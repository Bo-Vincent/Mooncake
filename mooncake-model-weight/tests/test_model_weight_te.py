from __future__ import annotations

import gc
import threading
import weakref
from dataclasses import replace
from math import prod

import pytest

from mooncake.model_weight.manifest import (
    ParallelRank,
    RuntimeFragment,
    RuntimeManifest,
    TensorDescriptor,
)
from mooncake.model_weight.planner import (
    CopyRange,
    TransferRegion,
    plan_runtime_transfer,
    plan_runtime_transfer_to_local_target,
)
from mooncake.model_weight.te import (
    MemoryRegistrationLease,
    MooncakeTransferEngineReader,
    MooncakeTransferEngineSink,
    TransferCompletionUnknownError,
    TransferEngineError,
)


class FakeTransferEngine:
    def __init__(self) -> None:
        self.calls = []
        self.fail_endpoint: str | None = None
        self.read_result: int | None = None
        self.register_calls: list[tuple[int, int]] = []
        self.unregister_calls: list[int] = []

    def register_memory(self, address: int, nbytes: int) -> int:
        self.register_calls.append((address, nbytes))
        return 0

    def unregister_memory(self, address: int) -> int:
        self.unregister_calls.append(address)
        return 0

    def batch_transfer_sync_write(
        self,
        endpoint: str,
        source_addresses: list[int],
        target_addresses: list[int],
        sizes: list[int],
    ) -> int:
        self.calls.append((endpoint, source_addresses, target_addresses, sizes))
        return -5 if endpoint == self.fail_endpoint else 0

    def batch_transfer_sync_read(
        self,
        endpoint: str,
        target_addresses: list[int],
        source_addresses: list[int],
        sizes: list[int],
    ) -> int:
        self.calls.append((endpoint, target_addresses, source_addresses, sizes))
        if self.read_result is not None:
            return self.read_result
        return -5 if endpoint == self.fail_endpoint else 0


class FakeBatchTransferTicket:
    def __init__(
        self,
        statuses: list[str],
        *,
        on_drain=None,
    ) -> None:
        self._statuses = list(statuses)
        self._on_drain = on_drain
        self.drain_calls: list[int] = []

    @property
    def status(self):
        return FakeCompletionStatus(self._statuses[0])

    @property
    def drained(self) -> bool:
        return self._statuses[0] != "COMPLETION_UNKNOWN"

    def drain(self, timeout_ms: int):
        self.drain_calls.append(timeout_ms)
        if self._on_drain is not None:
            self._on_drain()
        if len(self._statuses) > 1:
            self._statuses.pop(0)
        return FakeCompletionStatus(self._statuses[0])


class FakeCompletionStatus:
    def __init__(self, name: str) -> None:
        self.name = name


def manifests(
    tp: int,
    prefix: str,
    address_base: int,
    *,
    tensor: TensorDescriptor | None = None,
) -> tuple[RuntimeManifest, ...]:
    tensor = tensor or TensorDescriptor(
        tensor_id="layers.0.mlp.gate_up",
        global_shape=(8,),
        dtype="uint8",
        itemsize=1,
        partition_dim=0,
        layer_id=0,
        layout_fingerprint="sglang:qwen3.5:uint8:test",
    )
    dim = tensor.partition_dim
    assert dim is not None
    extent = tensor.global_shape[dim] // tp
    result = []
    for tp_rank in range(tp):
        worker_id = f"{prefix}-t{tp_rank}"
        local_shape = list(tensor.global_shape)
        local_shape[dim] = extent
        global_offset = [0] * len(tensor.global_shape)
        global_offset[dim] = tp_rank * extent
        fragment = RuntimeFragment(
            fragment_id=f"{worker_id}-fragment",
            tensor_id=tensor.tensor_id,
            global_offset=tuple(global_offset),
            local_shape=tuple(local_shape),
            address=address_base + tp_rank * 0x1000,
            nbytes=prod(local_shape) * tensor.itemsize,
            worker_id=worker_id,
            endpoint=f"{worker_id}:12345",
            device="cuda:0",
            rank=ParallelRank(tp=tp_rank),
            lease_generation=1,
        )
        result.append(
            RuntimeManifest(
                model_id="qwen3.5-0.8b",
                revision="step-42",
                instance_id=worker_id,
                tensors=(tensor,),
                fragments=(fragment,),
                lease_id=f"{worker_id}-runtime-lease",
            )
        )
    return tuple(result)


def registration_leases(
    manifests: tuple[RuntimeManifest, ...],
) -> tuple[MemoryRegistrationLease, ...]:
    return tuple(
        MemoryRegistrationLease.from_fragment(
            fragment,
            runtime_lease_id=manifest.lease_id,
        )
        for manifest in manifests
        for fragment in manifest.fragments
    )


@pytest.mark.parametrize(
    ("executor_kind", "side"),
    [
        ("sink", "source"),
        ("sink", "target"),
        ("reader", "source"),
        ("reader", "target"),
    ],
)
@pytest.mark.parametrize("mismatch", ["registration", "manifest"])
def test_te_pre_registered_paths_fence_runtime_lease_id_before_native_calls(
    executor_kind: str,
    side: str,
    mismatch: str,
) -> None:
    sources = manifests(tp=2, prefix="source", address_base=0x10000)
    targets = manifests(tp=2, prefix="target", address_base=0x40000)
    plan = (
        plan_runtime_transfer(sources, targets)
        if executor_kind == "sink"
        else plan_runtime_transfer_to_local_target(sources, targets[0])
    )
    current_sources = sources
    current_targets = targets
    if mismatch == "manifest":
        if side == "source":
            current_sources = (
                replace(sources[0], lease_id="rotated-source-runtime-lease"),
                *sources[1:],
            )
        else:
            current_targets = (
                replace(targets[0], lease_id="rotated-target-runtime-lease"),
                *targets[1:],
            )

    source_registrations = list(registration_leases(current_sources))
    target_registrations = list(registration_leases(current_targets))
    if mismatch == "registration":
        registrations = (
            source_registrations if side == "source" else target_registrations
        )
        registrations[0] = replace(
            registrations[0],
            runtime_lease_id=f"stale-{side}-runtime-lease",
        )

    engine = FakeTransferEngine()
    expected_error = (
        f"{side} registration lease mismatch"
        if mismatch == "registration"
        else f"{side} executor snapshot mismatch"
    )
    with pytest.raises(TransferEngineError, match=expected_error):
        try:
            if executor_kind == "sink":
                MooncakeTransferEngineSink(engine).execute(
                    plan,
                    current_sources[0],
                    current_targets,
                    target_registrations=tuple(target_registrations),
                    source_pre_registered=True,
                    source_registrations=tuple(source_registrations),
                )
            else:
                MooncakeTransferEngineReader(engine).execute(
                    plan,
                    current_sources,
                    current_targets[0],
                    source_registrations=tuple(source_registrations),
                    target_pre_registered=True,
                    target_registrations=tuple(target_registrations),
                )
        finally:
            assert engine.calls == []
            assert engine.register_calls == []
            assert engine.unregister_calls == []


def nd_te_manifests(
    prefix: str,
    address_base: int,
    *,
    source: bool,
) -> tuple[RuntimeManifest, ...]:
    tensor = TensorDescriptor(
        tensor_id="layers.0.experts.w1",
        global_shape=(4, 6, 8),
        dtype="uint8",
        itemsize=1,
        partition_dim=None,
        layer_id=0,
        expert_id=None,
        layout_fingerprint="framework:logical-contiguous:v2",
        shard_dims=(0,) if source else (2,),
    )
    result = []
    for rank in range(2):
        worker_id = f"{prefix}-{rank}"
        offset = (rank * 2, 0, 0) if source else (0, 0, rank * 4)
        shape = (2, 6, 8) if source else (4, 6, 4)
        parallel_rank = ParallelRank(ep=rank) if source else ParallelRank(tp=rank)
        fragment = RuntimeFragment(
            fragment_id=f"{worker_id}-fragment",
            tensor_id=tensor.tensor_id,
            global_offset=offset,
            local_shape=shape,
            address=address_base + rank * 0x1000,
            nbytes=prod(shape),
            worker_id=worker_id,
            endpoint=f"{worker_id}:12345",
            device="cuda:0",
            rank=parallel_rank,
            lease_generation=1,
        )
        result.append(
            RuntimeManifest(
                model_id="qwen-family-moe",
                revision="step-42",
                instance_id=worker_id,
                tensors=(tensor,),
                fragments=(fragment,),
                lease_id=f"{worker_id}-runtime-lease",
            )
        )
    return tuple(result)


def test_te_sink_lowers_nd_regions_in_bounded_batches() -> None:
    sources = nd_te_manifests("source", 0x10000, source=True)
    targets = nd_te_manifests("target", 0x40000, source=False)
    plan = plan_runtime_transfer(sources, targets)
    engine = FakeTransferEngine()

    receipts = MooncakeTransferEngineSink(
        engine,
        max_batch_operations=5,
        max_region_segments=12,
    ).execute(
        plan,
        sources[0],
        targets,
        target_registrations=registration_leases(targets),
    )

    assert [receipt.operation_count for receipt in receipts] == [12, 12]
    assert [receipt.nbytes for receipt in receipts] == [48, 48]
    assert [len(call[3]) for call in engine.calls] == [5, 5, 2, 5, 5, 2]
    assert max(len(call[3]) for call in engine.calls) == 5


def test_te_reader_lowers_nd_regions_in_bounded_batches() -> None:
    sources = nd_te_manifests("source", 0x10000, source=True)
    target = nd_te_manifests("target", 0x40000, source=False)[1]
    plan = plan_runtime_transfer_to_local_target(sources, target)
    engine = FakeTransferEngine()

    receipts = MooncakeTransferEngineReader(
        engine,
        max_batch_operations=5,
        max_region_segments=12,
    ).execute(
        plan,
        sources,
        target,
        source_registrations=registration_leases(sources),
        target_pre_registered=True,
        target_registrations=registration_leases((target,)),
    )

    assert [receipt.operation_count for receipt in receipts] == [12, 12]
    assert [receipt.nbytes for receipt in receipts] == [48, 48]
    assert [len(call[3]) for call in engine.calls] == [5, 5, 2, 5, 5, 2]
    assert max(len(call[3]) for call in engine.calls) == 5


@pytest.mark.parametrize("executor", ["sink", "reader"])
def test_te_rejects_nd_region_above_lowering_limit(executor: str) -> None:
    sources = nd_te_manifests("source", 0x10000, source=True)
    targets = nd_te_manifests("target", 0x40000, source=False)
    engine = FakeTransferEngine()

    with pytest.raises(TransferEngineError, match="max_region_segments"):
        if executor == "sink":
            plan = plan_runtime_transfer(sources, targets)
            MooncakeTransferEngineSink(engine, max_region_segments=11).execute(
                plan,
                sources[0],
                targets,
                target_registrations=registration_leases(targets),
            )
        else:
            target = targets[1]
            plan = plan_runtime_transfer_to_local_target(sources, target)
            MooncakeTransferEngineReader(engine, max_region_segments=11).execute(
                plan,
                sources,
                target,
                source_registrations=registration_leases(sources),
                target_pre_registered=True,
                target_registrations=registration_leases((target,)),
            )

    assert engine.calls == []
    assert engine.register_calls == []


def test_te_sink_keeps_legacy_copy_range_plan_executable() -> None:
    sources = manifests(tp=1, prefix="source", address_base=0x10000)
    targets = manifests(tp=1, prefix="target", address_base=0x40000)
    planned = plan_runtime_transfer(sources, targets)
    region = planned.operations[0]
    legacy = CopyRange(
        tensor_id=region.tensor_id,
        source=region.source,
        target=region.target,
        source_offset=region.source_offset,
        target_offset=region.target_offset,
        nbytes=region.nbytes,
        repeat=region.repeat,
        source_stride=region.source_stride,
        target_stride=region.target_stride,
    )
    plan = replace(planned, operations=(legacy,))
    engine = FakeTransferEngine()

    receipts = MooncakeTransferEngineSink(engine).execute(
        plan,
        sources[0],
        targets,
        target_registrations=registration_leases(targets),
    )

    assert receipts[0].nbytes == sources[0].fragments[0].nbytes
    assert len(engine.calls) == 1


def test_te_sink_executes_local_source_ranges_without_staging_buffer() -> None:
    sources = manifests(tp=2, prefix="source", address_base=0x10000)
    targets = manifests(tp=4, prefix="target", address_base=0x40000)
    plan = plan_runtime_transfer(sources, targets)
    engine = FakeTransferEngine()
    sink = MooncakeTransferEngineSink(engine)

    receipts = sink.execute(
        plan,
        sources[0],
        targets,
        target_registrations=registration_leases(targets),
    )

    assert receipts[0].source_worker_id == "source-t0"
    assert sum(receipt.nbytes for receipt in receipts) == 4
    assert engine.calls == [
        ("target-t0:12345", [0x10000], [0x40000], [2]),
        ("target-t1:12345", [0x10002], [0x41000], [2]),
    ]
    assert engine.register_calls == [(0x10000, 4)]
    assert engine.unregister_calls == [0x10000]


def test_te_sink_requires_generation_bound_target_registration_leases() -> None:
    sources = manifests(tp=2, prefix="source", address_base=0x10000)
    targets = manifests(tp=4, prefix="target", address_base=0x40000)
    plan = plan_runtime_transfer(sources, targets)
    sink = MooncakeTransferEngineSink(FakeTransferEngine())

    with pytest.raises(TransferEngineError, match="target registration"):
        sink.execute(plan, sources[0], targets)

    stale_leases = list(registration_leases(targets))
    stale_leases[0] = replace(stale_leases[0], lease_generation=2)
    with pytest.raises(TransferEngineError, match="target registration"):
        sink.execute(
            plan,
            sources[0],
            targets,
            target_registrations=tuple(stale_leases),
        )


def test_te_sink_surfaces_endpoint_failure() -> None:
    sources = manifests(tp=2, prefix="source", address_base=0x10000)
    targets = manifests(tp=4, prefix="target", address_base=0x40000)
    plan = plan_runtime_transfer(sources, targets)
    engine = FakeTransferEngine()
    engine.fail_endpoint = "target-t1:12345"

    with pytest.raises(TransferEngineError, match="target-t1:12345"):
        MooncakeTransferEngineSink(engine).execute(
            plan,
            sources[0],
            targets,
            target_registrations=registration_leases(targets),
        )


def test_te_sink_wraps_write_exception() -> None:
    sources = manifests(tp=2, prefix="source", address_base=0x10000)
    targets = manifests(tp=4, prefix="target", address_base=0x40000)
    plan = plan_runtime_transfer(sources, targets)
    engine = FakeTransferEngine()

    def fail_write(*args, **kwargs):
        raise RuntimeError("write exploded")

    engine.batch_transfer_sync_write = fail_write

    with pytest.raises(TransferEngineError, match="write exploded"):
        MooncakeTransferEngineSink(engine).execute(
            plan,
            sources[0],
            targets,
            target_registrations=registration_leases(targets),
        )

    assert engine.unregister_calls == [0x10000]


@pytest.mark.parametrize(
    ("executor", "terminal_status", "expect_error"),
    [
        ("sink", "COMPLETED", False),
        ("sink", "FAILED_DRAINED", True),
        ("reader", "COMPLETED", False),
        ("reader", "FAILED_DRAINED", True),
    ],
)
def test_te_ticket_path_keeps_registration_until_completion_is_known(
    executor: str,
    terminal_status: str,
    expect_error: bool,
) -> None:
    sources = manifests(tp=1, prefix="source", address_base=0x10000)
    targets = manifests(tp=1, prefix="target", address_base=0x40000)
    engine = FakeTransferEngine()
    ticket = FakeBatchTransferTicket(
        ["COMPLETION_UNKNOWN", "COMPLETION_UNKNOWN", terminal_status],
        on_drain=lambda: (
            pytest.fail("registration released before transfer drained")
            if engine.unregister_calls
            else None
        ),
    )

    def write_with_ticket(endpoint, source_addresses, target_addresses, sizes):
        engine.calls.append((endpoint, source_addresses, target_addresses, sizes))
        return ticket

    def read_with_ticket(endpoint, target_addresses, source_addresses, sizes):
        engine.calls.append((endpoint, target_addresses, source_addresses, sizes))
        return ticket

    engine.batch_transfer_sync_write_with_ticket = write_with_ticket
    engine.batch_transfer_sync_read_with_ticket = read_with_ticket

    if executor == "sink":
        plan = plan_runtime_transfer(sources, targets)

        def execute():
            return MooncakeTransferEngineSink(engine).execute(
                plan,
                sources[0],
                targets,
                target_registrations=registration_leases(targets),
            )

        expected_unregister = [0x10000]
    else:
        plan = plan_runtime_transfer_to_local_target(sources, targets[0])

        def execute():
            return MooncakeTransferEngineReader(engine).execute(
                plan,
                sources,
                targets[0],
                source_registrations=registration_leases(sources),
            )

        expected_unregister = [0x40000]

    if expect_error:
        with pytest.raises(TransferEngineError, match="FAILED_DRAINED"):
            execute()
    else:
        execute()

    assert ticket.drain_calls == [1000, 1000]
    assert engine.unregister_calls == expected_unregister


@pytest.mark.parametrize("executor_name", ["sink", "reader"])
def test_te_unknown_completion_moves_registration_to_queryable_quarantine(
    executor_name: str,
) -> None:
    sources = manifests(tp=1, prefix="source", address_base=0x10000)
    targets = manifests(tp=1, prefix="target", address_base=0x40000)
    engine = FakeTransferEngine()
    ticket = FakeBatchTransferTicket(["COMPLETION_UNKNOWN"] * 4)

    engine.batch_transfer_sync_write_with_ticket = lambda *args: ticket
    engine.batch_transfer_sync_read_with_ticket = lambda *args: ticket

    if executor_name == "sink":
        executor = MooncakeTransferEngineSink(
            engine,
            max_completion_drain_attempts=2,
        )

        def execute():
            return executor.execute(
                plan_runtime_transfer(sources, targets),
                sources[0],
                targets,
                target_registrations=registration_leases(targets),
            )

        expected_unregister = [0x10000]
    else:
        executor = MooncakeTransferEngineReader(
            engine,
            max_completion_drain_attempts=2,
        )

        def execute():
            return executor.execute(
                plan_runtime_transfer_to_local_target(sources, targets[0]),
                sources,
                targets[0],
                source_registrations=registration_leases(sources),
            )

        expected_unregister = [0x40000]

    with pytest.raises(TransferCompletionUnknownError) as raised:
        execute()

    pending_id = raised.value.pending_transfer_id
    assert ticket.drain_calls == [1000, 1000]
    assert engine.unregister_calls == []
    assert executor.pending_transfer_ids() == (pending_id,)
    assert executor.pending_transfer_status(pending_id) == "COMPLETION_UNKNOWN"

    ticket._statuses = ["COMPLETED"]
    assert executor.drain_pending_transfer(pending_id) == "COMPLETED"
    assert engine.unregister_calls == expected_unregister
    assert executor.pending_transfer_ids() == ()


@pytest.mark.parametrize("executor_name", ["sink", "reader"])
@pytest.mark.parametrize("interruption_type", [KeyboardInterrupt, SystemExit])
def test_te_interrupted_unknown_completion_quarantines_before_reraising(
    executor_name: str,
    interruption_type: type[BaseException],
) -> None:
    sources = manifests(tp=1, prefix="source", address_base=0x10000)
    targets = manifests(tp=1, prefix="target", address_base=0x40000)
    engine = FakeTransferEngine()

    def interrupt_drain() -> None:
        raise interruption_type("stop transfer wait")

    ticket = FakeBatchTransferTicket(
        ["COMPLETION_UNKNOWN"],
        on_drain=interrupt_drain,
    )
    engine.batch_transfer_sync_write_with_ticket = lambda *args: ticket
    engine.batch_transfer_sync_read_with_ticket = lambda *args: ticket

    if executor_name == "sink":
        executor = MooncakeTransferEngineSink(
            engine,
            max_completion_drain_attempts=1,
        )

        def execute():
            executor.execute(
                plan_runtime_transfer(sources, targets),
                sources[0],
                targets,
                target_registrations=registration_leases(targets),
            )

        expected_unregister = [0x10000]
    else:
        executor = MooncakeTransferEngineReader(
            engine,
            max_completion_drain_attempts=1,
        )

        def execute():
            executor.execute(
                plan_runtime_transfer_to_local_target(sources, targets[0]),
                sources,
                targets[0],
                source_registrations=registration_leases(sources),
            )

        expected_unregister = [0x40000]

    with pytest.raises(interruption_type, match="stop transfer wait"):
        execute()

    pending_ids = executor.pending_transfer_ids()
    assert len(pending_ids) == 1
    pending_id = pending_ids[0]
    assert engine.unregister_calls == []
    assert executor.pending_transfer_status(pending_id) == "COMPLETION_UNKNOWN"

    ticket._on_drain = None
    ticket._statuses = ["COMPLETED"]
    assert executor.drain_pending_transfer(pending_id) == "COMPLETED"
    assert engine.unregister_calls == expected_unregister
    assert executor.pending_transfer_ids() == ()


def test_te_reader_quarantines_legacy_completion_unknown_without_ticket() -> None:
    sources = manifests(tp=1, prefix="source", address_base=0x10000)
    targets = manifests(tp=1, prefix="target", address_base=0x40000)
    engine = FakeTransferEngine()
    engine.read_result = -2
    reader = MooncakeTransferEngineReader(engine)

    with pytest.raises(TransferCompletionUnknownError) as raised:
        reader.execute(
            plan_runtime_transfer_to_local_target(sources, targets[0]),
            sources,
            targets[0],
            source_registrations=registration_leases(sources),
        )

    pending_id = raised.value.pending_transfer_id
    assert "restart-required" in str(raised.value)
    assert reader.pending_transfer_ids() == (pending_id,)
    assert reader.pending_transfer_status(pending_id) == (
        "COMPLETION_UNKNOWN_RESTART_REQUIRED"
    )
    assert reader.drain_pending_transfer(pending_id, timeout_ms=0) == (
        "COMPLETION_UNKNOWN"
    )
    assert engine.unregister_calls == []


def test_te_legacy_completion_unknown_blocks_same_engine_until_restart() -> None:
    sources = manifests(tp=1, prefix="source", address_base=0x10000)
    targets = manifests(tp=1, prefix="target", address_base=0x40000)
    plan = plan_runtime_transfer_to_local_target(sources, targets[0])
    engine = FakeTransferEngine()
    engine.read_result = -2
    reader = MooncakeTransferEngineReader(engine)

    with pytest.raises(TransferCompletionUnknownError):
        reader.execute(
            plan,
            sources,
            targets[0],
            source_registrations=registration_leases(sources),
        )

    call_count = len(engine.calls)
    register_count = len(engine.register_calls)
    engine.read_result = 0
    with pytest.raises(TransferEngineError, match="restart-required"):
        MooncakeTransferEngineSink(engine).execute(
            plan_runtime_transfer(sources, targets),
            sources[0],
            targets,
            target_registrations=registration_leases(targets),
        )
    with pytest.raises(TransferEngineError, match="restart-required"):
        MooncakeTransferEngineReader(engine).execute(
            plan,
            sources,
            targets[0],
            source_registrations=registration_leases(sources),
        )

    assert len(engine.calls) == call_count
    assert len(engine.register_calls) == register_count
    assert engine.unregister_calls == []


@pytest.mark.parametrize("healthy_executor", ("sink", "reader"))
def test_te_pending_transfer_backpressure_is_engine_scoped(
    healthy_executor: str,
) -> None:
    sources = manifests(tp=1, prefix="source", address_base=0x10000)
    targets = manifests(tp=1, prefix="target", address_base=0x40000)
    blocked_engine = FakeTransferEngine()
    blocked_engine.read_result = -2

    with pytest.raises(TransferCompletionUnknownError):
        MooncakeTransferEngineReader(blocked_engine).execute(
            plan_runtime_transfer_to_local_target(sources, targets[0]),
            sources,
            targets[0],
            source_registrations=registration_leases(sources),
        )

    healthy_engine = FakeTransferEngine()
    if healthy_executor == "sink":
        receipts = MooncakeTransferEngineSink(healthy_engine).execute(
            plan_runtime_transfer(sources, targets),
            sources[0],
            targets,
            target_registrations=registration_leases(targets),
        )
    else:
        receipts = MooncakeTransferEngineReader(healthy_engine).execute(
            plan_runtime_transfer_to_local_target(sources, targets[0]),
            sources,
            targets[0],
            source_registrations=registration_leases(sources),
        )

    assert len(receipts) == 1
    assert len(healthy_engine.calls) == 1


@pytest.mark.parametrize("recovery_executor", ("sink", "reader"))
def test_te_pending_transfer_allows_new_transfer_after_drain_cleanup(
    recovery_executor: str,
) -> None:
    sources = manifests(tp=1, prefix="source", address_base=0x10000)
    targets = manifests(tp=1, prefix="target", address_base=0x40000)
    engine = FakeTransferEngine()
    pending_ticket = FakeBatchTransferTicket(["COMPLETION_UNKNOWN"] * 2)
    engine.batch_transfer_sync_write_with_ticket = lambda *args: pending_ticket
    sink = MooncakeTransferEngineSink(engine, max_completion_drain_attempts=1)

    with pytest.raises(TransferCompletionUnknownError) as raised:
        sink.execute(
            plan_runtime_transfer(sources, targets),
            sources[0],
            targets,
            target_registrations=registration_leases(targets),
        )

    completed_ticket = FakeBatchTransferTicket(["COMPLETED"])
    engine.batch_transfer_sync_write_with_ticket = lambda *args: completed_ticket
    engine.batch_transfer_sync_read_with_ticket = lambda *args: completed_ticket

    def recover() -> None:
        if recovery_executor == "sink":
            MooncakeTransferEngineSink(engine).execute(
                plan_runtime_transfer(sources, targets),
                sources[0],
                targets,
                target_registrations=registration_leases(targets),
            )
        else:
            MooncakeTransferEngineReader(engine).execute(
                plan_runtime_transfer_to_local_target(sources, targets[0]),
                sources,
                targets[0],
                source_registrations=registration_leases(sources),
            )

    with pytest.raises(TransferEngineError, match="drain_pending_transfer"):
        recover()

    pending_ticket._statuses = ["COMPLETED"]
    assert sink.drain_pending_transfer(raised.value.pending_transfer_id) == "COMPLETED"
    assert sink.pending_transfer_ids() == ()
    recover()


def test_te_canonical_engine_identity_fails_closed_for_broken_getter() -> None:
    engine = FakeTransferEngine()

    def fail_get_engine_ptr() -> int:
        raise RuntimeError("native engine handle unavailable")

    engine.get_engine_ptr = fail_get_engine_ptr
    with pytest.raises(TransferEngineError, match="canonical engine identity"):
        MooncakeTransferEngineSink(engine)


@pytest.mark.parametrize(
    ("first_executor_name", "second_executor_name", "use_wrapper_aliases"),
    [
        pytest.param("sink", "sink", False, id="sink-sink"),
        pytest.param("reader", "reader", False, id="reader-reader"),
        pytest.param("sink", "reader", False, id="sink-reader"),
        pytest.param("sink", "sink", True, id="wrapper-alias"),
    ],
)
def test_te_submission_reservation_is_atomic(
    first_executor_name: str,
    second_executor_name: str,
    use_wrapper_aliases: bool,
) -> None:
    class EngineAlias:
        def __init__(self, wrapped_engine) -> None:
            self._wrapped_engine = wrapped_engine

        def __getattr__(self, name: str):
            return getattr(self._wrapped_engine, name)

    sources = manifests(tp=1, prefix="source", address_base=0x10000)
    targets = manifests(tp=1, prefix="target", address_base=0x40000)
    sink_plan = plan_runtime_transfer(sources, targets)
    reader_plan = plan_runtime_transfer_to_local_target(sources, targets[0])
    engine = FakeTransferEngine()
    first_engine = engine
    second_engine = engine
    if use_wrapper_aliases:
        engine.get_engine_ptr = lambda: 0xCA110CA1
        first_engine = EngineAlias(engine)
        second_engine = EngineAlias(engine)

    first_entered = threading.Event()
    allow_first_to_finish = threading.Event()
    submission_lock = threading.Lock()
    submission_count = 0
    pending_ticket = FakeBatchTransferTicket(["COMPLETION_UNKNOWN"] * 2)
    completed_ticket = FakeBatchTransferTicket(["COMPLETED"])

    def transfer_with_ticket(*args):
        nonlocal submission_count
        with submission_lock:
            submission_count += 1
            call_number = submission_count
        if call_number == 1:
            first_entered.set()
            assert allow_first_to_finish.wait(timeout=2)
            return pending_ticket
        return completed_ticket

    engine.batch_transfer_sync_write_with_ticket = transfer_with_ticket
    engine.batch_transfer_sync_read_with_ticket = transfer_with_ticket

    def make_executor(executor_name: str, transfer_engine):
        executor_type = (
            MooncakeTransferEngineSink
            if executor_name == "sink"
            else MooncakeTransferEngineReader
        )
        return executor_type(
            transfer_engine,
            max_completion_drain_attempts=1,
        )

    def execute(executor_name: str, executor) -> None:
        if executor_name == "sink":
            executor.execute(
                sink_plan,
                sources[0],
                targets,
                target_registrations=registration_leases(targets),
            )
        else:
            executor.execute(
                reader_plan,
                sources,
                targets[0],
                source_registrations=registration_leases(sources),
            )

    first_executor = make_executor(first_executor_name, first_engine)
    second_executor = make_executor(second_executor_name, second_engine)
    first_errors = []
    second_errors = []

    def capture_error(executor_name: str, executor, errors: list) -> None:
        try:
            execute(executor_name, executor)
        except Exception as error:
            errors.append(error)

    first = threading.Thread(
        target=capture_error,
        args=(first_executor_name, first_executor, first_errors),
    )
    second = threading.Thread(
        target=capture_error,
        args=(second_executor_name, second_executor, second_errors),
    )
    first.start()
    try:
        assert first_entered.wait(timeout=2)
        second.start()
        second.join(timeout=2)
        assert not second.is_alive()
        register_calls_while_active = tuple(engine.register_calls)
        unregister_calls_while_active = tuple(engine.unregister_calls)
    finally:
        allow_first_to_finish.set()
        first.join(timeout=2)
        if second.ident is not None:
            second.join(timeout=2)

    assert not first.is_alive()
    assert len(first_errors) == 1
    assert isinstance(first_errors[0], TransferCompletionUnknownError)
    pending_id = first_errors[0].pending_transfer_id
    visible_pending_ids = second_executor.pending_transfer_ids()
    pending_ticket._statuses = ["COMPLETED"]
    assert second_executor.drain_pending_transfer(pending_id) == "COMPLETED"

    first_fragment = (
        sources[0].fragments[0]
        if first_executor_name == "sink"
        else targets[0].fragments[0]
    )
    assert visible_pending_ids == (pending_id,)
    assert submission_count == 1
    assert register_calls_while_active == (
        (first_fragment.address, first_fragment.nbytes),
    )
    assert unregister_calls_while_active == ()
    assert engine.unregister_calls == [first_fragment.address]
    assert len(second_errors) == 1
    assert isinstance(second_errors[0], TransferEngineError)
    assert "active weight transfer submission" in str(second_errors[0])


@pytest.mark.parametrize("executor_name", ("sink", "reader"))
def test_te_pending_transfer_cannot_drain_before_resource_handoff(
    executor_name: str,
) -> None:
    sources = manifests(tp=1, prefix="source", address_base=0x10000)
    targets = manifests(tp=1, prefix="target", address_base=0x40000)
    engine = FakeTransferEngine()
    ticket = FakeBatchTransferTicket(["COMPLETION_UNKNOWN"] * 2)
    engine.batch_transfer_sync_write_with_ticket = lambda *args: ticket
    engine.batch_transfer_sync_read_with_ticket = lambda *args: ticket

    if executor_name == "sink":
        executor = MooncakeTransferEngineSink(
            engine,
            max_completion_drain_attempts=1,
        )

        def execute() -> None:
            executor.execute(
                plan_runtime_transfer(sources, targets),
                sources[0],
                targets,
                target_registrations=registration_leases(targets),
            )

        pending_owner = executor
        owned_address = sources[0].fragments[0].address
    else:
        executor = MooncakeTransferEngineReader(
            engine,
            max_completion_drain_attempts=1,
        )

        def execute() -> None:
            executor.execute(
                plan_runtime_transfer_to_local_target(sources, targets[0]),
                sources,
                targets[0],
                source_registrations=registration_leases(sources),
            )

        pending_owner = executor._pending
        owned_address = targets[0].fragments[0].address

    original_retain = pending_owner._retain_pending_resources
    handoff_entered = threading.Event()
    allow_handoff = threading.Event()
    pending_ids = []

    def block_resource_handoff(
        pending_transfer_id: str,
        *,
        registrations,
        resources,
    ) -> None:
        pending_ids.append(pending_transfer_id)
        handoff_entered.set()
        assert allow_handoff.wait(timeout=2)
        original_retain(
            pending_transfer_id,
            registrations=registrations,
            resources=resources,
        )

    pending_owner._retain_pending_resources = block_resource_handoff
    execution_errors = []

    def run_execute() -> None:
        try:
            execute()
        except Exception as error:
            execution_errors.append(error)

    worker = threading.Thread(target=run_execute)
    worker.start()
    assert handoff_entered.wait(timeout=2)
    pending_id = pending_ids[0]
    ticket._statuses = ["COMPLETED"]
    drain_calls_before_handoff = tuple(ticket.drain_calls)
    try:
        with pytest.raises(TransferEngineError, match="resource handoff"):
            executor.drain_pending_transfer(pending_id)
        assert tuple(ticket.drain_calls) == drain_calls_before_handoff
        assert engine.unregister_calls == []
    finally:
        allow_handoff.set()
        worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(execution_errors) == 1
    assert isinstance(execution_errors[0], TransferCompletionUnknownError)
    assert executor.pending_transfer_status(pending_id) == "COMPLETED"
    assert executor.drain_pending_transfer(pending_id) == "COMPLETED"
    assert engine.unregister_calls == [owned_address]


def test_te_pending_transfer_retains_fragment_owner_until_terminal() -> None:
    class Owner:
        pass

    def create_pending():
        sources = manifests(tp=1, prefix="source", address_base=0x10000)
        targets = manifests(tp=1, prefix="target", address_base=0x40000)
        owner = Owner()
        owned_fragment = replace(targets[0].fragments[0], owner=owner)
        owned_target = replace(targets[0], fragments=(owned_fragment,))
        engine = FakeTransferEngine()
        ticket = FakeBatchTransferTicket(["COMPLETION_UNKNOWN"] * 4)
        engine.batch_transfer_sync_read_with_ticket = lambda *args: ticket
        reader = MooncakeTransferEngineReader(
            engine,
            max_completion_drain_attempts=1,
        )
        try:
            reader.execute(
                plan_runtime_transfer_to_local_target(sources, owned_target),
                sources,
                owned_target,
                source_registrations=registration_leases(sources),
            )
        except TransferCompletionUnknownError as error:
            pending_id = error.pending_transfer_id
        else:
            raise AssertionError("transfer must remain pending")
        return reader, ticket, pending_id, weakref.ref(owner)

    reader, ticket, pending_id, owner_ref = create_pending()
    gc.collect()
    assert owner_ref() is not None

    ticket._statuses = ["COMPLETED"]
    assert reader.drain_pending_transfer(pending_id) == "COMPLETED"
    gc.collect()
    assert owner_ref() is None


def test_te_pending_transfer_survives_executor_destruction() -> None:
    class Owner:
        pass

    def create_pending():
        sources = manifests(tp=1, prefix="source", address_base=0x10000)
        targets = manifests(tp=1, prefix="target", address_base=0x40000)
        owner = Owner()
        owned_fragment = replace(targets[0].fragments[0], owner=owner)
        owned_target = replace(targets[0], fragments=(owned_fragment,))
        engine = FakeTransferEngine()
        ticket = FakeBatchTransferTicket(["COMPLETION_UNKNOWN"] * 4)
        engine.batch_transfer_sync_read_with_ticket = lambda *args: ticket
        reader = MooncakeTransferEngineReader(
            engine,
            max_completion_drain_attempts=1,
        )
        with pytest.raises(TransferCompletionUnknownError) as raised:
            reader.execute(
                plan_runtime_transfer_to_local_target(sources, owned_target),
                sources,
                owned_target,
                source_registrations=registration_leases(sources),
            )
        return engine, ticket, raised.value.pending_transfer_id, weakref.ref(owner)

    engine, ticket, pending_id, owner_ref = create_pending()
    gc.collect()
    assert owner_ref() is not None

    recovery_reader = MooncakeTransferEngineReader(engine)
    assert recovery_reader.pending_transfer_status(pending_id) == "COMPLETION_UNKNOWN"
    ticket._statuses = ["COMPLETED"]
    assert recovery_reader.drain_pending_transfer(pending_id) == "COMPLETED"
    assert engine.unregister_calls == [0x40000]
    gc.collect()
    assert owner_ref() is None


@pytest.mark.parametrize("executor_name", ("sink", "reader"))
def test_te_pre_registered_unknown_retains_external_owner_until_terminal(
    executor_name: str,
) -> None:
    class ExternalOwner:
        pass

    def create_pending():
        sources = manifests(tp=1, prefix="source", address_base=0x10000)
        targets = manifests(tp=1, prefix="target", address_base=0x40000)
        engine = FakeTransferEngine()
        ticket = FakeBatchTransferTicket(
            ["COMPLETION_UNKNOWN", "COMPLETION_UNKNOWN"],
            on_drain=lambda: (
                pytest.fail("external registration released before transfer drained")
                if engine.unregister_calls
                else None
            ),
        )
        owner = ExternalOwner()
        if executor_name == "sink":
            owned_fragment = replace(sources[0].fragments[0], owner=owner)
            sources = (replace(sources[0], fragments=(owned_fragment,)),)
            executor = MooncakeTransferEngineSink(
                engine,
                max_completion_drain_attempts=1,
            )
            engine.batch_transfer_sync_write_with_ticket = lambda *args: ticket
        else:
            owned_fragment = replace(targets[0].fragments[0], owner=owner)
            targets = (replace(targets[0], fragments=(owned_fragment,)),)
            executor = MooncakeTransferEngineReader(
                engine,
                max_completion_drain_attempts=1,
            )
            engine.batch_transfer_sync_read_with_ticket = lambda *args: ticket

        external_address = owned_fragment.address
        assert engine.register_memory(external_address, owned_fragment.nbytes) == 0
        with pytest.raises(TransferCompletionUnknownError) as raised:
            if executor_name == "sink":
                executor.execute(
                    plan_runtime_transfer(sources, targets),
                    sources[0],
                    targets,
                    target_registrations=registration_leases(targets),
                    source_pre_registered=True,
                    source_registrations=registration_leases(sources),
                )
            else:
                executor.execute(
                    plan_runtime_transfer_to_local_target(sources, targets[0]),
                    sources,
                    targets[0],
                    source_registrations=registration_leases(sources),
                    target_pre_registered=True,
                    target_registrations=registration_leases(targets),
                )
        return (
            engine,
            ticket,
            raised.value.pending_transfer_id,
            type(executor),
            external_address,
            weakref.ref(owner),
        )

    engine, ticket, pending_id, executor_type, external_address, owner_ref = (
        create_pending()
    )
    gc.collect()
    assert engine.unregister_calls == []
    assert owner_ref() is not None

    recovery = executor_type(engine)
    assert recovery.pending_transfer_status(pending_id) == "COMPLETION_UNKNOWN"
    ticket._statuses = ["COMPLETED"]
    assert recovery.drain_pending_transfer(pending_id) == "COMPLETED"
    assert engine.unregister_calls == []
    gc.collect()
    assert owner_ref() is None

    assert engine.unregister_memory(external_address) == 0
    assert engine.unregister_calls == [external_address]


def test_te_continuous_drain_errors_remain_quarantined_until_terminal() -> None:
    sources = manifests(tp=1, prefix="source", address_base=0x10000)
    targets = manifests(tp=1, prefix="target", address_base=0x40000)
    engine = FakeTransferEngine()

    def fail_drain() -> None:
        raise RuntimeError("backend unavailable")

    ticket = FakeBatchTransferTicket(
        ["COMPLETION_UNKNOWN"],
        on_drain=fail_drain,
    )
    engine.batch_transfer_sync_write_with_ticket = lambda *args: ticket
    sink = MooncakeTransferEngineSink(
        engine,
        max_completion_drain_attempts=2,
    )

    with pytest.raises(TransferCompletionUnknownError) as raised:
        sink.execute(
            plan_runtime_transfer(sources, targets),
            sources[0],
            targets,
            target_registrations=registration_leases(targets),
        )

    pending_id = raised.value.pending_transfer_id
    assert ticket.drain_calls == [1000, 1000]
    assert engine.unregister_calls == []
    assert sink.drain_pending_transfer(pending_id) == "COMPLETION_UNKNOWN"
    assert engine.unregister_calls == []

    ticket._on_drain = None
    ticket._statuses = ["FAILED_DRAINED"]
    assert sink.drain_pending_transfer(pending_id) == "FAILED_DRAINED"
    assert engine.unregister_calls == [0x10000]


def test_te_rejects_concurrent_drain_for_same_pending_transfer() -> None:
    sources = manifests(tp=1, prefix="source", address_base=0x10000)
    targets = manifests(tp=1, prefix="target", address_base=0x40000)
    engine = FakeTransferEngine()
    ticket = FakeBatchTransferTicket(["COMPLETION_UNKNOWN"] * 2)
    engine.batch_transfer_sync_write_with_ticket = lambda *args: ticket
    sink = MooncakeTransferEngineSink(
        engine,
        max_completion_drain_attempts=1,
    )

    with pytest.raises(TransferCompletionUnknownError) as raised:
        sink.execute(
            plan_runtime_transfer(sources, targets),
            sources[0],
            targets,
            target_registrations=registration_leases(targets),
        )

    pending_id = raised.value.pending_transfer_id
    first_entered = threading.Event()
    allow_first_to_finish = threading.Event()

    def block_first_drain() -> None:
        first_entered.set()
        assert allow_first_to_finish.wait(timeout=2)

    ticket._statuses = ["COMPLETED"]
    ticket._on_drain = block_first_drain
    statuses = []
    errors = []

    def drain() -> None:
        try:
            statuses.append(sink.drain_pending_transfer(pending_id))
        except Exception as error:
            errors.append(error)

    first = threading.Thread(target=drain)
    second = threading.Thread(target=drain)
    first.start()
    assert first_entered.wait(timeout=2)
    second.start()
    second.join(timeout=0.2)
    allow_first_to_finish.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert statuses == ["COMPLETED"]
    assert len(errors) == 1
    assert isinstance(errors[0], TransferEngineError)
    assert "already being drained" in str(errors[0])
    assert engine.unregister_calls == [0x10000]


def test_te_sink_rejects_stale_source_generation() -> None:
    sources = manifests(tp=2, prefix="source", address_base=0x10000)
    targets = manifests(tp=4, prefix="target", address_base=0x40000)
    plan = plan_runtime_transfer(sources, targets)
    stale_fragment = replace(sources[0].fragments[0], lease_generation=2)
    stale_manifest = replace(
        sources[0],
        fragments=(stale_fragment,),
        generation=None,
    )

    with pytest.raises(TransferEngineError, match="source executor snapshot mismatch"):
        MooncakeTransferEngineSink(FakeTransferEngine()).execute(
            plan,
            stale_manifest,
            targets,
            target_registrations=registration_leases(targets),
        )


def test_te_sink_rejects_generation_scoped_source_id_rollover() -> None:
    sources = manifests(tp=2, prefix="source", address_base=0x10000)
    targets = manifests(tp=4, prefix="target", address_base=0x40000)
    plan = plan_runtime_transfer(sources, targets)
    replacement = replace(
        sources[0].fragments[0],
        fragment_id="replacement-source-fragment",
        worker_id="replacement-source-worker",
        lease_generation=2,
    )
    current = replace(
        sources[0],
        instance_id="replacement-source-instance",
        generation=None,
        fragments=(replacement,),
    )

    with pytest.raises(TransferEngineError, match="source executor snapshot mismatch"):
        MooncakeTransferEngineSink(FakeTransferEngine()).execute(
            plan,
            current,
            targets,
            target_registrations=registration_leases(targets),
        )


def test_te_sink_rejects_stale_target_address_and_generation() -> None:
    sources = manifests(tp=2, prefix="source", address_base=0x10000)
    targets = manifests(tp=4, prefix="target", address_base=0x40000)
    plan = plan_runtime_transfer(sources, targets)
    replacement = replace(targets[0].fragments[0], address=0x90000, lease_generation=2)
    current_targets = (
        replace(targets[0], fragments=(replacement,), generation=None),
        *targets[1:],
    )

    with pytest.raises(TransferEngineError, match="target executor snapshot mismatch"):
        MooncakeTransferEngineSink(FakeTransferEngine()).execute(
            plan,
            sources[0],
            current_targets,
            target_registrations=registration_leases(targets),
        )


def test_te_sink_rejects_generation_scoped_target_id_rollover() -> None:
    sources = manifests(tp=2, prefix="source", address_base=0x10000)
    targets = manifests(tp=4, prefix="target", address_base=0x40000)
    plan = plan_runtime_transfer(sources, targets)
    replacement = replace(
        targets[0].fragments[0],
        fragment_id="replacement-target-fragment",
        lease_generation=2,
    )
    current_targets = (
        replace(targets[0], fragments=(replacement,), generation=None),
        *targets[1:],
    )

    with pytest.raises(TransferEngineError, match="target executor snapshot mismatch"):
        MooncakeTransferEngineSink(FakeTransferEngine()).execute(
            plan,
            sources[0],
            current_targets,
            target_registrations=registration_leases(targets),
        )


@pytest.mark.parametrize("side", ["source", "target"])
def test_te_sink_rejects_revision_mismatch(side: str) -> None:
    sources = manifests(tp=2, prefix="source", address_base=0x10000)
    targets = manifests(tp=4, prefix="target", address_base=0x40000)
    plan = plan_runtime_transfer(sources, targets)
    if side == "source":
        sources = (replace(sources[0], revision="step-43"), *sources[1:])
    else:
        targets = (replace(targets[0], revision="step-43"), *targets[1:])

    with pytest.raises(TransferEngineError, match="revision mismatch"):
        MooncakeTransferEngineSink(FakeTransferEngine()).execute(
            plan,
            sources[0],
            targets,
            target_registrations=registration_leases(targets),
        )


def test_te_sink_expands_compact_ranges_in_bounded_batches() -> None:
    tensor = TensorDescriptor(
        tensor_id="layers.0.mlp.down_proj.weight",
        global_shape=(5, 8),
        dtype="uint8",
        itemsize=1,
        partition_dim=1,
        layer_id=0,
        layout_fingerprint="sglang:qwen3.5:uint8:test",
    )
    sources = manifests(
        tp=2,
        prefix="source",
        address_base=0x10000,
        tensor=tensor,
    )
    targets = manifests(
        tp=4,
        prefix="target",
        address_base=0x40000,
        tensor=tensor,
    )
    plan = plan_runtime_transfer(sources, targets)
    engine = FakeTransferEngine()

    receipts = MooncakeTransferEngineSink(engine, max_batch_operations=2).execute(
        plan,
        sources[0],
        targets,
        target_registrations=registration_leases(targets),
    )

    assert [len(call[1]) for call in engine.calls] == [2, 2, 1, 2, 2, 1]
    assert sum(receipt.operation_count for receipt in receipts) == 10
    assert sum(receipt.nbytes for receipt in receipts) == 20


def test_te_sink_allows_only_an_explicitly_planned_noop_source_executor() -> None:
    source_dp0 = manifests(tp=1, prefix="source-d0", address_base=0x10000)[0]
    source_dp1_fragment = replace(
        source_dp0.fragments[0],
        fragment_id="source-d1-fragment",
        address=0x20000,
        worker_id="source-d1-t0",
        endpoint="source-d1-t0:12345",
        rank=replace(source_dp0.fragments[0].rank, dp=1),
    )
    source_dp1 = replace(
        source_dp0,
        instance_id="source-d1-t0",
        fragments=(source_dp1_fragment,),
    )
    sources = (source_dp0, source_dp1)
    targets = manifests(tp=1, prefix="target", address_base=0x40000)
    plan = plan_runtime_transfer(sources, targets)
    engine = FakeTransferEngine()

    receipts = MooncakeTransferEngineSink(engine).execute(
        plan,
        source_dp1,
        targets,
        target_registrations=registration_leases(targets),
    )

    assert receipts == ()
    assert engine.calls == []
    assert engine.register_calls == []


def test_te_sink_fences_runtime_lease_for_explicit_noop_executor() -> None:
    source_dp0 = manifests(tp=1, prefix="source-d0", address_base=0x10000)[0]
    source_dp1_fragment = replace(
        source_dp0.fragments[0],
        fragment_id="source-d1-fragment",
        address=0x20000,
        worker_id="source-d1-t0",
        endpoint="source-d1-t0:12345",
        rank=replace(source_dp0.fragments[0].rank, dp=1),
    )
    source_dp1 = replace(
        source_dp0,
        instance_id="source-d1-t0",
        fragments=(source_dp1_fragment,),
    )
    targets = manifests(tp=1, prefix="target", address_base=0x40000)
    plan = plan_runtime_transfer((source_dp0, source_dp1), targets)
    stale = replace(
        source_dp1,
        generation=None,
        fragments=(
            replace(
                source_dp1_fragment,
                address=0x90000,
                endpoint="source-d1-t0:54321",
                lease_generation=2,
            ),
        ),
    )

    with pytest.raises(TransferEngineError, match="source executor snapshot mismatch"):
        MooncakeTransferEngineSink(FakeTransferEngine()).execute(
            plan,
            stale,
            targets,
            target_registrations=registration_leases(targets),
        )


def test_te_receipt_identifies_worker_instead_of_serving_instance() -> None:
    source = replace(
        manifests(tp=1, prefix="source", address_base=0x10000)[0],
        instance_id="serving-instance",
    )
    targets = manifests(tp=1, prefix="target", address_base=0x40000)
    plan = plan_runtime_transfer((source,), targets)

    receipts = MooncakeTransferEngineSink(FakeTransferEngine()).execute(
        plan,
        source,
        targets,
        target_registrations=registration_leases(targets),
    )

    assert receipts[0].source_worker_id == "source-t0"


def test_te_reader_pulls_local_target_ranges_without_source_rpc() -> None:
    sources = manifests(tp=2, prefix="source", address_base=0x10000)
    targets = manifests(tp=4, prefix="target", address_base=0x40000)
    target = targets[1]
    plan = plan_runtime_transfer_to_local_target(sources, target)
    engine = FakeTransferEngine()

    receipts = MooncakeTransferEngineReader(engine).execute(
        plan,
        sources,
        target,
        source_registrations=registration_leases(sources),
        target_pre_registered=True,
        target_registrations=registration_leases((target,)),
    )

    assert receipts[0].source_endpoint == "source-t0:12345"
    assert receipts[0].target_worker_id == "target-t1"
    assert receipts[0].nbytes == 2
    assert engine.calls == [("source-t0:12345", [0x41000], [0x10002], [2])]
    assert engine.register_calls == []
    assert engine.unregister_calls == []


def test_te_reader_wraps_target_registration_exception() -> None:
    sources = manifests(tp=2, prefix="source", address_base=0x10000)
    target = manifests(tp=4, prefix="target", address_base=0x40000)[1]
    plan = plan_runtime_transfer_to_local_target(sources, target)
    engine = FakeTransferEngine()

    def fail_register(*args, **kwargs):
        raise RuntimeError("register exploded")

    engine.register_memory = fail_register

    with pytest.raises(TransferEngineError, match="register exploded"):
        MooncakeTransferEngineReader(engine).execute(
            plan,
            sources,
            target,
            source_registrations=registration_leases(sources),
        )


def test_te_reader_requires_generation_bound_source_registration_leases() -> None:
    sources = manifests(tp=2, prefix="source", address_base=0x10000)
    target = manifests(tp=4, prefix="target", address_base=0x40000)[1]
    plan = plan_runtime_transfer_to_local_target(sources, target)
    reader = MooncakeTransferEngineReader(FakeTransferEngine())
    target_leases = registration_leases((target,))

    with pytest.raises(TransferEngineError, match="source registration leases"):
        reader.execute(
            plan,
            sources,
            target,
            target_pre_registered=True,
            target_registrations=target_leases,
        )

    stale_generation = list(registration_leases(sources))
    stale_generation[0] = replace(stale_generation[0], lease_generation=2)
    with pytest.raises(TransferEngineError, match="source registration lease mismatch"):
        reader.execute(
            plan,
            sources,
            target,
            source_registrations=stale_generation,
            target_pre_registered=True,
            target_registrations=target_leases,
        )

    stale_runtime_lease = list(registration_leases(sources))
    stale_runtime_lease[0] = replace(
        stale_runtime_lease[0], runtime_lease_id="stale-runtime-lease"
    )
    with pytest.raises(TransferEngineError, match="source registration lease mismatch"):
        reader.execute(
            plan,
            sources,
            target,
            source_registrations=stale_runtime_lease,
            target_pre_registered=True,
            target_registrations=target_leases,
        )


@pytest.mark.parametrize(
    "field, value",
    [
        ("address", 0x90000),
        ("nbytes", 1),
        ("lease_generation", 2),
        ("runtime_lease_id", "stale-runtime-lease"),
    ],
)
def test_te_reader_rejects_source_registration_snapshot_mismatch(
    field: str, value: int | str
) -> None:
    sources = manifests(tp=2, prefix="source", address_base=0x10000)
    target = manifests(tp=4, prefix="target", address_base=0x40000)[1]
    plan = plan_runtime_transfer_to_local_target(sources, target)
    leases = list(registration_leases(sources))
    leases[0] = replace(leases[0], **{field: value})

    with pytest.raises(TransferEngineError, match="source registration lease mismatch"):
        MooncakeTransferEngineReader(FakeTransferEngine()).execute(
            plan,
            sources,
            target,
            source_registrations=leases,
            target_pre_registered=True,
            target_registrations=registration_leases((target,)),
        )


def test_te_reader_legacy_one_shot_accepts_missing_runtime_lease_id() -> None:
    sources = tuple(
        replace(manifest, lease_id=None)
        for manifest in manifests(tp=2, prefix="source", address_base=0x10000)
    )
    target = replace(
        manifests(tp=4, prefix="target", address_base=0x40000)[1],
        lease_id=None,
    )
    plan = plan_runtime_transfer_to_local_target(sources, target)
    engine = FakeTransferEngine()

    receipts = MooncakeTransferEngineReader(engine).execute(
        plan,
        sources,
        target,
        source_registrations=registration_leases(sources),
        target_pre_registered=True,
        target_registrations=registration_leases((target,)),
    )

    assert sum(receipt.nbytes for receipt in receipts) == 2
    assert len(engine.calls) == 1


def test_te_reader_requires_registrations_only_for_used_source_fragments() -> None:
    sources = manifests(tp=2, prefix="source", address_base=0x10000)
    target = manifests(tp=4, prefix="target", address_base=0x40000)[1]
    plan = plan_runtime_transfer_to_local_target(sources, target)
    used_fragment_ids = {operation.source.fragment_id for operation in plan.operations}
    source_registrations = tuple(
        registration
        for registration in registration_leases(sources)
        if registration.fragment_id in used_fragment_ids
    )

    receipts = MooncakeTransferEngineReader(FakeTransferEngine()).execute(
        plan,
        sources,
        target,
        source_registrations=source_registrations,
        target_pre_registered=True,
        target_registrations=registration_leases((target,)),
    )

    assert sum(receipt.nbytes for receipt in receipts) == 2


def test_te_reader_batches_large_repeats_without_segment_tuple_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repeat = 8192
    tensor = TensorDescriptor(
        tensor_id="layers.0.mlp.down_proj.weight",
        global_shape=(repeat, 8),
        dtype="uint8",
        itemsize=1,
        partition_dim=1,
        layer_id=0,
        layout_fingerprint="sglang:qwen3.5:uint8:test",
    )
    sources = manifests(
        tp=2,
        prefix="source",
        address_base=0x10000,
        tensor=tensor,
    )
    target = manifests(
        tp=1,
        prefix="target",
        address_base=0x40000,
        tensor=tensor,
    )[0]
    plan = plan_runtime_transfer_to_local_target(sources, target)
    state = {"yielded": 0, "exhausted": False, "first_batch": None}

    class StreamingProbeTransferEngine(FakeTransferEngine):
        def batch_transfer_sync_read(
            self,
            endpoint,
            target_addresses,
            source_addresses,
            sizes,
        ):
            if state["first_batch"] is None:
                state["first_batch"] = (state["yielded"], state["exhausted"])
            return super().batch_transfer_sync_read(
                endpoint,
                target_addresses,
                source_addresses,
                sizes,
            )

    engine = StreamingProbeTransferEngine()

    assert [operation.repeat for operation in plan.operations] == [repeat, repeat]
    assert all(isinstance(operation, TransferRegion) for operation in plan.operations)

    original_iter_segments = TransferRegion.iter_segments

    def observe_streaming_segments(self: TransferRegion):
        try:
            for segment in original_iter_segments(self):
                state["yielded"] += 1
                yield segment
        finally:
            state["exhausted"] = True

    monkeypatch.setattr(TransferRegion, "iter_segments", observe_streaming_segments)

    receipts = MooncakeTransferEngineReader(engine, max_batch_operations=1024).execute(
        plan,
        sources,
        target,
        source_registrations=registration_leases(sources),
        target_pre_registered=True,
        target_registrations=registration_leases((target,)),
    )

    endpoints = [call[0] for call in engine.calls]
    assert endpoints == ["source-t0:12345"] * 8 + ["source-t1:12345"] * 8
    assert all(len(call[3]) == 1024 for call in engine.calls)
    assert [receipt.operation_count for receipt in receipts] == [repeat, repeat]
    assert [receipt.nbytes for receipt in receipts] == [repeat * 4, repeat * 4]
    assert state["first_batch"] == (1024, False)
    assert engine.calls[0][1][0] == 0x40000
    assert engine.calls[7][1][-1] == 0x40000 + (repeat - 1) * 8
    assert engine.calls[8][1][0] == 0x40004
    assert engine.calls[15][1][-1] == 0x40004 + (repeat - 1) * 8


def test_te_reader_requires_complete_planned_source_executor_set() -> None:
    sources = manifests(tp=2, prefix="source", address_base=0x10000)
    target = manifests(tp=4, prefix="target", address_base=0x40000)[1]
    plan = plan_runtime_transfer_to_local_target(sources, target)

    with pytest.raises(TransferEngineError, match="source executor set is incomplete"):
        MooncakeTransferEngineReader(FakeTransferEngine()).execute(
            plan,
            sources[:1],
            target,
            source_registrations=registration_leases(sources[:1]),
            target_pre_registered=True,
            target_registrations=registration_leases((target,)),
        )


def test_te_reader_surfaces_source_endpoint_failure() -> None:
    sources = manifests(tp=2, prefix="source", address_base=0x10000)
    target = manifests(tp=4, prefix="target", address_base=0x40000)[1]
    plan = plan_runtime_transfer_to_local_target(sources, target)
    engine = FakeTransferEngine()
    engine.fail_endpoint = "source-t0:12345"

    with pytest.raises(TransferEngineError, match="source-t0:12345"):
        MooncakeTransferEngineReader(engine).execute(
            plan,
            sources,
            target,
            source_registrations=registration_leases(sources),
            target_pre_registered=True,
            target_registrations=registration_leases((target,)),
        )


def test_te_reader_rejects_positive_nonzero_transfer_status() -> None:
    sources = manifests(tp=2, prefix="source", address_base=0x10000)
    target = manifests(tp=4, prefix="target", address_base=0x40000)[1]
    plan = plan_runtime_transfer_to_local_target(sources, target)
    engine = FakeTransferEngine()
    engine.read_result = 5

    with pytest.raises(TransferEngineError, match="failed: 5"):
        MooncakeTransferEngineReader(engine).execute(
            plan,
            sources,
            target,
            source_registrations=registration_leases(sources),
            target_pre_registered=True,
            target_registrations=registration_leases((target,)),
        )


def test_te_reader_rejects_leases_without_pre_registered_mode() -> None:
    sources = manifests(tp=2, prefix="source", address_base=0x10000)
    target = manifests(tp=4, prefix="target", address_base=0x40000)[1]
    plan = plan_runtime_transfer_to_local_target(sources, target)
    engine = FakeTransferEngine()

    with pytest.raises(TransferEngineError, match="target_pre_registered"):
        MooncakeTransferEngineReader(engine).execute(
            plan,
            sources,
            target,
            source_registrations=registration_leases(sources),
            target_registrations=registration_leases((target,)),
        )

    assert engine.register_calls == []
    assert engine.unregister_calls == []
