from __future__ import annotations

from dataclasses import replace

import pytest

from mooncake.reshard.weight.te import (
    MooncakeTransferEngineSink,
    TransferEngineError,
)

from .helpers import (
    FakeTransferEngine,
    RuntimeInputs,
    manifests,
    plan_transfer,
    registration_leases,
    with_revision,
)


def test_te_sink_executes_local_source_ranges_without_staging_buffer() -> None:
    sources = manifests(tp=2, prefix="source", address_base=0x10000)
    targets = manifests(tp=4, prefix="target", address_base=0x40000)
    plan = plan_transfer(sources, targets)
    engine = FakeTransferEngine()
    sink = MooncakeTransferEngineSink(engine)

    receipts = sink.execute(
        plan,
        sources.placement,
        sources.bindings[0],
        targets.placement,
        targets.bindings,
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
    plan = plan_transfer(sources, targets)
    sink = MooncakeTransferEngineSink(FakeTransferEngine())

    with pytest.raises(TransferEngineError, match="target registration"):
        sink.execute(
            plan,
            sources.placement,
            sources.bindings[0],
            targets.placement,
            targets.bindings,
        )

    stale_leases = list(registration_leases(targets))
    stale_leases[0] = replace(stale_leases[0], lease_generation=2)
    with pytest.raises(TransferEngineError, match="target registration"):
        sink.execute(
            plan,
            sources.placement,
            sources.bindings[0],
            targets.placement,
            targets.bindings,
            target_registrations=tuple(stale_leases),
        )


def test_te_sink_surfaces_endpoint_failure() -> None:
    sources = manifests(tp=2, prefix="source", address_base=0x10000)
    targets = manifests(tp=4, prefix="target", address_base=0x40000)
    plan = plan_transfer(sources, targets)
    engine = FakeTransferEngine()
    engine.fail_endpoint = "target-t1:12345"

    with pytest.raises(TransferEngineError, match="target-t1:12345"):
        MooncakeTransferEngineSink(engine).execute(
            plan,
            sources.placement,
            sources.bindings[0],
            targets.placement,
            targets.bindings,
            target_registrations=registration_leases(targets),
        )


def test_te_sink_wraps_write_exception() -> None:
    sources = manifests(tp=2, prefix="source", address_base=0x10000)
    targets = manifests(tp=4, prefix="target", address_base=0x40000)
    plan = plan_transfer(sources, targets)
    engine = FakeTransferEngine()

    def fail_write(*args, **kwargs):
        raise RuntimeError("write exploded")

    engine.batch_transfer_sync_write = fail_write

    with pytest.raises(TransferEngineError, match="write exploded"):
        MooncakeTransferEngineSink(engine).execute(
            plan,
            sources.placement,
            sources.bindings[0],
            targets.placement,
            targets.bindings,
            target_registrations=registration_leases(targets),
        )

    assert engine.unregister_calls == [0x10000]


def test_te_sink_rejects_stale_source_generation() -> None:
    sources = manifests(tp=2, prefix="source", address_base=0x10000)
    targets = manifests(tp=4, prefix="target", address_base=0x40000)
    plan = plan_transfer(sources, targets)
    stale_binding = replace(sources.bindings[0], generation=2)

    with pytest.raises(TransferEngineError, match="source executor snapshot mismatch"):
        MooncakeTransferEngineSink(FakeTransferEngine()).execute(
            plan,
            sources.placement,
            stale_binding,
            targets.placement,
            targets.bindings,
            target_registrations=registration_leases(targets),
        )


def test_te_sink_rejects_generation_scoped_source_id_rollover() -> None:
    sources = manifests(tp=2, prefix="source", address_base=0x10000)
    targets = manifests(tp=4, prefix="target", address_base=0x40000)
    plan = plan_transfer(sources, targets)
    replacement = replace(
        sources.bindings[0].fragments[0],
        fragment_id="replacement-source-fragment",
        worker_id="replacement-source-worker",
    )
    current_binding = replace(
        sources.bindings[0],
        instance_id="replacement-source-instance",
        generation=2,
        fragments=(replacement,),
    )

    with pytest.raises(TransferEngineError, match="source executor snapshot mismatch"):
        MooncakeTransferEngineSink(FakeTransferEngine()).execute(
            plan,
            sources.placement,
            current_binding,
            targets.placement,
            targets.bindings,
            target_registrations=registration_leases(targets),
        )


def test_te_sink_rejects_stale_target_address_and_generation() -> None:
    sources = manifests(tp=2, prefix="source", address_base=0x10000)
    targets = manifests(tp=4, prefix="target", address_base=0x40000)
    plan = plan_transfer(sources, targets)
    replacement = replace(
        targets.bindings[0].fragments[0],
        address=0x90000,
        storage_address=0x90000,
    )
    current_bindings = (
        replace(targets.bindings[0], fragments=(replacement,), generation=2),
        *targets.bindings[1:],
    )

    with pytest.raises(TransferEngineError, match="target executor snapshot mismatch"):
        MooncakeTransferEngineSink(FakeTransferEngine()).execute(
            plan,
            sources.placement,
            sources.bindings[0],
            targets.placement,
            current_bindings,
            target_registrations=registration_leases(targets),
        )


def test_te_sink_rejects_generation_scoped_target_id_rollover() -> None:
    sources = manifests(tp=2, prefix="source", address_base=0x10000)
    targets = manifests(tp=4, prefix="target", address_base=0x40000)
    plan = plan_transfer(sources, targets)
    replacement = replace(
        targets.bindings[0].fragments[0],
        fragment_id="replacement-target-fragment",
    )
    current_bindings = (
        replace(targets.bindings[0], fragments=(replacement,), generation=2),
        *targets.bindings[1:],
    )

    with pytest.raises(TransferEngineError, match="target executor snapshot mismatch"):
        MooncakeTransferEngineSink(FakeTransferEngine()).execute(
            plan,
            sources.placement,
            sources.bindings[0],
            targets.placement,
            current_bindings,
            target_registrations=registration_leases(targets),
        )


@pytest.mark.parametrize("side", ["source", "target"])
def test_te_sink_rejects_revision_mismatch(side: str) -> None:
    sources = manifests(tp=2, prefix="source", address_base=0x10000)
    targets = manifests(tp=4, prefix="target", address_base=0x40000)
    plan = plan_transfer(sources, targets)
    if side == "source":
        sources = with_revision(sources, "step-43")
    else:
        targets = with_revision(targets, "step-43")

    with pytest.raises(TransferEngineError, match="revision mismatch"):
        MooncakeTransferEngineSink(FakeTransferEngine()).execute(
            plan,
            sources.placement,
            sources.bindings[0],
            targets.placement,
            targets.bindings,
            target_registrations=registration_leases(targets),
        )


def test_te_receipt_identifies_worker_instead_of_serving_instance() -> None:
    source = manifests(tp=1, prefix="source", address_base=0x10000)
    source = RuntimeInputs(
        source.placement,
        (
            replace(
                source.bindings[0],
                instance_id="serving-instance",
            ),
        ),
    )
    targets = manifests(tp=1, prefix="target", address_base=0x40000)
    plan = plan_transfer(source, targets)

    receipts = MooncakeTransferEngineSink(FakeTransferEngine()).execute(
        plan,
        source.placement,
        source.bindings[0],
        targets.placement,
        targets.bindings,
        target_registrations=registration_leases(targets),
    )

    assert receipts[0].source_worker_id == "source-t0"
