from __future__ import annotations

import socket
import threading
from dataclasses import dataclass

import pytest

from benchmarks.heterogeneous_weight_reshard.case_spec import BenchmarkCase, MeshSpec
from benchmarks.heterogeneous_weight_reshard.mooncake_runtime import (
    LiveMeshAllocation,
    _expect_control,
    handle_target_connection,
    run_source_session,
    verify_target_allocation,
)


def _case(
    *,
    source_replicas: int = 1,
    source_shards: int = 4,
    source_dim: int = 0,
    target_shards: int = 4,
    target_dim: int = 1,
) -> BenchmarkCase:
    return BenchmarkCase(
        id="runtime-case",
        category="physical",
        source=MeshSpec(
            replicas=source_replicas,
            shards=source_shards,
            shard_dim=source_dim,
        ),
        target=MeshSpec(replicas=1, shards=target_shards, shard_dim=target_dim),
        global_shape=(8, 8),
        required_ranks=source_replicas * source_shards + target_shards,
    )


@dataclass
class FakeRuntime:
    device: int


class FakeBuffer:
    next_pointer = 0x100000
    buffers: list[FakeBuffer] = []

    def __init__(self, runtime: FakeRuntime, size: int, events: list[tuple]) -> None:
        self.device = runtime.device
        self.size = size
        self.pointer = FakeBuffer.next_pointer
        FakeBuffer.next_pointer += 0x1000
        self.data = bytearray(size)
        self.events = events
        self.closed = False
        FakeBuffer.buffers.append(self)
        self.events.append(("allocate", self.device, self.pointer, self.size))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def activate(self) -> None:
        self.events.append(("activate", self.device, self.pointer))

    def fill(self, value: int) -> None:
        self.data[:] = bytes([value]) * self.size

    def zero(self) -> None:
        self.fill(0)

    def read_range(self, offset: int, nbytes: int) -> bytes:
        return bytes(self.data[offset : offset + nbytes])

    def close(self) -> None:
        if not self.closed:
            self.events.append(("free", self.device, self.pointer))
            self.closed = True

    @classmethod
    def locate(cls, address: int, nbytes: int):
        for buffer in cls.buffers:
            if (
                not buffer.closed
                and buffer.pointer <= address
                and address + nbytes <= buffer.pointer + buffer.size
            ):
                return buffer, address - buffer.pointer
        raise RuntimeError(f"unmapped fake address: {address} + {nbytes}")


class FakeEngine:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events

    def register_memory(self, address: int, nbytes: int) -> int:
        self.events.append(("register", address, nbytes))
        return 0

    def unregister_memory(self, address: int) -> int:
        self.events.append(("unregister", address))
        return 0

    def batch_transfer_sync_write(
        self, endpoint, source_addresses, target_addresses, sizes
    ) -> int:
        self.events.append(("transfer", endpoint, len(sizes), sum(sizes)))
        for source, target, nbytes in zip(
            source_addresses, target_addresses, sizes, strict=True
        ):
            source_buffer, source_offset = FakeBuffer.locate(source, nbytes)
            target_buffer, target_offset = FakeBuffer.locate(target, nbytes)
            target_buffer.data[target_offset : target_offset + nbytes] = (
                source_buffer.data[source_offset : source_offset + nbytes]
            )
        return 0


def _open_allocation(case: BenchmarkCase, *, side: str):
    events = []
    engine = FakeEngine(events)
    allocation = LiveMeshAllocation.open(
        case=case,
        side=side,
        engine=engine,
        endpoint=f"{side}:12345",
        cuda_devices=tuple(range(8)),
        revision="revision-1",
        session_id="session-1",
        generation=7,
        runtime_factory=FakeRuntime,
        buffer_factory=lambda runtime, size: FakeBuffer(runtime, size, events),
    )
    return allocation, events


def test_live_allocation_registers_before_publish_and_unregisters_before_free() -> None:
    allocation, events = _open_allocation(_case(), side="target")

    assert len(allocation.buffers) == 4
    assert len(allocation.manifests) == 4
    assert len(allocation.registrations) == 4
    assert [buffer.device for buffer in allocation.buffers] == [0, 1, 2, 3]
    assert all(
        manifest.fragments[0].lease_generation == 7 for manifest in allocation.manifests
    )
    assert all(
        envelope.session_id == "session-1" and envelope.generation == 7
        for envelope in allocation.registrations
    )
    assert len([event for event in events if event[0] == "register"]) == 4

    allocation.close()

    unregister_indices = [
        index for index, event in enumerate(events) if event[0] == "unregister"
    ]
    free_indices = [index for index, event in enumerate(events) if event[0] == "free"]
    assert unregister_indices
    assert free_indices
    assert max(unregister_indices) < min(free_indices)


def test_source_fill_pattern_distinguishes_dp_replicas_and_shards() -> None:
    allocation, _ = _open_allocation(
        _case(source_replicas=2, source_shards=2, target_shards=4),
        side="source",
    )
    try:
        allocation.fill_source_pattern()
        assert [buffer.data[0] for buffer in allocation.buffers] == [1, 2, 3, 4]
    finally:
        allocation.close()


def test_target_validation_checks_cross_dim_content_in_bounded_chunks() -> None:
    case = _case(source_dim=0, target_dim=1)
    allocation, _ = _open_allocation(case, side="target")
    try:
        for manifest, buffer in zip(
            allocation.manifests, allocation.buffers, strict=True
        ):
            fragment = manifest.fragments[0]
            for offset, nbytes, value in allocation.expected_segments(
                fragment,
                selected_source_replica=0,
            ):
                buffer.data[offset : offset + nbytes] = bytes([value]) * nbytes

        result = verify_target_allocation(
            allocation,
            selected_source_replica=0,
            chunk_bytes=3,
        )
        assert result.checked_bytes == case.logical_bytes
        assert result.fragment_count == 4

        allocation.buffers[2].data[5] ^= 0xFF
        with pytest.raises(RuntimeError, match="target content mismatch"):
            verify_target_allocation(
                allocation,
                selected_source_replica=0,
                chunk_bytes=3,
            )
    finally:
        allocation.close()


def test_live_mesh_allocation_requires_one_device_per_logical_rank() -> None:
    case = _case(target_shards=4)
    with pytest.raises(ValueError, match="requires 4 CUDA devices"):
        LiveMeshAllocation.open(
            case=case,
            side="target",
            engine=FakeEngine([]),
            endpoint="target:12345",
            cuda_devices=(0, 1, 2),
            revision="revision-1",
            session_id="session-1",
            generation=7,
            runtime_factory=FakeRuntime,
            buffer_factory=lambda runtime, size: FakeBuffer(runtime, size, []),
        )


def test_control_error_preserves_target_detail() -> None:
    with pytest.raises(RuntimeError, match="target error: allocation failed"):
        _expect_control(
            {
                "type": "error",
                "schema_version": 1,
                "session_id": "session-1",
                "generation": 7,
                "detail": "allocation failed",
            },
            message_type="ready",
            session_id="session-1",
            generation=7,
        )


def test_source_target_session_runs_planner_sink_reset_and_validation() -> None:
    FakeBuffer.buffers.clear()
    case = _case(source_shards=2, target_shards=4, source_dim=0, target_dim=1)
    target_socket, source_socket = socket.socketpair()
    target_events: list[tuple] = []
    source_events: list[tuple] = []
    target_errors = []

    def target_worker() -> None:
        try:
            handle_target_connection(
                target_socket,
                engine=FakeEngine(target_events),
                endpoint="target:13000",
                cuda_devices=(4, 5, 6, 7),
                transport_init_seconds=0.012,
                runtime_factory=FakeRuntime,
                buffer_factory=lambda runtime, size: FakeBuffer(
                    runtime, size, target_events
                ),
                timeout_s=5.0,
            )
        except BaseException as error:
            target_errors.append(error)
        finally:
            target_socket.close()

    thread = threading.Thread(target=target_worker)
    thread.start()
    source_allocation = LiveMeshAllocation.open(
        case=case,
        side="source",
        engine=FakeEngine(source_events),
        endpoint="source:12000",
        cuda_devices=(0, 1),
        revision="revision-1",
        session_id="session-1",
        generation=3,
        runtime_factory=FakeRuntime,
        buffer_factory=lambda runtime, size: FakeBuffer(runtime, size, source_events),
    )
    source_allocation.fill_source_pattern()
    try:
        result = run_source_session(
            source_socket,
            case=case,
            revision="revision-1",
            session_id="session-1",
            generation=3,
            engine=FakeEngine(source_events),
            source_allocation=source_allocation,
            source_transport_init_seconds=0.010,
            warmups=2,
            iterations=3,
        )
    finally:
        source_socket.close()
        source_allocation.close()
        thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert target_errors == []
    assert result["backend"] == "mooncake-te-runtime-manifest"
    assert result["case_id"] == case.id
    assert result["validation"]["passed"] is True
    assert result["validation"]["checked_bytes"] == case.logical_bytes
    assert result["first_update"]["receipt_bytes"] == case.logical_bytes
    assert result["steady_update"]["total"]["count"] == 3
    assert "process_wall_ms" not in result
    assert result["protocol_wall_ms"] > 0
    assert result["target_protocol_wall_ms"] > 0
    assert result["control_plane_ms"]["source_transport_init"] == 10.0
    assert result["control_plane_ms"]["target_transport_init"] == 12.0
    assert len([event for event in source_events if event[0] == "transfer"]) > 0
    assert len([event for event in target_events if event[0] == "unregister"]) == 4


def test_source_target_cold_session_runs_exactly_one_unvalidated_update() -> None:
    FakeBuffer.buffers.clear()
    case = _case(source_shards=1, target_shards=1, source_dim=0, target_dim=0)
    target_socket, source_socket = socket.socketpair()
    target_events: list[tuple] = []
    source_events: list[tuple] = []
    target_errors = []

    def target_worker() -> None:
        try:
            handle_target_connection(
                target_socket,
                engine=FakeEngine(target_events),
                endpoint="target:13000",
                cuda_devices=(1,),
                transport_init_seconds=0.012,
                runtime_factory=FakeRuntime,
                buffer_factory=lambda runtime, size: FakeBuffer(
                    runtime, size, target_events
                ),
                timeout_s=5.0,
            )
        except BaseException as error:
            target_errors.append(error)
        finally:
            target_socket.close()

    thread = threading.Thread(target=target_worker)
    thread.start()
    source_allocation = LiveMeshAllocation.open(
        case=case,
        side="source",
        engine=FakeEngine(source_events),
        endpoint="source:12000",
        cuda_devices=(0,),
        revision="revision-cold",
        session_id="session-cold",
        generation=4,
        runtime_factory=FakeRuntime,
        buffer_factory=lambda runtime, size: FakeBuffer(runtime, size, source_events),
    )
    source_allocation.fill_source_pattern()
    try:
        result = run_source_session(
            source_socket,
            case=case,
            revision="revision-cold",
            session_id="session-cold",
            generation=4,
            engine=FakeEngine(source_events),
            source_allocation=source_allocation,
            source_transport_init_seconds=0.010,
            warmups=1,
            iterations=0,
            one_shot=True,
        )
    finally:
        source_socket.close()
        source_allocation.close()
        thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert target_errors == []
    assert result["phase"] == "cold"
    assert result["steady_update"] is None
    assert result["validation"] == {"passed": None}
    assert result["validation_update"] is None
    assert len([event for event in source_events if event[0] == "transfer"]) == 1
