from __future__ import annotations

from dataclasses import dataclass

import pytest

from benchmarks.heterogeneous_weight_reshard.case_spec import BenchmarkCase, MeshSpec
from benchmarks.heterogeneous_weight_reshard.mooncake_executor import (
    RegistrationEnvelope,
    TimedTransferEngine,
    case_from_wire,
    case_to_wire,
    execute_update,
    expected_fragment_segments,
    parse_prepare_message,
    parse_ready_message,
    prepare_message,
    ready_message,
    summarize_samples,
)
from benchmarks.heterogeneous_weight_reshard.runtime_layout import (
    build_runtime_manifests,
    registration_leases,
)
from mooncake.model_weight import MooncakeTransferEngineSink, plan_runtime_transfer


@dataclass
class FakeBuffer:
    pointer: int
    size: int


def _case(
    *,
    source_dim: int = 0,
    target_dim: int = 1,
    source_shards: int = 4,
    target_shards: int = 4,
) -> BenchmarkCase:
    return BenchmarkCase(
        id="cross_dim",
        category="physical",
        source=MeshSpec(replicas=1, shards=source_shards, shard_dim=source_dim),
        target=MeshSpec(replicas=1, shards=target_shards, shard_dim=target_dim),
        global_shape=(8, 8),
        required_ranks=source_shards + target_shards,
    )


def _target_manifests(case: BenchmarkCase):
    assert case.target is not None
    size = case.logical_bytes // case.target.shards
    buffers = [
        FakeBuffer(pointer=0x200000 + rank * 0x1000, size=size)
        for rank in range(case.target.total_ranks)
    ]
    return build_runtime_manifests(
        case,
        side="target",
        buffers=buffers,
        endpoint="172.16.1.108:13000",
        revision="revision-1",
        lease_generation=9,
    )


def test_case_wire_round_trip_keeps_only_explicit_geometry() -> None:
    case = _case()

    restored = case_from_wire(case_to_wire(case))

    assert restored == case
    assert set(case_to_wire(case)) == {
        "id",
        "category",
        "source",
        "target",
        "global_shape",
        "required_ranks",
    }


def test_prepare_message_round_trip_binds_session_generation_and_revision() -> None:
    message = prepare_message(
        session_id="session-1",
        generation=9,
        case=_case(),
        revision="revision-1",
    )

    prepared = parse_prepare_message(message)

    assert prepared.session_id == "session-1"
    assert prepared.generation == 9
    assert prepared.case == _case()
    assert prepared.revision == "revision-1"

    message["unexpected"] = True
    with pytest.raises(ValueError, match="prepare message fields"):
        parse_prepare_message(message)


def test_ready_message_round_trip_validates_registration_envelopes() -> None:
    case = _case()
    manifests = _target_manifests(case)
    envelopes = tuple(
        RegistrationEnvelope(
            allocation_id=f"allocation-{rank}",
            fragment_id=manifest.fragments[0].fragment_id,
            device_id=rank,
            base_address=manifest.fragments[0].address,
            nbytes=manifest.fragments[0].nbytes,
            session_id="session-1",
            generation=9,
        )
        for rank, manifest in enumerate(manifests)
    )
    message = ready_message(
        session_id="session-1",
        generation=9,
        endpoint="172.16.1.108:13000",
        manifests=manifests,
        registrations=envelopes,
        transport_init_ms=12.5,
        allocation_ms=4.75,
        registration_ms=3.25,
    )

    ready = parse_ready_message(
        message,
        expected_session_id="session-1",
        expected_generation=9,
    )

    assert ready.endpoint == "172.16.1.108:13000"
    assert ready.manifests == manifests
    assert ready.registrations == envelopes
    assert ready.transport_init_ms == pytest.approx(12.5)
    assert ready.allocation_ms == pytest.approx(4.75)
    assert ready.registration_ms == pytest.approx(3.25)


def test_ready_message_rejects_stale_generation_and_out_of_bounds_fragment() -> None:
    manifests = _target_manifests(_case())
    fragment = manifests[0].fragments[0]
    envelope = RegistrationEnvelope(
        allocation_id="allocation-0",
        fragment_id=fragment.fragment_id,
        device_id=0,
        base_address=fragment.address + 1,
        nbytes=fragment.nbytes,
        session_id="session-1",
        generation=9,
    )
    message = ready_message(
        session_id="session-1",
        generation=9,
        endpoint="172.16.1.108:13000",
        manifests=(manifests[0],),
        registrations=(envelope,),
        transport_init_ms=1.0,
        allocation_ms=1.0,
        registration_ms=1.0,
    )

    with pytest.raises(ValueError, match="registration bounds"):
        parse_ready_message(
            message,
            expected_session_id="session-1",
            expected_generation=9,
        )

    message["generation"] = 8
    with pytest.raises(ValueError, match="generation mismatch"):
        parse_ready_message(
            message,
            expected_session_id="session-1",
            expected_generation=9,
        )


def test_cross_dim_expected_segments_are_grouped_by_source_shard() -> None:
    case = _case(source_dim=0, target_dim=1)
    fragment = _target_manifests(case)[2].fragments[0]

    segments = tuple(expected_fragment_segments(case, fragment))

    assert segments == (
        (0, 4, 1),
        (4, 4, 2),
        (8, 4, 3),
        (12, 4, 4),
    )
    assert sum(nbytes for _, nbytes, _ in segments) == fragment.nbytes


def test_same_dim_expected_segment_does_not_expand_rows_or_elements() -> None:
    case = _case(source_dim=0, target_dim=0, target_shards=8)
    fragment = _target_manifests(case)[1].fragments[0]

    assert tuple(expected_fragment_segments(case, fragment)) == ((0, 8, 1),)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeEngine:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls = []
        self.marker = "delegated"

    def batch_transfer_sync_write(self, endpoint, sources, targets, sizes):
        self.calls.append((endpoint, sources, targets, sizes))
        self.clock.advance(0.025)
        return 0

    def batch_transfer_sync_write_with_ticket(self, endpoint, sources, targets, sizes):
        self.calls.append((endpoint, sources, targets, sizes))
        self.clock.advance(0.025)
        return type("CompletedTicket", (), {"status": "COMPLETED"})()


class LegacyFakeEngine:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls = []

    def batch_transfer_sync_write(self, endpoint, sources, targets, sizes):
        self.calls.append((endpoint, sources, targets, sizes))
        self.clock.advance(0.025)
        return 0


def test_timed_transfer_engine_measures_native_batches_and_delegates() -> None:
    clock = FakeClock()
    engine = FakeEngine(clock)
    timed = TimedTransferEngine(engine, clock=clock)

    result = timed.batch_transfer_sync_write("target:1", [1, 2], [3, 4], [5, 6])

    assert result == 0
    assert timed.native_transfer_seconds == pytest.approx(0.025)
    assert timed.batch_count == 1
    assert timed.operation_count == 2
    assert timed.wire_bytes == 11
    assert timed.marker == "delegated"
    timed.reset_measurements()
    assert timed.native_transfer_seconds == 0
    assert timed.batch_count == 0


def test_timed_transfer_engine_preserves_legacy_api_detection() -> None:
    case = _case(source_dim=0, target_dim=0, source_shards=1, target_shards=1)
    sources = build_runtime_manifests(
        case,
        side="source",
        buffers=[FakeBuffer(pointer=0x100000, size=64)],
        endpoint="172.16.1.107:12000",
        revision="revision-1",
    )
    targets = build_runtime_manifests(
        case,
        side="target",
        buffers=[FakeBuffer(pointer=0x200000, size=64)],
        endpoint="172.16.1.108:13000",
        revision="revision-1",
    )
    clock = FakeClock()
    engine = LegacyFakeEngine(clock)
    timed = TimedTransferEngine(engine, clock=clock)

    assert getattr(timed, "batch_transfer_sync_write_with_ticket", None) is None
    sample = execute_update(
        sink=MooncakeTransferEngineSink(timed),
        timed_engine=timed,
        plan=plan_runtime_transfer(sources, targets),
        source_manifests=sources,
        target_manifests=targets,
        source_registrations=registration_leases(sources),
        target_registrations=registration_leases(targets),
        clock=clock,
    )

    assert len(engine.calls) == 1
    assert sample.receipt_bytes == 64
    assert sample.wire_bytes == 64
    assert sample.operation_count == 1
    assert sample.batch_count == 1
    assert sample.native_transfer_seconds == pytest.approx(0.025)


def test_execute_update_uses_live_plan_leases_and_reports_full_update_time() -> None:
    case = _case(source_dim=0, target_dim=0, source_shards=2, target_shards=4)
    sources = build_runtime_manifests(
        case,
        side="source",
        buffers=[
            FakeBuffer(pointer=0x100000 + rank * 0x1000, size=32) for rank in range(2)
        ],
        endpoint="172.16.1.107:12000",
        revision="revision-1",
    )
    targets = build_runtime_manifests(
        case,
        side="target",
        buffers=[
            FakeBuffer(pointer=0x200000 + rank * 0x1000, size=16) for rank in range(4)
        ],
        endpoint="172.16.1.108:13000",
        revision="revision-1",
    )
    plan = plan_runtime_transfer(sources, targets)
    clock = FakeClock()
    timed_engine = TimedTransferEngine(FakeEngine(clock), clock=clock)
    sink = MooncakeTransferEngineSink(timed_engine)

    sample = execute_update(
        sink=sink,
        timed_engine=timed_engine,
        plan=plan,
        source_manifests=sources,
        target_manifests=targets,
        source_registrations=registration_leases(sources),
        target_registrations=registration_leases(targets),
        clock=clock,
    )

    assert sample.receipt_bytes == 64
    assert sample.wire_bytes == 64
    assert sample.operation_count == 4
    assert sample.batch_count > 0
    assert sample.total_seconds == pytest.approx(sample.native_transfer_seconds)
    assert sample.host_dispatch_seconds == pytest.approx(0.0)


def test_sample_summary_reports_e2e_latency_and_logical_throughput() -> None:
    summary = summarize_samples((1.0, 2.0, 3.0, 4.0), logical_bytes=2**30)

    assert summary["count"] == 4
    assert summary["mean_ms"] == pytest.approx(2500.0)
    assert summary["p50_ms"] == pytest.approx(2500.0)
    assert summary["p95_ms"] == pytest.approx(3850.0)
    assert summary["p50_logical_gibps"] == pytest.approx(0.4)
