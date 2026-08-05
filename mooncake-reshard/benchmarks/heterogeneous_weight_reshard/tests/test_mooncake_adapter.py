import pytest

from benchmarks.heterogeneous_weight_reshard.case_spec import (
    BenchmarkCase,
    MeshSpec,
)
from benchmarks.heterogeneous_weight_reshard.mooncake_adapter import (
    MooncakePlanningError,
    plan_mooncake_case,
)
from mooncake.reshard.weight.planner import TransferPlan, TransferRegion


GIB = 1024**3


def _case(
    *,
    source_replicas: int = 1,
    source_shards: int,
    source_dim: int,
    target_replicas: int = 1,
    target_shards: int,
    target_dim: int,
    shape: tuple[int, ...],
    category: str = "physical",
    case_id: str = "neutral_tensor_case",
) -> BenchmarkCase:
    source = MeshSpec(source_replicas, source_shards, source_dim)
    target = MeshSpec(target_replicas, target_shards, target_dim)
    return BenchmarkCase(
        id=case_id,
        category=category,
        source=source,
        target=target,
        global_shape=shape,
        required_ranks=source.total_ranks + target.total_ranks,
    )


@pytest.mark.parametrize(
    (
        "case",
        "runtime_binding_count",
        "region_count",
        "segment_count",
        "max_segments_per_region",
        "inner_bytes",
        "target_batch_counts",
        "plan_total_bytes",
    ),
    [
        (
            _case(
                source_shards=4,
                source_dim=0,
                target_shards=8,
                target_dim=0,
                shape=(2048, 2048, 2048),
            ),
            12,
            8,
            8,
            1,
            GIB,
            (1,) * 8,
            8 * GIB,
        ),
        (
            _case(
                source_shards=8,
                source_dim=0,
                target_shards=4,
                target_dim=0,
                shape=(2048, 2048, 2048),
            ),
            12,
            8,
            8,
            1,
            GIB,
            (2,) * 4,
            8 * GIB,
        ),
        (
            _case(
                source_shards=4,
                source_dim=0,
                target_shards=4,
                target_dim=1,
                shape=(2048, 2048, 2048),
            ),
            8,
            16,
            8192,
            512,
            1024**2,
            (4,) * 4,
            8 * GIB,
        ),
        (
            _case(
                source_shards=4,
                source_dim=0,
                target_shards=4,
                target_dim=2,
                shape=(2048, 2048, 2048),
                category="stress",
            ),
            8,
            16,
            16_777_216,
            1_048_576,
            512,
            (4096,) * 4,
            8 * GIB,
        ),
        (
            _case(
                source_shards=4,
                source_dim=0,
                target_shards=4,
                target_dim=2,
                shape=(128, 4096, 16384),
            ),
            8,
            16,
            2_097_152,
            131_072,
            4096,
            (512,) * 4,
            8 * GIB,
        ),
        (
            _case(
                source_replicas=2,
                source_shards=4,
                source_dim=0,
                target_shards=8,
                target_dim=0,
                shape=(2048, 2048, 1024),
            ),
            16,
            8,
            8,
            1,
            512 * 1024**2,
            (1,) * 8,
            4 * GIB,
        ),
    ],
)
def test_summarizes_static_nd_plan_exactly(
    case: BenchmarkCase,
    runtime_binding_count: int,
    region_count: int,
    segment_count: int,
    max_segments_per_region: int,
    inner_bytes: int,
    target_batch_counts: tuple[int, ...],
    plan_total_bytes: int,
) -> None:
    result = plan_mooncake_case(case)

    assert result.summary.placement_count == 2
    assert result.summary.runtime_binding_count == runtime_binding_count
    assert result.summary.region_count == region_count
    assert result.summary.total_segment_count == segment_count
    assert result.summary.max_segments_per_region == max_segments_per_region
    assert result.summary.inner_bytes == inner_bytes
    assert result.summary.target_batch_counts == target_batch_counts
    assert result.summary.plan_total_bytes == plan_total_bytes
    assert isinstance(result.plan, TransferPlan)
    assert result.plan.total_bytes == plan_total_bytes


def test_uses_nonzero_non_overlapping_fake_fragment_addresses() -> None:
    result = plan_mooncake_case(
        _case(
            source_shards=4,
            source_dim=0,
            target_shards=4,
            target_dim=1,
            shape=(2048, 2048, 2048),
            case_id="model.q_proj.must_not_define_geometry",
        )
    )

    ranges = sorted(result.fake_address_ranges)
    placements = (result.source_placement, result.target_placement)
    bindings = (*result.source_bindings, *result.target_bindings)
    placement_fragments = tuple(
        fragment for placement in placements for fragment in placement.fragments
    )
    binding_fragments = tuple(
        fragment for binding in bindings for fragment in binding.fragments
    )

    assert len(ranges) == result.summary.runtime_binding_count
    assert len(placements) == result.summary.placement_count
    assert len(bindings) == result.summary.runtime_binding_count
    assert ranges == sorted(
        (fragment.address, fragment.address + fragment.nbytes)
        for fragment in binding_fragments
    )
    assert result.tensor_id == "benchmark_tensor"
    assert all(
        fragment.tensor_id == result.tensor_id for fragment in placement_fragments
    )
    assert all(binding.generation == 1 for binding in bindings)
    assert all(start > 0 and end > start for start, end in ranges)
    assert all(
        left_end <= right_start
        for (_, left_end), (right_start, _) in zip(ranges, ranges[1:])
    )


def test_dp_source_replicas_are_selected_without_logical_duplication() -> None:
    result = plan_mooncake_case(
        _case(
            source_replicas=2,
            source_shards=4,
            source_dim=0,
            target_shards=8,
            target_dim=0,
            shape=(2048, 2048, 1024),
        )
    )

    assert result.summary.source_replica_count == 2
    assert result.summary.selected_source_replicas == (0,)
    assert result.summary.selected_source_fragment_count == 4
    assert result.summary.deduplicated_source_fragment_count == 4
    assert result.summary.region_count == 8
    assert result.summary.plan_total_bytes == 4 * GIB


def test_summary_never_materializes_transfer_segments(monkeypatch) -> None:
    def fail_if_materialized(self):
        raise AssertionError(f"materialized {self.segment_count} segments")

    monkeypatch.setattr(TransferRegion, "iter_segments", fail_if_materialized)
    case = _case(
        source_shards=4,
        source_dim=0,
        target_shards=4,
        target_dim=2,
        shape=(128, 4096, 16384),
    )

    summary = plan_mooncake_case(case).summary

    assert summary.total_segment_count == 2_097_152
    assert summary.max_segments_per_region == 131_072
    assert summary.target_batch_counts == (512,) * 4


def test_default_bounded_lowering_rejects_only_oversized_regions() -> None:
    stress_case = _case(
        source_shards=4,
        source_dim=0,
        target_shards=4,
        target_dim=2,
        shape=(2048, 2048, 2048),
        category="stress",
    )
    physical_case = _case(
        source_shards=4,
        source_dim=0,
        target_shards=4,
        target_dim=2,
        shape=(128, 4096, 16384),
    )

    rejected = plan_mooncake_case(stress_case).summary
    accepted = plan_mooncake_case(physical_case).summary

    assert rejected.bounded_lowering_allowed is False
    assert rejected.bounded_lowering_refusal_reasons == (
        "region requires 1048576 segments, exceeding max_region_segments 1000000",
    )
    assert accepted.bounded_lowering_allowed is True
    assert accepted.bounded_lowering_refusal_reasons == ()


def test_custom_bound_can_admit_the_dim2_stress_plan() -> None:
    case = _case(
        source_shards=4,
        source_dim=0,
        target_shards=4,
        target_dim=2,
        shape=(2048, 2048, 2048),
        category="stress",
    )

    summary = plan_mooncake_case(case, max_region_segments=1_048_576).summary

    assert summary.bounded_lowering_allowed is True
    assert summary.bounded_lowering_refusal_reasons == ()


def test_planner_only_case_without_geometry_fails_clearly() -> None:
    case = BenchmarkCase(
        id="planner_only_without_geometry",
        category="planner_only",
        reason="rank count exceeds the physical cluster",
    )

    with pytest.raises(
        MooncakePlanningError,
        match=(
            "planner_only_without_geometry: Mooncake planning requires source, "
            "target, and global_shape geometry"
        ),
    ):
        plan_mooncake_case(case)
