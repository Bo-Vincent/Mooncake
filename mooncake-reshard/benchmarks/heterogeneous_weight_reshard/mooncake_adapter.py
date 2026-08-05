"""Mooncake N-D planner 的纯控制面 benchmark adapter。"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from mooncake.reshard.weight import (
    WeightPlacementManifest,
    WeightRuntimeBindingManifest,
)
from mooncake.reshard.weight.planner import (
    TransferPlan,
    TransferRegion,
    bind_logical_transfer_plan,
    plan_placement_transfer,
)

from .case_spec import BenchmarkCase, MeshSpec
from .runtime_layout import build_runtime_topology


_MODEL_ID = "benchmark_model"
_REVISION = "dry_run"
_FAKE_ADDRESS_BASE = 0x1_0000_0000
_FAKE_ADDRESS_GAP = 4096
_DEFAULT_MAX_BATCH_OPERATIONS = 1024
_DEFAULT_MAX_REGION_SEGMENTS = 1_000_000
NEUTRAL_TENSOR_ID = "benchmark_tensor"


class MooncakePlanningError(ValueError):
    """Benchmark case 无法生成 Mooncake 静态计划。"""


@dataclass(frozen=True)
class MooncakePlanSummary:
    placement_count: int
    runtime_binding_count: int
    region_count: int
    total_segment_count: int
    max_segments_per_region: int
    inner_bytes: int
    target_batch_counts: tuple[int, ...]
    plan_total_bytes: int
    source_replica_count: int
    selected_source_replicas: tuple[int, ...]
    selected_source_fragment_count: int
    deduplicated_source_fragment_count: int
    bounded_lowering_allowed: bool
    bounded_lowering_refusal_reasons: tuple[str, ...]


@dataclass(frozen=True)
class MooncakeStaticPlan:
    tensor_id: str
    source_placement: WeightPlacementManifest
    source_bindings: tuple[WeightRuntimeBindingManifest, ...]
    target_placement: WeightPlacementManifest
    target_bindings: tuple[WeightRuntimeBindingManifest, ...]
    plan: TransferPlan
    summary: MooncakePlanSummary

    @property
    def fake_address_ranges(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (fragment.address, fragment.address + fragment.nbytes)
            for binding in (*self.source_bindings, *self.target_bindings)
            for fragment in binding.fragments
        )


@dataclass(frozen=True)
class _FakeRuntimeBuffer:
    pointer: int
    size: int
    device: int = 0


def _case_geometry(
    case: BenchmarkCase,
) -> tuple[MeshSpec, MeshSpec, tuple[int, ...], int]:
    if case.source is None or case.target is None or case.global_shape is None:
        raise MooncakePlanningError(
            f"{case.id}: Mooncake planning requires source, target, and "
            "global_shape geometry"
        )
    logical_bytes = case.logical_bytes
    elements = prod(case.global_shape)
    if logical_bytes is None or logical_bytes % elements:
        raise MooncakePlanningError(
            f"{case.id}: logical byte size is inconsistent with global_shape"
        )
    return case.source, case.target, case.global_shape, logical_bytes // elements


def _fake_buffers(
    mesh: MeshSpec,
    shape: tuple[int, ...],
    itemsize: int,
    next_address: int,
) -> tuple[tuple[_FakeRuntimeBuffer, ...], int]:
    shard_extent = shape[mesh.shard_dim] // mesh.shards
    local_shape = list(shape)
    local_shape[mesh.shard_dim] = shard_extent
    nbytes = prod(local_shape) * itemsize
    buffers = []
    for rank in range(mesh.total_ranks):
        buffers.append(_FakeRuntimeBuffer(next_address, nbytes, rank))
        next_address += nbytes + _FAKE_ADDRESS_GAP
    return tuple(buffers), next_address


def _target_batch_counts(
    plan: TransferPlan, max_batch_operations: int
) -> tuple[int, ...]:
    counts_by_worker = {executor.worker_id: 0 for executor in plan.target_executors}
    for source_executor in plan.source_executors:
        segments_by_target: dict[tuple[str, str], int] = {}
        for index in source_executor.operation_indices:
            region = plan.operations[index]
            key = (region.target.worker_id, region.target.endpoint)
            segments_by_target[key] = segments_by_target.get(key, 0) + region.repeat
        for (worker_id, _), segment_count in segments_by_target.items():
            counts_by_worker[worker_id] += (
                segment_count + max_batch_operations - 1
            ) // max_batch_operations
    return tuple(
        counts_by_worker[executor.worker_id] for executor in plan.target_executors
    )


def plan_mooncake_case(
    case: BenchmarkCase,
    *,
    max_batch_operations: int = _DEFAULT_MAX_BATCH_OPERATIONS,
    max_region_segments: int = _DEFAULT_MAX_REGION_SEGMENTS,
) -> MooncakeStaticPlan:
    """构造 placement/binding，并仅用 N-D region 静态算术汇总计划。"""

    for name, value in (
        ("max_batch_operations", max_batch_operations),
        ("max_region_segments", max_region_segments),
    ):
        if type(value) is not int or value <= 0:
            raise MooncakePlanningError(f"{name} must be a positive integer")

    source, target, shape, itemsize = _case_geometry(case)
    source_buffers, next_address = _fake_buffers(
        source, shape, itemsize, _FAKE_ADDRESS_BASE
    )
    target_buffers, _ = _fake_buffers(target, shape, itemsize, next_address)
    source_topology = build_runtime_topology(
        case,
        side="source",
        buffers=source_buffers,
        endpoint="dry-run://source",
        revision=_REVISION,
    )
    target_topology = build_runtime_topology(
        case,
        side="target",
        buffers=target_buffers,
        endpoint="dry-run://target",
        revision=_REVISION,
    )
    logical_plan = plan_placement_transfer(
        source_topology.placement,
        target_topology.placement,
    )
    plan = bind_logical_transfer_plan(
        logical_plan,
        target_topology.bindings,
        source_bindings=source_topology.bindings,
    )
    if not plan.operations or not all(
        isinstance(operation, TransferRegion) for operation in plan.operations
    ):
        raise MooncakePlanningError("Mooncake planner did not return N-D regions")

    regions = tuple(plan.operations)
    inner_bytes = {region.inner_bytes for region in regions}
    if len(inner_bytes) != 1:
        raise MooncakePlanningError("Mooncake plan has non-uniform inner bytes")
    oversized = sorted(
        {
            region.segment_count
            for region in regions
            if region.segment_count > max_region_segments
        }
    )
    refusal_reasons = tuple(
        f"region requires {segment_count} segments, exceeding "
        f"max_region_segments {max_region_segments}"
        for segment_count in oversized
    )
    selected_source_fragments = {
        region.source.placement_fragment_id for region in logical_plan.operations
    }
    selected_source_replicas = tuple(
        sorted({region.source.rank.dp for region in logical_plan.operations})
    )
    source_replica_count = len(
        {fragment.rank.dp for fragment in source_topology.placement.fragments}
    )
    summary = MooncakePlanSummary(
        placement_count=2,
        runtime_binding_count=(
            len(source_topology.bindings) + len(target_topology.bindings)
        ),
        region_count=len(regions),
        total_segment_count=sum(region.segment_count for region in regions),
        max_segments_per_region=max(region.segment_count for region in regions),
        inner_bytes=next(iter(inner_bytes)),
        target_batch_counts=_target_batch_counts(plan, max_batch_operations),
        plan_total_bytes=plan.total_bytes,
        source_replica_count=source_replica_count,
        selected_source_replicas=selected_source_replicas,
        selected_source_fragment_count=len(selected_source_fragments),
        deduplicated_source_fragment_count=(
            len(source_topology.placement.fragments) - len(selected_source_fragments)
        ),
        bounded_lowering_allowed=not refusal_reasons,
        bounded_lowering_refusal_reasons=refusal_reasons,
    )
    return MooncakeStaticPlan(
        tensor_id=NEUTRAL_TENSOR_ID,
        source_placement=source_topology.placement,
        source_bindings=source_topology.bindings,
        target_placement=target_topology.placement,
        target_bindings=target_topology.bindings,
        plan=plan,
        summary=summary,
    )
