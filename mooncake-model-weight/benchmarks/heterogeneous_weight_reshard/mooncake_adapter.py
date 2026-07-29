"""Mooncake N-D planner 的纯控制面 benchmark adapter。"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from mooncake.model_weight import (
    ParallelRank,
    RuntimeFragment,
    RuntimeManifest,
    TensorDescriptor,
    TransferPlan,
    TransferRegion,
    plan_runtime_transfer,
)

from .case_spec import BenchmarkCase, MeshSpec


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
    manifest_count: int
    region_count: int
    total_segment_count: int
    max_segments_per_region: int
    inner_bytes: int
    target_batch_counts: tuple[int, ...]
    plan_total_bytes: int
    source_replica_count: int
    selected_source_replicas: tuple[int, ...]
    selected_source_manifest_count: int
    deduplicated_source_manifest_count: int
    bounded_lowering_allowed: bool
    bounded_lowering_refusal_reasons: tuple[str, ...]


@dataclass(frozen=True)
class MooncakeStaticPlan:
    tensor_id: str
    source_manifests: tuple[RuntimeManifest, ...]
    target_manifests: tuple[RuntimeManifest, ...]
    plan: TransferPlan
    summary: MooncakePlanSummary

    @property
    def fake_address_ranges(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (fragment.address, fragment.address + fragment.nbytes)
            for manifest in (*self.source_manifests, *self.target_manifests)
            for fragment in manifest.fragments
        )


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


def _descriptor(
    shape: tuple[int, ...], itemsize: int, shard_dim: int
) -> TensorDescriptor:
    dtype = "uint8" if itemsize == 1 else f"opaque{itemsize * 8}"
    return TensorDescriptor(
        tensor_id=NEUTRAL_TENSOR_ID,
        global_shape=shape,
        dtype=dtype,
        itemsize=itemsize,
        partition_dim=None,
        layout_fingerprint="benchmark:logical-contiguous:v1",
        shard_dims=(shard_dim,),
    )


def _build_manifests(
    side: str,
    mesh: MeshSpec,
    shape: tuple[int, ...],
    itemsize: int,
    next_address: int,
) -> tuple[tuple[RuntimeManifest, ...], int]:
    descriptor = _descriptor(shape, itemsize, mesh.shard_dim)
    shard_extent = shape[mesh.shard_dim] // mesh.shards
    manifests = []
    for replica in range(mesh.replicas):
        for shard in range(mesh.shards):
            local_shape = list(shape)
            local_shape[mesh.shard_dim] = shard_extent
            global_offset = [0] * len(shape)
            global_offset[mesh.shard_dim] = shard * shard_extent
            local_shape_tuple = tuple(local_shape)
            nbytes = prod(local_shape_tuple) * itemsize
            worker_id = f"{side}-d{replica}-t{shard}"
            fragment = RuntimeFragment(
                fragment_id=f"{worker_id}-fragment",
                tensor_id=NEUTRAL_TENSOR_ID,
                global_offset=tuple(global_offset),
                local_shape=local_shape_tuple,
                address=next_address,
                nbytes=nbytes,
                worker_id=worker_id,
                endpoint=f"dry-run://{worker_id}",
                device="cuda:0",
                rank=ParallelRank(dp=replica, tp=shard),
                lease_generation=1,
            )
            manifests.append(
                RuntimeManifest(
                    model_id=_MODEL_ID,
                    revision=_REVISION,
                    instance_id=worker_id,
                    tensors=(descriptor,),
                    fragments=(fragment,),
                    lease_id=f"{worker_id}-lease",
                )
            )
            next_address += nbytes + _FAKE_ADDRESS_GAP
    return tuple(manifests), next_address


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
    """构造 runtime manifests，并仅用 N-D region 静态算术汇总计划。"""

    for name, value in (
        ("max_batch_operations", max_batch_operations),
        ("max_region_segments", max_region_segments),
    ):
        if type(value) is not int or value <= 0:
            raise MooncakePlanningError(f"{name} must be a positive integer")

    source, target, shape, itemsize = _case_geometry(case)
    source_manifests, next_address = _build_manifests(
        "source", source, shape, itemsize, _FAKE_ADDRESS_BASE
    )
    target_manifests, _ = _build_manifests(
        "target", target, shape, itemsize, next_address
    )
    plan = plan_runtime_transfer(source_manifests, target_manifests)
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
    selected_source_fragments = {region.source.fragment_id for region in regions}
    selected_source_replicas = tuple(
        sorted({region.source.rank.dp for region in regions})
    )
    source_replica_count = len(
        {
            fragment.rank.dp
            for manifest in source_manifests
            for fragment in manifest.fragments
        }
    )
    summary = MooncakePlanSummary(
        manifest_count=len(source_manifests) + len(target_manifests),
        region_count=len(regions),
        total_segment_count=sum(region.segment_count for region in regions),
        max_segments_per_region=max(region.segment_count for region in regions),
        inner_bytes=next(iter(inner_bytes)),
        target_batch_counts=_target_batch_counts(plan, max_batch_operations),
        plan_total_bytes=plan.total_bytes,
        source_replica_count=source_replica_count,
        selected_source_replicas=selected_source_replicas,
        selected_source_manifest_count=len(selected_source_fragments),
        deduplicated_source_manifest_count=(
            len(source_manifests) - len(selected_source_fragments)
        ),
        bounded_lowering_allowed=not refusal_reasons,
        bounded_lowering_refusal_reasons=refusal_reasons,
    )
    return MooncakeStaticPlan(
        tensor_id=NEUTRAL_TENSOR_ID,
        source_manifests=source_manifests,
        target_manifests=target_manifests,
        plan=plan,
        summary=summary,
    )
