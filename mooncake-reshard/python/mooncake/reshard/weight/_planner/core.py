from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..manifest import (
    ParallelRank,
    PlacementFragment,
    TensorDescriptor,
    WeightPlacementManifest,
    WeightRuntimeBindingManifest,
    validate_runtime_binding,
)
from . import geometry
from .contracts import (
    BoundWeightFragment,
    ExecutorTransferPlan,
    LogicalTransferPlan,
    PipelineRouteGroup,
    PlacementExecutorPlan,
    RuntimeLeaseSnapshot,
    SourceFragment,
    TargetFragment,
    TransferOperation,
    TransferPlan,
    TransferRegion,
)
from .ownership import (
    _boxes_exactly_cover,
    _validate_local_target_inventory,
    _validate_target_coverage,
    complete_parallel_source_replicas,
    parallel_tensor_owner,
)
from .validation import (
    _validate_tensor_compatibility,
    _validate_tensor_sets,
    _validate_tensor_subset,
)


@dataclass(frozen=True)
class _PlannedTransfer:
    resource_id: str
    revision: str
    operations: tuple[TransferOperation, ...]


def _collect_placements(
    manifests: Sequence[WeightPlacementManifest],
    label: str,
) -> tuple[dict[str, TensorDescriptor], list[PlacementFragment]]:
    if not manifests:
        raise ValueError(f"{label} placement manifests must not be empty")
    if any(not isinstance(manifest, WeightPlacementManifest) for manifest in manifests):
        raise ValueError(f"{label} placement manifests must be WeightPlacementManifest")
    resource_id = manifests[0].resource_id
    revision = manifests[0].revision
    placement_ids: set[str] = set()
    fragment_ids: set[str] = set()
    tensors: dict[str, TensorDescriptor] = {}
    fragments: list[PlacementFragment] = []
    for manifest in manifests:
        if manifest.resource_id != resource_id or manifest.revision != revision:
            raise ValueError(
                f"{label} placement manifests describe different revisions"
            )
        if manifest.placement_id in placement_ids:
            raise ValueError(f"duplicate {label} placement_id: {manifest.placement_id}")
        placement_ids.add(manifest.placement_id)
        for tensor in manifest.tensors:
            previous = tensors.setdefault(tensor.tensor_id, tensor)
            if previous != tensor:
                raise ValueError(
                    f"{label} placement tensor descriptor mismatch: {tensor.tensor_id}"
                )
        for fragment in manifest.fragments:
            if fragment.placement_fragment_id in fragment_ids:
                raise ValueError(
                    f"duplicate {label} placement fragment: "
                    f"{fragment.placement_fragment_id}"
                )
            fragment_ids.add(fragment.placement_fragment_id)
            fragments.append(fragment)
    return tensors, fragments


def _build_executor_plans(
    placements: Sequence[WeightPlacementManifest],
    bindings: Sequence[WeightRuntimeBindingManifest],
    operations: Sequence[TransferOperation],
    side: str,
) -> tuple[ExecutorTransferPlan, ...]:
    if side not in ("source", "target"):
        raise ValueError(f"invalid executor side: {side}")
    binding_by_placement_id = {binding.placement_id: binding for binding in bindings}
    if len(binding_by_placement_id) != len(bindings):
        raise ValueError(f"duplicate {side} runtime binding")
    if set(binding_by_placement_id) != {
        placement.placement_id for placement in placements
    }:
        raise ValueError(f"{side} placement and runtime binding IDs differ")

    result = []
    ranks: set[ParallelRank] = set()
    operation_indices_by_fragment: dict[str, list[int]] = {}
    for index, operation in enumerate(operations):
        fragment_id = getattr(operation, side).fragment_id
        operation_indices_by_fragment.setdefault(fragment_id, []).append(index)
    for placement in placements:
        if not placement.fragments:
            raise ValueError(f"{side} executor placement has no fragments")
        binding = binding_by_placement_id[placement.placement_id]
        runtime_by_placement_fragment_id = {
            fragment.placement_fragment_id: fragment for fragment in binding.fragments
        }
        fragments_by_rank: dict[ParallelRank, list[BoundWeightFragment]] = {}
        for placement_fragment in placement.fragments:
            runtime_fragment = runtime_by_placement_fragment_id[
                placement_fragment.placement_fragment_id
            ]
            fragment = BoundWeightFragment(
                placement=placement_fragment,
                binding=runtime_fragment,
                instance_id=binding.instance_id,
                runtime_lease_id=binding.lease_id,
                lease_generation=binding.generation,
                owner=runtime_fragment.owner,
            )
            fragments_by_rank.setdefault(fragment.rank, []).append(fragment)
        for rank, fragments in fragments_by_rank.items():
            workers = {fragment.worker_id for fragment in fragments}
            if len(workers) != 1:
                raise ValueError(f"{side} executor rank spans multiple workers: {rank}")
            if rank in ranks:
                raise ValueError(f"duplicate {side} executor rank: {rank}")
            ranks.add(rank)
            ordered_fragments = sorted(
                fragments, key=lambda fragment: fragment.fragment_id
            )
            fragment_ids = tuple(fragment.fragment_id for fragment in ordered_fragments)
            operation_indices = tuple(
                sorted(
                    index
                    for fragment_id in fragment_ids
                    for index in operation_indices_by_fragment.get(fragment_id, ())
                )
            )
            result.append(
                ExecutorTransferPlan(
                    instance_id=binding.instance_id,
                    placement_id=placement.placement_id,
                    placement_digest=placement.digest,
                    runtime_lease_id=binding.lease_id,
                    worker_id=next(iter(workers)),
                    rank=rank,
                    fragment_ids=fragment_ids,
                    fragment_leases=tuple(
                        RuntimeLeaseSnapshot.from_fragment(fragment)
                        for fragment in ordered_fragments
                    ),
                    operation_indices=operation_indices,
                )
            )
    result.sort(
        key=lambda item: (item.rank.dp, item.rank.pp, item.rank.ep, item.rank.tp)
    )
    return tuple(result)


def _build_placement_executor_plans(
    manifests: Sequence[WeightPlacementManifest],
    operations: Sequence[TransferOperation],
    side: str,
) -> tuple[PlacementExecutorPlan, ...]:
    if side not in ("source", "target"):
        raise ValueError(f"invalid placement executor side: {side}")
    operation_indices_by_fragment: dict[str, list[int]] = {}
    for index, operation in enumerate(operations):
        fragment = getattr(operation, side)
        if not isinstance(fragment, PlacementFragment):
            raise ValueError(f"placement executor operation {side} is runtime-bound")
        operation_indices_by_fragment.setdefault(
            fragment.placement_fragment_id, []
        ).append(index)

    result = []
    ranks: set[ParallelRank] = set()
    for manifest in manifests:
        if not manifest.fragments:
            raise ValueError(f"{side} placement executor has no fragments")
        fragments_by_rank: dict[ParallelRank, list[PlacementFragment]] = {}
        for fragment in manifest.fragments:
            fragments_by_rank.setdefault(fragment.rank, []).append(fragment)
        for rank, fragments in fragments_by_rank.items():
            if rank in ranks:
                raise ValueError(f"duplicate {side} placement executor rank: {rank}")
            ranks.add(rank)
            fragment_ids = tuple(
                sorted(fragment.placement_fragment_id for fragment in fragments)
            )
            operation_indices = tuple(
                sorted(
                    index
                    for fragment_id in fragment_ids
                    for index in operation_indices_by_fragment.get(fragment_id, ())
                )
            )
            result.append(
                PlacementExecutorPlan(
                    placement_id=manifest.placement_id,
                    rank=rank,
                    placement_fragment_ids=fragment_ids,
                    operation_indices=operation_indices,
                )
            )
    result.sort(
        key=lambda item: (item.rank.dp, item.rank.pp, item.rank.ep, item.rank.tp)
    )
    return tuple(result)


def resolve_executor_plans(
    plan: TransferPlan,
    placement: WeightPlacementManifest,
    binding: WeightRuntimeBindingManifest,
    side: str,
) -> tuple[ExecutorTransferPlan, ...]:
    if side == "source":
        executors = plan.source_executors
    elif side == "target":
        executors = plan.target_executors
    else:
        raise ValueError(f"invalid executor side: {side}")
    if not executors:
        raise ValueError(f"transfer plan has no {side} executor metadata")
    validate_runtime_binding(placement, binding)
    expected_executors = tuple(
        executor
        for executor in executors
        if executor.instance_id == binding.instance_id
    )
    if not expected_executors:
        raise ValueError(f"{side} executor snapshot mismatch: unknown instance")
    current_executors = _build_executor_plans(
        (placement,),
        (binding,),
        plan.operations,
        side,
    )
    if current_executors != expected_executors:
        raise ValueError(f"{side} executor snapshot mismatch")
    return expected_executors


def resolve_executor_plan(
    plan: TransferPlan,
    placement: WeightPlacementManifest,
    binding: WeightRuntimeBindingManifest,
    side: str,
) -> ExecutorTransferPlan:
    executors = resolve_executor_plans(plan, placement, binding, side)
    if len(executors) != 1:
        raise ValueError(f"{side} executor snapshot contains multiple ranks")
    return executors[0]


def _plan_transfer(
    resource_id: str,
    revision: str,
    source_tensors: dict[str, TensorDescriptor],
    source_fragments: Sequence[SourceFragment],
    target_tensors: dict[str, TensorDescriptor],
    target_fragments: Sequence[TargetFragment],
    *,
    local_target: bool = False,
) -> _PlannedTransfer:
    if local_target:
        _validate_tensor_subset(source_tensors, target_tensors)
        _validate_local_target_inventory(target_tensors, target_fragments)
    else:
        _validate_tensor_sets(source_tensors, target_tensors)
        _validate_target_coverage(target_tensors, target_fragments)
    if not all(
        isinstance(fragment, PlacementFragment) for fragment in source_fragments
    ):
        raise ValueError("source fragments must be logical placements")
    source_replicas = complete_parallel_source_replicas(
        source_tensors, source_fragments
    )
    source_dp_ranks = sorted(source_replicas)
    source_dp_by_target_dp = {
        target_dp: source_dp_ranks[target_dp % len(source_dp_ranks)]
        for target_dp in {fragment.rank.dp for fragment in target_fragments}
    }
    candidates: dict[str, dict[tuple, list[SourceFragment]]] = {}
    for fragment in source_fragments:
        candidates.setdefault(fragment.tensor_id, {}).setdefault(
            geometry._geometry_key(fragment), []
        ).append(fragment)
    for tensor_candidates in candidates.values():
        for group in tensor_candidates.values():
            group.sort(key=geometry._source_sort_key)
    candidate_indexes = {
        tensor_id: geometry._CandidateBoxIndex.build(
            tuple(tuple(group) for group in tensor_candidates.values())
        )
        for tensor_id, tensor_candidates in candidates.items()
    }

    operations: list[TransferRegion] = []
    for target in sorted(target_fragments, key=lambda item: item.fragment_id):
        target_tensor = target_tensors[target.tensor_id]
        source_tensor = source_tensors.get(target.tensor_id)
        if source_tensor is None:
            raise ValueError(f"missing source tensor: {target.tensor_id}")
        _validate_tensor_compatibility(source_tensor, target_tensor)

        overlaps: list[tuple[tuple[int, ...], tuple[int, ...], SourceFragment]] = []
        candidate_index = candidate_indexes.get(target.tensor_id)
        candidate_groups = (
            candidate_index.query(target) if candidate_index is not None else ()
        )
        for group in candidate_groups:
            source_dp = source_dp_by_target_dp[target.rank.dp]
            source_owner = source_replicas[source_dp][target.tensor_id]
            eligible = [
                fragment
                for fragment in group
                if fragment.rank.dp == source_dp
                and parallel_tensor_owner(source_tensor, fragment) == source_owner
            ]
            if not eligible:
                continue
            representative = eligible[0]
            selected = representative
            overlap = geometry._overlap_box(representative, target)
            if overlap is None:
                continue
            overlaps.append((*overlap, selected))
        overlaps.sort(key=lambda item: (item[0], item[1], item[2].fragment_id))

        if not _boxes_exactly_cover(
            target.global_offset,
            target.local_shape,
            tuple((offset, shape) for offset, shape, _ in overlaps),
        ):
            raise ValueError(
                f"target fragment is not fully covered: {target.fragment_id}"
            )
        operations.extend(
            geometry._transfer_region(
                target_tensor,
                source,
                target,
                overlap_offset,
                overlap_shape,
            )
            for overlap_offset, overlap_shape, source in overlaps
        )

    operations.sort(
        key=lambda item: (
            item.target.fragment_id,
            item.target_base_offset,
            item.source.fragment_id,
            item.source_base_offset,
        )
    )
    return _PlannedTransfer(
        resource_id=resource_id,
        revision=revision,
        operations=tuple(operations),
    )


def _build_pipeline_routes(
    operations: Sequence[TransferOperation],
) -> tuple[PipelineRouteGroup, ...]:
    indices_by_route: dict[tuple[int | None, int], list[int]] = {}
    for index, operation in enumerate(operations):
        source_pp = operation.source.rank.pp
        indices_by_route.setdefault((source_pp, operation.target.rank.pp), []).append(
            index
        )
    return tuple(
        PipelineRouteGroup(
            source_pp=source_pp,
            target_pp=target_pp,
            operation_indices=tuple(indices),
        )
        for (source_pp, target_pp), indices in sorted(
            indices_by_route.items(),
            key=lambda item: (
                -1 if item[0][0] is None else item[0][0],
                item[0][1],
            ),
        )
    )


def _logical_transfer_plan(
    transfer: _PlannedTransfer,
    *,
    source_tensors: dict[str, TensorDescriptor],
    target_tensors: dict[str, TensorDescriptor],
    source_placements: Sequence[WeightPlacementManifest],
    target_placements: Sequence[WeightPlacementManifest],
) -> LogicalTransferPlan:
    operations = transfer.operations
    return LogicalTransferPlan(
        resource_id=transfer.resource_id,
        revision=transfer.revision,
        source_placements=tuple(source_placements),
        target_placements=tuple(target_placements),
        source_tensors=tuple(
            sorted(source_tensors.values(), key=lambda item: item.tensor_id)
        ),
        target_tensors=tuple(
            sorted(target_tensors.values(), key=lambda item: item.tensor_id)
        ),
        operations=operations,
        source_executors=(
            _build_placement_executor_plans(source_placements, operations, "source")
            if source_placements
            else ()
        ),
        target_executors=_build_placement_executor_plans(
            target_placements, operations, "target"
        ),
        pipeline_routes=_build_pipeline_routes(operations),
    )


__all__ = [
    "resolve_executor_plan",
    "resolve_executor_plans",
]
