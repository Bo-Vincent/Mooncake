from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union

from ..._typing import TypeAlias

from ..._compat import _strict_zip
from ...contracts import ParticipantId, PlacementFragmentId, TensorId
from ...geometry import regions_exactly_cover
from ..manifest import (
    PlacementFragment,
    TensorDescriptor,
)
from ..storage_manifest import StoredFragment
from .contracts import (
    BoundWeightFragment,
    ExecutableTransferOperation,
    LogicalTransferOperation,
    LogicalTransferPlan,
    TransferPlan,
)
from .attestation import RuntimeBindingAttestation


IdentityKey: TypeAlias = tuple[object, ...]
TargetAddressSpace: TypeAlias = tuple[str, str, str]


def _validate_logical_target_coverage(logical_plan: LogicalTransferPlan) -> None:
    """Reject plans that would leave a selected target fragment unwritten."""

    target_parts = {
        part.participant_id: part for part in logical_plan.target_placement.parts
    }
    participant_by_fragment = {
        fragment.placement_fragment_id: part.participant_id
        for part in logical_plan.target_placement.parts
        for fragment in part.fragments
    }
    operation_participants: set[ParticipantId] = set()
    operations_by_fragment: dict[
        PlacementFragmentId, list[LogicalTransferOperation]
    ] = {}

    for operation in logical_plan.operations:
        target = operation.target
        if not isinstance(target, PlacementFragment):
            raise ValueError("logical transfer operation target must be a placement")
        target_participant_id = participant_by_fragment.get(
            target.placement_fragment_id
        )
        if target_participant_id is None or not any(
            fragment == target
            for fragment in target_parts[target_participant_id].fragments
        ):
            raise ValueError(
                "logical plan and target placement fragment snapshots differ"
            )
        operations_by_fragment.setdefault(target.placement_fragment_id, []).append(
            operation
        )
        operation_participants.add(target_participant_id)

    if not logical_plan.target_executors:
        raise ValueError("logical plan has no target executor metadata")

    seen_participants: set[ParticipantId] = set()
    selected_fragments: dict[PlacementFragmentId, PlacementFragment] = {}
    for executor in logical_plan.target_executors:
        if executor.placement_id != logical_plan.target_placement.placement_id:
            raise ValueError("logical target executor placement differs from target")
        if executor.participant_id in seen_participants:
            raise ValueError("logical plan has duplicate target executor participant")
        seen_participants.add(executor.participant_id)
        part = target_parts.get(executor.participant_id)
        if part is None or executor.rank != part.rank:
            raise ValueError("logical target executor differs from target placement")
        expected_fragment_ids = tuple(
            sorted(fragment.placement_fragment_id for fragment in part.fragments)
        )
        if executor.placement_fragment_ids != expected_fragment_ids:
            raise ValueError(
                "logical target executor fragments differ from target placement"
            )
        if not logical_plan.operation_indices_for_executor(executor, "target"):
            raise ValueError(
                "logical target executor operations differ from target placement"
            )
        selected_fragments.update(
            {fragment.placement_fragment_id: fragment for fragment in part.fragments}
        )

    if seen_participants != operation_participants:
        raise ValueError("logical target executors differ from planned operations")

    for fragment_id, target in selected_fragments.items():
        operations = operations_by_fragment.get(fragment_id, ())
        if not regions_exactly_cover(target, operations):
            raise ValueError(
                f"logical target fragment is not fully covered: {fragment_id}"
            )


def _validate_bound_target_coverage(plan: TransferPlan) -> None:
    """Require every selected target participant to be fully covered again."""

    target_placement = plan.target_placement
    target_parts = {part.participant_id: part for part in target_placement.parts}
    participant_by_fragment = {
        fragment.placement_fragment_id: part.participant_id
        for part in target_placement.parts
        for fragment in part.fragments
    }
    operations_by_fragment: dict[
        PlacementFragmentId, list[ExecutableTransferOperation]
    ] = {}
    operation_participants: set[ParticipantId] = set()
    for operation in plan.operations:
        target = operation.target
        attestation = target.attestation
        if (
            not isinstance(attestation, RuntimeBindingAttestation)
            or attestation.placement != target_placement
            or not attestation.validates(target.placement, target.binding)
        ):
            raise ValueError("bound target fragment differs from target placement")
        participant_id = participant_by_fragment.get(target.placement_fragment_id)
        if participant_id is None or not any(
            fragment == target.placement
            for fragment in target_parts[participant_id].fragments
        ):
            raise ValueError("bound target fragment differs from target placement")
        operations_by_fragment.setdefault(target.placement_fragment_id, []).append(
            operation
        )
        operation_participants.add(participant_id)

    if not plan.target_executors:
        raise ValueError("bound plan has no target executor metadata")

    seen_executor_keys: set[tuple[ParticipantId, str]] = set()
    selected_participants: set[ParticipantId] = set()
    selected_fragments: dict[PlacementFragmentId, PlacementFragment] = {}
    for executor in plan.target_executors:
        executor_key = (executor.participant_id, executor.worker_id)
        if executor_key in seen_executor_keys:
            raise ValueError("bound plan has duplicate target executor")
        seen_executor_keys.add(executor_key)
        part = target_parts.get(executor.participant_id)
        attestation = executor.attestation
        if (
            part is None
            or executor.rank != part.rank
            or executor.placement_id != target_placement.placement_id
            or executor.placement_digest != target_placement.digest
            or not isinstance(attestation, RuntimeBindingAttestation)
            or attestation.placement != target_placement
            or attestation.binding.participant_id != executor.participant_id
            or attestation.binding.instance_id != executor.instance_id
            or attestation.binding.lease_id != executor.runtime_lease_id
        ):
            raise ValueError("bound target executor differs from target placement")
        expected_fragment_ids = tuple(
            sorted(
                runtime.fragment_id
                for _, runtime in attestation.worker_fragment_pairs(executor.worker_id)
            )
        )
        if executor.fragment_ids != expected_fragment_ids:
            raise ValueError(
                "bound target executor fragments differ from target placement"
            )
        if not plan.operation_indices_for_executor(executor, "target"):
            raise ValueError(
                "bound target executor operations differ from target placement"
            )
        selected_participants.add(executor.participant_id)
        selected_fragments.update(
            {fragment.placement_fragment_id: fragment for fragment in part.fragments}
        )

    if selected_participants != operation_participants:
        raise ValueError("bound target executors differ from planned operations")

    for fragment_id, target in selected_fragments.items():
        operations = operations_by_fragment.get(fragment_id, ())
        if not regions_exactly_cover(target, operations):
            raise ValueError(
                f"bound target fragment is not fully covered: {fragment_id}"
            )


def _validate_tensor_compatibility(
    source: TensorDescriptor, target: TensorDescriptor
) -> None:
    if source.layout_fingerprint != target.layout_fingerprint:
        raise ValueError(f"layout mismatch for tensor {source.tensor_id}")
    if (
        source.global_shape != target.global_shape
        or source.dtype != target.dtype
        or source.itemsize != target.itemsize
        or source.layer_id != target.layer_id
        or source.expert_id != target.expert_id
    ):
        raise ValueError(f"tensor descriptor mismatch: {source.tensor_id}")


def _validate_tensor_sets(
    source_tensors: dict[TensorId, TensorDescriptor],
    target_tensors: dict[TensorId, TensorDescriptor],
) -> None:
    source_ids = set(source_tensors)
    target_ids = set(target_tensors)
    missing = sorted(source_ids - target_ids)
    if missing:
        raise ValueError(f"target manifests are missing tensors: {', '.join(missing)}")
    unexpected = sorted(target_ids - source_ids)
    if unexpected:
        raise ValueError(
            f"target manifests contain unknown tensors: {', '.join(unexpected)}"
        )


def _validate_tensor_subset(
    source_tensors: dict[TensorId, TensorDescriptor],
    target_tensors: dict[TensorId, TensorDescriptor],
) -> None:
    unexpected = sorted(set(target_tensors) - set(source_tensors))
    if unexpected:
        raise ValueError(
            f"target manifests contain unknown tensors: {', '.join(unexpected)}"
        )


def _is_complete_alias_group(
    fragment: Union[BoundWeightFragment, StoredFragment],
) -> bool:
    return len(fragment.aliases) > 1 and fragment.tensor_id in fragment.aliases


def _is_safe_declared_alias_overlap(
    left: ExecutableTransferOperation,
    right: ExecutableTransferOperation,
) -> bool:
    """Allow one exact write only for one complete, attested alias lifetime."""

    left_target = left.target
    right_target = right.target
    left_attestation = left_target.attestation
    right_attestation = right_target.attestation
    if (
        left.tensor_id == right.tensor_id
        or not _is_complete_alias_group(left.source)
        or not _is_complete_alias_group(right.source)
        or not _is_complete_alias_group(left_target)
        or not _is_complete_alias_group(right_target)
        or left.source.aliases != right.source.aliases
        or left_target.aliases != right_target.aliases
        or left.source.aliases != left_target.aliases
        or left.source.global_offset != right.source.global_offset
        or left.source.local_shape != right.source.local_shape
        or left.source_offset != right.source_offset
        or left.target_offset != right.target_offset
        or left.nbytes != right.nbytes
        or left.outer_loop_counts != right.outer_loop_counts
        or left.source_strides != right.source_strides
        or left.target_strides != right.target_strides
        or left_target.global_offset != right_target.global_offset
        or left_target.local_shape != right_target.local_shape
        or left_target.address != right_target.address
        or left_target.nbytes != right_target.nbytes
        or left_target.owner != right_target.owner
        or left_target.instance_id != right_target.instance_id
        or left_target.runtime_lease_id != right_target.runtime_lease_id
        or left_target.lease_generation != right_target.lease_generation
        or not isinstance(left_attestation, RuntimeBindingAttestation)
        or not isinstance(right_attestation, RuntimeBindingAttestation)
        or left_attestation.placement != right_attestation.placement
        or left_attestation.binding != right_attestation.binding
        or left.overlap_offset != right.overlap_offset
        or left.overlap_shape != right.overlap_shape
    ):
        return False
    return True


def _target_physical_bounds(operation: ExecutableTransferOperation) -> tuple[int, int]:
    begin = operation.target.address + operation.target_offset
    end = (
        begin
        + sum(
            (count - 1) * stride
            for count, stride in _strict_zip(
                operation.outer_loop_counts,
                operation.target_strides,
            )
        )
        + operation.nbytes
    )
    return begin, end


@dataclass
class _SegmentScanBudget:
    limit: int
    checked: int = 0

    def consume(self) -> None:
        self.checked += 1
        if self.checked > self.limit:
            raise ValueError("target physical segment scan budget exceeded")


def _absolute_target_segments(
    operation: ExecutableTransferOperation,
    *,
    max_segments: int,
) -> Iterable[tuple[int, int]]:
    for _, target_offset, nbytes in operation.iter_segments(max_segments=max_segments):
        begin = operation.target.address + target_offset
        yield begin, begin + nbytes


def _budgeted_target_segments(
    operation_index: int,
    operation: ExecutableTransferOperation,
    budget: _SegmentScanBudget,
) -> Iterable[tuple[int, int, int]]:
    for begin, end in _absolute_target_segments(
        operation,
        # The plan constructor already enforces its per-region bound. This
        # scanner owns a separate aggregate budget and must account segments
        # incrementally so its public error remains deterministic.
        max_segments=operation.segment_count,
    ):
        budget.consume()
        yield begin, end, operation_index


def _target_fragment_scan_key(
    operation: ExecutableTransferOperation,
) -> IdentityKey:
    target = operation.target
    return (
        target.fragment_id,
        target.tensor_id,
        target.instance_id,
        target.worker_id,
        target.device,
        target.lease_generation,
        target.address,
        target.nbytes,
        target.global_offset,
        target.local_shape,
    )


def _complete_target_fragment_segment(
    indexed_operations: Sequence[tuple[int, ExecutableTransferOperation]],
) -> Optional[tuple[int, int, int]]:
    if not indexed_operations:
        return None
    target = indexed_operations[0][1].target
    if not regions_exactly_cover(
        target,
        tuple(operation for _, operation in indexed_operations),
    ):
        return None
    return target.address, target.address + target.nbytes, indexed_operations[0][0]


def _validate_target_physical_ranges(
    operations: Sequence[ExecutableTransferOperation],
    *,
    max_segment_checks: int = 1_000_000,
) -> None:
    if type(max_segment_checks) is not int or max_segment_checks <= 0:
        raise ValueError("max_segment_checks must be a positive integer")
    by_address_space: dict[TargetAddressSpace, list[ExecutableTransferOperation]] = {}
    for operation in operations:
        by_address_space.setdefault(
            (
                operation.target.instance_id,
                operation.target.worker_id,
                operation.target.device,
            ),
            [],
        ).append(operation)

    for scoped_operations in by_address_space.values():
        by_fragment: dict[
            IdentityKey, list[tuple[int, ExecutableTransferOperation]]
        ] = {}
        for index, operation in enumerate(scoped_operations):
            by_fragment.setdefault(_target_fragment_scan_key(operation), []).append(
                (index, operation)
            )

        complete_segments: list[tuple[int, int, int]] = []
        incomplete_operation_indices: set[int] = set()
        for indexed_operations in by_fragment.values():
            complete_segment = _complete_target_fragment_segment(indexed_operations)
            if complete_segment is None:
                incomplete_operation_indices.update(
                    index for index, _ in indexed_operations
                )
            else:
                complete_segments.append(complete_segment)
        complete_segments.sort()

        budget = _SegmentScanBudget(max_segment_checks)
        segment_streams: list[Iterable[tuple[int, int, int]]] = []
        if complete_segments:
            segment_streams.append(iter(complete_segments))
        segment_streams.extend(
            (
                _budgeted_target_segments(index, operation, budget)
                for index, operation in enumerate(scoped_operations)
                if index in incomplete_operation_indices
            )
        )
        ordered_segments = heapq.merge(*segment_streams)
        previous_begin = -1
        previous_end = -1
        previous_operation_index: Optional[int] = None
        for begin, end, operation_index in ordered_segments:
            if begin < previous_end:
                if previous_operation_index is None:
                    raise AssertionError(
                        "target physical range scan lost its predecessor"
                    )
                previous_operation = scoped_operations[previous_operation_index]
                current_operation = scoped_operations[operation_index]
                if (
                    begin == previous_begin
                    and end == previous_end
                    and _is_safe_declared_alias_overlap(
                        previous_operation,
                        current_operation,
                    )
                ):
                    continue
                raise ValueError(
                    "conflicting target physical range: "
                    f"{scoped_operations[operation_index].target.fragment_id}"
                )
            previous_begin = begin
            previous_end = max(previous_end, end)
            previous_operation_index = operation_index
