from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..._compat import _strict_zip
from ..manifest import (
    PlacementFragment,
    TensorDescriptor,
)
from ..storage_manifest import StoredFragment
from .contracts import (
    BoundWeightFragment,
    CopyRange,
    TransferOperation,
    TransferRegion,
)
from .ownership import _boxes_exactly_cover


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
    source_tensors: dict[str, TensorDescriptor],
    target_tensors: dict[str, TensorDescriptor],
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
    source_tensors: dict[str, TensorDescriptor],
    target_tensors: dict[str, TensorDescriptor],
) -> None:
    unexpected = sorted(set(target_tensors) - set(source_tensors))
    if unexpected:
        raise ValueError(
            f"target manifests contain unknown tensors: {', '.join(unexpected)}"
        )


def _operation_loop_geometry(
    operation: TransferOperation, side: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if isinstance(operation, TransferRegion):
        strides = (
            operation.source_strides if side == "source" else operation.target_strides
        )
        return operation.outer_loop_counts, strides
    stride = operation.source_stride if side == "source" else operation.target_stride
    return (operation.repeat,), (stride,)


def _operation_sort_key(operation: TransferOperation) -> tuple:
    source_location = (
        (
            "runtime",
            operation.source.instance_id,
            operation.source.worker_id,
            operation.source.device,
            operation.source.lease_generation,
            operation.source.address + operation.source_offset,
        )
        if isinstance(operation.source, BoundWeightFragment)
        else (
            "stored",
            operation.source.object_key,
            operation.source.object_offset + operation.source_offset,
        )
    )
    return (
        operation.tensor_id,
        operation.target.instance_id,
        operation.target.worker_id,
        operation.target.device,
        operation.target.address + operation.target_offset,
        source_location,
    )


def _source_copy_identity(operation: TransferOperation) -> tuple:
    counts, strides = _operation_loop_geometry(operation, "source")
    if isinstance(operation.source, BoundWeightFragment):
        return (
            "runtime",
            operation.source.instance_id,
            operation.source.worker_id,
            operation.source.device,
            operation.source.lease_generation,
            operation.source.address + operation.source_offset,
            operation.nbytes,
            counts,
            strides,
        )
    return (
        "stored",
        operation.source.object_key,
        operation.source.object_offset + operation.source_offset,
        operation.nbytes,
        counts,
        strides,
    )


def _target_copy_identity(operation: TransferOperation) -> tuple:
    counts, strides = _operation_loop_geometry(operation, "target")
    return (
        operation.target.instance_id,
        operation.target.worker_id,
        operation.target.device,
        operation.target.lease_generation,
        operation.target.address + operation.target_offset,
        operation.nbytes,
        counts,
        strides,
    )


def _descriptor_alias_key(descriptor: TensorDescriptor) -> tuple:
    return (
        descriptor.global_shape,
        descriptor.dtype,
        descriptor.itemsize,
        descriptor.layer_id,
        descriptor.expert_id,
        descriptor.layout_fingerprint,
    )


def _is_declared_target_alias(
    left: TransferOperation,
    right: TransferOperation,
    source_tensors: dict[str, TensorDescriptor],
    target_tensors: dict[str, TensorDescriptor],
) -> bool:
    left_target = left.target
    right_target = right.target
    if (
        isinstance(left.source, StoredFragment)
        or isinstance(right.source, StoredFragment)
        or len(left_target.aliases) < 2
        or left_target.aliases != right_target.aliases
        or (
            isinstance(
                left.source,
                (BoundWeightFragment, PlacementFragment),
            )
            and (
                len(left.source.aliases) < 2
                or left.source.aliases != right.source.aliases
            )
        )
        or left_target.instance_id != right_target.instance_id
        or left_target.worker_id != right_target.worker_id
        or left_target.device != right_target.device
        or left_target.lease_generation != right_target.lease_generation
        or left_target.address != right_target.address
        or left_target.nbytes != right_target.nbytes
        or left_target.global_offset != right_target.global_offset
        or left_target.local_shape != right_target.local_shape
        or left.source.global_offset != right.source.global_offset
        or left.source.local_shape != right.source.local_shape
        or left.source_offset != right.source_offset
        or left.target_offset != right.target_offset
        or left.nbytes != right.nbytes
        or type(left) is not type(right)
        or _operation_loop_geometry(left, "source")
        != _operation_loop_geometry(right, "source")
        or _operation_loop_geometry(left, "target")
        != _operation_loop_geometry(right, "target")
        or (
            isinstance(left, TransferRegion)
            and isinstance(right, TransferRegion)
            and (
                left.overlap_offset != right.overlap_offset
                or left.overlap_shape != right.overlap_shape
            )
        )
    ):
        return False
    return (
        _descriptor_alias_key(source_tensors[left.tensor_id])
        == _descriptor_alias_key(source_tensors[right.tensor_id])
        == _descriptor_alias_key(target_tensors[left.tensor_id])
        == _descriptor_alias_key(target_tensors[right.tensor_id])
    )


def _deduplicate_target_copies(
    operations: Sequence[TransferOperation],
    source_tensors: dict[str, TensorDescriptor],
    target_tensors: dict[str, TensorDescriptor],
) -> tuple[TransferOperation, ...]:
    result = []
    seen = set()
    for operation in sorted(operations, key=_operation_sort_key):
        identity = (
            operation.tensor_id,
            _source_copy_identity(operation),
            _target_copy_identity(operation),
        )
        if identity in seen:
            continue
        seen.add(identity)
        result.append(operation)

    by_target: dict[tuple, TransferOperation] = {}
    deduplicated = []
    for operation in result:
        identity = _target_copy_identity(operation)
        previous = by_target.get(identity)
        if previous is None:
            by_target[identity] = operation
            deduplicated.append(operation)
            continue
        if _is_declared_target_alias(
            previous,
            operation,
            source_tensors,
            target_tensors,
        ):
            continue
        deduplicated.append(operation)

    result = sorted(
        deduplicated,
        key=lambda item: (
            item.target.fragment_id,
            item.target_offset,
            item.source.fragment_id,
            item.source_offset,
        ),
    )
    _validate_target_physical_ranges(result)
    return tuple(result)


def _target_physical_bounds(operation: TransferOperation) -> tuple[int, int]:
    begin = operation.target.address + operation.target_offset
    counts, strides = _operation_loop_geometry(operation, "target")
    end = (
        begin
        + sum((count - 1) * stride for count, stride in _strict_zip(counts, strides))
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
    operation: TransferOperation,
) -> Iterable[tuple[int, int]]:
    for _, target_offset, nbytes in operation.iter_segments():
        begin = operation.target.address + target_offset
        yield begin, begin + nbytes


def _budgeted_target_segments(
    operation_index: int,
    operation: TransferOperation,
    budget: _SegmentScanBudget,
) -> Iterable[tuple[int, int, int]]:
    for begin, end in _absolute_target_segments(operation):
        budget.consume()
        yield begin, end, operation_index


def _target_fragment_scan_key(operation: TransferOperation) -> tuple:
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
    indexed_operations: Sequence[tuple[int, TransferOperation]],
) -> tuple[int, int, int] | None:
    if not indexed_operations or not all(
        isinstance(operation, TransferRegion) for _, operation in indexed_operations
    ):
        return None
    target = indexed_operations[0][1].target
    boxes = tuple(
        (operation.overlap_offset, operation.overlap_shape)
        for _, operation in indexed_operations
        if isinstance(operation, TransferRegion)
    )
    if not _boxes_exactly_cover(target.global_offset, target.local_shape, boxes):
        return None
    return target.address, target.address + target.nbytes, indexed_operations[0][0]


def _validate_target_physical_ranges(
    operations: Sequence[TransferOperation],
    *,
    max_segment_checks: int = 1_000_000,
) -> None:
    if type(max_segment_checks) is not int or max_segment_checks <= 0:
        raise ValueError("max_segment_checks must be a positive integer")
    by_address_space: dict[tuple[str, str, str], list[TransferOperation]] = {}
    for operation in operations:
        if (
            isinstance(operation, CopyRange)
            and operation.repeat > 1
            and operation.target_stride < operation.nbytes
        ):
            raise ValueError(
                f"conflicting target physical range: {operation.target.fragment_id}"
            )
        by_address_space.setdefault(
            (
                operation.target.instance_id,
                operation.target.worker_id,
                operation.target.device,
            ),
            [],
        ).append(operation)

    for scoped_operations in by_address_space.values():
        by_fragment: dict[tuple, list[tuple[int, TransferOperation]]] = {}
        for index, operation in enumerate(scoped_operations):
            by_fragment.setdefault(_target_fragment_scan_key(operation), []).append(
                (index, operation)
            )

        complete_segments = []
        incomplete_operation_indices = set()
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
        previous_end = -1
        for begin, end, operation_index in ordered_segments:
            if begin < previous_end:
                raise ValueError(
                    "conflicting target physical range: "
                    f"{scoped_operations[operation_index].target.fragment_id}"
                )
            previous_end = max(previous_end, end)
