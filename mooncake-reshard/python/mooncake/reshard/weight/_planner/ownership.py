from __future__ import annotations

from math import prod
from typing import Sequence

from ..._compat import _strict_zip
from ..manifest import OwnershipAxis, PlacementFragment, TensorDescriptor
from .contracts import (
    RuntimeTensorOwner,
    SourceFragment,
    TargetFragment,
)
from .geometry import _box_contains


def _boxes_overlap(
    boxes: Sequence[tuple[tuple[int, ...], tuple[int, ...]]],
) -> bool:
    if len(boxes) < 2:
        return False
    ndim = len(boxes[0][0])
    sweep_dim = max(
        range(ndim),
        key=lambda dim: len(
            {(offset[dim], offset[dim] + shape[dim]) for offset, shape in boxes}
        ),
    )
    ordered = sorted(boxes, key=lambda item: item[0][sweep_dim])
    active: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for offset, shape in ordered:
        begin = offset[sweep_dim]
        active = [
            candidate
            for candidate in active
            if candidate[0][sweep_dim] + candidate[1][sweep_dim] > begin
        ]
        if any(
            all(
                left_begin < right_begin + right_extent
                and right_begin < left_begin + left_extent
                for left_begin, left_extent, right_begin, right_extent in _strict_zip(
                    candidate_offset,
                    candidate_shape,
                    offset,
                    shape,
                )
            )
            for candidate_offset, candidate_shape in active
        ):
            return True
        active.append((offset, shape))
    return False


def _boxes_exactly_cover(
    container_offset: tuple[int, ...],
    container_shape: tuple[int, ...],
    boxes: Sequence[tuple[tuple[int, ...], tuple[int, ...]]],
) -> bool:
    unique_boxes = tuple(dict.fromkeys(boxes))
    if not unique_boxes:
        return False
    if any(
        not _box_contains(container_offset, container_shape, offset, shape)
        for offset, shape in unique_boxes
    ):
        return False
    if sum(prod(shape) for _, shape in unique_boxes) != prod(container_shape):
        return False
    return not _boxes_overlap(unique_boxes)


def _fragments_fully_cover_tensor(
    tensor: TensorDescriptor,
    fragments: Sequence[SourceFragment | PlacementFragment],
) -> bool:
    geometries = {
        (fragment.global_offset, fragment.local_shape): fragment
        for fragment in fragments
        if fragment.tensor_id == tensor.tensor_id
    }
    boxes = tuple(geometries)
    return _boxes_exactly_cover(
        (0,) * len(tensor.global_shape), tensor.global_shape, boxes
    )


def parallel_tensor_owner(
    tensor: TensorDescriptor, fragment: PlacementFragment
) -> RuntimeTensorOwner:
    return tuple(
        (axis.kind, getattr(fragment.rank, axis.kind))
        for axis in tensor.parallel_axes
        if isinstance(axis, OwnershipAxis)
    )


def _validate_target_coverage(
    target_tensors: dict[str, TensorDescriptor],
    target_fragments: Sequence[TargetFragment],
) -> None:
    if not target_fragments:
        raise ValueError("target manifests have no fragments")
    fragments_by_dp_and_tensor: dict[int, dict[str, list[TargetFragment]]] = {}
    for fragment in target_fragments:
        fragments_by_dp_and_tensor.setdefault(fragment.rank.dp, {}).setdefault(
            fragment.tensor_id, []
        ).append(fragment)
    dp_ranks = sorted(fragments_by_dp_and_tensor)
    for dp_rank in dp_ranks:
        for tensor in target_tensors.values():
            fragments_by_owner: dict[RuntimeTensorOwner, list[TargetFragment]] = {}
            for fragment in fragments_by_dp_and_tensor[dp_rank].get(
                tensor.tensor_id, ()
            ):
                fragments_by_owner.setdefault(
                    parallel_tensor_owner(tensor, fragment), []
                ).append(fragment)
            if not fragments_by_owner or any(
                not _fragments_fully_cover_tensor(tensor, fragments)
                for fragments in fragments_by_owner.values()
            ):
                raise ValueError(
                    f"target tensor is not fully covered: {tensor.tensor_id}: "
                    f"dp={dp_rank}"
                )


def complete_parallel_source_replicas(
    source_tensors: dict[str, TensorDescriptor],
    source_fragments: Sequence[PlacementFragment],
) -> dict[int, dict[str, RuntimeTensorOwner]]:
    replicas: dict[int, dict[str, RuntimeTensorOwner]] = {}
    fragments_by_dp_and_tensor: dict[int, dict[str, list[PlacementFragment]]] = {}
    for fragment in source_fragments:
        dp_rank = fragment.rank.dp
        fragments_by_dp_and_tensor.setdefault(dp_rank, {}).setdefault(
            fragment.tensor_id, []
        ).append(fragment)
    for dp_rank in sorted(fragments_by_dp_and_tensor):
        owner_by_tensor: dict[str, RuntimeTensorOwner] = {}
        complete = True
        for tensor in source_tensors.values():
            fragments_by_owner: dict[RuntimeTensorOwner, list[PlacementFragment]] = {}
            for fragment in fragments_by_dp_and_tensor[dp_rank].get(
                tensor.tensor_id, ()
            ):
                fragments_by_owner.setdefault(
                    parallel_tensor_owner(tensor, fragment), []
                ).append(fragment)
            complete_owners = [
                owner
                for owner, fragments in fragments_by_owner.items()
                if _fragments_fully_cover_tensor(tensor, fragments)
            ]
            if not fragments_by_owner or len(complete_owners) != len(
                fragments_by_owner
            ):
                complete = False
                break
            owner_by_tensor[tensor.tensor_id] = min(complete_owners)
        if complete:
            replicas[dp_rank] = owner_by_tensor
    if not replicas:
        raise ValueError(
            "source manifests have no complete DP replica; tensors are not fully covered"
        )
    return replicas


def _validate_local_target_inventory(
    target_tensors: dict[str, TensorDescriptor],
    target_fragments: Sequence[TargetFragment],
) -> None:
    if not target_fragments:
        raise ValueError("local target manifest has no fragments")
    ranks = {fragment.rank for fragment in target_fragments}
    if len(ranks) != 1:
        raise ValueError("local target manifest must describe exactly one executor")
    missing = sorted(
        set(target_tensors) - {item.tensor_id for item in target_fragments}
    )
    if missing:
        raise ValueError(
            f"local target manifest is missing fragments: {', '.join(missing)}"
        )


__all__ = [
    "complete_parallel_source_replicas",
    "parallel_tensor_owner",
]
