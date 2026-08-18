"""Resource-neutral N-D logical-box helpers for reshard planners."""

from __future__ import annotations

from math import prod
from typing import Iterable, Protocol, Sequence

from ._compat import _strict_zip


class LogicalBox(Protocol):
    """An address-free logical box supplied by a resource-specific adapter."""

    @property
    def global_offset(self) -> tuple[int, ...]: ...

    @property
    def local_shape(self) -> tuple[int, ...]: ...


class OverlapRegion(Protocol):
    """The logical overlap portion of a resource-specific transfer region."""

    @property
    def overlap_offset(self) -> tuple[int, ...]: ...

    @property
    def overlap_shape(self) -> tuple[int, ...]: ...


def box_contains(
    outer_offset: tuple[int, ...],
    outer_shape: tuple[int, ...],
    inner_offset: tuple[int, ...],
    inner_shape: tuple[int, ...],
) -> bool:
    """Return whether one N-D logical box is wholly inside another."""

    return all(
        outer_begin <= inner_begin
        and inner_begin + inner_extent <= outer_begin + outer_extent
        for outer_begin, outer_extent, inner_begin, inner_extent in _strict_zip(
            outer_offset,
            outer_shape,
            inner_offset,
            inner_shape,
        )
    )


def boxes_overlap(
    boxes: Sequence[tuple[tuple[int, ...], tuple[int, ...]]],
) -> bool:
    """Return whether any two N-D boxes overlap."""

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


def boxes_exactly_cover(
    container_offset: tuple[int, ...],
    container_shape: tuple[int, ...],
    boxes: Sequence[tuple[tuple[int, ...], tuple[int, ...]]],
) -> bool:
    """Require in-bounds, non-overlapping boxes with exact logical volume."""

    if not boxes:
        return False
    if any(
        not box_contains(container_offset, container_shape, offset, shape)
        for offset, shape in boxes
    ):
        return False
    if sum(prod(shape) for _, shape in boxes) != prod(container_shape):
        return False
    return not boxes_overlap(boxes)


def regions_exactly_cover(
    target: LogicalBox,
    regions: Iterable[OverlapRegion],
) -> bool:
    """Require resource-neutral N-D regions to completely cover one target box."""

    return boxes_exactly_cover(
        target.global_offset,
        target.local_shape,
        tuple((region.overlap_offset, region.overlap_shape) for region in regions),
    )


__all__ = [
    "LogicalBox",
    "OverlapRegion",
    "box_contains",
    "boxes_exactly_cover",
    "boxes_overlap",
    "regions_exactly_cover",
]
