from __future__ import annotations

from dataclasses import dataclass

from hypothesis import given, settings, strategies as st

from mooncake.reshard.geometry import (
    LogicalBox,
    OverlapRegion,
    boxes_exactly_cover,
    regions_exactly_cover,
)


@dataclass(frozen=True)
class _Box:
    global_offset: tuple[int, ...]
    local_shape: tuple[int, ...]


@dataclass(frozen=True)
class _Region:
    overlap_offset: tuple[int, ...]
    overlap_shape: tuple[int, ...]


def test_geometry_contract_is_structural_and_resource_agnostic() -> None:
    target: LogicalBox = _Box(global_offset=(8, 4), local_shape=(2, 6))
    regions: tuple[OverlapRegion, ...] = (
        _Region(overlap_offset=(8, 4), overlap_shape=(1, 6)),
        _Region(overlap_offset=(9, 4), overlap_shape=(1, 6)),
    )

    assert regions_exactly_cover(target, regions)
    assert boxes_exactly_cover(
        target.global_offset,
        target.local_shape,
        tuple((region.overlap_offset, region.overlap_shape) for region in regions),
    )
    assert not regions_exactly_cover(target, (regions[0], regions[0], regions[1]))


@st.composite
def _tiled_target_and_regions(draw) -> tuple[_Box, tuple[_Region, ...]]:
    ndim = draw(st.integers(min_value=1, max_value=3))
    split_dim = draw(st.integers(min_value=0, max_value=ndim - 1))
    origin = tuple(draw(st.integers(min_value=0, max_value=8)) for _ in range(ndim))
    shape = tuple(
        draw(st.integers(min_value=2 if dim == split_dim else 1, max_value=6))
        for dim in range(ndim)
    )
    split = draw(st.integers(min_value=1, max_value=shape[split_dim] - 1))
    left_shape = tuple(
        split if dim == split_dim else extent for dim, extent in enumerate(shape)
    )
    right_offset = tuple(
        origin[dim] + split if dim == split_dim else origin[dim] for dim in range(ndim)
    )
    right_shape = tuple(
        extent - split if dim == split_dim else extent
        for dim, extent in enumerate(shape)
    )
    return _Box(origin, shape), (
        _Region(origin, left_shape),
        _Region(right_offset, right_shape),
    )


@settings(max_examples=80, deadline=None)
@given(_tiled_target_and_regions())
def test_regions_exactly_cover_random_nd_tilings(
    tiled: tuple[_Box, tuple[_Region, ...]],
) -> None:
    target, regions = tiled

    assert regions_exactly_cover(target, regions)
    assert not regions_exactly_cover(target, regions[:1])
