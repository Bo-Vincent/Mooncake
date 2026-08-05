from __future__ import annotations

from typing import Sequence

from ..manifest import WeightPlacementManifest
from .contracts import LogicalTransferPlan
from .core import (
    _collect_placements,
    _logical_transfer_plan,
    _plan_transfer,
)


def plan_placement_transfer(
    source_placements: Sequence[WeightPlacementManifest],
    target_placements: Sequence[WeightPlacementManifest],
) -> LogicalTransferPlan:
    """Plan a reshard using only address-free logical placements."""

    source_tensors, source_fragments = _collect_placements(
        source_placements,
        "source",
    )
    target_tensors, target_fragments = _collect_placements(
        target_placements,
        "target",
    )
    source = source_placements[0]
    target = target_placements[0]
    if source.resource_id != target.resource_id:
        raise ValueError("source and target resource_id differ")
    if source.revision != target.revision:
        raise ValueError("source and target revision differ")
    transfer = _plan_transfer(
        source.resource_id,
        source.revision,
        source_tensors,
        source_fragments,
        target_tensors,
        target_fragments,
    )
    return _logical_transfer_plan(
        transfer,
        source_tensors=source_tensors,
        target_tensors=target_tensors,
        source_placements=source_placements,
        target_placements=target_placements,
    )


def plan_placement_transfer_to_local_target(
    source_placements: Sequence[WeightPlacementManifest],
    target_placement: WeightPlacementManifest,
) -> LogicalTransferPlan:
    """Plan one target executor using address-free source and target layouts."""

    source_tensors, source_fragments = _collect_placements(
        source_placements,
        "source",
    )
    target_tensors, target_fragments = _collect_placements(
        (target_placement,),
        "target",
    )
    source = source_placements[0]
    if source.resource_id != target_placement.resource_id:
        raise ValueError("source and target resource_id differ")
    if source.revision != target_placement.revision:
        raise ValueError("source and target revision differ")
    transfer = _plan_transfer(
        source.resource_id,
        source.revision,
        source_tensors,
        source_fragments,
        target_tensors,
        target_fragments,
        local_target=True,
    )
    result = _logical_transfer_plan(
        transfer,
        source_tensors=source_tensors,
        target_tensors=target_tensors,
        source_placements=source_placements,
        target_placements=(target_placement,),
    )
    if len(result.target_executors) != 1:
        raise ValueError("local target placement must describe exactly one executor")
    return result


__all__ = [
    "plan_placement_transfer",
    "plan_placement_transfer_to_local_target",
]
