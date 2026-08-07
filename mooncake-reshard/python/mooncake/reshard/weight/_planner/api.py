from __future__ import annotations

from ..manifest import WeightPlacementManifest
from .contracts import LogicalTransferPlan
from .core import (
    _collect_placements,
    _logical_transfer_plan,
    _plan_transfer,
)


def plan_placement_transfer(
    source_placement: WeightPlacementManifest,
    target_placement: WeightPlacementManifest,
) -> LogicalTransferPlan:
    """Plan a reshard between two complete address-free placements."""

    source_tensors, source_fragments = _collect_placements(
        (source_placement,),
        "source",
    )
    target_tensors, target_fragments = _collect_placements(
        (target_placement,),
        "target",
    )
    source = source_placement
    target = target_placement
    if source.resource_id != target.resource_id:
        raise ValueError("source and target resource_id differ")
    if source.revision != target.revision:
        raise ValueError("source and target revision differ")
    if source.weight_generation != target.weight_generation:
        raise ValueError("source and target weight_generation differ")
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
        source_placement=source_placement,
        target_placement=target_placement,
    )


def plan_placement_transfer_to_local_target(
    source_placement: WeightPlacementManifest,
    target_placement: WeightPlacementManifest,
    target_participant_id: str | None = None,
) -> LogicalTransferPlan:
    """Plan one target executor using address-free source and target layouts."""

    source_tensors, source_fragments = _collect_placements(
        (source_placement,),
        "source",
    )
    if target_participant_id is None:
        if target_placement.topology.world_size != 1:
            raise ValueError(
                "target_participant_id is required for a multi-participant target"
            )
        target_participant_id = target_placement.parts[0].participant_id
    try:
        target_part = next(
            part
            for part in target_placement.parts
            if part.participant_id == target_participant_id
        )
    except StopIteration as error:
        raise ValueError(
            f"unknown target participant: {target_participant_id}"
        ) from error
    target_tensors = {tensor.tensor_id: tensor for tensor in target_part.tensors}
    target_fragments = list(target_part.fragments)
    source = source_placement
    if source.resource_id != target_placement.resource_id:
        raise ValueError("source and target resource_id differ")
    if source.revision != target_placement.revision:
        raise ValueError("source and target revision differ")
    if source.weight_generation != target_placement.weight_generation:
        raise ValueError("source and target weight_generation differ")
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
        source_placement=source_placement,
        target_placement=target_placement,
        target_participant_ids=frozenset({target_participant_id}),
    )
    if len(result.target_executors) != 1:
        raise ValueError("local target placement must describe exactly one executor")
    return result


__all__ = [
    "plan_placement_transfer",
    "plan_placement_transfer_to_local_target",
]
