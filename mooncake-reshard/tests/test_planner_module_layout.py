from __future__ import annotations

import mooncake.reshard.weight._planner.contracts as logical_contracts
import mooncake.reshard.weight.planner as planner
from mooncake.reshard.weight import (
    ParallelRank,
    PlacementFragment,
    RuntimeBindingFragment,
    TensorDescriptor,
    SplitAxis,
    WeightPlacementManifest,
    WeightRuntimeBindingManifest,
)
from mooncake.reshard.weight._planner.api import (
    plan_placement_transfer,
    plan_placement_transfer_to_local_target,
    plan_stored_transfer_to_target_placement,
)
from mooncake.reshard.weight._planner.binding import (
    bind_logical_transfer_plan,
)
from mooncake.reshard.weight._planner.bound_contracts import (
    ExecutorTransferPlan,
    TransferPlan,
)
from mooncake.reshard.weight._planner.contracts import (
    BoundWeightFragment,
    LogicalTransferPlan,
    PipelineRouteGroup,
    PlacementExecutorPlan,
    TransferRegion,
)
from model_weight_planner.helpers import global_placement_from_fragments


def _placement(
    *,
    placement_fragment_id: str,
    offset: int,
    extent: int,
    tp: int,
) -> WeightPlacementManifest:
    tensor = TensorDescriptor(
        tensor_id="layers.0.weight",
        global_shape=(8,),
        dtype="uint8",
        itemsize=1,
        shard_dims=(0,),
        layout_fingerprint="test:logical-box",
        layer_id=0,
        parallel_axes=(SplitAxis(kind="tp", dim=0),),
    )
    primary = PlacementFragment(
        placement_fragment_id=placement_fragment_id,
        tensor_id=tensor.tensor_id,
        global_offset=(offset,),
        local_shape=(extent,),
        nbytes=extent,
        rank=ParallelRank(tp=tp),
    )
    fragments = [primary]
    participant_ids = {primary.rank: f"{placement_fragment_id}-participant"}
    if extent != tensor.global_shape[0]:
        complement = PlacementFragment(
            placement_fragment_id=f"{placement_fragment_id}-complement",
            tensor_id=tensor.tensor_id,
            global_offset=(0,),
            local_shape=(offset,),
            nbytes=offset,
            rank=ParallelRank(tp=0),
        )
        fragments.append(complement)
        participant_ids[complement.rank] = f"{placement_fragment_id}-complement"
    return global_placement_from_fragments(
        resource_id="model",
        revision="revision",
        placement_set_id=placement_fragment_id,
        tensors=(tensor,),
        fragments=fragments,
        participant_ids=participant_ids,
    )


def _binding(
    placement: WeightPlacementManifest,
    *,
    placement_fragment_id: str,
    fragment_id: str,
    address: int,
    generation: int,
) -> WeightRuntimeBindingManifest:
    placement_part = next(
        part
        for part in placement.parts
        if any(
            fragment.placement_fragment_id == placement_fragment_id
            for fragment in part.fragments
        )
    )
    placement_fragment = next(
        fragment
        for fragment in placement_part.fragments
        if fragment.placement_fragment_id == placement_fragment_id
    )
    return WeightRuntimeBindingManifest(
        resource_id=placement.resource_id,
        revision=placement.revision,
        placement_id=placement.placement_id,
        placement_digest=placement.digest,
        instance_id=f"{fragment_id}-instance",
        participant_id=placement_part.participant_id,
        generation=generation,
        lease_id=f"{fragment_id}-lease",
        fragments=(
            RuntimeBindingFragment(
                placement_fragment_id=(placement_fragment.placement_fragment_id),
                fragment_id=fragment_id,
                address=address,
                nbytes=placement_fragment.nbytes,
                worker_id=f"{fragment_id}-worker",
                endpoint=f"{fragment_id}-endpoint",
                device="cuda:0",
                itemsize=1,
                local_shape=placement_fragment.local_shape,
                strides_bytes=(1,),
                storage_address=address,
                storage_nbytes=placement_fragment.nbytes,
                storage_offset_bytes=0,
            ),
        ),
    )


def test_planner_responsibility_modules_preserve_public_identity() -> None:
    assert planner.BoundWeightFragment is BoundWeightFragment
    assert planner.ExecutorTransferPlan is ExecutorTransferPlan
    assert planner.LogicalTransferPlan is LogicalTransferPlan
    assert planner.PipelineRouteGroup is PipelineRouteGroup
    assert planner.PlacementExecutorPlan is PlacementExecutorPlan
    assert planner.TransferPlan is TransferPlan
    assert planner.TransferRegion is TransferRegion
    assert planner.bind_logical_transfer_plan is bind_logical_transfer_plan
    assert planner.plan_placement_transfer is plan_placement_transfer
    assert (
        planner.plan_placement_transfer_to_local_target
        is plan_placement_transfer_to_local_target
    )
    assert (
        planner.plan_stored_transfer_to_target_placement
        is plan_stored_transfer_to_target_placement
    )


def test_logical_contracts_module_excludes_runtime_bound_plan_types() -> None:
    assert not hasattr(logical_contracts, "TransferPlan")
    assert not hasattr(logical_contracts, "ExecutorTransferPlan")
    assert not hasattr(logical_contracts, "RuntimeFragmentSnapshot")


def test_planner_does_not_export_combined_runtime_convenience_apis() -> None:
    assert not hasattr(planner, "RuntimeBindingInput")
    assert not hasattr(planner, "plan_runtime_transfer")
    assert not hasattr(planner, "plan_runtime_transfer_to_local_target")
    assert not hasattr(
        planner,
        "plan_runtime_transfer_to_local_target_placement",
    )
    assert not hasattr(planner, "plan_runtime_transfer_to_target_placements")
    assert not hasattr(planner, "plan_stored_transfer_to_target_placements")
    assert not hasattr(planner, "plan_stored_transfer")


def test_logical_plan_binds_placement_and_runtime_binding_without_manifest() -> None:
    source = _placement(
        placement_fragment_id="source-placement",
        offset=0,
        extent=8,
        tp=0,
    )
    target = _placement(
        placement_fragment_id="target-placement",
        offset=4,
        extent=4,
        tp=1,
    )
    target_participant_id = next(
        part.participant_id
        for part in target.parts
        if any(
            fragment.placement_fragment_id == "target-placement"
            for fragment in part.fragments
        )
    )
    logical = plan_placement_transfer_to_local_target(
        source,
        target,
        target_participant_id,
    )

    assert isinstance(logical.operations[0].source, PlacementFragment)
    assert isinstance(logical.operations[0].target, PlacementFragment)

    bound = bind_logical_transfer_plan(
        logical,
        (
            _binding(
                target,
                placement_fragment_id="target-placement",
                fragment_id="target",
                address=0x9000,
                generation=7,
            ),
        ),
        source_bindings=(
            _binding(
                source,
                placement_fragment_id="source-placement",
                fragment_id="source",
                address=0x1000,
                generation=3,
            ),
        ),
    )

    operation = bound.operations[0]
    assert isinstance(operation.source, BoundWeightFragment)
    assert isinstance(operation.target, BoundWeightFragment)
    assert operation.source.address == 0x1000
    assert operation.target.address == 0x9000
    assert operation.source.global_offset == (0,)
    assert operation.target.global_offset == (4,)
    assert bound.source_executors[0].runtime_lease_id == "source-lease"
    assert bound.source_executors[0].fragment_snapshots[0].lease_generation == 3
