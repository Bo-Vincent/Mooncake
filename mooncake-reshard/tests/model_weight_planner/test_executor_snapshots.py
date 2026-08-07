from __future__ import annotations

from dataclasses import replace
import pickle
from types import SimpleNamespace

import pytest

from mooncake.reshard.contracts import ResourceId, RevisionId
from mooncake.reshard.weight import (
    bind_logical_transfer_plan,
    plan_placement_transfer,
)
from mooncake.reshard.weight._planner.core import resolve_executor_plans
from mooncake.reshard.weight._planner.contracts import (
    CopyRange,
    TransferPlan,
    TransferRegion,
)

from .helpers import RuntimeInputs, plan_transfer, rebuild_placement, tp_manifests


def _replace_bindings(inputs: RuntimeInputs, bindings) -> RuntimeInputs:
    return RuntimeInputs(inputs.placement, tuple(bindings))


def test_bound_plan_snapshots_only_the_selected_source_dp_replica() -> None:
    sources = tp_manifests(
        tp=2,
        dp=2,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
    )
    targets = tp_manifests(
        tp=4,
        dp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x40000,
        worker_prefix="target",
    )

    plan = plan_transfer(sources, targets)
    assert len(plan.source_executors) == 2
    assert len(plan.target_executors) == len(targets)
    assert {executor.rank.dp for executor in plan.source_executors} == {0}


def test_bind_accepts_participant_local_source_generations() -> None:
    sources = tp_manifests(
        tp=2,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
    )
    targets = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x40000,
        worker_prefix="target",
    )
    logical = plan_placement_transfer(sources.placement, targets.placement)
    mixed_sources = (
        sources.bindings[0],
        replace(sources.bindings[1], generation=sources.bindings[0].generation + 1),
    )

    plan = bind_logical_transfer_plan(
        logical,
        targets.bindings,
        source_bindings=mixed_sources,
    )

    assert {
        executor.participant_id: executor.fragment_leases[0].lease_generation
        for executor in plan.source_executors
    } == {
        mixed_sources[0].participant_id: mixed_sources[0].generation,
        mixed_sources[1].participant_id: mixed_sources[1].generation,
    }


def test_binding_preserves_nd_region_geometry() -> None:
    sources = tp_manifests(
        tp=2,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
    )
    targets = tp_manifests(
        tp=4,
        pp_rank=0,
        ep_rank=0,
        address_base=0x40000,
        worker_prefix="target",
    )
    logical = plan_placement_transfer(sources.placement, targets.placement)
    plan = bind_logical_transfer_plan(
        logical,
        targets.bindings,
        source_bindings=sources.bindings,
    )

    assert len(plan.operations) == len(logical.operations)
    for logical_operation, live_operation in zip(
        logical.operations, plan.operations, strict=True
    ):
        assert live_operation.tensor_id == logical_operation.tensor_id
        assert live_operation.source_offset == logical_operation.source_offset
        assert live_operation.target_offset == logical_operation.target_offset
        assert live_operation.nbytes == logical_operation.nbytes
        assert live_operation.repeat == logical_operation.repeat
        assert live_operation.source_stride == logical_operation.source_stride
        assert live_operation.target_stride == logical_operation.target_stride
        assert isinstance(logical_operation, TransferRegion)
        assert isinstance(live_operation, TransferRegion)
        assert live_operation.overlap_offset == logical_operation.overlap_offset
        assert live_operation.overlap_shape == logical_operation.overlap_shape
        assert live_operation.source_base_offset == logical_operation.source_base_offset
        assert live_operation.target_base_offset == logical_operation.target_base_offset
        assert live_operation.inner_bytes == logical_operation.inner_bytes
        assert live_operation.outer_loop_counts == logical_operation.outer_loop_counts
        assert live_operation.source_strides == logical_operation.source_strides
        assert live_operation.target_strides == logical_operation.target_strides


def test_bind_rejects_duck_typed_runtime_binding() -> None:
    sources = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
    )
    targets = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x40000,
        worker_prefix="target",
    )
    logical = plan_placement_transfer(sources.placement, targets.placement)
    duck_binding = SimpleNamespace(**vars(sources.bindings[0]))

    with pytest.raises(ValueError, match="invalid source runtime binding input"):
        bind_logical_transfer_plan(
            logical,
            targets.bindings,
            source_bindings=(duck_binding,),
        )


@pytest.mark.parametrize("side", ["source", "target"])
def test_bound_plan_records_binding_lease_id_in_executor_snapshot(side: str) -> None:
    sources = tp_manifests(
        tp=2,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
    )
    targets = tp_manifests(
        tp=2,
        pp_rank=0,
        ep_rank=0,
        address_base=0x40000,
        worker_prefix="target",
    )
    sources = _replace_bindings(
        sources,
        (
            replace(binding, lease_id=f"source-lease-{index}")
            for index, binding in enumerate(sources.bindings)
        ),
    )
    targets = _replace_bindings(
        targets,
        (
            replace(binding, lease_id=f"target-lease-{index}")
            for index, binding in enumerate(targets.bindings)
        ),
    )

    plan = plan_transfer(sources, targets)
    bindings = sources.bindings if side == "source" else targets.bindings
    executors = plan.source_executors if side == "source" else plan.target_executors

    assert [executor.runtime_lease_id for executor in executors] == [
        binding.lease_id for binding in bindings
    ]


@pytest.mark.parametrize("side", ["source", "target"])
def test_executor_snapshot_rejects_binding_lease_id_change(side: str) -> None:
    source = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
    )
    target = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x40000,
        worker_prefix="target",
    )
    plan = plan_transfer(source, target)
    current = source if side == "source" else target
    binding = replace(
        current.bindings[0],
        lease_id=f"rotated-{side}-lease",
    )

    with pytest.raises(ValueError, match=f"{side} executor snapshot mismatch"):
        resolve_executor_plans(plan, current.placement, binding, side)


@pytest.mark.parametrize("side", ["source", "target"])
def test_executor_snapshot_rejects_device_change(side: str) -> None:
    source = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
    )
    target = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x40000,
        worker_prefix="target",
    )
    plan = plan_transfer(source, target)
    current = source if side == "source" else target
    binding = replace(
        current.bindings[0],
        fragments=(replace(current.bindings[0].fragments[0], device="cuda:1"),),
    )

    with pytest.raises(ValueError, match=f"{side} executor snapshot mismatch"):
        resolve_executor_plans(plan, current.placement, binding, side)


@pytest.mark.parametrize("side", ["source", "target"])
def test_executor_snapshot_rejects_replaced_placement(side: str) -> None:
    source = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
    )
    target = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x40000,
        worker_prefix="target",
    )
    plan = plan_transfer(source, target)
    current = source if side == "source" else target
    original = current.placement
    replacement = rebuild_placement(
        original,
        tensors=(
            replace(
                original.tensors[0],
                layout_fingerprint="changed:logical-layout",
            ),
        ),
    )
    binding = replace(
        current.bindings[0],
        placement_id=replacement.placement_id,
        placement_digest=replacement.digest,
    )

    with pytest.raises(ValueError, match=f"{side} executor snapshot mismatch"):
        resolve_executor_plans(plan, replacement, binding, side)


def test_executor_snapshot_accepts_the_exact_placement_and_binding() -> None:
    source = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
    )
    target = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x40000,
        worker_prefix="target",
    )
    plan = plan_transfer(source, target)

    assert resolve_executor_plans(
        plan,
        source.placement,
        source.bindings[0],
        "source",
    )
    assert resolve_executor_plans(
        plan,
        target.placement,
        target.bindings[0],
        "target",
    )


def test_public_transfer_plan_rejects_forged_identity() -> None:
    """An executable plan must attest the resource and revision it targets."""

    source = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
    )
    target = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x40000,
        worker_prefix="target",
    )
    with pytest.raises(ValueError, match="transfer plan identity differs"):
        replace(
            plan_transfer(source, target),
            resource_id=ResourceId("different-resource"),
            revision=RevisionId("different-revision"),
        )


@pytest.mark.parametrize("side", ["source", "target"])
def test_public_transfer_plan_rejects_forged_executor_provenance(side: str) -> None:
    source = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
    )
    target = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x40000,
        worker_prefix="target",
    )
    plan = plan_transfer(source, target)
    executor = (
        plan.source_executors[0] if side == "source" else plan.target_executors[0]
    )

    with pytest.raises(ValueError, match=f"{side} executor .*provenance"):
        replace(
            plan,
            **{
                f"{side}_executors": (
                    replace(executor, worker_id=f"forged-{side}-worker"),
                )
            },
        )


def test_public_transfer_plan_rejects_unattested_runtime_view() -> None:
    """Live fragment geometry must come from a fully validated binding."""

    source = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
    )
    target = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x40000,
        worker_prefix="target",
    )
    plan = plan_transfer(source, target)
    operation = plan.operations[0]
    malformed_source = replace(
        operation.source,
        binding=replace(
            operation.source.binding,
            strides_bytes=(
                operation.source.binding.strides_bytes[0],
                operation.source.binding.strides_bytes[1] * 2,
            ),
        ),
        attestation=None,
    )

    with pytest.raises(ValueError, match="attested runtime binding"):
        TransferPlan(
            resource_id=plan.resource_id,
            revision=plan.revision,
            weight_generation=plan.weight_generation,
            operations=(replace(operation, source=malformed_source),),
        )


def test_runtime_attestation_revalidates_after_pickle_round_trip() -> None:
    source = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
    )
    target = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x40000,
        worker_prefix="target",
    )
    plan = pickle.loads(pickle.dumps(plan_transfer(source, target)))

    assert resolve_executor_plans(
        plan,
        source.placement,
        source.bindings[0],
        "source",
    )
