from __future__ import annotations

from typing import Sequence

from ..manifest import (
    RuntimeBindingFragment,
    WeightPlacementManifest,
    WeightRuntimeBindingManifest,
    validate_runtime_binding,
)
from ..planner import (
    BoundWeightFragment,
    ExecutorTransferPlan,
    TransferPlan,
    resolve_executor_plans,
)
from .completion import TransferEngineError


def validate_lowering_limits(
    *,
    max_batch_operations: int,
    max_region_segments: int,
    max_completion_drain_attempts: int,
    completion_drain_timeout_ms: int,
) -> None:
    if (
        max_batch_operations <= 0
        or max_region_segments <= 0
        or max_completion_drain_attempts <= 0
        or completion_drain_timeout_ms < 0
    ):
        raise ValueError("transfer lowering limits must be positive")


def validate_plan_identity(
    plan: TransferPlan,
    placement: WeightPlacementManifest,
    label: str,
) -> None:
    if placement.resource_id != plan.resource_id:
        raise TransferEngineError(f"{label} resource_id mismatch")
    if placement.revision != plan.revision:
        raise TransferEngineError(f"{label} revision mismatch")
    if placement.weight_generation != plan.weight_generation:
        raise TransferEngineError(f"{label} weight_generation mismatch")


def validate_manifest_pair(
    plan: TransferPlan,
    placement: WeightPlacementManifest,
    binding: WeightRuntimeBindingManifest,
    label: str,
) -> None:
    validate_plan_identity(plan, placement, label)
    try:
        validate_runtime_binding(placement, binding)
    except ValueError as error:
        raise TransferEngineError(
            f"invalid {label} runtime binding: {error}"
        ) from error


def runtime_binding_fragment(
    fragment: RuntimeBindingFragment | BoundWeightFragment,
) -> RuntimeBindingFragment:
    if isinstance(fragment, RuntimeBindingFragment):
        return fragment
    if isinstance(fragment, BoundWeightFragment):
        return fragment.binding
    raise TransferEngineError(
        "transfer plan physical fragment must be a RuntimeBindingFragment "
        "or expose one as .binding"
    )


def pair_manifests(
    placement: WeightPlacementManifest,
    bindings: Sequence[WeightRuntimeBindingManifest],
    label: str,
) -> tuple[tuple[WeightPlacementManifest, WeightRuntimeBindingManifest], ...]:
    if not isinstance(placement, WeightPlacementManifest):
        raise TransferEngineError(
            f"{label} placement must be a WeightPlacementManifest"
        )
    binding_items = tuple(bindings)
    participant_ids = [binding.participant_id for binding in binding_items]
    if len(participant_ids) != len(set(participant_ids)):
        raise TransferEngineError(f"duplicate {label} runtime binding participant")
    if any(binding.placement_id != placement.placement_id for binding in binding_items):
        raise TransferEngineError(f"{label} placement and binding IDs differ")
    return tuple((placement, binding) for binding in binding_items)


def resolve_runtime_executors(
    plan: TransferPlan,
    placement: WeightPlacementManifest,
    binding: WeightRuntimeBindingManifest,
    label: str,
) -> tuple[ExecutorTransferPlan, ...]:
    return resolve_executor_plans(plan, placement, binding, label)
