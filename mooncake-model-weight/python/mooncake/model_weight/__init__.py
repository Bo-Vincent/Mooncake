"""Public contracts for framework-neutral model weight management."""

from .manifest import (
    ParallelRank,
    PlacementFragment,
    PlacementManifest,
    RuntimeBindingFragment,
    RuntimeBindingManifest,
    RuntimeFragment,
    RuntimeManifest,
    SourcePlacementManifest,
    StoredFragment,
    TargetPlacementManifest,
    TensorDescriptor,
    WeightManifest,
    bind_runtime_manifest,
    placement_manifest_from_runtime_manifest,
    runtime_binding_from_runtime_manifest,
)
from .planner import (
    CopyRange,
    ExecutorTransferPlan,
    LogicalTransferPlan,
    PipelineRouteGroup,
    PlacementExecutorPlan,
    TransferPlan,
    TransferRegion,
    bind_logical_transfer_plan,
    plan_placement_transfer,
    plan_placement_transfer_to_local_target,
    plan_runtime_transfer,
    plan_runtime_transfer_to_local_target,
    plan_runtime_transfer_to_local_target_placement,
    plan_runtime_transfer_to_target_placements,
    plan_stored_transfer,
    plan_stored_transfer_to_target_placements,
)

MODEL_WEIGHT_CAPABILITIES = frozenset(
    {
        "nd_logical_box",
        "placement_binding",
        "runtime_manifest",
        "store_weight_manifest",
    }
)


def supports_model_weight_capability(capability: str) -> bool:
    return capability in MODEL_WEIGHT_CAPABILITIES


__all__ = [
    "MODEL_WEIGHT_CAPABILITIES",
    "supports_model_weight_capability",
    "ParallelRank",
    "PlacementFragment",
    "PlacementManifest",
    "RuntimeBindingFragment",
    "RuntimeBindingManifest",
    "RuntimeFragment",
    "RuntimeManifest",
    "SourcePlacementManifest",
    "StoredFragment",
    "TargetPlacementManifest",
    "TensorDescriptor",
    "WeightManifest",
    "bind_runtime_manifest",
    "placement_manifest_from_runtime_manifest",
    "runtime_binding_from_runtime_manifest",
    "CopyRange",
    "ExecutorTransferPlan",
    "LogicalTransferPlan",
    "PipelineRouteGroup",
    "PlacementExecutorPlan",
    "TransferPlan",
    "TransferRegion",
    "bind_logical_transfer_plan",
    "plan_placement_transfer",
    "plan_placement_transfer_to_local_target",
    "plan_runtime_transfer",
    "plan_runtime_transfer_to_local_target",
    "plan_runtime_transfer_to_local_target_placement",
    "plan_runtime_transfer_to_target_placements",
    "plan_stored_transfer",
    "plan_stored_transfer_to_target_placements",
]
