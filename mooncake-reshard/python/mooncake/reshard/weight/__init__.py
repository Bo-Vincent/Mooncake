"""Framework-neutral model-weight reshard contracts and execution adapters."""

from .manifest import (
    OwnershipAxis,
    ParallelRank,
    ParallelTopology,
    PlacementFragment,
    ReplicatedAxis,
    RuntimeBindingFragment,
    SplitAxis,
    TensorDescriptor,
    TopologyParticipant,
    WeightPlacementManifest,
    WeightPlacementPart,
    WeightRuntimeBindingManifest,
    validate_runtime_binding,
    validate_runtime_bindings,
)
from .serde import weight_placement_from_json, weight_placement_to_json
from .planner import (
    BoundWeightFragment,
    ExecutorTransferPlan,
    LogicalTransferPlan,
    PlanningLimits,
    PipelineRouteGroup,
    PlacementExecutorPlan,
    RuntimeLeaseSnapshot,
    RuntimeBindingAttestation,
    TransferPlan,
    TransferRegion,
    bind_logical_transfer_plan,
    plan_placement_transfer,
    plan_placement_transfer_to_local_target,
    plan_stored_transfer_to_target_placement,
    resolve_executor_plan,
    resolve_executor_plans,
)
from .storage_manifest import StoredFragment, WeightManifest
from .store import (
    StoreRegistrationLease,
    UploadOperation,
    UploadReceipt,
    WeightLoadPlan,
    WeightStore,
    WeightStoreError,
    WeightUploadPlan,
)
from .te import (
    DirectReadReceipt,
    DirectTransferReceipt,
    MemoryRegistrationLease,
    MooncakeTransferEngineReader,
    MooncakeTransferEngineSink,
    TransferCompletionUnknownError,
    TransferEngineError,
    WeightAllocationGuardProvider,
    WeightAllocationGuardProviders,
)


WEIGHT_RESHARD_CAPABILITIES = frozenset(
    {
        "nd_logical_box",
        "placement_binding",
        "dependent_axis_projection",
        "store_weight_manifest",
        "te_execution",
    }
)


def supports_weight_reshard_capability(capability: str) -> bool:
    return capability in WEIGHT_RESHARD_CAPABILITIES


__all__ = [
    "WEIGHT_RESHARD_CAPABILITIES",
    "supports_weight_reshard_capability",
    "ParallelRank",
    "ParallelTopology",
    "PlacementFragment",
    "RuntimeBindingFragment",
    "SplitAxis",
    "ReplicatedAxis",
    "OwnershipAxis",
    "TensorDescriptor",
    "TopologyParticipant",
    "WeightPlacementManifest",
    "WeightPlacementPart",
    "WeightRuntimeBindingManifest",
    "validate_runtime_binding",
    "validate_runtime_bindings",
    "weight_placement_from_json",
    "weight_placement_to_json",
    "StoredFragment",
    "StoreRegistrationLease",
    "WeightManifest",
    "BoundWeightFragment",
    "ExecutorTransferPlan",
    "LogicalTransferPlan",
    "PlanningLimits",
    "PipelineRouteGroup",
    "PlacementExecutorPlan",
    "RuntimeLeaseSnapshot",
    "RuntimeBindingAttestation",
    "TransferPlan",
    "TransferRegion",
    "bind_logical_transfer_plan",
    "plan_placement_transfer",
    "plan_placement_transfer_to_local_target",
    "plan_stored_transfer_to_target_placement",
    "resolve_executor_plan",
    "resolve_executor_plans",
    "UploadOperation",
    "UploadReceipt",
    "WeightLoadPlan",
    "WeightStore",
    "WeightStoreError",
    "WeightUploadPlan",
    "DirectReadReceipt",
    "DirectTransferReceipt",
    "MemoryRegistrationLease",
    "MooncakeTransferEngineReader",
    "MooncakeTransferEngineSink",
    "TransferCompletionUnknownError",
    "TransferEngineError",
    "WeightAllocationGuardProvider",
    "WeightAllocationGuardProviders",
]
