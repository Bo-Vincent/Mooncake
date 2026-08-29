"""Negative static checks for canonical reshard contract categories.

This file is intentionally invalid. The type-check script requires pyright to
reject it after the production contract has passed strict checking.
"""

from mooncake.reshard.contracts import (
    ParticipantId,
    PlacementFragmentId,
    PlacementId,
    ResourceId,
    RuntimeFragmentId,
    StoredFragmentSnapshotId,
)
from mooncake.reshard.weight.planner import TransferPlan
from mooncake.reshard.weight.types import SplitAxis


participant_id = ParticipantId("participant-0")
placement_id: PlacementId = participant_id
resource_id: ResourceId = PlacementId("placement-0")
runtime_fragment_id: RuntimeFragmentId = PlacementFragmentId("placement-fragment-0")
stored_fragment_id: StoredFragmentSnapshotId = RuntimeFragmentId("runtime-fragment-0")
TransferPlan(
    resource_id=PlacementId("placement-1"),
    revision="revision",
    weight_generation=0,
    operations=(),
)
SplitAxis(kind="pp", dim=0)
