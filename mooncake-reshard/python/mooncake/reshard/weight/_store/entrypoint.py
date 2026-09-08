"""Native Store entry point for manifest-backed model-weight snapshots."""

from __future__ import annotations

from .store import WeightStore
from .snapshot import (
    WeightSnapshotAdapter,
    WeightSnapshotDescriptor,
)
from .writer import (
    WeightStoreWriter,
)


def begin_weight_snapshot(
    store: object,
    snapshot: WeightSnapshotDescriptor,
    adapter: WeightSnapshotAdapter,
) -> WeightStoreWriter:
    """Open an explicit manifest-backed model-weight snapshot writer."""

    return WeightStore(store).begin_weight_snapshot(snapshot, adapter)


def begin_managed_weight_snapshot(
    store: object,
    snapshot: WeightSnapshotDescriptor,
    adapter: WeightSnapshotAdapter,
    *,
    tenant_id: str = "default",
) -> WeightStoreWriter:
    """Open a snapshot writer published through the Store WeightCatalog."""

    return WeightStore(store).begin_managed_weight_snapshot(
        snapshot,
        adapter,
        tenant_id=tenant_id,
    )


__all__ = ["begin_managed_weight_snapshot", "begin_weight_snapshot"]
