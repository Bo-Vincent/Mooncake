from __future__ import annotations

import pytest

from mooncake.reshard.weight.management import (
    WeightAvailabilityState,
    WeightOperationKind,
    WeightRevisionIdentity,
)
from mooncake.reshard.weight.store import WeightSnapshotDescriptor

from .helpers import make_weight_store, source_manifests, target_manifests
from .test_managed_revision import ManagedInMemoryStore
from .test_snapshot_writer import _SnapshotAdapter


def test_managed_writer_abort_before_first_tensor_cleans_import() -> None:
    raw = ManagedInMemoryStore()
    _, weight_store = make_weight_store(raw)
    sources = source_manifests(dp=1, tp=1)

    with pytest.raises(RuntimeError, match="cancel upload"):
        with weight_store.weight_put(
            WeightSnapshotDescriptor(
                resource_id=sources.placement.resource_id,
                revision=sources.placement.revision,
                weight_generation=sources.placement.weight_generation,
            ),
            _SnapshotAdapter(sources),
            tenant_id="tenant-a",
        ):
            raise RuntimeError("cancel upload")

    assert all(
        metadata.availability is not WeightAvailabilityState.IMPORTING
        for metadata in raw.catalog.values()
    )


def test_managed_revision_public_workflow_end_to_end() -> None:
    raw = ManagedInMemoryStore()
    _, weight_store = make_weight_store(raw)
    sources = source_manifests(dp=1, tp=2)
    targets = target_manifests(dp=1, tp=1)

    writer = weight_store.weight_put(
        WeightSnapshotDescriptor(
            resource_id=sources.placement.resource_id,
            revision=sources.placement.revision,
            weight_generation=sources.placement.weight_generation,
            namespace="production",
        ),
        _SnapshotAdapter(sources),
        tenant_id="tenant-a",
    )
    identity = WeightRevisionIdentity(
        tenant_id="tenant-a",
        namespace="production",
        resource_id=sources.placement.resource_id,
        revision=sources.placement.revision,
        weight_generation=sources.placement.weight_generation,
    )
    assert raw.catalog[identity].availability is (WeightAvailabilityState.IMPORTING)

    for binding in sources.bindings:
        fragment = binding.fragments[0]
        writer.weight_put_tensor(
            sources.placement.tensors[0].tensor_id,
            fragment.placement_fragment_id,
        )
    manifest = writer.commit()

    view = weight_store.weight_get_metadata(identity)
    assert view.metadata.availability is WeightAvailabilityState.READY
    assert view.metadata.operation_id is not None
    assert (
        weight_store.weight_get_operation(
            view.metadata.operation_id,
            tenant_id="tenant-a",
        ).kind
        is WeightOperationKind.MIGRATING
    )
    assert view.metadata.manifest.manifest_key == manifest.manifest_key
    assert view.metadata.manifest.payload_group_id == manifest.group_id

    loaded = weight_store.weight_get(
        identity,
        targets.placement,
        targets.bindings,
    )
    assert loaded == manifest
    assert raw.released_leases == [1]
