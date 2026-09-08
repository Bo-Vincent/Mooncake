from __future__ import annotations

from mooncake.reshard.weight.management import WeightAvailabilityState

from .helpers import make_weight_store, source_manifests, target_manifests
from .test_managed_revision import ManagedInMemoryStore


def test_managed_revision_public_workflow_end_to_end() -> None:
    raw = ManagedInMemoryStore()
    _, weight_store = make_weight_store(raw)
    sources = source_manifests(dp=1, tp=2)
    targets = target_manifests(dp=1, tp=1)

    plan = weight_store.plan_managed_upload(
        sources.placement,
        sources.bindings,
        namespace="production",
        tenant_id="tenant-a",
    )
    assert plan.management_identity is not None
    assert raw.catalog[plan.management_identity].availability is (
        WeightAvailabilityState.IMPORTING
    )

    receipts = []
    for binding in sources.bindings:
        receipts.extend(weight_store.upload(plan, sources.placement, binding))
    manifest = weight_store.commit_upload(plan, receipts)

    view = weight_store.get_weight_revision(plan.management_identity)
    assert view.metadata.availability is WeightAvailabilityState.READY
    assert view.metadata.manifest.manifest_key == manifest.manifest_key
    assert view.metadata.manifest.payload_group_id == manifest.group_id

    loaded = weight_store.load_weight_revision(
        plan.management_identity,
        targets.placement,
        targets.bindings,
    )
    assert loaded == manifest
    assert raw.released_leases == [1]
