from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from mooncake.reshard.weight.management import (
    WeightAvailabilityState,
    WeightManifestReference,
    WeightOperationState,
    WeightResidencyState,
    WeightRevisionIdentity,
    WeightRevisionLease,
    WeightRevisionMetadata,
    WeightRevisionPage,
    WeightRevisionView,
)
from mooncake.reshard.weight.store import WeightStoreError

from .helpers import InMemoryStore, make_weight_store, source_manifests, target_manifests


def _identity_digest(identity: WeightRevisionIdentity) -> str:
    digest = hashlib.sha256()
    for value in (
        identity.tenant_id,
        identity.namespace,
        identity.resource_id,
        identity.revision,
        str(identity.weight_generation),
    ):
        encoded = value.encode()
        digest.update(str(len(encoded)).encode())
        digest.update(b":")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


class ManagedInMemoryStore(InMemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.catalog: dict[WeightRevisionIdentity, WeightRevisionMetadata] = {}
        self.next_lease_id = 1
        self.released_leases: list[int] = []

    @staticmethod
    def _identity(args: tuple[object, ...]) -> WeightRevisionIdentity:
        return WeightRevisionIdentity(*args[:5])

    def begin_weight_import(self, *args):
        identity = self._identity(args)
        group_id = f"weight:{_identity_digest(identity)}"
        current = self.catalog.get(identity)
        if current is None:
            current = WeightRevisionMetadata(
                identity=identity,
                manifest=WeightManifestReference(
                    manifest_key="",
                    manifest_sha256="",
                    payload_group_id=group_id,
                    payload_keys_sha256="",
                    payload_count=args[6],
                    logical_bytes=args[7],
                ),
                availability=WeightAvailabilityState.IMPORTING,
                residency=WeightResidencyState.UNKNOWN,
                operation=WeightOperationState.NONE,
                operation_id=0,
                metadata_generation=1,
                created_at_ms=1,
                updated_at_ms=1,
            )
            self.catalog[identity] = current
        return current, 0, 0

    def commit_weight_import(self, *args):
        identity = self._identity(args)
        current = self.catalog[identity]
        reference = WeightManifestReference(
            manifest_key=args[6],
            manifest_sha256=args[7],
            payload_group_id=args[8],
            payload_keys_sha256=args[9],
            payload_count=args[10],
            logical_bytes=args[11],
        )
        if current.availability is WeightAvailabilityState.READY:
            assert current.manifest == reference
            return current, 0, 0
        updated = replace(
            current,
            manifest=reference,
            availability=WeightAvailabilityState.READY,
            residency=WeightResidencyState.HOT,
            metadata_generation=2,
            updated_at_ms=2,
        )
        self.catalog[identity] = updated
        return updated, 0, 0

    def get_weight_revision(self, *args):
        metadata = self.catalog[self._identity(args)]
        return WeightRevisionView(metadata, 0, None), 0, 0

    def list_weight_revisions(self, *args):
        tenant_id, namespace, resource_id, _page_token, limit = args
        revisions = tuple(
            WeightRevisionView(metadata, 0, None)
            for identity, metadata in sorted(
                self.catalog.items(), key=lambda item: item[0].revision
            )
            if identity.tenant_id == tenant_id
            and identity.namespace == namespace
            and identity.resource_id == resource_id
        )[:limit]
        return WeightRevisionPage(revisions, ""), 0, 0

    def acquire_weight_revision_lease(self, *args):
        identity = self._identity(args)
        metadata = self.catalog[identity]
        lease = WeightRevisionLease(
            lease_id=self.next_lease_id,
            identity=identity,
            holder=args[6],
            expires_at_ms=1000 + args[7],
            fenced_metadata_generation=metadata.metadata_generation,
        )
        self.next_lease_id += 1
        return lease, 0, 0

    def release_weight_revision_lease(self, tenant_id, lease_id):
        assert tenant_id == "tenant-a"
        self.released_leases.append(lease_id)
        return True, 0, 0


def test_managed_revision_publish_resolve_load_and_release_lease() -> None:
    raw = ManagedInMemoryStore()
    _, weight_store = make_weight_store(raw)
    sources = source_manifests(dp=1, tp=2)
    targets = target_manifests(dp=1, tp=2)

    plan = weight_store.plan_managed_upload(
        sources.placement,
        sources.bindings,
        namespace="production",
        tenant_id="tenant-a",
    )
    assert plan.management_identity is not None
    assert plan.manifest.group_id.startswith("weight:")

    receipts = []
    for binding in sources.bindings:
        receipts.extend(weight_store.upload(plan, sources.placement, binding))
    manifest = weight_store.commit_upload(plan, receipts)
    view = weight_store.get_weight_revision(plan.management_identity)

    assert view.metadata.availability is WeightAvailabilityState.READY
    assert view.metadata.manifest.manifest_key == manifest.manifest_key
    assert view.metadata.manifest.manifest_sha256 == manifest.manifest_digest
    revision_keys = {
        manifest.manifest_key,
        *(fragment.object_key for fragment in manifest.fragments),
    }
    assert {raw.group_ids[key] for key in revision_keys} == {manifest.group_id}
    assert weight_store.list_weight_revisions(
        tenant_id="tenant-a",
        namespace="production",
        resource_id=manifest.resource_id,
    ).revisions == (view,)

    loaded = weight_store.load_weight_revision(
        plan.management_identity,
        targets.placement,
        targets.bindings,
    )
    assert loaded == manifest
    assert raw.released_leases == [1]


def test_managed_load_releases_lease_after_digest_rejection() -> None:
    raw = ManagedInMemoryStore()
    _, weight_store = make_weight_store(raw)
    sources = source_manifests(dp=1, tp=1)
    targets = target_manifests(dp=1, tp=1)
    plan = weight_store.plan_managed_upload(
        sources.placement,
        sources.bindings,
        tenant_id="tenant-a",
    )
    receipts = []
    for binding in sources.bindings:
        receipts.extend(weight_store.upload(plan, sources.placement, binding))
    weight_store.commit_upload(plan, receipts)
    assert plan.management_identity is not None
    raw.catalog[plan.management_identity] = replace(
        raw.catalog[plan.management_identity],
        manifest=replace(
            raw.catalog[plan.management_identity].manifest,
            manifest_sha256="f" * 64,
        ),
    )

    with pytest.raises(WeightStoreError, match="digest mismatch"):
        weight_store.load_weight_revision(
            plan.management_identity,
            targets.placement,
            targets.bindings,
        )
    assert raw.released_leases == [1]
