from __future__ import annotations

import hashlib
from dataclasses import replace
from threading import Event
from typing import cast

import pytest

from mooncake.reshard.weight.management import (
    WeightAvailabilityState,
    WeightManagementErrorCode,
    WeightManifestReference,
    WeightMigrationMode,
    WeightOperationKind,
    WeightResidencyState,
    WeightResidencyOperation,
    WeightRevisionIdentity,
    WeightRevisionLease,
    WeightRevisionMetadata,
    WeightRevisionPage,
    WeightRevisionView,
    WeightStoragePolicy,
)
from mooncake.reshard.weight.store import WeightStoreError

from .helpers import (
    InMemoryStore,
    make_weight_store,
    source_manifests,
    target_manifests,
)


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
        self.renewed_leases: list[int] = []
        self.renewed = Event()
        self.operations: dict[int, WeightResidencyOperation] = {}
        self.next_operation_id = 1
        self.cluster_policy = WeightStoragePolicy()
        self.received_policy_overrides: list[WeightStoragePolicy | None] = []

    @staticmethod
    def _identity(args: tuple[object, ...]) -> WeightRevisionIdentity:
        return WeightRevisionIdentity(
            tenant_id=cast(str, args[0]),
            namespace=cast(str, args[1]),
            resource_id=cast(str, args[2]),
            revision=cast(str, args[3]),
            weight_generation=cast(int, args[4]),
        )

    def begin_weight_import(self, *args):
        identity = self._identity(args)
        group_id = f"weight:{_identity_digest(identity)}"
        current = self.catalog.get(identity)
        if current is None:
            policy = (
                WeightStoragePolicy(
                    preferred_residency=WeightResidencyState(args[9]),
                    mixed_hot_ratio=args[10],
                    migration_mode=WeightMigrationMode(args[11]),
                )
                if args[8]
                else self.cluster_policy
            )
            self.received_policy_overrides.append(policy if args[8] else None)
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
                policy=policy,
                operation_id=None,
                affinity_count=args[12],
                affinity_digest=args[13],
                observed_hot_ratio=0.0,
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
            observed_hot_ratio=1.0,
            metadata_generation=2,
            updated_at_ms=2,
        )
        if updated.policy.preferred_residency is not WeightResidencyState.HOT:
            updated, _ = self._start_operation(
                updated,
                updated.policy.preferred_residency,
                preserve_generation=True,
            )
        self.catalog[identity] = updated
        return updated, 0, 0

    def _start_operation(
        self,
        current: WeightRevisionMetadata,
        target: WeightResidencyState,
        *,
        preserve_generation: bool = False,
    ) -> tuple[WeightRevisionMetadata, WeightResidencyOperation]:
        operation_id = self.next_operation_id
        self.next_operation_id += 1
        generation = (
            current.metadata_generation
            if preserve_generation
            else current.metadata_generation + 1
        )
        operation = WeightResidencyOperation(
            operation_id=operation_id,
            identity=current.identity,
            kind=WeightOperationKind.MIGRATING,
            target_residency=target,
            fenced_metadata_generation=generation,
            started_at_ms=10,
            updated_at_ms=10,
            processed_units=0,
            total_units=current.affinity_count,
            processed_bytes=0,
            total_bytes=current.manifest.logical_bytes,
            cursor="",
            message="",
        )
        self.operations[operation_id] = operation
        return replace(
            current,
            operation_id=operation_id,
            metadata_generation=generation,
        ), operation

    def complete_operation(self, operation_id: int) -> WeightRevisionMetadata:
        operation = self.operations[operation_id]
        current = self.catalog[operation.identity]
        ratio = {
            WeightResidencyState.HOT: 1.0,
            WeightResidencyState.COLD: 0.0,
        }.get(operation.target_residency, current.policy.mixed_hot_ratio)
        updated = replace(
            current,
            residency=operation.target_residency,
            operation_id=None,
            observed_hot_ratio=ratio,
            metadata_generation=current.metadata_generation + 1,
            updated_at_ms=current.updated_at_ms + 1,
        )
        self.catalog[current.identity] = updated
        self.operations[operation_id] = replace(
            operation,
            updated_at_ms=operation.updated_at_ms + 1,
            processed_units=operation.total_units,
            processed_bytes=operation.total_bytes,
            message="complete",
        )
        return updated

    def get_weight_revision(self, *args):
        metadata = self.catalog.get(self._identity(args))
        if metadata is None:
            return None, int(WeightManagementErrorCode.NOT_FOUND), 0
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

    def renew_weight_revision_lease(self, tenant_id, lease_id, ttl_ms):
        assert tenant_id == "tenant-a"
        self.renewed_leases.append(lease_id)
        self.renewed.set()
        identity = next(iter(self.catalog))
        return (
            WeightRevisionLease(
                lease_id=lease_id,
                identity=identity,
                holder="worker-0",
                expires_at_ms=1000 + ttl_ms,
                fenced_metadata_generation=self.catalog[identity].metadata_generation,
            ),
            0,
            0,
        )

    def update_weight_policy(self, *args):
        identity = self._identity(args)
        current = self.catalog[identity]
        if current.operation_id is not None:
            return None, int(WeightManagementErrorCode.BUSY), 0
        if args[5] != current.metadata_generation:
            return None, int(WeightManagementErrorCode.STALE_GENERATION), 0
        policy = WeightStoragePolicy(
            preferred_residency=WeightResidencyState(args[6]),
            mixed_hot_ratio=args[7],
            migration_mode=WeightMigrationMode(args[8]),
        )
        updated = replace(
            current,
            policy=policy,
            metadata_generation=current.metadata_generation + 1,
            updated_at_ms=current.updated_at_ms + 1,
        )
        self.catalog[identity] = updated
        return updated, 0, 0

    def start_weight_residency_operation(self, *args):
        identity = self._identity(args)
        current = self.catalog[identity]
        if current.operation_id is not None:
            return None, int(WeightManagementErrorCode.BUSY), 0
        if args[5] != current.metadata_generation:
            return None, int(WeightManagementErrorCode.STALE_GENERATION), 0
        updated, operation = self._start_operation(
            current, WeightResidencyState(args[6])
        )
        self.catalog[identity] = updated
        return operation, 0, 0

    def query_weight_operation(self, tenant_id, operation_id):
        assert tenant_id
        return self.operations[operation_id], 0, 0

    def delete_weight_revision(self, *args):
        identity = self._identity(args)
        current = self.catalog[identity]
        if current.operation_id is not None:
            return None, int(WeightManagementErrorCode.BUSY), 0
        if args[5] != current.metadata_generation:
            return None, int(WeightManagementErrorCode.STALE_GENERATION), 0
        tombstone = replace(
            current,
            availability=WeightAvailabilityState.DELETED,
            residency=WeightResidencyState.ABSENT,
            operation_id=None,
            observed_hot_ratio=0.0,
            metadata_generation=current.metadata_generation + 1,
            updated_at_ms=current.updated_at_ms + 1,
        )
        self.catalog[identity] = tombstone
        return tombstone, 0, 0


def test_managed_revision_publish_resolve_load_and_release_lease() -> None:
    raw = ManagedInMemoryStore()
    _, weight_store = make_weight_store(raw)
    sources = source_manifests(dp=1, tp=2)
    targets = target_manifests(dp=1, tp=2)

    plan = weight_store._weight_put_managed_plan(
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
    view = weight_store.weight_get_metadata(plan.management_identity)

    assert view.metadata.availability is WeightAvailabilityState.READY
    assert view.metadata.manifest.manifest_key == manifest.manifest_key
    assert view.metadata.manifest.manifest_sha256 == manifest.manifest_digest
    revision_keys = {
        manifest.manifest_key,
        *(fragment.object_key for fragment in manifest.fragments),
    }
    assert {raw.group_ids[key] for key in revision_keys} == {manifest.group_id}
    assert weight_store.weight_list(
        tenant_id="tenant-a",
        namespace="production",
        resource_id=manifest.resource_id,
    ).revisions == (view,)

    loaded = weight_store.weight_get(
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
    plan = weight_store._weight_put_managed_plan(
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
        weight_store.weight_get(
            plan.management_identity,
            targets.placement,
            targets.bindings,
        )
    assert raw.released_leases == [1]


def test_managed_load_renews_short_lease_until_transfer_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = ManagedInMemoryStore()
    _, weight_store = make_weight_store(raw)
    sources = source_manifests(dp=1, tp=1)
    targets = target_manifests(dp=1, tp=1)
    plan = weight_store._weight_put_managed_plan(
        sources.placement,
        sources.bindings,
        tenant_id="tenant-a",
    )
    receipts = []
    for binding in sources.bindings:
        receipts.extend(weight_store.upload(plan, sources.placement, binding))
    weight_store.commit_upload(plan, receipts)
    assert plan.management_identity is not None

    def wait_for_renewal(*args, **kwargs) -> None:
        assert raw.renewed.wait(timeout=1)

    monkeypatch.setattr(weight_store, "weight_get_payload", wait_for_renewal)
    weight_store.weight_get(
        plan.management_identity,
        targets.placement,
        targets.bindings,
        holder="worker-0",
        lease_ttl_ms=30,
    )

    assert raw.renewed_leases
    assert raw.released_leases == [1]


def test_weight_management_crud_and_operation_facade() -> None:
    raw = ManagedInMemoryStore()
    _, weight_store = make_weight_store(raw)
    sources = source_manifests(dp=1, tp=1)
    plan = weight_store._weight_put_managed_plan(
        sources.placement,
        sources.bindings,
        tenant_id="tenant-a",
    )
    receipts = weight_store.weight_put_payload(
        plan, sources.placement, sources.binding
    )
    manifest = weight_store.weight_put_commit(plan, receipts)
    assert plan.management_identity is not None
    identity = plan.management_identity

    initial = weight_store.weight_get_metadata(identity).metadata
    assert initial.operation_id is not None
    ready = raw.complete_operation(initial.operation_id)

    assert weight_store.weight_is_exist(identity)
    assert weight_store.weight_get_size(identity) == sum(
        fragment.nbytes for fragment in manifest.fragments
    )
    assert weight_store.weight_list(
        tenant_id="tenant-a",
        namespace=identity.namespace,
        resource_id=identity.resource_id,
    ).revisions

    policy = WeightStoragePolicy(
        preferred_residency=WeightResidencyState.COLD,
        migration_mode=WeightMigrationMode.MANUAL,
    )
    updated = weight_store.weight_update(
        identity,
        policy=policy,
        expected_metadata_generation=ready.metadata_generation,
    )
    assert updated.policy == policy

    operation = weight_store.weight_migrate(
        identity,
        target=WeightResidencyState.MIXED,
        mixed_hot_ratio=0.25,
        expected_metadata_generation=updated.metadata_generation,
    )
    assert weight_store.weight_get_operation(
        operation.operation_id, tenant_id="tenant-a"
    ) == operation

    migrated = raw.complete_operation(operation.operation_id)

    removed = weight_store.weight_remove(
        identity,
        expected_metadata_generation=migrated.metadata_generation,
    )
    assert removed.availability is WeightAvailabilityState.DELETED
    assert not weight_store.weight_is_exist(identity)


def test_weight_put_policy_precedence_is_call_then_instance_then_cluster() -> None:
    sources = source_manifests(dp=1, tp=1)
    instance_policy = WeightStoragePolicy(
        preferred_residency=WeightResidencyState.HOT,
        migration_mode=WeightMigrationMode.PINNED,
    )
    call_policy = WeightStoragePolicy(
        preferred_residency=WeightResidencyState.COLD,
        migration_mode=WeightMigrationMode.MANUAL,
    )

    raw = ManagedInMemoryStore()
    _, store = make_weight_store(raw)
    store.default_policy = instance_policy
    store._weight_put_managed_plan(
        sources.placement,
        sources.bindings,
        policy=call_policy,
    )
    assert raw.received_policy_overrides == [call_policy]

    raw = ManagedInMemoryStore()
    _, store = make_weight_store(raw)
    store.default_policy = instance_policy
    store._weight_put_managed_plan(sources.placement, sources.bindings)
    assert raw.received_policy_overrides == [instance_policy]

    raw = ManagedInMemoryStore()
    _, store = make_weight_store(raw)
    store._weight_put_managed_plan(sources.placement, sources.bindings)
    assert raw.received_policy_overrides == [None]
