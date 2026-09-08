from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace
from threading import Event, Lock, Thread
from typing import Callable, Literal, Optional
from uuid import uuid4

from ..manifest import (
    WeightPlacementManifest,
    WeightRuntimeBindingManifest,
)
from ...contracts import RuntimeFragmentId
from ..storage_manifest import StoredWeightManifest
from ..management import (
    WeightAvailabilityState,
    WeightManifestReference,
    WeightRevisionIdentity,
    WeightRevisionLease,
    WeightRevisionMetadata,
    WeightRevisionPage,
    WeightRevisionView,
)
from ...lifetime import TerminalTransferState
from ..lifetime import (
    WeightAllocationGuardProviders,
    acquire_weight_binding_token,
)
from .contracts import UploadReceipt, WeightLoadPlan, WeightUploadPlan
from .errors import WeightStoreError
from .backend import (
    StoreBackend,
    StoreConfigFactory,
    default_config_factory,
)
from .load import WeightLoadService
from .payload import PayloadStoreOperations
from .registration import StoreBufferRegistration, StoreRegistrationLease
from .transaction import WeightUploadTransaction
from .snapshot import (
    WeightSnapshotAdapter,
    WeightSnapshotDescriptor,
)
from .writer import (
    WeightStoreWriter,
)
from .upload import WeightUploadService


def _require_upload_plan(plan: WeightUploadPlan) -> None:
    if not isinstance(plan, WeightUploadPlan):
        raise WeightStoreError("plan must be a WeightUploadPlan")


def _require_load_plan(plan: WeightLoadPlan) -> None:
    if not isinstance(plan, WeightLoadPlan):
        raise WeightStoreError("plan must be a WeightLoadPlan")


def _payload_keys_sha256(keys: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        encoded = key.encode("utf-8")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def _payload_summary(manifest: StoredWeightManifest) -> tuple[tuple[str, ...], int]:
    object_sizes: dict[str, int] = {}
    for fragment in manifest.fragments:
        object_sizes[fragment.object_key] = max(
            object_sizes.get(fragment.object_key, 0),
            fragment.object_offset + fragment.nbytes,
        )
    return tuple(sorted(object_sizes)), sum(object_sizes.values())


class _RevisionLeaseGuard:
    def __init__(
        self,
        backend: StoreBackend,
        lease: WeightRevisionLease,
        ttl_ms: int,
    ) -> None:
        self._backend = backend
        self._lease = lease
        self._ttl_ms = ttl_ms
        self._stop = Event()
        self._error_lock = Lock()
        self._renewal_error: Optional[Exception] = None
        self._thread = Thread(
            target=self._renew_loop,
            name=f"weight-lease-{lease.lease_id}",
            daemon=True,
        )

    def __enter__(self) -> _RevisionLeaseGuard:
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> Literal[False]:
        self._stop.set()
        self._thread.join()
        renewal_error = self._get_renewal_error()
        release_error: Optional[Exception] = None
        try:
            self._backend.release_weight_revision_lease(
                tenant_id=self._lease.identity.tenant_id,
                lease_id=self._lease.lease_id,
            )
        except Exception as error:
            release_error = error
        if exc_type is None:
            if renewal_error is not None:
                raise renewal_error
            if release_error is not None:
                raise release_error
        return False

    def raise_if_failed(self) -> None:
        error = self._get_renewal_error()
        if error is not None:
            raise error

    def _get_renewal_error(self) -> Optional[Exception]:
        with self._error_lock:
            return self._renewal_error

    def _renew_loop(self) -> None:
        interval_seconds = max(0.001, min(self._ttl_ms / 3000, 30.0))
        while not self._stop.wait(interval_seconds):
            try:
                self._backend.renew_weight_revision_lease(
                    tenant_id=self._lease.identity.tenant_id,
                    lease_id=self._lease.lease_id,
                    ttl_ms=self._ttl_ms,
                )
            except Exception as error:
                with self._error_lock:
                    self._renewal_error = error
                return


class WeightStore:
    def __init__(
        self,
        store: object,
        *,
        key_prefix: str = "weights",
        config_factory: Optional[StoreConfigFactory] = None,
        max_range_bytes: int = 64 * 1024 * 1024,
        max_ranges_per_request: int = 1024,
        max_region_segments: int = 1_000_000,
    ) -> None:
        if (
            max_range_bytes <= 0
            or max_ranges_per_request <= 0
            or max_region_segments <= 0
        ):
            raise ValueError("range limits must be positive")
        self.store = StoreBackend(store)
        self.key_prefix = key_prefix.strip("/")
        self.config_factory = config_factory or default_config_factory
        self.max_range_bytes = max_range_bytes
        self.max_ranges_per_request = max_ranges_per_request
        self.max_region_segments = max_region_segments
        self.registration = StoreBufferRegistration(self.store)
        self._payloads = PayloadStoreOperations(self)
        self._transaction = WeightUploadTransaction(self, self._payloads)
        self._upload = WeightUploadService(self, self._payloads, self._transaction)
        self._load = WeightLoadService(self)

    def plan_upload(
        self,
        source_placement: WeightPlacementManifest,
        source_bindings: Sequence[WeightRuntimeBindingManifest],
        *,
        namespace: str = "default",
    ) -> WeightUploadPlan:
        return self._upload.plan_upload(
            source_placement,
            source_bindings,
            namespace=namespace,
        )

    def plan_managed_upload(
        self,
        source_placement: WeightPlacementManifest,
        source_bindings: Sequence[WeightRuntimeBindingManifest],
        *,
        namespace: str = "default",
        tenant_id: str = "default",
    ) -> WeightUploadPlan:
        """Begin and plan one Store-managed immutable weight revision."""

        if self.key_prefix != "weights":
            raise WeightStoreError(
                "managed weight revisions require the canonical weights key prefix"
            )
        draft = self.plan_upload(
            source_placement,
            source_bindings,
            namespace=namespace,
        )
        identity = WeightRevisionIdentity(
            tenant_id=tenant_id,
            namespace=namespace,
            resource_id=draft.manifest.resource_id,
            revision=draft.manifest.revision,
            weight_generation=draft.manifest.weight_generation,
        )
        payload_keys, logical_bytes = _payload_summary(draft.manifest)
        importing = self.store.begin_weight_import(
            identity,
            payload_group_id="",
            expected_payload_count=len(payload_keys),
            expected_logical_bytes=logical_bytes,
        )
        managed_manifest = replace(
            draft.manifest,
            group_id=importing.manifest.payload_group_id,
        )
        return replace(
            draft,
            manifest=managed_manifest,
            management_identity=importing.identity,
            management_generation=importing.metadata_generation,
        )

    def begin_weight_snapshot(
        self,
        snapshot: WeightSnapshotDescriptor,
        adapter: WeightSnapshotAdapter,
    ) -> WeightStoreWriter:
        """Create a manifest-backed writer for one immutable model snapshot."""

        return WeightStoreWriter(self, snapshot, adapter)

    def begin_managed_weight_snapshot(
        self,
        snapshot: WeightSnapshotDescriptor,
        adapter: WeightSnapshotAdapter,
        *,
        tenant_id: str = "default",
    ) -> WeightStoreWriter:
        """Create a writer whose manifest is published through WeightCatalog."""

        return WeightStoreWriter(
            self,
            snapshot,
            adapter,
            managed=True,
            tenant_id=tenant_id,
        )

    def upload(
        self,
        plan: WeightUploadPlan,
        source_placement: WeightPlacementManifest,
        source_binding: WeightRuntimeBindingManifest,
        *,
        source_worker_id: Optional[str] = None,
        source_allocation_guards: Optional[WeightAllocationGuardProviders] = None,
        registration_lease: Optional[StoreRegistrationLease] = None,
        transfer_id: Optional[str] = None,
    ) -> tuple[UploadReceipt, ...]:
        _require_upload_plan(plan)
        return self._upload.upload(
            plan,
            source_placement,
            source_binding,
            source_worker_id=source_worker_id,
            source_allocation_guards=source_allocation_guards,
            registration_lease=registration_lease,
            transfer_id=transfer_id,
        )

    def abort_upload(
        self,
        plan: WeightUploadPlan,
        receipts: Sequence[UploadReceipt],
    ) -> None:
        _require_upload_plan(plan)
        if plan.management_identity is not None:
            metadata = self.store.abort_weight_import(
                plan.management_identity,
                plan.management_generation or 0,
            )
            self.store.reconcile_weight_revision(metadata.identity)
            return
        self._transaction.abort_upload(plan, receipts)

    def finalize_upload_transaction(self, plan: WeightUploadPlan) -> None:
        _require_upload_plan(plan)
        self._transaction.finalize_upload_transaction(plan)

    def commit_upload(
        self,
        plan: WeightUploadPlan,
        receipts: Sequence[UploadReceipt],
    ) -> StoredWeightManifest:
        _require_upload_plan(plan)
        manifest = self._transaction.commit(plan, receipts)
        self._publish_managed_revision(plan, manifest)
        return manifest

    def _commit_upload_from_writer(
        self,
        plan: WeightUploadPlan,
        receipts: Sequence[UploadReceipt],
        *,
        on_commit_decision_may_exist: Callable[[], None],
    ) -> StoredWeightManifest:
        _require_upload_plan(plan)
        manifest = self._transaction.commit(
            plan,
            receipts,
            on_commit_decision_may_exist=on_commit_decision_may_exist,
        )
        self._publish_managed_revision(plan, manifest)
        return manifest

    def load_manifest(self, manifest_key: str) -> StoredWeightManifest:
        """Load an unmanaged manifest by key without lifecycle guarantees."""

        return self._load.load_manifest(manifest_key)

    def get_weight_revision(
        self, identity: WeightRevisionIdentity
    ) -> WeightRevisionView:
        return self.store.get_weight_revision(identity)

    def list_weight_revisions(
        self,
        *,
        namespace: str,
        resource_id: str,
        tenant_id: str = "default",
        page_token: str = "",
        limit: int = 100,
    ) -> WeightRevisionPage:
        return self.store.list_weight_revisions(
            tenant_id=tenant_id,
            namespace=namespace,
            resource_id=resource_id,
            page_token=page_token,
            limit=limit,
        )

    def load_weight_revision(
        self,
        identity: WeightRevisionIdentity,
        target_placement: WeightPlacementManifest,
        target_bindings: Sequence[WeightRuntimeBindingManifest],
        *,
        holder: Optional[str] = None,
        lease_ttl_ms: int = 24 * 60 * 60 * 1000,
        target_allocation_guards: Optional[WeightAllocationGuardProviders] = None,
    ) -> StoredWeightManifest:
        """Resolve, verify, and transfer an exact revision under one lease."""

        view = self.get_weight_revision(identity)
        lease = self.store.acquire_weight_revision_lease(
            identity,
            expected_metadata_generation=view.metadata.metadata_generation,
            holder=holder or f"reshard-{uuid4().hex}",
            ttl_ms=lease_ttl_ms,
        )
        with _RevisionLeaseGuard(self.store, lease, lease_ttl_ms) as lease_guard:
            manifest = self._load.load_manifest(view.metadata.manifest.manifest_key)
            self._verify_managed_manifest(view.metadata, manifest)
            plan = self.plan_load(manifest, target_placement, target_bindings)
            for binding in target_bindings:
                lease_guard.raise_if_failed()
                if target_allocation_guards is None:
                    self.load(plan, target_placement, binding)
                else:
                    self.load(
                        plan,
                        target_placement,
                        binding,
                        target_allocation_guards=target_allocation_guards,
                    )
                lease_guard.raise_if_failed()
            return manifest

    def plan_load(
        self,
        manifest: StoredWeightManifest,
        target_placement: WeightPlacementManifest,
        target_bindings: Sequence[WeightRuntimeBindingManifest],
    ) -> WeightLoadPlan:
        return self._load.plan_load(manifest, target_placement, target_bindings)

    def load(
        self,
        plan: WeightLoadPlan,
        target_placement: WeightPlacementManifest,
        target_binding: WeightRuntimeBindingManifest,
        *,
        target_worker_id: Optional[str] = None,
        target_allocation_guards: Optional[WeightAllocationGuardProviders] = None,
        registration_lease: Optional[StoreRegistrationLease] = None,
        transfer_id: Optional[str] = None,
    ) -> None:
        _require_load_plan(plan)
        self._load.load(
            plan,
            target_placement,
            target_binding,
            target_worker_id=target_worker_id,
            target_allocation_guards=target_allocation_guards,
            registration_lease=registration_lease,
            transfer_id=transfer_id,
        )

    def register_weight_buffers(
        self,
        binding: WeightRuntimeBindingManifest,
        *,
        fragment_ids: Sequence[RuntimeFragmentId],
        allocation_guards: Optional[WeightAllocationGuardProviders],
        side: str,
        transfer_id: Optional[str] = None,
    ) -> StoreRegistrationLease:
        """Create a typed long-lived Store registration under a framework pin."""

        if not isinstance(binding, WeightRuntimeBindingManifest):
            raise WeightStoreError("binding must be a WeightRuntimeBindingManifest")
        requested_ids = tuple(sorted(set(fragment_ids)))
        if not requested_ids:
            raise WeightStoreError("Store registration lease requires fragments")
        try:
            fresh_binding, tokens = acquire_weight_binding_token(
                transfer_id=transfer_id or uuid4().hex,
                expected_binding=binding,
                required_fragment_ids=requested_ids,
                side=side,
                providers=allocation_guards,
            )
        except ValueError as error:
            raise WeightStoreError(str(error)) from error
        fragments_by_id = {
            fragment.fragment_id: fragment for fragment in fresh_binding.fragments
        }
        try:
            fragments = tuple(
                fragments_by_id[fragment_id] for fragment_id in requested_ids
            )
        except KeyError as error:
            tokens.release_after_terminal(TerminalTransferState.ABORTED)
            raise WeightStoreError(
                f"Store registration fragment is missing: {error.args[0]}"
            ) from error
        return self.registration.acquire_lease(fresh_binding, fragments, tokens)

    def pending_registration_ids(self) -> tuple[str, ...]:
        return self.registration.pending_registration_ids()

    def drain_pending_registration(self, pending_registration_id: str) -> None:
        self.registration.drain_pending_registration(pending_registration_id)

    def _publish_managed_revision(
        self, plan: WeightUploadPlan, manifest: StoredWeightManifest
    ) -> Optional[WeightRevisionMetadata]:
        identity = plan.management_identity
        if identity is None:
            return None
        payload_keys, logical_bytes = _payload_summary(manifest)
        reference = WeightManifestReference(
            manifest_key=manifest.manifest_key,
            manifest_sha256=manifest.manifest_digest,
            payload_group_id=manifest.group_id,
            payload_keys_sha256=_payload_keys_sha256(payload_keys),
            payload_count=len(payload_keys),
            logical_bytes=logical_bytes,
        )
        return self.store.commit_weight_import(
            identity,
            expected_metadata_generation=plan.management_generation or 0,
            manifest=reference,
        )

    @staticmethod
    def _verify_managed_manifest(
        metadata: WeightRevisionMetadata, manifest: StoredWeightManifest
    ) -> None:
        if metadata.availability is not WeightAvailabilityState.READY:
            raise WeightStoreError("weight revision is not READY")
        identity = metadata.identity
        if (
            manifest.namespace != identity.namespace
            or manifest.resource_id != identity.resource_id
            or manifest.revision != identity.revision
            or manifest.weight_generation != identity.weight_generation
            or manifest.group_id != metadata.manifest.payload_group_id
            or manifest.manifest_key != metadata.manifest.manifest_key
        ):
            raise WeightStoreError("managed manifest identity mismatch")
        if manifest.manifest_digest != metadata.manifest.manifest_sha256:
            raise WeightStoreError("managed manifest digest mismatch")


__all__ = ["WeightStore", "WeightStoreError"]
