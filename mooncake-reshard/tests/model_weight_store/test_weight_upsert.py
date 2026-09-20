from __future__ import annotations

from dataclasses import replace

import mooncake.reshard.weight as weight
import pytest

from mooncake.reshard.weight import (
    WeightAvailabilityState,
    WeightMigrationMode,
    WeightResidencyState,
    WeightRevisionIdentity,
    WeightSnapshotDescriptor,
    WeightStoragePolicy,
    WeightStoreError,
)

from mooncake.reshard.weight.manifest import ParallelRank

from .helpers import make_weight_store, source_manifests, with_empty_participant
from .test_managed_revision import ManagedInMemoryStore, _identity_digest
from .test_snapshot_writer import _SnapshotAdapter


def test_weight_upsert_mode_wire_values_are_stable() -> None:
    assert hasattr(weight, "WeightUpsertMode")
    mode = weight.WeightUpsertMode
    assert int(mode.PUT_FIRST) == 0
    assert int(mode.DELETE_FIRST) == 1


def test_upsert_rejects_noncanonical_prefix_before_claim() -> None:
    raw = UpsertInMemoryStore()
    _, store = make_weight_store(raw)
    base = source_manifests(dp=1, tp=1, weight_generation=1)
    target = source_manifests(dp=1, tp=1, weight_generation=2)
    base_identity, _ = _commit_source(store, base)
    raw.upsert_events.clear()
    store.key_prefix = "custom"

    with pytest.raises(WeightStoreError, match="canonical weights key prefix"):
        store.weight_upsert(
            _descriptor(target),
            _SnapshotAdapter(target),
            replacing=base_identity,
            expected_metadata_generation=2,
            tenant_id="tenant-a",
        )

    assert raw.upsert_events == []


class UpsertInMemoryStore(ManagedInMemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.upsert_events: list[str] = []
        self.upsert_requests: dict[str, tuple[WeightRevisionIdentity, ...]] = {}
        self.upsert_modes: dict[str, weight.WeightUpsertMode] = {}
        self.released_upserts: set[str] = set()
        self.committed_upserts: set[str] = set()
        self.fail_begin_response_once = False
        self.fail_commit_response_once = False
        self.fail_abort_upsert_response_once = False
        self.reject_delete_first = False

    def batch_put_from(self, keys, addresses, sizes, config):
        # Native Put returns success for an existing immutable object.
        results = []
        for index, key in enumerate(keys):
            if key in self.objects:
                results.append(0)
                continue
            results.extend(
                super().batch_put_from(
                    [key],
                    [addresses[index]],
                    [sizes[index]],
                    replace(
                        config,
                        group_ids=[config.group_ids[index]],
                        residency_affinity_ids=(
                            [config.residency_affinity_ids[index]]
                            if config.residency_affinity_ids is not None
                            else None
                        ),
                    ),
                )
            )
        return results

    def begin_weight_upsert(self, *args):
        request_id = args[0]
        mode = weight.WeightUpsertMode(args[1])
        base = self._identity(args[2:7])
        target = self._identity(args[8:13])
        if self.reject_delete_first and mode is weight.WeightUpsertMode.DELETE_FIRST:
            return None, int(weight.WeightManagementErrorCode.BUSY), 0
        existing = self.upsert_requests.get(request_id)
        if existing is not None:
            assert existing == (base, target)
            assert self.upsert_modes[request_id] is mode
            self.upsert_events.append("begin")
            current = self.catalog[target]
            if (
                mode is weight.WeightUpsertMode.DELETE_FIRST
                and current.availability is WeightAvailabilityState.DELETED
            ):
                current = replace(
                    current,
                    availability=WeightAvailabilityState.IMPORTING,
                    residency=WeightResidencyState.UNKNOWN,
                    operation_id=None,
                    observed_hot_ratio=0.0,
                    metadata_generation=current.metadata_generation + 1,
                    updated_at_ms=current.updated_at_ms + 1,
                )
                self.catalog[target] = current
                self.upsert_events.append("target_restarted")
            return current, 0, 0
        if any(
            identities == (base, target)
            and active_request not in self.released_upserts
            and active_request not in self.committed_upserts
            for active_request, identities in self.upsert_requests.items()
        ):
            return None, int(weight.WeightManagementErrorCode.BUSY), 0
        self.upsert_events.append("begin")
        assert self.catalog[base].metadata_generation == args[7]
        if mode is weight.WeightUpsertMode.DELETE_FIRST:
            self.catalog[base] = replace(
                self.catalog[base],
                availability=WeightAvailabilityState.DELETED,
                residency=WeightResidencyState.ABSENT,
                operation_id=None,
                observed_hot_ratio=0.0,
                metadata_generation=self.catalog[base].metadata_generation + 1,
            )
            self.upsert_events.append("base_deleted")
        policy = (
            WeightStoragePolicy(
                preferred_residency=WeightResidencyState(args[17]),
                mixed_hot_ratio=args[18],
                migration_mode=WeightMigrationMode(args[19]),
            )
            if args[16]
            else self.cluster_policy
        )
        metadata = self._importing_metadata(
            target,
            payload_group_id=f"weight:{_identity_digest(target)}",
            payload_count=args[14],
            logical_bytes=args[15],
            policy=policy,
            affinity_count=args[20],
            affinity_digest=args[21],
        )
        self.catalog[target] = metadata
        self.upsert_requests[request_id] = (base, target)
        self.upsert_modes[request_id] = mode
        if self.fail_begin_response_once:
            self.fail_begin_response_once = False
            raise RuntimeError("begin response lost")
        return metadata, 0, 0

    def commit_weight_import(self, *args):
        self.upsert_events.append("target_ready")
        members = [key for key, group in self.group_ids.items() if group == args[8]]
        if len(members) != args[10] + 1:
            return None, int(weight.WeightManagementErrorCode.CONFLICT), 0
        return super().commit_weight_import(*args)

    def commit_weight_upsert(self, *args):
        request_id = args[0]
        base = self._identity(args[1:6])
        target = self._identity(args[6:11])
        assert self.upsert_requests[request_id] == (base, target)
        assert self.catalog[target].availability is WeightAvailabilityState.READY
        if request_id not in self.committed_upserts:
            self.catalog[base] = replace(
                self.catalog[base],
                availability=WeightAvailabilityState.DELETED,
                residency=WeightResidencyState.ABSENT,
                operation_id=None,
                observed_hot_ratio=0.0,
                metadata_generation=self.catalog[base].metadata_generation + 1,
            )
            self.committed_upserts.add(request_id)
        self.upsert_events.append("retirement_durable")
        if self.fail_commit_response_once:
            self.fail_commit_response_once = False
            raise RuntimeError("response lost")
        return self.catalog[target], 0, 0

    def abort_weight_upsert(self, *args):
        request_id = args[0]
        base = self._identity(args[1:6])
        target = self._identity(args[6:11])
        assert self.upsert_requests[request_id] == (base, target)
        if self.fail_abort_upsert_response_once:
            self.fail_abort_upsert_response_once = False
            raise RuntimeError("abort request not delivered")
        if self.upsert_modes[request_id] is weight.WeightUpsertMode.PUT_FIRST:
            self.released_upserts.add(request_id)
        self.upsert_events.append("claim_aborted")
        return self.catalog[base], 0, 0

    def reconcile_weight_revision(self, *args):
        identity = self._identity(args)
        payload_group_id = self.catalog[identity].manifest.payload_group_id
        result = super().reconcile_weight_revision(*args)
        if self.catalog[identity].availability is WeightAvailabilityState.DELETED:
            for key, group_id in tuple(self.group_ids.items()):
                if group_id == payload_group_id:
                    self.remove(key, force=True)
        return result

    def _importing_metadata(
        self,
        identity,
        *,
        payload_group_id,
        payload_count,
        logical_bytes,
        policy,
        affinity_count,
        affinity_digest,
    ):
        source = next(iter(self.catalog.values()))
        return replace(
            source,
            identity=identity,
            manifest=replace(
                source.manifest,
                manifest_key="",
                manifest_sha256="",
                payload_group_id=payload_group_id,
                payload_keys_sha256="",
                payload_count=payload_count,
                logical_bytes=logical_bytes,
            ),
            availability=WeightAvailabilityState.IMPORTING,
            residency=WeightResidencyState.UNKNOWN,
            policy=policy,
            operation_id=None,
            affinity_count=affinity_count,
            affinity_digest=affinity_digest,
            observed_hot_ratio=0.0,
            metadata_generation=1,
        )


def _descriptor(source) -> WeightSnapshotDescriptor:
    return WeightSnapshotDescriptor(
        resource_id=source.placement.resource_id,
        revision=source.placement.revision,
        weight_generation=source.placement.weight_generation,
        namespace="production",
    )


def _commit_source(weight_store, source):
    writer = weight_store.weight_put(
        _descriptor(source),
        _SnapshotAdapter(source),
        tenant_id="tenant-a",
        policy=WeightStoragePolicy(
            preferred_residency=WeightResidencyState.HOT,
            migration_mode=WeightMigrationMode.MANUAL,
        ),
    )
    for binding in source.bindings:
        fragment = binding.fragments[0]
        writer.weight_put_tensor(
            source.placement.tensors[0].tensor_id,
            fragment.placement_fragment_id,
        )
    return writer.identity, writer.commit()


def _commit_writer(writer, source):
    for binding in source.bindings:
        fragment = binding.fragments[0]
        writer.weight_put_tensor(
            source.placement.tensors[0].tensor_id,
            fragment.placement_fragment_id,
        )
    return writer.commit()


def test_put_first_publishes_target_before_retiring_base() -> None:
    raw = UpsertInMemoryStore()
    _, store = make_weight_store(raw)
    base = source_manifests(dp=1, tp=1, weight_generation=1)
    target = source_manifests(dp=1, tp=1, weight_generation=2)
    base_identity, _ = _commit_source(store, base)
    raw.upsert_events.clear()

    writer = store.weight_upsert(
        _descriptor(target),
        _SnapshotAdapter(target),
        replacing=base_identity,
        expected_metadata_generation=2,
        tenant_id="tenant-a",
        request_id="replace-1-with-2",
    )
    assert writer.request_id == "replace-1-with-2"
    assert raw.catalog[base_identity].availability is WeightAvailabilityState.READY

    _commit_writer(writer, target)

    assert raw.upsert_events == [
        "begin",
        "target_ready",
        "retirement_durable",
    ]
    assert raw.catalog[base_identity].availability is WeightAvailabilityState.DELETED
    assert raw.catalog[writer.identity].availability is WeightAvailabilityState.READY


def test_delete_first_reclaims_base_before_returning_writer() -> None:
    raw = UpsertInMemoryStore()
    _, store = make_weight_store(raw)
    base = source_manifests(dp=1, tp=1, weight_generation=1)
    target = source_manifests(dp=1, tp=1, weight_generation=2)
    base_identity, _ = _commit_source(store, base)
    raw.upsert_events.clear()

    writer = store.weight_upsert(
        _descriptor(target),
        _SnapshotAdapter(target),
        replacing=base_identity,
        expected_metadata_generation=2,
        mode=weight.WeightUpsertMode.DELETE_FIRST,
        tenant_id="tenant-a",
        request_id="delete-then-put",
    )

    assert raw.upsert_events == ["begin", "base_deleted"]
    assert raw.catalog[base_identity].availability is WeightAvailabilityState.DELETED
    _commit_writer(writer, target)


def test_delete_first_busy_fails_before_target_import() -> None:
    raw = UpsertInMemoryStore()
    _, store = make_weight_store(raw)
    base = source_manifests(dp=1, tp=1, weight_generation=1)
    target = source_manifests(dp=1, tp=1, weight_generation=2)
    base_identity, _ = _commit_source(store, base)
    raw.upsert_events.clear()
    raw.reject_delete_first = True

    with pytest.raises(weight.WeightManagementError) as error:
        store.weight_upsert(
            _descriptor(target),
            _SnapshotAdapter(target),
            replacing=base_identity,
            expected_metadata_generation=2,
            mode=weight.WeightUpsertMode.DELETE_FIRST,
            tenant_id="tenant-a",
            request_id="blocked-delete-first",
        )

    assert error.value.code is weight.WeightManagementErrorCode.BUSY
    assert raw.catalog[base_identity].availability is WeightAvailabilityState.READY
    assert all(identity.weight_generation != 2 for identity in raw.catalog)
    assert raw.upsert_events == []


@pytest.mark.parametrize("invalid_mode", [True, 0, 1])
def test_upsert_rejects_non_enum_mode_before_mutation(invalid_mode) -> None:
    raw = UpsertInMemoryStore()
    _, store = make_weight_store(raw)
    base = source_manifests(dp=1, tp=1, weight_generation=1)
    target = source_manifests(dp=1, tp=1, weight_generation=2)
    base_identity, _ = _commit_source(store, base)
    raw.upsert_events.clear()

    with pytest.raises(WeightStoreError, match="WeightUpsertMode"):
        store.weight_upsert(
            _descriptor(target),
            _SnapshotAdapter(target),
            replacing=base_identity,
            expected_metadata_generation=2,
            mode=invalid_mode,
            tenant_id="tenant-a",
        )

    assert raw.catalog[base_identity].availability is WeightAvailabilityState.READY
    assert raw.upsert_events == []


def test_upsert_abort_keeps_put_first_base_and_releases_claim() -> None:
    raw = UpsertInMemoryStore()
    _, store = make_weight_store(raw)
    base = source_manifests(dp=1, tp=1, weight_generation=1)
    target = source_manifests(dp=1, tp=1, weight_generation=2)
    base_identity, _ = _commit_source(store, base)
    raw.upsert_events.clear()

    with pytest.raises(RuntimeError, match="cancel replacement"):
        with store.weight_upsert(
            _descriptor(target),
            _SnapshotAdapter(target),
            replacing=base_identity,
            expected_metadata_generation=2,
            tenant_id="tenant-a",
            request_id="abort-replacement",
        ):
            raise RuntimeError("cancel replacement")

    assert raw.catalog[base_identity].availability is WeightAvailabilityState.READY
    assert raw.upsert_events[-1] == "claim_aborted"


def test_put_first_abort_retries_after_claim_abort_was_not_delivered() -> None:
    raw = UpsertInMemoryStore()
    _, store = make_weight_store(raw)
    base = source_manifests(dp=1, tp=1, weight_generation=1)
    target = source_manifests(dp=1, tp=1, weight_generation=2)
    base_identity, _ = _commit_source(store, base)
    writer = store.weight_upsert(
        _descriptor(target),
        _SnapshotAdapter(target),
        replacing=base_identity,
        expected_metadata_generation=2,
        tenant_id="tenant-a",
        request_id="retry-put-first-abort",
    )
    raw.fail_abort_upsert_response_once = True

    with pytest.raises(WeightStoreError, match="abort request not delivered"):
        writer.abort()

    assert raw.catalog[writer.identity].availability is WeightAvailabilityState.DELETED
    assert writer.request_id not in raw.released_upserts
    writer.abort()

    assert writer.request_id in raw.released_upserts
    assert raw.catalog[base_identity].availability is WeightAvailabilityState.READY


def test_weight_put_abort_still_deletes_importing_revision() -> None:
    raw = UpsertInMemoryStore()
    _, store = make_weight_store(raw)
    source = source_manifests(dp=1, tp=1, weight_generation=1)
    writer = store.weight_put(
        _descriptor(source),
        _SnapshotAdapter(source),
        tenant_id="tenant-a",
    )

    writer.abort()

    assert raw.catalog[writer.identity].availability is WeightAvailabilityState.DELETED


def test_put_first_payload_failure_keeps_base_ready() -> None:
    raw = UpsertInMemoryStore()
    _, store = make_weight_store(raw)
    base = source_manifests(dp=1, tp=1, weight_generation=1)
    target = source_manifests(dp=1, tp=1, weight_generation=2)
    base_identity, _ = _commit_source(store, base)
    writer = store.weight_upsert(
        _descriptor(target),
        _SnapshotAdapter(target),
        replacing=base_identity,
        expected_metadata_generation=2,
        tenant_id="tenant-a",
        request_id="failed-payload",
    )
    raw.fail_key = writer._plan.operations[0].target.object_key
    fragment = target.bindings[0].fragments[0]

    with pytest.raises(WeightStoreError, match="batch_put_from failed"):
        writer.weight_put_tensor(
            target.placement.tensors[0].tensor_id,
            fragment.placement_fragment_id,
        )

    assert raw.catalog[base_identity].availability is WeightAvailabilityState.READY
    assert raw.upsert_events[-1] == "claim_aborted"


def test_delete_first_payload_failure_recovers_with_same_request() -> None:
    raw = UpsertInMemoryStore()
    _, store = make_weight_store(raw)
    base = source_manifests(dp=1, tp=2, weight_generation=1)
    target = source_manifests(dp=1, tp=2, weight_generation=2)
    base_identity, _ = _commit_source(store, base)
    request_id = "recover-delete-first"
    writer = store.weight_upsert(
        _descriptor(target),
        _SnapshotAdapter(target),
        replacing=base_identity,
        expected_metadata_generation=2,
        mode=weight.WeightUpsertMode.DELETE_FIRST,
        tenant_id="tenant-a",
        request_id=request_id,
    )
    first_fragment = target.bindings[0].fragments[0]
    writer.weight_put_tensor(
        target.placement.tensors[0].tensor_id,
        first_fragment.placement_fragment_id,
    )
    first_payload_key = writer._plan.operations[0].target.object_key
    assert first_payload_key in raw.objects
    raw.fail_key = writer._plan.operations[1].target.object_key

    with pytest.raises(WeightStoreError, match="batch_put_from failed"):
        writer.weight_put_tensor(
            target.placement.tensors[0].tensor_id,
            target.bindings[1].fragments[0].placement_fragment_id,
        )

    assert first_payload_key not in raw.objects
    assert raw.catalog[writer.identity].availability is WeightAvailabilityState.DELETED
    with pytest.raises(weight.WeightManagementError) as error:
        store.weight_upsert(
            _descriptor(target),
            _SnapshotAdapter(target),
            replacing=base_identity,
            expected_metadata_generation=2,
            mode=weight.WeightUpsertMode.DELETE_FIRST,
            tenant_id="tenant-a",
            request_id="competing-delete-first",
        )
    assert error.value.code is weight.WeightManagementErrorCode.BUSY

    raw.fail_key = None
    retry = store.weight_upsert(
        _descriptor(target),
        _SnapshotAdapter(target),
        replacing=base_identity,
        expected_metadata_generation=2,
        mode=weight.WeightUpsertMode.DELETE_FIRST,
        tenant_id="tenant-a",
        request_id=request_id,
    )
    assert retry.request_id == request_id
    assert raw.catalog[retry.identity].availability is WeightAvailabilityState.IMPORTING

    _commit_writer(retry, target)

    assert raw.catalog[retry.identity].availability is WeightAvailabilityState.READY


def test_delete_first_begin_response_loss_retries_with_derived_request_id() -> None:
    raw = UpsertInMemoryStore()
    _, store = make_weight_store(raw)
    base = source_manifests(dp=1, tp=1, weight_generation=1)
    target = source_manifests(dp=1, tp=1, weight_generation=2)
    base_identity, _ = _commit_source(store, base)
    raw.fail_begin_response_once = True

    with pytest.raises(WeightStoreError, match="begin response lost"):
        store.weight_upsert(
            _descriptor(target),
            _SnapshotAdapter(target),
            replacing=base_identity,
            expected_metadata_generation=2,
            mode=weight.WeightUpsertMode.DELETE_FIRST,
            tenant_id="tenant-a",
        )

    persisted_request_id = next(iter(raw.upsert_requests))
    assert raw.catalog[base_identity].availability is WeightAvailabilityState.DELETED
    retry = store.weight_upsert(
        _descriptor(target),
        _SnapshotAdapter(target),
        replacing=base_identity,
        expected_metadata_generation=2,
        mode=weight.WeightUpsertMode.DELETE_FIRST,
        tenant_id="tenant-a",
    )

    assert retry.request_id == persisted_request_id
    _commit_writer(retry, target)
    assert raw.catalog[retry.identity].availability is WeightAvailabilityState.READY


def test_upsert_requires_same_lineage_and_increasing_generation() -> None:
    raw = UpsertInMemoryStore()
    _, store = make_weight_store(raw)
    base = source_manifests(dp=1, tp=1, weight_generation=1)
    base_identity, _ = _commit_source(store, base)
    raw.upsert_events.clear()

    with pytest.raises(WeightStoreError, match="generation must increase"):
        store.weight_upsert(
            _descriptor(base),
            _SnapshotAdapter(base),
            replacing=base_identity,
            expected_metadata_generation=2,
            tenant_id="tenant-a",
        )

    target = source_manifests(dp=1, tp=1, weight_generation=2)
    wrong_tenant = replace(base_identity, tenant_id="tenant-b")
    with pytest.raises(WeightStoreError, match="same lineage"):
        store.weight_upsert(
            _descriptor(target),
            _SnapshotAdapter(target),
            replacing=wrong_tenant,
            expected_metadata_generation=2,
            tenant_id="tenant-a",
        )
    assert raw.upsert_events == []


def test_upsert_derives_stable_request_id_from_immutable_arguments() -> None:
    def derive(
        *,
        mode=weight.WeightUpsertMode.PUT_FIRST,
        target_generation=2,
        expected_metadata_generation=2,
    ) -> str:
        raw = UpsertInMemoryStore()
        _, store = make_weight_store(raw)
        base = source_manifests(dp=1, tp=1, weight_generation=1)
        target = source_manifests(
            dp=1,
            tp=1,
            weight_generation=target_generation,
        )
        base_identity, _ = _commit_source(store, base)
        if expected_metadata_generation != 2:
            raw.catalog[base_identity] = replace(
                raw.catalog[base_identity],
                metadata_generation=expected_metadata_generation,
            )
        return store.weight_upsert(
            _descriptor(target),
            _SnapshotAdapter(target),
            replacing=base_identity,
            expected_metadata_generation=expected_metadata_generation,
            mode=mode,
            tenant_id="tenant-a",
        ).request_id

    first = derive()
    assert first == derive()
    assert len(first) == 64
    int(first, 16)
    assert first != derive(mode=weight.WeightUpsertMode.DELETE_FIRST)
    assert first != derive(target_generation=3)
    assert first != derive(expected_metadata_generation=3)


def test_upsert_rejects_explicit_empty_request_id() -> None:
    raw = UpsertInMemoryStore()
    _, store = make_weight_store(raw)
    base = source_manifests(dp=1, tp=1, weight_generation=1)
    target = source_manifests(dp=1, tp=1, weight_generation=2)
    base_identity, _ = _commit_source(store, base)

    with pytest.raises(WeightStoreError, match="request_id"):
        store.weight_upsert(
            _descriptor(target),
            _SnapshotAdapter(target),
            replacing=base_identity,
            expected_metadata_generation=2,
            tenant_id="tenant-a",
            request_id="",
        )


def test_put_first_commit_retry_reuses_request_after_uncertain_response() -> None:
    raw = UpsertInMemoryStore()
    _, store = make_weight_store(raw)
    base = source_manifests(dp=1, tp=1, weight_generation=1)
    target = source_manifests(dp=1, tp=1, weight_generation=2)
    base_identity, _ = _commit_source(store, base)
    writer = store.weight_upsert(
        _descriptor(target),
        _SnapshotAdapter(target),
        replacing=base_identity,
        expected_metadata_generation=2,
        tenant_id="tenant-a",
        request_id="stable-retry-id",
    )
    for binding in target.bindings:
        fragment = binding.fragments[0]
        writer.weight_put_tensor(
            target.placement.tensors[0].tensor_id,
            fragment.placement_fragment_id,
        )
    raw.fail_commit_response_once = True

    with pytest.raises(WeightStoreError, match="response lost"):
        writer.commit()

    assert writer.request_id == "stable-retry-id"
    assert raw.catalog[base_identity].availability is WeightAvailabilityState.DELETED
    assert writer.commit() == writer._plan.manifest


@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize("changed_source", [False, True])
def test_upsert_reconstructed_writer_resumes_upload(
    published, changed_source, monkeypatch
) -> None:
    raw = UpsertInMemoryStore()
    _, store = make_weight_store(raw)
    base = source_manifests(dp=1, tp=2, weight_generation=1)
    target = source_manifests(dp=1, tp=2, weight_generation=2)
    base_identity, _ = _commit_source(store, base)
    args = dict(
        replacing=base_identity,
        expected_metadata_generation=2,
        tenant_id="tenant-a",
        request_id="reconstructed-writer",
    )
    writer = store.weight_upsert(_descriptor(target), _SnapshotAdapter(target), **args)
    assert writer._plan.manifest.created_at == "1970-01-01T00:00:00.001Z"
    writer.weight_put_tensor(
        target.placement.tensors[0].tensor_id,
        target.bindings[0].fragments[0].placement_fragment_id,
    )
    if published:
        writer.weight_put_tensor(
            target.placement.tensors[0].tensor_id,
            target.bindings[1].fragments[0].placement_fragment_id,
        )
        original_commit = raw.commit_weight_import

        def cut_commit(*args):
            raise RuntimeError("interrupted after manifest publication")

        monkeypatch.setattr(raw, "commit_weight_import", cut_commit)
        with pytest.raises(WeightStoreError, match="interrupted"):
            writer.commit()
        monkeypatch.setattr(raw, "commit_weight_import", original_commit)

    _, rebuilt_store = make_weight_store(raw)
    target = source_manifests(dp=1, tp=2, weight_generation=2)
    if changed_source:
        target = with_empty_participant(
            target, participant_id="extra-stage", rank=ParallelRank(pp=1)
        )
    retry = rebuilt_store.weight_upsert(
        _descriptor(target), _SnapshotAdapter(target), **args
    )
    if changed_source:
        with pytest.raises(
            (WeightStoreError, weight.WeightManagementError), match="(?i)conflict"
        ):
            _commit_writer(retry, target)
        assert (
            raw.catalog[retry.identity].availability
            is WeightAvailabilityState.IMPORTING
        )
        return
    assert retry._plan.manifest == writer._plan.manifest
    assert retry._plan.control_key == writer._plan.control_key
    _commit_writer(retry, target)
    assert raw.catalog[retry.identity].availability is WeightAvailabilityState.READY
