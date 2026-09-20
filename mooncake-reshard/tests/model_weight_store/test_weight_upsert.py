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
