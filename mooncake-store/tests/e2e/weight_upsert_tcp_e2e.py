#!/usr/bin/env python3
"""Process-level TCP e2e for generation-aware weight upsert."""

from __future__ import annotations

import hashlib
import os
import sys
import time


def _require_store():
    try:
        import store  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(f"import_fail {exc}", flush=True)
        raise SystemExit(10)
    return store


def _payload_keys_sha256(keys: list[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        encoded = key.encode("utf-8")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def _require_success(result, operation: str):
    value, management_error, transport_error = result
    if value is None or management_error != 0 or transport_error != 0:
        raise AssertionError(
            f"{operation} failed: management_error={management_error}, "
            f"transport_error={transport_error}"
        )
    return value


def _run_replacements(client, store, identity, payload_key, payload, suffix):
    def view(generation):
        return _require_success(
            client.get_weight_revision(*identity[:4], generation),
            "replacement view",
        )

    def begin(mode, base, target, request_id, expected_generation=None):
        if expected_generation is None:
            expected_generation = view(base).metadata.metadata_generation
        return client.begin_weight_upsert(
            request_id,
            mode,
            *identity[:4],
            base,
            expected_generation,
            *identity[:4],
            target,
            "",
            1,
            len(payload),
            True,
            1,
            0.5,
            1,
            1,
            hashlib.sha256(b"rank-0").hexdigest(),
        )

    def publish(importing, generation):
        key = f"weights/{identity[2]}/{generation}/payload"
        manifest_key = (
            f"weights/{identity[1]}/{identity[2]}/{identity[3]}/{generation}/manifest"
        )
        config = store.ReplicateConfig()
        config.replica_num = 1
        config.with_hard_pin = True
        config.group_ids = [importing.manifest.payload_group_id]
        config.residency_affinity_ids = ["rank-0"]
        config.data_type = store.ObjectDataType.WEIGHT
        assert client.put(key, payload, config) == 0
        config.residency_affinity_ids = [""]
        config.data_type = store.ObjectDataType.METADATA
        assert client.put(manifest_key, b"{}", config) == 0
        _require_success(
            client.commit_weight_import(
                *identity[:4],
                generation,
                importing.metadata_generation,
                manifest_key,
                hashlib.sha256(b"{}").hexdigest(),
                importing.manifest.payload_group_id,
                _payload_keys_sha256([key]),
                1,
                len(payload),
            ),
            "replacement publish",
        )
        assert client.get(key) == payload
        return key

    base = identity[4]
    current = view(base).metadata
    lease = _require_success(
        client.acquire_weight_revision_lease(
            *identity,
            current.metadata_generation,
            "replacement-reader",
            30000,
        ),
        "replacement lease",
    )
    request_id = f"put-first-{suffix}"
    target = base + 2
    importing = _require_success(begin(0, base, target, request_id), "put-first begin")
    target_key = publish(importing, target)
    claim = _require_success(
        client.commit_weight_upsert(request_id, *identity, *identity[:4], target),
        "put-first commit",
    )
    assert claim.latest_claim.phase == store.WeightUpsertPhase.RETIRING_BASE
    assert client.get(payload_key) == payload
    released = client.release_weight_revision_lease(identity[0], lease.lease_id)
    assert released[1:] == (0, 0), released
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        claim = _require_success(
            client.commit_weight_upsert(request_id, *identity, *identity[:4], target),
            "put-first retire",
        )
        if claim.latest_claim.phase == store.WeightUpsertPhase.COMPLETED:
            break
        time.sleep(0.2)
    else:
        raise AssertionError("put-first did not retire the old generation")
    assert view(base).metadata.availability == store.WeightAvailabilityState.DELETED
    assert client.is_exist(payload_key) == 0
    assert client.get(target_key) == payload

    request_id = f"delete-first-{suffix}"
    next_generation = target + 1
    expected_generation = view(target).metadata.metadata_generation
    deadline = time.monotonic() + 30
    while True:
        result = begin(1, target, next_generation, request_id, expected_generation)
        if result[1] != int(store.WeightManagementError.BUSY):
            importing = _require_success(result, "delete-first begin")
            break
        assert time.monotonic() < deadline, "delete-first did not reclaim base"
        time.sleep(0.2)
    assert view(target).metadata.availability == store.WeightAvailabilityState.DELETED
    assert client.is_exist(target_key) == 0
    new_key = publish(importing, next_generation)
    claim = _require_success(
        client.commit_weight_upsert(
            request_id,
            *identity[:4],
            target,
            *identity[:4],
            next_generation,
        ),
        "delete-first commit",
    )
    assert claim.latest_claim.phase == store.WeightUpsertPhase.COMPLETED
    assert claim.committed_weight_generation == next_generation
    assert client.get(new_key) == payload
    print("put-first and delete-first replacement PASSED", flush=True)


def _run_lifecycle(client, store, suffix: str) -> None:
    identity = ("default", "e2e", f"lifecycle-{suffix}", "main", 1)
    prefix = "/".join(str(part) for part in identity[1:])
    manifest_key = f"weights/{prefix}/manifest"
    payloads = {
        f"weights/{prefix}/tensor-{index}": bytes([index + 1]) * 65536
        for index in range(2)
    }
    importing = _require_success(
        client.begin_weight_import(
            *identity,
            "",
            2,
            131072,
            True,
            1,
            0.5,
            1,
            2,
            hashlib.sha256(b"rank-0,rank-1").hexdigest(),
        ),
        "lifecycle begin",
    )
    group_id = importing.manifest.payload_group_id
    for index, (key, payload) in enumerate(payloads.items()):
        config = store.ReplicateConfig()
        config.replica_num = 1
        config.with_hard_pin = True
        config.group_ids = [group_id]
        config.residency_affinity_ids = [f"rank-{index}"]
        config.data_type = store.ObjectDataType.WEIGHT
        assert client.put(key, payload, config) == 0
    config = store.ReplicateConfig()
    config.replica_num = 1
    config.with_hard_pin = True
    config.group_ids = [group_id]
    config.data_type = store.ObjectDataType.METADATA
    assert client.put(manifest_key, b"{}", config) == 0
    ready = _require_success(
        client.commit_weight_import(
            *identity,
            importing.metadata_generation,
            manifest_key,
            hashlib.sha256(b"{}").hexdigest(),
            group_id,
            _payload_keys_sha256(list(payloads)),
            2,
            131072,
        ),
        "lifecycle commit",
    )
    assert ready.availability == store.WeightAvailabilityState.READY
    for key, payload in payloads.items():
        assert client.get(key) == payload
    initial_replicas = client.batch_get_replica_desc(list(payloads))
    for key in payloads:
        assert initial_replicas[key]
        assert all(replica.is_memory_replica() for replica in initial_replicas[key])

    lease = _require_success(
        client.acquire_weight_revision_lease(
            *identity,
            ready.metadata_generation,
            "e2e-reader",
            30000,
        ),
        "acquire lease",
    )
    current = _require_success(client.get_weight_revision(*identity), "get leased")
    blocked = client.delete_weight_revision(
        *identity,
        current.metadata.metadata_generation,
    )
    assert blocked[0] is None and blocked[1] == int(store.WeightManagementError.BUSY)
    assert blocked[2] == 0
    released = client.release_weight_revision_lease("default", lease.lease_id)
    assert released[1:] == (0, 0), released

    def wait_for(residency):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            result = _require_success(
                client.reconcile_weight_revision(*identity),
                "reconcile residency",
            )
            if result.residency == residency and result.operation_id is None:
                return result
            time.sleep(0.2)
        raise AssertionError(f"residency did not converge to {residency}")

    for target, hot_count in (
        (store.WeightResidencyState.COLD, 0),
        (store.WeightResidencyState.HOT, 2),
        (store.WeightResidencyState.MIXED, 1),
    ):
        current = _require_success(client.get_weight_revision(*identity), "get policy")
        updated = _require_success(
            client.update_weight_policy(
                *identity,
                current.metadata.metadata_generation,
                int(target),
                0.5,
                int(store.WeightMigrationMode.MANUAL),
            ),
            "update policy",
        )
        assert updated.policy.preferred_residency == target
        operation = _require_success(
            client.start_weight_residency_operation(
                *identity,
                updated.metadata_generation,
                int(target),
                0.5 if target == store.WeightResidencyState.MIXED else None,
            ),
            "start migration",
        )
        assert operation.operation_id != 0
        observed = wait_for(target)
        assert observed.availability == store.WeightAvailabilityState.READY
        replicas = client.batch_get_replica_desc(list(payloads) + [manifest_key])
        actual_hot = sum(
            any(replica.is_memory_replica() for replica in replicas[key])
            for key in payloads
        )
        assert actual_hot == hot_count, (target, actual_hot)
        assert any(replica.is_memory_replica() for replica in replicas[manifest_key])
        for key in payloads:
            if not any(replica.is_memory_replica() for replica in replicas[key]):
                assert any(replica.is_local_disk_replica() for replica in replicas[key])
        for key, payload in payloads.items():
            assert client.get(key) == payload
        print(f"lifecycle residency {target}: payload verified", flush=True)

    current = _require_success(
        client.get_weight_revision(*identity), "get before delete"
    )
    _require_success(
        client.delete_weight_revision(*identity, current.metadata.metadata_generation),
        "delete revision",
    )
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        result = _require_success(
            client.reconcile_weight_revision(*identity),
            "reconcile delete",
        )
        if result.availability == store.WeightAvailabilityState.DELETED:
            break
        time.sleep(0.2)
    else:
        raise AssertionError("delete did not complete")
    for key in list(payloads) + [manifest_key]:
        assert client.is_exist(key) == 0, key
    print("lifecycle lease, policy, migration, delete PASSED", flush=True)


def run() -> int:
    store = _require_store()
    master = os.getenv("MOONCAKE_MASTER", "127.0.0.1:50051")
    metadata = os.getenv("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE")
    protocol = os.getenv("MOONCAKE_PROTOCOL", "tcp")
    hostname = os.getenv("MOONCAKE_LOCAL_HOSTNAME", "localhost:17815")

    client = store.MooncakeDistributedStore()
    setup_ret = client.setup(
        hostname,
        metadata,
        64 * 1024 * 1024,
        64 * 1024 * 1024,
        protocol,
        "",
        master,
        None,
        True,
    )
    print(f"setup_ret {setup_ret}", flush=True)
    if setup_ret != 0:
        return setup_ret

    suffix = str(time.time_ns())
    tenant = "default"
    namespace = "e2e"
    resource = f"weight-upsert-{suffix}"
    revision = "main"
    base_generation = 7
    target_generation = 8
    payload = b"base-weight-payload"
    payload_key = f"weights/{resource}/{base_generation}/payload"
    manifest_key = (
        f"weights/{namespace}/{resource}/{revision}/{base_generation}/manifest"
    )
    affinity_digest = hashlib.sha256(b"rank-0").hexdigest()

    try:
        importing = _require_success(
            client.begin_weight_import(
                tenant,
                namespace,
                resource,
                revision,
                base_generation,
                "",
                1,
                len(payload),
                True,
                1,
                0.5,
                1,
                1,
                affinity_digest,
            ),
            "begin_weight_import",
        )
        group_id = importing.manifest.payload_group_id

        payload_config = store.ReplicateConfig()
        payload_config.replica_num = 1
        payload_config.with_hard_pin = True
        payload_config.group_ids = [group_id]
        payload_config.residency_affinity_ids = ["rank-0"]
        payload_config.data_type = store.ObjectDataType.WEIGHT
        assert client.put(payload_key, payload, payload_config) == 0

        manifest_config = store.ReplicateConfig()
        manifest_config.replica_num = 1
        manifest_config.with_hard_pin = True
        manifest_config.group_ids = [group_id]
        manifest_config.data_type = store.ObjectDataType.METADATA
        assert client.put(manifest_key, b"{}", manifest_config) == 0

        ready = _require_success(
            client.commit_weight_import(
                tenant,
                namespace,
                resource,
                revision,
                base_generation,
                importing.metadata_generation,
                manifest_key,
                hashlib.sha256(b"{}").hexdigest(),
                group_id,
                _payload_keys_sha256([payload_key]),
                1,
                len(payload),
            ),
            "commit_weight_import",
        )
        assert ready.availability == store.WeightAvailabilityState.READY

        stale = client.begin_weight_upsert(
            f"stale-{suffix}",
            0,
            tenant,
            namespace,
            resource,
            revision,
            base_generation,
            ready.metadata_generation - 1,
            tenant,
            namespace,
            resource,
            revision,
            target_generation,
            "",
            1,
            len(payload),
            True,
            1,
            0.5,
            1,
            1,
            affinity_digest,
        )
        assert stale[0] is None
        assert stale[1] == 4
        assert stale[2] == 0

        target = _require_success(
            client.begin_weight_upsert(
                f"replace-{suffix}",
                0,
                tenant,
                namespace,
                resource,
                revision,
                base_generation,
                ready.metadata_generation,
                tenant,
                namespace,
                resource,
                revision,
                target_generation,
                "",
                1,
                len(payload),
                True,
                1,
                0.5,
                1,
                1,
                affinity_digest,
            ),
            "begin_weight_upsert",
        )
        assert target.availability == store.WeightAvailabilityState.IMPORTING

        base_view = _require_success(
            client.get_weight_revision(
                tenant,
                namespace,
                resource,
                revision,
                base_generation,
            ),
            "get_weight_revision(base)",
        )
        assert base_view.metadata.availability == store.WeightAvailabilityState.READY

        lineage = _require_success(
            client.abort_weight_upsert(
                f"replace-{suffix}",
                tenant,
                namespace,
                resource,
                revision,
                base_generation,
                tenant,
                namespace,
                resource,
                revision,
                target_generation,
            ),
            "abort_weight_upsert",
        )
        assert lineage.latest_claim is not None
        assert lineage.latest_claim.phase == store.WeightUpsertPhase.ABORTED
        assert lineage.committed_weight_generation == base_generation

        base_view = _require_success(
            client.get_weight_revision(
                tenant,
                namespace,
                resource,
                revision,
                base_generation,
            ),
            "get_weight_revision(base after abort)",
        )
        assert base_view.metadata.availability == store.WeightAvailabilityState.READY
        _run_replacements(
            client,
            store,
            (tenant, namespace, resource, revision, base_generation),
            payload_key,
            payload,
            suffix,
        )
        _run_lifecycle(client, store, suffix)
        print("weight_upsert_tcp_e2e PASSED", flush=True)
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(run())
