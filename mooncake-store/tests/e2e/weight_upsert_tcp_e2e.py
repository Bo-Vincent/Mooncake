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
        print("weight_upsert_tcp_e2e PASSED", flush=True)
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(run())
