# Weight Management Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Promote one complete model-weight revision to a first-class Mooncake Store resource with durable discovery metadata, manifest-backed integrity, revision leases, and logically atomic group lifecycle operations.

**Architecture:** Add a Store Master-owned `WeightCatalog` whose authoritative record is one `WeightRevisionMetadata` per `(tenant, namespace, resource_id, revision, weight_generation)`. The catalog record is persisted through the existing Master snapshot and OpLog HA paths, references one immutable `StoredWeightManifest` object, and never duplicates tensor or fragment contents. The manifest and all referenced weight payload objects remain in the same Store `group_id`; Store object metadata continues to own replica placement, while the Weight Catalog owns discovery, availability, leases, and completion of group-level operations.

**Tech Stack:** C++20, Mooncake Store Master/RPC/HA, struct_pack and MessagePack serialization, pybind11, Python 3.10+, `mooncake.reshard.weight.WeightStore`, pytest, GoogleTest.

---

## 1. Locked decisions and scope

### 1.1 Aggregate boundary

One weight revision is the only managed aggregate. A tensor is a child described by the manifest and has no independent readiness, lease, eviction, or metadata record.

```text
WeightRevisionMetadata          Store Master WeightCatalog
  -> manifest_key + digest
  -> payload_group_id
  -> lifecycle / residency / operation / leases
           |
           v
StoredWeightManifest            immutable Store METADATA object
  -> tensors[]
  -> fragments[] -> object_key + object_offset + nbytes
           |
           v
Store ObjectMetadata            existing per-key replica authority
  -> memory / local disk / DFS / NoF replicas
```

### 1.2 Metadata location

`WeightRevisionMetadata` lives in an in-memory Store Master `WeightCatalog` and is durable through the same HA mechanisms as other Master authority:

- active Master: authoritative sharded/internally locked catalog;
- OpLog mode: durable-before-visible catalog mutations replicated to standby;
- snapshot mode: optional `weight_catalog` field in the Master metadata snapshot;
- no-HA/no-snapshot mode: the same restart durability boundary as current Master object metadata.

Do not store the catalog record inside the weight payload group. The record must remain discoverable while that group is cold, degraded, being deleted, or already absent.

### 1.3 Manifest location

Keep the current manifest storage contract:

```text
weights/<namespace>/<resource_id>/<revision>/<weight_generation>/manifest
```

It remains a hard-pinned `ObjectDataType::METADATA` Store object during import and shares `payload_group_id` with the committed `ObjectDataType::WEIGHT` fragments. The catalog stores only a validated reference and digest.

### 1.4 State model

Keep availability, physical residency, and in-progress operation separate.

```text
availability: IMPORTING -> READY -> DEGRADED -> READY
                         -> DELETING -> DELETED

residency:    UNKNOWN | HOT | COLD | MIXED | ABSENT

operation:    NONE | EVICTING | REHYDRATING | REPAIRING
```

`READY` means that the manifest is readable, its identity/digest validates, and every required payload object has at least one readable replica. DRAM eviction alone changes residency, not availability. A partially completed group operation never publishes a terminal residency or `DELETED` state.

### 1.5 Ownership boundary

- Store Weight Management owns revision discovery, import publication, manifest reference, readable-residency summary, revision leases, retention, reconciliation, and safe reclamation.
- `StoredWeightManifest` owns tensor descriptors and tensor-to-payload mapping.
- Store object metadata owns physical replicas, object leases, placement, promotion, and tier-level eviction.
- SGLang owns worker-local runtime binding, GPU addresses, allocation guards, and local snapshot generations.
- Slime/Kubernetes/Ray owns multi-worker activation barriers and traffic switching. Store records serving/read leases but does not select the globally active serving revision.

### 1.6 Required safety invariants

1. A catalog record may reference only one immutable manifest identity.
2. `READY` is published only after payload commit, manifest commit, and exact group membership validation.
3. `manifest_key`, `manifest_digest`, `payload_group_id`, and payload-key digest become immutable after `READY`.
4. All catalog mutations require `expected_metadata_generation`; stale writers fail closed.
5. An active revision lease blocks delete and any operation that would remove its last readable replica.
6. Generic pressure eviction must not independently evict objects belonging to a managed weight group; it must skip them or delegate to the weight lifecycle path.
7. A group operation may be physically partial while running, but it remains non-terminal and unavailable for conflicting operations until reconciliation completes.
8. Catalog metadata never contains tensors, fragments, GPU addresses, runtime endpoints, or framework objects.

## 2. Proposed public contracts

### 2.1 `WeightRevisionMetadata`

```cpp
struct WeightRevisionMetadata {
    TenantId tenant_id;
    std::string name_space;
    std::string resource_id;
    std::string revision;
    uint64_t weight_generation{0};

    std::string manifest_key;
    std::string manifest_sha256;
    std::string payload_group_id;
    std::string payload_keys_sha256;
    uint64_t payload_count{0};
    uint64_t logical_bytes{0};

    WeightAvailabilityState availability{WeightAvailabilityState::IMPORTING};
    WeightResidencyState residency{WeightResidencyState::UNKNOWN};
    WeightOperationState operation{WeightOperationState::NONE};
    uint64_t metadata_generation{1};

    uint64_t created_at_ms{0};
    uint64_t updated_at_ms{0};
};
```

Lease records are stored separately inside the catalog and returned only as an aggregate count/deadline in read APIs. Persist lease IDs, holders, expiry, revision identity, and the metadata generation they fence; do not rely on a mutable count as authority.

### 2.2 Store APIs

```text
BeginWeightImport(identity, payload_group_id, expected summary)
CommitWeightImport(identity, expected_generation, manifest_ref, payload summary)
AbortWeightImport(identity, expected_generation)

GetWeightRevision(identity)
ListWeightRevisions(namespace, resource_id, page_token, limit)

AcquireWeightRevisionLease(identity, expected_generation, holder, ttl)
RenewWeightRevisionLease(lease_id, ttl)
ReleaseWeightRevisionLease(lease_id)

StartWeightResidencyOperation(identity, expected_generation, target_residency)
QueryWeightOperation(operation_id)
ReconcileWeightRevision(identity)

DeleteWeightRevision(identity, expected_generation)
```

Do not add `active_revision` or traffic-routing APIs to Store in this change.

## 3. Implementation tasks

### Task 1: Define the native weight-management domain contract

**Files:**

- Create: `mooncake-store/include/weight_management.h`
- Create: `mooncake-store/tests/weight_management_contract_test.cpp`
- Modify: `mooncake-store/tests/CMakeLists.txt`

**Steps:**

1. Add failing tests for strict identity validation, SHA-256 formatting, enum serialization, generation bounds, and the absence of tensor/runtime fields.
2. Add transition-table tests: only `IMPORTING -> READY/DELETING`, `READY -> DEGRADED/DELETING`, `DEGRADED -> READY/DELETING`, and `DELETING -> DELETED` are legal.
3. Configure with `cmake -S . -B build -DBUILD_UNIT_TESTS=ON -DWITH_STORE_RUST=OFF`.
4. Run `cmake --build build --target weight_management_contract_test -j` and confirm the test target fails because the contract is absent.
5. Implement the enums, revision identity, manifest reference, revision metadata, lease record, operation record, and request/response wire structures with `YLT_REFL`/struct_pack support.
6. Reject empty identity components, malformed digests, zero operation IDs, and invalid state combinations in one shared validator.
7. Run `ctest --test-dir build -R weight_management_contract_test --output-on-failure` and confirm it passes.
8. Commit as `feat(store): define weight management contracts` with `Signed-off-by: Vincent Gao <vincentbo@linux.alibaba.com>`.

### Task 2: Add an authoritative in-memory `WeightCatalog`

**Files:**

- Create: `mooncake-store/include/weight_catalog.h`
- Create: `mooncake-store/src/weight_catalog.cpp`
- Create: `mooncake-store/tests/weight_catalog_test.cpp`
- Modify: `mooncake-store/CMakeLists.txt`
- Modify: `mooncake-store/tests/CMakeLists.txt`

**Steps:**

1. Write failing tests for begin/import idempotency, conflicting manifest identity, stale generation, exact-revision lookup, deterministic pagination, lease expiry, and operation exclusion.
2. Implement a catalog keyed by tenant plus revision identity, with a separate `payload_group_id -> revision identity` reverse index.
3. Use one documented lock order: catalog lock before no Store metadata-shard lock; never hold the catalog lock across Store I/O, OpLog durability waits, or callbacks.
4. Make every mutation return a candidate new record without publishing it; the Master layer publishes only after HA durability succeeds.
5. Make lease expiration idempotent and generation-fenced; a stale lease must not protect a recreated revision.
6. Run the new tests under repeated create/expire/recreate and concurrent CAS attempts.
7. Commit as `feat(store): add weight revision catalog`.

### Task 3: Integrate import publication with Store group authority

**Files:**

- Modify: `mooncake-store/include/master_service.h`
- Modify: `mooncake-store/src/master_service.cpp`
- Create: `mooncake-store/tests/master_service_weight_management_test.cpp`
- Modify: `mooncake-store/tests/CMakeLists.txt`

**Steps:**

1. Add failing tests proving `CommitWeightImport` rejects a missing manifest, wrong object type, wrong group, incomplete payload group, extra orphan payload, mismatched count/bytes, and stale generation.
2. Add a canonical digest over sorted committed payload keys. Persist only its SHA-256 and count in catalog metadata; do not copy the key list or manifest body.
3. Implement `BeginWeightImport` so Store allocates/validates the canonical `payload_group_id` and returns the catalog generation consumed by the writer.
4. Implement `CommitWeightImport` as: snapshot group membership, validate the manifest object and completed payload objects using read-only object metadata, recheck the catalog generation, durably publish `READY`, then return the final record.
5. Treat the client-provided manifest SHA-256 as an immutable reference attestation; verify it again when Python reads the manifest. Do not make Master read or understand tensor JSON.
6. Implement idempotent abort. Abort must not delete a manifest already attached to a `READY` record.
7. Add a failure test for `manifest committed -> response lost -> CommitWeightImport retry`.
8. Commit as `feat(store): publish manifest-backed weight revisions`.

### Task 4: Persist catalog mutations through OpLog

**Files:**

- Modify: `mooncake-store/include/ha/oplog/oplog_types.h`
- Modify: `mooncake-store/include/ha/oplog/oplog_applier.h`
- Modify: `mooncake-store/src/ha/oplog/oplog_applier.cpp`
- Modify: `mooncake-store/include/metadata_store.h`
- Modify: `mooncake-store/include/ha/standby_metadata_store.h`
- Modify: `mooncake-store/src/ha/standby_metadata_store.cpp`
- Modify: `mooncake-store/tests/ha/oplog/mock_metadata_store.h`
- Modify: `mooncake-store/tests/ha/oplog/oplog_types_test.cpp`
- Modify: `mooncake-store/tests/ha/oplog/oplog_applier_test.cpp`

**Steps:**

1. Add `WEIGHT_METADATA_UPSERT`, `WEIGHT_METADATA_DELETE`, `WEIGHT_LEASE_UPSERT`, and `WEIGHT_LEASE_DELETE` OpLog types without renumbering existing values.
2. Write RED tests for checksum/size validation, duplicate replay, future sequence rejection, stale catalog generation, lease replay, and delete tombstones.
3. Extend standby metadata storage with a separate weight-catalog namespace; never encode catalog entries as fake object metadata.
4. Serialize the complete post-mutation record so replay is idempotent and does not depend on earlier in-memory state.
5. Use durable-before-visible publication for every catalog state that authorizes discovery, load, or deletion.
6. Add a rollout capability gate: do not enable weight-management mutations until every standby can apply the new OpLog types. An old standby must fail closed, not silently skip entries.
7. Run OpLog codec/applier tests and a duplicate/reorder replay loop.
8. Commit as `feat(store): replicate weight catalog through oplog`.

### Task 5: Add backward-compatible Master snapshot persistence

**Files:**

- Modify: `mooncake-store/include/master_service.h`
- Modify: `mooncake-store/src/master_service.cpp`
- Modify: `mooncake-store/include/ha/snapshot/master_snapshot_codec.h`
- Modify: `mooncake-store/tests/ha/snapshot/master_snapshot_codec_test.cpp`
- Modify: `mooncake-store/tests/ha/snapshot/catalog_backed_snapshot_provider_test.cpp`

**Steps:**

1. Add RED round-trip tests covering catalog records, leases, operation records, and group reverse indexes.
2. Add an optional `weight_catalog` field to the existing Master metadata MessagePack map. Rebuild derived reverse indexes after restore.
3. Keep old snapshots valid: missing `weight_catalog` means an empty catalog. Reject malformed present fields rather than dropping them.
4. Ensure failed restore resets catalog state before the next snapshot candidate is attempted.
5. Test `snapshot -> restore -> expire lease -> resume EVICTING reconciliation`.
6. Test promotion from snapshot plus newer catalog OpLog entries.
7. Commit as `feat(store): snapshot weight catalog state`.

### Task 6: Expose catalog operations through Master RPC and native clients

**Files:**

- Modify: `mooncake-store/include/rpc_types.h`
- Modify: `mooncake-store/include/rpc_service.h`
- Modify: `mooncake-store/src/rpc_service.cpp`
- Modify: `mooncake-store/include/master_client.h`
- Modify: `mooncake-store/src/master_client.cpp`
- Modify: `mooncake-store/include/pyclient.h`
- Modify: `mooncake-store/include/real_client.h`
- Modify: `mooncake-store/src/real_client.cpp`
- Create: `mooncake-store/tests/weight_management_rpc_test.cpp`

**Steps:**

1. Add compile-failing client tests for all APIs listed in section 2.2.
2. Register every RPC explicitly and add `RpcNameTraits` entries; bind tenant identity through the existing request-tenant boundary.
3. Add bounded pagination and request-size validation. Do not add regex-based revision discovery.
4. Preserve exact error distinctions for not found, stale generation, not ready, active lease, conflicting operation, and invalid manifest/group.
5. Add RPC timeout tests proving an uncertain mutation can be retried with the same identity/generation without double transition.
6. Commit as `feat(store): expose weight management RPCs`.

### Task 7: Add Python bindings and typed Python management contracts

**Files:**

- Modify: `mooncake-integration/store/store_py.cpp`
- Create: `mooncake-reshard/python/mooncake/reshard/weight/management.py`
- Modify: `mooncake-reshard/python/mooncake/reshard/weight/__init__.py`
- Create: `mooncake-reshard/tests/model_weight_store/test_management_contracts.py`
- Modify: `mooncake-reshard/tests/model_weight_store/helpers.py`

**Steps:**

1. Add failing Python tests for enum/record conversion, exact identity round-trip, list pagination, stale generation, and typed error mapping.
2. Bind the native records and APIs without accepting arbitrary dictionaries or alternate field names.
3. Release the GIL around RPCs and retain no borrowed Python buffers in native catalog records.
4. Expose immutable Python `WeightRevisionMetadata`, `WeightRevisionIdentity`, and `WeightRevisionLease` types.
5. Keep tensor and fragment definitions exclusively in `StoredWeightManifest`; assert that management records contain no tensor fields.
6. Run `pytest -q mooncake-reshard/tests/model_weight_store/test_management_contracts.py`.
7. Commit as `feat(reshard): expose weight management metadata`.

### Task 8: Make `WeightStore` publish and resolve managed revisions

**Files:**

- Modify: `mooncake-reshard/python/mooncake/reshard/weight/_store/backend.py`
- Modify: `mooncake-reshard/python/mooncake/reshard/weight/_store/store.py`
- Modify: `mooncake-reshard/python/mooncake/reshard/weight/_store/upload.py`
- Modify: `mooncake-reshard/python/mooncake/reshard/weight/_store/transaction.py`
- Modify: `mooncake-reshard/python/mooncake/reshard/weight/_store/writer.py`
- Modify: `mooncake-reshard/python/mooncake/reshard/weight/_store/load.py`
- Create: `mooncake-reshard/tests/model_weight_store/test_managed_revision.py`

**Steps:**

1. Add RED tests for the complete production path: `begin -> payload upload -> manifest commit -> catalog READY -> resolve -> manifest digest check -> plan_load`.
2. Change managed snapshot creation to call `BeginWeightImport` and use the Store-issued `payload_group_id` and metadata generation instead of deriving an independent management identity.
3. Preserve current payload and manifest key layout inside that group.
4. After current transaction commit returns a `StoredWeightManifest`, call `CommitWeightImport` with `manifest_key`, manifest SHA-256, payload-key digest/count, and logical bytes.
5. Make commit retryable across both uncertainty windows: Store manifest put and catalog READY publication.
6. Add `get_weight_revision`, `list_weight_revisions`, and `load_weight_revision` APIs. `load_weight_revision` must acquire a revision lease before reading the manifest and retain it until Store-to-runtime transfer reaches a terminal state.
7. Verify manifest identity and digest against catalog metadata before planning any tensor range.
8. Keep legacy direct `load_manifest(manifest_key)` available only as an explicitly unmanaged compatibility path; document that it has no lifecycle guarantee.
9. Run the complete `model_weight_store` suite.
10. Commit as `feat(reshard): manage Store-backed weight revisions`.

### Task 9: Add revision lease management

**Files:**

- Modify: `mooncake-store/include/weight_catalog.h`
- Modify: `mooncake-store/src/weight_catalog.cpp`
- Modify: `mooncake-store/include/master_service.h`
- Modify: `mooncake-store/src/master_service.cpp`
- Modify: `mooncake-reshard/python/mooncake/reshard/weight/_store/load.py`
- Create: `mooncake-store/tests/weight_revision_lease_test.cpp`
- Modify: `mooncake-reshard/tests/model_weight_store/test_managed_revision.py`

**Steps:**

1. Add RED tests for acquire/renew/release, expiry, stale-generation fencing, failover restore, and exactly-once release.
2. Reject lease acquisition unless availability is `READY` and no conflicting group operation is running.
3. Make delete and last-readable-replica removal fail while any live revision lease exists.
4. Let lease renewal extend only the revision lease; do not silently rewrite tensor manifests or runtime binding leases.
5. Ensure exceptions and cancellation in `load_weight_revision` release or expire the revision lease only after Store transfer resources are safely drained.
6. Commit as `feat(store): fence weight revisions with leases`.

### Task 10: Add managed group residency, deletion, and reconciliation

**Files:**

- Modify: `mooncake-store/include/master_service.h`
- Modify: `mooncake-store/src/master_service.cpp`
- Modify: `mooncake-store/tests/master_service_group_test.cpp`
- Modify: `mooncake-store/tests/master_service_evict_scenario_test.cpp`
- Create: `mooncake-store/tests/weight_group_lifecycle_test.cpp`

**Steps:**

1. Add failing tests proving generic `BatchEvict`, tenant quota eviction, NoF eviction, explicit remove, and background cleanup cannot independently remove a member of a managed weight group while its catalog record is `READY`.
2. Add a managed-group check through the catalog reverse index. Generic eviction skips managed groups; only a weight operation may change their residency.
3. Implement `StartWeightResidencyOperation` as a durable operation record with target residency, operation ID, cursor/progress, and expected catalog generation.
4. Reuse existing group membership and per-object replica primitives, but run them under the weight operation state machine. Never claim physical atomicity.
5. After each pass, verify every required member. Publish terminal residency only when all members satisfy the target.
6. If a member is busy, pinned unexpectedly, missing, or fails persistence, retain the operation and report `IN_PROGRESS`/`DEGRADED`; retry from observed state.
7. Implement deletion as `READY/DEGRADED -> DELETING -> DELETED`: block new leases, wait for existing leases, remove payload members, remove the manifest last, verify absence, then retain a catalog tombstone.
8. Add restart/failover tests for interruption after the first member, after all payloads but before manifest removal, and after physical completion but before catalog publication.
9. Commit as `feat(store): manage weight group lifecycle`.

### Task 11: Add background reconciliation and observability

**Files:**

- Modify: `mooncake-store/include/master_service.h`
- Modify: `mooncake-store/src/master_service.cpp`
- Modify: `mooncake-store/include/master_metric_manager.h`
- Modify: `mooncake-store/src/master_metric_manager.cpp`
- Create: `mooncake-store/tests/weight_reconciliation_test.cpp`

**Steps:**

1. Add deterministic, clock-injected tests for lease expiry, abandoned `IMPORTING`, partial group operations, manifest loss, and payload loss.
2. Implement a bounded reconciliation worker that snapshots catalog candidates without holding locks across object metadata access.
3. Reconcile `IMPORTING` only through an explicit retry/abort policy; never infer READY from key prefixes alone.
4. Recompute residency from actual replicas and transition READY revisions to DEGRADED when a required object has no readable replica.
5. Emit bounded metrics for revisions by state/residency, active leases, pending operations, reconciliation failures, and operation age. Do not add Prometheus/OpenTelemetry dependencies.
6. Commit as `feat(store): reconcile managed weight revisions`.

### Task 12: Validate HA, compatibility, and end-to-end behavior

**Files:**

- Modify: `mooncake-store/tests/ha/master_service_ha_test.cpp`
- Modify: `mooncake-store/tests/e2e/store_client_e2e.py`
- Create: `mooncake-reshard/tests/model_weight_store/test_management_e2e.py`

**Steps:**

1. Run an exact-head native Store E2E covering import, READY discovery, lease-protected load, cold residency transition, rehydrate, and delete.
2. Kill/promote the active Master after each durable boundary and verify the promoted node returns the same catalog generation and operation state.
3. Verify an older snapshot without `weight_catalog` restores with an empty catalog and existing KV objects remain accessible.
4. Verify an unsupported standby prevents feature enablement before any new OpLog type is emitted.
5. Verify unmanaged ordinary Store objects retain their current eviction and removal behavior.
6. Run:

   ```bash
   cmake --build build -j
   ctest --test-dir build --output-on-failure
   PYTHONPATH=mooncake-reshard/python:mooncake-reshard/tests \
     python3 -m pytest -q mooncake-reshard/tests/model_weight_store
   python3 -m pyright mooncake-reshard/python/mooncake/reshard
   pre-commit run --from-ref 37fbb8daeedb99e6af58acaf77efe09014812aab \
     --to-ref HEAD
   ```

7. Scan the current diff and confirm every added/modified code comment is English.
8. Commit as `test(store): validate weight management lifecycle`.

### Task 13: Document the public architecture and migration

**Files:**

- Create: `docs/source/design/weight-management.md`
- Modify: `docs/source/design/model-weight-store-upload-planning.md`
- Modify: `docs/source/design/reshard-manifest.md`
- Modify: `docs/source/index.md`
- Modify: `docs/source/api-reference/cpp/mooncake-store.md`
- Modify: `docs/source/api-reference/python/mooncake-store.md`

**Steps:**

1. Document the three authorities: Weight Catalog, immutable manifest, and per-object Store metadata.
2. Document exact state meanings, lease behavior, group-operation visibility, recovery, and the boundary with SGLang/Slime activation.
3. State explicitly that tensor-level management metadata is not introduced.
4. Document the migration from unmanaged `load_manifest(manifest_key)` to catalog-based exact-revision lookup.
5. Build documentation with `cd docs && make html` and inspect the generated Weight Management page and navigation.
6. Commit as `docs(store): document weight management architecture`.

## 4. Commit and review gates

Before every commit in this worktree:

```bash
git config user.name "Vincent Gao"
git config user.email vincentbo@linux.alibaba.com
git config author.name "Vincent Gao"
git config author.email vincentbo@linux.alibaba.com
git config committer.name "Vincent Gao"
git config committer.email vincentbo@linux.alibaba.com
git var GIT_AUTHOR_IDENT
git var GIT_COMMITTER_IDENT
```

Each semantic commit must build and pass its own focused tests without relying on a later commit. Before handoff, review the complete stack against the original base and record:

- exact base and head SHA;
- manifest/catalog/group ownership matrix;
- wrong-accept and wrong-reject tests for every state transition;
- snapshot and OpLog compatibility results;
- current diff comment-language scan;
- full local test commands and unverified environment boundaries.

## 5. Acceptance criteria

The feature is complete only when all of the following are demonstrated:

1. A caller can discover an exact READY weight revision without knowing `manifest_key`.
2. The returned catalog record points to one immutable manifest with a verified identity and SHA-256.
3. The manifest resolves every tensor range to Store object keys; no tensor-level metadata catalog exists.
4. A live revision lease prevents unsafe reclamation.
5. No generic eviction path can independently evict a managed weight-group member.
6. Partial residency/delete work is represented as an unfinished operation and never as a terminal state.
7. Snapshot restore and OpLog promotion preserve catalog generations, leases, and resumable operations.
8. Existing unmanaged KV objects and their eviction behavior do not regress.
9. Current WeightStore upload/load tests, native Store tests, Pyright, pre-commit, and docs build pass.
