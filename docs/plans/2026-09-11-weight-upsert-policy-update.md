# Weight Upsert and Policy Update Implementation Plan

**Goal:** Add a generation-aware `weight_upsert` workflow with `PUT_FIRST`
and `DELETE_FIRST` modes, and rename the policy-only API to
`weight_update_policy`.

**Architecture:** Store continues to serve exact immutable weight identities and
does not own serving activation. A durable lineage-scoped replacement claim
serializes successors, fences the base revision during retirement, and lets HA
recovery resume cleanup. `PUT_FIRST` publishes the target before retiring the
base; `DELETE_FIRST` deletes the base before allowing target payload upload.

**Tech Stack:** C++20, struct_pack RPC/OpLog/snapshot state, pybind11, Python 3,
pytest, GoogleTest.

---

### Task 1: Lock the public contracts with failing tests

**Files:**

- Modify: `mooncake-store/tests/weight_management_contract_test.cpp`
- Modify: `mooncake-store/tests/weight_metadata_store_test.cpp`
- Modify: `mooncake-reshard/tests/model_weight_store/test_weight_store_api.py`
- Create: `mooncake-reshard/tests/model_weight_store/test_weight_upsert.py`

**Steps:**

1. Add contract tests for `WeightUpsertMode::{PUT_FIRST, DELETE_FIRST}` and a
   lineage-scoped claim carrying request ID, base/target identities and phase.
2. Add metadata-store tests proving only one successor claim can exist, the
   same request is idempotent, another request conflicts, and target generation
   must be greater than the base and current committed watermark.
3. Change the Python public API assertion from `weight_update` to
   `weight_update_policy` and add `weight_upsert`.
4. Add public-path Python tests for both replacement orders and failure paths.
5. Run the focused tests and capture the expected failures caused by the
   missing contracts and methods.

### Task 2: Implement the native lineage claim state machine

**Files:**

- Modify: `mooncake-store/include/weight_management.h`
- Modify: `mooncake-store/include/weight_metadata_store.h`
- Modify: `mooncake-store/src/weight_metadata_store.cpp`
- Modify: `mooncake-store/tests/weight_metadata_store_test.cpp`
- Modify: `mooncake-store/tests/weight_revision_lease_test.cpp`

**Steps:**

1. Add `WeightLineageIdentity`, `WeightUpsertMode`, `WeightUpsertPhase`,
   `WeightUpsertClaim`, and begin/commit/abort/query request contracts.
2. Persist one lineage record per `(tenant, namespace, resource_id, revision)`.
   Its generation is a CAS fence and its committed watermark is ordering
   metadata, not a serving-active pointer.
3. Implement claim creation and legal phase transitions. Require a non-empty
   request ID, identical lineage, and a strictly newer target generation.
4. Make same-request retries idempotent and reject competing successors.
5. Fence new base leases and renewals once `PUT_FIRST` reaches retirement or
   immediately after `DELETE_FIRST` starts.
6. Export and restore lineage state with snapshot schema v3, accepting v2 as an
   empty-lineage baseline.
7. Run the focused metadata and lease tests until green.

### Task 3: Add Master, HA, RPC and pybind orchestration

**Files:**

- Modify: `mooncake-store/include/master_service.h`
- Modify: `mooncake-store/src/master_service_weight_management.cpp`
- Modify: `mooncake-store/src/master_service_weight_lifecycle.cpp`
- Modify: `mooncake-store/include/ha/oplog/oplog_types.h`
- Modify: `mooncake-store/include/ha/oplog/oplog_applier.h`
- Modify: `mooncake-store/src/ha/oplog/oplog_applier.cpp`
- Modify: `mooncake-store/include/master_client.h`
- Modify: `mooncake-store/src/master_client.cpp`
- Modify: `mooncake-store/include/real_client.h`
- Modify: `mooncake-store/src/real_client.cpp`
- Modify: `mooncake-store/include/rpc_service.h`
- Modify: `mooncake-store/src/rpc_service.cpp`
- Modify: `mooncake-integration/store/store_py.cpp`
- Modify: native weight management, lifecycle, RPC and HA tests nearest to the
  touched code.

**Steps:**

1. Add failing Master/RPC/HA tests for durable claim creation, phase replay,
   generation fencing, lease drain, and restart recovery.
2. Add append-only `WEIGHT_LINEAGE_UPSERT` OpLog encoding and replay; do not
   renumber existing operation types.
3. Implement `BeginWeightUpsert`, `CommitWeightUpsert`, `AbortWeightUpsert` and
   `GetWeightLineage` across Master, client and RPC boundaries.
4. `PUT_FIRST`: reserve the claim, import the target, publish it READY, then
   enter base retirement. Existing base leases remain valid; new leases and
   renewals are fenced; reconciliation deletes the base after drain.
5. `DELETE_FIRST`: reserve and fence the base only when there are no active
   leases or lifecycle operations, delete it in bounded batches, then allow the
   target import. A failure after deletion remains explicitly unavailable and
   retryable with the same request ID.
6. Ensure direct revision deletion and generic Store mutation cannot bypass an
   active replacement claim.
7. Bind only internal native methods in pybind; keep the public application API
   in `WeightStore`.
8. Run focused C++ contract, metadata, lifecycle, RPC and HA tests until green.

### Task 4: Implement the Python WeightStore API

**Files:**

- Modify: `mooncake-reshard/python/mooncake/reshard/weight/management.py`
- Modify: `mooncake-reshard/python/mooncake/reshard/weight/_store/backend.py`
- Modify: `mooncake-reshard/python/mooncake/reshard/weight/_store/contracts.py`
- Create: `mooncake-reshard/python/mooncake/reshard/weight/_store/upsert.py`
- Modify: `mooncake-reshard/python/mooncake/reshard/weight/_store/store.py`
- Modify: `mooncake-reshard/python/mooncake/reshard/weight/_store/writer.py`
- Modify: package exports under `mooncake-reshard/python/mooncake/reshard/weight/`
- Modify: `mooncake-wheel/tests/test_weight_snapshot_api.py`
- Modify: Python tests named in Task 1 and nearby writer/recovery tests.

**Steps:**

1. Rename public and internal Python policy methods to
   `weight_update_policy`; the native RPC name remains `update_weight_policy`.
2. Add `weight_upsert(snapshot, adapter, replacing=..., mode=PUT_FIRST,
   request_id=None, ...)` and validate the complete lineage plus increasing
   generation before native mutation.
3. When omitted, derive a stable SHA-256 request ID from the complete base and
   target identities, mode, and expected metadata generation. Retain it on the
   writer so an identical retry after response loss reuses the same claim.
4. Keep normal tensor upload logic shared with `weight_put`; add narrow commit
   and abort hooks for replacement finalization.
5. For `DELETE_FIRST`, do not expose a writer until the base group is fully
   deleted and its space reclaimed. Return `BUSY` before side effects when an
   active lease or operation exists.
6. For `PUT_FIRST`, keep the base READY until the target is READY. Upload or
   commit failure must leave the base usable.
7. Run model-weight Store tests, public API contract tests and Pyright until
   green.

### Task 5: Update and polish the RFC and user documentation

**Files:**

- Modify: `docs/plans/2026-09-10-rfc-first-class-weight-management-zh.md`
- Modify: `docs/plans/2026-09-10-rfc-first-class-weight-management-en.md`
- Modify: `docs/plans/2026-09-10-weight-management-api-policy-design.md`
- Modify: `docs/source/design/weight-management.md`
- Modify: `docs/source/api-reference/python/mooncake-store.md`
- Modify: `mooncake-reshard/README.md`

**Steps:**

1. Replace `weight_update` with `weight_update_policy` everywhere it means a
   policy-only mutation.
2. Document `weight_upsert` as storage-lifecycle replacement, not serving
   activation and not generic Store in-place upsert.
3. Define `PUT_FIRST` as the default and `DELETE_FIRST` as explicit
   capacity-saving downtime mode.
4. Document generation ordering, request-id idempotency, lease fencing,
   failure windows, HA recovery and the relationship to sparse-weight RFCs.
5. Keep the Chinese and English RFC structurally equivalent.
6. Remove repetitive summaries, invented guarantees and process narration;
   preserve only externally useful contract and rationale.
7. Run Markdown/pre-commit checks and a forced Sphinx build.

### Task 6: Integration review and verification

**Steps:**

1. Audit the final diff against every accepted API and state-machine invariant.
2. Run exact public-path RED/GREEN checks for any review finding before fixing
   it.
3. Run the complete Reshard pytest suite, Pyright, pre-commit and Sphinx.
4. Build native Store and run CTest in an isolated Linux checkout; repeat the
   weight contract/lifecycle/RPC tests on `erdma-1` and `erdma-2` when needed.
5. Verify changed comments are English and review prose for repository-native
   wording.
6. Record exact head, patch digest, commands, pass counts and remaining external
   boundaries. Do not push or create a PR without a separate user request.
