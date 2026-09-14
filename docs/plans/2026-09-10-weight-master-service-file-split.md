# Weight MasterService File Split Implementation Plan

**Goal:** Move Weight-specific `MasterService` orchestration out of the general Store implementation while preserving the exact public API, state machine, locking, serialization, and runtime behavior.

**Architecture:** Keep `MasterService` as the owner and split only its out-of-line method definitions. Put metadata-facing APIs and durable publication in `master_service_weight_management.cpp`; put reconciliation, physical residency changes, managed-group protection, and deletion in `master_service_weight_lifecycle.cpp`. Leave generic Store group operations and their Weight guard call sites in `master_service.cpp`.

**Tech Stack:** C++20, CMake, GoogleTest, Mooncake Store Master/RPC/HA, Python Reshard tests.

---

### Task 1: Capture the structural RED and baseline

**Files:**

- Inspect: `mooncake-store/src/master_service.cpp`
- Inspect: `mooncake-store/src/CMakeLists.txt`

**Steps:**

1. Record exact HEAD and the pre-existing RFC working-tree file.
2. Assert that `master_service_weight_management.cpp` and
   `master_service_weight_lifecycle.cpp` do not exist.
3. Assert that `BeginWeightImport`, `ReconcileWeightRevision`, and
   `DeleteWeightRevision` are still defined in `master_service.cpp`.
4. Treat that expected structural failure as the RED for the requested file
   ownership change; do not alter behavior tests to manufacture a failure.

### Task 2: Extract metadata-facing MasterService methods

**Files:**

- Create: `mooncake-store/src/master_service_weight_management.cpp`
- Modify: `mooncake-store/src/master_service.cpp`

**Move without semantic edits:**

- `AcquireWeightGroupOperationLock`
- `BeginWeightImport`, `CommitWeightImport`, `AbortWeightImport`
- `GetWeightRevision`, `ListWeightRevisions`, `UpdateWeightPolicy`
- `AcquireWeightRevisionLease`, `RenewWeightRevisionLease`,
  `ReleaseWeightRevisionLease`
- `StartWeightResidencyOperation`, `StartWeightResidencyOperationLocked`,
  `QueryWeightOperation`
- `ValidateWeightGroupForCommit`
- `PersistAndPublishWeightMutation`
- `PersistAndPublishWeightOperationMutation`
- `PersistAndPublishWeightLeaseMutation`

Keep generic `RegisterGroupMember`, `UnregisterGroupMember`, and
`GetGroupMemberKeys` in `master_service.cpp`.

### Task 3: Extract physical Weight lifecycle methods

**Files:**

- Create: `mooncake-store/src/master_service_weight_lifecycle.cpp`
- Modify: `mooncake-store/src/master_service.cpp`

**Move without semantic edits:**

- `ReconcileWeightRevision`, `ReconcileWeightMetadataOnce`
- `DeleteWeightRevision`
- `RunWeightReconciliationForTesting`, `DropWeightGroupMemberForTesting`
- `SnapshotWeightGroup`, `SnapshotManagedWeightGroups`,
  `IsManagedWeightObject`
- `EvictManagedWeightGroupToCold`, `EvictManagedWeightMembersToCold`,
  `QueueManagedWeightMemberOffload`

Keep `EvictGroupOrObject` and generic eviction call sites in
`master_service.cpp`. Move only Weight-only constants and helpers; retain
`GetResidencyAffinityIdForKey` in the general file because ordinary put paths
consume it.

### Task 4: Wire and format the new translation units

**Files:**

- Modify: `mooncake-store/src/CMakeLists.txt`
- Modify: the two new `.cpp` files

**Steps:**

1. Add both sources to the same Store object/library target that owns
   `master_service.cpp`.
2. Give each file only the headers needed by its moved definitions.
3. Run the repository C++ formatter on the touched C++ source and header files.
4. Verify that no Weight management/lifecycle method definition remains in
   `master_service.cpp`, except generic Store integration helpers and call
   sites.
5. Scan changed source comments and confirm they are English-only.

### Task 5: Update the Chinese RFC

**Files:**

- Modify: `docs/plans/2026-09-10-rfc-first-class-weight-management-zh.md`

**Steps:**

1. Add the concrete Store source layout and responsibilities.
2. State that the split is translation-unit-only and does not introduce a
   second service, repository, or authority.
3. Explain why RPC, HA, config, and generic Store guard changes remain in their
   owning files.
4. Keep implementation status non-normative and retain all unverified
   boundaries.

### Task 6: Verify behavior and final structure

**Local checks:**

1. Run `git diff --check` and PR-scoped pre-commit on touched files.
2. Run the complete Python Reshard suite and Pyright.
3. Verify both new source files are in CMake and the main file no longer owns
   Weight orchestration definitions.

**Linux checks:**

1. Apply the exact local diff to the existing exact-head test checkout on
   `erdma-1` and `erdma-2` without committing it.
2. Reconfigure/build the affected Store targets with `USE_CUDA=ON`.
3. Run the ten Weight Management and directly affected Store CTest targets on
   both hosts.
4. Run focused lease/migration/delete concurrency cases.
5. Restore the remote source checkouts to their exact clean HEAD after
   collecting results and regenerate the baseline build configuration.

### Task 7: Community-owner review

1. Fix exact base and post-split head/working-tree diff.
2. Review only the new split diff and its direct call graph because the prior
   exact head already received an APPROVE verdict.
3. Obtain one independent reviewer second opinion.
4. Report only findings backed by a runnable exact-tree public/production-path
   RED; otherwise record them as unverified boundaries.
5. Do not commit or push in this task unless the user separately requests it.
