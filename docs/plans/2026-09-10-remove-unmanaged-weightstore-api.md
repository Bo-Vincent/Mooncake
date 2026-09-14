# Remove Unmanaged WeightStore API Implementation Plan

**Goal:** Remove the unused unmanaged WeightStore surface so managed `weight_*`
APIs are the only supported application path.

**Architecture:** Keep the existing upload, manifest, transaction, planning, and
transfer implementations as internal building blocks. Remove legacy aliases and
generic Store convenience bindings; make `WeightStore.weight_put` and
`WeightStore.weight_get` own the revision lifecycle, while private calls retain
the `_weight_put_*` and `_weight_get_*` naming convention.

**Tech Stack:** Python 3, pytest, Pyright, pybind11/C++, Sphinx/Markdown,
pre-commit.

---

### Task 1: Lock the public API contract

**Files:**

- Modify: `mooncake-reshard/tests/test_store_module_layout.py`
- Modify: `mooncake-reshard/tests/model_weight_store/test_snapshot_writer.py`

**Steps:**

1. Add assertions that `WeightStore` does not expose `plan_upload`,
   `begin_weight_snapshot`, `upload`, `abort_upload`,
   `finalize_upload_transaction`, `commit_upload`, `load_manifest`, `plan_load`,
   or `load`.
2. Assert that the public `mooncake.reshard.weight.store` facade and internal
   package do not export `begin_weight_snapshot` or `plan_weight_upload`.
3. Assert that `WeightStoreWriter` no longer exposes `write_tensor` and its
   constructor has no unmanaged `managed` switch.
4. Run the focused tests and confirm they fail because the legacy surface is
   still present.

### Task 2: Remove facade aliases and unmanaged writer mode

**Files:**

- Modify: `mooncake-reshard/python/mooncake/reshard/weight/_store/store.py`
- Modify: `mooncake-reshard/python/mooncake/reshard/weight/_store/writer.py`
- Delete: `mooncake-reshard/python/mooncake/reshard/weight/_store/entrypoint.py`
- Modify: `mooncake-reshard/python/mooncake/reshard/weight/_store/__init__.py`
- Modify: `mooncake-reshard/python/mooncake/reshard/weight/store.py`
- Modify: `mooncake-reshard/python/mooncake/reshard/weight/__init__.py`

**Steps:**

1. Delete every legacy alias named in Task 1.
2. Replace transaction-internal manifest lookup with a private
   `_weight_get_manifest` helper; do not restore a public manifest-key API.
3. Make `WeightStoreWriter` always create a managed plan and retain only
   `weight_put_tensor` as its write method.
4. Stop exporting the legacy module entrypoint and standalone planner.
5. Run the focused contract tests and confirm they pass.

### Task 3: Migrate internal tests to the new call chain

**Files:**

- Modify: `mooncake-reshard/tests/model_weight_store/*.py`
- Modify: `mooncake-reshard/tests/test_store_module_layout.py`

**Steps:**

1. Replace low-level legacy aliases with private `_weight_put_*` and
   `_weight_get_*` equivalents.
2. Use private manifest lookup only in tests that explicitly exercise internal
   transaction recovery; application-path tests must use `weight_get`.
3. Convert writer tests to `WeightStore.weight_put` and
   `WeightStoreWriter.weight_put_tensor`.
4. Run the complete Reshard test suite.

### Task 4: Remove generic Store convenience bindings

**Files:**

- Modify: `mooncake-integration/store/store_py.cpp`
- Modify: `mooncake-wheel/tests/test_weight_snapshot_api.py`

**Steps:**

1. Remove `MooncakeDistributedStore.begin_weight_snapshot` and the dangling
   `begin_managed_weight_snapshot` binding.
2. Keep Weight management RPC bindings and ordinary Store APIs unchanged.
3. Update the installed-wheel contract to reject both removed shortcuts.
4. Build the affected pybind/native targets on Linux and run the installed-wheel
   contract against that exact module.

### Task 5: Update normative and user-facing documentation

**Files:**

- Modify: `docs/plans/2026-09-10-rfc-first-class-weight-management-zh.md`
- Modify: `docs/source/design/weight-management.md`
- Modify: `docs/source/design/model-weight-store-upload-planning.md`
- Modify: `docs/source/api-reference/python/mooncake-store.md`
- Modify: `mooncake-reshard/README.md`

**Steps:**

1. Remove the unmanaged compatibility promise and migration instructions.
2. State that manifest keys and payload transfer primitives are internal
   implementation details.
3. Present `weight_put`, `weight_get`, metadata query, policy, migration, and
   remove as the only application API.
4. Remove stale examples based on `begin_weight_snapshot`, `upload`,
   `load_manifest`, or `load`.

### Task 6: Review and verify

**Steps:**

1. Search production Python, C++, tests, and rendered docs for every removed
   public symbol; allow only historical plan text that is explicitly marked as
   superseded.
2. Run the complete Python Reshard suite and Pyright.
3. Run PR-scoped pre-commit, `git diff --check`, and changed-comment language
   checks.
4. Apply the exact final diff to `erdma-1` and `erdma-2`; build the affected
   native targets and run the Weight management CTest set.
5. Obtain an independent community-owner review of the final exact diff.
6. Restore both remote test checkouts to their clean exact HEAD.
7. Do not commit or push unless the user requests it separately.
