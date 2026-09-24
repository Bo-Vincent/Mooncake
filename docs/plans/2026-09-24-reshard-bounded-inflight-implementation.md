# Reshard Bounded In-Flight Transfer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add bounded concurrent Scatter submission for model-weight resharding without weakening completion, registration, or allocation lifetime guarantees.

**Architecture:** Expose the existing asynchronous C++ `submitScatter()` operation through non-draining Python bindings. Add a resource-neutral bounded ticket window to the reshard executor, then feed it one deterministic batch stream from reader and sink. Composite pending tickets preserve all unresolved operations through the existing recovery boundary.

**Tech Stack:** C++17, pybind11, Python 3.12, pytest, Mooncake Transfer Engine.

---

### Task 1: Non-draining Scatter bindings

**Files:**
- Modify: `mooncake-integration/transfer_engine/transfer_engine_py.cpp`
- Test: `mooncake-wheel/tests/transfer_engine_initiator_test.py`

1. Add native tests that submit a Scatter write/read ticket and assert the call
   returns before explicit `drain()`.
2. Verify the tests fail because the submit methods are absent.
3. Extract common Scatter ticket construction and expose
   `submit_scatter_write()` and `submit_scatter_read()` without an initial
   drain. Keep both synchronous methods unchanged.
4. Build the Python wheel target and run the focused native tests.
5. Commit the binding and tests together.

### Task 2: Composite completion contract

**Files:**
- Modify: `mooncake-reshard/python/mooncake/reshard/transfer_engine/completion.py`
- Test: `mooncake-reshard/tests/reshard_transfer_engine/test_executor.py`

1. Add failing tests for composite success, known failure, unknown child,
   interruption, restart-required propagation, and idempotent repeated drain.
2. Add a private composite completion ticket that aggregates child tickets and
   drains every child before publishing a terminal result.
3. Run the focused completion tests and type checks.
4. Commit the completion contract.

### Task 3: Bounded executor window

**Files:**
- Modify: `mooncake-reshard/python/mooncake/reshard/transfer_engine/executor.py`
- Modify: `mooncake-reshard/python/mooncake/reshard/transfer_engine/__init__.py`
- Test: `mooncake-reshard/tests/reshard_transfer_engine/test_executor.py`

1. Add RED tests proving the current peak in-flight count is one.
2. Add tests for exact window bounds, deterministic receipt order, capability
   rejection before side effects, known-failure drain, and unknown completion
   handoff.
3. Refactor batch argument preparation so synchronous and non-draining paths
   share one implementation.
4. Add `TransferSubmission.execute_batches()` with a bounded FIFO window.
5. Run executor tests, Pyright, and Ruff.
6. Commit the executor implementation.

### Task 4: Reader and sink integration

**Files:**
- Modify: `mooncake-reshard/python/mooncake/reshard/weight/_te/execution.py`
- Modify: `mooncake-reshard/python/mooncake/reshard/weight/_te/reader.py`
- Modify: `mooncake-reshard/python/mooncake/reshard/weight/_te/sink.py`
- Test: `mooncake-reshard/tests/model_weight_te/test_batch_lowering_limits.py`
- Test: `mooncake-reshard/tests/model_weight_te/test_reader.py`
- Test: `mooncake-reshard/tests/model_weight_te/test_sink.py`
- Test: `mooncake-reshard/tests/model_weight_te/test_pending_lifecycle.py`

1. Add constructor validation for `max_inflight_batches` and preserve default 1.
2. Add RED tests with at least two endpoints proving overlap only when enabled.
3. Feed all endpoint batches into one executor window and aggregate receipts by
   endpoint.
4. Add failure-path tests proving registrations and allocation tokens remain
   pinned until all child tickets are terminal or handed to pending recovery.
5. Run all model-weight TE tests and type checks.
6. Commit adapter integration and tests.

### Task 5: Exact-head validation and benchmark

**Files:**
- Modify: `docs/source/design/mooncake-reshard/model-weight-reshard-planner.md`
- Create: benchmark artifact under `benchmark-artifacts/reshard-bounded-inflight/`

1. Document the default synchronous behavior, opt-in bounded window, and
   composite recovery semantics.
2. Run the full reshard test suite, Pyright, PR-scoped pre-commit, changed-line
   formatting, Sphinx, and `git diff --check`.
3. Build and run native TCP smoke; run RDMA/GDR smoke on the available hardware.
4. Benchmark window sizes 1, 2, 4, and 8 from identical initial state and report
   throughput plus end-to-end time.
5. Request independent community-owner review on the exact head. Reproduce every
   formal finding as RED and rerun the same case as GREEN after fixes.
