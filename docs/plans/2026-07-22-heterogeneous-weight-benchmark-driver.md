# Heterogeneous Weight Benchmark Driver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a non-executing dry-run driver that maps one canonical logical tensor case to NCCL M2N placement/arguments and Mooncake runtime manifest/`WeightTransferPlan` statistics.

**Architecture:** A strict case-schema module owns all backend-neutral validation. Independent M2N and Mooncake adapters consume the immutable case object and emit structured dry-run plans; a thin CLI serializes deterministic JSON and intentionally exposes no execution command.

**Tech Stack:** Python 3.10+, dataclasses, stdlib JSON/argparse, Mooncake `model_weight` manifest/planner APIs, pytest, ruff.

---

> 用户已明确要求先验证、不提交。本计划中的每个常规 commit checkpoint 均替换为
> diff review checkpoint；不得执行 `git commit`、`git push` 或创建 PR。

### Task 1: Strict case schema and execution gate

**Files:**
- Create: `mooncake-reshard/benchmarks/heterogeneous_weight_reshard/__init__.py`
- Create: `mooncake-reshard/benchmarks/heterogeneous_weight_reshard/case_spec.py`
- Create: `mooncake-reshard/benchmarks/heterogeneous_weight_reshard/tests/test_case_spec.py`

**Step 1: Write failing schema tests**

Cover valid config loading, exact logical bytes, source/target/world rank counts,
shape divisibility, 2-D/3-D M2N rank, duplicate IDs across categories, per-host GPU
capacity, `required_ranks` consistency and unknown dtype rejection.

**Step 2: Verify RED**

Run:

```bash
PYTHONPATH=mooncake-reshard/python:mooncake-reshard \
python3 -m pytest -q \
  mooncake-reshard/benchmarks/heterogeneous_weight_reshard/tests/test_case_spec.py
```

Expected: import failure because `case_spec.py` does not exist.

**Step 3: Implement immutable schema types**

Add `PlacementSpec`, `MeshSpec`, `BenchmarkCase`, `BenchmarkConfig`, strict JSON
loading, dtype itemsize mapping and `execution_is_authorized(env)` requiring both
configuration and environment gates.

**Step 4: Verify GREEN and review diff**

Run the Task 1 test file and `git diff --check`. Do not commit.

### Task 2: M2N dry-run adapter

**Files:**
- Create: `mooncake-reshard/benchmarks/heterogeneous_weight_reshard/m2n_adapter.py`
- Create: `mooncake-reshard/benchmarks/heterogeneous_weight_reshard/tests/test_m2n_adapter.py`

**Step 1: Write failing adapter tests**

Assert exact `ncclMesh` summaries, tensor/shard CLI arguments, world size and MPMD
argv for TP4 -> TP8, TP8 -> TP4, dim0 -> dim1/dim2 and DP2/TP4 -> DP1/TP8.
Assert source ranks map only to the source host and target ranks only to the target
host.

**Step 2: Verify RED**

Run only `test_m2n_adapter.py`; expect an import failure.

**Step 3: Implement structured M2N plan generation**

Generate argv as a tuple, use two OpenMPI app contexts separated by `:`, repeat the
same M2N descriptor arguments in both contexts, always include validation, and never
call subprocess.

**Step 4: Verify GREEN and review diff**

Run Task 1 and Task 2 tests plus `git diff --check`. Do not commit.

### Task 3: Mooncake static planner adapter

**Files:**
- Create: `mooncake-reshard/benchmarks/heterogeneous_weight_reshard/mooncake_adapter.py`
- Create: `mooncake-reshard/benchmarks/heterogeneous_weight_reshard/tests/test_mooncake_adapter.py`

**Step 1: Write failing planner tests**

Assert exact region/segment/inner-byte values for all approved cases, exact per-target
batch counts, DP source dedup, fake addresses remaining nonzero/non-overlapping, and
stress rejection when a region exceeds `1_000_000` segments.

**Step 2: Verify RED**

Run only `test_mooncake_adapter.py`; expect an import failure.

**Step 3: Implement manifest and plan construction**

Construct `TensorDescriptor`, one `WeightPlacementManifest` and one
`WeightRuntimeBindingManifest` per logical rank with stable fake fragments, call
`plan_placement_transfer` followed by `bind_logical_transfer_plan`, then summarize
operations without iterating individual segments. Never instantiate a transfer engine.

**Step 4: Verify GREEN and review diff**

Run Task 1 and Task 3 tests plus the existing N-D planner test. Do not commit.

### Task 4: Deterministic dry-run CLI and canonical cases

**Files:**
- Create: `mooncake-reshard/benchmarks/heterogeneous_weight_reshard/__main__.py`
- Create: `mooncake-reshard/benchmarks/heterogeneous_weight_reshard/cases.json`
- Create: `mooncake-reshard/benchmarks/heterogeneous_weight_reshard/tests/test_cli.py`

**Step 1: Write failing CLI tests**

Cover deterministic JSON, all-case and single-case selection, unknown case failure,
stress case visibility, absent `run` command and proof that no subprocess API is used.

**Step 2: Verify RED**

Run only `test_cli.py`; expect the module entry point to be missing.

**Step 3: Implement CLI and approved case split**

Add only `dry-run --config --case --m2n-binary --json`. Move the original
`[2048,2048,2048]` dim0 -> dim2 case to `stress_cases`; add physical
`[128,4096,16384]` dim0 -> dim2. Preserve TP, DP and dim1 cases.

**Step 4: Verify GREEN and review diff**

Run all new tests and `git diff --check`. Do not commit.

### Task 5: Benchmark documentation and environment sync

**Files:**
- Create: `mooncake-reshard/benchmarks/heterogeneous_weight_reshard/README.md`
- Modify: `.vin_stage/rdma-m2n-env/README.md` outside the Git worktree only if needed
- Replace remote: `/nvme/tmp/vin-m2n-benchmark/env/benchmark_cases.json`

**Step 1: Document exact boundaries**

Describe dry-run usage, output fields, MPMD placement, execution gates, stress case,
future executor/result metrics and the prohibition on comparing generic TE results to
M2N.

**Step 2: Run local static validation**

Run the CLI for all cases and each physical case. Confirm output contains no timing or
throughput samples and `execution_authorized=false`.

**Step 3: Sync with rsync**

Use `rsync`, not `scp`, to copy the benchmark directory/canonical case config to both
nodes without executing it.

**Step 4: Run remote dry-run/import checks**

On both nodes, run schema tests and CLI dry-run using the isolated Mooncake runtime.
Do not set `VIN_RUN_BENCHMARK`, do not change `execution_enabled`, and do not invoke
M2N/TE binaries.

### Task 6: Regression and review

**Files:**
- Review all files under `mooncake-reshard/benchmarks/heterogeneous_weight_reshard/`

**Step 1: Run focused suite**

Run all new benchmark tests plus existing manifest/planner/TE tests.

**Step 2: Run style checks**

Run ruff check/format check when available and `git diff --check`.

**Step 3: Independent reviewer pass**

Review for accidental execution paths, shell injection, incorrect rank placement,
logical-byte double counting, segment materialization and Mooncake/M2N coupling.

**Step 4: Final worktree audit**

Report changed files and exact validation evidence. Leave all changes unstaged and
uncommitted for the user to inspect and commit manually.
