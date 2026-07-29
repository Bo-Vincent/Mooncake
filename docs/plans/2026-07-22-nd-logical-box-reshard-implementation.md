# N-D Logical-Box Resharding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将现有 manifest + TransferPlan 升级为向后兼容的 N-D logical-box reshard planner，并完成 PP/EP/TP/DP、Store、TE、runtime G2G 和真实推理验证。

**Architecture:** Runtime/Store manifest 提供 logical tensor 和物理 fragment 的全部事实；planner 只做 N-D box coverage、intersection、DP replica 选择和 PP route grouping，输出 backend-neutral `TransferRegion`。TE/Store 使用有界 lazy lowering，未来 N-D/M2N executor 可直接消费 region，不改变 manifest 或 planner。

**Tech Stack:** Python 3.10+、pytest、Mooncake Transfer Engine Python binding、Mooncake Store、SGLang runtime manifest、CUDA/H20 integration tests。

---

### Task 1: Manifest v2 schema 和 v1 兼容

**Files:**
- Modify: `mooncake-model-weight/python/mooncake/model_weight/manifest.py`
- Test: `mooncake-model-weight/tests/test_model_weight_manifest.py`

**Step 1: 写失败测试**

新增测试覆盖：

```python
def test_runtime_manifest_accepts_multi_axis_logical_box():
    tensor = TensorDescriptor(
        tensor_id="layers.0.experts.w1",
        global_shape=(8, 16, 32),
        dtype="bfloat16",
        itemsize=2,
        partition_dim=None,
        shard_dims=(0, 1),
        layer_id=0,
    )
    fragment = RuntimeFragment(
        global_offset=(3, 8, 0),
        local_shape=(1, 8, 32),
        ...,
    )
    RuntimeManifest(..., tensors=(tensor,), fragments=(fragment,))
```

同时测试：

- 单轴 `partition_dim` 行为不变；
- 多维 shard box 合法；
- 非 shard dimension 上的 partial box 被拒绝；
- `partition_dim` 与 `shard_dims` 冲突被拒绝；
- WeightManifest JSON exact fields 和 round-trip。

**Step 2: 运行 RED**

```bash
PYTHONPATH=mooncake-wheel:mooncake-model-weight/python python -m pytest \
  mooncake-model-weight/tests/test_model_weight_manifest.py -q
```

预期：因为 `shard_dims` 尚不存在而失败。

**Step 3: 最小实现**

在 `TensorDescriptor` 增加 `shard_dims` 和规范化属性：

```python
@property
def effective_shard_dims(self) -> tuple[int, ...]:
    if self.shard_dims is not None:
        return self.shard_dims
    return () if self.partition_dim is None else (self.partition_dim,)
```

让 `_validate_fragment_geometry` 使用 `effective_shard_dims`。Runtime inventory
和 persisted WeightManifest 共用一套无版本契约。

**Step 4: 运行 GREEN 和旧回归**

```bash
PYTHONPATH=mooncake-wheel:mooncake-model-weight/python python -m pytest \
  mooncake-model-weight/tests/test_model_weight_manifest.py \
  mooncake-model-weight/tests/test_weight_store.py -q
```

预期：新测试和单轴回归全部通过。

**Step 5: 提交**

```bash
git add mooncake-model-weight/python/mooncake/model_weight/manifest.py \
  mooncake-model-weight/tests/test_model_weight_manifest.py
git commit -m "feat(model_weight): add manifest v2 shard boxes"
```

### Task 2: Backend-neutral TransferRegion

**Files:**
- Modify: `mooncake-model-weight/python/mooncake/model_weight/planner.py`
- Modify: `mooncake-model-weight/python/mooncake/model_weight/__init__.py`
- Test: `mooncake-model-weight/tests/test_model_weight_planner.py`

**Step 1: 写失败测试**

直接构造 source/target fragments，断言 dim0 -> dim1 和 dim0 -> dim2 的：

```python
assert region.overlap_offset == (...)
assert region.overlap_shape == (...)
assert region.source_base_offset == ...
assert region.target_base_offset == ...
assert region.inner_bytes == ...
assert region.outer_loop_counts == (...)
assert region.source_strides == (...)
assert region.target_strides == (...)
```

增加 bounds 测试，使用全部 outer dimensions 计算最大地址。

**Step 2: 运行 RED**

```bash
PYTHONPATH=mooncake-wheel:mooncake-model-weight/python python -m pytest \
  mooncake-model-weight/tests/test_model_weight_planner.py \
  -k "transfer_region or cross_dim" -q
```

预期：`TransferRegion` 尚不存在。

**Step 3: 最小实现**

新增：

```python
@dataclass(frozen=True)
class TransferRegion:
    tensor_id: str
    source: SourceFragment
    target: RuntimeFragment
    overlap_offset: tuple[int, ...]
    overlap_shape: tuple[int, ...]
    source_base_offset: int
    target_base_offset: int
    inner_bytes: int
    outer_loop_counts: tuple[int, ...]
    source_strides: tuple[int, ...]
    target_strides: tuple[int, ...]
```

实现：

- strict tuple/integer/rank validation；
- `segment_count`、`total_bytes`；
- N-D bounds；
- mixed-radix `iter_segments()`；
- 对 0/1 outer loop 的 CopyRange-compatible properties；
- `TransferPlan.regions` alias。

保留现有 `CopyRange`，让 manually constructed legacy plans 继续可执行。

**Step 4: 运行 GREEN**

```bash
PYTHONPATH=mooncake-wheel:mooncake-model-weight/python python -m pytest \
  mooncake-model-weight/tests/test_model_weight_planner.py -q
```

**Step 5: 提交**

```bash
git add mooncake-model-weight/python/mooncake/model_weight/planner.py \
  mooncake-model-weight/python/mooncake/model_weight/__init__.py \
  mooncake-model-weight/tests/test_model_weight_planner.py
git commit -m "feat(model_weight): define N-D transfer regions"
```

### Task 3: N-D coverage、intersection 和旧 TP 收敛

**Files:**
- Modify: `mooncake-model-weight/python/mooncake/model_weight/planner.py`
- Test: `mooncake-model-weight/tests/test_model_weight_planner.py`

**Step 1: 写失败测试**

覆盖：

- v1 TP4 -> TP8；
- v1 TP8 -> TP4；
- dim0 -> dim1；
- dim0 -> dim2；
- hole、overlap、out-of-bounds box；
- planner operation 数量等于 fragment intersections，而不是 segment 数。

每个测试通过模拟 global tensor 内容，按 region segments 执行 copy 后比较 target
与 global reference。

**Step 2: 运行 RED**

```bash
PYTHONPATH=mooncake-wheel:mooncake-model-weight/python python -m pytest \
  mooncake-model-weight/tests/test_model_weight_planner.py \
  -k "tp4_to_tp8 or tp8_to_tp4 or cross_dim or box_coverage" -q
```

预期：partition dimension mismatch 或 coverage 失败。

**Step 3: 最小实现**

实现：

```python
def _box_intersection(source, target) -> tuple[offset, shape] | None: ...
def _boxes_exactly_cover(shape, boxes) -> bool: ...
def _build_transfer_region(descriptor, source, target, offset, shape): ...
```

coverage 使用 volume + sweep-axis overlap detection；planner compatibility 不比较
source/target `partition_dim/shard_dims`。删除 1-D `_target_interval/_overlap/_copy_ranges`
主路径，让 v1 也经 N-D box 算法生成 region。

**Step 4: 运行 GREEN 和完整 planner 回归**

```bash
PYTHONPATH=mooncake-wheel:mooncake-model-weight/python python -m pytest \
  mooncake-model-weight/tests/test_model_weight_planner.py -q
```

**Step 5: 提交**

```bash
git add mooncake-model-weight/python/mooncake/model_weight/planner.py \
  mooncake-model-weight/tests/test_model_weight_planner.py
git commit -m "feat(model_weight): plan N-D logical-box overlaps"
```

### Task 4: PP、EP 和四轴组合

**Files:**
- Modify: `mooncake-model-weight/python/mooncake/model_weight/planner.py`
- Test: `mooncake-model-weight/tests/test_model_weight_planner.py`
- Test: `mooncake-model-weight/tests/test_weight_all_axes_native_e2e.py`

**Step 1: 写失败测试**

增加统一 fixture builder，显式给出每个 fragment 的 logical box 和 physical rank，
不通过参数名推断：

- PP2 -> PP4；
- PP4 -> PP2；
- EP8 -> EP2；
- EP2 -> EP8；
- source TP4/PP2/EP8/DP2 -> target TP8/PP4/EP2/DP4。

断言：

```python
assert {(g.source_pp, g.target_pp) for g in plan.pipeline_routes} == expected
assert every_target_box_is_exactly_covered(plan)
assert execute_plan_bytes(plan) == expected_target_bytes
```

**Step 2: 运行 RED**

预期：当前 `_complete_runtime_source_replicas` 把 EP 当 owner label，无法让多个 EP
fragments 共同覆盖一个 logical tensor。

**Step 3: 最小实现**

- source replica completeness 改为按 DP + logical tensor boxes；
- 相同 geometry 的 replica 只选择一个 deterministic physical fragment；
- target coverage 跨 TP/EP fragments 计算；
- 新增 `PipelineRouteGroup` 并按 `(src_pp, dst_pp)` 构建 indices；
- Store source route 使用 `source_pp=None`；
- 保留 executor lease snapshots 和 DP modulo mapping。

**Step 4: 运行 GREEN**

```bash
PYTHONPATH=mooncake-wheel:mooncake-model-weight/python python -m pytest \
  mooncake-model-weight/tests/test_model_weight_planner.py \
  mooncake-model-weight/tests/test_weight_all_axes_native_e2e.py -q
```

**Step 5: 提交**

```bash
git add mooncake-model-weight/python/mooncake/model_weight/planner.py \
  mooncake-model-weight/tests/test_model_weight_planner.py \
  mooncake-model-weight/tests/test_weight_all_axes_native_e2e.py
git commit -m "feat(model_weight): route PP and EP logical boxes"
```

### Task 5: Alias dedup 和 N-D physical range safety

**Files:**
- Modify: `mooncake-model-weight/python/mooncake/model_weight/planner.py`
- Test: `mooncake-model-weight/tests/test_model_weight_planner.py`

**Step 1: 写失败测试**

- 两个 N-D regions 写入相交 target segments 时失败；
- manifest 显式 alias 且 source geometry/region 完全相同时只复制一次；
- 仅 pattern/shape 相同但不是物理 alias 时不得 dedup；
- 多 outer loops 的最后 segment 越界时失败。

**Step 2: 运行 RED**

预期：旧 dedup identity 和 physical overlap 只理解一层 stride。

**Step 3: 最小实现**

dedup identity 使用完整 region geometry、source/target bases、counts 和 strides。
physical overlap 使用 lazy segment sweep，并设置计算预算，不能展开成常驻列表。

**Step 4: 运行 GREEN**

```bash
PYTHONPATH=mooncake-wheel:mooncake-model-weight/python python -m pytest \
  mooncake-model-weight/tests/test_model_weight_planner.py -q
```

**Step 5: 提交**

```bash
git add mooncake-model-weight/python/mooncake/model_weight/planner.py \
  mooncake-model-weight/tests/test_model_weight_planner.py
git commit -m "fix(model_weight): validate N-D physical writes"
```

### Task 6: TE bounded lowering

**Files:**
- Modify: `mooncake-model-weight/python/mooncake/model_weight/te.py`
- Test: `mooncake-model-weight/tests/test_model_weight_te.py`

**Step 1: 写失败测试**

- 2-D outer loop region 被正确复制；
- 每次 fake engine batch 长度不超过 `max_batch_operations`；
- planner/TE 不创建 segment-count 大小的 CopyRange list；
- `segment_count > max_region_segments` fail closed；
- stale source/target generation、registration mismatch、address bounds 继续失败；
- legacy CopyRange plan 继续执行。

**Step 2: 运行 RED**

```bash
PYTHONPATH=mooncake-wheel:mooncake-model-weight/python python -m pytest \
  mooncake-model-weight/tests/test_model_weight_te.py -q
```

**Step 3: 最小实现**

Sink/Reader 接受 `CopyRange | TransferRegion`，统一通过 lazy segment iterator
填充固定上界 arrays。构造函数增加 `max_region_segments`，在任何 backend 调用前
校验 region 上界、lease 和 registration。

**Step 4: 运行 GREEN**

```bash
PYTHONPATH=mooncake-wheel:mooncake-model-weight/python python -m pytest \
  mooncake-model-weight/tests/test_model_weight_te.py -q
```

**Step 5: 提交**

```bash
git add mooncake-model-weight/python/mooncake/model_weight/te.py \
  mooncake-model-weight/tests/test_model_weight_te.py
git commit -m "feat(model_weight): lower N-D regions in bounded batches"
```

### Task 7: Store manifest v2 和 Store source

**Files:**
- Modify: `mooncake-model-weight/python/mooncake/model_weight/store.py`
- Modify: `mooncake-model-weight/python/mooncake/model_weight/manifest.py`
- Test: `mooncake-model-weight/tests/test_weight_store.py`
- Test: `mooncake-model-weight/tests/test_weight_store_gpu_e2e.py`

**Step 1: 写失败测试**

- 多个独立 expert fragments 保存后仍是独立 Store objects/fragments；
- v2 manifest 保留 logical tensor、shard dims 和 box；
- Store -> dim1/dim2 target 内容一致；
- Store batch arrays 有界；
- v1 Store source 结果不变。

**Step 2: 运行 RED**

预期：旧 stored coverage 和 reader 只支持 1-D CopyRange。

**Step 3: 最小实现**

- Store writer 按 runtime fragments 原有 box 持久化，不 pack/all-gather；
- stored coverage 使用 N-D exact coverage；
- Store reader 使用同一 lazy region lowering；
- v1/v2 manifest publication、generation 和 group semantics 不变。

**Step 4: 运行 GREEN**

```bash
PYTHONPATH=mooncake-wheel:mooncake-model-weight/python python -m pytest \
  mooncake-model-weight/tests/test_weight_store.py \
  mooncake-model-weight/tests/test_weight_store_gpu_e2e.py -q
```

**Step 5: 提交**

```bash
git add mooncake-model-weight/python/mooncake/model_weight/store.py \
  mooncake-model-weight/python/mooncake/model_weight/manifest.py \
  mooncake-model-weight/tests/test_weight_store.py \
  mooncake-model-weight/tests/test_weight_store_gpu_e2e.py
git commit -m "feat(weight_store): persist N-D logical fragments"
```

### Task 8: SGLang runtime manifest v2 adapter

**Files:**
- Modify: `python/sglang/srt/model_executor/weight_runtime_manifest.py`
- Modify: `python/sglang/srt/model_executor/weight_semantics/qwen3_5.py`
- Modify: `python/sglang/srt/model_executor/weight_semantics/qwen3_next.py`
- Test: `test/registered/unit/model_executor/test_weight_runtime_manifest.py`
- Test: `test/registered/unit/model_executor/fixtures/qwen3_5_moe_runtime_manifest.json`

**Step 1: 写失败测试**

对 adapter 的结构化输出断言：

```python
assert view.tensor_id == "<framework-produced logical expert family id>"
assert view.global_shape == (num_experts, out, in_)
assert view.global_offset[0] == expert_id
assert view.local_shape[0] == 1
assert view.shard_dims == (0, tp_dim)
```

测试同一 physical parameter 的多个 views 保留不同 `byte_offset`，不分配新 buffer。

**Step 2: 运行 RED**

预期：现有 adapter 为每个 expert 生成独立 tensor ID 和 2-D logical tensor。

**Step 3: 最小实现**

- `LogicalTensorView`/`RuntimeWeightTensor` 增加 `shard_dims`；
- runtime manifest format 升级为 v2；
- Qwen adapters 在框架层输出 expert-family logical tensors；
- dense/legacy views 保持原 tensor IDs 和几何；
- 不在 Mooncake 中加入任何模型名匹配。

**Step 4: 运行 GREEN 和 SGLang 定向回归**

```bash
python -m pytest \
  test/registered/unit/model_executor/test_weight_runtime_manifest.py \
  test/registered/unit/model_executor/model_runner_components/test_remote_instance_weight_transporter.py \
  test/registered/unit/model_loader/test_remote_instance_heterogeneous.py -q
```

**Step 5: 提交到 SGLang 个人分支**

```bash
git add python/sglang/srt/model_executor/weight_runtime_manifest.py \
  python/sglang/srt/model_executor/weight_semantics/qwen3_5.py \
  python/sglang/srt/model_executor/weight_semantics/qwen3_next.py \
  test/registered/unit/model_executor/test_weight_runtime_manifest.py
git commit -m "feat(model_runner): export N-D logical weight boxes"
```

### Task 9: 集成、native 和真实推理验证

**Files:**
- Modify only if a test exposes a defect in the files above.
- Artifacts: `/data1/vin/<timestamped-validation-dir>` on H20.

**Step 1: Mooncake Python suite**

```bash
PYTHONPATH=mooncake-wheel:mooncake-model-weight/python python -m pytest \
  mooncake-model-weight/tests/test_model_weight_manifest.py \
  mooncake-model-weight/tests/test_model_weight_planner.py \
  mooncake-model-weight/tests/test_model_weight_te.py \
  mooncake-model-weight/tests/test_weight_store.py \
  mooncake-model-weight/tests/test_weight_all_axes_native_e2e.py -q
```

**Step 2: CUDA/native matrix**

在 H20 上运行：

- runtime G2G all-axis test；
- Store CUDA all-axis test；
- TP4 -> TP8、TP8 -> TP4；
- EP/TP dim0 -> dim1、dim0 -> dim2；
- four-axis source TP4/PP2/EP8/DP2 -> target TP8/PP4/EP2/DP4。

要求 byte-for-byte correctness、无越界、无 stale lease 绕过。

**Step 3: 真实推理 reuse benchmark**

使用相同模型、相同目标拓扑和相同 prompt 对比：

- cold start；
- manifest planning；
- live TE transfer；
- target load/refresh；
- first successful inference。

至少运行 3 次，报告 median/p95；校验 token IDs、文本、finish reason 和容差内
logprobs。权重复用总启动时间必须相对 cold start 有稳定且明显改善。

**Step 4: 服务连续性**

source 在 snapshot lease 有效期间持续处理请求；记录成功数、错误数、p95 latency，
确认 transfer 不要求 source 停服，更新写入在 lease 期间被 fencing。

### Task 10: 格式、review 和个人分支交付

**Files:**
- Review all changed files only.

**Step 1: 格式和静态检查**

```bash
pre-commit run --all-files --show-diff-on-failure
git diff --check origin/main...HEAD
```

如果全仓 hooks 修改无关文件，只保留本任务文件的格式结果。

**Step 2: 独立 review**

重点检查：

- v1 schema/behavior regression；
- N-D coverage 和 alias correctness；
- lease/generation/address bounds；
- Store publication semantics；
- segment explosion 和 CPU memory；
- Mooncake/SGLang 模型语义边界。

**Step 3: exact-SHA 复验**

在最终 Mooncake 和 SGLang commit SHA 上重跑定向 unit、native CUDA、Store 和真实
推理 benchmark；之前 SHA 的结果不能替代最终 SHA。

**Step 4: 推送个人 fork**

只推送：

- `Bo-Vincent/Mooncake` 的个人异构转换分支；
- `Bo-Vincent/vin-sglang` 的个人异构转换分支。

不创建、重开或更新社区 PR。最终交付 exact SHAs、测试命令、结果、benchmark
artifact 路径和已知限制。
