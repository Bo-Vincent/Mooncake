# 双侧 Placement / Runtime Binding 实施记录

日期：2026-07-23

## 完成定义

转换层必须能够仅根据 source placement 和 target placement 生成
`LogicalTransferPlan`。逻辑计划不得持有任何运行时地址或 lease；source/target 地址
均在 bind 阶段注入。旧 one-shot RuntimeManifest、Store 和 TE 路径保持可用。

## 实施内容

### 1. 角色中立 manifest

- 将 placement schema 提升为 `PlacementManifest`；
- 保留 `TargetPlacementManifest`，新增 `SourcePlacementManifest` 兼容公开名；
- `RuntimeBindingManifest` 继续由 source/target 共用；
- 新增 runtime manifest 到 placement/binding 的兼容投影函数；
- 旧 runtime 缺少 placement ID 时只生成 instance-scoped ID，不宣称跨重启稳定。

### 2. 纯逻辑 planner

- 新增 `plan_placement_transfer`；
- 新增 `plan_placement_transfer_to_local_target`；
- source/target operations 都使用 `PlacementFragment`；
- source/target executors 都使用 `PlacementExecutorPlan`；
- DP coverage 只读取 logical geometry/rank；
- PP route 同时支持 source placement；
- Runtime/Store one-shot planner 保持原接口。

### 3. 双侧 late binding

- `bind_logical_transfer_plan` 接收 target bindings，并通过 `source_bindings` 接收
  source bindings；
- binding 输入可为 `RuntimeBindingManifest` 或已绑定的 `RuntimeManifest`；
- 双侧均严格校验 placement identity、fragment set、geometry、rank、aliases 和
  nbytes；
- source 跨 DP generation 不一致时 fail closed；
- 绑定后重新生成 source/target `RuntimeFragment`、executor lease snapshots 和
  pipeline routes；
- Store source 不要求 source binding。

### 4. Runtime projection

- `plan_runtime_transfer_to_target_placements` 和 local 版本把 runtime source
  投影成 placement 后再规划；
- logical plan 不含 source 地址；
- 执行时调用方显式把 source runtime manifests 传给 `source_bindings`；
- 最终 `TransferPlan` 与直接 runtime planner 逐 operation、executor 和 route 等价。

### 5. SGLang source 协议接入

- source runtime 支持返回 `placement_binding`，分别导出稳定的
  `source_weight_placements` 和本次 snapshot 的
  `source_weight_runtime_bindings`；
- source scheduler 仍对每个 rank 创建并持有 address-stable lease，split 只改变
  控制面表达，不改变 source GPU allocation 生命周期；
- tokenizer 控制面校验多 DP placement 语义一致、placement/binding 一一对应、
  fragment ID/nbytes、worker 唯一性和 generation 一致性；
- target loader 先用双侧 placement 生成逻辑计划，再绑定双侧 runtime binding，
  最后仍把完整 runtime manifest 交给既有 TE 注册和执行路径；
- client 和 server 使用同一套无版本 placement/binding 契约，不提供旧 schema
  自动回退。

### 6. 参数与元数据大小

- planner 的必选参数由“source full runtime manifest + target placement”收敛为
  “source placement + target placement”；address、endpoint、lease、generation 等
  动态字段不再进入逻辑规划；
- binder 仍必须得到 source/target runtime binding，因此系统所需事实没有减少，
  只是按生命周期拆开；
- 若一次 RPC 同时发送 placement 和 binding，首轮线上的总字节数通常与旧 full
  manifest 接近，甚至会因关联 ID 略有增加；
- placement 可按 `placement_id` 缓存；当前 SGLang 会把 weight generation 编入
  `revision`，因此本版主要复用同一 generation 内的重试和地址重绑；若要跨权重
  generation 只刷新 binding，还需把稳定 layout revision 与动态 content generation
  进一步分离；
- logical plan 可按 model revision、source placement IDs 和 target placement IDs
  缓存，target 地址尚未分配时也可提前规划。

## 验证矩阵

定向 pytest 覆盖：

- placement JSON/digest 与旧 `partition_dim`；
- source/target 双侧无 runtime location；
- source address/instance/worker/generation 变化只触发 rebinding；
- source binding 缺失、错误 placement、nbytes、ownership 和 generation；
- TP4 -> TP8、TP8 -> TP4；
- PP2 -> PP4、PP4 -> PP2；
- EP8 -> EP2、EP2 -> EP8；
- `[experts, out, in]` dim0 -> dim1/dim2；
- TP4/PP2/EP8/DP2 -> TP8/PP4/EP2/DP4；
- Store source、TE reader/sink、旧 manifest 和 old planner；
- operation 数量和逐字节内容验证。
- SGLang 当前 Qwen3.5 MoE adapter 生成的 v2 golden 与 Mooncake fixture
  完全一致；
- SGLang TP4/PP2/EP4/DP2 source placement/binding 到
  TP2/PP4/EP2/DP4 local target 的 plan、bind、lease、TE read 和逐 tensor
  内容验证。

Mooncake model-weight 完整定向矩阵：

```bash
SGLANG_CONTRACT_REQUIRED=1 \
SGLANG_SOURCE_ROOT=/path/to/sglang \
PYTHONPATH=mooncake-wheel:mooncake-model-weight/python \
python3.12 -m pytest -q mooncake-model-weight/tests/test_weight*.py
```

当前结果：`319 passed, 20 skipped, 8 subtests passed`。严格模式下缺少
`SGLANG_SOURCE_ROOT` 会直接失败，不会把跨仓门禁静默记为 skip。skip 项是本机缺少
GPU/RDMA 运行条件的 native cases；逻辑、Store mock、TE mock 和基于真实 SGLang
Qwen adapter 的 CPU memory-copy contract 均已执行。

SGLang 当前 checkout 的轻量定向验证：

- runtime manifest、Qwen semantics、placement/binding manager 和 update fence：
  `74 passed`；
- source HTTP client、格式协商、world broadcast、heartbeat 和失败释放：
  `25 passed`。

Mooncake heterogeneous benchmark harness 为 `168 passed`；两仓改动均通过
`ruff format --check`、`ruff check` 和 `git diff --check`。

远端 `vin` 的 `mooncake-pro-dev` 容器重新构建 `engine` 和
`transfer_engine_completion_test` 后，completion CTest `1/1` 通过，Python 扩展可
导入 read/write ticket API。当前本地与该容器都不包含完整 PyTorch SGLang 开发
环境，因此 SGLang 定向测试使用轻量 package bootstrap 屏蔽与目标模块无关的可选
import；实际被测源码和断言均来自当前 checkout。这不能替代后续真实
SGLang + GPU/RDMA E2E。

## 后续远端验证

在 GPU/RDMA 机器上继续运行：

1. source/target placement 由真实 SGLang runtime 导出；
2. source/target binding 指向真实 CUDA allocation；
3. direct G2G 对 target tensor 做逐字节校验；
4. Store `get_into_ranges` 对 target tensor 做逐字节校验；
5. source stale generation、lease 失效和注册范围越界必须在执行前失败。

远端验证不改变本次 planner/binder 契约，也不要求接入 NCCL M2N runtime。
