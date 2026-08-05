# 源/目标 Placement 与 Runtime Binding 分层设计

日期：2026-07-23

## 目标

异构权重转换的逻辑层只回答一个问题：给定 source placement 和 target
placement，哪些 source logical boxes 应当写入哪些 target logical boxes。

GPU 地址、worker、endpoint、generation 和 lease 不参与 N-D overlap 计算，只在
执行前绑定。最终形成下面的稳定分层：

```text
Source Placement + Target Placement
                 |
                 v
       LogicalTransferPlan
                 |
Source Binding + Target Binding
                 |
                 v
           TransferPlan
                 |
          TE / Store executor
```

## 数据契约

### PlacementManifest

`PlacementManifest` 是 source/target 共用的角色中立契约：

```text
resource_id, revision, placement_id
tensors[]:
  tensor_id, global_shape, dtype, itemsize
  shard_dims
  parallel_axes: split / replicated / ownership
  layer_id, expert_id, layout_fingerprint
fragments[]:
  placement_fragment_id
  tensor_id, global_offset, local_shape, nbytes
  rank(dp, tp, pp, ep), aliases
```

它不包含 address、instance、worker、endpoint、generation、lease 或 owner。
`SourcePlacementManifest` 和 `TargetPlacementManifest` 都指向该角色中立契约，
不会产生两套 schema。

每个 framework participant 先独立生成 `WeightPlacementPart`，collection barrier
再组装一份完整 `WeightPlacementManifest`。全局 contract 校验 participant 集、
split rank participation、每个 replica/owner 的独立完整覆盖以及未声明 axis，
planner 因而只接收一份完整 source placement 和一份完整 target placement。

### RuntimeBindingManifest

binding 只描述某次运行实例的物理位置和生命周期：

```text
resource_id, placement_id, placement_digest
instance_id, generation, lease_id
fragments[]:
  placement_fragment_id
  fragment_id, address, nbytes
  worker_id, endpoint, owner
```

`WeightPlacementManifest + WeightRuntimeBindingManifest` 经严格校验后绑定为
planner 私有的 `BoundWeightFragment` 执行视图。该对象不可序列化，也不是第三种
manifest。

### LogicalTransferPlan

逻辑计划保存 source/target placements 和 backend-neutral N-D regions：

- overlap offset/shape；
- source/target base offset；
- inner bytes；
- outer loop counts；
- source/target strides；
- `(source_pp, target_pp)` route；
- source/target placement executor 分组。

逻辑计划的完整对象图中不得出现 `BoundWeightFragment`、
`RuntimeLeaseSnapshot`、地址、endpoint、worker、generation、lease 或 owner。

## Planner

主入口为：

```python
logical = plan_placement_transfer(source_placement, target_placement)
```

单 target executor 启动时使用：

```python
logical = plan_placement_transfer_to_local_target(
    source_placement,
    target_placement,
    target_participant_id,
)
```

TP 和 expert-family EP 由 N-D logical box overlap 统一处理。独立 expert tensor
继续使用 `expert_id` 和 EP owner；PP 由 tensor/layer owner 和
`(src_pp, dst_pp)` route 处理。DP 只选择完整 source replica，不产生额外权重副本。
完整规划要求 source 至少存在一个完整 DP replica，并要求每个 target DP/tensor
owner 完整覆盖；hole、部分重叠或跨 owner 拼接都会 fail closed。local-target
规划只验证当前 executor 的 fragments，集群完整性由框架 startup barrier 保证。

placement planner 不读取 generation 或 worker，因此同一 placement 可跨地址变化复用。

## Runtime Binding

绑定入口同时接收 source 和 target runtime bindings：

```python
plan = bind_logical_transfer_plan(
    logical,
    target_bindings,
    source_bindings=source_bindings,
)
```

binder 必须完成：

1. model、revision、placement ID 和 fragment 集合完全匹配；
2. nbytes、logical geometry、rank 和 aliases 不能被 runtime 改写；
3. source 各 DP replica 的 lease generation 必须一致；
4. source/target `BoundWeightFragment` 和 executor lease snapshot 重新构建；
5. N-D operation bounds、target alias dedup 和物理地址冲突检查继续执行；
6. pipeline routes 必须覆盖全部 operations。

地址变化只重新运行 binding，不重新计算 overlap。binding 后的 `TransferPlan` 仍包含
TE/Store 所需的地址与 fencing 信息。

## 公开路径

- 逻辑规划只暴露 `plan_placement_transfer`、
  `plan_placement_transfer_to_local_target` 和 Store placement planner。
- 运行时执行只接受独立的 `WeightRuntimeBindingManifest`；没有 combined runtime
  manifest、runtime projection 或旧 schema 自动回退。
- Store 的持久化 `WeightManifest` 继续可用，但它与 runtime binding 属于不同
  生命周期和存储语义。
- Store source 没有 source runtime binding，仍只绑定 target，并继续走
  `get_into_ranges`。
- 直接 G2G 最终仍走现有 TE batch read/write，不改变数据面 API。

## 能力协商

框架不得通过“模块能 import”推断 split manifest 可用。Mooncake wheel 暴露机器可读
capability，SGLang 同时据此选择 source 请求格式和 target builder：

- 支持 `placement_binding`：source/target 都使用 placement + binding；
- placement/binding 不能只在 source 侧启用，否则 target 会在 planning 后晚失败；
- capability 不暴露 backend 专用的 `transfer_scatter`；executor 在运行时探测
  scatter binding，不可用时回退到原有 batch read/write；
- 当前接口不接受版本参数，也不提供旧 schema 自动回退。

## TE Lowering

默认执行器保留现有 bounded batch read/write。绑定了 Mooncake PR #3000 接口时，
lowering 会保留 manifest fragment 的 allocation base/capacity，将同一 allocation
pair 内的 segment 压缩为 scatter range，并使用 completion-fenced scatter ticket。
接口不可用时，同一 batch 确定性展开为原有绝对地址数组。

它主要适合 target-initiated、多 source fan-in 的 TP/EP merge；对单 source split 或
连续 region 不保证更快。planner、manifest 和 Store schema 都不依赖该 API，未满足
条件时必须无损回退到 batch TE，不能自动重试完成状态未知的 batch。

## 框架边界

SGLang 等框架负责输出 tensor semantic、logical box、PP/EP owner、稳定 placement
fragment ID，并在 runtime 地址就绪后输出 binding。Mooncake 不根据 `q_proj`、
`down_proj` 或 expert 名称猜模型语义。

框架显式提供的 `placement_fragment_id` 必须跨进程重启保持稳定；同一 logical
layout 不能在显式 ID 和派生 ID 之间切换。框架 adapter 还必须把
`numpy.int64` 等整数标量归一化为 Python `int`，Mooncake contract 会拒绝
`bool` 和其他非 `int` 类型。

同一套 placement/binding 可以用于 source 或 target，因此框架无需维护两套模型
adapter。Mooncake planner 只处理角色中立的逻辑几何，TE、Store 和未来 M2N lowering
共享同一个 `LogicalTransferPlan`。

runtime binding 的 address 必须是可传输 view 的正地址；`0` 保留为 null sentinel。
binding 后仍执行 unsigned 64-bit address bounds、registration、lease 和 generation
校验。

## 验收边界

1. source/target placement 均不含 runtime location；
2. source/target placement 可以独立生成完整 `LogicalTransferPlan`；
3. 同一 logical plan 可绑定不同 source instance/address/generation；
4. source binding 缺失、placement 不匹配、nbytes 不一致或跨 DP generation 不一致时
   fail closed；
5. TP、PP、EP、cross-dim 和四轴组合与旧 one-shot runtime planner 等价；
6. Store source、旧 manifest、TE lease/address bounds 和 operation-count guardrail 回归
   不变。
