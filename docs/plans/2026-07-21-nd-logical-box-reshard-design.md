# N-D Logical-Box 异构权重转换设计

日期：2026-07-21
状态：已实现，并通过单元、Store、CUDA TE 和真实推理 smoke 验证

## 1. 目标

在现有 runtime manifest、WeightManifest 和 TransferPlan 基础上，将只支持
单一 `partition_dim` 的 1-D overlap planner 升级为 N-D logical-box planner。

设计需要同时满足：

- runtime manifest 是 live G2G 场景下权重物理位置和模型语义的唯一事实来源；
- source 和 target 可以沿不同 tensor dimension 分片；
- PP 按 manifest 给出的 layer/tensor ownership 路由；
- EP 是 logical expert tensor 的 leading coordinate；
- planner 输出 backend-neutral 的 N-D transfer region；
- TE 和 Store 对 region 做有界、懒惰 lowering；
- 旧 manifest、旧 TP、Store source 和现有 TE 路径保持兼容；
- 不依赖 NCCL M2N runtime、`ncclMemAlloc` 或模型参数命名规则。

## 2. 参考边界

借鉴 NVIDIA NCCL M2N `computeTransferPlan` 的三个通用思想：

1. source tile 和 target tile 都映射到同一个 global logical box；
2. 对每个维度计算 box intersection；
3. 把共同连续的 suffix 合并为 `innerSize`，其余维度保留为
   `outerCounts + src/dst strides`。

不复用以下 M2N 专属约束：

- 2-D rank mesh；
- 每侧只能有一个 SHARD placement；
- 对称 window 和相同 window offset；
- `ncclMemAlloc`、`ncclWindow_t` 或 M2N lifecycle；
- benchmark 中根据参数名推断 TP/EP 语义；
- benchmark 中根据 `layer * pp / num_layers` 推断 PP stage；
- benchmark 的 expert grouping 和 pattern dedup。

Mooncake 处理的是 manifest 已经声明好的 logical tensor 和物理 fragment，
不重新推断模型结构。

## 3. 当前限制

当前 `TensorDescriptor.partition_dim` 同时承担三个职责：

- 描述分片语义；
- 限制 fragment 只能沿一个维度变化；
- 驱动 planner 的 1-D interval overlap。

当前 `CopyRange` 只能表达：

- 一个连续 byte range；或
- 一层 `repeat + src/dst stride`。

因此它不能严谨表达：

- source dim0 到 target dim1/dim2；
- EP 和 TP 同时作用于 `[experts, out, in]`；
- 两层及以上 outer loops；
- 一个 logical tensor 由多个独立 expert allocation 共同覆盖。

## 4. Manifest v2

### 4.1 TensorDescriptor

`TensorDescriptor` 增加：

```python
shard_dims: tuple[int, ...] | None = None
```

兼容规则：

- v1 或 `shard_dims is None`：
  - `partition_dim is None` 等价于 `shard_dims=()`；
  - `partition_dim=d` 等价于 `shard_dims=(d,)`。
- v2 显式提供 `shard_dims`：
  - 允许零个、一个或多个维度；
  - 多维分片时要求 `partition_dim is None`，防止两个字段冲突；
  - 单维时可同时提供一致的 `partition_dim`，用于过渡期兼容。
- `shard_dims` 不参与 source/target logical tensor compatibility；source 和
  target 本来就允许使用不同 shard dimensions。

logical tensor compatibility 只比较：

- `tensor_id`；
- `global_shape`；
- `dtype` 和 `itemsize`；
- `layer_id` 和 logical expert family 语义；
- `layout_fingerprint`。

### 4.2 Fragment

`RuntimeFragment` 和 `StoredFragment` 已经具备 N-D box 所需字段：

```text
global_offset: box 在 logical tensor 中的起点
local_shape:   box 的各维 extent
address/object_key + object_offset: 物理位置
nbytes:        物理 allocation/view 的 byte 数
```

v2 校验规则：

- box rank 必须等于 `global_shape` rank；
- box 必须完全位于 global shape 内；
- 非 `shard_dims` 维必须覆盖完整 logical extent；
- `nbytes == product(local_shape) * itemsize`；
- runtime view 继续要求 canonical contiguous local layout。

这允许一个 fragment 同时沿 EP dim0 和 TP dim1/dim2 分片，而每个 fragment
仍然指向自己的独立 allocation。

### 4.3 EP leading coordinate

一个 expert family 由同一个 logical `tensor_id` 表示：

```text
logical shape: [num_experts, out, in]
```

例如 expert 7 的独立 allocation 可以声明为：

```text
global_offset = [7, out_offset, in_offset]
local_shape   = [1, local_out, local_in]
shard_dims    = [0, 1] 或 [0, 2]
```

Mooncake 不解析 `expert`、`gate_proj`、`down_proj` 等名字。把多个 runtime
参数映射到同一个 logical tensor、确定 leading expert coordinate，是 SGLang
等框架 adapter 的职责。

### 4.4 单一契约

- runtime inventory 直接提供 `partition_dim` 和可选的 `shard_dims`；
- `partition_dim` 是单轴 shorthand，`shard_dims` 表达 N-D logical box；
- `WeightManifest` 始终持久化 `shard_dims` 字段；
- Store manifest 是 runtime manifest 的持久化投影，不补充、猜测或改写模型语义；
- 当前接口不提供版本参数或旧 schema 兼容分支。

## 5. N-D TransferRegion

新增 backend-neutral `TransferRegion`：

```python
@dataclass(frozen=True)
class TransferRegion:
    tensor_id: str
    source: RuntimeFragment | StoredFragment
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

不在 region 中复制 address、endpoint、generation 等事实。它们仍由 source
和 target fragment 提供，避免形成第二份物理位置状态。

`TransferPlan.operations` 继续保留现有名称，但 planner 生成的元素升级为
`TransferRegion`。增加 `regions` 只读别名，便于新 executor 使用。

旧 `CopyRange` 类型继续保留，供旧调用方和定向测试构造。`TransferRegion`
对零层或一层 outer loop 提供旧 `source_offset/target_offset/nbytes/repeat/stride`
兼容属性，因此既有单维 TP 行为和断言不变。

## 6. Planner 算法

### 6.1 完整 DP replica

DP 不参与 tensor geometry。planner 对每个 source DP rank：

1. 按 `tensor_id` 收集 fragments；
2. 对完全相同的 logical geometry 去重，只选择一个物理 replica；
3. 校验每个 tensor 的 boxes 两两不重叠；
4. 校验 box volume 总和等于 global tensor volume；
5. 校验同一 DP replica 的 lease generation 一致。

只有覆盖全部 logical tensors 的 DP rank 才是可选 source replica。target DP
rank 继续按确定性的 modulo 规则映射到一个完整 source DP replica。

box overlap 校验使用 sweep axis，而不是按 element 建 bitmap，避免大模型下
内存随 tensor element 数增长。

### 6.2 Target coverage

非 local-target 规划要求每个 target DP rank 对每个 logical tensor完整覆盖。
coverage 按 logical owner 分组：PP 始终属于 owner；有独立 `expert_id` 的 tensor
同时按 EP owner 分组；expert-family tensor 则把 expert 作为 logical box 的 leading
coordinate。TP 以及 expert-family EP 的覆盖由 owner 内所有 boxes 的 union 决定，
不能把不同 PP 或独立 expert owner 的不完整 shard 拼成一份完整 tensor。

local-target 规划只要求当前 worker 声明的 fragments 能被 source 覆盖；全局
startup barrier 和其他 target workers 仍由框架控制。

### 6.3 Region 生成

对每个 target fragment：

1. 从选定 source DP replica 中选择相同 `tensor_id` 的 source fragments；
2. 逐维计算：

```text
overlap_begin[d] = max(src_begin[d], dst_begin[d])
overlap_end[d]   = min(src_end[d], dst_end[d])
```

3. 对非空 intersection 生成一个 region；
4. 校验所有 intersection boxes 不重叠，且 volume 总和等于 target fragment
   volume；
5. 不按 row、element 或 outer-loop iteration 创建 `CopyRange`。

### 6.4 Base offset 和 strides

source 和 target local tensors 都是 canonical row-major contiguous。对每侧：

```text
local_start[d] = overlap_offset[d] - fragment.global_offset[d]
base_offset    = sum(local_start[d] * local_stride[d]) * itemsize
```

从最后一维向前寻找 source 和 target 都共同连续的 suffix：

- suffix 内所有更低维都必须完整覆盖两侧 local extent；
- suffix 合并后的 byte 数为 `inner_bytes`；
- suffix 之前的 overlap extents 成为 `outer_loop_counts`；
- 对应 local byte strides 成为 `source_strides/target_strides`。

示例：`[experts, out, in]` 的 dim0 -> dim1：

```text
inner_bytes       = overlap_out * in * itemsize
outer_loop_counts = [overlap_experts]
source_strides    = [src_out * in * itemsize]
target_strides    = [dst_out * in * itemsize]
```

dim0 -> dim2：

```text
inner_bytes       = overlap_in * itemsize
outer_loop_counts = [overlap_experts, overlap_out]
source_strides    = [src_out * src_in * itemsize, src_in * itemsize]
target_strides    = [dst_out * dst_in * itemsize, dst_in * itemsize]
```

## 7. PP routing

PP ownership完全来自 fragment 的 `rank.pp` 和 descriptor 的 `layer_id/tensor_id`。
Mooncake 不计算 layer 到 stage 的映射。

`TransferPlan` 增加：

```python
PipelineRouteGroup(
    source_pp: int | None,
    target_pp: int,
    operation_indices: tuple[int, ...],
)
```

- runtime G2G 按 `(source.rank.pp, target.rank.pp)` 分组；
- Store source 没有 live source PP，使用 `(None, target.rank.pp)`；
- executor plans 继续保留 worker/rank/lease fencing；
- route group 只是调度索引，不复制 region 或地址。

## 8. TE 和 Store 有界 lowering

### 8.1 懒惰 mixed-radix iterator

`TransferRegion.iter_segments()` 使用 mixed-radix counter 按需生成：

```text
(source_base_offset + sum(index[d] * source_stride[d]),
 target_base_offset + sum(index[d] * target_stride[d]),
 inner_bytes)
```

planner 内不保存展开后的 segments。

### 8.2 两级上界

- `max_batch_operations`：单次 TE/Store backend 调用最多携带的 contiguous
  segments；
- `max_region_segments`：单个 region 允许 lowering 的 segment 总数，超过时
  fail closed，提示使用能够原生消费 N-D region 的 executor。

target 物理冲突校验不会把整个模型的 segment 总数当成固定拒绝阈值。对于
同一 runtime fragment，只有当所有 `TransferRegion` 都是 canonical geometry，
且 logical boxes 可证明完整、无重叠地覆盖 fragment 时，校验器才把它压缩成
一个 `[address, address + nbytes)` 区间；不同完整 fragment 之间直接检查这些
区间。无法作出该证明的手工或不完整 operations 仍使用有预算的 lazy segment
扫描并 fail closed。这样既不会因大模型累计超过 100 万 segments 而误拒绝，
也不会取消真实物理地址冲突校验。

因此 planner operation 数量只与 fragment-box overlap 数量相关，不与 row 或
element 数量相关；TE 的临时 host metadata 始终受 batch 上界限制。

planner 对每个 logical tensor 的 source boxes 建立纯几何 interval index，选择
source box 区分度最高的维度做候选查询，再调用完整 N-D overlap 和 coverage
校验。该索引不读取 tensor 名字，也不假设查询维度等于 target shard dimension；
因此 dim0 -> dim1/dim2 时会返回所有真实相交 source boxes，但不会让每个
target expert allocation 线性扫描全部 source experts。

对于本次基于 contiguous copy API 的 TE，跨维转换在数学上仍可能需要多个
实际 contiguous writes。该设计限制的是 planner 对象数量和 lowering 内存，
并对病态 region 设置硬上限；未来 N-D/M2N executor 可以直接消费 region，
把 outer loops 放到 GPU kernel 中执行。

### 8.3 未来 executor 预留

满足以下条件的 region 可被未来 live G2G N-D executor 直接消费：

- source 和 target 都是 `RuntimeFragment`；
- dtype、itemsize 和 logical layout 相容；
- lease/generation snapshot 有效；
- source/target address bounds 校验通过；
- executor 声明支持该 tensor rank 和 loop rank。

本次不加入 NCCL header、动态库检测、M2N communicator、symmetric allocation
或 backend-specific plan 字段。

## 9. 安全与一致性

以下校验不可因 N-D 支持而弱化：

- model ID 和 revision 一致；
- runtime lease generation 一致；
- executor snapshot 和 fragment inventory 一致；
- source/target memory registration lease 一致；
- 每个 region 的最大 source/target byte address 不超过 fragment `nbytes`；
- target physical ranges 不冲突，除非是 manifest 显式声明的物理 alias；
- Store object range 不超过 stored fragment；
- target logical coverage 完整且无重复写入。

N-D bounds 使用所有 outer dimensions 的最大 offset：

```text
max_offset = base + sum((count[d] - 1) * stride[d]) + inner_bytes
```

## 10. 框架边界

Mooncake core 负责：

- manifest v1/v2 schema 和严格校验；
- N-D box coverage/intersection；
- backend-neutral TransferRegion；
- DP replica selection、PP route grouping；
- TE/Store bounded lowering 和安全校验。

SGLang 等框架 adapter 负责：

- 从 runtime parameter/view 得到 `tensor_id`、`global_shape` 和 box；
- 把同一 expert family 映射成 leading-coordinate logical tensor；
- 声明 `layer_id`、`shard_dims` 和 `layout_fingerprint`；
- 提供真实 PP/EP/TP/DP rank ownership；
- 保持 parameter owner 活着并协调 snapshot lease。

Mooncake 不包含 Qwen、DeepSeek、q_proj、down_proj 或 expert 参数名规则。

### 10.1 live G2G snapshot 生命周期

一次 target model world 只获取一份 source snapshot：

1. target global rank 0 调用 source control plane 获取 manifest、`transfer_id`
   和 lease timeout，并在启动后台线程前同步续租一次；
2. rank 0 将同一 session 广播给所有 target ranks，各 rank 只根据自己的
   target manifest 生成和执行局部 plan；
3. 每个 rank 即使本地 manifest/plan 失败，也必须到达一次固定的 world
   readiness gate；rank 0 把 heartbeat 健康状态并入该 gate，只有全部 rank
   ready 才允许任何 rank 开始 TE DMA；
4. 各 rank 汇总 `(transfer_success, release_safe)`，只有全部 rank 都能证明
   TE 调用完成后，rank 0 才停止 heartbeat 并释放 source lease 一次；
5. 任一 rank 无法证明完成、collective 失败或 TE 返回不确定状态时，不主动
   释放 source lease，由 TTL 保留 source allocation；
6. source 的权重更新、offload 和 pointer 失效操作先在每个 rank 预留本地
   snapshot update token，再做 world collective；全员成功才进入含 barrier 的
   mutation body，否则取消已取得的 token 并让所有 rank 一致失败；
7. source manager 同时记录 transfer deadline，scheduler 周期性清理已过 TTL
   的 `transfer_id -> lease_id` bookkeeping；实际 lease 仍由 snapshot
   coordinator 的 generation/TTL 语义裁决。

这套控制面只用于 live runtime source。Store source 使用已发布 manifest 中的
object generation 和 range bounds，不需要 source runtime heartbeat，也不改变
Store 的 publication 语义。

### 10.2 当前 TE lowering 与未来 executor

本次 live G2G 使用同步 `execute`，并把每个 backend batch 限制为最多 8192 个
contiguous operations。这个选择不是 planner 限制：当前 Python async binding
在部分提交或状态查询失败时不能稳定返回可等待的 batch handle，也就无法证明
DMA 已经终止；此时注销或复用 source allocation 会破坏 lease 安全条件。

未来 async 或 M2N lowering 必须提供有所有权的执行 handle，并满足：

- 部分提交失败仍返回可追踪 handle；
- 能等待到成功或失败的 terminal 状态；
- terminal 后显式释放 handle，并在释放失败时 fail closed；
- executor 继续消费同一个 backend-neutral `TransferRegion`，不修改 manifest
  或 planner，也不要求 `ncclMemAlloc`。

因此当前同步 TE、未来安全 async TE 和未来 M2N executor 是同一 plan 的不同
lowering，而不是三套模型语义或三套 reshard planner。

### 10.3 当前性能证据

H20 上 Qwen3-Next、source TP4/EP4、target TP2/EP2、RDMA G2G 的单轮 smoke：

- 38,057 个 compact regions，lowering 为 431,081 个 contiguous segments；
- profile 定位旧实现 `_overlap_box` 调用约 1,887 万次；
- interval index 后 plan 从约 69.4 秒降到 5.1 秒；
- manifest target spawn-to-ready 69.43 秒，cold checkpoint 108.78 秒，约
  1.57x 加速；
- source 在两次 target 启动期间持续推理，0 error、0 mismatch；target 与
  cold baseline 的文本、token IDs、token 数、finish reason 完全相同，
  token logprobs 在 1e-4 绝对/相对误差内一致。

这是单轮 smoke，不替代最终多轮提交 SHA benchmark；但它证明 planner 不再
被 expert candidate 的笛卡尔扫描主导。

## 11. 验收矩阵

必须覆盖：

1. v1 TP4 -> TP8 和 TP8 -> TP4，region/legacy fields 与现状一致；
2. PP2 -> PP4 和 PP4 -> PP2，route group 与实际 layer owner 一致；
3. EP8 -> EP2 和 EP2 -> EP8，expert leading coordinate 完整覆盖；
4. `[experts, out, in]` dim0 -> dim1；
5. `[experts, out, in]` dim0 -> dim2；
6. source TP4/PP2/EP8/DP2 -> target TP8/PP4/EP2/DP4；
7. 每个 target logical tensor byte-for-byte 内容一致；
8. v1 JSON manifest round-trip；
9. v2 RuntimeManifest -> Store -> Runtime round-trip；
10. runtime G2G source、Store source、local target 和 alias dedup 回归；
11. stale generation、invalid registration、address overflow 继续失败；
12. planner region 数量不随 row/element 展开；
13. backend batch 不超过 `max_batch_operations`；
14. region 超过 `max_region_segments` 时 fail closed；
15. 最新分支上的真实推理权重复用 correctness、服务连续性和冷启动对比。

当前验证结果（H20，2026-07-22）：

- Mooncake 权重单元/集成：253 passed，12 skipped；
- native CUDA Store：TP split/merge、四轴组合、8 个独立 expert allocation
  cross-dim 共 3 passed；
- RDMA CUDA TE：all-axis Sink、packed Reader、cross-dim Sink/Reader 共
  4 passed；
- SGLang manifest、planner adapter、loader lifecycle 和 benchmark harness：
  114 passed；
- Qwen3-Next TP4/EP4 -> TP2/EP2 live G2G 单轮 smoke：结果与 cold baseline
  一致、source 服务持续可用，spawn-to-ready 约 1.57x 加速。

真实 Qwen3-Next 当前 adapter 要求首尾 PP stage 同 rank，因此 PP2 <-> PP4 和
包含 DP/PP 的四轴组合以 synthetic/native CUDA 数据验证；这属于模型 runtime
adapter 的当前限制，不是 planner 的 logical-box 或 PP route 限制。量化模型若
runtime 无法导出满足校验的 canonical manifest，同样 fail closed，不按参数名
在 Mooncake core 中猜测或修补。

## 12. 交付边界

- Mooncake 和 SGLang 分别提交到个人 fork 分支；
- 不创建、重开或更新社区 PR；
- 完成 Python unit/integration、Store、native TE/CUDA 和真实推理性能验证；
- 最终提供 exact SHA、命令、测试结果、性能数据和仍然受限的 runtime 组合。
