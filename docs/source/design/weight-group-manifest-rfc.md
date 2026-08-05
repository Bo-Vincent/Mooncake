# RFC: 基于 Weight Manifest 的异构并行权重复用

状态：Draft，实验性 POC，尚未进入发布版本
作者：Mooncake contributors
最后更新：2026-08-06

## 摘要

本文提出一套以框架语义为入口、以 Mooncake 为数据平面的模型权重复用方案。
框架的每个 participant 把运行时参数解释为地址无关的
`WeightPlacementPart`。控制面把同一模型、revision、weight generation 和
placement set 的 parts 聚合为一份完整 `WeightPlacementManifest`；每个实际
参与传输的进程再独立导出带 lease 的 `WeightRuntimeBindingManifest`。
Mooncake 按全局 Tensor 区间一次性规划 DP、TP、PP、EP 变化，并把同一个逻辑
计划绑定、编译到：

- Mooncake Store：持久化权重，并从内存或后续分层存储加载到目标地址。
- Mooncake Transfer Engine：从 source GPU 直接传到 target GPU。
- 测试用 Host backend：验证计划正确性，不属于生产数据面。

POC 已实现 Manifest、四轴联合 Planner、Store backend、TE backend，以及
SGLang 的 Qwen3/Qwen3.5 文本、静态 MoE、VL 适配和 canonical serialized
FP8 的源码级 adapter contract。SGLang 已接入跨 worker 的
begin/renew/release 控制面和 snapshot lease 生命周期；集群级 source
discovery、目标 revision 原子激活和 slime 调度接线仍属于框架或外部控制面
工作。

## 调研结论

### slime 当前权重更新

调研基线为 slime `origin/main@50f2d944`（2026-07-19）。Megatron 侧当前按
参数执行 TP all-gather，再做 HF 名称/布局转换，并按
`update_weight_buffer_size` 组成 bucket；Expert 参数还会在一个有界 batch
内做 EP gather。因此：

- 临时空间是“当前参数的 TP shards + 拼接结果 + 当前 bucket”，不是整模型。
- full Tensor 或 bucket 当前位于 GPU，生命周期结束后即可释放。
- 这条路径适合训练框架先还原参数，再交给 SGLang 自己切到目标 TP。
- PP/EP owner、模型 canonical 名称和 source/target address 没有形成可持久化的
  模型级控制面。

这套 POC 不删除 slime 的现有路径。后续 slime 可以选择继续先 gather，也可以
直接把训练侧 shard 导出为 placement 和 runtime binding，让 Mooncake 按
range 传输。

### Mooncake 当前基础

调研基线为 Mooncake `origin/main@5d87a43d`（2026-07-18）。现有统一并行
Tensor IO 已有 `ParallelAxis`、`ReadTarget` 和 `get_into_ranges`，能够把
source shard 和 target shard 映射到同一个全局 split dimension。例如
TP4 rank0 覆盖 `[0, N/4)`，TP8 rank0/rank1 分别覆盖其前后两半，区间 overlap
即可生成 split plan；反向则生成 merge plan。

这解决的是单 Tensor 的 TP 区间重建。它原本不知道：

- Group 代表哪个模型 revision。
- 完整 revision 应包含哪些 Tensor 和 Fragment。
- PP stage 和 EP owner 如何变化。
- GPU address 属于哪个 worker generation。
- 多个 Payload 何时共同构成可见 revision。

Mooncake Group semantics 提供对象的物理组织、路由、续租和回收，但 Group
不是多 Object 事务。Weight Manifest 在其上增加模型语义和可见性提交点，
而不是替代 Group。

### 可行性结论

同模型、同 dtype、同 canonical layout 下，异构 DP/TP/PP/EP 可以完成。
关键不是 rank 数互为倍数，而是 source 和 target 对每个 logical Tensor 的
global range 能完整覆盖且 layout 兼容：

- DP 是副本选择和 fan-out，不改变 Tensor 字节。
- PP 是 layer owner 变化，不改变单层 Tensor 字节。
- EP 是 global expert owner 变化，不改变单 Expert Tensor 字节。
- TP 是 split dimension 上的区间拆分或合并。

TP2 到 TP3 这类非整数倍变化，只要每个 shard 的边界可表示并完整覆盖，也能
由 overlap 规划；如果框架要求等分且维度不能整除，则应由框架适配器拒绝。

## 设计目标

1. 同一份语义模型支持 Runtime 到 Store、Store 到 Runtime、Runtime 到
   Runtime。
2. 一次规划同时处理 DP、TP、PP、EP，不创建串行中间布局。
3. 不要求整模型或整 Tensor all-gather。
4. GPU 地址只存在于带 lease 的临时 runtime binding。
5. 模型和框架规则留在 SGLang、vLLM、slime 等框架侧。
6. 普通 KVCache Store 和普通 TE 调用不经过本模块。

模型权重语义、规划器以及 Store/TE 调用适配器由顶层
`mooncake-reshard` 模块维护。通用传输执行和物理存储管理仍分别属于
`mooncake-transfer-engine` 与 `mooncake-store`。

首版可传输框架已解释为 canonical byte view 的序列化 FP8 Tensor，但不处理
在线量化、量化 recipe 或 dtype 转换、转置、未知 fused/packed layout、动态
EPLB 或跨模型架构转换。

## 分层架构

```text
Framework model semantics adapter
  canonical name / shape / layer / expert / split axis / byte view
                         |
                         v
WeightPlacementManifest             WeightRuntimeBindingManifest(s)
  topology / participant parts        participant / worker / endpoint
  tensor semantics / N-D boxes        GPU address / nbytes / generation / lease
  PP/EP/TP/DP ownership
                  \                    /
                   \ validated bind  /
                    v               v
                BoundWeightFragment
                (planner-private execution view)
                         |
                         v
Mooncake validation and planner
  exact tensor set / coverage / overlap / DP source selection / fencing
                         |
                         v
TransferPlan + per-executor subplans
              |                         |
              v                         v
Store backend                          TE backend
Payload + WeightManifest              GPU address -> GPU address
```

这不是修改 Store/TE 默认路径的全局开关。只有调用
`mooncake.reshard.weight` 并提供 Manifest 时才进入该模块；KVCache 仍使用
原有 key/value 和 disaggregated KV 路径。

### 框架必须负责

- 在 load、量化、TP、EP、LoRA 等最终布局稳定后解释 resident parameters。
- 把运行时名称映射为稳定的 canonical tensor ID。
- 提供 global shape、global offset、local shape、split dimension、layer ID、
  expert ID 和 layout fingerprint。
- 提供 target placement；target 当前实际持有哪些参数是 PP/EP owner 的权威
  来源，并在目标 allocation 就绪后独立提供 runtime binding。
- 在 source lease 期间固定参数地址和内容，在 target 激活前固定目标地址。
- 对所有 worker 使用同一个全局 revision，并在全部 worker 更新成功后才导出。
- 完成目标校验、barrier 和 revision 激活。

### Mooncake 必须负责

- 严格校验 schema、Tensor 集合、Fragment coverage、placement digest 和布局
  兼容性。
- 选择一份 DP source，加载时向每个 target DP fan-out。
- 按 global range 生成 CopyRange 和每个 worker 的子计划。
- 校验 worker、address、nbytes、generation 和 registration lease。
- 将计划编译为 Store ranged IO 或 TE batch transfer。
- 用 Group 保存 Weight Payload 和最终 Weight Manifest。

### 可选的框架能力

- SGLang worker RPC：远程获取 placement，并获取/续租/释放 runtime binding；
  POC 已实现对应的 snapshot 生命周期基础设施。
- K8s/Ray/slime 控制面：发现 source、收集 Manifest、选择 Store/TE/fallback。
- target 双 buffer 或新实例切换：实现不中断的原子 revision 激活。
- 模型家族 adapter：Qwen、DeepSeek、Llama 等可以独立合入各框架社区。

## Manifest 数据模型

### ParallelTopology 与 WeightPlacementPart

`ParallelTopology` 描述完整的 `tp_size/pp_size/ep_size/dp_size`，以及本次
placement 明确选择的 `(participant_id, rank)` 集合。`world_size` 是显式
participant 数量，不是四个轴 size 的乘积；框架可以让 TP/EP 共用 worker，也
可以只选择一套完整 DP replica，同时保留 runtime 的真实 `dp_size`。

每个 worker 导出一个 `WeightPlacementPart`：

```text
resource_id / revision / weight_generation
placement_set_id / topology_id
participant_id / dp / tp / pp / ep rank
tensors[] / fragments[]
```

`participant_id` 必须来自稳定的逻辑 placement 身份，不能使用会随 session、
transfer 或进程重启变化的 runtime `instance_id`。part 可以为空，例如某个 PP
participant 不拥有任何参数；空 part 不需要 runtime binding。

### WeightPlacementManifest

一份 `WeightPlacementManifest` 表示一整个模型权重 generation 的完整逻辑
placement，而不是单进程 shard：

```text
resource_id / revision / weight_generation / placement_set_id
topology
parts[]
derived tensors[] / derived fragments[]
canonical placement_id / digest
```

聚合时必须精确匹配 topology participant 集，并校验所有 parts 属于同一模型、
revision、generation、placement set 和 topology。每个被选择的 DP replica 必须
独立完整覆盖全部 logical Tensor；多个 PP owner 可以各自持有完整副本，但不能
把不同 PP owner 的残缺片段拼成伪完整 Tensor。

### TensorDescriptor

```text
tensor_id
global_shape
dtype / itemsize
shard_dims
parallel_axes[] -> split(kind, dim) / replicated(kind) / ownership(kind)
layer_id / expert_id
layout_fingerprint
```

`shard_dims` 是 logical box 可变化的维度集合；`parallel_axes` 明确区分 collective
split、独立完整 replica 和 owner-only placement。框架 adapter 必须给出这些
typed semantics，Mooncake 不接受 `partition_dim`、可空 `split_dim` 或别名字段。
即使某个 runtime rank 当前持有完整 Tensor，它仍可保留 canonical split axis，
并用 `global_offset=0, local_shape=global_shape` 表示全量 fragment。

### PlacementFragment 与 RuntimeBindingFragment

```text
PlacementFragment:
placement_fragment_id / tensor_id
global_offset / local_shape / nbytes
aliases / dp / tp / pp / ep rank

RuntimeBindingFragment:
placement_fragment_id / fragment_id
address / nbytes / worker_id / endpoint / device
```

`WeightRuntimeBindingManifest` 只绑定一个 `participant_id`，通过全局 placement
ID 与 digest 证明它属于哪一份完整 placement，并包含 `instance_id`、
`generation` 和 `lease_id`。完整逻辑 placement 可以先于任意 target GPU
allocation 生成；真正执行时只为 plan 引用的非空 participants 提供 bindings。
binding 是小体积控制面 metadata，可以放在内存、RPC 消息或临时控制面中；
不得把 GPU 地址持久化到 Store。逻辑 planning 只读取 placement，真正执行前才
把 placement 和 binding 绑定为 planner 私有的 `BoundWeightFragment`；不存在
第三种可序列化的 combined runtime manifest。

### WeightManifest

```text
namespace / resource_id / revision / weight_generation
group_id / manifest_key
tensors[]
stored_fragments[] -> object_key / object_offset / nbytes
created_at
```

Weight Manifest 不含 endpoint、GPU address 或 Runtime lease。它描述一份
不可变 revision 的完整成员集合，是 Store 中的 READY 提交点。

### TransferPlan

Planner 输出 logical operations 和每个 source/target executor 的子计划。
只有被 operation 引用的 participant 才进入 executor 子计划；每个实际 executor
在执行时仍核对 Fragment、address、generation 和 lease 快照。未拥有 Tensor 的
空 participant 不要求虚构 GPU 地址或 runtime binding。

## 四种并行如何联合处理

四个轴不是依次“先转 PP，再转 EP，再转 TP”。Planner 一次读取完整的 source
和 target placements：

1. 按 canonical tensor ID 匹配两份完整 placement 中的同一个逻辑参数。
2. target placement 决定新的 PP layer owner 和 EP expert owner。
3. DP 中相同 geometry 的副本只选择一个 source。
4. source/target 的 N-D logical boxes 求 overlap；两侧可以使用不同 shard
   dimensions，并直接生成最终 transfer region。
5. CopyRange 按 target worker 汇总为可执行子计划。

例如 source 为 `DP2/TP2/PP2/EP2`，target 为
`DP3/TP4/PP4/EP1`。对于 `layer.6.expert.3.w1`：

- source manifest 可能只在 PP0、EP1 的两个 TP rank 上包含它，并有 DP0/DP1
  两套副本。
- Planner 选择一套 DP source，不复制第二套到 Store。
- target manifest 可能把它放在 PP1、EP0 的四个 TP rank 上，并有三个 DP
  副本。
- 两个 source TP ranges 与四个 target ranges 直接求交，得到四组目标写入；
  每组再 fan-out 到三个 target DP。

中间不存在 `TP2/PP4/EP1` 或完整模型 buffer。PP/EP 变化体现在 owner，不
要求搬运一份额外的“中间布局”。

对于非零 split axis，Planner 用
`repeat/source_stride/target_stride` 表示规则 strided range；sink 分批展开，
不会为矩阵每一行创建计划对象。

## Store 持久化协议

### 为什么写入仍要经过转换层

假设 source 运行在 TP4/EP4。框架 adapter 先把 fused 参数中的 logical
Tensor view、global expert ID 和 TP range 写入 placement，并把当前地址写入
独立 binding；这是 metadata 转换，不复制权重。Store upload planner 随后：

- 去除 DP 重复。
- 为每个选中的 source fragment 创建 Payload Object。
- 直接从已注册 source address 调用 `batch_put_from`。

首版存的是 source-native Fragment，不强制转换为固定 TP8/EP8/PP8 Store
格式，也不做 Store 内 all-gather。持久语义与来源拓扑无关，未来可以在不
改变 Manifest 的前提下增加固定大小 chunk 或小 Tensor pack。

### Group、Manifest 与 upload session

资源 Group：

```text
weights/<namespace>/<resource_id>/<revision>/<weight_generation>
```

Payload 使用 `ObjectDataType.WEIGHT`，最终 Manifest 使用 metadata 类型，
都属于资源 Group。资源 Group 提供物理放置、hard pin、续租和回收；Manifest
提供模型语义、完整成员集合和 revision 可见性。

每次上传另有独立 session Group：

```text
weights/<namespace>/<resource_id>/<revision>/<weight_generation>/sessions/<upload_id>
```

不可变 `commit` 或 `abort` decision 写入 session Group，使控制对象与长期
Weight Payload 可以独立治理。当前 POC 把 terminal decision 保留为 durable
tombstone；`finalize_upload_session` 只确认 commit 对应 Manifest 已存在，或
abort/冲突 Payload 已清理，不物理删除 decision。立即删除会重新开放同一个
session，使迟到 abort 能破坏 READY revision，或使迟到 upload 复活 abort。
未来只有在具备明确 retention、请求 fencing 和可证明没有迟到调用后，控制面
才能回收 session tombstone。

完整提交顺序：

1. 收集 source parts，聚合为唯一的完整 source placement，并在上传前绑定实际
   参与写入的 source bindings，生成 UploadPlan。
2. worker 写入自己负责的 Payload，并用 `batch_is_exist` 确认 COMPLETE。
3. Coordinator 收齐 receipts，再次确认全部 required Payload COMPLETE。
4. 首写者写入不可变 COMMIT decision；ABORT 与它竞争同一个 session key。
5. COMMIT 后再次检查 Payload COMPLETE。
6. 最后写入不可变 Weight Manifest。
7. 读端只认完整、可解析且 coverage 通过的 Manifest。
8. finalize 核验终态并清理 loser/abort 的非引用 Payload，保留 terminal
   decision 作为幂等 tombstone。

Group 本身不被描述成事务。Manifest 是模型级提交点，session decision 解决
commit/abort 竞争和重试，二者职责不同。

### Store 加载

Receiver 先构建目标模型并导出 target placement；目标 allocation 就绪后再
导出 target binding。Planner 从 `StoredFragment` 到最终 target range 直接
生成逻辑计划，Store backend 在 bind 后用 `get_into_ranges` 写入目标参数地址，
不先恢复 source TP/PP/EP。

当前真实 ranged read 的重要边界是：内存 replica 已验证；部分磁盘/L3/OSS
后端可能仍需要读取或暂存接近整个 Object。生产上要么把 Weight 作为 memory
hard pin，要么增加固定 chunk 和后端原生 partial read，不能仅凭
`max_range_bytes` 承诺所有层级的峰值内存。

## GPU 到 GPU 协议

TE backend 使用与 Store 相同的 region，但在执行前把 source/target placement
与 binding 解析为当前地址。执行前必须提供 generation-bound registration
leases，绑定：

```text
fragment_id / worker_id / address / nbytes / lease_generation
```

任一 worker、address、nbytes 或 generation 变化都拒绝旧计划。source 不需要
复制整模型到额外 GPU buffer；TE 可直接读取 resident parameter range。需要
的额外空间只有 TE/transport 的有界内部队列和协议 staging，而不是 397B 或
1.8T 模型大小。

source 服务可以继续只读推理，但传输会共享 HBM、PCIe、NVLink 和 NIC 带宽。
框架必须阻止同时发生的原地权重更新；训练场景可以按参数或 bucket 获取 read
lease，传完立即释放。若必须保证严格无抖动，控制面应选择低负载副本或限流。

## SGLang POC

适配基线为 SGLang `origin/main@b8ec5449`（2026-07-19）。新增能力由
`--enable-weight-runtime-manifest` 显式开启，默认关闭，因此普通推理不会
创建 coordinator，也不会遍历参数。

当前实现包括：

- `ModelRunner.get_weight_runtime_manifest(...)` 和显式 lease release。
- 对 disk、distributed、tensor、IPC 四条在线更新入口的本地 generation
  协调；活跃 snapshot lease 会阻止更新，失败更新会 poison 后续 snapshot。
- 任一更新后先进入 pending，只有控制面完成整 revision 后显式调用
  `commit_weight_runtime_revision()`，本 worker 才重新允许 snapshot。
- aliases、CUDA address、shape、stride、storage offset、generation 导出。
- Qwen3.5 dense、静态 MoE、GDN、fused QKV/gate-up 和 Qwen3.5 VL 视觉塔语义。
- Qwen3 dense/MoE，以及 Triton block FP8 `e4m3fn 128x128` weight 和 FP32
  inverse scale 的 canonical serialized byte view 源码级 contract。
- tied embedding 与 LM head 保留两个 canonical identities。
- Scheduler、Tokenizer 和 HTTP 层的 begin/renew/release RPC；target 生成并
  复用 transfer ID，source 对活跃、过期和已释放 session 做幂等校验。
- 有界 lease TTL、heartbeat renew、失败 release 重试和迟到 begin 终态拒绝。

以下能力尚未实现：

- 集群级 source discovery。
- 所有 worker 对同一 revision 的 prepare/commit barrier。
- target 全量完成后的原子激活或流量切换。
- source 进程崩溃后的外部 lease 回收。

在线量化、跨 dtype/recipe、DeepGEMM/Marlin/swizzle/shuffle/pack、LoRA、
DP attention、动态 EPLB/elastic EP 继续 fail-closed。DeepSeek、Llama、
其他模型家族通过各自模型语义 adapter 扩展，不把模型规则放进 Mooncake。

## 当前支持边界

| 能力 | 状态 | 说明 |
| --- | --- | --- |
| DP source 去重和 target fan-out | 已验证 | Planner 单测和四轴 E2E |
| TP 度数变化 | 已验证 | 相同 dtype、split axis、layout；支持 split/merge |
| PP 度数变化 | 已验证 | target manifest 决定 layer owner |
| 静态 EP 度数变化 | 已验证 | global expert ID 决定 owner |
| DP/TP/PP/EP 同时变化 | 已验证 | 一次联合规划，无中间布局 |
| Store 权重持久化 | 已验证 | Payload、session decision、final Manifest |
| Store CUDA source/target | 已验证 | 原生 Store 扩展、多 GPU 地址 |
| TE TCP managed buffer | 已验证 | 两个独立 endpoint |
| TE intra-node CUDA P2P | 已验证 | 双进程 CUDA IPC/NVLink transport |
| TE completion fencing | 已验证 | ticket 和无-ticket `-2` 均不提前释放；永久 UNKNOWN 有界移交进程级 quarantine |
| 跨节点 RDMA/GDR | 条件支持，未验证 | 代码路径具备，缺少本轮 RDMA 环境证据 |
| L3/OSS ranged restore | 规划中 | 需 chunk 或后端原生 partial read 验证 |
| SGLang Qwen3.5 runtime export | 已验证 | 真实 Qwen3.5-0.8B CUDA 模型 |
| SGLang 多 worker RPC 和 lease lifecycle | 已验证 | begin/renew/release、TTL、幂等和失败回收单测 |
| SGLang target 原子 activation | 未实现 | 属于框架/外部控制面接线 |
| Canonical serialized Triton FP8 | 契约已验证 | 源码 fixture 覆盖 `e4m3fn 128x128` weight、FP32 inverse scale、W31/W13 logical split；未做真实量化模型 GPU E2E |
| slime 新 Planner 接入 | 未实现 | 现有 all-gather 路径保持可用 |
| 在线量化/跨 recipe/packed layout | 不支持 | DeepGEMM、Marlin、swizzle、shuffle、pack 明确 fail-closed |
| LoRA/动态 EPLB | 不支持 | 明确 fail-closed |

“已验证”表示 POC 对应模块和 E2E 已运行，不表示已合入 Mooncake/SGLang main
或已形成发布级 API 稳定性承诺。

## 性能与容量边界

- 不做模型级 all-gather，也不分配模型大小的中间 buffer。
- 连续 suffix 合并为 `inner_bytes`，其余维度保留为 compact strided region。
- Store 按 `max_range_bytes` 和 `max_ranges_per_request` 分批。
- TE 按 endpoint 和 batch 上限提交，支持预注册 allocator block 以摊薄注册成本。
- 首版每个 source Fragment 一个 Object，超大 MoE 的 Object 数可能较多；
  chunk/pack 是后续性能优化，不应改变 correctness schema。
- checksum 当前可选；coverage、nbytes、generation 和 Store 完成状态是首版门槛。

在 4×A10、无 RDMA HCA 的节点上，既有 256 MiB/1 GiB 测试中，Store
CUDA/TCP 数据面约为 8.7-12.1 GB/s，TE TCP managed path 约为
4.8-5.2 GB/s。数字只用于证明没有明显的规划器性能回退，不代表 RDMA/GDR
上限或生产 SLO。intra-node CUDA 测试当前只作为正确性证据。

## 验证记录

截至 2026-08-06：

- Mooncake 定向模块套件：`643 passed, 18 skipped, 8 subtests passed`。
- Mooncake 与 SGLang golden contract：SGLang adapter 将 Qwen3.5 MoE TP4/EP4
  runtime inventory 规范化为 typed placement 与 binding，并可直接生成 Store
  UploadPlan；Mooncake core 不读取 SGLang runtime object。
- Host、原生 Store/CUDA、TE/TCP、TE/CUDA 均完成
  `DP2/TP2/PP2/EP2 -> DP3/TP4/PP4/EP1` 全字节校验。
- Store/CUDA 使用 GPU0/1 source 和 GPU2/3 target；TE/CUDA 使用两个进程，
  target 进程拥有并注册目标 GPU memory。
- SGLang 定向单元测试：`182 passed`，在 H20 完整 Torch 环境通过。
- TE completion 原生 C++ 测试通过，覆盖完成、已 drain 失败和完成态未知时
  resource quarantine 的有界移交；Mooncake pending registry 保留 backing
  owner，SGLang 在永久 UNKNOWN 时连同目标上下文和模型一起保留，且不释放
  source lease。
- 真实 `Qwen3.5-0.8B` 在 H20 上由标准 `ModelRunner` 加载后导出 594 个
  logical tensors、2,214,531,776 bytes 的 CUDA resident views，并成功释放
  runtime lease。
- 真实 SGLang serving E2E 覆盖 Qwen3-8B `TP2 -> TP4`，以及
  Qwen3-30B-A3B `TP2/PP1/EP2 -> TP1/PP4/EP1`。cold 与 manifest 两种启动
  均通过 16 条严格 smoke、64 条并发请求、KVCache 命中和 50 条 GSM8K；30B
  两种模式均为 `47/50 = 0.94`，逐条 predictions、token、finish reason 和
  bounded logprob 一致，source probe 为 `0 error / 0 mismatch`。

尚未完成跨节点 RDMA/GDR、L3/OSS 原生 ranged restore、集群级 source
discovery 和 target serving activation，因此这些能力不得从上述 E2E 推导为
已完成。

## 合入建议

为降低社区 review 风险，建议拆成四组可独立验证的变更：

1. Mooncake Manifest、Planner、Store backend 和纯 Python/Store tests。
2. Mooncake TE backend、registration lease 和 native CUDA E2E。
3. SGLang 默认关闭的 runtime exporter、update coordinator、worker RPC 和
   Qwen3/Qwen3.5 adapters。
4. slime/K8s/Ray source discovery、revision barrier 和 activation。

Mooncake PR 不依赖 SGLang import；SGLang PR 不绑定 Mooncake transport。两边只
共享 versioned placement/binding contract，因此 SGLang 也可以把同一逻辑计划
编译到其他传输后端，而 Mooncake 可以接入 vLLM 或其他框架导出的契约。

## 后续工作

1. 将已实现的 begin/renew/release worker RPC 接入集群级 source discovery
   和全局 revision prepare/commit/activate 协议。
2. 增加 DeepSeek、Llama、更多 Qwen dense/MoE/VL adapter contract fixtures。
3. 增加固定 chunk、小 Tensor pack 和 disk/L3/OSS partial read。
4. 在 RDMA/GDR 集群运行跨节点 GPU-direct correctness 和性能矩阵。
5. 评估把 `resource_kind=WEIGHT`、manifest digest、READY state 和 serving
   lease count 作为紧凑 Group Descriptor 放入 Master；大体积 Tensor 列表仍
   保留在 Manifest Object，避免放大 Master snapshot 和 HA log。

参考：Mooncake Group semantics 讨论见
[GitHub issue #2282](https://github.com/kvcache-ai/Mooncake/issues/2282)。
