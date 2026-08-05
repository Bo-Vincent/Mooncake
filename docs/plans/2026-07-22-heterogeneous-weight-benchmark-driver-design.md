# 异构权重 Benchmark Driver 设计

日期：2026-07-22

## 目标

为 Mooncake runtime manifest + `WeightTransferPlan` 路径与 NVIDIA NCCL M2N
`ncclReshardWithWindow` 路径提供同一份 logical tensor case、物理 rank 放置和
机器可读结果契约。第一阶段只实现 dry-run、静态规划与执行门禁，不初始化
MPI/NCCL、CUDA、Mooncake TE 或 RDMA，也不采集性能数据。

## 非目标

- 不在 Mooncake core 引入 NCCL M2N runtime、`ncclMemAlloc` 或模型命名依赖。
- 不根据 `q_proj`、`down_proj`、`expert` 等名字推断模型语义。
- 不复用 NVIDIA model benchmark 的 PP 公式或 dedup 逻辑。
- 不把独立 expert allocation pack 或 all-gather 成连续大 buffer。
- 不用通用 `transfer_engine_bench` 代替异构权重 benchmark。

## 目录与边界

新增独立目录：

```text
mooncake-reshard/benchmarks/heterogeneous_weight_reshard/
  case_spec.py
  m2n_adapter.py
  mooncake_adapter.py
  __main__.py
  cases.json
  README.md
  tests/
```

该目录是 benchmark harness，不属于 Mooncake 产品运行时。Mooncake adapter
只调用公开的 manifest/planner API；M2N adapter 第一阶段只生成命令和 descriptor
摘要，不链接或执行 M2N。后续 executor 可独立演进，不能反向污染
`mooncake.reshard.weight` 的语义边界。

## 数据模型

`BenchmarkConfig` 保存：

- schema version、dtype、warmup、iterations 和指标列表；
- source/target host、每节点 GPU 数和总 GPU 数；
- `physical`、`stress`、`planner_only` 三类 case；
- 双重执行门禁：配置 `execution_enabled` 与环境变量名。

`BenchmarkCase` 保存显式 logical tensor shape 和 source/target mesh：

```text
MeshSpec = replicas x shards @ shard_dim
```

所有 geometry 都来自配置，不从 tensor name 或模型类型推导。physical case 必须
满足 tensor rank 2..3、shard 维可整除、source/target rank 分别不超过单节点
GPU 数、world size 不超过物理 GPU 总数。case ID 在所有分类中全局唯一。

## Rank 放置

不能仅使用 `hosts` 文件和 `mpirun -np N`，否则 TP4 -> TP8 的前 8 个 rank
可能全部落到 source 节点。driver 生成 OpenMPI MPMD app contexts：

```text
rank [0, src_total)             -> source host
rank [src_total, world_size)    -> target host
```

M2N 两个 app context 执行同一个 binary/参数；Mooncake 后续 executor 使用 source
与 target 两个不同 role。结构化 argv 是事实来源，shell command 只作为展示，避免
字符串拼接成为执行接口。

## Backend-neutral Dry-run

每个 case 输出：

- logical bytes、world size、source/target rank 与 host 放置；
- M2N mesh、tensor descriptor 参数和 MPMD argv；
- Mooncake manifest 数、N-D region 数、总 segment 数、最大单 region segment、
  inner bytes、每 target batch 数和 plan total bytes；
- 是否允许后续执行，以及所有拒绝原因。

Mooncake 静态规划使用非零 fake address；这些地址只用于 manifest bounds 与
geometry 校验，永远不会交给 TE。DP source replica 由现有 planner 选择；PP 和
EP 的 multi-tensor case 后续继续使用显式 owner/coordinate，不改变本阶段的单
tensor schema。

## Dim2 Case 决策

原 `[2048, 2048, 2048]`、dim0 -> dim2 case 生成 16 个 N-D region，但 TE
lowering 共需 16,777,216 个 512 B segment，单 region 为 1,048,576，超过默认
`max_region_segments=1,000,000`。该 case 保留为 `stress`，用于暴露当前 TE
lowering 边界，不进入默认物理执行集合。

新增 `[128, 4096, 16384]`、dim0 -> dim2 physical case。它仍为 8 GiB uint8
logical tensor，但单 region 为 131,072 segments、inner range 为 4 KiB，不突破
默认单 region guard；实际性能好坏仍由后续测试如实报告。

## 执行安全

第一阶段 CLI 只提供 `dry-run`，不存在 `run` 子命令。后续 executor 必须同时
满足：

```text
config.execution_enabled == true
环境变量 VIN_RUN_BENCHMARK == "1"
```

并在启动任何进程前重新执行双机 isolation gate。validation 未通过时禁止生成
吞吐；lease、generation、address bounds 与 operation guard 不能因 benchmark
而关闭。

## 测试

- case schema：字段类型、整除、rank 上限、重复 ID、分类和执行门禁；
- M2N adapter：TP expand/shrink、cross-dim、DP replica、MPMD rank 放置；
- Mooncake adapter：region/segment/inner-byte 精确值、DP 去重、dim2 stress 拒绝；
- CLI：确定性 JSON、case filter、只有 dry-run、无 subprocess/GPU 副作用；
- 回归：既有 model-weight 单测保持通过。

## 执行阶段

dry-run 合同稳定后，执行阶段仍留在独立 benchmark harness，不修改 Mooncake
core。M2N 与 Mooncake 共用 case、warmup/iteration、正确性门禁和结果 schema，
但分别使用各自原生执行方式。

### Mooncake 跨机角色

target 进程在 target host 初始化自己的 `TransferEngine`，按 target mesh 在指定
GPU 上逐 shard 分配独立 CUDA allocation 并注册内存。target 通过长度前缀 JSON
控制通道返回由真实 endpoint、GPU address、shape、rank、lease generation 和
runtime lease ID 构成的 v2 runtime manifest；不返回或持久化 CUDA buffer 内容。

source 进程在 source host 完成同样的本地 allocation/registration，先用收到的
target placement 与本地 source placement 调用 `plan_placement_transfer`，再用
双侧 runtime bindings 调用 `bind_logical_transfer_plan`。实际发送仍由
`MooncakeTransferEngineSink.execute` 完成，
并显式传入 source/target registration leases；benchmark 不绕过 lease、
generation、snapshot、bounds 或 region-count 校验。

控制通道只负责 `prepare`、`validate`、`shutdown` 和错误传播。target buffer 不在
每轮重复分配，source/target 注册也不放进 steady-state 数据面时间。正确性校验在
计时轮次之后由 target 对本地 CUDA buffer 分块执行，不把完整权重读回 source，
也不把多个 expert allocation pack 成大 buffer。

Mooncake 外层 distributed runner 分别运行独立 cold 与 steady source/target
进程对。每个阶段都先启动远端 target SSH 进程，再启动本地 source，从第一个
角色启动前计时到两个角色都退出并完成进程回收。两端 stdout/stderr 在解析前
落盘；角色非零退出、超时或结果 marker 无效时保留日志但不写成功结果。
`cold_e2e_ms` 与 `steady_process_wall_ms` 分别表达两个阶段，
`benchmark_process_wall_ms` 为两者之和；角色内部只报告 `protocol_wall_ms`。

### M2N 执行

M2N executor 直接执行 dry-run 已生成的结构化 OpenMPI MPMD argv。热更新时间以
官方 benchmark 输出的 `ncclReshardWithWindow + cudaStreamSynchronize +
MPI_Barrier` 为准；Python wrapper 另外用单调时钟记录完整子进程 wall time。
one-shot cold run 与 warm steady run 分开执行，避免把官方 hot timing 误写成
完整 E2E。wrapper 不链接 M2N、不调用 `ncclMemAlloc`，也不在 Mooncake 路径中
引入 M2N runtime。MPMD 的每个 app context 都显式转发 `PATH` 与
`LD_LIBRARY_PATH`；不能只在第一个 context 设置，否则 target rank 会加载不到
固定版本的 NCCL/M2N 动态库。

多网卡环境中，两个 app context 还必须同时设置 OpenMPI control interface 并
显式导出 NCCL bootstrap/HCA 环境，避免 source/target 在 NCCL init 选择不同
网络。由于 NVIDIA benchmark 使用 `cudaSetDevice(global_rank % numDevices)`，
adapter 按每个 context 的起始 global rank 旋转 `CUDA_VISIBLE_DEVICES`，保证
target global rank 4 对应 target physical GPU 0，而不是 GPU 4。该处理只修正
benchmark 物理放置，不修改 M2N runtime 或 logical tensor descriptor。

### 结果口径

每个 backend/case 至少输出：

- `cold_e2e_ms`：独立 cold one-shot 进程的完整 wall；
- `steady_process_wall_ms`：独立 steady 进程的完整 wall；
- `process_wall_ms`：steady 进程 wall 的兼容字段；
- `benchmark_process_wall_ms`：cold 与 steady 进程 wall 之和；
- `protocol_wall_ms`：Mooncake 角色内部从启动到 release/shutdown 完成，仅用于
  诊断，不替代完整进程 E2E；
- `first_update_ms`：首次已初始化 update；
- `steady_update_ms`：warmup 后 update 样本的 p50/mean/p95；
- `transport_init_ms`、`registration_ms`、`plan_ms`：可独立测量的控制面阶段；
- `data_plane_ms`：backend 原生 update 调用及其同步完成时间；
- `logical_GiBps`：唯一 logical tensor bytes 除以 update 时间，DP replica 不重复；
- correctness 状态与原始 backend 输出路径。

Mooncake 当前 sink 在遍历 N-D region 时边 lowering 边提交 batch，因此在没有
额外核心 instrumentation 时，`lowering_ms` 单独记为不可用，明确包含在
`data_plane_ms`，不能伪造拆分值。`wire_GiBps` 仅在能对所有实际使用 NIC 取得
可靠 counter delta 时输出；否则记为不可用，而不是用 logical bytes 代替。

### 失败与清理

任一角色 schema 不匹配、超时、TE 返回非零、receipt bytes 不完整、M2N validate
失败或 target 内容不一致，整个 case 都失败且不发布吞吐。target 始终按注册逆序
unregister，再释放 CUDA allocation。正式采数前记录并停止 `amperf.service`，
结束后恢复原状态；远端不可达时只允许继续本地实现和静态验证，不得生成模拟性能
结果。
