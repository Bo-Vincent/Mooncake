# 异构权重 Reshard Benchmark Driver

这个目录用同一份 logical tensor case 描述 Mooncake placement/binding +
`WeightTransferPlan` 与 NVIDIA NCCL M2N `ncclReshardWithWindow` 两条路径。
dry-run 用于审阅几何、rank 放置和 operation 上限；独立执行模块用于受门禁保护的
NCCL M2N 与 Mooncake CUDA/TE 双机测试。canonical 配置默认禁止真实执行。

## 当前能力

- 严格校验 2-D/3-D tensor、source/target replica、shard 数、shard dimension、
  rank 数和双机容量。
- 为 NCCL M2N 生成显式 source/target host 放置的 OpenMPI MPMD argv。
- source/target 各构造一个完整的 Mooncake `WeightPlacementManifest`，每个
  participant 单独提供带非零 fake address 的 `WeightRuntimeBindingManifest`，
  再调用 N-D planner 生成静态摘要。
- 统计 region 数、segment 数、inner bytes、每个 target 的 lowering batch 数和
  `max_region_segments` 拒绝原因，不按 row 或 element 展开操作。
- PP/EP/四轴语义 case 可以作为 planner-only 项保留；不会根据模型权重名字
  猜测 layer 或 expert，也不会把独立 expert allocation pack 成大 buffer。
- Mooncake source/target role 各自分配并注册本机 CUDA shard；target 只通过有界
  控制协议分别发布真实 placement、runtime binding 和 registration envelope。
- M2N executor 分开运行 cold one-shot 与 validated steady benchmark，同时保留
  stdout、stderr、进程 wall time 和官方 hot timing。
- Mooncake distributed runner 也分开运行 cold one-shot 与 validated steady
  进程；每个阶段都从启动远端 target 前计时，直到 source 和 target 都退出并被
  回收，避免把首次 update、warmup 和 steady iteration 混进同一个 E2E。
- M2N MPMD 两个 app context 可显式绑定 MPI control interface、导出 NCCL 环境，
  并按 context 起始 global rank 旋转 `CUDA_VISIBLE_DEVICES`，保证 benchmark 内部
  `global_rank % device_count` 映射到两台机器上相同的 local-rank GPU。

## 使用方式

从 Mooncake 仓库根目录运行：

```bash
PYTHONPATH=mooncake-wheel:mooncake-reshard/python:mooncake-reshard \
python3 -m \
  benchmarks.heterogeneous_weight_reshard dry-run \
  --config mooncake-reshard/benchmarks/heterogeneous_weight_reshard/cases.json \
  --m2n-binary /path/to/reshard_bench \
  --json
```

使用 `--case tp4_to_tp8_dim0` 只查看一个 case。dry-run CLI 仍然没有 `run`
子命令；输出中的 MPMD command 仅用于审阅，不会被该命令执行。

## 真实执行

真实执行使用单独模块，并要求运行配置显式设置 `execution_enabled=true`，同时环境
中有 `VIN_RUN_BENCHMARK=1`。推荐在 source host 使用外层 runner；它根据配置中的
placement 通过 SSH 启动 target，并测量两个角色的完整进程 wall：

```bash
VIN_RUN_BENCHMARK=1 \
PYTHONPATH=mooncake-wheel:mooncake-reshard/python:mooncake-reshard \
python3 -m \
  benchmarks.heterogeneous_weight_reshard.mooncake_distributed_benchmark \
  --config /path/to/authorized-cases.json \
  --case tp4_to_tp8_dim0 --control-port 29600 \
  --output-dir /path/to/results/mooncake/tp4_to_tp8_dim0
```

该 runner 依次启动独立的 `cold` 和 `steady` source/target 进程对。每个阶段会先
保存 `source.*.log` 和 `target.*.log`，再解析角色结果；失败运行也保留日志。
`cold_e2e_ms` 是单次未校验 update 的完整进程 wall，`steady_process_wall_ms` 是
warmup/iteration/validation 进程的完整 wall，`benchmark_process_wall_ms` 是两者
之和。`protocol_wall_ms` 只保留为角色内部诊断时间。

以下手工启动方式只用于调试角色协议；它不能单独提供完整双进程 E2E。先在 target
host 启动 Mooncake target role：

```bash
VIN_RUN_BENCHMARK=1 \
PYTHONPATH=mooncake-wheel:mooncake-reshard/python:mooncake-reshard \
python3 -m \
  benchmarks.heterogeneous_weight_reshard.mooncake_benchmark \
  --config /path/to/authorized-cases.json target \
  --bind-host 0.0.0.0 --control-port 29600 \
  --engine-host 172.16.1.108 --protocol rdma --device '<rdma-device>' \
  --cuda-devices 0,1,2,3,4,5,6,7
```

再在 source host 启动 source role：

```bash
VIN_RUN_BENCHMARK=1 \
PYTHONPATH=mooncake-wheel:mooncake-reshard/python:mooncake-reshard \
python3 -m \
  benchmarks.heterogeneous_weight_reshard.mooncake_benchmark \
  --config /path/to/authorized-cases.json source \
  --case tp4_to_tp8_dim0 \
  --phase steady \
  --target-control-host 172.16.1.108 --target-control-port 29600 \
  --engine-host 172.16.1.107 --protocol rdma --device '<rdma-device>' \
  --cuda-devices 0,1,2,3,4,5,6,7
```

每个 logical rank 使用独立 CUDA allocation，并要求一 rank 对应一张 GPU。steady
计时复用 allocation、registration、placement/binding 和 plan；计时后 target
先清零，再执行
一轮不计时 update 并做完整分块校验，防止旧内容掩盖覆盖缺失。退出顺序固定为
unregister 后 cudaFree。

在 source host 运行 M2N：

```bash
VIN_RUN_BENCHMARK=1 \
PYTHONPATH=mooncake-wheel:mooncake-reshard/python:mooncake-reshard \
python3 -m \
  benchmarks.heterogeneous_weight_reshard.m2n_benchmark \
  --config /path/to/authorized-cases.json \
  --case tp4_to_tp8_dim0 \
  --binary /path/to/reshard_bench \
  --mpi-interface eth0 \
  --export-env NCCL_SOCKET_IFNAME \
  --export-env NCCL_IB_HCA \
  --output-dir /path/to/results/tp4_to_tp8_dim0
```

M2N 先执行 `warmup=0, iterations=1, validate=no` 的独立 cold run，再执行配置中
warmup/iterations 的 validated steady run。`result.json` 与四份原始 stdout/stderr
日志一起保留；任一进程超时、非零退出、结果区不完整或 validation 不完整时，不
发布吞吐。OpenMPI 的 `-x` 在 MPMD 中属于 app-context 参数，因此 source 和
target context 都显式导出 `PATH` 与 `LD_LIBRARY_PATH`，保证远端 rank 使用同一套
自建 NCCL/M2N 动态库。

`--mpi-interface` 会在两个 MPMD app context 中同时设置 OpenMPI
`oob_tcp_if_include` 与 `btl_tcp_if_include`；每个 `--export-env NAME` 也会在两个
context 中分别生成 `-x NAME`。这用于把 MPI/NCCL bootstrap 固定到控制网卡，
把 NCCL data interface/HCA 交给显式环境配置，避免多网卡自动选择在 NCCL init
阶段挂起。

NVIDIA `reshard_bench` 使用 `cudaSetDevice(global_rank % numDevices)`。当 target
global rank 从 4 开始时，直接运行会把 target local rank 0..3 放到物理 GPU 4..7，
而 Mooncake 使用物理 GPU 0..3，形成不公平拓扑。adapter 因此只在 MPMD
app-context 层旋转 `CUDA_VISIBLE_DEVICES`：例如起始 rank 4 使用
`4,5,6,7,0,1,2,3`，使 global rank 4 最终落到物理 GPU 0。该适配不修改 NVIDIA
benchmark、NCCL M2N runtime 或模型数据布局。

## Case 分类

`cases.json` 将输入分成三类：

- `physical_cases`：source 和 target 分别不超过单节点 8 GPU，world size 不超过
  16，可作为后续真实双机测试候选。
- `stress_cases`：几何合法，但可能突破当前 TE 有界 lowering 限制；dry-run 会
  如实给出拒绝原因。
- `planner_only_cases`：用于表达 PP ownership、独立 expert tensor 和超过物理
  GPU 数的四轴组合，不伪装成同规模性能结果。

原始 `[2048, 2048, 2048]` dim0 -> dim2 case 会生成 16,777,216 个 TE
segment，单 region 为 1,048,576，超过默认
`max_region_segments=1,000,000`，因此归入 `stress_cases`。物理候选改用
同为 8 GiB 的 `[128, 4096, 16384]`，单 region 为 131,072 segment。

## Rank 放置

M2N dry-run 使用两个 OpenMPI app context：

```text
global rank [0, source_total)          -> placement.source_host
global rank [source_total, world_size) -> placement.target_host
```

两个 context 使用同一个 M2N binary 和同一份 mesh/tensor 参数。driver 不依赖
通用 hostfile 的默认 rank 分配，因为那可能把 TP4 -> TP8 的 source 和 target
放到错误节点。M2N 摘要还会分别输出 `physical_execution_eligible` 和 refusal
reasons：超出当前每节点 GPU 数的 stress/planner-only case 仍可生成静态描述，
但不会被误标成可执行的物理 case。

## 执行门禁

启动真实进程前必须同时满足：

```text
config.execution_enabled == true
env[config.execution_guard] == "1"
```

canonical 配置保持 `execution_enabled=false`，guard 名为
`VIN_RUN_BENCHMARK`。dry-run 不读取全局环境，也不会因为 guard 已设置而执行。
真实执行还必须重新通过双机 isolation gate；benchmark 不允许关闭 Mooncake
lease、generation、address bounds 或 operation-count 校验。

## 公平比较口径

NVIDIA `reshard_bench` 自带时间只覆盖热循环中的
`ncclReshardWithWindow`、stream synchronize 与 MPI barrier，不等于完整 E2E。
双方 executor 同时报告：

- `cold_e2e_ms`：独立 cold one-shot 进程的完整 wall
- `steady_process_wall_ms`：独立 steady 进程的完整 wall
- `process_wall_ms`：`steady_process_wall_ms` 的兼容别名
- `benchmark_process_wall_ms`：cold 与 steady 两个进程 wall 之和
- `protocol_wall_ms`：Mooncake 两端从角色启动到 release/shutdown 完成
- `first_update_ms`
- `steady_update_ms`
- `data_plane_ms`
- `transport_init_ms`
- `registration_ms`
- `plan_ms`
- `lowering_ms`
- `logical_GiBps`
- `wire_GiBps`

正确性校验单独计时，并作为每个 case 的硬门槛。logical throughput 只按唯一
logical tensor bytes 计算，不重复计算 DP replica；wire throughput 使用 NIC
counter 增量；没有可靠的全 NIC counter 时该字段保持不可用，不用 logical bytes
伪装。Mooncake 当前 Sink 一边迭代 N-D region 一边提交 native batch，因而
`lowering_ms` 保持不可用，并另报完整 update、native blocking TE 与
host lowering/dispatch 差值。Mooncake 通用 `transfer_engine_bench` 只验证 TE
构建和裸传输，
不表达 weight placement/binding 或异构 reshard，不能作为这组对比的 Mooncake
主结果。
Mooncake source/target 角色自身不会把协议 wall time 冒充完整进程 E2E；正式双机
结果必须由外层 runner 从启动 target 前计时，直到 source 与 target 都正常退出。
M2N 每个 cold/steady 子进程结束后立即持久化 stdout/stderr，再做结果解析，因此
超时、非零退出和 validation 失败也会保留原始证据。

## 设计边界

- `WeightPlacementManifest` 是完整 topology、shape、dtype、layer/expert 和
  ownership 的事实来源；`WeightRuntimeBindingManifest` 按 participant 独立提供
  物理地址、worker、endpoint、generation 和 lease，不合成为第三种 manifest。
- benchmark adapter 不进入 Mooncake core，也不引入 NCCL M2N runtime、
  `ncclMemAlloc` 或模型命名依赖。
- PP 是 tensor ownership routing；EP 是独立 logical expert tensor 的 coordinate；
  DP 是 replica selection；TP/EP tensor 内重组由 N-D logical-box overlap 表达。
- 当前物理性能 case 是单 logical tensor 的 canonical descriptor。PP、独立 expert
  tensor 和四轴 multi-tensor case 继续只做 planner correctness；未来物理 executor
  必须按显式 ownership/coordinate 拆分 communicator 或 transfer plan。
