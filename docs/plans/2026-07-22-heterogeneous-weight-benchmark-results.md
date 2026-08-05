# Mooncake N-D Reshard 与 NCCL M2N 实测报告

日期：2026-07-22

## 结论

当前 `runtime manifest + WeightTransferPlan + Mooncake TE` 路径已经在真实双机
GPU-to-GPU 环境完成 TP expand/shrink、DP replica 去重和两种 EP/TP cross-dim
几何的内容校验。五个物理 case 的每个 target logical tensor 都完整覆盖且内容
一致，说明 N-D logical-box planner 与 runtime manifest 数据面链路可用。

性能上需要分两层看：

- 热 update：NCCL M2N 在五个 case 中都更快。连续或较连续布局快约
  `1.68x-1.99x`；最内维 `dim0 -> dim2` 快约 `5.00x`。
- benchmark 进程 E2E：Mooncake 在四个连续或较连续 case 中更短，因为 M2N 每次
  重新承担 MPI/NCCL 进程初始化；`dim0 -> dim2` 则由 Mooncake 的 209 万次
  CopyRange lowering 主导，E2E 明显更慢。

因此当前实现已经证明“能正确转换”，但还不能声称所有 N-D layout 都有可接受的
热更新性能。下一步的核心不是修改 planner 语义，而是为满足条件的 strided N-D
region 增加压缩 lowering 或未来 M2N executor，继续保留 lease、generation、
address bounds 和 operation-count 校验。

## 测试版本与环境

- Mooncake：`vin/heterogeneous-weight-transfer-v2`，`755598254defcb1327a0868aa7b10e38ddc06a31`
- NCCL M2N：`5067397c2676d5aed50042fc39e5c8ee96eb0027`
- NCCL/CUDA/OpenMPI：NCCL 2.30.7、CUDA 12.8、OpenMPI 4.1.6
- 物理资源：每节点 8 张 H20；每节点 4 个 `mlx5_bond_*` HCA
- workload：除 DP case 为 4 GiB 外，其余均为 8 GiB logical uint8 tensor；
  warmup 3 次、steady 20 次

这不是完整模型 benchmark。PP、独立 expert 和四轴 multi-tensor case 当前只覆盖
planner correctness；本报告中的 EP/TP cross-dim 是单个 logical expert tensor
的物理数据面验证。

## 公平性修正

正式结果采用以下约束：

1. MPI OOB/BTL 与 NCCL bootstrap 固定到 `eth0`，NCCL data HCA 固定到四个
   `mlx5_bond_*`，避免多网卡自动选择导致 NCCL init 挂起。
2. 两个 MPMD app context 都显式导出 `PATH`、`LD_LIBRARY_PATH` 和 NCCL 环境。
3. 按 app-context 起始 global rank 旋转 `CUDA_VISIBLE_DEVICES`。这抵消 NVIDIA
   benchmark 的 `cudaSetDevice(global_rank % numDevices)`，让 source/target 都
   使用对应 local-rank 的同一物理 GPU。
4. M2N 与 Mooncake 都使用独立 cold one-shot 进程和独立 steady 进程；E2E 从
   角色启动前计到全部进程退出。
5. 所有吞吐都以完整内容校验通过为前提。DP logical bytes 只计唯一副本。

GPU 映射修正前，target global rank 从 4 开始的 TP4 -> TP8 和两个 cross-dim
M2N 样本不进入最终统计；target 从 rank 8 开始的 TP8 -> TP4 与 DP case 不受该
问题影响。

## 热 Update

M2N 使用 3 个独立进程样本的中位数和 min/max。Mooncake 展示修正 E2E 口径后的
validated steady p50；两次 Mooncake hot run 的数值范围另列在后文。

| Case | Logical bytes | M2N steady 中位数 [min, max] | M2N GiB/s | Mooncake steady p50 | Mooncake GiB/s | M2N latency 优势 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TP4 -> TP8 dim0 | 8 GiB | 132.500 ms [132.446, 143.359] | 60.38 | 225.739 ms | 35.44 | 1.704x |
| TP8 -> TP4 dim0 | 8 GiB | 143.253 ms [139.899, 223.445] | 55.85 | 243.762 ms | 32.82 | 1.702x |
| dim0 -> dim1 | 8 GiB | 134.623 ms [133.288, 749.650] | 59.43 | 267.459 ms | 29.91 | 1.987x |
| dim0 -> dim2 | 8 GiB | 1007.077 ms [344.093, 1349.825] | 7.94 | 5038.308 ms | 1.59 | 5.003x |
| DP2/TP4 -> DP1/TP8 | 4 GiB | 66.316 ms [66.312, 66.321] | 60.32 | 111.686 ms | 35.81 | 1.684x |

M2N 在 `dim0 -> dim1` 和 `dim0 -> dim2` 上存在明显进程级抖动，所以不能只引用
最佳值。DP case 的三个 M2N 样本只有 0.009 ms 范围，说明 replica 去重口径稳定。

Mooncake 两次 hot run 的 p50 范围如下：

- TP4 -> TP8：`225.698-225.739 ms`
- TP8 -> TP4：`243.762-269.509 ms`
- dim0 -> dim1：`267.459-268.779 ms`
- dim0 -> dim2：`5038.308-5059.105 ms`
- DP2/TP4 -> DP1/TP8：`111.686-116.617 ms`

## 进程 E2E

`benchmark_process_wall_ms = cold_e2e_ms + steady_process_wall_ms`。M2N 为三个
独立样本的中位数和范围；Mooncake 为完成公平性修正后的一个 validated 样本，
因此这里只能比较当前观测值，不能当作稳定分布结论。

| Case | M2N total E2E 中位数 [min, max] | Mooncake total E2E | 当前观测 |
| --- | ---: | ---: | --- |
| TP4 -> TP8 dim0 | 41.557 s [41.368, 42.164] | 23.733 s | Mooncake 短 17.824 s |
| TP8 -> TP4 dim0 | 37.656 s [37.035, 43.035] | 23.481 s | Mooncake 短 14.175 s |
| dim0 -> dim1 | 31.461 s [30.292, 43.225] | 22.631 s | Mooncake 短 8.830 s |
| dim0 -> dim2 | 45.302 s [31.699, 54.450] | 141.625 s | Mooncake 长 96.323 s |
| DP2/TP4 -> DP1/TP8 | 41.169 s [36.945, 41.428] | 19.626 s | Mooncake 短 21.543 s |

进程 E2E 不是常驻服务热更新时间：它刻意包含进程、CUDA allocation、注册、
transport/MPI/NCCL 初始化、warmup、iteration、validation 和退出。真实在线权重
更新应优先参考 steady update，同时单独评估常驻服务中的 pause/drain 与控制面。

## Mooncake Lowering 分解

| Case | CopyRange operations | Batches | Host dispatch p50 | Native TE p50 | Total p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP4 -> TP8 dim0 | 8 | 4 | 0.402 ms | 225.364 ms | 225.739 ms |
| TP8 -> TP4 dim0 | 8 | 8 | 0.452 ms | 243.285 ms | 243.762 ms |
| dim0 -> dim1 | 8,192 | 8 | 12.044 ms | 255.199 ms | 267.459 ms |
| dim0 -> dim2 | 2,097,152 | 2,048 | 3165.373 ms | 1899.923 ms | 5038.308 ms |
| DP2/TP4 -> DP1/TP8 | 8 | 4 | 0.601 ms | 111.097 ms | 111.686 ms |

当前 operation guard 确实避免了无界按 element 展开，也没有 pack/all-gather 整个
tensor，但 `dim0 -> dim2` 的有界结果仍然过大。建议保留 backend-neutral N-D
region，在 executor 层增加以下优先级：

1. 能表达为少量 strided copy 的 region，使用压缩 descriptor/native batch。
2. 满足 live G2G、连续注册、bounds/lease 等条件的 region，预留 M2N lowering。
3. 不满足条件时继续使用当前 TE CopyRange fallback，并保持 operation guard 拒绝。

## 正确性与边界

- 五个物理 case 的 M2N 和 Mooncake target 内容校验全部通过。
- runtime manifest 仍是地址、shape、dtype 和模型语义的唯一事实来源。
- benchmark adapter 没有修改 Mooncake core，也没有引入 NCCL M2N runtime、
  `ncclMemAlloc` 或模型命名依赖。
- 本轮没有物理执行 PP2 -> PP4、PP4 -> PP2、EP8 -> EP2、EP2 -> EP8 或完整
  TP/PP/EP/DP 四轴 multi-tensor 模型；这些仍需在 runtime 导出的 layer/expert
  manifest 上做下一阶段 E2E。
- `wire_GiBps` 没有可靠的全 HCA counter delta，本报告只给 logical GiB/s。

原始机器可读 JSON 与 stdout/stderr 保存在工作区外的
`.vin_stage/rdma-m2n-results/20260722/`，不进入个人分支或后续社区提交。
