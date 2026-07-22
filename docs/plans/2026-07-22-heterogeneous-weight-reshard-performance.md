# 异构权重转换完整性与性能验证报告

日期：2026-07-22
分支：`vin/heterogeneous-weight-transfer-v2`
N-D 核心基线：Mooncake `9e850de122c1e13af732c03330fa827a774f7d7c`
SGLang adapter 基线：`2db0c78425eb7952651149dc0c662fd3ea6f6108`

## 1. 结论

1. manifest v2、N-D logical-box planner、PP route、EP leading coordinate、
   DP replica 选择、Store/TE 有界 lowering 已经实现；旧 manifest 和旧 TP
   路径保持兼容。
2. Mooncake 异构 reshard 的主对照必须是 NVIDIA `nccl_m2n` 的
   `ncclReshardWithWindow`，不能用普通 NCCL `send/recv` 代替。此前基于普通
   P2P 得出的“Mooncake 超过 NCCL”结论已撤回；相关数字只保留为裸链路诊断。
3. 固定 NCCL `5067397c`（2.30.7）的 M2N 已在 H20 构建成功，但当前 eRDMA
   环境没有产出任何有效 M2N 性能样本。普通 NCCL eRDMA collective 在兼容
   `max_recv_wr=0` 后可通过；M2N 最小 1 -> 1 和 TP4 -> TP2 仍在 GIN `IPut`
   上返回 verbs `status=21`，随后 QP `local work queue catastrophic error`，
   `DIRECT`/`RING` 均无法完成 validated iteration。
4. 正确性能结论必须同时包含有效吞吐和 E2E：warm reshard API E2E、cold
   communicator/window/setup E2E，以及框架 target 首次可用 E2E。当前只有
   Mooncake 的 raw TE、Store 和 Qwen3-Next E2E 数据；同机 M2N 列为 `N/A`，
   因而暂时不能给出 Mooncake 与 M2N 的快慢比。
5. Qwen3-Next 框架级 E2E 中，从目标进程拉起到首次确定性推理完成，
   checkpoint cold p50 为 `107.101 s`，live manifest reuse p50 为
   `68.998 s`，节省 `38.104 s`，缩短 `35.6%`，加速 `1.55x`。
6. 当前 RDMA fallback 单路平均 `20.389 GiB/s`。它适合跨节点，不应和
   同机 NVLink 数字混用。
7. WeightStore 的六个核心 API 和 CUDA 数据路径已实现，可以作为 POC/library
   使用；但生产控制面、revision barrier、target activation、故障恢复、完整
   lease 消费仍未接通，因此不能宣称生产服务已经完整。
8. Store CUDA ranged load 当前经过 host 临时 buffer，再 H2D。H20 实测
   upload 约 `9.5-10.3 GB/s`，load 约 `5.0-5.3 GB/s`；下一阶段最高 ROI
   是让 MEMORY ranged GET 直接写入已注册 GPU range。

## 2. 测量边界

### 2.1 环境

- 机器：8 x NVIDIA H20，单卡 97,871 MiB；GPU 两两拓扑均为 `NV18`。
- CUDA：12.8。
- NCCL：官方默认分支 `master`，
  `5067397c2676d5aed50042fc39e5c8ee96eb0027`，版本 2.30.7。
- nccl-tests：官方默认分支 `master`，
  `a0b82b2260cf5152b9f8c061bbf7eaf0ba096432`。
- Mooncake：Release + CUDA + `USE_INTRA_NVLINK=ON`。

### 2.2 公平口径

- 主对照必须调用 M2N `reshard_bench` 的 `ncclReshardWithWindow`。source 和
  target 使用不重叠 rank、相同 global tensor/dtype、相同 source/target mesh、
  相同 shard dimensions，并且每组结果必须通过 `--validate`。普通 NCCL
  `ncclSend/ncclRecv` 不是 reshard executor，只能表示裸 P2P transport ceiling。
- warm reshard E2E：在 communicator、symmetric allocation、window、manifest
  和 tensor descriptor 已就绪后，测一次 API 提交到所有 target 数据可见。
  M2N 口径是 `ncclReshardWithWindow -> cudaStreamSynchronize -> MPI_Barrier`，
  使用所有 rank 的最大 iteration time；Mooncake 口径应是
  `plan/lowering -> submit -> completion -> target barrier`。两边都报告 p50、
  p95 和 max，不能只用异步提交吞吐代替完成时间。
- warm 有效吞吐统一定义为 `global logical tensor bytes / max-rank E2E`。
  这里计算唯一逻辑 payload，不把 source replica、destination replica、ring
  forwarding 或 PP route 中的重复物理流量重复记入分子。
- cold reshard E2E：从 source/target topology 已知开始，包含 communicator 或
  TE session 建立、memory allocation/registration、window/segment 建立、manifest
  获取、planning/lowering、首次 validated transfer 和 target barrier。官方
  `reshard_bench` 的结果只计 warm loop，cold setup 必须由外层 harness 单独计时。
- framework E2E：从 target process/pod 拉起开始，包含 runtime 初始化、权重
  获取与转换、revision activation，截止首次确定性推理响应完成；source 已在
  运行且持续 serving。该指标和单 tensor reshard E2E 分开报告。
- 辅助 raw TE/P2P 数据仍按原口径保存：Mooncake 使用 64 MiB block 和 CUDA
  registered buffer；普通 NCCL 每个 GPU pair 使用独立 communicator 和
  `ncclCommRegister`。这些数据不进入 Mooncake/M2N 胜负结论。
- `GB/s` 为十进制；`GiB/s` 为二进制。主对比统一使用 `GiB/s`。
- Store 测试使用一个进程内的 source GPU0-3、target GPU4-7，3 次 warmup、
  10 次正式采样。Store E2E 从 `prepare_upload` 开始，包含 upload、commit、
  manifest get、plan load 和 load，截止 target 数据到位；不包含 Store setup、
  GPU buffer 分配/注册、结果校验和测试清理。
- Store 的机器可读 payload 将 backend 记为协议无关的 CUDA/pre-registered
  路径，并把请求协议、`MC_STORE_MEMCPY` 和“运行时选择策略”单独记录。
  Python API 当前不返回最终 `TransferStrategy`；下表的 local-copy/RDMA 判断
  来自本次 Store transport 日志，不仅依据请求协议。
- 框架 E2E 使用每轮 `spawn_to_ready_s + first_generation_s` 后取中位数，边界
  是“target process spawn 到首次确定性推理响应完成”；source 已在运行，其
  启动时间不计入，期间持续探活并校验响应一致性。

## 3. NCCL M2N 主对照

### 3.1 当前可比性结论

| 指标 | Mooncake 当前样本 | NCCL M2N 当前 H20 样本 | 可否比较 |
| --- | --- | --- | --- |
| validated N-D reshard 吞吐 | 尚未按 M2N 相同 mesh/shape 重跑 | `N/A`，GIN iteration 未完成 | 否 |
| warm reshard API E2E | 尚未按 `plan -> barrier` 统一口径重跑 | `N/A` | 否 |
| cold reshard setup E2E | Store full lifecycle 有独立数据 | `N/A`，官方 bench 不计 setup | 否 |
| framework target 首次可用 E2E | Qwen3-Next p50 `68.998 s` | `N/A` | 否 |
| raw 单 region transport | Mooncake TE 有效 | 普通 NCCL P2P 有效，但不是 M2N | 仅诊断 |

因此，当前能确认的是 Mooncake 功能链和自身性能边界，不能确认 Mooncake 比
M2N 快或慢。后续只有在同一台支持 GIN 的机器上完成相同 topology、logical
bytes、validation 和 E2E 边界后，才允许计算 speedup。

### 3.2 H20 上的真实 M2N 结果

M2N 使用固定 NCCL `5067397c`、版本 2.30.7，成功构建
`libnccl_m2n.so` 和 `reshard_bench`。本地运行经历了以下阶段：

1. 上游 NCCL 创建部分 RC QP 时设置 `max_recv_wr=0`，H20 的 eRDMA verbs
   provider 返回 `EINVAL`。独立 verbs 探针验证 `recv_wr=0` 失败、`recv_wr=1`
   成功。
2. 只在 benchmark NCCL 源码的通用 `ncclIbQpCreate` 入口将 0 提升为 1 后，
   强制 `NCCL_NET=IB` 且禁用 P2P/SHM 的 2-rank all-reduce 从 1 MiB 到
   16 MiB 全部通过，`#wrong=0`。该补丁不属于 Mooncake，也没有提交上游。
3. M2N `DIRECT` 最小 1 source -> 1 destination 在第一轮 warmup 仍不能完成。
   GIN proxy 的 `IPut` completion 返回
   `status=21 (IBV_WC_GENERAL_ERR)`，随后 eRDMA QP 报
   `local work queue catastrophic error`；20 秒后由外层 timeout 终止。NCCL
   `ncclRmaIbProxyIPutSignal` 的 WR chain 是 payload `IBV_WR_RDMA_WRITE` 加
   signal `IBV_WR_ATOMIC_FETCH_AND_ADD`；失败 completion 的 `opcode=4` 对应
   后者，因此故障点是 M2N GIN atomic signal 路径，不是 logical-box planner。
4. M2N `RING` TP4 -> TP2 已越过 `ncclDevCommCreate`，但在 warmup 内等待
   GIN signal，20 秒受控复测仍未完成。禁用 IB 时 communicator 没有任何
   可用 GIN，
   `ncclDevCommCreate` 明确拒绝创建。

这说明普通 NCCL eRDMA transport 可运行，不代表 M2N 所需的 Device API GIN
数据面可运行。上述 timeout 不是延迟样本，失败前的字节也不能计算吞吐；本报告
不生成“超时下限吞吐”或使用未校验数据。

### 3.3 正确的复测矩阵

支持 GIN 的平台需要让 Mooncake 和 M2N 执行同一组 case：

| Case | source -> target | logical tensor | 目的 |
| --- | --- | --- | --- |
| TP shrink | TP4 -> TP2，dim0 -> dim0 | 64 MiB/256 MiB/1 GiB | split shard merge |
| TP expand | TP2 -> TP4，dim0 -> dim0 | 64 MiB/256 MiB/1 GiB | shard fan-out |
| cross-dim | TP4 -> TP2，dim0 -> dim1 | `[experts,out,in]` | N-D strided reshard |
| cross-dim | TP4 -> TP2，dim0 -> dim2 | `[experts,out,in]` | innermost-axis change |
| multi-tensor | 显式 PP route + expert tensor batch | 同一 manifest tensor 集 | 真实权重 revision |

每个 case 至少报告以下指标：

- warm `E2E p50/p95/max`：API 调用、设备完成和所有 target barrier；
- cold `E2E p50/p95`：runtime/session、allocation/registration、metadata、plan、
  首次 validated transfer；
- `validated logical GiB/s`：global unique bytes 除以 max-rank warm E2E；
- physical bytes、operation/region 数、峰值额外显存和 source serving 影响。

M2N 的 synthetic benchmark 使用 2D mesh 表示 replicate/shard，并不理解模型的
PP、EP、DP 语义。公平比较 multi-tensor case 时，由 Mooncake manifest 和测试
harness 显式提供 layer/expert ownership，再把每个 logical tensor descriptor
交给各 executor；不能让 M2N benchmark 的模型名推断或 dedup 代替真实语义。

### 3.4 官方 M2N E2E 参考

NVIDIA M2N README 给出的 GB200 NVL72 `reshard_model_bench` 结果是当前唯一可用
的真实 M2N E2E 参考，但硬件、模型、GPU 数和 workload 都与本次 H20 不同，
只能证明 M2N 的目标量级，不能和本地 Mooncake 数字计算比值：

| 模型 | GPU 配置 | M2N RING max E2E | M2N DIRECT max E2E |
| --- | --- | ---: | ---: |
| DeepSeek-V3 | 128T + 128G | `1447.44 ms` | `3837.74 ms` |
| Qwen3-235B | 64T + 64G | `988.75 ms` | `2175.16 ms` |

这组结果使用 `--validate --no-dedup`，统计所有 layer pattern；RING 相对 DIRECT
分别为 `2.65x` 和 `2.20x`。它是 NVIDIA 环境内两种 M2N algorithm 的对比，
不是 Mooncake 与 M2N 的对比。

### 3.5 辅助诊断：裸 G2G，不是 M2N

以下旧数据只用于判断 H20/NVLink 的 transport ceiling 和 Mooncake TE 自身
开销，不进入异构 reshard 主结论：

| 路径 | 拓扑/region | 饱和吞吐 |
| --- | --- | ---: |
| Mooncake `nvlink_intra` | 0 -> 4，64 MiB | `369.062 GiB/s` |
| 普通 NCCL strict one-way P2P | 0 -> 4，64 MiB | `314.629 GiB/s` |
| Mooncake `nvlink_intra` | 0-3 -> 4-7，64 MiB | `1474.255 GiB/s` |
| 普通 NCCL strict one-way P2P | 4 pair，64 MiB | `1254.082 GiB/s` |
| Mooncake RDMA | 0 -> 4，双 eRDMA | `20.389 GiB/s` |

| Payload | Mooncake 单 region completion | 普通 NCCL P2P completion |
| --- | ---: | ---: |
| 64 MiB | `0.181021 ms` | `0.225312 ms` |
| 256 MiB | `0.689038 ms` | `0.771438 ms` |
| 1 GiB | `2.718195 ms` | `3.234601 ms` |
| 4 GiB | `10.835100 ms` | `12.832107 ms` |

这些 completion 数据不包含 manifest、N-D planning、多个 source/destination
协调、GIN/ring forwarding、PP route 或 target activation。它们既不是完整
revision E2E，也不能用于估算 M2N 的 reshard 吞吐。

## 4. Store 权重 save/load E2E

| Protocol | 大小 | TP | upload p50 | load p50 | 完整 E2E p50/p95 | E2E 速率 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| TCP/local-copy | 256 MiB | 4 -> 2 | `28.363 ms` | `54.141 ms` | `84.025/94.722 ms` | `6.389 GB/s` |
| TCP/local-copy | 256 MiB | 2 -> 4 | `27.900 ms` | `54.105 ms` | `83.449/83.508 ms` | `6.434 GB/s` |
| TCP/local-copy | 1 GiB | 4 -> 2 | `112.339 ms` | `215.945 ms` | `329.753/330.094 ms` | `6.512 GB/s` |
| TCP/local-copy | 1 GiB | 2 -> 4 | `112.181 ms` | `215.920 ms` | `329.565/329.710 ms` | `6.516 GB/s` |
| RDMA, 2 HCA | 256 MiB | 4 -> 2 | `26.469 ms` | `50.285 ms` | `78.357/119.895 ms` | `6.852 GB/s` |
| RDMA, 2 HCA | 256 MiB | 2 -> 4 | `26.100 ms` | `50.569 ms` | `78.174/78.327 ms` | `6.868 GB/s` |

完整 E2E 包含 prepare、upload、commit、manifest get、plan load 和 load。
E2E 速率的 logical bytes 是 `upload bytes + load bytes`，即同一份权重按两次
数据移动计数；它不能与第 3 节只搬一次的 G2G GiB/s 直接比较。TCP TP4 -> TP2
和 RDMA TP4 -> TP2 的 p95 被个别 commit/metadata 长尾拉高，但 upload/load
本身稳定；因此表中同时保留 p50 与 p95，不能只报最好吞吐。

典型 metadata/control p50：prepare `0.17-0.20 ms`，manifest get
`0.11-0.13 ms`，plan load `0.35-0.36 ms`，commit `0.80-0.90 ms`。因此
256 MiB 以上对象的主瓶颈明确在数据路径，不在 manifest/planner。

本测试 master 未配置持久化 root，测量的是 RAM/L3 Store，不是 SSD 或 OSS。
GPU ranged load 在 `RealClient::execute_ranged_read` 中分配 host temp buffer，
Store GET 完成后再 `scatter_host_to_maybe_device`；这解释了 load 约为 upload
一半。未来直写 GPU 仍须保留 registered-buffer bounds、generation 和 lease
检查，不能用性能优化绕过这些校验。

## 5. 框架级可用 E2E

Qwen3-Next 实测使用 source TP4/EP4、target TP2/EP2，PP1、DP1；每个 target
启动前 drop page cache，交替执行 cold 与 manifest reuse，各 3 轮。

| 模式 | spawn -> ready p50 | spawn -> 首次推理完成 p50 | 相对 cold |
| --- | ---: | ---: | ---: |
| Checkpoint cold | `106.139 s` | `107.101 s` | 基线 |
| Live manifest reuse | `68.040 s` | `68.998 s` | 少 `38.104 s`，低 `35.6%`，`1.55x` |

`spawn -> 首次推理完成` 按每一轮的
`spawn_to_ready_s + first_generation_s` 求和后取中位数，不是简单相加两个独立
p50。manifest reuse 期间 source 共完成 202 次 probe，错误和响应 mismatch 都为
0；target 的文本、token IDs、token counts、finish reason 完全一致，token
logprobs 在既定容差内。这个指标最接近用户等待新 replica 可用的真实时间，
但 source 本身已预先运行，因此不包含 source 冷启动。

## 6. 完整性判断

| 层 | 状态 | 说明 |
| --- | --- | --- |
| Runtime manifest v1/v2 | 已实现 | 物理地址、shape、dtype、logical box、rank/generation 的唯一事实来源 |
| N-D planner | 已实现 | source/target 可使用不同 shard dims；输出 backend-neutral region |
| PP | 已实现 | 按显式 layer/tensor owner，输出 `(src_pp,dst_pp)` route |
| EP | 已实现 | expert 是 logical tensor dim0 coordinate，不依赖参数名 |
| DP | 已实现 | 选择完整 source replica；target DP 不引入重复存储 |
| TE lowering | 已实现 | lazy bounded segments；保留未来 M2N lowering 接口边界 |
| Store upload/load | 已实现（POC/library） | `prepare_upload/upload/commit/load_manifest/plan_load/load` |
| SGLang adapter | 已实现（个人分支） | runtime 导出模型语义和 GPU address inventory |
| 生产控制面 | 未实现 | revision barrier、节点选择、activation、retry/recovery |
| Lease/checksum | 部分实现 | generation/bounds 已校验；lease_id 未完整消费；checksum 未强制生成/验证 |

独立 expert allocation 不会被强制 pack/all-gather；planner 只在 logical box 上
求交，executor 按每个 fragment 的原始地址传输。alias 去重前，每个被选中的
非空 source/target fragment intersection 生成一个 `TransferRegion`；最终
operation 数可能因物理 alias 去重而减少。outer loops 在 lowering 时有界展开，
不按 row/element 预生成海量 `CopyRange`。

## 7. 本轮发现并修复的问题

1. classic `transfer_engine_bench` worker 未调用已有的
   `setWorkerDeviceIfNeeded()`。非 0 GPU worker 默认 current device 为 0，
   导致 CUDA IPC handle 和 sync event 建在错误 context；`GPU1 -> 5` 只有
   `32.632 GiB/s`。补齐一行后，同一原始复现达到 `369.199 GiB/s`，四路
   也全部恢复到约 `369 GiB/s`。
2. Store 性能 payload 把 RDMA 请求也硬编码成 `store-cuda-tcp-*`，同时把
   请求协议误当成实际 data plane。现已将 backend 改为协议无关标识，并在
   `transport` 字段单独记录 requested protocol、运行时策略选择和
   `MC_STORE_MEMCPY`；单元测试直接覆盖最终 payload 的 TCP/RDMA、单卡/多卡
   组合。

## 8. 验证

- 当前增量：C++ `transfer_engine_bench` Release 重建通过。
- 当前增量：`test_weight_store_gpu_e2e.py` 为 `45 passed, 4 skipped`。
- 精确 legacy TP 回归：TP4 -> TP8、TP8 -> TP4 均通过；planner 测试文件
  `49 passed`。
- 当前 HEAD 六个相关测试文件重新执行：`252 passed, 4 skipped`。
- 当前增量：ruff check/format check 通过，`git diff --check` 通过。
- N-D 基线：Mooncake `253 passed, 12 skipped`；SGLang `114 passed`；CUDA
  RDMA TE 4 项和 CUDA Store 3 项通过。
- Qwen3-Next 真实启动：spawn-to-ready p50 为 `106.139/68.040 s`；
  spawn-to-first-response p50 为 `107.101/68.998 s`；source serving
  continuity 和响应一致性通过。该项是框架级 E2E，不等同于 raw TE 吞吐。
- NCCL `5067397c` 和 `nccl_m2n` 构建通过。benchmark-only eRDMA QP 兼容后，
  强制走 IB 的 2-rank all-reduce 1-16 MiB 全部 `#wrong=0`。
- M2N 负向验证：1 -> 1 DIRECT 和 TP4 -> TP2 RING 均进入 warmup，但 GIN
  `IPut`/QP 失败并由 20 秒 timeout 终止；没有 `VALIDATION PASSED`，因此没有
  纳入任何吞吐或 E2E 样本。
- 当前环境没有 `clang-format` 可执行文件；本轮 C++ 变更只有一行并已编译，
  但未执行独立 clang-format check。

## 9. 证据与后续

性能原始日志已同步到：

```text
/Users/gaobo/Documents/mooncake/.vin_stage/results-20260722/
```

其中 `mooncake-nvlink-*`、`mooncake-rdma-*`、`nccl-one-way-*` 和
`store-cuda-*` 分别支撑第 3.5、4 节数据。普通 NCCL 日志只支持 raw P2P
辅助诊断，不支持 M2N 结论。Qwen3-Next 启动结果位于：

```text
/Users/gaobo/Documents/mooncake/.vin_stage/qwen3-next-final-9e850de1-2db0c7842-benchmark-result.json
```

单 region E2E 原始日志位于：

```text
/Users/gaobo/Documents/mooncake/.vin_stage/results-20260722-e2e/
```

`mooncake-e2e-*` 是 batch1/thread1 的串行 completion 结果；
`nccl-warm-e2e-*` 是 64 GiB 等字节 warmup 后的正式同步结果；
`nccl-e2e-*` 保留初始短 warmup 及其双峰，不作为正式中位数输入；
`nccl-debug-1g/*` 记录 direct-pointer、channel 和 chunk 配置证据。

NCCL M2N 构建和运行证据位于：

```text
/Users/gaobo/Documents/mooncake/.vin_stage/results-20260722-m2n/
```

`nccl-erdma-after-qp-compat.log` 记录普通 collective 的通过结果；
`m2n-direct-1to1-info.log` 记录最小 GIN `IPut status=21`；
`m2n-ring-tp4-to-tp2-after-qp-compat.log` 记录真实 TP4 -> TP2 RING 在 warmup
中的 QP fatal event。benchmark-only 补丁保存在：

```text
/Users/gaobo/Documents/mooncake/.vin_stage/nccl-erdma-max-recv-wr.patch
```

证据边界：Release build、当前 pytest/ruff 以及 N-D 基线 pass count 是本轮和
实现阶段的命令输出摘要，未混入吞吐日志目录。M2N 负向日志已经归档，只能
证明当前 H20/eRDMA 环境没有有效 M2N iteration，不能证明其他 GIN 平台上的
M2N 性能。第三方独立复核时应在上述固定 commit 上重新执行构建和测试。

推荐下一阶段按优先级推进：

1. 为 Store MEMORY ranged GET 增加 registered GPU destination 的直接 RDMA/
   CUDA path，消除 host staging，并增加 direct/staged 字节计数。
2. 在 SGLang/slime 控制面接 revision barrier、target activation 和失败回滚。
3. 让 lease_id 成为执行期强校验，补强 checksum 生成和 load 校验。
4. 在支持 NCCL Device API GIN 的平台复测 M2N executor，再决定是否增加
   `TransferRegion -> M2N` lowering；manifest 和 planner 无需因此改变。

公开参考：

- [NVIDIA NCCL M2N README（固定到 5067397c）](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/contrib/nccl_m2n/README.md)
- [NVIDIA M2N reshard benchmark（固定到 5067397c）](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/contrib/nccl_m2n/benchmarks/reshard_bench.cc)
- [Mooncake group semantics RFC #2282](https://github.com/kvcache-ai/Mooncake/issues/2282)
