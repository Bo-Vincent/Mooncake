# 异构权重转换完整性与性能验证报告

日期：2026-07-22
分支：`vin/heterogeneous-weight-transfer-v2`
N-D 核心基线：Mooncake `9e850de122c1e13af732c03330fa827a774f7d7c`
SGLang adapter 基线：`2db0c78425eb7952651149dc0c662fd3ea6f6108`

## 1. 结论

1. manifest v2、N-D logical-box planner、PP route、EP leading coordinate、
   DP replica 选择、Store/TE 有界 lowering 已经实现；旧 manifest 和旧 TP
   路径保持兼容。
2. H20 同机 live G2G 的主路径应使用 `nvlink_intra`。在吞吐饱和模式下，
   64 MiB region 的 Mooncake 单路为 `369.062 GiB/s`，严格单向 NCCL
   send/recv 为 `314.629 GiB/s`，Mooncake 高 `17.3%`。
3. 单 region warm E2E 不能用饱和吞吐直接倒推。按每次提交后等待完成的
   同步口径，64 MiB/256 MiB/1 GiB/4 GiB 的 Mooncake E2E 分别为
   `0.181/0.689/2.718/10.835 ms`；NCCL 为
   `0.225/0.771/3.235/12.832 ms`。Mooncake 延迟低 `10.7%-19.7%`。
4. Qwen3-Next 框架级 E2E 中，从目标进程拉起到首次确定性推理完成，
   checkpoint cold p50 为 `107.101 s`，live manifest reuse p50 为
   `68.998 s`，节省 `38.104 s`，缩短 `35.6%`，加速 `1.55x`。
5. 当前 RDMA fallback 单路平均 `20.389 GiB/s`。它适合跨节点，不应和
   同机 NVLink 数字混用。
6. WeightStore 的六个核心 API 和 CUDA 数据路径已实现，可以作为 POC/library
   使用；但生产控制面、revision barrier、target activation、故障恢复、完整
   lease 消费仍未接通，因此不能宣称生产服务已经完整。
7. Store CUDA ranged load 当前经过 host 临时 buffer，再 H2D。H20 实测
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

- 饱和吞吐模式：Mooncake 使用 64 MiB block、batch 8、8 worker threads、
  8 GiB circular buffer，持续单向 write 15 秒；NCCL 先异步提交多次
  `ncclSend/ncclRecv`，末尾统一同步。两者都排除初始化和注册时间。
- warm region E2E：Mooncake 使用 batch 1、1 worker，串行执行
  `allocateBatchID -> submitTransfer -> poll completion -> freeBatchID`；每轮
  运行 5 秒，取 5 个独立进程的 operation mean 中位数。NCCL 每个 operation
  提交一组 send/recv 后同步 source/target 两条 stream，直接记录每次 wall
  time；每种大小先搬约 64 GiB，再取 5 个独立进程 mean 的中位数。
- warm region E2E 不包含 engine/communicator 初始化、显存分配、memory
  registration、segment open 和 manifest/planner。它回答“运行中的 source 与
  target 已建立后，一次 region 提交到完成多久”，不是冷启动时间。
- NCCL strict one-way：每个 GPU pair 使用独立 2-rank communicator，
  `ncclCommRegister`，只执行 source 到 target 的 `ncclSend/ncclRecv`；分母
  只计算唯一 payload，末尾校验 target 首尾内容。
- 官方 `sendrecv_perf` 是双向同时传输，只作为 full-duplex reference，不能
  直接当作单向权重迁移。
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

## 3. Raw G2G 吞吐与 E2E

### 3.1 饱和吞吐

| 路径 | 拓扑/region | 结果 | 备注 |
| --- | --- | ---: | --- |
| Mooncake `nvlink_intra` | 0 -> 4，64 MiB | `369.062 GiB/s` | 3 轮：369.034/369.025/369.127 |
| NCCL strict one-way | 0 -> 4，64 MiB | `314.629 GiB/s` | 200 iterations，validation OK |
| Mooncake `nvlink_intra` | 0-3 -> 4-7，64 MiB | `1474.255 GiB/s` | 4 路：368.406/368.330/368.436/369.083 |
| NCCL strict one-way | 4 pair，64 MiB | `1254.082 GiB/s` | 4 个独立 communicator，validation OK |
| NCCL strict one-way | 1 pair，256 MiB | `336.087 GiB/s` | 50 iterations |
| NCCL strict one-way | 4 pair，256 MiB | `1342.477 GiB/s` | 唯一 payload 合计 |
| NCCL strict one-way | 1 pair，1/4 GiB | `311.911/312.439 GiB/s` | 大单操作点 |
| NCCL strict one-way | 4 pair，1/4 GiB | `1247.063/1249.601 GiB/s` | 大单操作点 |
| Mooncake RDMA | 0 -> 4，双 eRDMA | `20.389 GiB/s` | 3 轮：20.346/20.493/20.327 |

#### 同口径 G2G 对比

| 单向 workload | Mooncake | NCCL | Mooncake/NCCL | 结论 |
| --- | ---: | ---: | ---: | --- |
| 1 pair，64 MiB | `369.062 GiB/s` | `314.629 GiB/s` | `1.1730x` | 高 `17.301%` |
| 4 pair，64 MiB 唯一 payload 合计 | `1474.255 GiB/s` | `1254.082 GiB/s` | `1.1756x` | 高 `17.557%` |

因此在当前 H20/NV18、64 MiB 权重 region、单向 G2G 口径下，Mooncake 已经
对齐并超过 NCCL。该结论只适用于本表环境和 workload，不外推为所有 GPU、
消息大小或双向 collective 下都更快。

Mooncake 与 NCCL 的 64 MiB 点最接近当前 Store 数据路径：本次 raw TE
benchmark 使用 64 MiB block，WeightStore 的 `max_range_bytes` 默认值也为
64 MiB。TE 的 N-D lowering 限制 batch operation 和 region segment 数量，
但不会强制把每个 region 切成 64 MiB。Mooncake 四路的十进制吞吐约为
`1.583 TB/s`。

官方 `sendrecv_perf` 的 4 GiB 点为 `341.08 GB/s/方向`，但测试同时发送和
接收，物理 aggregate 约为两倍；本报告不使用它证明 Mooncake 超过 NCCL。

### 3.2 单 region warm E2E

| Payload | Mooncake E2E | NCCL E2E | Mooncake 延迟降低 | Mooncake 串行速率 | NCCL 同步速率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 MiB | `0.181021 ms` | `0.225312 ms` | `19.66%` | `345.263 GiB/s` | `277.394 GiB/s` |
| 256 MiB | `0.689038 ms` | `0.771438 ms` | `10.68%` | `362.825 GiB/s` | `324.070 GiB/s` |
| 1 GiB | `2.718195 ms` | `3.234601 ms` | `15.97%` | `367.891 GiB/s` | `309.157 GiB/s` |
| 4 GiB | `10.835100 ms` | `12.832107 ms` | `15.56%` | `369.171 GiB/s` | `311.718 GiB/s` |

表中 E2E 都是 5 个独立进程的 operation mean 中位数。Mooncake 各轮 mean
范围依次为 `0.180910-0.182117 ms`、`0.688565-0.690094 ms`、
`2.718043-2.718454 ms`、`10.834301-10.836680 ms`。NCCL 对应范围为
`0.225242-0.226106 ms`、`0.770142-0.771688 ms`、
`3.232938-3.235064 ms`、`12.830775-12.835337 ms`；各进程 p95 的中位数
分别为 `0.227633/0.778682/3.243100/12.850401 ms`。当前 Mooncake benchmark
只记录进程内 operation mean，没有保存逐 operation histogram，因此不能伪造
Mooncake p95；后续若把 latency histogram 加入 benchmark，再补严格 tail 对比。
Mooncake 的 operation mean 来自 batch1/thread1 串行循环的 wall duration /
completed batch count；日志中的 duration 只显示两位小数，因此表内使用与该比值
等价的 `block GiB / 六位精度 GiB/s` 恢复数值。这和从 batch8/thread8 饱和
吞吐推算单次延迟不是同一个口径。

第一次只 warmup 10 次的 NCCL 采样出现过进程级双峰：64 MiB 一轮为
`2.553 ms`，1 GiB 也出现过约 `7.95 ms`，而其他进程明显更快。INFO 日志确认
1 GiB debug 对照的 6 个进程都使用 32 个 channel、`P2P/direct pointer` 和
512 KiB P2P chunk，没有切到网络 transport。将每种大小的 warmup 统一为约
64 GiB 后，5 轮数据落入上述窄区间。旧数据仍保存在 `nccl-e2e-*`，正式数据使用
`nccl-warm-e2e-*`；这里只能确认它属于进程冷态/瞬态，尚不能把根因严格归结
为 GPU 时钟或某个 NCCL protocol。

这一节补齐了单个 region 的完成时间，但还不是完整权重 revision 的总耗时。
完整 revision 会受 tensor 数量、N-D region 数、并发窗口、PP route 分组和
控制面 barrier 影响；不能简单用“模型字节数 / 本表速率”替代实际 E2E。

### 3.3 NCCL M2N 状态

当前 `nccl_m2n` 是 NVIDIA `contrib/` 下的 experimental standalone preview，
不是 NCCL core runtime。本机已构建 `libnccl_m2n.so` 和 `reshard_bench`，但
运行时在 eRDMA 上的 `ncclDevCommCreate`/GIN 能力检查失败；禁用 IB 后仍因
缺少 GIN 失败。因此本报告将 M2N 标为“构建成功，环境不支持运行”，不拿
普通 NCCL P2P 结果冒充 M2N reshard 结果。

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
- 当前环境没有 `clang-format` 可执行文件；本轮 C++ 变更只有一行并已编译，
  但未执行独立 clang-format check。

## 9. 证据与后续

性能原始日志已同步到：

```text
/Users/gaobo/Documents/mooncake/.vin_stage/results-20260722/
```

其中 `mooncake-nvlink-*`、`mooncake-rdma-*`、`nccl-one-way-*` 和
`store-cuda-*` 分别支撑第 3、4 节吞吐数据。Qwen3-Next 启动结果位于：

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

证据边界：Release build、当前 pytest/ruff 以及 N-D 基线 pass count 是本轮和
实现阶段的命令输出摘要，未混入吞吐日志目录；M2N 失败日志也未归档，因此
M2N 结论只表示本机环境探测结果，不作为性能对比证据。第三方独立复核时应在
上述固定 commit 上重新执行构建和测试。

推荐下一阶段按优先级推进：

1. 为 Store MEMORY ranged GET 增加 registered GPU destination 的直接 RDMA/
   CUDA path，消除 host staging，并增加 direct/staged 字节计数。
2. 在 SGLang/slime 控制面接 revision barrier、target activation 和失败回滚。
3. 让 lease_id 成为执行期强校验，补强 checksum 生成和 load 校验。
4. 在支持 NCCL Device API GIN 的平台复测 M2N executor，再决定是否增加
   `TransferRegion -> M2N` lowering；manifest 和 planner 无需因此改变。

公开参考：

- [NVIDIA NCCL M2N README（固定到 5067397c）](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/contrib/nccl_m2n/README.md)
- [NVIDIA nccl-tests sendrecv 实现（固定到 a0b82b22）](https://github.com/NVIDIA/nccl-tests/blob/a0b82b2260cf5152b9f8c061bbf7eaf0ba096432/src/sendrecv.cu)
- [Mooncake group semantics RFC #2282](https://github.com/kvcache-ai/Mooncake/issues/2282)
