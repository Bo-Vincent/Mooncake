# 异构权重转换完整性与性能验证报告

日期：2026-07-22
分支：`vin/heterogeneous-weight-transfer-v2`
N-D 核心基线：Mooncake `9e850de122c1e13af732c03330fa827a774f7d7c`
SGLang adapter 基线：`2db0c78425eb7952651149dc0c662fd3ea6f6108`

## 1. 结论

1. manifest v2、N-D logical-box planner、PP route、EP leading coordinate、
   DP replica 选择、Store/TE 有界 lowering 已经实现；旧 manifest 和旧 TP
   路径保持兼容。
2. H20 同机 live G2G 的主路径应使用 `nvlink_intra`。64 MiB region 下，
   Mooncake 单路平均 `369.062 GiB/s`，四路唯一 payload 合计
   `1474.255 GiB/s`；严格单向 NCCL send/recv 对照分别为
   `314.629 GiB/s` 和 `1254.082 GiB/s`，Mooncake 高 `17.3%` 和 `17.6%`。
3. 当前 RDMA fallback 单路平均 `20.389 GiB/s`。它适合跨节点，不应和
   同机 NVLink 数字混用。
4. WeightStore 的六个核心 API 和 CUDA 数据路径已实现，可以作为 POC/library
   使用；但生产控制面、revision barrier、target activation、故障恢复、完整
   lease 消费仍未接通，因此不能宣称生产服务已经完整。
5. Store CUDA ranged load 当前经过 host 临时 buffer，再 H2D。H20 实测
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

- Mooncake raw TE：64 MiB block，batch 8，8 worker threads，8 GiB circular
  buffer，15 秒持续单向 write；初始化、注册和 open segment 不计时。
- NCCL strict one-way：每个 GPU pair 使用独立 2-rank communicator，
  `ncclCommRegister`，warmup 后计时，只执行 source 到 target 的
  `ncclSend/ncclRecv`；分母只计算唯一 payload，末尾校验 target 首尾内容。
- 官方 `sendrecv_perf` 是双向同时传输，只作为 full-duplex reference，不能
  直接当作单向权重迁移。
- `GB/s` 为十进制；`GiB/s` 为二进制。主对比统一使用 `GiB/s`。
- Store 测试使用一个进程内的 source GPU0-3、target GPU4-7，3 次 warmup、
  10 次正式采样；metadata、prepare、commit、plan 与数据传输分别计时。
- Store 的机器可读 payload 将 backend 记为协议无关的 CUDA/pre-registered
  路径，并把请求协议、`MC_STORE_MEMCPY` 和“运行时选择策略”单独记录。
  Python API 当前不返回最终 `TransferStrategy`；下表的 local-copy/RDMA 判断
  来自本次 Store transport 日志，不仅依据请求协议。

## 3. Raw G2G 性能

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

Mooncake 与 NCCL 的 64 MiB 点最接近当前 Store 数据路径：本次 raw TE
benchmark 使用 64 MiB block，WeightStore 的 `max_range_bytes` 默认值也为
64 MiB。TE 的 N-D lowering 限制 batch operation 和 region segment 数量，
但不会强制把每个 region 切成 64 MiB。Mooncake 四路的十进制吞吐约为
`1.583 TB/s`。

官方 `sendrecv_perf` 的 4 GiB 点为 `341.08 GB/s/方向`，但测试同时发送和
接收，物理 aggregate 约为两倍；本报告不使用它证明 Mooncake 超过 NCCL。

### 3.1 NCCL M2N 状态

当前 `nccl_m2n` 是 NVIDIA `contrib/` 下的 experimental standalone preview，
不是 NCCL core runtime。本机已构建 `libnccl_m2n.so` 和 `reshard_bench`，但
运行时在 eRDMA 上的 `ncclDevCommCreate`/GIN 能力检查失败；禁用 IB 后仍因
缺少 GIN 失败。因此本报告将 M2N 标为“构建成功，环境不支持运行”，不拿
普通 NCCL P2P 结果冒充 M2N reshard 结果。

## 4. Store 权重 save/load 性能

| Protocol | 大小 | TP | upload p50 | load p50 | e2e p50 |
| --- | ---: | --- | ---: | ---: | ---: |
| TCP/local-copy | 256 MiB | 4 -> 2 | `9.464 GB/s` | `4.958 GB/s` | `6.389 GB/s` |
| TCP/local-copy | 256 MiB | 2 -> 4 | `9.621 GB/s` | `4.961 GB/s` | `6.434 GB/s` |
| TCP/local-copy | 1 GiB | 4 -> 2 | `9.558 GB/s` | `4.972 GB/s` | `6.512 GB/s` |
| TCP/local-copy | 1 GiB | 2 -> 4 | `9.572 GB/s` | `4.973 GB/s` | `6.516 GB/s` |
| RDMA, 2 HCA | 256 MiB | 4 -> 2 | `10.142 GB/s` | `5.338 GB/s` | `6.852 GB/s` |
| RDMA, 2 HCA | 256 MiB | 2 -> 4 | `10.285 GB/s` | `5.308 GB/s` | `6.868 GB/s` |

典型 metadata/control p50：prepare `0.17-0.20 ms`，manifest get
`0.11-0.13 ms`，plan load `0.35-0.36 ms`，commit `0.80-0.90 ms`。因此
256 MiB 以上对象的主瓶颈明确在数据路径，不在 manifest/planner。

本测试 master 未配置持久化 root，测量的是 RAM/L3 Store，不是 SSD 或 OSS。
GPU ranged load 在 `RealClient::execute_ranged_read` 中分配 host temp buffer，
Store GET 完成后再 `scatter_host_to_maybe_device`；这解释了 load 约为 upload
一半。未来直写 GPU 仍须保留 registered-buffer bounds、generation 和 lease
检查，不能用性能优化绕过这些校验。

## 5. 完整性判断

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

## 6. 本轮发现并修复的问题

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

## 7. 验证

- 当前增量：C++ `transfer_engine_bench` Release 重建通过。
- 当前增量：`test_weight_store_gpu_e2e.py` 为 `45 passed, 4 skipped`。
- 当前增量：ruff check/format check 通过，`git diff --check` 通过。
- N-D 基线：Mooncake `253 passed, 12 skipped`；SGLang `114 passed`；CUDA
  RDMA TE 4 项和 CUDA Store 3 项通过。
- Qwen3-Next 真实启动基线：checkpoint cold p50 `106.139 s`，live manifest
  reuse p50 `68.040 s`，加速 `1.56x`；source serving continuity 和响应一致性
  通过。该项是完整框架启动数据，不等同于 raw TE 吞吐。
- 当前环境没有 `clang-format` 可执行文件；本轮 C++ 变更只有一行并已编译，
  但未执行独立 clang-format check。

## 8. 证据与后续

性能原始日志已同步到：

```text
/Users/gaobo/Documents/mooncake/.vin_stage/results-20260722/
```

其中 `mooncake-nvlink-*`、`mooncake-rdma-*`、`nccl-one-way-*` 和
`store-cuda-*` 分别支撑第 3、4 节吞吐数据。Qwen3-Next 启动结果位于：

```text
/Users/gaobo/Documents/mooncake/.vin_stage/qwen3-next-final-9e850de1-2db0c7842-benchmark-result.json
```

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
