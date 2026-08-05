# 异构权重 Benchmark 执行阶段计划

日期：2026-07-22

> 用户要求所有改动保留在个人分支工作树中，不执行 commit、push 或社区 PR。

## Task 1：控制协议与 runtime manifest wire contract

- 新增长度前缀 JSON 控制协议，限制单消息大小并保留 socket timeout。
- 新增 v1/v2 runtime manifest 的显式序列化与反序列化；不序列化 `owner`。
- 先写 round-trip、畸形 schema、超长消息和连接中断测试，再实现代码。

验证：定向 pytest、ruff、`git diff --check`。

## Task 2：统一结果 schema 与 M2N wrapper

- 解析官方 `reshard_bench` validate/timing 输出。
- 分开执行 one-shot cold run 和 warm steady run。
- 用结构化 argv 启动 OpenMPI，记录 subprocess wall time、退出码和原始日志。
- 每个 cold/steady 子进程返回后先落 stdout/stderr，再解析和判定 validation，确保
  失败运行也有证据。
- backend 输出不完整或 validate 失败时禁止计算吞吐。

验证：fixture stdout parser、超时/非零退出、cold/steady 聚合单测。

## Task 3：Mooncake 跨机 G2G runner

- target role：初始化 TE、按 target mesh 分配/注册 CUDA buffer、返回真实 manifest、
  本地分块验证、逆序清理。
- source role：按 source mesh 分配/注册/fill，接收 target manifest，规划并执行
  warmup/iteration，校验 receipt bytes。
- 保持一 fragment 一 allocation；DP 只由 planner 选择 replica；cross-dim 由 N-D
  region lowering 执行。

验证：pure geometry、manifest round-trip、fake engine 生命周期、socketpair 双角色
测试；远端恢复后补真实 CUDA/RDMA smoke。

## Task 4：双机编排与环境门禁

- runner 只在配置 `execution_enabled=true` 且 `VIN_RUN_BENCHMARK=1` 时运行。
- 通过 SSH/rsync 启动 target/source，不使用 scp。
- 外层 runner 从启动 target 前计时到 source/target 都退出并完成进程回收，生成与
  M2N 同层级的 `process_wall_ms`；两端日志先落盘再解析。
- 正式运行前记录并停止两台机器的 `amperf.service`，执行 strict isolation gate。
- 任一 case 结束均保留机器可读 JSON 和原始日志。

验证：小尺寸 TP2 -> TP4 correctness smoke，随后 TP4 -> TP8 与 TP8 -> TP4。

## Task 5：正式对比与恢复

- 对每个可物理执行 case 分别采集 M2N 与 Mooncake one-shot cold 和 warm steady。
- 同时报告 E2E、first update、steady p50/mean/p95 和 logical GiB/s。
- cross-dim operation 数量过大时如实报告 lowering 拒绝或性能，不修改 guard。
- 恢复 `amperf.service` 原状态，确认无残留 benchmark 进程，再输出中文报告。

## 执行结果

Task 1-5 已完成。双机环境已恢复，五个物理 case 的 Mooncake 与 NCCL M2N
correctness 均通过；公平性修正、hot update、进程 E2E、抖动范围和 lowering
瓶颈见 [实测报告](2026-07-22-heterogeneous-weight-benchmark-results.md)。
