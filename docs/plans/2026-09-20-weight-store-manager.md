# WeightStoreManager 重构实施方案

**目标：** 在同一 Master 进程和 RPC server 内，将 Weight 管理编排移入独立组件，保留完整集成分支功能，按所属功能提交重写历史，并重新完成 review、测试和真实 E2E。

**架构：** WeightStoreManager 持有 WeightMetadataStore、Weight 操作锁及生命周期配置。MasterStoreBackend 提供少量真实需要的底层操作，由 MasterService 适配现有 object/group、replica、durable oplog 能力。HA 仍统一协调 snapshot 和 promotion，调用 manager 导出/恢复 Weight 状态。

**技术栈：** C++、现有 coro_rpc/struct_pack、GoogleTest、Python WeightStore、TCP E2E。

## 基线与约束

- PR1 已合入基线：`97030cacfbf8db7d0098e35da46c58abbae04593`。
- 按最新主线重新集成：`9b5adcd4d`；Reshard 设计文档统一位于
  `docs/source/design/mooncake-reshard/`，API 文档交叉链接跟随新目录。
- 原 PR2：`19a6dc5f15708d9ffad6fbdafd5a227a99712809`。
- 原集成头：`1644a877c9d5ec618fde59c88072c769924ecdad`。
- PR1 独立 metadata 状态机原则上保持不变；发现必须调整时记录具体契约与证据。
- 普通 Store 的数据结构和业务行为保持现有实现，仅调整必要接线与 Weight 保护调用。
- 保留现有 RPC 地址和 wire contract。RPC wrapper 可保留稳定入口符号，内部直接转发 manager，避免更换类导致 RPC 标识变化。
- 不将 MasterService 指针或内部 map/锁暴露给 manager；不建立第二套 object metadata、replica accounting 或 HA writer。
- 新增/修改代码注释使用英文。该文档是实施记录，不是 PR 描述。

## 组件边界

```text
Master RPC server
  ├─ 普通 Store handlers → MasterService
  └─ Weight handlers → WeightStoreManager
                         ├─ WeightMetadataStore
                         ├─ group/lineage 操作锁
                         ├─ policy / lease / upsert / reconcile 编排
                         └─ MasterStoreBackend
                              └─ 现有 object/group、replica、oplog 基础设施
Master HA coordinator → manager snapshot export/restore
```

Backend 首先只提供 group 成员观察和 durable append；删除、promotion、offload 等能力随对应功能提交加入。底层返回值包含执行所需的身份与状态，manager 不使用裸内部引用。

Weight 锁的所有权属于 manager；object/replica 锁仍由 Store 持有。固定 lineage → group → 底层对象操作的获取顺序。普通 Store 的删除、驱逐和后台任务仍需检查 managed-group、lease/generation fence；不能用 manager 的单独锁替代这些保护。durable callback 必须在 manager 销毁前完成或停止，snapshot 保持原有持久化序列边界。

## 执行步骤与提交划分

1. 保存 PR2/集成精确 SHA 备份，建立 `vin/` 前缀隔离工作分支；记录原提交到新提交映射。
2. 从 PR1 基线重建 PR2。按职责拆为：backend 契约与适配；manager import/commit/query；durable publication/oplog；snapshot/standby；revision lease 与并发测试。每个提交同时带上对应测试，避免末尾集中修补或独立大格式提交。
3. 在集成分支逐段重放下游提交，将 group lifecycle、reconciliation、RPC、policy/migration、Python、auto migration、upsert 改到各自引入功能的提交。每个功能保持明确边界；必要时拆分过大的历史提交，fixup 归入所属功能。
4. 对物理存储操作保留 Master 实现，通过 backend 调用；对 Weight 决策、状态发布、锁、配置和后台调度移入 manager。同步测试 peer 与 snapshot 调用方。
5. 更新架构文档中的类/所有权和调用路径。输出实际 commit 列表与逐项变更统计。
6. 使用 review skill 审阅 PR2 和完整集成差异。正式问题必须有精确提交上的真实路径 RED；修复在所属提交并验证同一用例 GREEN。
7. 构建和测试各语义边界；完成 PR2、集成最终提交的格式、pre-commit、定向 CTest、Python 测试及真实 TCP E2E。
8. 验证 PR2 是集成祖先，使用 range-diff/功能矩阵检查无遗漏，保持备份后通过 ssh vin 同步个人 fork，并回读 SHA。

## 验证与验收

- [ ] PR1 是否需要修改的结论与代码依据。
- [ ] manager 不直接访问 Master 私有字段；backend 无重复数据所有权。
- [ ] import/commit/retry、manifest 对象核验、lease acquire/renew/release 语义保持。
- [ ] durable append 失败/延迟回调、snapshot、standby promotion 保持原有时序。
- [ ] 普通 Store 删除/驱逐与 Weight lease、delete、migration/upsert 并发保护仍生效。
- [ ] policy HOT/COLD/MIXED、PINNED/MANUAL/AUTO、reconciliation 和 upsert 两模式完整保留。
- [ ] RPC tenant binding、RPC 标识、客户端调用链完整保留。
- [ ] 每个语义提交可独立构建并通过对应测试；最终 PR2 与集成分别验证。
- [ ] PR2 定向 CTest；集成 16 项 CTest、Python model_weight_store 套件通过。
- [ ] 构建真实 `mooncake_master` 和 Python `store` 扩展，执行 `mooncake-store/tests/e2e/run_weight_upsert_tcp_e2e.sh`。
- [ ] 审计现有 E2E 的行为覆盖；补充其缺失的 managed put/get、lease/delete、policy/migration 真实路径，不能以 mock 或 CTest 替代。
- [ ] E2E 记录主机、源码 SHA、二进制/扩展来源、完整命令、日志和结果。TCP 成功不标作 RDMA 验证。
- [ ] review / AI taste / changed-line format / PR-scoped pre-commit 完成。
- [ ] 实际提交映射、分支祖先关系、远端 SHA 完整核对。
