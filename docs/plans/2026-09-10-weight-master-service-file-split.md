# Weight 管理组件拆分

原方案仅将 MasterService 成员函数移到两个文件。当前方案改为独立的
WeightStoreManager 组件；本文件保留原路径，避免已有文档链接失效。

## 所有权与调用路径

```text
同一个 Master RPC server
  ├─ 普通 Store RPC → MasterService
  └─ Weight RPC → WeightStoreManager
                    ├─ WeightMetadataStore
                    ├─ group / lineage 操作锁
                    ├─ policy / lease / upsert / migration / delete
                    └─ WeightStoreBackend 接口
                         └─ MasterStoreBackend → 现有 Store 基础设施
Master HA → manager.ExportSnapshot / RestoreSnapshot
```

- WeightMetadataStore 是原有 metadata 状态机，不新增重复的 metadata。
- Manager 负责 Weight 决策和流程；backend 只适配实际需要的底层操作。
- MasterService 保留 object/group、replica、segment 和 HA writer 的原有状态。
- RPC 地址、handler 标识和 wire contract 保持不变，不启动第二个服务。
- 通用 Store 路径中的 managed-group 与 lease 保护继续生效。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| weight_store_manager.h / .cpp | Weight 状态所有权、锁和生命周期编排 |
| weight_store_backend.h | 窄接口与 Master 适配器声明 |
| master_store_backend.cpp | 调用已有 group/object、promotion/offload、删除和 OpLog 能力 |
| weight_metadata_store.h / .cpp | revision、lineage、lease、operation 状态机与快照 |
| master_service.cpp | 通用 Store 流程、物理操作和必要接线 |
| rpc_service.cpp | 保留 RPC handler，绑定 tenant 后转发 manager |

文件均位于 mooncake-store 的 include/ 或 src/ 下。

## 并发与恢复约束

锁按 lineage → group → 底层对象操作获取。底层对象锁仍由 Store 管理，不能用
manager 的操作锁替代。删除、迁移和 upsert 保持原有 lease/generation fence。

状态发布必须等待 durable append；HA 快照和恢复仍由 Master 协调。Master 停止
后台线程并排空 OpLog callback 后，才可销毁 manager 和 backend。

## 提交与验收

PR1 的独立 metadata 核心保持原基线。PR2 引入 manager/backend 与 import、durable
publication、HA、lease；后续集成能力按所属功能提交迁移，不在末尾堆积修补。

验收包含精确提交构建、PR2 和集成 CTest、Python 测试，以及真实 master + Python
扩展的 TCP E2E。还需验证普通 Store 保护和 Weight 并发路径、RPC 兼容性与 HA
恢复。结构检索只能证明代码归属，不能代替运行测试或作为正式缺陷复现。

实施过程和未完成项见 [重构实施方案](2026-09-20-weight-store-manager.md)。
