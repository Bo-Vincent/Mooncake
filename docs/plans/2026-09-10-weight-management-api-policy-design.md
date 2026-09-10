# Weight Management API 与驻留策略补充设计

状态：提议

范围：`mooncake.reshard.weight.WeightStore` 与 Mooncake Store Weight
Management

基础：现有 manifest-backed Weight Management

依赖：不依赖 Transfer Engine Reshard PR

当前状态（2026-09-10）：上游 `origin/main` 为 `b9059252`，尚未包含
revision-level managed Weight Management；main 已有 unmanaged manifest-backed
`WeightStore`。新增管理代码只存在于当前 feature branch，因此必须在合入前直接
完成 `WeightStore` 公共 API 的命名收敛，不保留未发布 managed API 的兼容层。

## 1. 目标与边界

把一个不可变 Weight Revision 作为 Store 的一等公民，并继续由
`WeightStore` 提供领域管理接口。本设计补齐：

- 统一的 `weight_*` Python API；
- 可持久化、可更新的 revision-level policy；
- `HOT`、`COLD`、`MIXED`、`MIGRATING` 语义；
- 显式和自动 residency migration；
- query、update、operation query 与 group delete；
- group 级生命周期，禁止单个 Tensor/fragment 独立驱逐。

管理聚合根是 Weight Revision。Tensor 语义保留在不可变 manifest 中，
不建立独立 Tensor metadata record。

## 2. 架构与数据权威

```text
Slime/SGLang/Megatron/未来 DTensor Adapter
                    |
                    v
             WeightStore.weight_*
        policy / lease / manifest / reshard plan
                    |
                    v
              Mooncake Store
        +-----------+------------+
        |                        |
 revision metadata       manifest + payload group
```

| 数据 | 位置 | 权威内容 |
| --- | --- | --- |
| `WeightRevisionMetadata` | Store Master metadata、OpLog、Master snapshot | identity、policy、availability、observed residency、operation ID、generation、lease、manifest reference |
| `StoredWeightManifest` | Store `METADATA` object | Tensor 描述、fragment 几何、alias、object key 与 range |
| Store object metadata | 现有 Store metadata | memory/disk/DFS/NoF 物理副本及可读状态 |

revision metadata 只保存 `WeightManifestReference`，通过它找到并校验
manifest；manifest 再定位 payload。revision metadata 不保存 Tensor shape、
fragment、GPU address 或 framework 对象。

manifest 与所有 payload 共用一个 `payload_group_id`。revision metadata 位于
group 之外，因此 group 处于 COLD、DEGRADED、MIGRATING、DELETING 或 ABSENT 时
仍可查询。

架构中不引入独立的发现服务或对应公共 API。公开管理者只有 `WeightStore`；Store
Master 内部只维护 Weight metadata、索引、lease 和 operation 的持久化状态。

## 3. `WeightStore` 公共 API

命名遵循 Store 原语优先原则：存在等价 Store API 时，保留完整原名并只增加
`weight_` 前缀。

| Store API | `WeightStore` API |
| --- | --- |
| `put` | `weight_put` |
| `get` | `weight_get` |
| `is_exist` | `weight_is_exist` |
| `remove` | `weight_remove` |
| `get_size` | `weight_get_size` |
| `get_batch` | `weight_get_batch`（需要批量接口时） |
| `batch_is_exist` | `weight_batch_is_exist`（需要批量接口时） |
| `batch_remove` | `weight_batch_remove`（需要批量接口时） |

Store 没有等价原语的管理操作才使用领域动词，例如 `weight_update`、
`weight_migrate` 和 `weight_list`。查询 metadata 和 operation 统一使用 `get`
语义，不再使用 `query`。

该规则覆盖整个 Python `WeightStore` 组件，即
`mooncake.reshard.weight._store` 内从 facade 到 writer、upload/load service、
transaction 和 backend adapter 的调用链。只要某一步与 Store 的
`put/get/is_exist/remove/get_size` 等原语具有对应语义，就以
`weight_<Store method>` 为词根，阶段名追加在后面。

规则止于 `WeightStore` 与原始 Store/native management API 的边界，不要求
pybind、native client、RPC、`MasterService` 或 metadata state machine 机械改名。
边界以内外的典型调用链为：

```text
WeightStore.weight_put
  -> WeightStore.weight_put_plan
  -> WeightStoreWriter.weight_put_tensor
  -> WeightUploadService.weight_put_payload
  -> StoreBackend.weight_batch_put_from
  -> [raw Store boundary] batch_put_from
  -> WeightUploadTransaction.weight_put_commit

WeightStore.weight_get
  -> WeightLoadService.weight_get_manifest
  -> WeightLoadService.weight_get_plan
  -> WeightLoadService.weight_get_payload
  -> StoreBackend.weight_get_into_ranges
  -> [raw Store boundary] get_into_ranges

WeightStore.weight_get_metadata
  -> StoreBackend.weight_get_metadata
  -> [native management boundary] get_weight_revision

WeightStore.weight_remove
  -> StoreBackend.weight_remove
  -> [native management boundary] delete_weight_revision

WeightStore.weight_update
  -> StoreBackend.weight_update
  -> [native management boundary] update_weight_policy

WeightStore.weight_migrate
  -> StoreBackend.weight_migrate
  -> [native management boundary] start_weight_residency_operation

WeightStore.weight_get_operation
  -> StoreBackend.weight_get_operation
  -> [native management boundary] query_weight_operation
```

`weight_update`、`weight_migrate` 等没有普通 Store 等价原语，但它们仍属于
`WeightStore` 公共和内部调用链，因此保持相同的 `weight_` 前缀。边界外继续使用
现有准确的领域命名，不复制一套 Store 数据面实现。

```python
weight_store = WeightStore(store, default_policy=default_policy)

with weight_store.weight_put(
    snapshot,
    adapter,
    tenant_id="default",
    policy=policy,
) as writer:
    writer.weight_put_tensor(tensor_id, tensor)

exists = weight_store.weight_is_exist(identity)
view = weight_store.weight_get_metadata(identity)
page = weight_store.weight_list(namespace=namespace, resource_id=resource_id)

manifest = weight_store.weight_get(
    identity,
    target_placement,
    target_bindings,
)

updated = weight_store.weight_update(
    identity,
    policy=new_policy,
    expected_metadata_generation=view.metadata.metadata_generation,
)

operation = weight_store.weight_migrate(
    identity,
    target=WeightResidencyState.MIXED,
    mixed_hot_ratio=0.5,
    expected_metadata_generation=updated.metadata_generation,
)
operation = weight_store.weight_get_operation(operation.operation_id)

removed = weight_store.weight_remove(
    identity,
    expected_metadata_generation=updated.metadata_generation,
)
```

| API | 语义 |
| --- | --- |
| `weight_put` | 以 HOT 完成上传并发布 READY；需要时异步迁移到 preferred |
| `weight_get` | 获取 revision lease，解析 metadata/manifest，并加载或 Reshard 到目标布局 |
| `weight_is_exist` | 仅当精确 revision 为 `READY` 且可加载时返回 `True` |
| `weight_get_metadata` | 获取轻量 revision metadata 和 manifest reference |
| `weight_get_size` | 获取 revision 的 logical payload bytes |
| `weight_list` | 分页查询 revision metadata，不读取完整 manifest |
| `weight_update` | 只更新 policy，使用 metadata generation fencing |
| `weight_migrate` | 发起一次显式 residency 迁移 |
| `weight_get_operation` | 获取异步迁移进度和错误 |
| `weight_remove` | 显式整组删除 revision 并保留 tombstone |

当前 feature branch 新增的 verbose managed API 尚未进入 main，直接重命名且不保留
alias。main 已有的 `begin_weight_snapshot` 和 manifest-key API 继续作为明确的
unmanaged 兼容路径，不提供生命周期保证，也不在本设计中改名。

## 4. Policy 与状态模型

```python
@dataclass(frozen=True)
class WeightStoragePolicy:
    preferred_residency: WeightResidencyState = WeightResidencyState.MIXED
    mixed_hot_ratio: float = 0.5
    migration_mode: WeightMigrationMode = WeightMigrationMode.AUTO
```

```text
policy:
    preferred_residency: HOT | COLD | MIXED
    mixed_hot_ratio:     float，默认 0.5
    migration_mode:      PINNED | MANUAL | AUTO

metadata:
    availability: IMPORTING | READY | DEGRADED | DELETING | DELETED
    residency:    UNKNOWN | HOT | COLD | MIXED | ABSENT
    operation_id: Optional[int]

operation:
    kind:             MIGRATING | REPAIRING
    target_residency: HOT | COLD | MIXED
    progress:         processed_units / total_units / bytes / message
```

`MIGRATING` 是 operation，不是稳定 residency。一次查询可以同时返回：

```text
metadata.residency          = MIXED
metadata.operation_id       = 42
operation.kind              = MIGRATING
operation.target_residency  = COLD
```

对外可在关联 operation 的 kind 为 `MIGRATING` 时显示有效状态 `MIGRATING`，但
metadata 必须保留真实 residency，供进度判断和故障恢复使用。没有活动 operation
时 `operation_id=None`，不再额外保存一份 `NONE` operation state。

Policy 语义：

- `HOT + PINNED`：payload 始终保留可读内存副本；
- `HOT + AUTO`：优先内存，内存压力下允许安全下沉；
- `MIXED + AUTO`：优先配置的 HOT 比例，并随压力/访问调整；
- `COLD + AUTO`：优先冷存储，访问后允许提升；
- `* + MANUAL`：仅接受 `weight_migrate`；
- `* + PINNED`：更新 policy 前不改变 residency。

`PINNED` 只约束物理驻留，不阻止无 lease/operation 时显式
`weight_remove`。第一版只允许显式删除，不设置只有单一取值的 retention policy；
未来真正支持 TTL/GC 后再增加该字段。

Policy 解析优先级：

```text
weight_put(policy=...)
    > WeightStore(default_policy=...)
    > Store Master cluster default
```

解析后的 policy 必须写入 revision metadata。默认值为 `MIXED`、`0.5`、`AUTO`。

所有 `weight_put` 都先通过 Store memory replica 完成 HOT 写入。若 preferred 为
MIXED/COLD，初始迁移与 `READY + HOT` 在同一持久化 mutation 中建立关联，但不阻塞
`weight_put` 返回。该初始迁移用于兑现 preferred，不属于 pressure-driven AUTO；
`migration_mode` 只控制初始迁移完成后的后续 residency 调整。

## 5. HOT、COLD 与 MIXED

Residency 只统计 `WEIGHT` payload：

- `HOT`：所有必需 payload 都有可读 memory replica；
- `COLD`：所有必需 payload 都有可读 cold replica，且不依赖 memory；
- `MIXED`：部分完整 placement unit 为 HOT，其余为 COLD；
- `ABSENT`：payload group 已被物理删除；
- `UNKNOWN`：import/recovery 尚未得出聚合状态。

revision metadata 始终可查询。manifest 始终保留至少一个 hard-pinned、可读的
memory replica；因此 `COLD` 只表示 payload COLD。

`MIXED` 的比例定义为：

```text
hot logical payload bytes / total logical payload bytes
```

`0 < mixed_hot_ratio < 1`，默认 `0.5`；端点分别使用 `HOT` 和 `COLD`。

一个逻辑 Tensor（包括全部 fragments 和 alias group）是不可拆分的 residency
unit。`WeightStore` 根据 manifest 为 payload 生成不透明的
`residency_affinity_id`。Store 只按相同 affinity 整组迁移，不理解 Tensor
shape 或 framework 语义；revision metadata 只保存 affinity count/digest，
不复制 Tensor metadata。

MIXED planner 以确定性算法选择完整 affinity unit，使 HOT bytes 最接近目标值；
当超大 Tensor 导致无法精确达到比例时，返回实际 `observed_hot_ratio`。

## 6. 数据流与生命周期

### Put

```text
weight_put
  -> BeginWeightImport(policy)
  -> batch_put_from(payload group)
  -> 最后提交 StoredWeightManifest
  -> CommitWeightImport 原子发布：
       availability=READY
       residency=HOT
       preferred=HOT   时 operation_id=None
       preferred!=HOT 时 operation=MIGRATING(target=preferred)
  -> weight_put 返回，后台继续向 preferred 收敛
```

`READY` 只证明 manifest 和完整 payload 已可读，不要求当前 residency 已等于
preferred。初始迁移失败时 revision 仍保持 `READY`，operation 保留错误和进度并由
reconciliation 重试；只有必需 payload 失去全部可读副本时才进入 `DEGRADED`。

写入 L3 时保留源端 fragment，不预先转换成目标 TP/PP/EP。真正 Reshard 发生在
`weight_get`：

```text
identity -> metadata -> manifest/source ranges
         -> target placement/bindings -> get_into_ranges
```

### Update

`weight_update` 只接受 `WeightStoragePolicy`，禁止调用方修改 identity、manifest
reference、availability、observed residency 或 operation。更新必须
durable-before-visible；相同请求幂等，过期 generation 返回
`STALE_GENERATION`。`AUTO` 自动收敛，`MANUAL` 等待显式 migrate。

### Migrate

`weight_migrate` 创建一个持久化 operation 并立即返回：

- 到 `COLD`：先建立并验证 cold replica，再释放 memory replica；
- 到 `HOT`：为所有 payload unit 建立并验证 memory replica；
- 到 `MIXED`：提升选中的 unit，安全下沉未选中的 unit。

迁移可物理部分完成，但在全组验证通过前关联 operation 始终保持
`kind=MIGRATING`。`weight_get` 在迁移期间仍可获取 revision lease 并读取数据；
迁移可以继续创建新副本，但 active lease 会推迟释放现有副本。这样迁移不降低
`READY` revision 的可用性。

### Delete 与 eviction

普通 Store eviction、quota eviction、单 key remove 和 cleanup 必须跳过 managed
weight group。已有 cold durability 后释放 memory replica 属于 migration，不是
逻辑 Weight eviction。

第一版只允许显式整组删除：

```text
READY/DEGRADED -> DELETING
  -> 分批删除 payload
  -> 最后删除 manifest
  -> 验证 group 已空
  -> DELETED + ABSENT tombstone
```

进入 `DELETING` 后拒绝新 lease；已有 lease 或 migration 时返回 `BUSY`。

## 7. 自动迁移

自动迁移复用 active Master 的 bounded Weight reconciliation loop。候选必须满足：

```text
availability == READY
migration_mode == AUTO
operation_id is None
active_lease_count == 0
```

内存超过 high watermark 时，按最久未访问、可释放 HOT bytes、稳定 revision
identity 排序，执行 `HOT -> MIXED -> COLD`。内存低于 low watermark 或 COLD
revision 被访问时，可向 `preferred_residency` 提升。

每个 revision 使用 cooldown；每轮限制 member 数和 bytes，防止抖动及迁移队列
饥饿。自动决策必须先持久化 operation target、operation ID、generation 和
progress，再执行物理工作；切主后从真实副本状态继续。

## 8. 并发、HA 与错误

`weight_update`、`weight_migrate`、`weight_remove` 必须携带
`expected_metadata_generation`。同一 revision 同时只允许一个 operation；metadata
只保存可选 operation ID，operation kind、target 和 progress 存在独立 operation
record 中。

主要错误包括：`NOT_FOUND`、`NOT_READY`、`STALE_GENERATION`、`BUSY`、
`CONFLICT`、`POLICY_UNSATISFIABLE` 和 `DURABILITY_FAILED`。

Policy、operation target、affinity summary、operation progress 与 deletion
tombstone 都进入现有 Store Master metadata OpLog 和 snapshot。由于这些 managed
metadata 尚未进入 main，无需兼容旧内部字段名；standby 未具备新 schema 能力时，
capability gate 必须禁止新 mutation。

## 9. 实施切片

### 9.1 内部命名收敛

Master 端仍需要一个内部组件持有 revision metadata、group reverse index、lease、
operation 和 generation state machine，但它不是独立发现服务，也不对外暴露。
合入前直接完成以下重命名：

```text
WeightCatalog              -> WeightMetadataStore
WeightCatalogMutation      -> WeightMetadataMutation
WeightCatalogMutationKind  -> WeightMetadataMutationKind
WeightCatalogSnapshot      -> WeightMetadataSnapshot
WeightCatalogError         -> WeightManagementError
weight_catalog_            -> weight_metadata_
weight_catalog.h/.cpp      -> weight_metadata_store.h/.cpp
weight_catalog_test.cpp    -> weight_metadata_store_test.cpp
snapshot field weight_catalog -> weight_metadata
```

最终公开管理者仍然只有 Python `WeightStore`；`WeightMetadataStore` 是
`MasterService` 的私有实现细节。

### 9.2 功能切片

1. Contracts/API：完成上述重命名，增加 policy、update contract、独立
   `WeightOperation`，统一 `WeightStore.weight_*` 门面。
2. Management CRUD：补齐 `weight_is_exist`、`weight_get_metadata`、
   `weight_get_size`、`weight_update`、`weight_get_operation` 与
   `weight_remove`。
3. WeightStore internal naming：把 writer、upload/load service、transaction 和
   backend adapter 中具有 Store 对应语义的方法统一到 `weight_*` 词根。
4. MIXED：实现 payload-only residency、manifest HOT、affinity unit 和比例规划。
5. AUTO：实现 pressure/access 触发、cooldown、限批和 failover resume。
6. HA/docs：扩展 OpLog/snapshot、Python binding、API 文档和兼容迁移说明。

前四项不依赖 Transfer Engine PR；Store upload 继续使用已存在的 registered-buffer
`batch_put_from`。未来 DTensor-native adapter 只替换 placement/binding 的生成方式，
不改变本设计的 revision metadata、manifest、policy 和 lifecycle API。

## 10. 验收标准

- 新增 `weight_*` managed API 不得弱化既有 unmanaged 兼容入口的公开契约；
  `upload`、`load` 等既有 API 必须保留显式 keyword-only 参数、类型标注和可 introspect
  的签名，完整 reshard contract suite 必须通过；

以下标准必须在同一个 exact implementation head 上逐项验证。所有“最终收敛”类
断言使用有 deadline 的状态轮询，不以固定 `sleep` 代替；无法在当前环境执行的
项目必须明确记录为未验证边界，不能计为通过。

### 10.1 API 与命名

- `weight_put -> weight_get_metadata -> weight_get -> weight_remove` E2E 通过；
- `weight_is_exist` 对 `IMPORTING/READY/DEGRADED/DELETING/DELETED` 分别返回
  `False/True/False/False/False`，其中 `READY + MIGRATING` 仍返回 `True`；
- `weight_get_size` 返回 manifest reference 中经过提交校验的 logical bytes；
- `weight_list` 的过滤、稳定分页和 page token 校验通过；
- 异步接口返回值语义明确：`weight_migrate` 返回已持久化 operation；
  `weight_remove` 对大组允许先返回 `DELETING`。调用方必须使用有 deadline 的
  metadata/operation 轮询确认物理完成，不能把首次 RPC 成功等同于终态；
- `WeightStore` 组件内从 facade 到 backend adapter 的方法，在 Store 存在同语义
  API 时使用对应 `weight_<Store method>` 词根；
- pybind、native RPC、`MasterService` 等边界外代码不因该规则机械改名；
- 除内部类型迁移说明外，不再出现旧 `WeightCatalog*`/`weight_catalog_*` 命名。

### 10.2 Put 与 policy

- policy 的 cluster default、`WeightStore` default、单次 `weight_put` override
  优先级正确，最终值持久化到 revision metadata；
- preferred 为 MIXED 但 revision 少于两个完整 affinity unit 时，begin 在写入任何
  payload 前返回 `POLICY_UNSATISFIABLE`，不能创建永远无法完成的 migration；
- cluster default 只影响之后新建的 revision；修改默认值后，已有 revision 的已
  持久化 policy、generation 和目标 residency 不发生变化；
- cluster default 的 YAML/CLI 配置等价；非法枚举、非法 MIXED ratio、high/low
  watermark 逆序、零迁移批限额等配置在启动时 fail closed；
- 所有 `weight_put` 先完整写入 HOT payload，并在 manifest 最后提交后发布
  `READY + HOT`；任何 payload/manifest 不完整的路径都不能发布 `READY`；
- import 在 payload、manifest 或 commit 任一阶段失败、超时或调用方退出时，
  revision 不得对 `weight_is_exist/weight_get` 可见；相同 identity 的恢复、重试或
  abort 必须 generation-fenced，不能留下可被误判为 READY 的半组数据或重复
  manifest；
- preferred 为 HOT 时 `operation_id=None`；preferred 为 MIXED/COLD 时，READY
  metadata 与对应 `MIGRATING(target=preferred)` operation 在同一个 durable
  mutation 中发布，且不阻塞 `weight_put` 返回；
- `READY` 发布和 preferred 收敛是两个独立检查点：前者只要求 manifest 与全部
  payload 已可读并且初始 residency 为 HOT；后者由异步 reconciliation 完成，
  不得把 `READY` 推迟到 MIXED/COLD 迁移结束；
- preferred 为 MIXED/COLD 时，commit 返回的 view 或暂停 reconciliation 的契约
  测试必须先验证 `READY + HOT + active MIGRATING`；随后在有 deadline 的轮询内
  收敛到目标 residency。公共 E2E 不依赖短暂中间态的竞态窗口，但测试集不得只
  验证最终状态而漏掉 READY 发布点；
- response 丢失后的相同 put/commit 重试幂等，不生成第二个 group、manifest 或
  operation；
- 首次 HOT 向 preferred 的收敛对 `PINNED/MANUAL/AUTO` 都执行；首次收敛完成后，
  `PINNED` 固定 residency，`MANUAL` 仅接受显式迁移，`AUTO` 才响应压力和访问；
- 初始收敛完成后，`PINNED` 下显式迁移被拒绝，`MANUAL/AUTO` 下合法显式迁移被
  接受；非法 MIXED ratio 和无法满足 durability 的目标返回确定错误且不改变状态；
- `weight_update` 只能修改 policy；同值更新幂等，过期 generation 返回
  `STALE_GENERATION`，不允许修改 identity、manifest 或观测状态；
- `weight_update` 不同步篡改 observed residency：更新后若 policy 为 `AUTO` 且当前
  residency 与 preferred 不一致，则先持久化新 operation 再异步收敛；`MANUAL` 和
  `PINNED` 不自动创建迁移 operation；
- 并发 update/migrate/remove 只能有一个 generation-fenced mutation 成功，其余
  返回 `STALE_GENERATION` 或 `BUSY`。

### 10.3 Migration

- HOT/COLD/MIXED 之间所有显式合法迁移通过，`weight_get_operation` 返回稳定的
  kind、target、processed/total、bytes 和 terminal message；
- 显式迁移响应丢失后的相同请求返回原 operation，而不是创建第二个 operation；若
  residency 已满足相同 target，则返回确定的幂等结果且不重复执行物理迁移；
- metadata 只保存可选 operation ID，不重复保存 kind、target 或 progress；
- operation ID 在重试、Master restart 和 failover 后保持不变；processed units/bytes
  及错误信息只能单调推进，不能回退、重复累计或被旧 operation 的迟到回调覆盖；
- 迁移成功后 metadata 的 residency/`observed_hot_ratio` 与实际 payload replica
  一致，active `operation_id` 被清空；原 operation ID 仍可查询终态，重复
  reconciliation 不重复迁移或重复累计进度；
- 到 COLD/MIXED 时，cold replica 未验证可读前绝不释放对应的最后一个 memory
  replica；到 HOT 时必须验证所有必需 payload 都有可读 memory replica；
- MIXED 默认 50%、支持 override、按完整 Tensor/alias affinity 迁移，并返回实际
  `observed_hot_ratio`；
- payload 为 COLD 时 manifest 仍保持 memory-readable；
- `weight_get` 在 MIGRATING 期间保持可用；active lease 允许创建新副本，但暂停
  现有副本释放和 revision 删除；
- `weight_get` 必须先原子获取与当前 revision generation 对应的 lease，再解析
  manifest/payload；成功、异常和取消路径都会释放 lease，调用方崩溃后 lease 能按
  TTL 过期，Master restart/failover 后不得永久泄漏或提前失效；
- migration 失败但数据仍完整可读时保持 `READY` 并保留可重试 operation；只有
  必需 payload 失去全部可读副本时进入 `DEGRADED`；
- AUTO 在 high/low watermark 和访问事件下按预期迁移，并遵守 cooldown 与每轮
  member/bytes 上限；压力下沉和空闲提升只选择无 active lease 的 revision；
- COLD/MIXED revision 的访问提升在授予本次 lease 前，以同一个 generation fence
  先持久化 promotion operation，再基于新 generation 授予 lease；该 lease 只延迟
  旧副本释放，不阻止创建、校验 HOT 副本，也不生成重复 operation；
- 压力候选按可持久化的最近访问时间、可释放 HOT bytes 和稳定 identity 确定性
  排序；Master 重启后相同输入得到相同顺序，迁移批次至少保证一个完整 affinity
  unit 前进，且绝不为满足 bytes 上限拆分 Tensor/alias affinity。

### 10.4 Remove、恢复与非回归

- `weight_remove` 执行 `DELETING -> payload 分批删除 -> manifest 最后删除 ->
  DELETED/ABSENT`，并在 active lease 或 migration 存在时返回 `BUSY`；
- 任一删除批次失败时保持 `DELETING` 和已持久化进度；重试或切主后从剩余 member
  继续，manifest 仍最后删除，不重新删除已完成 member；
- remove 响应丢失后的相同请求幂等完成，tombstone 保持可查询且
  `weight_is_exist=False`；
- payload 超过单批删除上限时，首次 `weight_remove` 返回 `DELETING`，后台或相同
  请求重试从剩余 member 继续，最终在 deadline 内到达 `DELETED + ABSENT`；
- 普通 eviction、quota eviction、单 key remove 和 cleanup 不能部分删除 managed
  weight group；
- metadata 能定位并校验 manifest identity/digest，manifest 能定位全部 payload
  range，metadata 不复制 Tensor/runtime binding；
- partial migration、policy update 和 remove 在 Master restart/failover 后从已持久化
  generation、operation 和实际 replica 状态继续；
- 状态不变量始终成立：`IMPORTING` 对应 `UNKNOWN`；`READY` 只对应
  `HOT/COLD/MIXED`；`DELETED` 对应 `ABSENT` 且无 active lease/operation；metadata
  中的 active operation ID 必须指向同 revision 的唯一非终态 operation，终态
  operation 可按原 ID 查询但不再挂在 metadata 上；
- active/standby schema capability 未确认时，新 Weight mutation fail closed；
- 旧 snapshot/OpLog 不含 weight metadata 时按兼容约定恢复为空；新 weight schema
  未被 standby 明确支持时 fail closed，不得静默丢失 revision、lease 或 operation；
- revision 到达终态后不残留 active lease、active operation、offload/promotion task
  或额外 group member；保留的 terminal operation 与 tombstone 仅承担查询和幂等；
- 普通 Store、unmanaged WeightStore 和 KVCache 的 put/get/eviction/remove 行为无
  回归。

### 10.5 交付门禁

在同一个 exact implementation head 上全部通过：

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
PYTHONPATH=mooncake-reshard/python:mooncake-reshard/tests \
  python3 -m pytest -q mooncake-reshard/tests/model_weight_store
npx --yes pyright --project mooncake-reshard/pyrightconfig.json
pre-commit run --from-ref origin/main --to-ref HEAD
cd docs && make html
```

另外必须在 Linux 环境完成至少一条真实公共路径 E2E，而不是只使用 fake client：

- 启动该 exact head 构建出的 Store Master，并使用同一 head 的 native Python
  binding 执行 `weight_put -> READY/HOT -> 异步收敛 preferred -> weight_get ->
  weight_remove`；
- COLD/MIXED E2E 必须实际创建并验证 cold replica，覆盖迁移期间持 lease 读取、
  lease 释放后再下沉 memory replica，以及 manifest 始终可从内存读取；
- 在 operation 已 durable、物理迁移未完成的位置重启 Master，验证恢复后从真实
  replica 状态继续，而不是重新生成 group 或 operation；
- 至少注入一次 cold 写入/校验失败，验证 revision 保持 `READY`、已有数据可读、
  operation 错误可查询且可重试；
- E2E 使用 deadline 轮询等待状态变化，并记录各关键状态及 generation；固定延时
  后只检查最终结果不算通过。

最终记录 exact base/head SHA、构建参数、测试数量、E2E 拓扑、失败注入结果和未
验证硬件边界。若验证主机没有 RDMA HCA，只能确认 TCP/本地磁盘路径，不能声称
RDMA data-plane E2E 已通过。最后扫描当前 diff，确认所有新增或修改的代码注释均
为英文。
