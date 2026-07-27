# Heterogeneous Weight Conversion (Rust)

这是一个独立的控制面 Rust crate，用完整的 source/target runtime manifest
生成 sCR 可直接执行的显存拷贝计划。它不依赖 Mooncake Transfer Engine，
也不直接调用 memlib。

## 稳定接口

输入保持为 Python 版本的 `ConversionRequest`：

```text
ConversionRequest {
  format_version,
  plan_id,
  source_manifest,
  target_manifest
}
```

输出保持为 `ScrTransferPlan`：

```text
ScrTransferPlan {
  format_version,
  plan_id,
  transfers[] {
    key,
    tensor_id,
    source { dev, addr },
    target { dev, addr },
    nbytes
  }
}
```

每条 transfer 都是连续的 byte-copy。TP、PP、EP、DP 的 overlap、拆分、
聚合和 strided tensor 展开均已在 planner/lowering 内完成，sCR 不需要再次
理解 tensor 几何。

## Rust API

```rust
use heterogeneous_weight_conversion::{
    ConversionRequest, ManifestWeightConversionPlugin,
};

let request = ConversionRequest::from_json(request_json)?;
let plan = ManifestWeightConversionPlugin::default()
    .plan_scr(&request, Some(64 * 1024 * 1024))?;
let output_json = plan.to_json()?;
```

也可以通过 `create_plan(...)` 直接传入两侧 `RuntimeModelManifest`。

## CLI

CLI 从 stdin 读取一份 request，并把计划写到 stdout：

```bash
cargo run --bin hweight-plan -- --max-chunk-bytes 67108864 < request.json
```

## 边界

- runtime/framework 负责提供完整、已绑定地址的 source 和 target manifest。
- 本 crate 负责确定性 reshard、copy geometry、chunk 和物理安全校验。
- sCR 按输出顺序把每条 transfer 映射为 memlib `reg/load`。
- TE/memlib 负责数据面执行、注册、传输和生命周期管理。
- 执行落盘或跨进程收到的计划前，应使用同一 request 和 chunk 参数重新规划，
  并调用 exact plan validation。

## 验证

```bash
cargo fmt --check
cargo test --all-targets
cargo clippy --all-targets --all-features -- -D warnings
```

`tests/python_ab.py` 可用当前 Python 插件作为 oracle，逐字节比较随机异构
布局下的输出计划。

Qwen3、Qwen3-MoE、Qwen3.5 和 Qwen3-VL 的真实 tensor 名称、shape、packed
fragment 与 TP/EP/PP/DP 转换用例见
[`docs/qwen3-scr-test-cases.md`](docs/qwen3-scr-test-cases.md)。可以通过
`cargo run --example qwen3_scr_cases -- --list` 生成 sCR 测试输入或预期 plan。
