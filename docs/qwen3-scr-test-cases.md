# Qwen3 系列异构权重转换测试

最后核对日期：2026-07-27

## 测试目标

这些用例用于验证独立 Rust 插件能把 Qwen3 系列的 source/target runtime
manifest 转换为 sCR 可直接执行的连续 byte-copy plan。

这里的“真实”指：

- 使用公开模型配置中的真实 hidden size、intermediate size、expert 数和层数。
- 使用 SGLang runtime 中的真实 fused tensor 名称。
- 按 QKV、gate/up、w1/w3 的实际 packed fragment 表达 TP 切分。
- 同时覆盖 TP、EP、PP、DP 变化，并检查传输总字节数、设备集合和 exact rebind。
- 使用一个缩小尺寸的 MoE w13 用例真正回放每条 transfer 并逐字节校验结果。

测试不会下载模型文件，也不会分配 235B 模型的 GPU 显存。大模型用例验证的是
真实 shape 下的 manifest、reshard 和 lowering；逐字节执行由 miniature 用例完成。

## 用例矩阵

| 用例 | Runtime tensor | Source | Target | 重点 |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | `model.layers.20.self_attn.qkv_proj.weight` `[4096,1024]` | TP2/PP1/DP2 | TP4/PP2/DP1 | GQA 的 Q/K/V 三段 packed TP fragment |
| Qwen3-8B | `model.layers.18.mlp.gate_up_proj.weight` `[24576,4096]` | TP4/PP2 | TP2/PP4 | gate/up 两段分别聚合，PP owner 改变 |
| Qwen3-30B-A3B | `model.layers.30.mlp.experts.w13_weight` `[128,1536,2048]` | TP2/EP4/PP4/DP2 | TP4/EP2/PP2/DP1 | TP、EP、PP、DP 同时变化 |
| Qwen3-235B-A22B | `model.layers.60.mlp.experts.w2_weight` `[128,1536,4096]` | TP4/EP8/PP8 | TP8/EP4/PP4 | SGLang Triton E-N-K 物理语义 |
| Qwen3.5-0.8B | `model.layers.8.linear_attn.in_proj_qkvz.weight` `[8192,1024]` | TP2/PP1/DP1 | TP4/PP4/DP2 | GDN Q/K/V/Z 四段 packed TP 和 DP fan-out |
| Qwen3-VL-8B | `visual.blocks.0.attn.qkv_proj.weight` `[3456,1152]` | TP2 | TP4 | vision stack 的 Q/K/V packed tensor |
| Qwen3-235B-A22B-FP8 | `model.layers.60.mlp.experts.w13_weight_scale_inv` `[128,24,32]` | TP4/EP1 | TP4/EP4 | 128x128 block scale 的 EP 重分片 |
| Qwen3-30B-A3B miniature | `model.layers.30.mlp.experts.w13_weight` `[8,16,8]` | TP2/EP4/PP4 | TP4/EP2/PP2 | sCR 可分配 buffer 并执行的逐字节 fixture |

`tests/qwen3_real_reshard.rs` 还包含两个边界用例：

1. Triton `w13` scale 与 FlashInfer/CUTLASS `w31` scale 的
   `representation_id` 不同，copy-only planner 必须拒绝，不能静默搬错字节。
2. miniature MoE w13 使用 `[8,16,8]`，执行 TP2/EP4/PP4 到
   TP4/EP2/PP2 的全部 transfer，并校验每个目标字节。

## 为什么使用 fragment

例如 Qwen3-0.6B 的 runtime `qkv_proj.weight` 不是把
`[Q全部,K全部,V全部]` 沿 dim0 粗暴均分。TP rank 上实际存的是：

```text
[Q的本rank分片, K的本rank分片, V的本rank分片]
```

因此一张卡上的一个 runtime tensor 会在 manifest 中展开为三个具有相同
`tensor_id`、`rank` 和 `dev` 的逻辑 fragment。每个 fragment 记录自己的
`logical_offset`、`logical_shape`、`addr` 和 stride。gate/up、w1/w3 和
Q/K/V/Z 同理。planner 仍然只按统一的全局 tensor 空间求 overlap，sCR 最终只看
降低后的连续 transfer。

## 运行

执行全部 Qwen3 转换测试：

```bash
cargo test --test qwen3_real_reshard
```

列出可直接提供给 sCR 的请求：

```bash
cargo run --example qwen3_scr_cases -- --list
```

生成指定用例的 input manifest：

```bash
cargo run --example qwen3_scr_cases -- \
  qwen3_30b_a3b_moe_w13 request > request.json
```

生成插件预期输出的 sCR plan：

```bash
cargo run --example qwen3_scr_cases -- \
  qwen3_30b_a3b_moe_w13 plan > expected-plan.json
```

需要真正分配小 buffer 并执行数据面时，使用：

```bash
cargo run --example qwen3_scr_cases -- \
  qwen3_30b_a3b_moe_w13_mini request > executable-request.json
```

sCR 测试可以把 `request.json` 交给 `create_plan`，再逐条执行输出中的
`source {dev,addr}`、`target {dev,addr}` 和 `nbytes`。测试地址与 dev 是
确定性的逻辑值，不应在真实 GPU 上直接解引用；接真实 runtime 时由 sCR 提供
同结构、绑定真实 dev/addr 的 manifest。

## 适用边界

- planner 只支持 source/target `representation_id` 一致的 copy-only 转换。
- backend 引起的 w13/w31 交换、transpose、clamp 或量化格式变化，必须先由
  framework adapter 产出一致的 canonical representation，或由单独 transform
  kernel 完成。
- PP 不复制整个 source stage。每个 tensor 只从它的 source PP owner 传到
  target PP owner。
- DP source 只选择一份完整 replica；target DP 增大时才按目标 manifest fan-out。

## 依据

- [Qwen3-0.6B config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c916fa4defd319b7d4e4da17604ca7338f4d99f5/config.json)
- [Qwen3-8B config](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json)
- [Qwen3-30B-A3B config](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json)
- [Qwen3-235B-A22B config](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/refs%2Fpr%2F2/config.json)
- [Qwen3.5-0.8B config](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json)
- [Qwen3-VL-8B config](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json)
- [SGLang qwen3.py](https://github.com/sgl-project/sglang/blob/08af5aea570a96f5edd9c1d9e0c28c690fdd842a/python/sglang/srt/models/qwen3.py)
- [SGLang qwen3_moe.py](https://github.com/sgl-project/sglang/blob/08af5aea570a96f5edd9c1d9e0c28c690fdd842a/python/sglang/srt/models/qwen3_moe.py)
- [SGLang qwen3_5.py](https://github.com/sgl-project/sglang/blob/08af5aea570a96f5edd9c1d9e0c28c690fdd842a/python/sglang/srt/models/qwen3_5.py)
- [SGLang qwen3_vl.py](https://github.com/sgl-project/sglang/blob/08af5aea570a96f5edd9c1d9e0c28c690fdd842a/python/sglang/srt/models/qwen3_vl.py)
