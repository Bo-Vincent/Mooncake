use heterogeneous_weight_conversion::{
    ConversionRequest, ParallelConfig, ParallelRank, RuntimeModelManifest, RuntimeTensorManifest,
    TensorSemantic,
};

const SOURCE_BASE_DEV: u64 = 0;
const TARGET_BASE_DEV: u64 = 1024;
const SOURCE_BASE_ADDR: u64 = 0x0000_1000_0000_0000;
const TARGET_BASE_ADDR: u64 = 0x4000_1000_0000_0000;
const DEVICE_SLOT_BYTES: u64 = 64 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub struct ParallelLayout {
    pub tp: u64,
    pub pp: u64,
    pub ep: u64,
    pub dp: u64,
}

#[derive(Debug, Clone, Copy)]
pub enum TpSharding {
    Contiguous {
        axis: usize,
    },
    Packed {
        axis: usize,
        component_extents: &'static [u64],
    },
}

#[derive(Debug, Clone, Copy)]
pub struct TensorSpec {
    pub model_id: &'static str,
    pub tensor_id: &'static str,
    pub model_part: &'static str,
    pub stack: &'static str,
    pub layer_id: u64,
    pub num_layers: u64,
    pub module_path: &'static [&'static str],
    pub parameter_role: &'static str,
    pub representation_id: &'static str,
    pub global_shape: &'static [u64],
    pub tp_sharding: TpSharding,
    pub ep_axis: Option<usize>,
    pub dtype: &'static str,
    pub itemsize: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct Qwen3Case {
    pub name: &'static str,
    pub description: &'static str,
    pub tensor: TensorSpec,
    pub source: ParallelLayout,
    pub target: ParallelLayout,
}

impl Qwen3Case {
    pub fn request(&self) -> ConversionRequest {
        build_request(self.name, self.tensor, self.source, self.target)
    }
}

pub fn all_cases() -> Vec<Qwen3Case> {
    vec![
        Qwen3Case {
            name: "qwen3_0_6b_dense_qkv",
            description: "Qwen3-0.6B qkv_proj: TP2/PP1/DP2 -> TP4/PP2/DP1",
            tensor: TensorSpec {
                model_id: "Qwen/Qwen3-0.6B",
                tensor_id: "model.layers.20.self_attn.qkv_proj.weight",
                model_part: "language",
                stack: "decoder",
                layer_id: 20,
                num_layers: 28,
                module_path: &["self_attn", "qkv_proj"],
                parameter_role: "qkv_weight",
                representation_id: "bf16:sglang-packed-qkv:canonical:v1",
                global_shape: &[4096, 1024],
                tp_sharding: TpSharding::Packed {
                    axis: 0,
                    component_extents: &[2048, 1024, 1024],
                },
                ep_axis: None,
                dtype: "bfloat16",
                itemsize: 2,
            },
            source: ParallelLayout {
                tp: 2,
                pp: 1,
                ep: 1,
                dp: 2,
            },
            target: ParallelLayout {
                tp: 4,
                pp: 2,
                ep: 1,
                dp: 1,
            },
        },
        Qwen3Case {
            name: "qwen3_8b_dense_gate_up",
            description: "Qwen3-8B gate_up_proj: TP4/PP2 -> TP2/PP4",
            tensor: TensorSpec {
                model_id: "Qwen/Qwen3-8B",
                tensor_id: "model.layers.18.mlp.gate_up_proj.weight",
                model_part: "language",
                stack: "decoder",
                layer_id: 18,
                num_layers: 36,
                module_path: &["mlp", "gate_up_proj"],
                parameter_role: "gate_up_weight",
                representation_id: "bf16:sglang-packed-gate-up:canonical:v1",
                global_shape: &[24576, 4096],
                tp_sharding: TpSharding::Packed {
                    axis: 0,
                    component_extents: &[12288, 12288],
                },
                ep_axis: None,
                dtype: "bfloat16",
                itemsize: 2,
            },
            source: ParallelLayout {
                tp: 4,
                pp: 2,
                ep: 1,
                dp: 1,
            },
            target: ParallelLayout {
                tp: 2,
                pp: 4,
                ep: 1,
                dp: 1,
            },
        },
        Qwen3Case {
            name: "qwen3_30b_a3b_moe_w13",
            description: "Qwen3-30B-A3B w13: TP2/EP4/PP4/DP2 -> TP4/EP2/PP2/DP1",
            tensor: TensorSpec {
                model_id: "Qwen/Qwen3-30B-A3B",
                tensor_id: "model.layers.30.mlp.experts.w13_weight",
                model_part: "language",
                stack: "decoder",
                layer_id: 30,
                num_layers: 48,
                module_path: &["mlp", "experts"],
                parameter_role: "w13_weight",
                representation_id: "bf16:sglang-triton-w13:canonical:v1",
                global_shape: &[128, 1536, 2048],
                tp_sharding: TpSharding::Packed {
                    axis: 1,
                    component_extents: &[768, 768],
                },
                ep_axis: Some(0),
                dtype: "bfloat16",
                itemsize: 2,
            },
            source: ParallelLayout {
                tp: 2,
                pp: 4,
                ep: 4,
                dp: 2,
            },
            target: ParallelLayout {
                tp: 4,
                pp: 2,
                ep: 2,
                dp: 1,
            },
        },
        Qwen3Case {
            name: "qwen3_235b_a22b_moe_w2",
            description: "Qwen3-235B-A22B Triton w2: TP4/EP8/PP8 -> TP8/EP4/PP4",
            tensor: TensorSpec {
                model_id: "Qwen/Qwen3-235B-A22B",
                tensor_id: "model.layers.60.mlp.experts.w2_weight",
                model_part: "language",
                stack: "decoder",
                layer_id: 60,
                num_layers: 94,
                module_path: &["mlp", "experts"],
                parameter_role: "w2_weight",
                representation_id: "bf16:sglang-triton-w2-e-n-k:canonical:v1",
                global_shape: &[128, 1536, 4096],
                tp_sharding: TpSharding::Contiguous { axis: 1 },
                ep_axis: Some(0),
                dtype: "bfloat16",
                itemsize: 2,
            },
            source: ParallelLayout {
                tp: 4,
                pp: 8,
                ep: 8,
                dp: 1,
            },
            target: ParallelLayout {
                tp: 8,
                pp: 4,
                ep: 4,
                dp: 1,
            },
        },
        Qwen3Case {
            name: "qwen3_5_0_8b_gdn_qkvz",
            description: "Qwen3.5-0.8B GDN qkvz: TP2/PP1/DP1 -> TP4/PP4/DP2",
            tensor: TensorSpec {
                model_id: "Qwen/Qwen3.5-0.8B",
                tensor_id: "model.layers.8.linear_attn.in_proj_qkvz.weight",
                model_part: "language",
                stack: "decoder",
                layer_id: 8,
                num_layers: 24,
                module_path: &["linear_attn", "in_proj_qkvz"],
                parameter_role: "gdn_qkvz_weight",
                representation_id: "bf16:sglang-packed-qkvz:canonical:v1",
                global_shape: &[8192, 1024],
                tp_sharding: TpSharding::Packed {
                    axis: 0,
                    component_extents: &[2048, 2048, 2048, 2048],
                },
                ep_axis: None,
                dtype: "bfloat16",
                itemsize: 2,
            },
            source: ParallelLayout {
                tp: 2,
                pp: 1,
                ep: 1,
                dp: 1,
            },
            target: ParallelLayout {
                tp: 4,
                pp: 4,
                ep: 1,
                dp: 2,
            },
        },
        Qwen3Case {
            name: "qwen3_vl_8b_vision_qkv",
            description: "Qwen3-VL-8B vision qkv_proj: TP2 -> TP4",
            tensor: TensorSpec {
                model_id: "Qwen/Qwen3-VL-8B-Instruct",
                tensor_id: "visual.blocks.0.attn.qkv_proj.weight",
                model_part: "vision",
                stack: "encoder",
                layer_id: 0,
                num_layers: 27,
                module_path: &["blocks", "attn", "qkv_proj"],
                parameter_role: "vision_qkv_weight",
                representation_id: "bf16:sglang-vision-packed-qkv:canonical:v1",
                global_shape: &[3456, 1152],
                tp_sharding: TpSharding::Packed {
                    axis: 0,
                    component_extents: &[1152, 1152, 1152],
                },
                ep_axis: None,
                dtype: "bfloat16",
                itemsize: 2,
            },
            source: ParallelLayout {
                tp: 2,
                pp: 1,
                ep: 1,
                dp: 1,
            },
            target: ParallelLayout {
                tp: 4,
                pp: 1,
                ep: 1,
                dp: 1,
            },
        },
        Qwen3Case {
            name: "qwen3_235b_a22b_fp8_w13_scale",
            description: "Qwen3-235B-A22B FP8 w13 block scale: TP4/EP1 -> TP4/EP4",
            tensor: TensorSpec {
                model_id: "Qwen/Qwen3-235B-A22B-FP8",
                tensor_id: "model.layers.60.mlp.experts.w13_weight_scale_inv",
                model_part: "language",
                stack: "decoder",
                layer_id: 60,
                num_layers: 94,
                module_path: &["mlp", "experts"],
                parameter_role: "w13_weight_scale_inv",
                representation_id: "fp8-block128:sglang-triton-w13-scale-inv:canonical:v1",
                global_shape: &[128, 24, 32],
                tp_sharding: TpSharding::Packed {
                    axis: 1,
                    component_extents: &[12, 12],
                },
                ep_axis: Some(0),
                dtype: "float32",
                itemsize: 4,
            },
            source: ParallelLayout {
                tp: 4,
                pp: 1,
                ep: 1,
                dp: 1,
            },
            target: ParallelLayout {
                tp: 4,
                pp: 1,
                ep: 4,
                dp: 1,
            },
        },
        Qwen3Case {
            name: "qwen3_30b_a3b_moe_w13_mini",
            description: "Executable miniature w13: TP2/EP4/PP4 -> TP4/EP2/PP2",
            tensor: TensorSpec {
                model_id: "Qwen/Qwen3-30B-A3B",
                tensor_id: "model.layers.30.mlp.experts.w13_weight",
                model_part: "language",
                stack: "decoder",
                layer_id: 30,
                num_layers: 48,
                module_path: &["mlp", "experts"],
                parameter_role: "w13_weight",
                representation_id: "uint8:sglang-triton-w13:miniature:v1",
                global_shape: &[8, 16, 8],
                tp_sharding: TpSharding::Packed {
                    axis: 1,
                    component_extents: &[8, 8],
                },
                ep_axis: Some(0),
                dtype: "uint8",
                itemsize: 1,
            },
            source: ParallelLayout {
                tp: 2,
                pp: 4,
                ep: 4,
                dp: 1,
            },
            target: ParallelLayout {
                tp: 4,
                pp: 2,
                ep: 2,
                dp: 1,
            },
        },
    ]
}

pub fn find_case(name: &str) -> Option<Qwen3Case> {
    all_cases().into_iter().find(|case| case.name == name)
}

pub fn build_request(
    plan_id: &str,
    tensor: TensorSpec,
    source: ParallelLayout,
    target: ParallelLayout,
) -> ConversionRequest {
    ConversionRequest {
        format_version: 1,
        plan_id: plan_id.to_owned(),
        source_manifest: build_manifest(tensor, source, SOURCE_BASE_DEV, SOURCE_BASE_ADDR),
        target_manifest: build_manifest(tensor, target, TARGET_BASE_DEV, TARGET_BASE_ADDR),
    }
}

#[allow(dead_code)]
pub fn expected_payload_bytes(tensor: TensorSpec, target_dp: u64) -> u64 {
    tensor
        .global_shape
        .iter()
        .copied()
        .try_fold(tensor.itemsize, u64::checked_mul)
        .and_then(|bytes| bytes.checked_mul(target_dp))
        .expect("Qwen3 case byte count must fit in u64")
}

fn build_manifest(
    tensor: TensorSpec,
    layout: ParallelLayout,
    base_dev: u64,
    base_addr: u64,
) -> RuntimeModelManifest {
    let parallel = ParallelConfig {
        tp_size: layout.tp,
        pp_size: layout.pp,
        ep_size: layout.ep,
        dp_size: layout.dp,
    };
    let pp_rank = pp_owner(tensor.layer_id, tensor.num_layers, layout.pp);
    let mut tensors = Vec::new();

    for dp in 0..layout.dp {
        let ep_ranks = if tensor.ep_axis.is_some() {
            layout.ep
        } else {
            1
        };
        for ep in 0..ep_ranks {
            for tp in 0..layout.tp {
                tensors.extend(build_rank_fragments(
                    tensor, layout, pp_rank, dp, tp, ep, base_dev, base_addr,
                ));
            }
        }
    }

    RuntimeModelManifest {
        model_id: tensor.model_id.to_owned(),
        parallel,
        tensors,
        revision: None,
        instance_id: None,
        generation: None,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_rank_fragments(
    tensor: TensorSpec,
    layout: ParallelLayout,
    pp: u64,
    dp: u64,
    tp: u64,
    ep: u64,
    base_dev: u64,
    base_addr: u64,
) -> Vec<RuntimeTensorManifest> {
    let rank = ParallelRank { dp, tp, pp, ep };
    let device_index = (((dp * layout.pp + pp) * layout.tp + tp) * layout.ep) + ep;
    let dev = base_dev
        .checked_add(device_index)
        .expect("test device id must fit in u64");
    let tensor_base_addr = base_addr
        .checked_add(
            device_index
                .checked_mul(DEVICE_SLOT_BYTES)
                .expect("test address slot must fit in u64"),
        )
        .expect("test address must fit in u64");

    let mut full_local_shape = tensor.global_shape.to_vec();
    let mut base_logical_offset = vec![0; tensor.global_shape.len()];
    if let Some(axis) = tensor.ep_axis {
        assert_ne!(axis, tp_axis(tensor.tp_sharding));
        let local_extent = split_extent(tensor.global_shape[axis], layout.ep);
        full_local_shape[axis] = local_extent;
        base_logical_offset[axis] = ep * local_extent;
    }

    match tensor.tp_sharding {
        TpSharding::Contiguous { axis } => {
            let local_extent = split_extent(tensor.global_shape[axis], layout.tp);
            full_local_shape[axis] = local_extent;
            base_logical_offset[axis] = tp * local_extent;
            let strides = contiguous_strides(&full_local_shape, tensor.itemsize);
            vec![make_fragment(
                tensor,
                rank,
                dev,
                tensor_base_addr,
                base_logical_offset,
                full_local_shape.clone(),
                full_local_shape,
                strides,
            )]
        }
        TpSharding::Packed {
            axis,
            component_extents,
        } => {
            assert_eq!(
                component_extents.iter().sum::<u64>(),
                tensor.global_shape[axis]
            );
            let local_components: Vec<_> = component_extents
                .iter()
                .map(|extent| split_extent(*extent, layout.tp))
                .collect();
            full_local_shape[axis] = local_components.iter().sum();
            let strides = contiguous_strides(&full_local_shape, tensor.itemsize);
            let mut global_component_offset = 0_u64;
            let mut local_component_offset = 0_u64;
            let mut fragments = Vec::with_capacity(component_extents.len());

            for (global_extent, local_extent) in component_extents
                .iter()
                .copied()
                .zip(local_components.iter().copied())
            {
                let mut logical_offset = base_logical_offset.clone();
                logical_offset[axis] = global_component_offset + tp * local_extent;
                let mut logical_shape = full_local_shape.clone();
                logical_shape[axis] = local_extent;
                let addr = tensor_base_addr
                    .checked_add(
                        local_component_offset
                            .checked_mul(strides[axis])
                            .expect("packed component address must fit in u64"),
                    )
                    .expect("packed component address must fit in u64");
                fragments.push(make_fragment(
                    tensor,
                    rank,
                    dev,
                    addr,
                    logical_offset,
                    logical_shape.clone(),
                    logical_shape,
                    strides.clone(),
                ));
                global_component_offset += global_extent;
                local_component_offset += local_extent;
            }
            fragments
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn make_fragment(
    tensor: TensorSpec,
    rank: ParallelRank,
    dev: u64,
    addr: u64,
    logical_offset: Vec<u64>,
    logical_shape: Vec<u64>,
    local_shape: Vec<u64>,
    strides_bytes: Vec<u64>,
) -> RuntimeTensorManifest {
    RuntimeTensorManifest {
        tensor_id: tensor.tensor_id.to_owned(),
        semantic: TensorSemantic {
            model_part: tensor.model_part.to_owned(),
            stack: tensor.stack.to_owned(),
            layer_id: Some(tensor.layer_id),
            module_path: tensor
                .module_path
                .iter()
                .map(|item| (*item).to_owned())
                .collect(),
            parameter_role: tensor.parameter_role.to_owned(),
            representation_id: tensor.representation_id.to_owned(),
            expert: None,
        },
        global_shape: tensor.global_shape.to_vec(),
        logical_offset,
        logical_shape,
        rank,
        local_shape: local_shape.clone(),
        view_kind: "canonical-affine:v1".to_owned(),
        dtype: tensor.dtype.to_owned(),
        itemsize: tensor.itemsize,
        strides_bytes: strides_bytes.clone(),
        dev,
        addr,
        storage_nbytes: storage_span(&local_shape, &strides_bytes, tensor.itemsize),
        fragment_id: None,
        worker_id: None,
        lease_generation: None,
        postprocess: Vec::new(),
        legacy_semantic_fingerprint: None,
        legacy_physical_layout_id: None,
    }
}

fn pp_owner(layer_id: u64, num_layers: u64, pp_size: u64) -> u64 {
    (layer_id * pp_size / num_layers).min(pp_size - 1)
}

fn tp_axis(sharding: TpSharding) -> usize {
    match sharding {
        TpSharding::Contiguous { axis } | TpSharding::Packed { axis, .. } => axis,
    }
}

fn split_extent(extent: u64, parts: u64) -> u64 {
    assert_eq!(extent % parts, 0);
    extent / parts
}

fn contiguous_strides(shape: &[u64], itemsize: u64) -> Vec<u64> {
    let mut strides = vec![itemsize; shape.len()];
    for dim in (0..shape.len().saturating_sub(1)).rev() {
        strides[dim] = strides[dim + 1]
            .checked_mul(shape[dim + 1])
            .expect("test stride must fit in u64");
    }
    strides
}

fn storage_span(shape: &[u64], strides: &[u64], itemsize: u64) -> u64 {
    shape
        .iter()
        .zip(strides)
        .try_fold(itemsize, |span, (extent, stride)| {
            (extent - 1)
                .checked_mul(*stride)
                .and_then(|contribution| span.checked_add(contribution))
        })
        .expect("test storage span must fit in u64")
}
