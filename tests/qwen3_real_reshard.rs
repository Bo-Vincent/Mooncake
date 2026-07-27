#[path = "../examples/support/qwen3_cases.rs"]
mod qwen3_cases;

use std::collections::{BTreeMap, BTreeSet};

use heterogeneous_weight_conversion::{
    ConversionRequest, ManifestWeightConversionPlugin, RuntimeModelManifest, RuntimeTensorManifest,
};
use qwen3_cases::{all_cases, expected_payload_bytes, find_case};

const CHUNK_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug)]
struct DeviceBuffer {
    base_addr: u64,
    bytes: Vec<u8>,
}

fn assert_real_case(case_name: &str) {
    let case = find_case(case_name).expect("Qwen3 case must exist");
    let request = case.request();
    request.validate().expect("request must validate");

    let encoded = request.to_json().expect("request must serialize");
    let decoded = ConversionRequest::from_json(&encoded).expect("request must round-trip");
    assert_eq!(decoded, request);

    let plugin = ManifestWeightConversionPlugin::default();
    let plan = plugin
        .plan_scr(&request, Some(CHUNK_BYTES))
        .expect("real Qwen3 reshard case must plan");
    plugin
        .validate_scr_transfer_plan(&request, &plan, Some(CHUNK_BYTES))
        .expect("fresh Qwen3 plan must exactly rebind");

    assert!(!plan.transfers.is_empty());
    assert!(plan
        .transfers
        .iter()
        .all(|task| task.tensor_id == case.tensor.tensor_id));

    let copied_bytes: u64 = plan.transfers.iter().map(|task| task.nbytes).sum();
    assert_eq!(
        copied_bytes,
        expected_payload_bytes(case.tensor, case.target.dp)
    );

    let selected_source_devs: BTreeSet<_> = request
        .source_manifest
        .tensors
        .iter()
        .filter(|tensor| tensor.rank.dp == 0)
        .map(|tensor| tensor.dev)
        .collect();
    assert!(plan
        .transfers
        .iter()
        .all(|task| selected_source_devs.contains(&task.source.dev)));

    let expected_target_devs: BTreeSet<_> = request
        .target_manifest
        .tensors
        .iter()
        .map(|tensor| tensor.dev)
        .collect();
    let actual_target_devs: BTreeSet<_> =
        plan.transfers.iter().map(|task| task.target.dev).collect();
    assert_eq!(actual_target_devs, expected_target_devs);

    let source_pp = request.source_manifest.tensors[0].rank.pp;
    let target_pp = request.target_manifest.tensors[0].rank.pp;
    let source_key = format!("/S-p{source_pp}");
    let target_key = format!("/T-p{target_pp}");
    assert!(plan
        .transfers
        .iter()
        .all(|task| task.key.contains(&source_key) && task.key.contains(&target_key)));

    let encoded_plan = plan.to_json().expect("plan must serialize");
    assert_eq!(
        heterogeneous_weight_conversion::ScrTransferPlan::from_json(encoded_plan)
            .expect("plan must round-trip"),
        plan
    );
}

#[test]
fn qwen3_0_6b_dense_qkv_tp_and_pp_reshard() {
    assert_real_case("qwen3_0_6b_dense_qkv");
}

#[test]
fn qwen3_8b_dense_gate_up_tp_merge_and_pp_reshard() {
    assert_real_case("qwen3_8b_dense_gate_up");
}

#[test]
fn qwen3_30b_a3b_moe_w13_changes_tp_ep_pp_and_dp_together() {
    assert_real_case("qwen3_30b_a3b_moe_w13");
}

#[test]
fn qwen3_235b_a22b_moe_w2_changes_tp_ep_and_pp_together() {
    assert_real_case("qwen3_235b_a22b_moe_w2");
}

#[test]
fn qwen3_5_0_8b_gdn_qkvz_changes_tp_pp_and_fans_out_dp() {
    assert_real_case("qwen3_5_0_8b_gdn_qkvz");
}

#[test]
fn qwen3_vl_8b_vision_qkv_uses_real_packed_fragments() {
    assert_real_case("qwen3_vl_8b_vision_qkv");
}

#[test]
fn qwen3_235b_a22b_fp8_w13_scale_changes_ep() {
    assert_real_case("qwen3_235b_a22b_fp8_w13_scale");
}

#[test]
fn qwen3_fp8_w13_scale_rejects_cross_backend_representation_change() {
    let mut request = find_case("qwen3_235b_a22b_fp8_w13_scale")
        .expect("FP8 case must exist")
        .request();
    for tensor in &mut request.target_manifest.tensors {
        tensor.semantic.representation_id =
            "fp8-block128:flashinfer-cutlass-w31-scale-inv:canonical:v1".to_owned();
    }

    let error = ManifestWeightConversionPlugin::default()
        .plan_scr(&request, Some(CHUNK_BYTES))
        .expect_err("copy-only planner must reject a backend-format conversion");
    assert!(error.to_string().contains("semantic fingerprint mismatch"));
}

#[test]
fn qwen3_moe_w13_miniature_case_replays_every_transfer_byte() {
    let case = find_case("qwen3_30b_a3b_moe_w13_mini").expect("miniature case must exist");
    let request = case.request();
    let plan = ManifestWeightConversionPlugin::default()
        .plan_scr(&request, Some(7))
        .expect("miniature Qwen3 case must plan");

    let mut source_buffers = allocate_buffers(&request.source_manifest);
    for tensor in &request.source_manifest.tensors {
        write_canonical_fragment(
            source_buffers
                .get_mut(&tensor.dev)
                .expect("source device buffer must exist"),
            tensor,
        );
    }
    let mut target_buffers = allocate_buffers(&request.target_manifest);

    for task in &plan.transfers {
        let source = source_buffers
            .get(&task.source.dev)
            .expect("source transfer device must exist");
        let source_begin = usize::try_from(task.source.addr - source.base_addr)
            .expect("source offset must fit usize");
        let source_end =
            source_begin + usize::try_from(task.nbytes).expect("source size must fit usize");
        let payload = source.bytes[source_begin..source_end].to_vec();

        let target = target_buffers
            .get_mut(&task.target.dev)
            .expect("target transfer device must exist");
        let target_begin = usize::try_from(task.target.addr - target.base_addr)
            .expect("target offset must fit usize");
        let target_end = target_begin + payload.len();
        target.bytes[target_begin..target_end].copy_from_slice(&payload);
    }

    for tensor in &request.target_manifest.tensors {
        assert_canonical_fragment(
            target_buffers
                .get(&tensor.dev)
                .expect("target device buffer must exist"),
            tensor,
        );
    }
    assert_eq!(
        plan.transfers.iter().map(|task| task.nbytes).sum::<u64>(),
        expected_payload_bytes(case.tensor, 1)
    );
}

#[test]
fn qwen3_case_names_are_unique_and_requests_validate() {
    let cases = all_cases();
    let names: BTreeSet<_> = cases.iter().map(|case| case.name).collect();
    assert_eq!(names.len(), cases.len());
    for case in cases {
        case.request().validate().expect(case.description);
    }
}

fn allocate_buffers(manifest: &RuntimeModelManifest) -> BTreeMap<u64, DeviceBuffer> {
    let mut ranges: BTreeMap<u64, (u64, u64)> = BTreeMap::new();
    for tensor in &manifest.tensors {
        let end = tensor
            .addr
            .checked_add(tensor.storage_nbytes)
            .expect("test tensor address must fit in u64");
        ranges
            .entry(tensor.dev)
            .and_modify(|range| {
                range.0 = range.0.min(tensor.addr);
                range.1 = range.1.max(end);
            })
            .or_insert((tensor.addr, end));
    }
    ranges
        .into_iter()
        .map(|(dev, (begin, end))| {
            let size = usize::try_from(end - begin).expect("test buffer size must fit usize");
            (
                dev,
                DeviceBuffer {
                    base_addr: begin,
                    bytes: vec![0; size],
                },
            )
        })
        .collect()
}

fn write_canonical_fragment(buffer: &mut DeviceBuffer, tensor: &RuntimeTensorManifest) {
    for_each_3d_coordinate(&tensor.logical_shape, |local| {
        let global = [
            tensor.logical_offset[0] + local[0],
            tensor.logical_offset[1] + local[1],
            tensor.logical_offset[2] + local[2],
        ];
        let offset = physical_offset(buffer, tensor, local);
        buffer.bytes[offset] = canonical_value(global);
    });
}

fn assert_canonical_fragment(buffer: &DeviceBuffer, tensor: &RuntimeTensorManifest) {
    for_each_3d_coordinate(&tensor.logical_shape, |local| {
        let global = [
            tensor.logical_offset[0] + local[0],
            tensor.logical_offset[1] + local[1],
            tensor.logical_offset[2] + local[2],
        ];
        let offset = physical_offset(buffer, tensor, local);
        assert_eq!(
            buffer.bytes[offset],
            canonical_value(global),
            "target mismatch for {} at global coordinate {global:?}",
            tensor.tensor_id
        );
    });
}

fn for_each_3d_coordinate(shape: &[u64], mut visitor: impl FnMut([u64; 3])) {
    assert_eq!(shape.len(), 3);
    for dim0 in 0..shape[0] {
        for dim1 in 0..shape[1] {
            for dim2 in 0..shape[2] {
                visitor([dim0, dim1, dim2]);
            }
        }
    }
}

fn physical_offset(
    buffer: &DeviceBuffer,
    tensor: &RuntimeTensorManifest,
    local: [u64; 3],
) -> usize {
    let fragment_offset = tensor.addr - buffer.base_addr;
    let local_offset = local
        .iter()
        .zip(&tensor.strides_bytes)
        .map(|(index, stride)| index * stride)
        .sum::<u64>();
    usize::try_from(fragment_offset + local_offset).expect("physical offset must fit usize")
}

fn canonical_value(global: [u64; 3]) -> u8 {
    ((global[0] * 41 + global[1] * 13 + global[2] * 3) % 251) as u8
}
