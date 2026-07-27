use heterogeneous_weight_conversion::ConversionRequest;
use serde_json::{json, Value};

const REQUEST_JSON: &str = include_str!("fixtures/scr42_request.json");

fn request_value() -> Value {
    serde_json::from_str(REQUEST_JSON).expect("fixture must parse")
}

fn encoded(value: &Value) -> String {
    serde_json::to_string(value).expect("value must serialize")
}

#[test]
fn rejects_unknown_and_duplicate_request_fields() {
    let mut unknown = request_value();
    unknown["unexpected"] = json!(true);
    assert!(ConversionRequest::from_json(&encoded(&unknown)).is_err());

    let duplicate = REQUEST_JSON.replacen(
        r#""plan_id":"scr42""#,
        r#""plan_id":"scr42","plan_id":"other""#,
        1,
    );
    assert!(ConversionRequest::from_json(&duplicate).is_err());
}

#[test]
fn required_nullable_and_optional_non_null_fields_stay_distinct() {
    let mut missing_layer = request_value();
    missing_layer["source_manifest"]["tensors"][0]["semantic"]
        .as_object_mut()
        .expect("semantic must be an object")
        .remove("layer_id");
    assert!(ConversionRequest::from_json(&encoded(&missing_layer)).is_err());

    let mut null_layer = request_value();
    null_layer["source_manifest"]["tensors"][0]["semantic"]["layer_id"] = Value::Null;
    assert!(ConversionRequest::from_json(&encoded(&null_layer)).is_ok());

    let mut null_expert = request_value();
    null_expert["source_manifest"]["tensors"][0]["semantic"]["expert"] = Value::Null;
    assert!(ConversionRequest::from_json(&encoded(&null_expert)).is_err());
}

#[test]
fn accepted_legacy_fields_are_removed_from_canonical_json() {
    let mut legacy = request_value();
    legacy["source_manifest"]["revision"] = json!("old-revision");
    legacy["source_manifest"]["instance_id"] = json!("source-17");
    legacy["source_manifest"]["generation"] = json!(7);
    let tensor = &mut legacy["source_manifest"]["tensors"][0];
    tensor["fragment_id"] = json!("old-fragment");
    tensor["endpoint"] = json!("dev:0");
    tensor["worker_id"] = json!("worker-0");
    tensor["lease_generation"] = json!(5);
    tensor["semantic_fingerprint"] = json!("uint8:plain:canonical:v1");
    tensor["physical_layout_id"] = json!("canonical-affine:v1");
    tensor["postprocess"] = json!([]);

    let parsed = ConversionRequest::from_json(&encoded(&legacy)).expect("legacy input must parse");
    assert_eq!(
        parsed.to_json().expect("request must serialize"),
        REQUEST_JSON.trim_end()
    );
}

#[test]
fn uint64_address_space_end_is_inclusive_but_overflow_is_rejected() {
    let mut boundary = request_value();
    boundary["source_manifest"]["parallel"]["tp_size"] = json!(1);
    boundary["source_manifest"]["tensors"]
        .as_array_mut()
        .expect("tensors must be an array")
        .truncate(1);
    for side in ["source_manifest", "target_manifest"] {
        let tensor = &mut boundary[side]["tensors"][0];
        tensor["global_shape"] = json!([1]);
        tensor["logical_offset"] = json!([0]);
        tensor["logical_shape"] = json!([1]);
        tensor["local_shape"] = json!([1]);
        tensor["strides_bytes"] = json!([1]);
        tensor["storage_nbytes"] = json!(1);
        tensor["rank"]["tp"] = json!(0);
    }
    boundary["source_manifest"]["tensors"][0]["addr"] = json!(u64::MAX);
    assert!(ConversionRequest::from_json(&encoded(&boundary)).is_ok());

    boundary["source_manifest"]["tensors"][0]["storage_nbytes"] = json!(2);
    assert!(ConversionRequest::from_json(&encoded(&boundary)).is_err());
}

#[test]
fn target_manifest_must_declare_every_dp_replica_per_tensor() {
    let mut missing_dp = request_value();
    missing_dp["target_manifest"]["parallel"]["dp_size"] = json!(2);
    assert!(ConversionRequest::from_json(&encoded(&missing_dp)).is_err());
}
