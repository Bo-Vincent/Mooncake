use heterogeneous_weight_conversion::{
    create_plan, ConversionRequest, ManifestWeightConversionPlugin,
};
use serde_json::{json, Value};

const REQUEST_JSON: &str = include_str!("fixtures/scr42_request.json");

fn request_value() -> Value {
    serde_json::from_str(REQUEST_JSON).expect("fixture must parse")
}

fn request_from_value(value: &Value) -> ConversionRequest {
    ConversionRequest::from_json(&serde_json::to_string(value).expect("value must serialize"))
        .expect("request must parse")
}

#[test]
fn create_plan_and_plugin_facade_return_the_same_plan() {
    let request = ConversionRequest::from_json(REQUEST_JSON).expect("request must parse");
    let expected = ManifestWeightConversionPlugin::default()
        .plan_scr(&request, Some(1))
        .expect("plugin must plan");
    let actual = create_plan(
        request.plan_id.clone(),
        request.source_manifest.clone(),
        request.target_manifest.clone(),
        Some(1),
    )
    .expect("create_plan must plan");

    assert_eq!(actual, expected);
}

#[test]
fn exact_rebind_rejects_a_structurally_valid_but_stale_plan() {
    let request = ConversionRequest::from_json(REQUEST_JSON).expect("request must parse");
    let plugin = ManifestWeightConversionPlugin::default();
    let plan = plugin
        .plan_scr(&request, Some(1))
        .expect("plugin must plan");
    plugin
        .validate_scr_transfer_plan(&request, &plan, Some(1))
        .expect("fresh plan must rebind");

    let mut stale = plan;
    stale.transfers[0].target.addr += 4096;
    assert!(plugin
        .validate_scr_transfer_plan(&request, &stale, Some(1))
        .is_err());
}

#[test]
fn source_dp_selection_falls_back_without_mixing_replicas() {
    let mut value = request_value();
    value["source_manifest"]["parallel"]["dp_size"] = json!(2);
    let source_tensors = value["source_manifest"]["tensors"]
        .as_array_mut()
        .expect("tensors must be an array");
    source_tensors.remove(1);
    let mut dp1_tp0 = source_tensors[0].clone();
    dp1_tp0["rank"]["dp"] = json!(1);
    dp1_tp0["dev"] = json!(2);
    dp1_tp0["addr"] = json!(12288);
    let mut dp1_tp1 = dp1_tp0.clone();
    dp1_tp1["rank"]["tp"] = json!(1);
    dp1_tp1["logical_offset"] = json!([0, 2]);
    dp1_tp1["dev"] = json!(3);
    dp1_tp1["addr"] = json!(16384);
    source_tensors.push(dp1_tp0);
    source_tensors.push(dp1_tp1);

    let request = request_from_value(&value);
    let plan = ManifestWeightConversionPlugin::default()
        .plan_scr(&request, Some(1))
        .expect("DP1 must independently cover the tensor");

    assert!(plan
        .transfers
        .iter()
        .all(|transfer| matches!(transfer.source.dev, 2 | 3)));
}

#[test]
fn source_record_order_does_not_change_the_plan() {
    let original = request_from_value(&request_value());
    let mut reordered_value = request_value();
    reordered_value["source_manifest"]["tensors"]
        .as_array_mut()
        .expect("tensors must be an array")
        .reverse();
    let reordered = request_from_value(&reordered_value);
    let plugin = ManifestWeightConversionPlugin::default();

    assert_eq!(
        plugin
            .plan_scr(&original, Some(1))
            .expect("original must plan"),
        plugin
            .plan_scr(&reordered, Some(1))
            .expect("reordered request must plan")
    );
}
