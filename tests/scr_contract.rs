use heterogeneous_weight_conversion::{
    ConversionRequest, ManifestWeightConversionPlugin, ScrTransferPlan,
};
use pretty_assertions::assert_eq;

const REQUEST_JSON: &str = include_str!("fixtures/scr42_request.json");
const PLAN_JSON: &str = include_str!("fixtures/scr42_plan_chunk1.json");

#[test]
fn stable_request_round_trips_without_schema_changes() {
    let request = ConversionRequest::from_json(REQUEST_JSON).expect("request must parse");

    assert_eq!(
        request.to_json().expect("request must serialize"),
        REQUEST_JSON.trim_end()
    );
}

#[test]
fn rust_planner_matches_the_python_scr_plan() {
    let request = ConversionRequest::from_json(REQUEST_JSON).expect("request must parse");
    let expected = ScrTransferPlan::from_json(PLAN_JSON).expect("plan must parse");

    let actual = ManifestWeightConversionPlugin::default()
        .plan_scr(&request, Some(1))
        .expect("planning must succeed");

    assert_eq!(actual, expected);
    assert_eq!(
        actual.to_json().expect("plan must serialize"),
        PLAN_JSON.trim_end()
    );
}
