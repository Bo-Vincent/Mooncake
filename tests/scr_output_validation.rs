use heterogeneous_weight_conversion::{
    ConversionLimits, DeviceMemoryLocation, Result, ScrTransferPlan, ScrTransferTask,
};
use pretty_assertions::assert_eq;

const VALID_PLAN: &str = concat!(
    r#"{"format_version":1,"plan_id":"scr42","transfers":[{"#,
    r#""key":"wt/tx-scr42-copy00000123-4-0/S-p1t2e0/T-p0t1e2","#,
    r#""nbytes":1048576,"source":{"addr":17592186044416,"dev":2},"#,
    r#""target":{"addr":35184372088832,"dev":11},"#,
    r#""tensor_id":"model.layers.12.mlp.experts.7.w13_weight"}]}"#,
);

fn key(plan_id: &str, operation: &str, segment: usize, chunk: usize) -> String {
    format!("wt/tx-{plan_id}-{operation}-{segment}-{chunk}/S-p0t0e0/T-p0t0e0")
}

fn task(
    plan_id: &str,
    operation: &str,
    source_dev: u32,
    source_addr: u64,
    target_dev: u32,
    target_addr: u64,
    nbytes: u64,
) -> ScrTransferTask {
    ScrTransferTask::new(
        key(plan_id, operation, 0, 0),
        "tensor.weight",
        DeviceMemoryLocation::new(source_dev.into(), source_addr).unwrap(),
        DeviceMemoryLocation::new(target_dev.into(), target_addr).unwrap(),
        nbytes,
    )
    .unwrap()
}

fn parse_error(json: &str) -> String {
    ScrTransferPlan::from_json(json).unwrap_err().to_string()
}

fn plan(plan_id: &str, transfers: Vec<ScrTransferTask>) -> Result<ScrTransferPlan> {
    ScrTransferPlan::new(plan_id, transfers, &ConversionLimits::default())
}

#[test]
fn parses_and_emits_python_canonical_wire_format() {
    let plan = ScrTransferPlan::from_json(VALID_PLAN).unwrap();

    assert_eq!(plan.format_version, 1);
    assert_eq!(plan.plan_id, "scr42");
    assert_eq!(plan.transfers.len(), 1);
    assert_eq!(plan.transfers[0].source.dev, 2);
    assert_eq!(plan.transfers[0].target.dev, 11);
    assert_eq!(plan.to_json().unwrap(), VALID_PLAN);
}

#[test]
fn canonical_json_sorts_every_object_and_preserves_utf8() {
    let input = concat!(
        r#"{"transfers":[{"tensor_id":"层.权重","target":{"dev":1,"addr":32},"#,
        r#""source":{"dev":0,"addr":16},"nbytes":4,"#,
        r#""key":"wt/tx-p-copy0-0-0/S-p0t0e0/T-p0t0e0"}],"plan_id":"p","#,
        r#""format_version":1}"#,
    );
    let expected = concat!(
        r#"{"format_version":1,"plan_id":"p","transfers":[{"#,
        r#""key":"wt/tx-p-copy0-0-0/S-p0t0e0/T-p0t0e0","nbytes":4,"#,
        r#""source":{"addr":16,"dev":0},"target":{"addr":32,"dev":1},"#,
        r#""tensor_id":"层.权重"}]}"#,
    );

    assert_eq!(
        ScrTransferPlan::from_json(input)
            .unwrap()
            .to_json()
            .unwrap(),
        expected
    );
}

#[test]
fn serde_rejects_unknown_and_duplicate_fields_at_every_level() {
    let unknown_plan = VALID_PLAN.replacen(
        r#""format_version":1"#,
        r#""format_version":1,"extra":0"#,
        1,
    );
    let unknown_task =
        VALID_PLAN.replacen(r#""nbytes":1048576"#, r#""nbytes":1048576,"extra":0"#, 1);
    let unknown_location = VALID_PLAN.replacen(
        r#""addr":17592186044416,"dev":2"#,
        r#""addr":17592186044416,"dev":2,"extra":0"#,
        1,
    );
    let duplicate_plan_id = VALID_PLAN.replacen(
        r#""plan_id":"scr42""#,
        r#""plan_id":"scr42","plan_id":"scr42""#,
        1,
    );

    assert!(parse_error(&unknown_plan).contains("unknown field"));
    assert!(parse_error(&unknown_task).contains("unknown field"));
    assert!(parse_error(&unknown_location).contains("unknown field"));
    assert!(parse_error(&duplicate_plan_id).contains("duplicate field"));
}

#[test]
fn rejects_unsupported_version_and_plan_id_characters() {
    assert!(parse_error(
        &VALID_PLAN.replacen(r#""format_version":1"#, r#""format_version":2"#, 1,)
    )
    .contains("unsupported format_version"));
    assert!(
        parse_error(&VALID_PLAN.replacen(r#""plan_id":"scr42""#, r#""plan_id":"scr/42""#, 1,))
            .contains("plan_id contains unsupported key characters")
    );
}

#[test]
fn rejects_unbound_and_duplicate_transfer_keys() {
    let unbound = VALID_PLAN.replacen("tx-scr42-", "tx-other-", 1);
    assert!(parse_error(&unbound).contains("transfer key is not bound"));

    let duplicate_task = VALID_PLAN.strip_suffix("]}").unwrap().to_owned()
        + ","
        + VALID_PLAN
            .split_once("\"transfers\":[")
            .unwrap()
            .1
            .strip_suffix("]}")
            .unwrap()
        + "]}";
    assert!(parse_error(&duplicate_task).contains("duplicate keys"));
}

#[test]
fn enforces_json_task_and_key_limits() {
    let mut limits = ConversionLimits::default();
    limits.max_json_bytes = VALID_PLAN.len() - 1;
    assert!(ScrTransferPlan::from_json_with_limits(VALID_PLAN, &limits)
        .unwrap_err()
        .to_string()
        .contains("max_json_bytes"));

    let mut limits = ConversionLimits::default();
    limits.max_transfer_tasks = 1;
    let first = task("p", "copy0", 0, 100, 1, 200, 4);
    let second = task("p", "copy1", 0, 104, 1, 204, 4);
    let two_tasks = plan("p", vec![first, second]).unwrap().to_json().unwrap();
    assert!(ScrTransferPlan::from_json_with_limits(&two_tasks, &limits)
        .unwrap_err()
        .to_string()
        .contains("max_transfer_tasks"));

    let mut limits = ConversionLimits::default();
    limits.max_key_bytes = 8;
    assert!(ScrTransferPlan::from_json_with_limits(VALID_PLAN, &limits)
        .unwrap_err()
        .to_string()
        .contains("max_key_bytes"));
}

#[test]
fn rejects_unsorted_overlapping_target_writes_but_allows_adjacency() {
    let overlapping = vec![
        task("p", "copy0", 0, 100, 2, 204, 8),
        task("p", "copy1", 0, 108, 2, 200, 8),
    ];
    assert!(plan("p", overlapping)
        .unwrap_err()
        .to_string()
        .contains("overlapping target writes"));

    let adjacent = vec![
        task("p", "copy0", 0, 100, 2, 200, 4),
        task("p", "copy1", 0, 104, 2, 204, 4),
    ];
    plan("p", adjacent).unwrap();
}

#[test]
fn rejects_source_target_hazards_except_same_task_exact_noop() {
    let hazard = vec![
        task("p", "copy0", 0, 100, 1, 300, 8),
        task("p", "copy1", 2, 400, 0, 104, 8),
    ];
    assert!(plan("p", hazard)
        .unwrap_err()
        .to_string()
        .contains("source-target physical hazard"));

    let cross_task_exact_range = vec![
        task("p", "copy0", 0, 100, 1, 300, 8),
        task("p", "copy1", 2, 400, 0, 100, 8),
    ];
    assert!(plan("p", cross_task_exact_range)
        .unwrap_err()
        .to_string()
        .contains("source-target physical hazard"));

    let exact_noop = vec![task("p", "copy0", 0, 100, 0, 100, 8)];
    plan("p", exact_noop).unwrap();
}

#[test]
fn validates_device_and_uint64_address_space_boundaries() {
    DeviceMemoryLocation::new(i32::MAX as u64, u64::MAX).unwrap();
    assert!(DeviceMemoryLocation::new(i32::MAX as u64 + 1, 1).is_err());
    assert!(DeviceMemoryLocation::new(0, 0).is_err());

    task("p", "copy0", 0, u64::MAX, 1, u64::MAX, 1);
    assert!(ScrTransferTask::new(
        key("p", "copy0", 0, 0),
        "tensor.weight",
        DeviceMemoryLocation::new(0, u64::MAX).unwrap(),
        DeviceMemoryLocation::new(1, 1).unwrap(),
        2,
    )
    .unwrap_err()
    .to_string()
    .contains("64-bit address space"));
    assert!(ScrTransferTask::new(
        key("p", "copy0", 0, 0),
        "tensor.weight",
        DeviceMemoryLocation::new(0, 1).unwrap(),
        DeviceMemoryLocation::new(1, 2).unwrap(),
        0,
    )
    .is_err());
}
