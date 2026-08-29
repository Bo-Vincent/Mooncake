import copy
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from benchmarks.heterogeneous_weight_reshard.case_spec import (
    BenchmarkCase,
    BenchmarkConfig,
    CaseSpecError,
    MeshSpec,
    PlacementSpec,
    load_benchmark_config,
    load_benchmark_config_file,
    load_benchmark_config_json,
)


_CANONICAL_CASES = Path(__file__).parents[1] / "cases.json"


def _valid_config() -> dict:
    return {
        "schema_version": 1,
        "execution_enabled": False,
        "execution_guard": "VIN_RUN_BENCHMARK",
        "dtype": "uint8",
        "warmups": 3,
        "iterations": 20,
        "physical_gpus": 16,
        "placement": {
            "source_host": "source-node",
            "target_host": "target-node",
            "gpus_per_host": 8,
        },
        "metrics": ["process_wall_ms", "logical_GiBps"],
        "physical_cases": [
            {
                "id": "dp2_tp4_to_dp1_tp8_dim0",
                "source": {"replicas": 2, "shards": 4, "shard_dim": 0},
                "target": {"replicas": 1, "shards": 8, "shard_dim": 0},
                "global_shape": [2048, 2048, 1024],
                "required_ranks": 16,
            }
        ],
        "stress_cases": [
            {
                "id": "cross_dim0_to_dim2",
                "source": {"replicas": 1, "shards": 4, "shard_dim": 0},
                "target": {"replicas": 1, "shards": 4, "shard_dim": 2},
                "global_shape": [2048, 2048, 2048],
                "required_ranks": 8,
            }
        ],
        "planner_only_cases": [
            {
                "id": "tp_pp_ep_conversion",
                "reason": "literal rank counts exceed the physical cluster",
            }
        ],
    }


def _at(value: object, path: tuple[object, ...]) -> object:
    for part in path:
        value = value[part]  # type: ignore[index]
    return value


def _set(value: object, path: tuple[object, ...], replacement: object) -> None:
    parent = _at(value, path[:-1])
    parent[path[-1]] = replacement  # type: ignore[index]


def test_loads_valid_config_into_frozen_schema() -> None:
    raw = _valid_config()

    config = load_benchmark_config(raw)

    assert isinstance(config, BenchmarkConfig)
    assert isinstance(config.placement, PlacementSpec)
    assert config.placement == PlacementSpec(
        source_host="source-node",
        target_host="target-node",
        gpus_per_host=8,
    )
    assert config.metrics == ("process_wall_ms", "logical_GiBps")
    assert isinstance(config.physical_cases, tuple)
    assert isinstance(config.physical_cases[0], BenchmarkCase)
    assert config.physical_cases[0].category == "physical"
    assert config.stress_cases[0].category == "stress"
    assert config.planner_only_cases[0].category == "planner_only"
    assert config.physical_cases[0].global_shape == (2048, 2048, 1024)
    assert config.physical_cases[0].source == MeshSpec(2, 4, 0)
    with pytest.raises(FrozenInstanceError):
        config.iterations = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        config.placement.source_host = "other"  # type: ignore[misc]


def test_loading_detaches_immutable_values_from_input() -> None:
    raw = _valid_config()
    config = load_benchmark_config(raw)

    raw["metrics"].append("wire_GiBps")
    raw["physical_cases"][0]["global_shape"][0] = 16

    assert config.metrics == ("process_wall_ms", "logical_GiBps")
    assert config.physical_cases[0].global_shape == (2048, 2048, 1024)


def test_loads_from_json_text_and_file(tmp_path) -> None:
    raw = _valid_config()
    encoded = json.dumps(raw)
    path = tmp_path / "cases.json"
    path.write_text(encoded, encoding="utf-8")

    expected = load_benchmark_config(raw)

    assert load_benchmark_config_json(encoded) == expected
    assert load_benchmark_config_file(path) == expected
    assert BenchmarkConfig.from_dict(raw) == expected
    assert BenchmarkConfig.from_json(encoded) == expected
    assert BenchmarkConfig.from_json_file(path) == expected


def test_canonical_cases_are_valid_and_execution_stays_disabled() -> None:
    config = load_benchmark_config_file(_CANONICAL_CASES)

    assert config.execution_enabled is False
    assert config.execution_guard == "VIN_RUN_BENCHMARK"
    assert [case.id for case in config.physical_cases] == [
        "tp4_to_tp8_dim0",
        "tp8_to_tp4_dim0",
        "cross_dim0_to_dim1",
        "cross_dim0_to_dim2_physical",
        "dp2_tp4_to_dp1_tp8_dim0",
    ]
    assert [case.id for case in config.stress_cases] == ["cross_dim0_to_dim2_stress"]
    assert {case.id for case in config.planner_only_cases} == {
        "pp2_to_pp4_then_reverse",
        "ep8_to_ep2_then_reverse",
        "tp4_pp2_ep8_dp2_to_tp8_pp4_ep2_dp4",
    }


def test_reports_invalid_json() -> None:
    with pytest.raises(CaseSpecError, match=r"^config: invalid JSON:"):
        load_benchmark_config_json("{")


def test_computes_logical_bytes_and_rank_counts_exactly() -> None:
    config = load_benchmark_config(_valid_config())
    physical = config.physical_cases[0]
    stress = config.stress_cases[0]

    assert physical.logical_bytes == 4_294_967_296
    assert physical.source.total_ranks == 8
    assert physical.target.total_ranks == 8
    assert physical.source_total == 8
    assert physical.target_total == 8
    assert physical.world_size == 16
    assert stress.logical_bytes == 8_589_934_592
    assert stress.source_total == 4
    assert stress.target_total == 4
    assert stress.world_size == 8


def test_planner_only_case_can_omit_geometry() -> None:
    case = load_benchmark_config(_valid_config()).planner_only_cases[0]

    assert case.id == "tp_pp_ep_conversion"
    assert case.reason == "literal rank counts exceed the physical cluster"
    assert case.global_shape is None
    assert case.source is None
    assert case.target is None
    assert case.required_ranks is None
    assert case.logical_bytes is None
    assert case.source_total is None
    assert case.target_total is None
    assert case.world_size is None


def test_planner_only_case_can_keep_complete_geometry_beyond_capacity() -> None:
    raw = _valid_config()
    raw["planner_only_cases"] = [
        {
            "id": "large_mesh",
            "reason": "planner coverage only",
            "source": {"replicas": 2, "shards": 8, "shard_dim": 0},
            "target": {"replicas": 4, "shards": 8, "shard_dim": 1},
            "global_shape": [2048, 2048],
            "required_ranks": 48,
        }
    ]

    case = load_benchmark_config(raw).planner_only_cases[0]

    assert case.source_total == 16
    assert case.target_total == 32
    assert case.world_size == 48
    assert case.logical_bytes == 4_194_304


@pytest.mark.parametrize(
    ("container_path", "field", "message"),
    [
        ((), "stress_cases", r"config: missing fields: stress_cases"),
        (
            ("placement",),
            "target_host",
            r"config\.placement: missing fields: target_host",
        ),
        (
            ("physical_cases", 0),
            "required_ranks",
            r"physical_cases\[0\]: missing fields: required_ranks",
        ),
        (
            ("physical_cases", 0, "source"),
            "replicas",
            r"physical_cases\[0\]\.source: missing fields: replicas",
        ),
        (
            ("planner_only_cases", 0),
            "reason",
            r"planner_only_cases\[0\]: missing fields: reason",
        ),
    ],
)
def test_rejects_missing_fields(
    container_path: tuple[object, ...], field: str, message: str
) -> None:
    raw = _valid_config()
    del _at(raw, container_path)[field]  # type: ignore[index]

    with pytest.raises(CaseSpecError, match=message):
        load_benchmark_config(raw)


@pytest.mark.parametrize(
    ("container_path", "message"),
    [
        ((), r"config: unknown fields: surprise"),
        (("placement",), r"config\.placement: unknown fields: surprise"),
        (
            ("physical_cases", 0),
            r"physical_cases\[0\]: unknown fields: surprise",
        ),
        (
            ("physical_cases", 0, "source"),
            r"physical_cases\[0\]\.source: unknown fields: surprise",
        ),
        (
            ("planner_only_cases", 0),
            r"planner_only_cases\[0\]: unknown fields: surprise",
        ),
    ],
)
def test_rejects_unknown_fields(
    container_path: tuple[object, ...], message: str
) -> None:
    raw = _valid_config()
    _at(raw, container_path)["surprise"] = 1  # type: ignore[index]

    with pytest.raises(CaseSpecError, match=message):
        load_benchmark_config(raw)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("schema_version",), True, r"config\.schema_version: expected integer"),
        (("warmups",), False, r"config\.warmups: expected positive integer"),
        (
            ("placement", "gpus_per_host"),
            True,
            r"config\.placement\.gpus_per_host: expected positive integer",
        ),
        (
            ("physical_cases", 0, "source", "replicas"),
            True,
            r"physical_cases\[0\]\.source\.replicas: expected positive integer",
        ),
        (
            ("physical_cases", 0, "source", "shard_dim"),
            False,
            r"physical_cases\[0\]\.source\.shard_dim: expected non-negative integer",
        ),
        (
            ("physical_cases", 0, "global_shape"),
            [2048, True, 1024],
            r"physical_cases\[0\]\.global_shape\[1\]: expected positive integer",
        ),
        (
            ("physical_cases", 0, "required_ranks"),
            True,
            r"physical_cases\[0\]\.required_ranks: expected positive integer",
        ),
    ],
)
def test_bool_never_satisfies_integer_fields(
    path: tuple[object, ...], value: object, message: str
) -> None:
    raw = _valid_config()
    _set(raw, path, value)

    with pytest.raises(CaseSpecError, match=message):
        load_benchmark_config(raw)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("warmups",), 0, r"config\.warmups: expected positive integer"),
        (("iterations",), -1, r"config\.iterations: expected positive integer"),
        (("physical_gpus",), 0, r"config\.physical_gpus: expected positive integer"),
        (
            ("placement", "gpus_per_host"),
            0,
            r"config\.placement\.gpus_per_host: expected positive integer",
        ),
        (
            ("physical_cases", 0, "source", "replicas"),
            0,
            r"physical_cases\[0\]\.source\.replicas: expected positive integer",
        ),
        (
            ("physical_cases", 0, "target", "shards"),
            -1,
            r"physical_cases\[0\]\.target\.shards: expected positive integer",
        ),
        (
            ("physical_cases", 0, "global_shape"),
            [2048, 0, 1024],
            r"physical_cases\[0\]\.global_shape\[1\]: expected positive integer",
        ),
        (
            ("physical_cases", 0, "required_ranks"),
            0,
            r"physical_cases\[0\]\.required_ranks: expected positive integer",
        ),
    ],
)
def test_rejects_non_positive_integer_fields(
    path: tuple[object, ...], value: object, message: str
) -> None:
    raw = _valid_config()
    _set(raw, path, value)

    with pytest.raises(CaseSpecError, match=message):
        load_benchmark_config(raw)


def test_rejects_unsupported_schema_version() -> None:
    raw = _valid_config()
    raw["schema_version"] = 2

    with pytest.raises(
        CaseSpecError,
        match=r"config\.schema_version: unsupported version 2; expected 1",
    ):
        load_benchmark_config(raw)


def test_rejects_non_bool_execution_enabled() -> None:
    raw = _valid_config()
    raw["execution_enabled"] = 1

    with pytest.raises(
        CaseSpecError, match=r"config\.execution_enabled: expected bool"
    ):
        load_benchmark_config(raw)


@pytest.mark.parametrize("guard", ["", "VIN_RUN_BENCHMARK=1", "9INVALID"])
def test_rejects_invalid_execution_guard_name(guard: str) -> None:
    raw = _valid_config()
    raw["execution_guard"] = guard

    with pytest.raises(
        CaseSpecError,
        match=r"config\.execution_guard: expected an environment variable name",
    ):
        load_benchmark_config(raw)


def test_rejects_unknown_dtype() -> None:
    raw = _valid_config()
    raw["dtype"] = "float16"

    with pytest.raises(
        CaseSpecError,
        match=r"config\.dtype: unsupported dtype 'float16'; supported: uint8",
    ):
        load_benchmark_config(raw)


@pytest.mark.parametrize("shape", [[2048], [1, 2, 3, 4]])
def test_rejects_shape_rank_outside_two_or_three(shape: list[int]) -> None:
    raw = _valid_config()
    raw["physical_cases"][0]["global_shape"] = shape

    with pytest.raises(
        CaseSpecError,
        match=rf"physical_cases\[0\]\.global_shape: expected rank 2 or 3, got {len(shape)}",
    ):
        load_benchmark_config(raw)


@pytest.mark.parametrize("side", ["source", "target"])
@pytest.mark.parametrize("shard_dim", [-1, 3])
def test_rejects_shard_dim_outside_shape(side: str, shard_dim: int) -> None:
    raw = _valid_config()
    raw["physical_cases"][0][side]["shard_dim"] = shard_dim

    qualifier = "expected non-negative integer" if shard_dim < 0 else "must be in"
    with pytest.raises(
        CaseSpecError,
        match=rf"physical_cases\[0\]\.{side}\.shard_dim: {qualifier}",
    ):
        load_benchmark_config(raw)


@pytest.mark.parametrize("side", ["source", "target"])
def test_rejects_non_divisible_shard_dimension(side: str) -> None:
    raw = _valid_config()
    raw["physical_cases"][0]["global_shape"] = [10, 2048, 1024]
    if side == "source":
        raw["physical_cases"][0]["target"] = {
            "replicas": 1,
            "shards": 2,
            "shard_dim": 0,
        }
        raw["physical_cases"][0]["required_ranks"] = 10
    else:
        raw["physical_cases"][0]["source"] = {
            "replicas": 2,
            "shards": 2,
            "shard_dim": 0,
        }
        raw["physical_cases"][0]["required_ranks"] = 12

    with pytest.raises(
        CaseSpecError,
        match=(
            rf"physical_cases\[0\]\.global_shape\[0\] \(10\) must be divisible "
            rf"by {side}\.shards"
        ),
    ):
        load_benchmark_config(raw)


def test_rejects_required_rank_mismatch() -> None:
    raw = _valid_config()
    raw["physical_cases"][0]["required_ranks"] = 15

    with pytest.raises(
        CaseSpecError,
        match=(
            r"physical_cases\[0\]\.required_ranks: expected 16 from "
            r"source_total \+ target_total, got 15"
        ),
    ):
        load_benchmark_config(raw)


@pytest.mark.parametrize("side", ["source", "target"])
def test_rejects_physical_side_exceeding_single_host_capacity(side: str) -> None:
    raw = _valid_config()
    raw["physical_cases"][0][side] = {
        "replicas": 3,
        "shards": 3,
        "shard_dim": 0,
    }
    raw["physical_cases"][0]["global_shape"] = [6144, 2048, 1024]
    other_total = 8
    raw["physical_cases"][0]["required_ranks"] = 9 + other_total
    raw["physical_gpus"] = 32

    with pytest.raises(
        CaseSpecError,
        match=(
            rf"physical_cases\[0\]\.{side}: requires 9 ranks but "
            r"config\.placement\.gpus_per_host is 8"
        ),
    ):
        load_benchmark_config(raw)


def test_rejects_physical_world_larger_than_available_gpus() -> None:
    raw = _valid_config()
    raw["physical_gpus"] = 15

    with pytest.raises(
        CaseSpecError,
        match=r"physical_cases\[0\]\.world_size: 16 exceeds config\.physical_gpus 15",
    ):
        load_benchmark_config(raw)


def test_stress_case_is_not_rejected_by_physical_capacity_gate() -> None:
    raw = _valid_config()
    raw["stress_cases"][0] = {
        "id": "oversized_stress",
        "source": {"replicas": 2, "shards": 8, "shard_dim": 0},
        "target": {"replicas": 2, "shards": 8, "shard_dim": 1},
        "global_shape": [2048, 2048],
        "required_ranks": 32,
    }

    assert load_benchmark_config(raw).stress_cases[0].world_size == 32


def test_rejects_duplicate_ids_across_categories() -> None:
    raw = _valid_config()
    raw["planner_only_cases"][0]["id"] = raw["physical_cases"][0]["id"]

    with pytest.raises(
        CaseSpecError,
        match=(
            r"config: duplicate case id 'dp2_tp4_to_dp1_tp8_dim0' in "
            r"physical_cases\[0\] and planner_only_cases\[0\]"
        ),
    ):
        load_benchmark_config(raw)


def test_rejects_partial_planner_only_geometry() -> None:
    raw = _valid_config()
    raw["planner_only_cases"][0]["global_shape"] = [2048, 2048]

    with pytest.raises(
        CaseSpecError,
        match=(
            r"planner_only_cases\[0\]: geometry fields must be provided together: "
            r"global_shape, required_ranks, source, target"
        ),
    ):
        load_benchmark_config(raw)


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("metrics",), [], r"config\.metrics: expected a non-empty list"),
        (("metrics",), ["ok", 3], r"config\.metrics\[1\]: expected non-empty string"),
        (("physical_cases",), {}, r"config\.physical_cases: expected list"),
        (
            ("physical_cases", 0, "global_shape"),
            "2048,2048",
            r"physical_cases\[0\]\.global_shape: expected list",
        ),
    ],
)
def test_rejects_invalid_collection_types(
    path: tuple[object, ...], replacement: object, message: str
) -> None:
    raw = _valid_config()
    _set(raw, path, replacement)

    with pytest.raises(CaseSpecError, match=message):
        load_benchmark_config(raw)


@pytest.mark.parametrize(
    ("enabled", "env_value", "authorized"),
    [
        (False, None, False),
        (False, "1", False),
        (True, None, False),
        (True, "0", False),
        (True, 1, False),
        (True, "1", True),
    ],
)
def test_execution_requires_both_gates(
    enabled: bool, env_value: object, authorized: bool
) -> None:
    raw = _valid_config()
    raw["execution_enabled"] = enabled
    config = load_benchmark_config(raw)
    env = {} if env_value is None else {"VIN_RUN_BENCHMARK": env_value}

    assert config.execution_is_authorized(env) is authorized


def test_execution_gate_never_reads_global_environment(monkeypatch) -> None:
    raw = _valid_config()
    raw["execution_enabled"] = True
    config = load_benchmark_config(raw)
    monkeypatch.setenv("VIN_RUN_BENCHMARK", "1")

    assert config.execution_is_authorized({}) is False


def test_input_is_not_mutated_during_failed_load() -> None:
    raw = _valid_config()
    raw["dtype"] = "float16"
    before = copy.deepcopy(raw)

    with pytest.raises(CaseSpecError):
        load_benchmark_config(raw)

    assert raw == before
