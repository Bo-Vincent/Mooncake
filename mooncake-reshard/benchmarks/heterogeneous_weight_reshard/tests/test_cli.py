import copy
import inspect
import json
from pathlib import Path

import pytest

from benchmarks.heterogeneous_weight_reshard import __main__ as cli


_CASES = Path(__file__).parents[1] / "cases.json"


def _argv(*extra: str) -> list[str]:
    return [
        "dry-run",
        "--config",
        str(_CASES),
        "--m2n-binary",
        "/opt/m2n/reshard_bench",
        "--json",
        *extra,
    ]


def _run_json(capsys, *extra: str) -> tuple[str, dict[str, object]]:
    assert cli.main(_argv(*extra)) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    return captured.out, json.loads(captured.out)


def test_all_case_json_is_deterministic_and_contains_no_samples(capsys) -> None:
    first_text, first = _run_json(capsys)
    second_text, second = _run_json(capsys)

    assert first_text == second_text
    assert first == second
    assert first["mode"] == "dry-run"
    assert first["execution_authorized"] is False
    assert first["execution_enabled"] is False
    assert first["execution_guard"] == "VIN_RUN_BENCHMARK"
    assert first["metrics"] == [
        "process_wall_ms",
        "cold_e2e_ms",
        "first_update_ms",
        "steady_update_ms",
        "data_plane_ms",
        "logical_GiBps",
        "wire_GiBps",
        "transport_init_ms",
        "registration_ms",
        "plan_ms",
        "lowering_ms",
    ]
    assert "samples" not in first
    assert "results" not in first


def test_all_categories_are_visible_with_explicit_backend_status(capsys) -> None:
    _, report = _run_json(capsys)
    cases = {case["id"]: case for case in report["cases"]}

    assert list(cases) == [
        "tp4_to_tp8_dim0",
        "tp8_to_tp4_dim0",
        "cross_dim0_to_dim1",
        "cross_dim0_to_dim2_physical",
        "dp2_tp4_to_dp1_tp8_dim0",
        "cross_dim0_to_dim2_stress",
        "pp2_to_pp4_then_reverse",
        "ep8_to_ep2_then_reverse",
        "tp4_pp2_ep8_dp2_to_tp8_pp4_ep2_dp4",
    ]
    assert cases["tp4_to_tp8_dim0"]["m2n"]["status"] == "planned"
    assert cases["tp4_to_tp8_dim0"]["mooncake"]["status"] == "planned"
    assert (
        cases["cross_dim0_to_dim2_stress"]["mooncake"]["summary"][
            "bounded_lowering_allowed"
        ]
        is False
    )
    planner_only = cases["pp2_to_pp4_then_reverse"]
    assert planner_only["category"] == "planner_only"
    assert planner_only["logical_bytes"] is None
    assert planner_only["m2n"]["status"] == "not_planned"
    assert planner_only["mooncake"]["status"] == "not_planned"


def test_case_filter_returns_only_the_requested_case(capsys) -> None:
    _, report = _run_json(capsys, "--case", "cross_dim0_to_dim1")

    assert [case["id"] for case in report["cases"]] == ["cross_dim0_to_dim1"]
    case = report["cases"][0]
    assert case["global_shape"] == [2048, 2048, 2048]
    assert case["m2n"]["world_size"] == 8
    assert case["mooncake"]["summary"]["region_count"] == 16


def test_unknown_case_fails_before_any_backend_execution(capsys) -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.main(_argv("--case", "does-not-exist"))

    captured = capsys.readouterr()
    assert "unknown case id: does-not-exist" in captured.err
    assert captured.out == ""


def test_cli_has_no_run_command(capsys) -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.main(["run"])

    assert "invalid choice" in capsys.readouterr().err


def test_dry_run_never_uses_global_execution_guard(
    tmp_path, monkeypatch, capsys
) -> None:
    raw = json.loads(_CASES.read_text(encoding="utf-8"))
    enabled = copy.deepcopy(raw)
    enabled["execution_enabled"] = True
    path = tmp_path / "enabled.json"
    path.write_text(json.dumps(enabled), encoding="utf-8")
    monkeypatch.setenv("VIN_RUN_BENCHMARK", "1")

    argv = _argv()
    argv[argv.index(str(_CASES))] = str(path)
    assert cli.main(argv) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["execution_enabled"] is True
    assert report["execution_authorized"] is False
    assert report["execution_refusal_reasons"] == [
        "dry-run never executes transfer backends"
    ]


def test_cli_source_contains_no_process_launch_path() -> None:
    source = inspect.getsource(cli)

    assert "subprocess" not in source
    assert "os.environ" not in source
    assert "Popen" not in source
