from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest

from benchmarks.heterogeneous_weight_reshard.mooncake_distributed_benchmark import (
    DistributedBenchmarkError,
    main,
    plan_distributed_commands,
    run_process_pair,
)
from benchmarks.heterogeneous_weight_reshard.case_spec import (
    BenchmarkCase,
    MeshSpec,
    PlacementSpec,
)


def _python(script: str) -> tuple[str, ...]:
    return (sys.executable, "-c", script)


CASES = Path(__file__).parents[1] / "cases.json"


def test_distributed_commands_use_explicit_mesh_hosts_and_devices() -> None:
    case = BenchmarkCase(
        id="tp4_to_tp8",
        category="physical",
        source=MeshSpec(replicas=1, shards=4, shard_dim=0),
        target=MeshSpec(replicas=1, shards=8, shard_dim=0),
        global_shape=(8, 8, 8),
        required_ranks=12,
    )

    plan = plan_distributed_commands(
        case,
        PlacementSpec("172.16.1.107", "172.16.1.108", 8),
        config_path="/nvme/benchmark/cases.json",
        python_binary="/usr/bin/python3",
        ssh_binary="ssh",
        control_port=19091,
        protocol="rdma",
        timeout_s=300.0,
        remote_env={"PYTHONPATH": "/runtime:/source", "LD_LIBRARY_PATH": "/lib"},
    )

    assert plan.target_argv[:2] == ("ssh", "172.16.1.108")
    remote = shlex.split(plan.target_argv[2])
    assert remote[:3] == [
        "env",
        "LD_LIBRARY_PATH=/lib",
        "PYTHONPATH=/runtime:/source",
    ]
    assert remote[remote.index("--engine-host") + 1] == "172.16.1.108"
    assert remote[remote.index("--cuda-devices") + 1] == "0,1,2,3,4,5,6,7"
    assert plan.source_argv[plan.source_argv.index("--engine-host") + 1] == (
        "172.16.1.107"
    )
    assert (
        plan.source_argv[plan.source_argv.index("--target-control-host") + 1]
        == "172.16.1.108"
    )
    assert plan.source_argv[plan.source_argv.index("--cuda-devices") + 1] == ("0,1,2,3")


def test_distributed_commands_forward_cold_phase_to_source_role() -> None:
    case = BenchmarkCase(
        id="tp1_to_tp1",
        category="physical",
        source=MeshSpec(replicas=1, shards=1, shard_dim=0),
        target=MeshSpec(replicas=1, shards=1, shard_dim=0),
        global_shape=(8, 8),
        required_ranks=2,
    )

    plan = plan_distributed_commands(
        case,
        PlacementSpec("source", "target", 8),
        config_path="/cases.json",
        python_binary="python3",
        ssh_binary="ssh",
        control_port=19091,
        protocol="rdma",
        timeout_s=30.0,
        remote_env={},
        phase="cold",
    )

    assert plan.source_argv[plan.source_argv.index("--phase") + 1] == "cold"


def test_main_requires_exact_execution_gate_before_spawning(tmp_path: Path) -> None:
    called = False

    def runner(*args, **kwargs):
        nonlocal called
        called = True
        del args, kwargs
        return {}

    with pytest.raises(DistributedBenchmarkError, match="not authorized"):
        main(
            [
                "--config",
                str(CASES),
                "--case",
                "tp4_to_tp8_dim0",
                "--output-dir",
                str(tmp_path),
            ],
            environ={"VIN_RUN_BENCHMARK": "1"},
            runner=runner,
        )

    assert called is False


def test_main_builds_and_runs_authorized_distributed_job(tmp_path: Path) -> None:
    raw = json.loads(CASES.read_text())
    raw["execution_enabled"] = True
    raw["placement"] = {
        "source_host": "172.16.1.107",
        "target_host": "172.16.1.108",
        "gpus_per_host": 8,
    }
    config = tmp_path / "cases.json"
    config.write_text(json.dumps(raw))
    observed = []

    def runner(target_argv, source_argv, **kwargs):
        phase = source_argv[source_argv.index("--phase") + 1]
        observed.append(
            {
                "phase": phase,
                "target_argv": target_argv,
                "source_argv": source_argv,
                "kwargs": kwargs,
            }
        )
        return {
            "backend": "mooncake-te-placement-binding",
            "case_id": "tp4_to_tp8_dim0",
            "process_wall_ms": 12.0 if phase == "cold" else 20.0,
            "first_update": {"total_ms": 3.0},
            "validation": {"passed": None if phase == "cold" else True},
        }

    result = main(
        [
            "--config",
            str(config),
            "--case",
            "tp4_to_tp8_dim0",
            "--output-dir",
            str(tmp_path / "output"),
            "--python",
            "/usr/bin/python3",
        ],
        environ={
            "VIN_RUN_BENCHMARK": "1",
            "PYTHONPATH": "/runtime:/source",
            "LD_LIBRARY_PATH": "/lib",
        },
        runner=runner,
    )

    assert result["cold_e2e_ms"] == 12.0
    assert result["steady_process_wall_ms"] == 20.0
    assert result["process_wall_ms"] == 20.0
    assert result["benchmark_process_wall_ms"] == 32.0
    assert result["validation"]["passed"] is True
    assert [item["phase"] for item in observed] == ["cold", "steady"]
    assert observed[0]["target_argv"][:2] == ("ssh", "172.16.1.108")
    assert "VIN_RUN_BENCHMARK=1" in shlex.split(observed[0]["target_argv"][2])
    assert observed[0]["source_argv"][0] == "/usr/bin/python3"
    assert observed[0]["kwargs"]["output_dir"] == tmp_path / "output" / "cold"
    assert observed[1]["kwargs"]["output_dir"] == tmp_path / "output" / "steady"


def test_process_pair_reports_full_wall_and_persists_role_logs(tmp_path: Path) -> None:
    target_ready = {
        "control_host": "0.0.0.0",
        "control_port": 19091,
        "engine_endpoint": "172.16.1.108:12345",
    }
    role_result = {
        "schema_version": 1,
        "backend": "mooncake-te-placement-binding",
        "protocol_wall_ms": 7.0,
    }
    target = _python(
        "import json, time; "
        f"print('TARGET_CONTROL_READY=' + json.dumps({target_ready!r}), flush=True); "
        "time.sleep(0.05)"
    )
    source = _python(
        "import json, time; time.sleep(0.02); "
        f"print('BENCHMARK_RESULT=' + json.dumps({role_result!r}), flush=True)"
    )

    result = run_process_pair(
        target,
        source,
        output_dir=tmp_path,
        timeout_s=2.0,
        poll_interval_s=0.001,
    )

    assert result["process_wall_ms"] > 0
    assert result["benchmark_process_wall_ms"] == result["process_wall_ms"]
    assert result["protocol_wall_ms"] == 7.0
    assert result["target_control_ready"] == target_ready
    assert "TARGET_CONTROL_READY=" in (tmp_path / "target.stdout.log").read_text()
    assert "BENCHMARK_RESULT=" in (tmp_path / "source.stdout.log").read_text()
    assert json.loads((tmp_path / "result.json").read_text()) == result


def test_process_pair_preserves_logs_when_target_fails(tmp_path: Path) -> None:
    target = _python(
        "import sys; print('target fatal', file=sys.stderr, flush=True); sys.exit(3)"
    )
    source = _python("import time; time.sleep(10)")

    with pytest.raises(DistributedBenchmarkError, match="target exited with 3"):
        run_process_pair(
            target,
            source,
            output_dir=tmp_path,
            timeout_s=2.0,
            poll_interval_s=0.001,
        )

    assert (tmp_path / "target.stderr.log").read_text() == "target fatal\n"
    assert (tmp_path / "source.stdout.log").exists()
