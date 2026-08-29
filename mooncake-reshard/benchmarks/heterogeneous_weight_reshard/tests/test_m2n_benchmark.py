from __future__ import annotations

import pytest

from benchmarks.heterogeneous_weight_reshard.case_spec import (
    BenchmarkCase,
    MeshSpec,
    PlacementSpec,
)
from benchmarks.heterogeneous_weight_reshard.m2n_benchmark import run_m2n_case
from benchmarks.heterogeneous_weight_reshard.m2n_executor import (
    M2NExecutionResult,
    M2NParsedResult,
    M2NSectionMetrics,
)


def _case() -> BenchmarkCase:
    return BenchmarkCase(
        id="tp2_to_tp4",
        category="physical",
        source=MeshSpec(replicas=1, shards=2, shard_dim=0),
        target=MeshSpec(replicas=1, shards=4, shard_dim=0),
        global_shape=(8, 8),
        required_ranks=6,
    )


def _parsed(*, hot_ms: float, validated: bool) -> M2NParsedResult:
    section = M2NSectionMetrics(
        time_min_ms=hot_ms / 2,
        time_max_ms=hot_ms,
        bandwidth_min_gib_s=1.0,
        bandwidth_max_gib_s=2.0,
        bandwidth_avg_gib_s=1.5,
    )
    return M2NParsedResult(
        global_shape=(8, 8),
        iterations=3 if validated else 1,
        warmup=2 if validated else 0,
        source_ranks=2,
        destination_ranks=4,
        total_data_bytes=64,
        algorithm="RING",
        lb_mode="UNIFORM",
        validation_requested=validated,
        validation_passed=True if validated else None,
        destination_validation_ranks=(2, 3, 4, 5) if validated else (),
        overall=section,
        sources=section,
        destinations=section,
        effective_bandwidth_gib_s=1.75,
    )


def test_run_case_uses_separate_cold_and_validated_steady_processes() -> None:
    executions = {
        b"cold": M2NExecutionResult(b"cold", b"", 0, False, 100.0),
        b"steady": M2NExecutionResult(b"steady", b"", 0, False, 250.0),
    }
    calls = []

    def execute(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        key = b"cold" if "--validate" not in argv else b"steady"
        return executions[key]

    def parse(stdout, *, rc):
        assert rc == 0
        return (
            _parsed(hot_ms=10.0, validated=False)
            if stdout == b"cold"
            else _parsed(hot_ms=20.0, validated=True)
        )

    run = run_m2n_case(
        _case(),
        PlacementSpec("source", "target", 8),
        binary="/opt/m2n/reshard_bench",
        launcher="mpirun",
        algorithm="ring",
        lb_mode="uniform",
        warmups=2,
        iterations=3,
        timeout_s=30.0,
        execute=execute,
        parse=parse,
    )

    cold_argv, steady_argv = calls[0][0], calls[1][0]
    assert cold_argv[cold_argv.index("--warmup") + 1] == "0"
    assert cold_argv[cold_argv.index("--iterations") + 1] == "1"
    assert "--validate" not in cold_argv
    assert steady_argv[steady_argv.index("--warmup") + 1] == "2"
    assert steady_argv[steady_argv.index("--iterations") + 1] == "3"
    assert "--validate" in steady_argv
    assert run.result["cold_e2e_ms"] == 100.0
    assert run.result["first_update_ms"] == 10.0
    assert run.result["steady_update_ms"] == 20.0
    assert run.result["steady_process_wall_ms"] == 250.0
    assert run.result["benchmark_process_wall_ms"] == 350.0
    assert run.result["validation"]["passed"] is True
    assert run.result["logical_gibps"] == 64 / (2**30) / 0.020


def test_run_case_applies_network_launch_options_to_both_processes() -> None:
    calls = []

    def execute(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        validated = "--validate" in argv
        return M2NExecutionResult(
            b"steady" if validated else b"cold",
            b"",
            0,
            False,
            20.0,
        )

    def parse(stdout, *, rc):
        assert rc == 0
        return _parsed(hot_ms=2.0, validated=stdout == b"steady")

    run_m2n_case(
        _case(),
        PlacementSpec("source", "target", 8),
        binary="reshard_bench",
        launcher="mpirun",
        algorithm="ring",
        lb_mode="uniform",
        warmups=2,
        iterations=3,
        timeout_s=30.0,
        mpi_interface="eth0",
        export_env_names=("NCCL_SOCKET_IFNAME", "NCCL_IB_HCA"),
        execute=execute,
        parse=parse,
    )

    for argv, _ in calls:
        assert argv.count("oob_tcp_if_include") == 2
        assert argv.count("btl_tcp_if_include") == 2
        assert argv.count("NCCL_SOCKET_IFNAME") == 2
        assert argv.count("NCCL_IB_HCA") == 2


def test_run_case_records_raw_phase_before_parse_failure() -> None:
    recorded = []

    def execute(argv, **kwargs):
        del argv, kwargs
        return M2NExecutionResult(b"partial", b"fatal", 1, False, 12.0)

    def parse(stdout, *, rc):
        del stdout, rc
        raise ValueError("invalid output")

    with pytest.raises(ValueError, match="invalid output"):
        run_m2n_case(
            _case(),
            PlacementSpec("source", "target", 8),
            binary="/opt/m2n/reshard_bench",
            launcher="mpirun",
            algorithm="ring",
            lb_mode="uniform",
            warmups=2,
            iterations=3,
            timeout_s=30.0,
            execute=execute,
            parse=parse,
            record_execution=lambda phase, execution: recorded.append(
                (phase, execution.stdout, execution.stderr)
            ),
        )

    assert recorded == [("cold", b"partial", b"fatal")]
