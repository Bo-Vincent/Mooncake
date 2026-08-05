"""Cold and steady benchmark runner for NVIDIA NCCL M2N."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .case_spec import BenchmarkCase, BenchmarkConfig, PlacementSpec
from .m2n_adapter import plan_m2n_case
from .m2n_executor import (
    M2NExecutionResult,
    M2NOutputError,
    M2NParsedResult,
    execute_m2n,
    parse_m2n_stdout,
)
from .mooncake_benchmark import BenchmarkExecutionError, ensure_execution_authorized


@dataclass(frozen=True)
class M2NCaseRun:
    result: dict[str, object]
    cold_execution: M2NExecutionResult
    steady_execution: M2NExecutionResult
    cold_parsed: M2NParsedResult
    steady_parsed: M2NParsedResult


def _validate_parsed(
    case: BenchmarkCase,
    parsed: M2NParsedResult,
    *,
    validation_requested: bool,
    warmups: int,
    iterations: int,
) -> None:
    if case.global_shape is None or case.source is None or case.target is None:
        raise M2NOutputError("case geometry is incomplete")
    if parsed.global_shape != case.global_shape:
        raise M2NOutputError("M2N output global shape does not match case")
    if parsed.source_ranks != case.source.total_ranks:
        raise M2NOutputError("M2N output source rank count does not match case")
    if parsed.destination_ranks != case.target.total_ranks:
        raise M2NOutputError("M2N output target rank count does not match case")
    if parsed.validation_requested is not validation_requested:
        raise M2NOutputError("M2N output validation mode does not match run")
    if (parsed.warmup, parsed.iterations) != (warmups, iterations):
        raise M2NOutputError("M2N output iteration counts do not match run")


def run_m2n_case(
    case: BenchmarkCase,
    placement: PlacementSpec,
    *,
    binary: str,
    launcher: str,
    algorithm: str,
    lb_mode: str,
    warmups: int,
    iterations: int,
    timeout_s: float,
    mpi_interface: str | None = None,
    export_env_names: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
    execute: Callable[..., M2NExecutionResult] = execute_m2n,
    parse: Callable[..., M2NParsedResult] = parse_m2n_stdout,
    record_execution: Callable[[str, M2NExecutionResult], None] | None = None,
) -> M2NCaseRun:
    cold_plan = plan_m2n_case(
        case,
        placement,
        binary=binary,
        launcher=launcher,
        algorithm=algorithm,
        lb_mode=lb_mode,
        warmups=0,
        iterations=1,
        validate=False,
        mpi_interface=mpi_interface,
        export_env_names=export_env_names,
    )
    cold_execution = execute(
        cold_plan.mpirun_argv,
        timeout_s=timeout_s,
        env=env,
    )
    if record_execution is not None:
        record_execution("cold", cold_execution)
    if cold_execution.timed_out:
        raise M2NOutputError("M2N cold run timed out")
    cold_parsed = parse(cold_execution.stdout, rc=cold_execution.rc)
    _validate_parsed(
        case,
        cold_parsed,
        validation_requested=False,
        warmups=0,
        iterations=1,
    )

    steady_plan = plan_m2n_case(
        case,
        placement,
        binary=binary,
        launcher=launcher,
        algorithm=algorithm,
        lb_mode=lb_mode,
        warmups=warmups,
        iterations=iterations,
        validate=True,
        mpi_interface=mpi_interface,
        export_env_names=export_env_names,
    )
    steady_execution = execute(
        steady_plan.mpirun_argv,
        timeout_s=timeout_s,
        env=env,
    )
    if record_execution is not None:
        record_execution("steady", steady_execution)
    if steady_execution.timed_out:
        raise M2NOutputError("M2N steady run timed out")
    steady_parsed = parse(steady_execution.stdout, rc=steady_execution.rc)
    _validate_parsed(
        case,
        steady_parsed,
        validation_requested=True,
        warmups=warmups,
        iterations=iterations,
    )
    if steady_parsed.validation_passed is not True:
        raise M2NOutputError("M2N steady validation did not pass")
    if case.logical_bytes is None:
        raise M2NOutputError("case logical bytes are unavailable")
    steady_seconds = steady_parsed.hot_latency_ms / 1000.0
    logical_gibps = case.logical_bytes / (2**30) / steady_seconds
    result = {
        "schema_version": 1,
        "backend": "nccl-m2n",
        "case_id": case.id,
        "logical_bytes": case.logical_bytes,
        "topology": {
            "source_replicas": case.source.replicas,
            "source_shards": case.source.shards,
            "source_shard_dim": case.source.shard_dim,
            "target_replicas": case.target.replicas,
            "target_shards": case.target.shards,
            "target_shard_dim": case.target.shard_dim,
        },
        "algorithm": steady_parsed.algorithm.lower(),
        "load_balance_mode": steady_parsed.lb_mode,
        "warmups": warmups,
        "iterations": iterations,
        "process_wall_ms": steady_execution.process_wall_ms,
        "steady_process_wall_ms": steady_execution.process_wall_ms,
        "benchmark_process_wall_ms": (
            cold_execution.process_wall_ms + steady_execution.process_wall_ms
        ),
        "cold_e2e_ms": cold_execution.process_wall_ms,
        "first_update_ms": cold_parsed.hot_latency_ms,
        "steady_update_ms": steady_parsed.hot_latency_ms,
        "data_plane_ms": steady_parsed.hot_latency_ms,
        "logical_gibps": logical_gibps,
        "m2n_reported_effective_gibps": (steady_parsed.effective_bandwidth_gib_s),
        "validation": {
            "passed": True,
            "target_ranks": list(steady_parsed.destination_validation_ranks),
        },
        "transport_init_ms": None,
        "registration_ms": None,
        "plan_ms": None,
        "lowering_ms": None,
        "timing_boundary": (
            "steady update is Overall/Max and includes ncclReshardWithWindow, "
            "CUDA stream synchronization, and per-iteration MPI barrier"
        ),
    }
    return M2NCaseRun(
        result=result,
        cold_execution=cold_execution,
        steady_execution=steady_execution,
        cold_parsed=cold_parsed,
        steady_parsed=steady_parsed,
    )


def _physical_case(config: BenchmarkConfig, case_id: str) -> BenchmarkCase:
    matches = [case for case in config.physical_cases if case.id == case_id]
    if len(matches) != 1:
        raise BenchmarkExecutionError(f"unknown physical case: {case_id}")
    return matches[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NCCL M2N reshard benchmark")
    parser.add_argument("--config", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--launcher", default="mpirun")
    parser.add_argument("--algorithm", choices=("ring", "direct"), default="ring")
    parser.add_argument("--lb-mode", choices=("uniform", "node"), default="uniform")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--mpi-interface")
    parser.add_argument("--export-env", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    args = _parser().parse_args(argv)
    if args.timeout <= 0:
        raise BenchmarkExecutionError("timeout must be positive")
    config = BenchmarkConfig.from_json_file(args.config)
    execution_env = dict(os.environ if environ is None else environ)
    ensure_execution_authorized(config, execution_env)
    if os.geteuid() == 0:
        execution_env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
        execution_env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    missing_exports = [name for name in args.export_env if not execution_env.get(name)]
    if missing_exports:
        raise BenchmarkExecutionError(
            "requested M2N environment variables are unset: "
            + ", ".join(missing_exports)
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def record_execution(phase: str, execution: M2NExecutionResult) -> None:
        (output_dir / f"{phase}.stdout.log").write_bytes(execution.stdout)
        (output_dir / f"{phase}.stderr.log").write_bytes(execution.stderr)

    run = run_m2n_case(
        _physical_case(config, args.case),
        config.placement,
        binary=args.binary,
        launcher=args.launcher,
        algorithm=args.algorithm,
        lb_mode=args.lb_mode,
        warmups=config.warmups,
        iterations=config.iterations,
        timeout_s=args.timeout,
        mpi_interface=args.mpi_interface,
        export_env_names=tuple(args.export_env),
        env=execution_env,
        record_execution=record_execution,
    )
    (output_dir / "result.json").write_text(
        json.dumps(run.result, indent=2, sort_keys=True) + "\n"
    )
    print("BENCHMARK_RESULT=" + json.dumps(run.result, sort_keys=True), flush=True)
    return run.result


if __name__ == "__main__":
    try:
        main()
    except (BenchmarkExecutionError, M2NOutputError, ValueError) as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
