"""Outer process runner for a distributed Mooncake benchmark job."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .case_spec import BenchmarkCase, BenchmarkConfig, PlacementSpec


class DistributedBenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True)
class DistributedCommandPlan:
    target_argv: tuple[str, ...]
    source_argv: tuple[str, ...]


_ROLE_MODULE = "benchmarks.heterogeneous_weight_reshard.mooncake_benchmark"


def plan_distributed_commands(
    case: BenchmarkCase,
    placement: PlacementSpec,
    *,
    config_path: str,
    python_binary: str,
    ssh_binary: str,
    control_port: int,
    protocol: str,
    timeout_s: float,
    remote_env: Mapping[str, str],
    phase: str = "steady",
) -> DistributedCommandPlan:
    if case.source is None or case.target is None or case.category != "physical":
        raise ValueError("distributed execution requires a physical case")
    if (
        case.source.total_ranks > placement.gpus_per_host
        or case.target.total_ranks > placement.gpus_per_host
    ):
        raise ValueError("case role rank count exceeds GPUs per host")
    if not config_path or not python_binary or not ssh_binary:
        raise ValueError("command paths must be non-empty")
    if not 1 <= control_port <= 65535:
        raise ValueError("control port is invalid")
    if protocol not in ("rdma", "tcp"):
        raise ValueError("protocol is invalid")
    if phase not in ("cold", "steady"):
        raise ValueError("phase is invalid")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout must be finite and positive")
    if any(
        not isinstance(key, str) or not key.isidentifier() or not isinstance(value, str)
        for key, value in remote_env.items()
    ):
        raise ValueError("remote environment is invalid")

    common = (
        python_binary,
        "-m",
        _ROLE_MODULE,
        "--config",
        config_path,
    )
    target_role = (
        *common,
        "target",
        "--bind-host",
        "0.0.0.0",
        "--control-port",
        str(control_port),
        "--engine-host",
        placement.target_host,
        "--protocol",
        protocol,
        "--cuda-devices",
        ",".join(str(rank) for rank in range(case.target.total_ranks)),
        "--timeout",
        str(timeout_s),
    )
    remote_command = shlex.join(
        (
            "env",
            *(f"{key}={remote_env[key]}" for key in sorted(remote_env)),
            "timeout",
            "--signal=TERM",
            "--kill-after=5s",
            f"{math.ceil(timeout_s + 10)}s",
            *target_role,
        )
    )
    source_argv = (
        *common,
        "source",
        "--case",
        case.id,
        "--target-control-host",
        placement.target_host,
        "--target-control-port",
        str(control_port),
        "--engine-host",
        placement.source_host,
        "--protocol",
        protocol,
        "--cuda-devices",
        ",".join(str(rank) for rank in range(case.source.total_ranks)),
        "--timeout",
        str(timeout_s),
        "--phase",
        phase,
    )
    return DistributedCommandPlan(
        target_argv=(ssh_binary, placement.target_host, remote_command),
        source_argv=source_argv,
    )


def _argv(value: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{label} argv must be a sequence of strings")
    normalized = tuple(value)
    if not normalized or any(
        not isinstance(argument, str) or not argument for argument in normalized
    ):
        raise ValueError(f"{label} argv must contain non-empty strings")
    return normalized


def _signal_process_group(
    process: subprocess.Popen[bytes], sig: signal.Signals
) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def _stop_process(process: subprocess.Popen[bytes], grace_s: float) -> None:
    if process.poll() is not None:
        return
    _signal_process_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        _signal_process_group(process, signal.SIGKILL)
        process.wait()


def _prefixed_json(text: str, prefix: str, label: str) -> dict[str, object]:
    values = [
        line[len(prefix) :] for line in text.splitlines() if line.startswith(prefix)
    ]
    if len(values) != 1:
        raise DistributedBenchmarkError(f"expected exactly one {label} marker")
    try:
        value = json.loads(values[0])
    except json.JSONDecodeError as error:
        raise DistributedBenchmarkError(f"invalid {label} JSON") from error
    if not isinstance(value, Mapping):
        raise DistributedBenchmarkError(f"{label} payload must be an object")
    return dict(value)


def run_process_pair(
    target_argv: Sequence[str],
    source_argv: Sequence[str],
    *,
    output_dir: str | os.PathLike[str],
    timeout_s: float,
    terminate_grace_s: float = 2.0,
    poll_interval_s: float = 0.05,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Run both roles and measure wall time through complete process reaping."""

    target_command = _argv(target_argv, "target")
    source_command = _argv(source_argv, "source")
    if timeout_s <= 0 or terminate_grace_s < 0 or poll_interval_s <= 0:
        raise ValueError("timeouts and poll interval are invalid")

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "target_stdout": directory / "target.stdout.log",
        "target_stderr": directory / "target.stderr.log",
        "source_stdout": directory / "source.stdout.log",
        "source_stderr": directory / "source.stderr.log",
    }

    target: subprocess.Popen[bytes] | None = None
    source: subprocess.Popen[bytes] | None = None
    timed_out = False
    started = clock()
    with (
        paths["target_stdout"].open("wb") as target_stdout,
        paths["target_stderr"].open("wb") as target_stderr,
        paths["source_stdout"].open("wb") as source_stdout,
        paths["source_stderr"].open("wb") as source_stderr,
    ):
        try:
            target = subprocess.Popen(
                target_command,
                stdout=target_stdout,
                stderr=target_stderr,
                start_new_session=True,
            )
            source = subprocess.Popen(
                source_command,
                stdout=source_stdout,
                stderr=source_stderr,
                start_new_session=True,
            )
            deadline = started + timeout_s
            while True:
                target_rc = target.poll()
                source_rc = source.poll()
                if target_rc is not None and target_rc != 0:
                    _stop_process(source, terminate_grace_s)
                    break
                if source_rc is not None and source_rc != 0:
                    _stop_process(target, terminate_grace_s)
                    break
                if target_rc is not None and source_rc is not None:
                    break
                if clock() >= deadline:
                    timed_out = True
                    _stop_process(source, terminate_grace_s)
                    _stop_process(target, terminate_grace_s)
                    break
                sleep(poll_interval_s)
        except BaseException:
            if source is not None:
                _stop_process(source, terminate_grace_s)
            if target is not None:
                _stop_process(target, terminate_grace_s)
            raise

    finished = clock()
    if target is None or source is None:
        raise RuntimeError("role processes were not started")
    if target.returncode is None:
        target.wait()
    if source.returncode is None:
        source.wait()
    if timed_out:
        raise DistributedBenchmarkError("Mooncake distributed job timed out")
    if target.returncode != 0:
        raise DistributedBenchmarkError(f"target exited with {target.returncode}")
    if source.returncode != 0:
        raise DistributedBenchmarkError(f"source exited with {source.returncode}")

    target_text = paths["target_stdout"].read_text(errors="replace")
    source_text = paths["source_stdout"].read_text(errors="replace")
    ready = _prefixed_json(
        target_text,
        "TARGET_CONTROL_READY=",
        "target readiness",
    )
    result = _prefixed_json(source_text, "BENCHMARK_RESULT=", "benchmark result")
    if "process_wall_ms" in result or "benchmark_process_wall_ms" in result:
        raise DistributedBenchmarkError("role result contains outer process timing")
    process_wall_ms = (finished - started) * 1000.0
    result.update(
        {
            "process_wall_ms": process_wall_ms,
            "benchmark_process_wall_ms": process_wall_ms,
            "target_control_ready": ready,
            "process_wall_boundary": (
                "before target process spawn through source and target process reap"
            ),
        }
    )
    (directory / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and measure both Mooncake heterogeneous benchmark roles"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python", dest="python_binary", default=sys.executable)
    parser.add_argument("--ssh", dest="ssh_binary", default="ssh")
    parser.add_argument("--control-port", type=int, default=19091)
    parser.add_argument("--protocol", choices=("rdma", "tcp"), default="rdma")
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser


def _physical_case(config: BenchmarkConfig, case_id: str) -> BenchmarkCase:
    matches = [case for case in config.physical_cases if case.id == case_id]
    if len(matches) != 1:
        raise DistributedBenchmarkError(f"unknown physical case: {case_id}")
    return matches[0]


def _process_wall(result: Mapping[str, object], phase: str) -> float:
    value = result.get("process_wall_ms")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise DistributedBenchmarkError(f"{phase} process wall time is invalid")
    return float(value)


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    runner: Callable[..., dict[str, object]] = run_process_pair,
) -> dict[str, object]:
    args = _parser().parse_args(argv)
    config_path = str(Path(args.config).resolve())
    config = BenchmarkConfig.from_json_file(config_path)
    execution_env = dict(os.environ if environ is None else environ)
    if not config.execution_is_authorized(execution_env):
        raise DistributedBenchmarkError(
            "benchmark execution is not authorized by config and environment guard"
        )
    remote_env = {
        key: execution_env[key]
        for key in (
            "PATH",
            "LD_LIBRARY_PATH",
            "PYTHONPATH",
            config.execution_guard,
        )
        if key in execution_env
    }
    case = _physical_case(config, args.case)
    plans = {
        phase: plan_distributed_commands(
            case,
            config.placement,
            config_path=config_path,
            python_binary=args.python_binary,
            ssh_binary=args.ssh_binary,
            control_port=args.control_port,
            protocol=args.protocol,
            timeout_s=args.timeout,
            remote_env=remote_env,
            phase=phase,
        )
        for phase in ("cold", "steady")
    }
    output_dir = Path(args.output_dir)
    phase_results = {}
    for phase in ("cold", "steady"):
        plan = plans[phase]
        phase_results[phase] = runner(
            plan.target_argv,
            plan.source_argv,
            output_dir=output_dir / phase,
            timeout_s=args.timeout + 15.0,
        )

    cold = phase_results["cold"]
    steady = phase_results["steady"]
    cold_wall_ms = _process_wall(cold, "cold")
    steady_wall_ms = _process_wall(steady, "steady")
    validation = steady.get("validation")
    if not isinstance(validation, Mapping) or validation.get("passed") is not True:
        raise DistributedBenchmarkError("steady validation did not pass")
    result = dict(steady)
    result.update(
        {
            "cold_e2e_ms": cold_wall_ms,
            "cold_process_wall_ms": cold_wall_ms,
            "steady_process_wall_ms": steady_wall_ms,
            "process_wall_ms": steady_wall_ms,
            "benchmark_process_wall_ms": cold_wall_ms + steady_wall_ms,
            "cold_first_update": cold.get("first_update"),
            "cold_control_plane_ms": cold.get("control_plane_ms"),
            "cold_target_control_ready": cold.get("target_control_ready"),
            "timing_boundary": (
                "cold is a separate one-update process; steady is a separate "
                "warmup/iteration/validation process"
            ),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print("BENCHMARK_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    return result


__all__ = [
    "DistributedBenchmarkError",
    "DistributedCommandPlan",
    "main",
    "plan_distributed_commands",
    "run_process_pair",
]


if __name__ == "__main__":
    try:
        main()
    except (DistributedBenchmarkError, ValueError) as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
