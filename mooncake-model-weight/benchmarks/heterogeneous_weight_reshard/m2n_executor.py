"""Execution and stdout parsing for NVIDIA NCCL M2N reshard_bench."""

from __future__ import annotations

import math
import os
import re
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod
from os import PathLike


class M2NOutputError(ValueError):
    """Raised when reshard_bench did not produce a publishable result."""


@dataclass(frozen=True)
class M2NSectionMetrics:
    time_min_ms: float
    time_max_ms: float
    bandwidth_min_gib_s: float
    bandwidth_max_gib_s: float
    bandwidth_avg_gib_s: float


@dataclass(frozen=True)
class M2NParsedResult:
    global_shape: tuple[int, ...]
    iterations: int
    warmup: int
    source_ranks: int
    destination_ranks: int
    total_data_bytes: int
    algorithm: str
    lb_mode: str | None
    validation_requested: bool
    validation_passed: bool | None
    destination_validation_ranks: tuple[int, ...]
    overall: M2NSectionMetrics
    sources: M2NSectionMetrics
    destinations: M2NSectionMetrics
    effective_bandwidth_gib_s: float

    @property
    def hot_latency_ms(self) -> float:
        """Barrier-inclusive hot latency from the Overall/Max field."""

        return self.overall.time_max_ms


@dataclass(frozen=True)
class M2NExecutionResult:
    stdout: bytes
    stderr: bytes
    rc: int
    timed_out: bool
    process_wall_ms: float

    @property
    def returncode(self) -> int:
        return self.rc


_FLOAT_TOKEN = r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)|inf|nan)"
_TIME_RE = re.compile(
    rf"^Time per iteration \(ms\):\s+Min=({_FLOAT_TOKEN})\s+"
    rf"Max=({_FLOAT_TOKEN})$",
    re.IGNORECASE,
)
_BANDWIDTH_RE = re.compile(
    rf"^Bandwidth \(GB/s\):\s+Min=({_FLOAT_TOKEN})\s+"
    rf"Max=({_FLOAT_TOKEN})\s+Avg=({_FLOAT_TOKEN})$",
    re.IGNORECASE,
)
_EFFECTIVE_RE = re.compile(
    rf"^Total data throughput:\s+({_FLOAT_TOKEN}) GB/s$", re.IGNORECASE
)
_SOURCE_SECTION_RE = re.compile(r"^--- Sources only \((\d+) ranks\) ---$")
_DESTINATION_SECTION_RE = re.compile(r"^--- Destinations only \((\d+) ranks\) ---$")

_GLOBAL_SHAPE_RE = re.compile(
    r"^Global tensor: \[([0-9]+(?:,\s*[0-9]+){1,2})\] \(([23])D\)$",
    re.MULTILINE,
)
_SHARD_HEADER_RE = re.compile(
    r"^Source shard dim: (\d+), Dest shard dim: (\d+) "
    r"\((same-dim|CROSS-DIM!)\)$",
    re.MULTILINE,
)
_ROLE_RE_TEMPLATE = (
    r"^{role}: (\d+) ranks = (\d+) reps x (\d+) shards, "
    r"local=\[([0-9]+(?:,\s*[0-9]+){{1,2}})\]$"
)
_CONFIG_RE = re.compile(
    r"^Iterations: (\d+) \(warmup: (\d+)\), Validate: (yes|no)$",
    re.MULTILINE,
)
_SUMMARY_ITERATIONS_RE = re.compile(
    r"^Iterations: (\d+) \(warmup: (\d+)\)$", re.MULTILINE
)
_TOTAL_DATA_RE = re.compile(r"^Total data: (\d+) bytes \([^)]+ MB\)$", re.MULTILINE)
_SUMMARY_RANKS_RE = re.compile(
    r"^Sources: (\d+) ranks, Destinations: (\d+) ranks$", re.MULTILINE
)
_SUMMARY_SHARD_RE = re.compile(
    r"^Sharding: src_dim=(\d+), dst_dim=(\d+) "
    r"\((same-dim|cross-dim)\)$",
    re.MULTILINE,
)
_ALGORITHM_RE = re.compile(r"^Algorithm: (RING|DIRECT)$", re.MULTILINE)
_LB_MODE_RE = re.compile(r"^Load Balance Mode: (UNIFORM|NODE_AWARE)$", re.MULTILINE)
_VALIDATION_PASS_RE = re.compile(
    r"^\[Rank\s+(\d+)\] VALIDATION PASSED: (\d+) bytes correct$",
    re.MULTILINE,
)


def _single_match(pattern: re.Pattern[str], text: str, label: str) -> re.Match[str]:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise M2NOutputError(f"expected exactly one {label}")
    return matches[0]


def _dimensions(value: str, label: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in value.split(","))
    if len(dims) not in (2, 3) or any(dim <= 0 for dim in dims):
        raise M2NOutputError(f"invalid {label}")
    return dims


def _finite_nonnegative(value: str, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise M2NOutputError(f"{label} must be finite")
    if parsed < 0:
        raise M2NOutputError(f"{label} must be non-negative")
    return parsed


def _section_metrics(
    fields: dict[str, tuple[str, ...]], label: str
) -> M2NSectionMetrics:
    if "time" not in fields or "bandwidth" not in fields:
        raise M2NOutputError(f"incomplete {label} results section")
    time_min = _finite_nonnegative(fields["time"][0], f"{label} time min")
    time_max = _finite_nonnegative(fields["time"][1], f"{label} time max")
    bandwidth_min = _finite_nonnegative(
        fields["bandwidth"][0], f"{label} bandwidth min"
    )
    bandwidth_max = _finite_nonnegative(
        fields["bandwidth"][1], f"{label} bandwidth max"
    )
    bandwidth_avg = _finite_nonnegative(
        fields["bandwidth"][2], f"{label} bandwidth avg"
    )
    if time_min > time_max:
        raise M2NOutputError(f"invalid {label} time range")
    if bandwidth_min > bandwidth_max:
        raise M2NOutputError(f"invalid {label} bandwidth range")
    return M2NSectionMetrics(
        time_min_ms=time_min,
        time_max_ms=time_max,
        bandwidth_min_gib_s=bandwidth_min,
        bandwidth_max_gib_s=bandwidth_max,
        bandwidth_avg_gib_s=bandwidth_avg,
    )


def _parse_results_sections(
    results_text: str, source_ranks: int, destination_ranks: int
) -> tuple[M2NSectionMetrics, M2NSectionMetrics, M2NSectionMetrics, float]:
    sections: dict[str, dict[str, tuple[str, ...]]] = {}
    section_rank_counts: dict[str, int] = {}
    current: str | None = None
    effective_value: str | None = None

    for raw_line in results_text.splitlines():
        line = raw_line.strip()
        if line == "--- Overall (all ranks) ---":
            current = "Overall"
        else:
            source_heading = _SOURCE_SECTION_RE.fullmatch(line)
            destination_heading = _DESTINATION_SECTION_RE.fullmatch(line)
            if source_heading:
                current = "Sources"
                section_rank_counts[current] = int(source_heading.group(1))
            elif destination_heading:
                current = "Destinations"
                section_rank_counts[current] = int(destination_heading.group(1))
            elif line == "--- Effective bandwidth ---":
                current = "Effective"
            else:
                time_match = _TIME_RE.fullmatch(line)
                bandwidth_match = _BANDWIDTH_RE.fullmatch(line)
                effective_match = _EFFECTIVE_RE.fullmatch(line)
                if time_match and current in ("Overall", "Sources", "Destinations"):
                    fields = sections.setdefault(current, {})
                    if "time" in fields:
                        raise M2NOutputError(f"duplicate {current} time result")
                    fields["time"] = time_match.groups()
                elif bandwidth_match and current in (
                    "Overall",
                    "Sources",
                    "Destinations",
                ):
                    fields = sections.setdefault(current, {})
                    if "bandwidth" in fields:
                        raise M2NOutputError(f"duplicate {current} bandwidth result")
                    fields["bandwidth"] = bandwidth_match.groups()
                elif effective_match and current == "Effective":
                    if effective_value is not None:
                        raise M2NOutputError("duplicate effective bandwidth result")
                    effective_value = effective_match.group(1)
                continue

        if current in sections:
            raise M2NOutputError(f"duplicate {current} results section")
        if current in ("Overall", "Sources", "Destinations"):
            sections[current] = {}

    if "Overall" not in sections:
        raise M2NOutputError("missing Overall results section")
    if "Sources" not in sections or "Destinations" not in sections:
        raise M2NOutputError("missing role results section")
    if section_rank_counts.get("Sources") != source_ranks:
        raise M2NOutputError("Sources results rank count mismatch")
    if section_rank_counts.get("Destinations") != destination_ranks:
        raise M2NOutputError("Destinations results rank count mismatch")
    if effective_value is None:
        raise M2NOutputError("missing effective bandwidth result")

    overall = _section_metrics(sections["Overall"], "Overall")
    sources = _section_metrics(sections["Sources"], "Sources")
    destinations = _section_metrics(sections["Destinations"], "Destinations")
    effective = _finite_nonnegative(effective_value, "effective bandwidth")
    return overall, sources, destinations, effective


def parse_m2n_stdout(stdout: bytes | str, *, rc: int) -> M2NParsedResult:
    """Parse one successful reshard_bench stdout stream.

    Failed, incomplete, or internally inconsistent runs raise M2NOutputError so
    callers cannot accidentally publish timing from an invalid run.
    """

    if isinstance(stdout, bytes):
        text = stdout.decode("utf-8", errors="replace")
    elif isinstance(stdout, str):
        text = stdout
    else:
        raise TypeError("stdout must be bytes or str")

    if "=== Tensor Reshard Benchmark ===" not in text:
        raise M2NOutputError("output is not reshard_bench stdout")
    if rc != 0:
        raise M2NOutputError(f"reshard_bench exit code was {rc}")
    if "*** VALIDATION FAILED ***" in text or (
        "Benchmark completed with VALIDATION FAILURES." in text
    ):
        raise M2NOutputError("reshard_bench reported validation failure")

    results_marker = "       BENCHMARK RESULTS\n"
    completion_marker = "Benchmark completed successfully!"
    if text.count(results_marker) != 1:
        raise M2NOutputError("missing or duplicate results marker")
    if text.count(completion_marker) != 1:
        raise M2NOutputError("missing or duplicate completion marker")
    results_start = text.index(results_marker)
    completion_start = text.index(completion_marker)
    if completion_start <= results_start:
        raise M2NOutputError("completion marker precedes results")

    global_match = _single_match(_GLOBAL_SHAPE_RE, text, "global tensor header")
    global_shape = _dimensions(global_match.group(1), "global tensor shape")
    if len(global_shape) != int(global_match.group(2)):
        raise M2NOutputError("global tensor rank mismatch")

    shard_match = _single_match(_SHARD_HEADER_RE, text, "sharding header")
    source_shard_dim = int(shard_match.group(1))
    destination_shard_dim = int(shard_match.group(2))
    if source_shard_dim >= len(global_shape) or destination_shard_dim >= len(
        global_shape
    ):
        raise M2NOutputError("shard dimension is outside the tensor rank")

    role_values: dict[str, tuple[int, int, int, tuple[int, ...]]] = {}
    for role in ("Source", "Dest"):
        pattern = re.compile(_ROLE_RE_TEMPLATE.format(role=role), re.MULTILINE)
        match = _single_match(pattern, text, f"{role} header")
        local_shape = _dimensions(match.group(4), f"{role} local shape")
        if len(local_shape) != len(global_shape):
            raise M2NOutputError(f"{role} local tensor rank mismatch")
        role_values[role] = (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            local_shape,
        )

    source_ranks, source_replicas, source_shards, _ = role_values["Source"]
    destination_ranks, destination_replicas, destination_shards, destination_shape = (
        role_values["Dest"]
    )
    if source_ranks != source_replicas * source_shards:
        raise M2NOutputError("Source rank geometry mismatch")
    if destination_ranks != destination_replicas * destination_shards:
        raise M2NOutputError("Dest rank geometry mismatch")

    config_match = _single_match(_CONFIG_RE, text, "iteration header")
    iterations = int(config_match.group(1))
    warmup = int(config_match.group(2))
    validation_requested = config_match.group(3) == "yes"
    if iterations <= 0:
        raise M2NOutputError("iterations must be positive")

    summary_iterations = _single_match(
        _SUMMARY_ITERATIONS_RE, text, "summary iteration line"
    )
    if (int(summary_iterations.group(1)), int(summary_iterations.group(2))) != (
        iterations,
        warmup,
    ):
        raise M2NOutputError("summary iteration count mismatch")

    total_data_bytes = int(
        _single_match(_TOTAL_DATA_RE, text, "total data line").group(1)
    )
    if total_data_bytes != prod(global_shape):
        raise M2NOutputError("total data does not match the uint8 global tensor")

    summary_ranks = _single_match(_SUMMARY_RANKS_RE, text, "summary rank line")
    if (int(summary_ranks.group(1)), int(summary_ranks.group(2))) != (
        source_ranks,
        destination_ranks,
    ):
        raise M2NOutputError("summary rank count mismatch")

    summary_shard = _single_match(_SUMMARY_SHARD_RE, text, "summary sharding line")
    if (int(summary_shard.group(1)), int(summary_shard.group(2))) != (
        source_shard_dim,
        destination_shard_dim,
    ):
        raise M2NOutputError("summary shard dimensions mismatch")
    expected_shard_kind = (
        "same-dim" if source_shard_dim == destination_shard_dim else "cross-dim"
    )
    if summary_shard.group(3) != expected_shard_kind:
        raise M2NOutputError("summary sharding kind mismatch")

    algorithm = _single_match(_ALGORITHM_RE, text, "algorithm header").group(1)
    lb_matches = list(_LB_MODE_RE.finditer(text))
    if algorithm == "RING" and len(lb_matches) != 1:
        raise M2NOutputError("RING output requires one load-balance header")
    if algorithm == "DIRECT" and lb_matches:
        raise M2NOutputError("DIRECT output must not contain a load-balance header")
    lb_mode = lb_matches[0].group(1) if lb_matches else None

    results_text = text[results_start:completion_start]
    overall, sources, destinations, effective = _parse_results_sections(
        results_text, source_ranks, destination_ranks
    )

    validation_matches = list(_VALIDATION_PASS_RE.finditer(text))
    if validation_requested:
        if text.count("*** VALIDATION PASSED ***") != 1:
            raise M2NOutputError("missing global validation pass marker")
        expected_ranks = tuple(range(source_ranks, source_ranks + destination_ranks))
        observed_ranks = tuple(int(match.group(1)) for match in validation_matches)
        if tuple(sorted(observed_ranks)) != expected_ranks or len(
            set(observed_ranks)
        ) != len(observed_ranks):
            raise M2NOutputError("destination validation passes are incomplete")
        expected_bytes = prod(destination_shape)
        if any(int(match.group(2)) != expected_bytes for match in validation_matches):
            raise M2NOutputError("destination validation byte count mismatch")
        validation_passed: bool | None = True
        destination_validation_ranks = expected_ranks
    else:
        if validation_matches or "*** VALIDATION PASSED ***" in text:
            raise M2NOutputError("unexpected validation output")
        validation_passed = None
        destination_validation_ranks = ()

    return M2NParsedResult(
        global_shape=global_shape,
        iterations=iterations,
        warmup=warmup,
        source_ranks=source_ranks,
        destination_ranks=destination_ranks,
        total_data_bytes=total_data_bytes,
        algorithm=algorithm,
        lb_mode=lb_mode,
        validation_requested=validation_requested,
        validation_passed=validation_passed,
        destination_validation_ranks=destination_validation_ranks,
        overall=overall,
        sources=sources,
        destinations=destinations,
        effective_bandwidth_gib_s=effective,
    )


def _signal_process_group(
    process: subprocess.Popen[bytes], sig: signal.Signals
) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def execute_m2n(
    argv: Sequence[str],
    *,
    timeout_s: float | None = None,
    terminate_grace_s: float = 1.0,
    env: Mapping[str, str] | None = None,
    cwd: str | PathLike[str] | None = None,
) -> M2NExecutionResult:
    """Execute structured argv and measure the complete child process wall."""

    if isinstance(argv, (str, bytes)):
        raise TypeError("argv must be a sequence of strings")
    normalized_argv = tuple(argv)
    if not normalized_argv or any(
        not isinstance(argument, str) or not argument for argument in normalized_argv
    ):
        raise ValueError("argv must contain non-empty strings")
    if timeout_s is not None and timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if terminate_grace_s < 0:
        raise ValueError("terminate_grace_s must be non-negative")

    started_ns = time.perf_counter_ns()
    process = subprocess.Popen(
        normalized_argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=None if env is None else dict(env),
        cwd=cwd,
        start_new_session=True,
    )
    timed_out = False
    try:
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _signal_process_group(process, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=terminate_grace_s)
            except subprocess.TimeoutExpired:
                _signal_process_group(process, signal.SIGKILL)
                stdout, stderr = process.communicate()
    except BaseException:
        _signal_process_group(process, signal.SIGKILL)
        process.communicate()
        raise
    finished_ns = time.perf_counter_ns()

    if process.returncode is None:
        raise RuntimeError("subprocess was not reaped")
    return M2NExecutionResult(
        stdout=stdout,
        stderr=stderr,
        rc=process.returncode,
        timed_out=timed_out,
        process_wall_ms=(finished_ns - started_ns) / 1_000_000.0,
    )


__all__ = [
    "M2NExecutionResult",
    "M2NOutputError",
    "M2NParsedResult",
    "M2NSectionMetrics",
    "execute_m2n",
    "parse_m2n_stdout",
]
