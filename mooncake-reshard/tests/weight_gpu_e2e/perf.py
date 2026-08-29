from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from statistics import fmean, pstdev
from typing import Callable, Mapping


@dataclass(frozen=True)
class PerfConfig:
    total_bytes: int
    warmups: int
    iterations: int
    max_cv: float
    max_p95_p50: float
    min_store_gbps: float
    min_te_gbps: float
    topologies: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class PerfSummary:
    phase: str
    logical_bytes: int
    iterations: int
    mean_seconds: float
    p50_seconds: float
    p95_seconds: float
    p50_gbps: float | None
    cv: float
    p95_p50_ratio: float


def _parse_int(
    environ: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
) -> int:
    raw = environ.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if str(value) != raw.strip() or value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _parse_float(
    environ: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float,
    inclusive: bool,
) -> float:
    raw = environ.get(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    below_minimum = value < minimum if inclusive else value <= minimum
    if not math.isfinite(value) or below_minimum:
        comparator = "at least" if inclusive else "greater than"
        raise ValueError(f"{name} must be {comparator} {minimum}")
    return value


def _parse_topologies(raw: str) -> tuple[tuple[int, int], ...]:
    name = "MOONCAKE_WEIGHT_PERF_TOPOLOGIES"
    result = []
    try:
        for item in raw.split(","):
            source, target = item.split(":", maxsplit=1)
            source_tp = int(source)
            target_tp = int(target)
            if source_tp <= 0 or target_tp <= 0:
                raise ValueError
            result.append((source_tp, target_tp))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain positive source:target pairs") from error
    if not result:
        raise ValueError(f"{name} must not be empty")
    return tuple(result)


def _perf_config_from_environ(environ: Mapping[str, str]) -> PerfConfig:
    return PerfConfig(
        total_bytes=_parse_int(
            environ,
            "MOONCAKE_WEIGHT_PERF_BYTES",
            256 * 1024 * 1024,
            minimum=1,
        ),
        warmups=_parse_int(
            environ,
            "MOONCAKE_WEIGHT_PERF_WARMUPS",
            3,
            minimum=0,
        ),
        iterations=_parse_int(
            environ,
            "MOONCAKE_WEIGHT_PERF_ITERATIONS",
            30,
            minimum=1,
        ),
        max_cv=_parse_float(
            environ,
            "MOONCAKE_WEIGHT_PERF_MAX_CV",
            0.20,
            minimum=0.0,
            inclusive=False,
        ),
        max_p95_p50=_parse_float(
            environ,
            "MOONCAKE_WEIGHT_PERF_MAX_P95_P50",
            1.50,
            minimum=1.0,
            inclusive=True,
        ),
        min_store_gbps=_parse_float(
            environ,
            "MOONCAKE_WEIGHT_PERF_MIN_STORE_GBPS",
            0.0,
            minimum=0.0,
            inclusive=True,
        ),
        min_te_gbps=_parse_float(
            environ,
            "MOONCAKE_WEIGHT_PERF_MIN_TE_GBPS",
            0.0,
            minimum=0.0,
            inclusive=True,
        ),
        topologies=_parse_topologies(
            environ.get("MOONCAKE_WEIGHT_PERF_TOPOLOGIES", "4:2,2:4")
        ),
    )


def _percentile(samples: tuple[float, ...], percentile: float) -> float:
    ordered = tuple(sorted(samples))
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summarize_perf_samples(
    *,
    phase: str,
    logical_bytes: int,
    samples: tuple[float, ...],
) -> PerfSummary:
    if not phase or logical_bytes < 0 or not samples:
        raise ValueError("performance summary inputs must not be empty")
    if any(not math.isfinite(sample) or sample <= 0 for sample in samples):
        raise ValueError("performance samples must be finite and positive")
    mean_seconds = fmean(samples)
    p50_seconds = _percentile(samples, 0.50)
    p95_seconds = _percentile(samples, 0.95)
    return PerfSummary(
        phase=phase,
        logical_bytes=logical_bytes,
        iterations=len(samples),
        mean_seconds=mean_seconds,
        p50_seconds=p50_seconds,
        p95_seconds=p95_seconds,
        p50_gbps=(
            (logical_bytes / 1_000_000_000) / p50_seconds if logical_bytes else None
        ),
        cv=pstdev(samples) / mean_seconds,
        p95_p50_ratio=p95_seconds / p50_seconds,
    )


def _collect_perf_samples(
    *,
    warmups: int,
    iterations: int,
    run_once: Callable[[int], Mapping[str, float]],
) -> dict[str, tuple[float, ...]]:
    if warmups < 0 or iterations <= 0:
        raise ValueError("benchmark iteration counts are invalid")
    samples: dict[str, list[float]] = {}
    phases: tuple[str, ...] | None = None
    for iteration in range(warmups + iterations):
        current = run_once(iteration)
        current_phases = tuple(sorted(current))
        if not current_phases or (phases is not None and current_phases != phases):
            raise ValueError("benchmark phases changed between iterations")
        phases = current_phases
        for phase in current_phases:
            duration = current[phase]
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"benchmark phase is invalid: {phase}")
            if iteration >= warmups:
                samples.setdefault(phase, []).append(duration)
    return {phase: tuple(values) for phase, values in samples.items()}


def _assert_perf_gate(
    summary: PerfSummary,
    *,
    max_cv: float,
    max_p95_p50: float,
    min_gbps: float,
) -> None:
    if summary.cv > max_cv:
        raise AssertionError(
            f"{summary.phase} CV {summary.cv:.4f} exceeds {max_cv:.4f}"
        )
    if summary.p95_p50_ratio > max_p95_p50:
        raise AssertionError(
            f"{summary.phase} p95/p50 {summary.p95_p50_ratio:.4f} "
            f"exceeds {max_p95_p50:.4f}"
        )
    if summary.p50_gbps is None:
        if min_gbps == 0:
            return
        raise AssertionError(f"{summary.phase} has no GB/s measurement")
    if summary.p50_gbps < min_gbps:
        raise AssertionError(
            f"{summary.phase} {summary.p50_gbps:.4f} GB/s is below {min_gbps:.4f} GB/s"
        )


def _perf_result_payload(
    *,
    backend: str,
    total_bytes: int,
    source_tp: int,
    target_tp: int,
    warmups: int,
    iterations: int,
    samples: Mapping[str, tuple[float, ...]],
    logical_bytes: Mapping[str, int],
    placement: Mapping[str, object] | None = None,
    transport: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if set(samples) != set(logical_bytes):
        raise ValueError("performance phase byte accounting is incomplete")
    summaries = {
        phase: _summarize_perf_samples(
            phase=phase,
            logical_bytes=logical_bytes[phase],
            samples=durations,
        )
        for phase, durations in sorted(samples.items())
    }
    payload = {
        "schema_version": 1,
        "backend": backend,
        "total_bytes": total_bytes,
        "topology": {"source_tp": source_tp, "target_tp": target_tp},
        "warmups": warmups,
        "iterations": iterations,
        "phases": {phase: asdict(summary) for phase, summary in summaries.items()},
    }
    if placement is not None:
        payload["placement"] = dict(placement)
    if transport is not None:
        payload["transport"] = dict(transport)
    return payload


def _store_perf_result_payload(
    *,
    protocol: str,
    multi_gpu: bool,
    environ: Mapping[str, str],
    total_bytes: int,
    source_tp: int,
    target_tp: int,
    warmups: int,
    iterations: int,
    samples: Mapping[str, tuple[float, ...]],
    logical_bytes: Mapping[str, int],
    placement: Mapping[str, object] | None = None,
) -> dict[str, object]:
    normalized_protocol = protocol.strip().lower()
    if not normalized_protocol:
        raise ValueError("Store performance protocol must not be empty")
    memcpy_override = environ.get("MC_STORE_MEMCPY")
    memcpy_policy = (
        "auto" if memcpy_override is None else memcpy_override.strip().lower()
    )
    suffix = "-multi-gpu" if multi_gpu else ""
    return _perf_result_payload(
        backend=f"store-cuda-preregistered{suffix}",
        total_bytes=total_bytes,
        source_tp=source_tp,
        target_tp=target_tp,
        warmups=warmups,
        iterations=iterations,
        samples=samples,
        logical_bytes=logical_bytes,
        placement=placement,
        transport={
            "requested_protocol": normalized_protocol,
            "strategy": "runtime-selected",
            "mc_store_memcpy": memcpy_policy,
        },
    )


def _emit_perf_result(payload: Mapping[str, object]) -> None:
    print("MODEL_WEIGHT_PERF=" + json.dumps(payload, sort_keys=True))
