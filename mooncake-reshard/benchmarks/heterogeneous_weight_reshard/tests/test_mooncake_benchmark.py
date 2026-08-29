from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.heterogeneous_weight_reshard.mooncake_benchmark import (
    BenchmarkExecutionError,
    ensure_execution_authorized,
    parse_cuda_devices,
)
from benchmarks.heterogeneous_weight_reshard.case_spec import BenchmarkConfig


CASES = Path(__file__).parents[1] / "cases.json"


def _config(*, execution_enabled: bool) -> BenchmarkConfig:
    raw = json.loads(CASES.read_text())
    raw["execution_enabled"] = execution_enabled
    return BenchmarkConfig.from_dict(raw)


def test_parse_cuda_devices_requires_unique_non_negative_ids() -> None:
    assert parse_cuda_devices("0,2,7") == (0, 2, 7)

    for invalid in ("", "0,0", "-1", "0, 1", "false"):
        with pytest.raises(BenchmarkExecutionError, match="CUDA devices"):
            parse_cuda_devices(invalid)


def test_execution_requires_both_config_and_exact_environment_guard() -> None:
    with pytest.raises(BenchmarkExecutionError, match="not authorized"):
        ensure_execution_authorized(
            _config(execution_enabled=False),
            {"VIN_RUN_BENCHMARK": "1"},
        )
    with pytest.raises(BenchmarkExecutionError, match="not authorized"):
        ensure_execution_authorized(
            _config(execution_enabled=True),
            {"VIN_RUN_BENCHMARK": "true"},
        )

    ensure_execution_authorized(
        _config(execution_enabled=True),
        {"VIN_RUN_BENCHMARK": "1"},
    )
