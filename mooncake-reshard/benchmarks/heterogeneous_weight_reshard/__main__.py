"""Command-line entry point for heterogeneous weight reshard dry-runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .case_spec import (
    BenchmarkCase,
    BenchmarkConfig,
    CaseSpecError,
    MeshSpec,
    load_benchmark_config_file,
)
from .m2n_adapter import M2NPlanningError, plan_m2n_case
from .mooncake_adapter import MooncakePlanningError, plan_mooncake_case


class BenchmarkDriverError(ValueError):
    """Raised when a dry-run report cannot be constructed."""


def _mesh_summary(mesh: MeshSpec | None) -> dict[str, int] | None:
    if mesh is None:
        return None
    return {
        "replicas": mesh.replicas,
        "shards": mesh.shards,
        "shard_dim": mesh.shard_dim,
        "total_ranks": mesh.total_ranks,
    }


def _select_cases(
    config: BenchmarkConfig, case_ids: Sequence[str] | None
) -> tuple[BenchmarkCase, ...]:
    all_cases = (
        *config.physical_cases,
        *config.stress_cases,
        *config.planner_only_cases,
    )
    if not case_ids:
        return all_cases
    by_id = {case.id: case for case in all_cases}
    selected = []
    for case_id in case_ids:
        case = by_id.get(case_id)
        if case is None:
            raise BenchmarkDriverError(f"unknown case id: {case_id}")
        selected.append(case)
    return tuple(selected)


def _case_report(
    case: BenchmarkCase,
    config: BenchmarkConfig,
    *,
    m2n_binary: str,
    launcher: str,
    max_batch_operations: int,
    max_region_segments: int,
) -> dict[str, object]:
    report: dict[str, object] = {
        "id": case.id,
        "category": case.category,
        "reason": case.reason,
        "global_shape": (
            None if case.global_shape is None else list(case.global_shape)
        ),
        "logical_bytes": case.logical_bytes,
        "required_ranks": case.required_ranks,
        "world_size": case.world_size,
        "source": _mesh_summary(case.source),
        "target": _mesh_summary(case.target),
    }
    if case.source is None or case.target is None or case.global_shape is None:
        reason = case.reason or "case has no single-tensor geometry"
        report["m2n"] = {"status": "not_planned", "reason": reason}
        report["mooncake"] = {"status": "not_planned", "reason": reason}
        return report

    m2n = plan_m2n_case(
        case,
        config.placement,
        binary=m2n_binary,
        launcher=launcher,
        warmups=config.warmups,
        iterations=config.iterations,
    )
    mooncake = plan_mooncake_case(
        case,
        max_batch_operations=max_batch_operations,
        max_region_segments=max_region_segments,
    )
    report["m2n"] = {"status": "planned", **m2n.to_dict()}
    report["mooncake"] = {
        "status": "planned",
        "tensor_id": mooncake.tensor_id,
        "source_participant_count": mooncake.source_placement.topology.world_size,
        "target_participant_count": mooncake.target_placement.topology.world_size,
        "summary": asdict(mooncake.summary),
    }
    return report


def build_dry_run_report(
    config: BenchmarkConfig,
    *,
    case_ids: Sequence[str] | None = None,
    m2n_binary: str = "reshard_bench",
    launcher: str = "mpirun",
    max_batch_operations: int = 1024,
    max_region_segments: int = 1_000_000,
) -> dict[str, object]:
    """Build a deterministic report without inspecting execution state."""

    cases = _select_cases(config, case_ids)
    return {
        "schema_version": config.schema_version,
        "mode": "dry-run",
        "execution_enabled": config.execution_enabled,
        "execution_guard": config.execution_guard,
        "execution_authorized": False,
        "execution_refusal_reasons": ["dry-run never executes transfer backends"],
        "dtype": config.dtype,
        "warmups": config.warmups,
        "iterations": config.iterations,
        "physical_gpus": config.physical_gpus,
        "placement": asdict(config.placement),
        "metrics": list(config.metrics),
        "cases": [
            _case_report(
                case,
                config,
                m2n_binary=m2n_binary,
                launcher=launcher,
                max_batch_operations=max_batch_operations,
                max_region_segments=max_region_segments,
            )
            for case in cases
        ],
    }


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.heterogeneous_weight_reshard"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    dry_run = commands.add_parser(
        "dry-run", help="generate backend-neutral plans without execution"
    )
    dry_run.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("cases.json"),
    )
    dry_run.add_argument("--case", action="append", dest="case_ids")
    dry_run.add_argument("--m2n-binary", default="reshard_bench")
    dry_run.add_argument("--launcher", default="mpirun")
    dry_run.add_argument("--max-batch-operations", type=_positive_integer, default=1024)
    dry_run.add_argument(
        "--max-region-segments", type=_positive_integer, default=1_000_000
    )
    dry_run.add_argument("--json", action="store_true", dest="json_output")
    return parser


def _print_human(report: dict[str, object]) -> None:
    cases = report["cases"]
    assert isinstance(cases, list)
    print(f"mode=dry-run cases={len(cases)} execution_authorized=false")
    for case in cases:
        assert isinstance(case, dict)
        mooncake = case["mooncake"]
        assert isinstance(mooncake, dict)
        print(
            f"case={case['id']} category={case['category']} "
            f"mooncake={mooncake['status']}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = load_benchmark_config_file(args.config)
        report = build_dry_run_report(
            config,
            case_ids=args.case_ids,
            m2n_binary=args.m2n_binary,
            launcher=args.launcher,
            max_batch_operations=args.max_batch_operations,
            max_region_segments=args.max_region_segments,
        )
    except (
        BenchmarkDriverError,
        CaseSpecError,
        M2NPlanningError,
        MooncakePlanningError,
    ) as error:
        parser.error(str(error))

    if args.json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
