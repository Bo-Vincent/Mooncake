"""Strict, backend-neutral benchmark case schema.

The schema is intentionally small. It describes logical tensor geometry and rank
placement only; neither Mooncake nor NCCL M2N execution details belong here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Mapping


class CaseSpecError(ValueError):
    """Raised when a benchmark configuration violates the schema."""


_SUPPORTED_SCHEMA_VERSION = 1
_DTYPE_ITEMSIZE = {"uint8": 1}
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class PlacementSpec:
    source_host: str
    target_host: str
    gpus_per_host: int


@dataclass(frozen=True)
class MeshSpec:
    replicas: int
    shards: int
    shard_dim: int

    @property
    def total_ranks(self) -> int:
        return self.replicas * self.shards


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    category: str
    source: MeshSpec | None = None
    target: MeshSpec | None = None
    global_shape: tuple[int, ...] | None = None
    required_ranks: int | None = None
    reason: str | None = None
    _dtype_itemsize: int = field(default=1, repr=False, compare=False)

    @property
    def logical_bytes(self) -> int | None:
        if self.global_shape is None:
            return None
        return reduce(mul, self.global_shape, self._dtype_itemsize)

    @property
    def source_total(self) -> int | None:
        return None if self.source is None else self.source.total_ranks

    @property
    def target_total(self) -> int | None:
        return None if self.target is None else self.target.total_ranks

    @property
    def world_size(self) -> int | None:
        if self.source_total is None or self.target_total is None:
            return None
        return self.source_total + self.target_total


@dataclass(frozen=True)
class BenchmarkConfig:
    schema_version: int
    execution_enabled: bool
    execution_guard: str
    dtype: str
    warmups: int
    iterations: int
    physical_gpus: int
    placement: PlacementSpec
    metrics: tuple[str, ...]
    physical_cases: tuple[BenchmarkCase, ...]
    stress_cases: tuple[BenchmarkCase, ...]
    planner_only_cases: tuple[BenchmarkCase, ...]

    @classmethod
    def from_dict(cls, value: object) -> "BenchmarkConfig":
        return load_benchmark_config(value)

    @classmethod
    def from_json(cls, value: str) -> "BenchmarkConfig":
        return load_benchmark_config_json(value)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "BenchmarkConfig":
        return load_benchmark_config_file(path)

    def execution_is_authorized(self, env: Mapping[str, object]) -> bool:
        return self.execution_enabled and env.get(self.execution_guard) == "1"


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CaseSpecError(f"{path}: expected object")
    if not all(isinstance(key, str) for key in value):
        raise CaseSpecError(f"{path}: expected string field names")
    return value


def _fields(
    value: Mapping[str, object],
    path: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = sorted(required - value.keys())
    if missing:
        raise CaseSpecError(f"{path}: missing fields: {', '.join(missing)}")
    unknown = sorted(value.keys() - required - optional)
    if unknown:
        raise CaseSpecError(f"{path}: unknown fields: {', '.join(unknown)}")


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise CaseSpecError(f"{path}: expected non-empty string")
    return value


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CaseSpecError(f"{path}: expected integer")
    return value


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CaseSpecError(f"{path}: expected positive integer")
    return value


def _non_negative_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CaseSpecError(f"{path}: expected non-negative integer")
    return value


def _placement(value: object) -> PlacementSpec:
    path = "config.placement"
    raw = _mapping(value, path)
    _fields(
        raw,
        path,
        required={"source_host", "target_host", "gpus_per_host"},
    )
    return PlacementSpec(
        source_host=_string(raw["source_host"], f"{path}.source_host"),
        target_host=_string(raw["target_host"], f"{path}.target_host"),
        gpus_per_host=_positive_integer(raw["gpus_per_host"], f"{path}.gpus_per_host"),
    )


def _mesh(value: object, path: str) -> MeshSpec:
    raw = _mapping(value, path)
    _fields(raw, path, required={"replicas", "shards", "shard_dim"})
    return MeshSpec(
        replicas=_positive_integer(raw["replicas"], f"{path}.replicas"),
        shards=_positive_integer(raw["shards"], f"{path}.shards"),
        shard_dim=_non_negative_integer(raw["shard_dim"], f"{path}.shard_dim"),
    )


def _shape(value: object, path: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise CaseSpecError(f"{path}: expected list")
    if len(value) not in (2, 3):
        raise CaseSpecError(f"{path}: expected rank 2 or 3, got {len(value)}")
    return tuple(
        _positive_integer(dimension, f"{path}[{index}]")
        for index, dimension in enumerate(value)
    )


def _validate_geometry(
    *,
    path: str,
    shape: tuple[int, ...],
    source: MeshSpec,
    target: MeshSpec,
    required_ranks: int,
) -> None:
    for side, mesh in (("source", source), ("target", target)):
        if mesh.shard_dim >= len(shape):
            raise CaseSpecError(
                f"{path}.{side}.shard_dim: must be in [0, {len(shape)})"
            )
        dimension = shape[mesh.shard_dim]
        if dimension % mesh.shards:
            raise CaseSpecError(
                f"{path}.global_shape[{mesh.shard_dim}] ({dimension}) must be "
                f"divisible by {side}.shards ({mesh.shards})"
            )

    expected_ranks = source.total_ranks + target.total_ranks
    if required_ranks != expected_ranks:
        raise CaseSpecError(
            f"{path}.required_ranks: expected {expected_ranks} from "
            f"source_total + target_total, got {required_ranks}"
        )


def _geometry_case(
    raw: Mapping[str, object],
    *,
    path: str,
    category: str,
    dtype_itemsize: int,
    reason: str | None = None,
) -> BenchmarkCase:
    source = _mesh(raw["source"], f"{path}.source")
    target = _mesh(raw["target"], f"{path}.target")
    shape = _shape(raw["global_shape"], f"{path}.global_shape")
    required_ranks = _positive_integer(raw["required_ranks"], f"{path}.required_ranks")
    _validate_geometry(
        path=path,
        shape=shape,
        source=source,
        target=target,
        required_ranks=required_ranks,
    )
    return BenchmarkCase(
        id=_string(raw["id"], f"{path}.id"),
        category=category,
        source=source,
        target=target,
        global_shape=shape,
        required_ranks=required_ranks,
        reason=reason,
        _dtype_itemsize=dtype_itemsize,
    )


def _required_geometry_case(
    value: object, *, path: str, category: str, dtype_itemsize: int
) -> BenchmarkCase:
    raw = _mapping(value, path)
    _fields(
        raw,
        path,
        required={"id", "source", "target", "global_shape", "required_ranks"},
    )
    return _geometry_case(
        raw,
        path=path,
        category=category,
        dtype_itemsize=dtype_itemsize,
    )


def _planner_only_case(
    value: object, *, path: str, dtype_itemsize: int
) -> BenchmarkCase:
    raw = _mapping(value, path)
    geometry_fields = {"source", "target", "global_shape", "required_ranks"}
    _fields(
        raw,
        path,
        required={"id", "reason"},
        optional=geometry_fields,
    )
    present_geometry = geometry_fields & raw.keys()
    if present_geometry and present_geometry != geometry_fields:
        raise CaseSpecError(
            f"{path}: geometry fields must be provided together: "
            "global_shape, required_ranks, source, target"
        )

    reason = _string(raw["reason"], f"{path}.reason")
    if present_geometry:
        return _geometry_case(
            raw,
            path=path,
            category="planner_only",
            dtype_itemsize=dtype_itemsize,
            reason=reason,
        )
    return BenchmarkCase(
        id=_string(raw["id"], f"{path}.id"),
        category="planner_only",
        reason=reason,
        _dtype_itemsize=dtype_itemsize,
    )


def _case_list(
    value: object,
    *,
    field_name: str,
    category: str,
    dtype_itemsize: int,
) -> tuple[BenchmarkCase, ...]:
    path = f"config.{field_name}"
    if not isinstance(value, list):
        raise CaseSpecError(f"{path}: expected list")
    cases = []
    for index, item in enumerate(value):
        item_path = f"{field_name}[{index}]"
        if category == "planner_only":
            case = _planner_only_case(
                item, path=item_path, dtype_itemsize=dtype_itemsize
            )
        else:
            case = _required_geometry_case(
                item,
                path=item_path,
                category=category,
                dtype_itemsize=dtype_itemsize,
            )
        cases.append(case)
    return tuple(cases)


def load_benchmark_config(value: object) -> BenchmarkConfig:
    """Validate and detach a benchmark configuration from its mutable input."""

    raw = _mapping(value, "config")
    required_fields = {
        "schema_version",
        "execution_enabled",
        "execution_guard",
        "dtype",
        "warmups",
        "iterations",
        "physical_gpus",
        "placement",
        "metrics",
        "physical_cases",
        "stress_cases",
        "planner_only_cases",
    }
    _fields(raw, "config", required=required_fields)

    schema_version = _integer(raw["schema_version"], "config.schema_version")
    if schema_version != _SUPPORTED_SCHEMA_VERSION:
        raise CaseSpecError(
            f"config.schema_version: unsupported version {schema_version}; "
            f"expected {_SUPPORTED_SCHEMA_VERSION}"
        )

    execution_enabled = raw["execution_enabled"]
    if not isinstance(execution_enabled, bool):
        raise CaseSpecError("config.execution_enabled: expected bool")

    execution_guard = raw["execution_guard"]
    if (
        not isinstance(execution_guard, str)
        or _ENVIRONMENT_NAME.fullmatch(execution_guard) is None
    ):
        raise CaseSpecError(
            "config.execution_guard: expected an environment variable name"
        )

    dtype = _string(raw["dtype"], "config.dtype")
    if dtype not in _DTYPE_ITEMSIZE:
        supported = ", ".join(sorted(_DTYPE_ITEMSIZE))
        raise CaseSpecError(
            f"config.dtype: unsupported dtype {dtype!r}; supported: {supported}"
        )
    dtype_itemsize = _DTYPE_ITEMSIZE[dtype]

    metrics_raw = raw["metrics"]
    if not isinstance(metrics_raw, list) or not metrics_raw:
        raise CaseSpecError("config.metrics: expected a non-empty list")
    metrics = tuple(
        _string(metric, f"config.metrics[{index}]")
        for index, metric in enumerate(metrics_raw)
    )

    placement = _placement(raw["placement"])
    physical_gpus = _positive_integer(raw["physical_gpus"], "config.physical_gpus")
    physical_cases = _case_list(
        raw["physical_cases"],
        field_name="physical_cases",
        category="physical",
        dtype_itemsize=dtype_itemsize,
    )
    stress_cases = _case_list(
        raw["stress_cases"],
        field_name="stress_cases",
        category="stress",
        dtype_itemsize=dtype_itemsize,
    )
    planner_only_cases = _case_list(
        raw["planner_only_cases"],
        field_name="planner_only_cases",
        category="planner_only",
        dtype_itemsize=dtype_itemsize,
    )

    for index, case in enumerate(physical_cases):
        case_path = f"physical_cases[{index}]"
        assert case.source is not None and case.target is not None
        assert case.world_size is not None
        for side, mesh in (("source", case.source), ("target", case.target)):
            if mesh.total_ranks > placement.gpus_per_host:
                raise CaseSpecError(
                    f"{case_path}.{side}: requires {mesh.total_ranks} ranks but "
                    f"config.placement.gpus_per_host is {placement.gpus_per_host}"
                )
        if case.world_size > physical_gpus:
            raise CaseSpecError(
                f"{case_path}.world_size: {case.world_size} exceeds "
                f"config.physical_gpus {physical_gpus}"
            )

    seen: dict[str, str] = {}
    for field_name, cases in (
        ("physical_cases", physical_cases),
        ("stress_cases", stress_cases),
        ("planner_only_cases", planner_only_cases),
    ):
        for index, case in enumerate(cases):
            path = f"{field_name}[{index}]"
            if case.id in seen:
                raise CaseSpecError(
                    f"config: duplicate case id {case.id!r} in {seen[case.id]} "
                    f"and {path}"
                )
            seen[case.id] = path

    return BenchmarkConfig(
        schema_version=schema_version,
        execution_enabled=execution_enabled,
        execution_guard=execution_guard,
        dtype=dtype,
        warmups=_positive_integer(raw["warmups"], "config.warmups"),
        iterations=_positive_integer(raw["iterations"], "config.iterations"),
        physical_gpus=physical_gpus,
        placement=placement,
        metrics=metrics,
        physical_cases=physical_cases,
        stress_cases=stress_cases,
        planner_only_cases=planner_only_cases,
    )


def load_benchmark_config_json(value: str) -> BenchmarkConfig:
    try:
        raw = json.loads(value)
    except (json.JSONDecodeError, TypeError) as error:
        raise CaseSpecError(f"config: invalid JSON: {error}") from error
    return load_benchmark_config(raw)


def load_benchmark_config_file(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path)
    try:
        contents = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise CaseSpecError(f"config: cannot read {config_path}: {error}") from error
    return load_benchmark_config_json(contents)
