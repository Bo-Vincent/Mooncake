"""Pure control-plane adapter for NVIDIA NCCL M2N reshard_bench."""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

from .case_spec import BenchmarkCase, MeshSpec, PlacementSpec


class M2NPlanningError(ValueError):
    """Raised when a case cannot be represented by the M2N benchmark."""


@dataclass(frozen=True)
class M2NMeshSummary:
    dims: tuple[int, int]
    start_rank: int
    placement: tuple[str, str]

    @property
    def total_ranks(self) -> int:
        return self.dims[0] * self.dims[1]

    def to_dict(self) -> dict[str, object]:
        return {
            "dims": list(self.dims),
            "start_rank": self.start_rank,
            "placement": list(self.placement),
            "total_ranks": self.total_ranks,
        }


@dataclass(frozen=True)
class M2NAppContext:
    role: str
    host: str
    rank_begin: int
    rank_end: int
    argv: tuple[str, ...]

    @property
    def rank_count(self) -> int:
        return self.rank_end - self.rank_begin

    @property
    def rank_ids(self) -> tuple[int, ...]:
        return tuple(range(self.rank_begin, self.rank_end))

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "host": self.host,
            "rank_begin": self.rank_begin,
            "rank_end": self.rank_end,
            "rank_count": self.rank_count,
            "rank_ids": list(self.rank_ids),
            "argv": list(self.argv),
        }


@dataclass(frozen=True)
class M2NDryRunPlan:
    case_id: str
    binary: str
    launcher: str
    global_shape: tuple[int, ...]
    source_mesh: M2NMeshSummary
    target_mesh: M2NMeshSummary
    physical_execution_eligible: bool
    physical_execution_refusal_reasons: tuple[str, ...]
    descriptor_argv: tuple[str, ...]
    app_contexts: tuple[M2NAppContext, M2NAppContext]
    mpirun_argv: tuple[str, ...]

    @property
    def world_size(self) -> int:
        return self.source_mesh.total_ranks + self.target_mesh.total_ranks

    @property
    def rendered_command(self) -> str:
        return shlex.join(self.mpirun_argv)

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "binary": self.binary,
            "launcher": self.launcher,
            "global_shape": list(self.global_shape),
            "world_size": self.world_size,
            "source_mesh": self.source_mesh.to_dict(),
            "target_mesh": self.target_mesh.to_dict(),
            "physical_execution_eligible": self.physical_execution_eligible,
            "physical_execution_refusal_reasons": list(
                self.physical_execution_refusal_reasons
            ),
            "descriptor_argv": list(self.descriptor_argv),
            "app_contexts": [context.to_dict() for context in self.app_contexts],
            "mpirun_argv": list(self.mpirun_argv),
            "rendered_command": self.rendered_command,
        }


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise M2NPlanningError(f"{name} must be a non-empty string")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise M2NPlanningError(f"{name} must be a positive integer")
    return value


def _non_negative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise M2NPlanningError(f"{name} must be a non-negative integer")
    return value


def _geometry(
    case: BenchmarkCase,
) -> tuple[MeshSpec, MeshSpec, tuple[int, ...]]:
    if case.source is None or case.target is None or case.global_shape is None:
        raise M2NPlanningError(f"{case.id}: M2N planning requires geometry")
    source = case.source
    target = case.target
    shape = case.global_shape
    if len(shape) not in (2, 3):
        raise M2NPlanningError(f"{case.id}: M2N supports only rank-2 or rank-3 tensors")
    if any(type(extent) is not int or extent <= 0 for extent in shape):
        raise M2NPlanningError(f"{case.id}: global_shape must be positive")
    for side, mesh in (("source", source), ("target", target)):
        if (
            type(mesh.replicas) is not int
            or mesh.replicas <= 0
            or type(mesh.shards) is not int
            or mesh.shards <= 0
        ):
            raise M2NPlanningError(f"{case.id}: {side} mesh is invalid")
        if (
            type(mesh.shard_dim) is not int
            or mesh.shard_dim < 0
            or mesh.shard_dim >= len(shape)
        ):
            raise M2NPlanningError(f"{case.id}: {side} shard dimension is invalid")
        if shape[mesh.shard_dim] % mesh.shards:
            raise M2NPlanningError(
                f"{case.id}: {side} shard dimension is not divisible"
            )
    expected_ranks = source.total_ranks + target.total_ranks
    if case.required_ranks != expected_ranks:
        raise M2NPlanningError(f"{case.id}: required_ranks must equal {expected_ranks}")
    logical_bytes = case.logical_bytes
    if logical_bytes is None or logical_bytes != prod(shape):
        raise M2NPlanningError(
            f"{case.id}: reshard_bench currently requires uint8 geometry"
        )
    return source, target, shape


def _mesh(mesh: MeshSpec, start_rank: int) -> M2NMeshSummary:
    return M2NMeshSummary(
        dims=(mesh.replicas, mesh.shards),
        start_rank=start_rank,
        placement=("replicate", f"shard({mesh.shard_dim})"),
    )


def _cuda_visible_devices(start_rank: int, device_count: int) -> str:
    shift = start_rank % device_count
    return ",".join(
        str((visible_index - shift) % device_count)
        for visible_index in range(device_count)
    )


def plan_m2n_case(
    case: BenchmarkCase,
    placement: PlacementSpec,
    *,
    binary: str,
    warmups: int,
    iterations: int,
    launcher: str = "mpirun",
    algorithm: str = "ring",
    lb_mode: str = "uniform",
    validate: bool = True,
    mpi_interface: str | None = None,
    export_env_names: Sequence[str] = (),
) -> M2NDryRunPlan:
    """Map one immutable case to structured OpenMPI MPMD arguments."""

    binary = _nonempty_string(binary, "binary")
    launcher = _nonempty_string(launcher, "launcher")
    warmups = _non_negative_integer(warmups, "warmups")
    iterations = _positive_integer(iterations, "iterations")
    if type(validate) is not bool:
        raise M2NPlanningError("validate must be a boolean")
    if algorithm not in ("ring", "direct"):
        raise M2NPlanningError("algorithm must be 'ring' or 'direct'")
    if lb_mode not in ("uniform", "node"):
        raise M2NPlanningError("lb_mode must be 'uniform' or 'node'")
    if mpi_interface is not None:
        mpi_interface = _nonempty_string(mpi_interface, "mpi_interface")
    export_names = tuple(
        dict.fromkeys(
            (
                "PATH",
                "LD_LIBRARY_PATH",
                *(
                    _nonempty_string(name, "export_env_names item")
                    for name in export_env_names
                ),
            )
        )
    )
    source_host = _nonempty_string(placement.source_host, "source_host")
    target_host = _nonempty_string(placement.target_host, "target_host")
    gpus_per_host = _positive_integer(placement.gpus_per_host, "gpus_per_host")

    source, target, shape = _geometry(case)
    physical_refusal_reasons = tuple(
        f"{side} requires {mesh.total_ranks} ranks but gpus_per_host is {gpus_per_host}"
        for side, mesh in (("source", source), ("target", target))
        if mesh.total_ranks > gpus_per_host
    )
    if case.category == "physical" and physical_refusal_reasons:
        raise M2NPlanningError(f"{case.id}: {'; '.join(physical_refusal_reasons)}")
    max_sources = 16 if algorithm == "ring" else 32
    if source.total_ranks > max_sources or target.total_ranks > 64:
        raise M2NPlanningError(
            f"{case.id}: mesh exceeds NCCL M2N {algorithm} static limits"
        )

    source_mesh = _mesh(source, 0)
    target_mesh = _mesh(target, source.total_ranks)
    descriptor_argv = (
        "--src-mesh-dims",
        f"{source.replicas},{source.shards}",
        "--dst-mesh-dims",
        f"{target.replicas},{target.shards}",
        "--tensor-dims",
        ",".join(str(extent) for extent in shape),
        "--src-shard-dim",
        str(source.shard_dim),
        "--dst-shard-dim",
        str(target.shard_dim),
        "--iterations",
        str(iterations),
        "--warmup",
        str(warmups),
        "--algorithm",
        algorithm,
        "--lb-mode",
        lb_mode,
        *(("--validate",) if validate else ()),
    )
    source_argv = (binary, *descriptor_argv)
    target_argv = (binary, *descriptor_argv)
    source_context = M2NAppContext(
        role="source",
        host=source_host,
        rank_begin=0,
        rank_end=source.total_ranks,
        argv=source_argv,
    )
    target_context = M2NAppContext(
        role="target",
        host=target_host,
        rank_begin=source.total_ranks,
        rank_end=source.total_ranks + target.total_ranks,
        argv=target_argv,
    )
    network_argv = (
        ()
        if mpi_interface is None
        else (
            "--mca",
            "oob_tcp_if_include",
            mpi_interface,
            "--mca",
            "btl_tcp_if_include",
            mpi_interface,
        )
    )
    export_argv = tuple(item for name in export_names for item in ("-x", name))
    context_argv = (*network_argv, *export_argv)
    source_device_argv = (
        "-x",
        "CUDA_VISIBLE_DEVICES="
        + _cuda_visible_devices(source_context.rank_begin, gpus_per_host),
    )
    target_device_argv = (
        "-x",
        "CUDA_VISIBLE_DEVICES="
        + _cuda_visible_devices(target_context.rank_begin, gpus_per_host),
    )
    mpirun_argv = (
        launcher,
        *context_argv,
        *source_device_argv,
        "-np",
        str(source_context.rank_count),
        "--host",
        f"{source_host}:{source_context.rank_count}",
        *source_argv,
        ":",
        *context_argv,
        *target_device_argv,
        "-np",
        str(target_context.rank_count),
        "--host",
        f"{target_host}:{target_context.rank_count}",
        *target_argv,
    )
    return M2NDryRunPlan(
        case_id=case.id,
        binary=binary,
        launcher=launcher,
        global_shape=shape,
        source_mesh=source_mesh,
        target_mesh=target_mesh,
        physical_execution_eligible=not physical_refusal_reasons,
        physical_execution_refusal_reasons=physical_refusal_reasons,
        descriptor_argv=descriptor_argv,
        app_contexts=(source_context, target_context),
        mpirun_argv=mpirun_argv,
    )
