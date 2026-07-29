"""Build live runtime manifests from explicit benchmark mesh geometry."""

from __future__ import annotations

from math import prod
from typing import Protocol, Sequence

from mooncake.model_weight import (
    MemoryRegistrationLease,
    ParallelRank,
    RuntimeFragment,
    RuntimeManifest,
    TensorDescriptor,
)

from .case_spec import BenchmarkCase, MeshSpec


MODEL_ID = "benchmark_model"
TENSOR_ID = "benchmark_tensor"
LAYOUT_FINGERPRINT = "benchmark:logical-contiguous:v1"


class RuntimeBuffer(Protocol):
    pointer: int
    size: int


def _geometry(case: BenchmarkCase, side: str) -> tuple[MeshSpec, tuple[int, ...], int]:
    if side not in ("source", "target"):
        raise ValueError("side must be 'source' or 'target'")
    mesh = case.source if side == "source" else case.target
    if mesh is None or case.global_shape is None or case.logical_bytes is None:
        raise ValueError(f"{case.id}: {side} runtime geometry is unavailable")
    elements = prod(case.global_shape)
    if elements <= 0 or case.logical_bytes % elements:
        raise ValueError(f"{case.id}: logical bytes do not match shape")
    return mesh, case.global_shape, case.logical_bytes // elements


def build_runtime_manifests(
    case: BenchmarkCase,
    *,
    side: str,
    buffers: Sequence[RuntimeBuffer],
    endpoint: str,
    revision: str,
    lease_generation: int = 1,
) -> tuple[RuntimeManifest, ...]:
    mesh, global_shape, itemsize = _geometry(case, side)
    if len(buffers) != mesh.total_ranks:
        raise ValueError(
            f"{side} requires {mesh.total_ranks} buffers, got {len(buffers)}"
        )
    shard_extent = global_shape[mesh.shard_dim] // mesh.shards
    local_shape = list(global_shape)
    local_shape[mesh.shard_dim] = shard_extent
    local_shape_tuple = tuple(local_shape)
    expected_nbytes = prod(local_shape_tuple) * itemsize
    descriptor = TensorDescriptor(
        tensor_id=TENSOR_ID,
        global_shape=global_shape,
        dtype="uint8" if itemsize == 1 else f"opaque{itemsize * 8}",
        itemsize=itemsize,
        partition_dim=None,
        layout_fingerprint=LAYOUT_FINGERPRINT,
        shard_dims=(mesh.shard_dim,),
    )

    manifests = []
    for replica in range(mesh.replicas):
        for shard in range(mesh.shards):
            index = replica * mesh.shards + shard
            buffer = buffers[index]
            if type(buffer.pointer) is not int or buffer.pointer <= 0:
                raise ValueError(f"{side} buffer {index} has an invalid address")
            if type(buffer.size) is not int or buffer.size != expected_nbytes:
                raise ValueError(
                    f"{side} buffer {index} must contain {expected_nbytes} bytes"
                )
            global_offset = [0] * len(global_shape)
            global_offset[mesh.shard_dim] = shard * shard_extent
            worker_id = f"{side}-d{replica}-t{shard}"
            fragment = RuntimeFragment(
                fragment_id=f"{worker_id}-fragment",
                tensor_id=TENSOR_ID,
                global_offset=tuple(global_offset),
                local_shape=local_shape_tuple,
                address=buffer.pointer,
                nbytes=buffer.size,
                worker_id=worker_id,
                endpoint=endpoint,
                device="cuda:0",
                rank=ParallelRank(dp=replica, tp=shard),
                lease_generation=lease_generation,
                owner=buffer,
            )
            manifests.append(
                RuntimeManifest(
                    model_id=MODEL_ID,
                    revision=revision,
                    instance_id=worker_id,
                    tensors=(descriptor,),
                    fragments=(fragment,),
                    lease_id=f"{revision}:{worker_id}:lease",
                )
            )
    return tuple(manifests)


def registration_leases(
    manifests: Sequence[RuntimeManifest],
) -> tuple[MemoryRegistrationLease, ...]:
    leases = []
    for manifest in manifests:
        for fragment in manifest.fragments:
            leases.append(
                MemoryRegistrationLease.from_fragment(
                    fragment,
                    runtime_lease_id=manifest.lease_id,
                )
            )
    return tuple(leases)
