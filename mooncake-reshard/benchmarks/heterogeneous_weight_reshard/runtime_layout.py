"""Build independent weight placements and runtime bindings for benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Protocol, Sequence

from mooncake.reshard._compat import _strict_zip
from mooncake.reshard.weight import (
    ParallelRank,
    ParallelTopology,
    PlacementFragment,
    RuntimeBindingFragment,
    TensorDescriptor,
    ReplicatedAxis,
    SplitAxis,
    TopologyParticipant,
    WeightPlacementManifest,
    WeightPlacementPart,
    WeightRuntimeBindingManifest,
    validate_runtime_bindings,
)
from mooncake.reshard.weight.te import MemoryRegistrationLease

from .case_spec import BenchmarkCase, MeshSpec


MODEL_ID = "benchmark_model"
TENSOR_ID = "benchmark_tensor"
LAYOUT_FINGERPRINT = "benchmark:logical-contiguous:v1"


class RuntimeBuffer(Protocol):
    pointer: int
    size: int


@dataclass(frozen=True)
class RuntimeTopology:
    """One complete placement plus per-participant runtime bindings."""

    placement: WeightPlacementManifest
    bindings: tuple[WeightRuntimeBindingManifest, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.placement, WeightPlacementManifest):
            raise ValueError("runtime topology must contain one complete placement")
        validate_runtime_bindings(self.placement, self.bindings)


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


def _contiguous_strides_bytes(shape: tuple[int, ...], itemsize: int) -> tuple[int, ...]:
    stride = itemsize
    strides = []
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return tuple(reversed(strides))


def build_runtime_topology(
    case: BenchmarkCase,
    *,
    side: str,
    buffers: Sequence[RuntimeBuffer],
    endpoint: str,
    revision: str,
    lease_generation: int = 1,
) -> RuntimeTopology:
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
        layout_fingerprint=LAYOUT_FINGERPRINT,
        parallel_axes=(
            ReplicatedAxis(kind="dp"),
            SplitAxis(kind="tp", dim=mesh.shard_dim),
        ),
        shard_dims=(mesh.shard_dim,),
    )

    participants = []
    fragments = []
    buffers_by_participant = []
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
            placement_fragment_id = f"{worker_id}-placement-fragment"
            rank = ParallelRank(dp=replica, tp=shard)
            participants.append(
                TopologyParticipant(participant_id=worker_id, rank=rank)
            )
            fragments.append(
                PlacementFragment(
                    placement_fragment_id=placement_fragment_id,
                    tensor_id=TENSOR_ID,
                    global_offset=tuple(global_offset),
                    local_shape=local_shape_tuple,
                    nbytes=buffer.size,
                    rank=rank,
                )
            )
            buffers_by_participant.append((worker_id, placement_fragment_id, buffer))

    topology = ParallelTopology(
        tp_size=mesh.shards,
        pp_size=1,
        ep_size=1,
        dp_size=mesh.replicas,
        participants=tuple(participants),
    )
    placement_set_id = f"{revision}:{side}:benchmark-placement"
    parts = tuple(
        WeightPlacementPart(
            resource_id=MODEL_ID,
            revision=revision,
            weight_generation=0,
            placement_set_id=placement_set_id,
            topology_id=topology.topology_id,
            participant_id=participant.participant_id,
            rank=participant.rank,
            tensors=(descriptor,),
            fragments=(fragment,),
        )
        for participant, fragment in _strict_zip(participants, fragments)
    )
    placement = WeightPlacementManifest(
        resource_id=MODEL_ID,
        revision=revision,
        weight_generation=0,
        placement_set_id=placement_set_id,
        topology=topology,
        parts=parts,
    )

    bindings = []
    for worker_id, placement_fragment_id, buffer in buffers_by_participant:
        bindings.append(
            WeightRuntimeBindingManifest(
                resource_id=MODEL_ID,
                revision=revision,
                placement_id=placement.placement_id,
                placement_digest=placement.digest,
                instance_id=worker_id,
                participant_id=worker_id,
                generation=lease_generation,
                lease_id=f"{revision}:{worker_id}:lease",
                fragments=(
                    RuntimeBindingFragment(
                        placement_fragment_id=placement_fragment_id,
                        fragment_id=f"{worker_id}-runtime-fragment",
                        address=buffer.pointer,
                        nbytes=buffer.size,
                        worker_id=worker_id,
                        endpoint=endpoint,
                        device=f"cuda:{getattr(buffer, 'device', 0)}",
                        itemsize=itemsize,
                        local_shape=local_shape_tuple,
                        strides_bytes=_contiguous_strides_bytes(
                            local_shape_tuple, itemsize
                        ),
                        storage_address=buffer.pointer,
                        storage_nbytes=buffer.size,
                        storage_offset_bytes=0,
                        owner=buffer,
                    ),
                ),
            )
        )
    return RuntimeTopology(placement, tuple(bindings))


def registration_leases(
    bindings: Sequence[WeightRuntimeBindingManifest],
) -> tuple[MemoryRegistrationLease, ...]:
    leases = []
    for binding in bindings:
        for fragment in binding.fragments:
            leases.append(
                MemoryRegistrationLease.from_fragment(
                    fragment,
                    lease_generation=binding.generation,
                    runtime_lease_id=binding.lease_id,
                )
            )
    return tuple(leases)
