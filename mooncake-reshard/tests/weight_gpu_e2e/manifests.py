from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from mooncake.reshard.weight import (
    ParallelRank,
    PlacementFragment,
    TensorDescriptor,
    SplitAxis,
    WeightPlacementManifest,
    WeightRuntimeBindingManifest,
    validate_runtime_bindings,
)
from global_placement_helpers import global_placement, runtime_fragment

from .buffers import TransferBuffer


@dataclass(frozen=True)
class RuntimeInputs:
    """Benchmark-local carrier for one global placement and its bindings."""

    placement: WeightPlacementManifest
    bindings: tuple[WeightRuntimeBindingManifest, ...]

    def __post_init__(self) -> None:
        validate_runtime_bindings(self.placement, self.bindings)

    def pairs(
        self,
    ) -> Iterator[tuple[WeightPlacementManifest, WeightRuntimeBindingManifest]]:
        return iter((self.placement, binding) for binding in self.bindings)

    def single(
        self,
    ) -> tuple[WeightPlacementManifest, WeightRuntimeBindingManifest]:
        if len(self.bindings) != 1:
            raise ValueError("runtime inputs do not identify one executor")
        return self.placement, self.bindings[0]


def _expected_tp_shard(
    *,
    total_bytes: int,
    source_tp: int,
    target_tp: int,
    target_rank: int,
) -> bytes:
    return b"".join(
        bytes([value]) * nbytes
        for _, nbytes, value in _expected_tp_segments(
            total_bytes=total_bytes,
            source_tp=source_tp,
            target_tp=target_tp,
            target_rank=target_rank,
        )
    )


def _expected_tp_segments(
    *,
    total_bytes: int,
    source_tp: int,
    target_tp: int,
    target_rank: int,
) -> tuple[tuple[int, int, int], ...]:
    if (
        total_bytes <= 0
        or source_tp <= 0
        or source_tp > 255
        or target_tp <= 0
        or not 0 <= target_rank < target_tp
        or total_bytes % source_tp
        or total_bytes % target_tp
    ):
        raise ValueError("invalid TP shard geometry")
    source_extent = total_bytes // source_tp
    target_extent = total_bytes // target_tp
    target_begin = target_rank * target_extent
    target_end = target_begin + target_extent
    result = []
    target_offset = 0
    for source_rank in range(source_tp):
        source_begin = source_rank * source_extent
        source_end = source_begin + source_extent
        overlap = max(
            0,
            min(source_end, target_end) - max(source_begin, target_begin),
        )
        if overlap:
            result.append((target_offset, overlap, source_rank + 1))
            target_offset += overlap
    if target_offset != target_extent:
        raise AssertionError("TP shard expectation is incomplete")
    return tuple(result)


def _verify_tp_buffers(
    buffers: list[TransferBuffer],
    *,
    total_bytes: int,
    source_tp: int,
    target_tp: int,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> None:
    if len(buffers) != target_tp or chunk_bytes <= 0:
        raise ValueError("target buffer geometry is invalid")
    expected_size = total_bytes // target_tp
    for rank, buffer in enumerate(buffers):
        if buffer.size != expected_size:
            raise ValueError(f"target TP rank {rank} has an invalid buffer size")
        for offset, nbytes, value in _expected_tp_segments(
            total_bytes=total_bytes,
            source_tp=source_tp,
            target_tp=target_tp,
            target_rank=rank,
        ):
            for chunk_offset in range(0, nbytes, chunk_bytes):
                current_size = min(chunk_bytes, nbytes - chunk_offset)
                actual = buffer.read_range(offset + chunk_offset, current_size)
                if actual != bytes([value]) * current_size:
                    raise AssertionError(
                        f"target TP rank {rank} differs at byte {offset + chunk_offset}"
                    )


def _tensor(total_bytes: int) -> TensorDescriptor:
    return TensorDescriptor(
        tensor_id="layers.0.mlp.gate_proj.weight",
        global_shape=(total_bytes,),
        dtype="uint8",
        itemsize=1,
        shard_dims=(0,),
        layer_id=0,
        layout_fingerprint="e2e:contiguous:uint8:v1",
        parallel_axes=(SplitAxis("tp", dim=0),),
    )


def _rank_manifests(
    *,
    tensor: TensorDescriptor,
    revision: str,
    prefix: str,
    buffers: list[TransferBuffer],
    endpoint: str | None = None,
) -> RuntimeInputs:
    if not buffers:
        raise ValueError("rank buffers must not be empty")
    extent = tensor.global_shape[0] // len(buffers)
    fragments = tuple(
        PlacementFragment(
            tensor_id=tensor.tensor_id,
            global_offset=(rank * extent,),
            local_shape=(extent,),
            nbytes=buffer.size,
            rank=ParallelRank(tp=rank),
        )
        for rank, buffer in enumerate(buffers)
    )
    placement = global_placement(
        resource_id="weight-store-gpu-e2e",
        revision=revision,
        placement_set_id=f"{prefix}-placement",
        tensors=(tensor,),
        fragments=fragments,
    )
    buffer_by_rank = {
        ParallelRank(tp=rank): buffer for rank, buffer in enumerate(buffers)
    }
    bindings = tuple(
        WeightRuntimeBindingManifest(
            resource_id=placement.resource_id,
            revision=placement.revision,
            placement_id=placement.placement_id,
            placement_digest=placement.digest,
            participant_id=part.participant_id,
            instance_id=f"{prefix}-tp{part.rank.tp}",
            generation=1,
            lease_id=f"{prefix}-tp{part.rank.tp}-lease",
            fragments=tuple(
                runtime_fragment(
                    placement=fragment,
                    tensor=tensor,
                    fragment_id=f"{prefix}-tp{part.rank.tp}-weight",
                    address=buffer_by_rank[part.rank].pointer,
                    worker_id=f"{prefix}-tp{part.rank.tp}",
                    endpoint=endpoint or f"{prefix}-tp{part.rank.tp}:12345",
                    owner=buffer_by_rank[part.rank],
                )
                for fragment in part.fragments
            ),
        )
        for part in placement.parts
    )
    return RuntimeInputs(placement, bindings)
