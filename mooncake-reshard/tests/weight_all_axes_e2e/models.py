from __future__ import annotations

from dataclasses import dataclass

from mooncake.reshard.weight import (
    OwnershipAxis,
    ReplicatedAxis,
    SplitAxis,
    TensorDescriptor,
)


_TENSOR_BYTES = 64 * 1024
_READER_BUFFER_BYTES = 256
_READER_SOURCE_GUARD = 0x3C
_READER_TARGET_SENTINEL = 0xD7
_READER_VIEW_OFFSETS = (64, 128, 192)
_CROSS_DIM_SHAPE = (4, 4, 6)


@dataclass(frozen=True)
class LogicalWeight:
    tensor_id: str
    layer_id: int
    expert_id: int | None
    source_pp: int
    source_ep: int
    target_pp: int
    target_ep: int
    pattern: int

    def descriptor(self) -> TensorDescriptor:
        parallel_axes = [
            ReplicatedAxis("dp"),
            OwnershipAxis("pp"),
            OwnershipAxis("ep"),
        ]
        parallel_axes.append(SplitAxis("tp", dim=0))
        return TensorDescriptor(
            tensor_id=self.tensor_id,
            global_shape=(_TENSOR_BYTES,),
            dtype="uint8",
            itemsize=1,
            shard_dims=(0,),
            layer_id=self.layer_id,
            expert_id=self.expert_id,
            layout_fingerprint="all-axes:contiguous:uint8:v1",
            parallel_axes=tuple(parallel_axes),
        )


def _weights() -> tuple[LogicalWeight, ...]:
    return (
        LogicalWeight(
            tensor_id="layers.0.self_attn.q_proj.weight",
            layer_id=0,
            expert_id=None,
            source_pp=0,
            source_ep=0,
            target_pp=0,
            target_ep=0,
            pattern=10,
        ),
        LogicalWeight(
            tensor_id="layers.3.mlp.gate_proj.weight",
            layer_id=3,
            expert_id=None,
            source_pp=1,
            source_ep=0,
            target_pp=3,
            target_ep=0,
            pattern=30,
        ),
        LogicalWeight(
            tensor_id="layers.1.mlp.experts.0.gate_proj.weight",
            layer_id=1,
            expert_id=0,
            source_pp=0,
            source_ep=0,
            target_pp=1,
            target_ep=0,
            pattern=50,
        ),
        LogicalWeight(
            tensor_id="layers.2.mlp.experts.1.down_proj.weight",
            layer_id=2,
            expert_id=1,
            source_pp=1,
            source_ep=1,
            target_pp=2,
            target_ep=0,
            pattern=70,
        ),
    )
