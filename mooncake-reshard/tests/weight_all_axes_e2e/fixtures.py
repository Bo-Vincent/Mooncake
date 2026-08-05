from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

from mooncake.reshard._compat import _strict_zip
from mooncake.reshard.weight import (
    WeightPlacementManifest,
    WeightRuntimeBindingManifest,
)
from weight_all_axes_e2e.models import (
    LogicalWeight,
    _READER_BUFFER_BYTES,
    _TENSOR_BYTES,
)
from weight_gpu_e2e.buffers import TransferBuffer


@dataclass(frozen=True)
class RuntimeInputs:
    """Test-only carrier for one global placement and runtime bindings."""

    placement: WeightPlacementManifest
    bindings: tuple[WeightRuntimeBindingManifest, ...]

    def __post_init__(self) -> None:
        expected = {part.participant_id for part in self.placement.parts}
        participant_ids = [binding.participant_id for binding in self.bindings]
        actual = set(participant_ids)
        if not actual.issubset(expected) or len(actual) != len(participant_ids):
            raise ValueError("runtime bindings do not belong to the global placement")

    def pairs(
        self,
    ) -> Iterator[tuple[WeightPlacementManifest, WeightRuntimeBindingManifest]]:
        return iter((self.placement, binding) for binding in self.bindings)

    def active_pairs(
        self,
    ) -> Iterator[tuple[WeightPlacementManifest, WeightRuntimeBindingManifest]]:
        active = {
            part.participant_id for part in self.placement.parts if part.fragments
        }
        return iter(
            (self.placement, binding)
            for binding in self.bindings
            if binding.participant_id in active
        )

    def single(
        self,
    ) -> tuple[WeightPlacementManifest, WeightRuntimeBindingManifest]:
        if len(self.bindings) != 1:
            raise ValueError("runtime inputs do not identify one executor")
        return self.placement, self.bindings[0]


@dataclass
class AllAxesFixture:
    sources: RuntimeInputs
    targets: RuntimeInputs
    source_buffers: dict[tuple[str, int, int], TransferBuffer]
    target_buffers: dict[tuple[str, int, int], TransferBuffer]
    weights: tuple[LogicalWeight, ...]
    source_tp: int
    target_tp: int

    def verify(self) -> None:
        source_extent = _TENSOR_BYTES // self.source_tp
        target_extent = _TENSOR_BYTES // self.target_tp
        patterns = {weight.tensor_id: weight.pattern for weight in self.weights}
        for (tensor_id, _dp_rank, tp_rank), buffer in self.target_buffers.items():
            expected = bytearray()
            target_begin = tp_rank * target_extent
            for offset in range(target_extent):
                source_rank = (target_begin + offset) // source_extent
                expected.append(patterns[tensor_id] + source_rank)
            assert buffer.read_range(0, buffer.size) == bytes(expected)


@dataclass
class PackedReaderFixture:
    source: RuntimeInputs
    targets: RuntimeInputs
    source_buffer: TransferBuffer
    target_buffers: tuple[TransferBuffer, ...]
    source_expected: bytes
    target_expected: tuple[bytes, ...]

    def verify(self) -> None:
        assert self.source_buffer.read_range(0, _READER_BUFFER_BYTES) == (
            self.source_expected
        )
        for buffer, expected in _strict_zip(self.target_buffers, self.target_expected):
            assert buffer.read_range(0, _READER_BUFFER_BYTES) == expected


@dataclass
class CrossDimReaderFixture:
    sources: RuntimeInputs
    targets: RuntimeInputs
    source_buffers: tuple[TransferBuffer, ...]
    target_buffers: tuple[TransferBuffer, ...]
    target_expected: tuple[bytes, ...]

    def verify(self) -> None:
        for buffer, expected in _strict_zip(self.target_buffers, self.target_expected):
            assert buffer.read_range(0, buffer.size) == expected


def _native_store_config_factory(group_ids: Sequence[str], record_type: str):
    from mooncake.store import ObjectDataType, ReplicateConfig

    config = ReplicateConfig()
    if hasattr(config, "group_ids"):
        config.group_ids = list(group_ids)
    config.with_hard_pin = True
    if record_type == "payload":
        config.data_type = ObjectDataType.WEIGHT
    elif record_type == "metadata":
        config.data_type = ObjectDataType.METADATA
    else:
        raise ValueError(f"invalid weight Store record type: {record_type}")
    return config
