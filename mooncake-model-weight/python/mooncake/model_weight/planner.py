from __future__ import annotations

import heapq
from dataclasses import dataclass, replace
from math import prod
from typing import Iterable, Sequence

from .manifest import (
    ParallelRank,
    PlacementManifest,
    PlacementFragment,
    RuntimeBindingManifest,
    RuntimeFragment,
    RuntimeManifest,
    StoredFragment,
    TensorDescriptor,
    WeightManifest,
    bind_runtime_manifest,
    placement_manifest_from_runtime_manifest,
    runtime_binding_from_runtime_manifest,
)
from .binding import _logical_fragment_id
from .types import _canonical_tensor_descriptor


SourceFragment = RuntimeFragment | StoredFragment | PlacementFragment
TargetFragment = RuntimeFragment | PlacementFragment
RuntimeTensorOwner = tuple[int, int | None]
RuntimeBindingInput = RuntimeBindingManifest | RuntimeManifest


@dataclass(frozen=True)
class RuntimeLeaseSnapshot:
    fragment_id: str
    tensor_id: str
    global_offset: tuple[int, ...]
    local_shape: tuple[int, ...]
    address: int
    nbytes: int
    worker_id: str
    endpoint: str
    lease_generation: int

    @classmethod
    def from_fragment(cls, fragment: RuntimeFragment) -> RuntimeLeaseSnapshot:
        return cls(
            fragment_id=fragment.fragment_id,
            tensor_id=fragment.tensor_id,
            global_offset=fragment.global_offset,
            local_shape=fragment.local_shape,
            address=fragment.address,
            nbytes=fragment.nbytes,
            worker_id=fragment.worker_id,
            endpoint=fragment.endpoint,
            lease_generation=fragment.lease_generation,
        )


@dataclass(frozen=True)
class ExecutorTransferPlan:
    instance_id: str
    runtime_lease_id: str | None
    worker_id: str
    rank: ParallelRank
    fragment_ids: tuple[str, ...]
    fragment_leases: tuple[RuntimeLeaseSnapshot, ...]
    operation_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fragment_ids", tuple(self.fragment_ids))
        object.__setattr__(self, "fragment_leases", tuple(self.fragment_leases))
        object.__setattr__(self, "operation_indices", tuple(self.operation_indices))
        if not self.instance_id or not self.worker_id or not self.fragment_ids:
            raise ValueError("executor plan identifiers must not be empty")
        if self.runtime_lease_id is not None and (
            type(self.runtime_lease_id) is not str or not self.runtime_lease_id
        ):
            raise ValueError("executor plan runtime lease ID must be non-empty")
        if len(self.fragment_ids) != len(set(self.fragment_ids)):
            raise ValueError("executor plan has duplicate fragment IDs")
        if not all(
            isinstance(lease, RuntimeLeaseSnapshot) for lease in self.fragment_leases
        ):
            raise ValueError("executor plan has invalid runtime lease metadata")
        if (
            tuple(lease.fragment_id for lease in self.fragment_leases)
            != self.fragment_ids
        ):
            raise ValueError("executor plan fragment lease IDs do not match")
        if any(type(index) is not int or index < 0 for index in self.operation_indices):
            raise ValueError("executor operation indices must be non-negative integers")


@dataclass(frozen=True)
class PipelineRouteGroup:
    source_pp: int | None
    target_pp: int
    operation_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_indices", tuple(self.operation_indices))
        if self.source_pp is not None and (
            type(self.source_pp) is not int or self.source_pp < 0
        ):
            raise ValueError("pipeline route source_pp must be non-negative")
        if type(self.target_pp) is not int or self.target_pp < 0:
            raise ValueError("pipeline route target_pp must be non-negative")
        if any(type(index) is not int or index < 0 for index in self.operation_indices):
            raise ValueError("pipeline route indices must be non-negative integers")
        if len(self.operation_indices) != len(set(self.operation_indices)):
            raise ValueError("pipeline route has duplicate operation indices")


@dataclass(frozen=True)
class CopyRange:
    tensor_id: str
    source: SourceFragment
    target: TargetFragment
    source_offset: int
    target_offset: int
    nbytes: int
    repeat: int = 1
    source_stride: int = 0
    target_stride: int = 0

    def __post_init__(self) -> None:
        if not self.tensor_id:
            raise ValueError("copy range tensor_id must not be empty")
        if (
            self.source.tensor_id != self.tensor_id
            or self.target.tensor_id != self.tensor_id
        ):
            raise ValueError("copy range tensor mismatch")
        for name in (
            "source_offset",
            "target_offset",
            "nbytes",
            "repeat",
            "source_stride",
            "target_stride",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise ValueError(f"copy range {name} must be an integer")
        if (
            min(
                self.source_offset,
                self.target_offset,
                self.source_stride,
                self.target_stride,
            )
            < 0
        ):
            raise ValueError("copy range values must be non-negative")
        if self.nbytes <= 0 or self.repeat <= 0:
            raise ValueError("copy range size and repeat must be positive")
        self.validate_bounds()

    def validate_bounds(self) -> None:
        source_end = (
            self.source_offset + (self.repeat - 1) * self.source_stride + self.nbytes
        )
        if source_end > self.source.nbytes:
            raise ValueError("copy range exceeds source fragment")
        target_end = (
            self.target_offset + (self.repeat - 1) * self.target_stride + self.nbytes
        )
        if target_end > self.target.nbytes:
            raise ValueError("copy range exceeds target fragment")

    @property
    def total_bytes(self) -> int:
        return self.nbytes * self.repeat

    def iter_segments(self) -> Iterable[tuple[int, int, int]]:
        for index in range(self.repeat):
            yield (
                self.source_offset + index * self.source_stride,
                self.target_offset + index * self.target_stride,
                self.nbytes,
            )


@dataclass(frozen=True)
class TransferRegion:
    tensor_id: str
    source: SourceFragment
    target: TargetFragment
    overlap_offset: tuple[int, ...]
    overlap_shape: tuple[int, ...]
    source_base_offset: int
    target_base_offset: int
    inner_bytes: int
    outer_loop_counts: tuple[int, ...]
    source_strides: tuple[int, ...]
    target_strides: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
            "overlap_offset",
            "overlap_shape",
            "outer_loop_counts",
            "source_strides",
            "target_strides",
        ):
            value = getattr(self, name)
            if isinstance(value, (str, bytes, bytearray)):
                raise ValueError(f"transfer region {name} must contain integers")
            try:
                normalized = tuple(value)
            except TypeError as error:
                raise ValueError(
                    f"transfer region {name} must contain integers"
                ) from error
            if any(type(item) is not int for item in normalized):
                raise ValueError(f"transfer region {name} must contain integers")
            object.__setattr__(self, name, normalized)

        if not self.tensor_id:
            raise ValueError("transfer region tensor_id must not be empty")
        if (
            self.source.tensor_id != self.tensor_id
            or self.target.tensor_id != self.tensor_id
        ):
            raise ValueError("transfer region tensor mismatch")
        ndim = len(self.overlap_offset)
        if (
            ndim == 0
            or len(self.overlap_shape) != ndim
            or len(self.source.global_offset) != ndim
            or len(self.target.global_offset) != ndim
        ):
            raise ValueError("transfer region logical rank mismatch")
        if any(offset < 0 for offset in self.overlap_offset) or any(
            extent <= 0 for extent in self.overlap_shape
        ):
            raise ValueError("transfer region logical box is invalid")
        if not _box_contains(
            self.source.global_offset,
            self.source.local_shape,
            self.overlap_offset,
            self.overlap_shape,
        ):
            raise ValueError("transfer region exceeds source logical fragment")
        if not _box_contains(
            self.target.global_offset,
            self.target.local_shape,
            self.overlap_offset,
            self.overlap_shape,
        ):
            raise ValueError("transfer region exceeds target logical fragment")

        for name in ("source_base_offset", "target_base_offset", "inner_bytes"):
            value = getattr(self, name)
            if type(value) is not int:
                raise ValueError(f"transfer region {name} must be an integer")
        if self.source_base_offset < 0 or self.target_base_offset < 0:
            raise ValueError("transfer region base offsets must be non-negative")
        if self.inner_bytes <= 0:
            raise ValueError("transfer region inner_bytes must be positive")
        if not (
            len(self.outer_loop_counts)
            == len(self.source_strides)
            == len(self.target_strides)
        ):
            raise ValueError("transfer region outer loop rank mismatch")
        if any(count <= 0 for count in self.outer_loop_counts):
            raise ValueError("transfer region outer loop counts must be positive")
        if any(stride < 0 for stride in self.source_strides) or any(
            stride < 0 for stride in self.target_strides
        ):
            raise ValueError("transfer region strides must be non-negative")

        source_itemsize = _fragment_itemsize(self.source)
        target_itemsize = _fragment_itemsize(self.target)
        if source_itemsize != target_itemsize:
            raise ValueError("transfer region source and target itemsize differ")
        (
            expected_source_offset,
            expected_target_offset,
            expected_inner_bytes,
            expected_outer_loop_counts,
            expected_source_strides,
            expected_target_strides,
        ) = _derive_region_geometry(
            self.source,
            self.target,
            self.overlap_offset,
            self.overlap_shape,
        )
        expected_bytes = prod(self.overlap_shape) * source_itemsize
        if self.total_bytes != expected_bytes:
            raise ValueError("transfer region loop geometry does not match overlap")
        if self.source_base_offset != expected_source_offset:
            raise ValueError("transfer region source base offset is inconsistent")
        if self.target_base_offset != expected_target_offset:
            raise ValueError("transfer region target base offset is inconsistent")
        if (
            self.inner_bytes,
            self.outer_loop_counts,
            self.source_strides,
            self.target_strides,
        ) != (
            expected_inner_bytes,
            expected_outer_loop_counts,
            expected_source_strides,
            expected_target_strides,
        ):
            raise ValueError("transfer region loop geometry is not canonical")

        _validate_outer_strides(
            self.outer_loop_counts,
            self.source_strides,
            self.inner_bytes,
            "source",
        )
        _validate_outer_strides(
            self.outer_loop_counts,
            self.target_strides,
            self.inner_bytes,
            "target",
        )
        self.validate_bounds()

    @property
    def segment_count(self) -> int:
        return prod(self.outer_loop_counts)

    @property
    def total_bytes(self) -> int:
        return self.inner_bytes * self.segment_count

    @property
    def source_offset(self) -> int:
        return self.source_base_offset

    @property
    def target_offset(self) -> int:
        return self.target_base_offset

    @property
    def nbytes(self) -> int:
        return self.inner_bytes

    @property
    def repeat(self) -> int:
        return self.segment_count

    @property
    def source_stride(self) -> int:
        if not self.source_strides:
            return 0
        if len(self.source_strides) == 1:
            return self.source_strides[0]
        raise ValueError("N-D transfer region has multiple source strides")

    @property
    def target_stride(self) -> int:
        if not self.target_strides:
            return 0
        if len(self.target_strides) == 1:
            return self.target_strides[0]
        raise ValueError("N-D transfer region has multiple target strides")

    def validate_bounds(self) -> None:
        source_end = (
            self.source_base_offset
            + sum(
                (count - 1) * stride
                for count, stride in zip(
                    self.outer_loop_counts, self.source_strides, strict=True
                )
            )
            + self.inner_bytes
        )
        if source_end > self.source.nbytes:
            raise ValueError("transfer region exceeds source fragment")
        target_end = (
            self.target_base_offset
            + sum(
                (count - 1) * stride
                for count, stride in zip(
                    self.outer_loop_counts, self.target_strides, strict=True
                )
            )
            + self.inner_bytes
        )
        if target_end > self.target.nbytes:
            raise ValueError("transfer region exceeds target fragment")

    def iter_segments(self) -> Iterable[tuple[int, int, int]]:
        if not self.outer_loop_counts:
            yield self.source_base_offset, self.target_base_offset, self.inner_bytes
            return
        indices = [0] * len(self.outer_loop_counts)
        while True:
            yield (
                self.source_base_offset
                + sum(
                    index * stride
                    for index, stride in zip(indices, self.source_strides, strict=True)
                ),
                self.target_base_offset
                + sum(
                    index * stride
                    for index, stride in zip(indices, self.target_strides, strict=True)
                ),
                self.inner_bytes,
            )
            for dim in range(len(indices) - 1, -1, -1):
                indices[dim] += 1
                if indices[dim] < self.outer_loop_counts[dim]:
                    break
                indices[dim] = 0
            else:
                return


def _box_contains(
    outer_offset: tuple[int, ...],
    outer_shape: tuple[int, ...],
    inner_offset: tuple[int, ...],
    inner_shape: tuple[int, ...],
) -> bool:
    return all(
        outer_begin <= inner_begin
        and inner_begin + inner_extent <= outer_begin + outer_extent
        for outer_begin, outer_extent, inner_begin, inner_extent in zip(
            outer_offset,
            outer_shape,
            inner_offset,
            inner_shape,
            strict=True,
        )
    )


def _fragment_itemsize(fragment: SourceFragment | PlacementFragment) -> int:
    elements = prod(fragment.local_shape)
    if elements <= 0 or fragment.nbytes % elements != 0:
        raise ValueError("transfer region fragment byte size is invalid")
    itemsize = fragment.nbytes // elements
    if itemsize <= 0:
        raise ValueError("transfer region fragment itemsize is invalid")
    return itemsize


def _logical_byte_offset(
    fragment: SourceFragment | TargetFragment,
    overlap_offset: tuple[int, ...],
    itemsize: int,
) -> int:
    linear_offset = 0
    stride = 1
    for global_begin, local_begin, extent in zip(
        reversed(overlap_offset),
        reversed(fragment.global_offset),
        reversed(fragment.local_shape),
        strict=True,
    ):
        linear_offset += (global_begin - local_begin) * stride
        stride *= extent
    return linear_offset * itemsize


def _validate_outer_strides(
    counts: tuple[int, ...],
    strides: tuple[int, ...],
    inner_bytes: int,
    side: str,
) -> None:
    span = inner_bytes
    for count, stride in zip(reversed(counts), reversed(strides), strict=True):
        if count > 1 and stride < span:
            raise ValueError(f"transfer region {side} strides overlap")
        span += (count - 1) * stride


TransferOperation = CopyRange | TransferRegion


@dataclass(frozen=True)
class TransferPlan:
    model_id: str
    revision: str
    operations: tuple[TransferOperation, ...]
    source_executors: tuple[ExecutorTransferPlan, ...] = ()
    target_executors: tuple[ExecutorTransferPlan, ...] = ()
    pipeline_routes: tuple[PipelineRouteGroup, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(self, "source_executors", tuple(self.source_executors))
        object.__setattr__(self, "target_executors", tuple(self.target_executors))
        object.__setattr__(self, "pipeline_routes", tuple(self.pipeline_routes))
        if not self.model_id or not self.revision:
            raise ValueError("transfer plan identifiers must not be empty")
        if any(
            not isinstance(operation.target, RuntimeFragment)
            for operation in self.operations
        ):
            raise ValueError("executable transfer plan target must be runtime-bound")
        if any(
            isinstance(operation.source, PlacementFragment)
            for operation in self.operations
        ):
            raise ValueError("executable transfer plan source must be runtime-bound")
        for executor in (*self.source_executors, *self.target_executors):
            if any(
                index >= len(self.operations) for index in executor.operation_indices
            ):
                raise ValueError("executor operation index is out of range")
        route_indices = []
        route_keys = set()
        for route in self.pipeline_routes:
            if not isinstance(route, PipelineRouteGroup):
                raise ValueError("transfer plan has invalid pipeline route metadata")
            key = (route.source_pp, route.target_pp)
            if key in route_keys:
                raise ValueError("transfer plan has duplicate pipeline route groups")
            route_keys.add(key)
            if any(index >= len(self.operations) for index in route.operation_indices):
                raise ValueError("pipeline route operation index is out of range")
            route_indices.extend(route.operation_indices)
        if self.pipeline_routes and sorted(route_indices) != list(
            range(len(self.operations))
        ):
            raise ValueError("pipeline routes must cover every operation exactly once")

    @property
    def total_bytes(self) -> int:
        return sum(operation.total_bytes for operation in self.operations)

    @property
    def regions(self) -> tuple[TransferOperation, ...]:
        return self.operations


@dataclass(frozen=True)
class PlacementExecutorPlan:
    placement_id: str
    rank: ParallelRank
    placement_fragment_ids: tuple[str, ...]
    operation_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "placement_fragment_ids", tuple(self.placement_fragment_ids)
        )
        object.__setattr__(self, "operation_indices", tuple(self.operation_indices))
        if not self.placement_id or not self.placement_fragment_ids:
            raise ValueError("placement executor identifiers must not be empty")
        if len(self.placement_fragment_ids) != len(set(self.placement_fragment_ids)):
            raise ValueError("placement executor has duplicate fragment IDs")
        if any(type(index) is not int or index < 0 for index in self.operation_indices):
            raise ValueError(
                "placement executor operation indices must be non-negative integers"
            )


@dataclass(frozen=True)
class LogicalTransferPlan:
    model_id: str
    revision: str
    source_placements: tuple[PlacementManifest, ...]
    target_placements: tuple[PlacementManifest, ...]
    source_tensors: tuple[TensorDescriptor, ...]
    target_tensors: tuple[TensorDescriptor, ...]
    operations: tuple[TransferOperation, ...]
    source_executors: tuple[PlacementExecutorPlan, ...] = ()
    target_executors: tuple[PlacementExecutorPlan, ...] = ()
    pipeline_routes: tuple[PipelineRouteGroup, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "source_placements",
            "target_placements",
            "source_tensors",
            "target_tensors",
            "operations",
            "source_executors",
            "target_executors",
            "pipeline_routes",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.model_id or not self.revision or not self.target_placements:
            raise ValueError("logical transfer plan identifiers must not be empty")
        for side, placements in (
            ("source", self.source_placements),
            ("target", self.target_placements),
        ):
            placement_ids = [placement.placement_id for placement in placements]
            if len(placement_ids) != len(set(placement_ids)):
                raise ValueError(
                    f"logical transfer plan has duplicate {side} placement IDs"
                )
            if any(
                placement.model_id != self.model_id
                or placement.revision != self.revision
                for placement in placements
            ):
                raise ValueError(
                    f"logical transfer plan {side} placement identity differs"
                )
        if self.source_placements and any(
            not isinstance(operation.source, PlacementFragment)
            for operation in self.operations
        ):
            raise ValueError("logical transfer plan source must be a placement")
        if not self.source_placements and any(
            not isinstance(operation.source, StoredFragment)
            for operation in self.operations
        ):
            raise ValueError("logical transfer plan source has no placement")
        if any(
            not isinstance(operation.target, PlacementFragment)
            for operation in self.operations
        ):
            raise ValueError("logical transfer plan target must be a placement")
        for executor in (*self.source_executors, *self.target_executors):
            if any(
                index >= len(self.operations) for index in executor.operation_indices
            ):
                raise ValueError("logical executor operation index is out of range")
        route_indices = [
            index for route in self.pipeline_routes for index in route.operation_indices
        ]
        if self.pipeline_routes and sorted(route_indices) != list(
            range(len(self.operations))
        ):
            raise ValueError(
                "logical pipeline routes must cover every operation exactly once"
            )

    @property
    def total_bytes(self) -> int:
        return sum(operation.total_bytes for operation in self.operations)

    @property
    def source_placement_ids(self) -> tuple[str, ...]:
        return tuple(placement.placement_id for placement in self.source_placements)

    @property
    def target_placement_ids(self) -> tuple[str, ...]:
        return tuple(placement.placement_id for placement in self.target_placements)


@dataclass(frozen=True)
class _PlannedTransfer:
    model_id: str
    revision: str
    operations: tuple[TransferOperation, ...]


def _collect_manifests(
    manifests: Sequence[RuntimeManifest], label: str
) -> tuple[dict[str, TensorDescriptor], list[RuntimeFragment]]:
    if not manifests:
        raise ValueError(f"{label} manifests must not be empty")

    model_id = manifests[0].model_id
    revision = manifests[0].revision
    tensors: dict[str, TensorDescriptor] = {}
    fragments: list[RuntimeFragment] = []
    fragment_ids: set[str] = set()
    for manifest in manifests:
        if manifest.model_id != model_id or manifest.revision != revision:
            raise ValueError(f"{label} manifests describe different revisions")
        for tensor in manifest.tensors:
            previous = tensors.setdefault(tensor.tensor_id, tensor)
            if previous != tensor:
                raise ValueError(
                    f"{label} tensor descriptor mismatch: {tensor.tensor_id}"
                )
        for fragment in manifest.fragments:
            if fragment.fragment_id in fragment_ids:
                raise ValueError(
                    f"duplicate {label} fragment_id: {fragment.fragment_id}"
                )
            fragment_ids.add(fragment.fragment_id)
            fragments.append(fragment)
    return tensors, fragments


def _collect_placements(
    manifests: Sequence[PlacementManifest],
    label: str,
) -> tuple[dict[str, TensorDescriptor], list[PlacementFragment]]:
    if not manifests:
        raise ValueError(f"{label} placement manifests must not be empty")
    model_id = manifests[0].model_id
    revision = manifests[0].revision
    placement_ids: set[str] = set()
    fragment_ids: set[str] = set()
    tensors: dict[str, TensorDescriptor] = {}
    fragments: list[PlacementFragment] = []
    for manifest in manifests:
        if manifest.model_id != model_id or manifest.revision != revision:
            raise ValueError(
                f"{label} placement manifests describe different revisions"
            )
        if manifest.placement_id in placement_ids:
            raise ValueError(f"duplicate {label} placement_id: {manifest.placement_id}")
        placement_ids.add(manifest.placement_id)
        for tensor in manifest.tensors:
            previous = tensors.setdefault(tensor.tensor_id, tensor)
            if previous != tensor:
                raise ValueError(
                    f"{label} placement tensor descriptor mismatch: {tensor.tensor_id}"
                )
        for fragment in manifest.fragments:
            if fragment.placement_fragment_id in fragment_ids:
                raise ValueError(
                    f"duplicate {label} placement fragment: "
                    f"{fragment.placement_fragment_id}"
                )
            fragment_ids.add(fragment.placement_fragment_id)
            fragments.append(fragment)
    return tensors, fragments


def _build_executor_plans(
    manifests: Sequence[RuntimeManifest],
    operations: Sequence[TransferOperation],
    side: str,
) -> tuple[ExecutorTransferPlan, ...]:
    if side not in ("source", "target"):
        raise ValueError(f"invalid executor side: {side}")
    result = []
    ranks: set[ParallelRank] = set()
    operation_indices_by_fragment: dict[str, list[int]] = {}
    for index, operation in enumerate(operations):
        fragment_id = getattr(operation, side).fragment_id
        operation_indices_by_fragment.setdefault(fragment_id, []).append(index)
    for manifest in manifests:
        if not manifest.fragments:
            raise ValueError(f"{side} executor manifest has no fragments")
        fragments_by_rank: dict[ParallelRank, list[RuntimeFragment]] = {}
        for fragment in manifest.fragments:
            fragments_by_rank.setdefault(fragment.rank, []).append(fragment)
        for rank, fragments in fragments_by_rank.items():
            workers = {fragment.worker_id for fragment in fragments}
            if len(workers) != 1:
                raise ValueError(f"{side} executor rank spans multiple workers: {rank}")
            if rank in ranks:
                raise ValueError(f"duplicate {side} executor rank: {rank}")
            ranks.add(rank)
            ordered_fragments = sorted(
                fragments, key=lambda fragment: fragment.fragment_id
            )
            fragment_ids = tuple(fragment.fragment_id for fragment in ordered_fragments)
            operation_indices = tuple(
                sorted(
                    index
                    for fragment_id in fragment_ids
                    for index in operation_indices_by_fragment.get(fragment_id, ())
                )
            )
            result.append(
                ExecutorTransferPlan(
                    instance_id=manifest.instance_id,
                    runtime_lease_id=manifest.lease_id,
                    worker_id=next(iter(workers)),
                    rank=rank,
                    fragment_ids=fragment_ids,
                    fragment_leases=tuple(
                        RuntimeLeaseSnapshot.from_fragment(fragment)
                        for fragment in ordered_fragments
                    ),
                    operation_indices=operation_indices,
                )
            )
    result.sort(
        key=lambda item: (item.rank.dp, item.rank.pp, item.rank.ep, item.rank.tp)
    )
    return tuple(result)


def _build_placement_executor_plans(
    manifests: Sequence[PlacementManifest],
    operations: Sequence[TransferOperation],
    side: str,
) -> tuple[PlacementExecutorPlan, ...]:
    if side not in ("source", "target"):
        raise ValueError(f"invalid placement executor side: {side}")
    operation_indices_by_fragment: dict[str, list[int]] = {}
    for index, operation in enumerate(operations):
        fragment = getattr(operation, side)
        if not isinstance(fragment, PlacementFragment):
            raise ValueError(f"placement executor operation {side} is runtime-bound")
        operation_indices_by_fragment.setdefault(
            fragment.placement_fragment_id, []
        ).append(index)

    result = []
    ranks: set[ParallelRank] = set()
    for manifest in manifests:
        if not manifest.fragments:
            raise ValueError(f"{side} placement executor has no fragments")
        fragments_by_rank: dict[ParallelRank, list[PlacementFragment]] = {}
        for fragment in manifest.fragments:
            fragments_by_rank.setdefault(fragment.rank, []).append(fragment)
        for rank, fragments in fragments_by_rank.items():
            if rank in ranks:
                raise ValueError(f"duplicate {side} placement executor rank: {rank}")
            ranks.add(rank)
            fragment_ids = tuple(
                sorted(fragment.placement_fragment_id for fragment in fragments)
            )
            operation_indices = tuple(
                sorted(
                    index
                    for fragment_id in fragment_ids
                    for index in operation_indices_by_fragment.get(fragment_id, ())
                )
            )
            result.append(
                PlacementExecutorPlan(
                    placement_id=manifest.placement_id,
                    rank=rank,
                    placement_fragment_ids=fragment_ids,
                    operation_indices=operation_indices,
                )
            )
    result.sort(
        key=lambda item: (item.rank.dp, item.rank.pp, item.rank.ep, item.rank.tp)
    )
    return tuple(result)


def resolve_executor_plans(
    plan: TransferPlan,
    manifest: RuntimeManifest,
    side: str,
) -> tuple[ExecutorTransferPlan, ...]:
    if side == "source":
        executors = plan.source_executors
    elif side == "target":
        executors = plan.target_executors
    else:
        raise ValueError(f"invalid executor side: {side}")
    if not executors:
        raise ValueError(f"transfer plan has no {side} executor metadata")
    if not manifest.fragments:
        raise ValueError(f"{side} executor snapshot mismatch: no fragments")
    expected_executors = tuple(
        executor
        for executor in executors
        if executor.instance_id == manifest.instance_id
    )
    if not expected_executors:
        raise ValueError(f"{side} executor snapshot mismatch: unknown instance")
    executor_by_rank = {executor.rank: executor for executor in expected_executors}
    fragments_by_rank: dict[ParallelRank, list[RuntimeFragment]] = {}
    for fragment in manifest.fragments:
        fragments_by_rank.setdefault(fragment.rank, []).append(fragment)
    if set(fragments_by_rank) != set(executor_by_rank):
        raise ValueError(f"{side} executor snapshot mismatch: executor set changed")
    result = []
    for rank, fragments in fragments_by_rank.items():
        executor = executor_by_rank.get(rank)
        workers = {fragment.worker_id for fragment in fragments}
        ordered_fragments = sorted(fragments, key=lambda fragment: fragment.fragment_id)
        current_ids = tuple(fragment.fragment_id for fragment in ordered_fragments)
        current_leases = tuple(
            RuntimeLeaseSnapshot.from_fragment(fragment)
            for fragment in ordered_fragments
        )
        if executor is None or len(workers) != 1:
            raise ValueError(f"{side} executor snapshot mismatch: unknown rank {rank}")
        if (
            manifest.instance_id != executor.instance_id
            or manifest.lease_id != executor.runtime_lease_id
            or next(iter(workers)) != executor.worker_id
            or current_ids != executor.fragment_ids
            or current_leases != executor.fragment_leases
        ):
            raise ValueError(f"{side} executor snapshot mismatch")
        result.append(executor)
    result.sort(
        key=lambda item: (item.rank.dp, item.rank.pp, item.rank.ep, item.rank.tp)
    )
    return tuple(result)


def resolve_executor_plan(
    plan: TransferPlan,
    manifest: RuntimeManifest,
    side: str,
) -> ExecutorTransferPlan:
    executors = resolve_executor_plans(plan, manifest, side)
    if len(executors) != 1:
        raise ValueError(f"{side} executor snapshot contains multiple ranks")
    return executors[0]


def _validate_tensor_compatibility(
    source: TensorDescriptor, target: TensorDescriptor
) -> None:
    if source.layout_fingerprint != target.layout_fingerprint:
        raise ValueError(f"layout mismatch for tensor {source.tensor_id}")
    if (
        source.global_shape != target.global_shape
        or source.dtype != target.dtype
        or source.itemsize != target.itemsize
        or source.layer_id != target.layer_id
        or source.expert_id != target.expert_id
    ):
        raise ValueError(f"tensor descriptor mismatch: {source.tensor_id}")


def _validate_tensor_sets(
    source_tensors: dict[str, TensorDescriptor],
    target_tensors: dict[str, TensorDescriptor],
) -> None:
    source_ids = set(source_tensors)
    target_ids = set(target_tensors)
    missing = sorted(source_ids - target_ids)
    if missing:
        raise ValueError(f"target manifests are missing tensors: {', '.join(missing)}")
    unexpected = sorted(target_ids - source_ids)
    if unexpected:
        raise ValueError(
            f"target manifests contain unknown tensors: {', '.join(unexpected)}"
        )


def _validate_tensor_subset(
    source_tensors: dict[str, TensorDescriptor],
    target_tensors: dict[str, TensorDescriptor],
) -> None:
    unexpected = sorted(set(target_tensors) - set(source_tensors))
    if unexpected:
        raise ValueError(
            f"target manifests contain unknown tensors: {', '.join(unexpected)}"
        )


def _validate_target_coverage(
    target_tensors: dict[str, TensorDescriptor],
    target_fragments: Sequence[TargetFragment],
) -> None:
    if not target_fragments:
        raise ValueError("target manifests have no fragments")
    fragments_by_dp_and_tensor: dict[int, dict[str, list[TargetFragment]]] = {}
    for fragment in target_fragments:
        fragments_by_dp_and_tensor.setdefault(fragment.rank.dp, {}).setdefault(
            fragment.tensor_id, []
        ).append(fragment)
    dp_ranks = sorted(fragments_by_dp_and_tensor)
    for dp_rank in dp_ranks:
        for tensor in target_tensors.values():
            fragments_by_owner: dict[RuntimeTensorOwner, list[TargetFragment]] = {}
            for fragment in fragments_by_dp_and_tensor[dp_rank].get(
                tensor.tensor_id, ()
            ):
                fragments_by_owner.setdefault(
                    _parallel_tensor_owner(tensor, fragment), []
                ).append(fragment)
            if not fragments_by_owner or any(
                not _fragments_fully_cover_tensor(tensor, fragments)
                for fragments in fragments_by_owner.values()
            ):
                raise ValueError(
                    f"target tensor is not fully covered: {tensor.tensor_id}: "
                    f"dp={dp_rank}"
                )


def _fragments_fully_cover_tensor(
    tensor: TensorDescriptor,
    fragments: Sequence[SourceFragment | PlacementFragment],
) -> bool:
    geometries = {
        (fragment.global_offset, fragment.local_shape): fragment
        for fragment in fragments
        if fragment.tensor_id == tensor.tensor_id
    }
    boxes = tuple(geometries)
    return _boxes_exactly_cover(
        (0,) * len(tensor.global_shape), tensor.global_shape, boxes
    )


def _boxes_exactly_cover(
    container_offset: tuple[int, ...],
    container_shape: tuple[int, ...],
    boxes: Sequence[tuple[tuple[int, ...], tuple[int, ...]]],
) -> bool:
    unique_boxes = tuple(dict.fromkeys(boxes))
    if not unique_boxes:
        return False
    if any(
        not _box_contains(container_offset, container_shape, offset, shape)
        for offset, shape in unique_boxes
    ):
        return False
    if sum(prod(shape) for _, shape in unique_boxes) != prod(container_shape):
        return False
    return not _boxes_overlap(unique_boxes)


def _boxes_overlap(
    boxes: Sequence[tuple[tuple[int, ...], tuple[int, ...]]],
) -> bool:
    if len(boxes) < 2:
        return False
    ndim = len(boxes[0][0])
    sweep_dim = max(
        range(ndim),
        key=lambda dim: len(
            {(offset[dim], offset[dim] + shape[dim]) for offset, shape in boxes}
        ),
    )
    ordered = sorted(boxes, key=lambda item: item[0][sweep_dim])
    active: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for offset, shape in ordered:
        begin = offset[sweep_dim]
        active = [
            candidate
            for candidate in active
            if candidate[0][sweep_dim] + candidate[1][sweep_dim] > begin
        ]
        if any(
            all(
                left_begin < right_begin + right_extent
                and right_begin < left_begin + left_extent
                for left_begin, left_extent, right_begin, right_extent in zip(
                    candidate_offset,
                    candidate_shape,
                    offset,
                    shape,
                    strict=True,
                )
            )
            for candidate_offset, candidate_shape in active
        ):
            return True
        active.append((offset, shape))
    return False


def _parallel_tensor_owner(
    tensor: TensorDescriptor, fragment: RuntimeFragment | PlacementFragment
) -> RuntimeTensorOwner:
    return (
        fragment.rank.pp,
        fragment.rank.ep if tensor.expert_id is not None else None,
    )


# Store upload keeps importing the historical private runtime helper names.
_runtime_tensor_owner = _parallel_tensor_owner


def _complete_parallel_source_replicas(
    source_tensors: dict[str, TensorDescriptor],
    source_fragments: Sequence[RuntimeFragment | PlacementFragment],
) -> dict[int, dict[str, RuntimeTensorOwner]]:
    replicas: dict[int, dict[str, RuntimeTensorOwner]] = {}
    generation_by_dp: dict[int, int] = {}
    fragments_by_dp_and_tensor: dict[
        int, dict[str, list[RuntimeFragment | PlacementFragment]]
    ] = {}
    generations_by_dp: dict[int, set[int]] = {}
    runtime_sources = all(
        isinstance(fragment, RuntimeFragment) for fragment in source_fragments
    )
    for fragment in source_fragments:
        dp_rank = fragment.rank.dp
        fragments_by_dp_and_tensor.setdefault(dp_rank, {}).setdefault(
            fragment.tensor_id, []
        ).append(fragment)
        if isinstance(fragment, RuntimeFragment):
            generations_by_dp.setdefault(dp_rank, set()).add(fragment.lease_generation)
    for dp_rank in sorted(fragments_by_dp_and_tensor):
        generations = generations_by_dp.get(dp_rank, set())
        if runtime_sources and len(generations) != 1:
            continue
        owner_by_tensor: dict[str, RuntimeTensorOwner] = {}
        complete = True
        for tensor in source_tensors.values():
            fragments_by_owner: dict[
                RuntimeTensorOwner, list[RuntimeFragment | PlacementFragment]
            ] = {}
            for fragment in fragments_by_dp_and_tensor[dp_rank].get(
                tensor.tensor_id, ()
            ):
                fragments_by_owner.setdefault(
                    _parallel_tensor_owner(tensor, fragment), []
                ).append(fragment)
            complete_owners = [
                owner
                for owner, fragments in fragments_by_owner.items()
                if _fragments_fully_cover_tensor(tensor, fragments)
            ]
            if not fragments_by_owner or len(complete_owners) != len(
                fragments_by_owner
            ):
                complete = False
                break
            owner_by_tensor[tensor.tensor_id] = min(complete_owners)
        if complete:
            replicas[dp_rank] = owner_by_tensor
            if runtime_sources:
                generation_by_dp[dp_rank] = next(iter(generations))
    if not replicas:
        raise ValueError(
            "source manifests have no complete DP replica; tensors are not fully covered"
        )
    if runtime_sources and len(set(generation_by_dp.values())) != 1:
        raise ValueError("source DP replicas have inconsistent lease generations")
    return replicas


_complete_runtime_source_replicas = _complete_parallel_source_replicas


def _validate_local_target_inventory(
    target_tensors: dict[str, TensorDescriptor],
    target_fragments: Sequence[TargetFragment],
) -> None:
    if not target_fragments:
        raise ValueError("local target manifest has no fragments")
    ranks = {fragment.rank for fragment in target_fragments}
    if len(ranks) != 1:
        raise ValueError("local target manifest must describe exactly one executor")
    if all(isinstance(fragment, RuntimeFragment) for fragment in target_fragments):
        workers = {
            fragment.worker_id
            for fragment in target_fragments
            if isinstance(fragment, RuntimeFragment)
        }
        if len(workers) != 1:
            raise ValueError("local target manifest must describe exactly one executor")
    missing = sorted(
        set(target_tensors) - {item.tensor_id for item in target_fragments}
    )
    if missing:
        raise ValueError(
            f"local target manifest is missing fragments: {', '.join(missing)}"
        )


def _geometry_key(fragment: SourceFragment) -> tuple:
    return fragment.tensor_id, fragment.global_offset, fragment.local_shape


def _source_sort_key(fragment: SourceFragment) -> tuple:
    if isinstance(fragment, StoredFragment):
        return (0, 0, 0, 0, fragment.object_key, fragment.fragment_id)
    return (
        fragment.rank.dp,
        fragment.rank.pp,
        fragment.rank.ep,
        fragment.rank.tp,
        fragment.worker_id if isinstance(fragment, RuntimeFragment) else "",
        fragment.fragment_id,
    )


@dataclass(frozen=True)
class _CandidateInterval:
    begin: int
    end: int
    group: tuple[SourceFragment, ...]


@dataclass(frozen=True)
class _CandidateIntervalNode:
    center: int
    crossing_by_begin: tuple[_CandidateInterval, ...]
    crossing_by_end: tuple[_CandidateInterval, ...]
    left: _CandidateIntervalNode | None = None
    right: _CandidateIntervalNode | None = None

    @classmethod
    def build(
        cls, intervals: Sequence[_CandidateInterval]
    ) -> _CandidateIntervalNode | None:
        if not intervals:
            return None
        center = sorted((interval.begin + interval.end) // 2 for interval in intervals)[
            len(intervals) // 2
        ]
        left = []
        right = []
        crossing = []
        for interval in intervals:
            if interval.end <= center:
                left.append(interval)
            elif interval.begin > center:
                right.append(interval)
            else:
                crossing.append(interval)
        return cls(
            center=center,
            crossing_by_begin=tuple(
                sorted(crossing, key=lambda item: (item.begin, item.end))
            ),
            crossing_by_end=tuple(
                sorted(crossing, key=lambda item: (item.end, item.begin))
            ),
            left=cls.build(left),
            right=cls.build(right),
        )

    def query(
        self,
        begin: int,
        end: int,
        result: list[tuple[SourceFragment, ...]],
    ) -> None:
        if end <= self.center:
            for interval in self.crossing_by_begin:
                if interval.begin >= end:
                    break
                result.append(interval.group)
            if self.left is not None:
                self.left.query(begin, end, result)
            return
        if begin > self.center:
            for interval in reversed(self.crossing_by_end):
                if interval.end <= begin:
                    break
                result.append(interval.group)
            if self.right is not None:
                self.right.query(begin, end, result)
            return

        result.extend(interval.group for interval in self.crossing_by_begin)
        if begin < self.center and self.left is not None:
            self.left.query(begin, end, result)
        if self.right is not None:
            self.right.query(begin, end, result)


@dataclass(frozen=True)
class _CandidateBoxIndex:
    dimension: int
    root: _CandidateIntervalNode

    @classmethod
    def build(cls, groups: Sequence[tuple[SourceFragment, ...]]) -> _CandidateBoxIndex:
        if not groups:
            raise ValueError("candidate box index requires source fragments")
        ndim = len(groups[0][0].global_offset)
        dimension = max(
            range(ndim),
            key=lambda dim: (
                len(
                    {
                        (
                            group[0].global_offset[dim],
                            group[0].global_offset[dim] + group[0].local_shape[dim],
                        )
                        for group in groups
                    }
                ),
                -dim,
            ),
        )
        root = _CandidateIntervalNode.build(
            tuple(
                _CandidateInterval(
                    begin=group[0].global_offset[dimension],
                    end=(
                        group[0].global_offset[dimension]
                        + group[0].local_shape[dimension]
                    ),
                    group=group,
                )
                for group in groups
            )
        )
        assert root is not None
        return cls(dimension=dimension, root=root)

    def query(self, target: TargetFragment) -> tuple[tuple[SourceFragment, ...], ...]:
        begin = target.global_offset[self.dimension]
        result = []
        self.root.query(
            begin,
            begin + target.local_shape[self.dimension],
            result,
        )
        return tuple(result)


def _overlap_box(
    source: SourceFragment,
    target: TargetFragment,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    overlap_offset = tuple(
        max(source_begin, target_begin)
        for source_begin, target_begin in zip(
            source.global_offset, target.global_offset, strict=True
        )
    )
    overlap_end = tuple(
        min(
            source_begin + source_extent,
            target_begin + target_extent,
        )
        for source_begin, source_extent, target_begin, target_extent in zip(
            source.global_offset,
            source.local_shape,
            target.global_offset,
            target.local_shape,
            strict=True,
        )
    )
    overlap_shape = tuple(
        end - begin for begin, end in zip(overlap_offset, overlap_end, strict=True)
    )
    if any(extent <= 0 for extent in overlap_shape):
        return None
    return overlap_offset, overlap_shape


def _canonical_byte_strides(shape: tuple[int, ...], itemsize: int) -> tuple[int, ...]:
    result = []
    running = itemsize
    for extent in reversed(shape):
        result.append(running)
        running *= extent
    return tuple(reversed(result))


def _derive_region_geometry(
    source: SourceFragment,
    target: TargetFragment,
    overlap_offset: tuple[int, ...],
    overlap_shape: tuple[int, ...],
) -> tuple[
    int,
    int,
    int,
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    source_itemsize = _fragment_itemsize(source)
    target_itemsize = _fragment_itemsize(target)
    if source_itemsize != target_itemsize:
        raise ValueError("transfer region source and target itemsize differ")

    source_byte_strides = _canonical_byte_strides(source.local_shape, source_itemsize)
    target_byte_strides = _canonical_byte_strides(target.local_shape, target_itemsize)
    source_base_offset = sum(
        (overlap_begin - fragment_begin) * stride
        for overlap_begin, fragment_begin, stride in zip(
            overlap_offset,
            source.global_offset,
            source_byte_strides,
            strict=True,
        )
    )
    target_base_offset = sum(
        (overlap_begin - fragment_begin) * stride
        for overlap_begin, fragment_begin, stride in zip(
            overlap_offset,
            target.global_offset,
            target_byte_strides,
            strict=True,
        )
    )

    suffix_begin = len(overlap_shape) - 1
    inner_bytes = overlap_shape[-1] * source_itemsize
    for dim in range(len(overlap_shape) - 2, -1, -1):
        if (
            source_byte_strides[dim] != inner_bytes
            or target_byte_strides[dim] != inner_bytes
        ):
            break
        inner_bytes *= overlap_shape[dim]
        suffix_begin = dim

    return (
        source_base_offset,
        target_base_offset,
        inner_bytes,
        overlap_shape[:suffix_begin],
        source_byte_strides[:suffix_begin],
        target_byte_strides[:suffix_begin],
    )


def _transfer_region(
    tensor: TensorDescriptor,
    source: SourceFragment,
    target: TargetFragment,
    overlap_offset: tuple[int, ...],
    overlap_shape: tuple[int, ...],
) -> TransferRegion:
    if (
        _fragment_itemsize(source) != tensor.itemsize
        or _fragment_itemsize(target) != tensor.itemsize
    ):
        raise ValueError("transfer region fragment itemsize differs from descriptor")
    (
        source_base_offset,
        target_base_offset,
        inner_bytes,
        outer_loop_counts,
        source_strides,
        target_strides,
    ) = _derive_region_geometry(source, target, overlap_offset, overlap_shape)

    return TransferRegion(
        tensor_id=tensor.tensor_id,
        source=source,
        target=target,
        overlap_offset=overlap_offset,
        overlap_shape=overlap_shape,
        source_base_offset=source_base_offset,
        target_base_offset=target_base_offset,
        inner_bytes=inner_bytes,
        outer_loop_counts=outer_loop_counts,
        source_strides=source_strides,
        target_strides=target_strides,
    )


def _plan_transfer(
    model_id: str,
    revision: str,
    source_tensors: dict[str, TensorDescriptor],
    source_fragments: Sequence[SourceFragment],
    target_tensors: dict[str, TensorDescriptor],
    target_fragments: Sequence[TargetFragment],
    *,
    local_target: bool = False,
) -> _PlannedTransfer:
    if local_target:
        _validate_tensor_subset(source_tensors, target_tensors)
        _validate_local_target_inventory(target_tensors, target_fragments)
    else:
        _validate_tensor_sets(source_tensors, target_tensors)
        _validate_target_coverage(target_tensors, target_fragments)
    runtime_sources = all(
        isinstance(fragment, RuntimeFragment) for fragment in source_fragments
    )
    placement_sources = all(
        isinstance(fragment, PlacementFragment) for fragment in source_fragments
    )
    stored_sources = all(
        isinstance(fragment, StoredFragment) for fragment in source_fragments
    )
    if sum((runtime_sources, placement_sources, stored_sources)) != 1:
        raise ValueError(
            "source fragments mix runtime, placement, and stored locations"
        )
    parallel_sources = runtime_sources or placement_sources
    source_replicas = (
        _complete_parallel_source_replicas(source_tensors, source_fragments)
        if parallel_sources
        else {}
    )
    source_dp_ranks = sorted(source_replicas)
    source_dp_by_target_dp = (
        {
            target_dp: source_dp_ranks[target_dp % len(source_dp_ranks)]
            for target_dp in {fragment.rank.dp for fragment in target_fragments}
        }
        if parallel_sources
        else {}
    )
    candidates: dict[str, dict[tuple, list[SourceFragment]]] = {}
    for fragment in source_fragments:
        candidates.setdefault(fragment.tensor_id, {}).setdefault(
            _geometry_key(fragment), []
        ).append(fragment)
    for tensor_candidates in candidates.values():
        for group in tensor_candidates.values():
            group.sort(key=_source_sort_key)
    candidate_indexes = {
        tensor_id: _CandidateBoxIndex.build(
            tuple(tuple(group) for group in tensor_candidates.values())
        )
        for tensor_id, tensor_candidates in candidates.items()
    }

    operations: list[TransferRegion] = []
    for target in sorted(target_fragments, key=lambda item: item.fragment_id):
        target_tensor = target_tensors[target.tensor_id]
        source_tensor = source_tensors.get(target.tensor_id)
        if source_tensor is None:
            raise ValueError(f"missing source tensor: {target.tensor_id}")
        _validate_tensor_compatibility(source_tensor, target_tensor)

        overlaps: list[tuple[tuple[int, ...], tuple[int, ...], SourceFragment]] = []
        candidate_index = candidate_indexes.get(target.tensor_id)
        candidate_groups = (
            candidate_index.query(target) if candidate_index is not None else ()
        )
        for group in candidate_groups:
            if parallel_sources:
                source_dp = source_dp_by_target_dp[target.rank.dp]
                source_owner = source_replicas[source_dp][target.tensor_id]
                eligible = [
                    fragment
                    for fragment in group
                    if isinstance(fragment, (RuntimeFragment, PlacementFragment))
                    and fragment.rank.dp == source_dp
                    and _parallel_tensor_owner(source_tensor, fragment) == source_owner
                ]
                if not eligible:
                    continue
                representative = eligible[0]
                selected = representative
            else:
                representative = group[0]
                selected = representative
            overlap = _overlap_box(representative, target)
            if overlap is None:
                continue
            overlaps.append((*overlap, selected))
        overlaps.sort(key=lambda item: (item[0], item[1], item[2].fragment_id))

        if not _boxes_exactly_cover(
            target.global_offset,
            target.local_shape,
            tuple((offset, shape) for offset, shape, _ in overlaps),
        ):
            raise ValueError(
                f"target fragment is not fully covered: {target.fragment_id}"
            )
        operations.extend(
            _transfer_region(
                target_tensor,
                source,
                target,
                overlap_offset,
                overlap_shape,
            )
            for overlap_offset, overlap_shape, source in overlaps
        )

    operations.sort(
        key=lambda item: (
            item.target.fragment_id,
            item.target_base_offset,
            item.source.fragment_id,
            item.source_base_offset,
        )
    )
    return _PlannedTransfer(
        model_id=model_id,
        revision=revision,
        operations=tuple(operations),
    )


def _operation_loop_geometry(
    operation: TransferOperation, side: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if isinstance(operation, TransferRegion):
        strides = (
            operation.source_strides if side == "source" else operation.target_strides
        )
        return operation.outer_loop_counts, strides
    stride = operation.source_stride if side == "source" else operation.target_stride
    return (operation.repeat,), (stride,)


def _operation_sort_key(operation: TransferOperation) -> tuple:
    source_location = (
        (
            "runtime",
            operation.source.worker_id,
            operation.source.endpoint,
            operation.source.lease_generation,
            operation.source.address + operation.source_offset,
        )
        if isinstance(operation.source, RuntimeFragment)
        else (
            "stored",
            operation.source.object_key,
            operation.source.object_offset + operation.source_offset,
        )
    )
    return (
        operation.tensor_id,
        operation.target.worker_id,
        operation.target.endpoint,
        operation.target.address + operation.target_offset,
        source_location,
    )


def _source_copy_identity(operation: TransferOperation) -> tuple:
    counts, strides = _operation_loop_geometry(operation, "source")
    if isinstance(operation.source, RuntimeFragment):
        return (
            "runtime",
            operation.source.worker_id,
            operation.source.endpoint,
            operation.source.lease_generation,
            operation.source.address + operation.source_offset,
            operation.nbytes,
            counts,
            strides,
        )
    return (
        "stored",
        operation.source.object_key,
        operation.source.object_offset + operation.source_offset,
        operation.nbytes,
        counts,
        strides,
    )


def _target_copy_identity(operation: TransferOperation) -> tuple:
    counts, strides = _operation_loop_geometry(operation, "target")
    return (
        operation.target.worker_id,
        operation.target.endpoint,
        operation.target.lease_generation,
        operation.target.address + operation.target_offset,
        operation.nbytes,
        counts,
        strides,
    )


def _descriptor_alias_key(descriptor: TensorDescriptor) -> tuple:
    return (
        descriptor.global_shape,
        descriptor.dtype,
        descriptor.itemsize,
        descriptor.layer_id,
        descriptor.expert_id,
        descriptor.layout_fingerprint,
    )


def _is_declared_target_alias(
    left: TransferOperation,
    right: TransferOperation,
    source_tensors: dict[str, TensorDescriptor],
    target_tensors: dict[str, TensorDescriptor],
) -> bool:
    left_target = left.target
    right_target = right.target
    if (
        isinstance(left.source, StoredFragment)
        or isinstance(right.source, StoredFragment)
        or len(left_target.aliases) < 2
        or left_target.aliases != right_target.aliases
        or (
            isinstance(left.source, (RuntimeFragment, PlacementFragment))
            and (
                len(left.source.aliases) < 2
                or left.source.aliases != right.source.aliases
            )
        )
        or left_target.worker_id != right_target.worker_id
        or left_target.endpoint != right_target.endpoint
        or left_target.lease_generation != right_target.lease_generation
        or left_target.address != right_target.address
        or left_target.nbytes != right_target.nbytes
        or left_target.global_offset != right_target.global_offset
        or left_target.local_shape != right_target.local_shape
        or left.source.global_offset != right.source.global_offset
        or left.source.local_shape != right.source.local_shape
        or left.source_offset != right.source_offset
        or left.target_offset != right.target_offset
        or left.nbytes != right.nbytes
        or type(left) is not type(right)
        or _operation_loop_geometry(left, "source")
        != _operation_loop_geometry(right, "source")
        or _operation_loop_geometry(left, "target")
        != _operation_loop_geometry(right, "target")
        or (
            isinstance(left, TransferRegion)
            and isinstance(right, TransferRegion)
            and (
                left.overlap_offset != right.overlap_offset
                or left.overlap_shape != right.overlap_shape
            )
        )
    ):
        return False
    return (
        _descriptor_alias_key(source_tensors[left.tensor_id])
        == _descriptor_alias_key(source_tensors[right.tensor_id])
        == _descriptor_alias_key(target_tensors[left.tensor_id])
        == _descriptor_alias_key(target_tensors[right.tensor_id])
    )


def _deduplicate_target_copies(
    operations: Sequence[TransferOperation],
    source_tensors: dict[str, TensorDescriptor],
    target_tensors: dict[str, TensorDescriptor],
) -> tuple[TransferOperation, ...]:
    result = []
    seen = set()
    for operation in sorted(operations, key=_operation_sort_key):
        identity = (
            operation.tensor_id,
            _source_copy_identity(operation),
            _target_copy_identity(operation),
        )
        if identity in seen:
            continue
        seen.add(identity)
        result.append(operation)

    by_target: dict[tuple, TransferOperation] = {}
    deduplicated = []
    for operation in result:
        identity = _target_copy_identity(operation)
        previous = by_target.get(identity)
        if previous is None:
            by_target[identity] = operation
            deduplicated.append(operation)
            continue
        if _is_declared_target_alias(
            previous,
            operation,
            source_tensors,
            target_tensors,
        ):
            continue
        deduplicated.append(operation)

    result = sorted(
        deduplicated,
        key=lambda item: (
            item.target.fragment_id,
            item.target_offset,
            item.source.fragment_id,
            item.source_offset,
        ),
    )
    _validate_target_physical_ranges(result)
    return tuple(result)


def _target_physical_bounds(operation: TransferOperation) -> tuple[int, int]:
    begin = operation.target.address + operation.target_offset
    counts, strides = _operation_loop_geometry(operation, "target")
    end = (
        begin
        + sum(
            (count - 1) * stride for count, stride in zip(counts, strides, strict=True)
        )
        + operation.nbytes
    )
    return begin, end


@dataclass
class _SegmentScanBudget:
    limit: int
    checked: int = 0

    def consume(self) -> None:
        self.checked += 1
        if self.checked > self.limit:
            raise ValueError("target physical segment scan budget exceeded")


def _absolute_target_segments(
    operation: TransferOperation,
) -> Iterable[tuple[int, int]]:
    for _, target_offset, nbytes in operation.iter_segments():
        begin = operation.target.address + target_offset
        yield begin, begin + nbytes


def _budgeted_target_segments(
    operation_index: int,
    operation: TransferOperation,
    budget: _SegmentScanBudget,
) -> Iterable[tuple[int, int, int]]:
    for begin, end in _absolute_target_segments(operation):
        budget.consume()
        yield begin, end, operation_index


def _target_fragment_scan_key(operation: TransferOperation) -> tuple:
    target = operation.target
    return (
        target.fragment_id,
        target.tensor_id,
        target.worker_id,
        target.endpoint,
        target.lease_generation,
        target.address,
        target.nbytes,
        target.global_offset,
        target.local_shape,
    )


def _complete_target_fragment_segment(
    indexed_operations: Sequence[tuple[int, TransferOperation]],
) -> tuple[int, int, int] | None:
    if not indexed_operations or not all(
        isinstance(operation, TransferRegion) for _, operation in indexed_operations
    ):
        return None
    target = indexed_operations[0][1].target
    boxes = tuple(
        (operation.overlap_offset, operation.overlap_shape)
        for _, operation in indexed_operations
        if isinstance(operation, TransferRegion)
    )
    if not _boxes_exactly_cover(target.global_offset, target.local_shape, boxes):
        return None
    return target.address, target.address + target.nbytes, indexed_operations[0][0]


def _validate_target_physical_ranges(
    operations: Sequence[TransferOperation],
    *,
    max_segment_checks: int = 1_000_000,
) -> None:
    if type(max_segment_checks) is not int or max_segment_checks <= 0:
        raise ValueError("max_segment_checks must be a positive integer")
    by_executor: dict[tuple[str, str], list[TransferOperation]] = {}
    for operation in operations:
        if (
            isinstance(operation, CopyRange)
            and operation.repeat > 1
            and operation.target_stride < operation.nbytes
        ):
            raise ValueError(
                f"conflicting target physical range: {operation.target.fragment_id}"
            )
        by_executor.setdefault(
            (operation.target.worker_id, operation.target.endpoint), []
        ).append(operation)

    for scoped_operations in by_executor.values():
        by_fragment: dict[tuple, list[tuple[int, TransferOperation]]] = {}
        for index, operation in enumerate(scoped_operations):
            by_fragment.setdefault(_target_fragment_scan_key(operation), []).append(
                (index, operation)
            )

        complete_segments = []
        incomplete_operation_indices = set()
        for indexed_operations in by_fragment.values():
            complete_segment = _complete_target_fragment_segment(indexed_operations)
            if complete_segment is None:
                incomplete_operation_indices.update(
                    index for index, _ in indexed_operations
                )
            else:
                complete_segments.append(complete_segment)
        complete_segments.sort()

        budget = _SegmentScanBudget(max_segment_checks)
        segment_streams: list[Iterable[tuple[int, int, int]]] = []
        if complete_segments:
            segment_streams.append(iter(complete_segments))
        segment_streams.extend(
            (
                _budgeted_target_segments(index, operation, budget)
                for index, operation in enumerate(scoped_operations)
                if index in incomplete_operation_indices
            )
        )
        ordered_segments = heapq.merge(*segment_streams)
        previous_end = -1
        for begin, end, operation_index in ordered_segments:
            if begin < previous_end:
                raise ValueError(
                    "conflicting target physical range: "
                    f"{scoped_operations[operation_index].target.fragment_id}"
                )
            previous_end = max(previous_end, end)


def _build_pipeline_routes(
    operations: Sequence[TransferOperation],
) -> tuple[PipelineRouteGroup, ...]:
    indices_by_route: dict[tuple[int | None, int], list[int]] = {}
    for index, operation in enumerate(operations):
        source_pp = (
            operation.source.rank.pp
            if isinstance(operation.source, (RuntimeFragment, PlacementFragment))
            else None
        )
        indices_by_route.setdefault((source_pp, operation.target.rank.pp), []).append(
            index
        )
    return tuple(
        PipelineRouteGroup(
            source_pp=source_pp,
            target_pp=target_pp,
            operation_indices=tuple(indices),
        )
        for (source_pp, target_pp), indices in sorted(
            indices_by_route.items(),
            key=lambda item: (
                -1 if item[0][0] is None else item[0][0],
                item[0][1],
            ),
        )
    )


def _logical_transfer_plan(
    transfer: _PlannedTransfer,
    *,
    source_tensors: dict[str, TensorDescriptor],
    target_tensors: dict[str, TensorDescriptor],
    source_placements: Sequence[PlacementManifest],
    target_placements: Sequence[PlacementManifest],
) -> LogicalTransferPlan:
    operations = transfer.operations
    return LogicalTransferPlan(
        model_id=transfer.model_id,
        revision=transfer.revision,
        source_placements=tuple(source_placements),
        target_placements=tuple(target_placements),
        source_tensors=tuple(
            sorted(source_tensors.values(), key=lambda item: item.tensor_id)
        ),
        target_tensors=tuple(
            sorted(target_tensors.values(), key=lambda item: item.tensor_id)
        ),
        operations=operations,
        source_executors=(
            _build_placement_executor_plans(source_placements, operations, "source")
            if source_placements
            else ()
        ),
        target_executors=_build_placement_executor_plans(
            target_placements, operations, "target"
        ),
        pipeline_routes=_build_pipeline_routes(operations),
    )


def plan_placement_transfer(
    source_placements: Sequence[PlacementManifest],
    target_placements: Sequence[PlacementManifest],
) -> LogicalTransferPlan:
    """Plan a runtime-to-runtime reshard using only logical placements."""

    source_tensors, source_fragments = _collect_placements(source_placements, "source")
    target_tensors, target_fragments = _collect_placements(target_placements, "target")
    source = source_placements[0]
    target = target_placements[0]
    if source.model_id != target.model_id:
        raise ValueError("source and target model_id differ")
    if source.revision != target.revision:
        raise ValueError("source and target revision differ")
    transfer = _plan_transfer(
        source.model_id,
        source.revision,
        source_tensors,
        source_fragments,
        target_tensors,
        target_fragments,
    )
    return _logical_transfer_plan(
        transfer,
        source_tensors=source_tensors,
        target_tensors=target_tensors,
        source_placements=source_placements,
        target_placements=target_placements,
    )


def plan_placement_transfer_to_local_target(
    source_placements: Sequence[PlacementManifest],
    target_placement: PlacementManifest,
) -> LogicalTransferPlan:
    """Plan one target executor using address-free source and target layouts."""

    source_tensors, source_fragments = _collect_placements(source_placements, "source")
    target_tensors, target_fragments = _collect_placements(
        (target_placement,), "target"
    )
    source = source_placements[0]
    if source.model_id != target_placement.model_id:
        raise ValueError("source and target model_id differ")
    if source.revision != target_placement.revision:
        raise ValueError("source and target revision differ")
    transfer = _plan_transfer(
        source.model_id,
        source.revision,
        source_tensors,
        source_fragments,
        target_tensors,
        target_fragments,
        local_target=True,
    )
    result = _logical_transfer_plan(
        transfer,
        source_tensors=source_tensors,
        target_tensors=target_tensors,
        source_placements=source_placements,
        target_placements=(target_placement,),
    )
    if len(result.target_executors) != 1:
        raise ValueError("local target placement must describe exactly one executor")
    return result


def plan_runtime_transfer_to_target_placements(
    source_manifests: Sequence[RuntimeManifest],
    target_placements: Sequence[PlacementManifest],
) -> LogicalTransferPlan:
    """Project runtime sources into placements before logical planning."""

    return plan_placement_transfer(
        tuple(
            placement_manifest_from_runtime_manifest(manifest)
            for manifest in source_manifests
        ),
        target_placements,
    )


def plan_runtime_transfer_to_local_target_placement(
    source_manifests: Sequence[RuntimeManifest],
    target_placement: PlacementManifest,
) -> LogicalTransferPlan:
    """Plan runtime sources against one address-free target placement."""

    return plan_placement_transfer_to_local_target(
        tuple(
            placement_manifest_from_runtime_manifest(manifest)
            for manifest in source_manifests
        ),
        target_placement,
    )


def plan_stored_transfer_to_target_placements(
    source_manifest: WeightManifest,
    target_placements: Sequence[PlacementManifest],
) -> LogicalTransferPlan:
    target_tensors, target_fragments = _collect_placements(target_placements, "target")
    target = target_placements[0]
    if source_manifest.model_id != target.model_id:
        raise ValueError("source and target model_id differ")
    if source_manifest.revision != target.revision:
        raise ValueError("source and target revision differ")
    source_tensors = {tensor.tensor_id: tensor for tensor in source_manifest.tensors}
    transfer = _plan_transfer(
        source_manifest.model_id,
        source_manifest.revision,
        source_tensors,
        source_manifest.fragments,
        target_tensors,
        target_fragments,
    )
    return _logical_transfer_plan(
        transfer,
        source_tensors=source_tensors,
        target_tensors=target_tensors,
        source_placements=(),
        target_placements=target_placements,
    )


def _bind_placement_manifests(
    placements: Sequence[PlacementManifest],
    binding_inputs: Sequence[RuntimeBindingInput],
    label: str,
) -> tuple[RuntimeManifest, ...]:
    if not placements:
        if binding_inputs:
            raise ValueError(f"logical plan has no {label} placements")
        return ()
    if not binding_inputs:
        raise ValueError(f"logical plan requires {label} runtime bindings")

    placement_by_id = {placement.placement_id: placement for placement in placements}
    binding_by_id: dict[str, RuntimeBindingManifest] = {}
    runtime_by_id: dict[str, RuntimeManifest] = {}
    for item in binding_inputs:
        if isinstance(item, RuntimeManifest):
            actual_placement = placement_manifest_from_runtime_manifest(item)
            effective_placement_id = item.placement_id or actual_placement.placement_id
            expected_placement = placement_by_id.get(effective_placement_id)
            if expected_placement is None:
                raise ValueError(f"logical plan and {label} placement IDs differ")
            if actual_placement != expected_placement:
                raise ValueError(
                    f"runtime fragment does not match {label} placement: "
                    f"{effective_placement_id}"
                )
            runtime_binding_from_runtime_manifest(
                item, placement_id=effective_placement_id
            )
            if effective_placement_id in runtime_by_id:
                raise ValueError(
                    f"duplicate {label} runtime binding: {effective_placement_id}"
                )
            runtime_by_id[effective_placement_id] = item
            continue
        elif isinstance(item, RuntimeBindingManifest):
            binding = item
        else:
            raise ValueError(f"invalid {label} runtime binding input")
        if binding.placement_id in binding_by_id:
            raise ValueError(
                f"duplicate {label} runtime binding: {binding.placement_id}"
            )
        binding_by_id[binding.placement_id] = binding

    if set(binding_by_id).intersection(runtime_by_id):
        duplicate = min(set(binding_by_id).intersection(runtime_by_id))
        raise ValueError(f"duplicate {label} runtime binding: {duplicate}")
    if set(binding_by_id).union(runtime_by_id) != set(placement_by_id):
        raise ValueError(f"logical plan and {label} placement IDs differ")
    return tuple(
        runtime_by_id.get(placement.placement_id)
        or bind_runtime_manifest(placement, binding_by_id[placement.placement_id])
        for placement in placements
    )


def _bound_fragments_by_placement_id(
    placements: Sequence[PlacementManifest],
    manifests: Sequence[RuntimeManifest],
    label: str,
) -> tuple[dict[str, TensorDescriptor], dict[str, RuntimeFragment]]:
    tensors, fragments = _collect_manifests(manifests, label)
    tensors = {
        tensor_id: _canonical_tensor_descriptor(tensor)
        for tensor_id, tensor in tensors.items()
    }
    by_placement_id: dict[str, RuntimeFragment] = {}
    manifest_by_placement_id = {
        (
            manifest.placement_id
            or placement_manifest_from_runtime_manifest(manifest).placement_id
        ): manifest
        for manifest in manifests
    }
    for placement in placements:
        manifest = manifest_by_placement_id[placement.placement_id]
        for fragment in manifest.fragments:
            placement_fragment_id = _logical_fragment_id(fragment)
            if placement_fragment_id in by_placement_id:
                raise ValueError(
                    f"duplicate bound {label} placement fragment: "
                    f"{placement_fragment_id}"
                )
            by_placement_id[placement_fragment_id] = fragment
    return tensors, by_placement_id


def bind_logical_transfer_plan(
    logical_plan: LogicalTransferPlan,
    target_bindings: Sequence[RuntimeBindingInput],
    *,
    source_bindings: Sequence[RuntimeBindingInput] = (),
) -> TransferPlan:
    target_manifests = _bind_placement_manifests(
        logical_plan.target_placements, target_bindings, "target"
    )
    source_manifests = _bind_placement_manifests(
        logical_plan.source_placements, source_bindings, "source"
    )
    if source_manifests:
        source_generations = {
            fragment.lease_generation
            for manifest in source_manifests
            for fragment in manifest.fragments
        }
        if len(source_generations) != 1:
            raise ValueError("source DP replicas have inconsistent lease generations")
    target_tensors, runtime_targets = _bound_fragments_by_placement_id(
        logical_plan.target_placements, target_manifests, "target"
    )
    runtime_sources: dict[str, RuntimeFragment] = {}
    if source_manifests:
        source_runtime_tensors, runtime_sources = _bound_fragments_by_placement_id(
            logical_plan.source_placements, source_manifests, "source"
        )
        expected_source_tensors = {
            tensor.tensor_id: tensor for tensor in logical_plan.source_tensors
        }
        if source_runtime_tensors != expected_source_tensors:
            raise ValueError("logical plan and bound source tensor descriptors differ")

    operations = []
    for operation in logical_plan.operations:
        placement_source = operation.source
        placement_target = operation.target
        if not isinstance(placement_target, PlacementFragment):
            raise ValueError("logical transfer operation target is runtime-bound")
        runtime_target = runtime_targets.get(placement_target.placement_fragment_id)
        if runtime_target is None:
            raise ValueError(
                "missing runtime binding for placement fragment: "
                f"{placement_target.placement_fragment_id}"
            )
        if isinstance(placement_source, PlacementFragment):
            runtime_source = runtime_sources.get(placement_source.placement_fragment_id)
            if runtime_source is None:
                raise ValueError(
                    "missing source runtime binding for placement fragment: "
                    f"{placement_source.placement_fragment_id}"
                )
        elif isinstance(placement_source, StoredFragment):
            runtime_source = placement_source
        else:
            raise ValueError("logical transfer operation source is runtime-bound")
        operations.append(
            replace(operation, source=runtime_source, target=runtime_target)
        )

    source_tensors = {
        tensor.tensor_id: tensor for tensor in logical_plan.source_tensors
    }
    expected_target_tensors = {
        tensor.tensor_id: tensor for tensor in logical_plan.target_tensors
    }
    if target_tensors != expected_target_tensors:
        raise ValueError("logical plan and bound target tensor descriptors differ")
    bound_operations = _deduplicate_target_copies(
        operations, source_tensors, target_tensors
    )
    return TransferPlan(
        model_id=logical_plan.model_id,
        revision=logical_plan.revision,
        operations=bound_operations,
        source_executors=(
            _build_executor_plans(source_manifests, bound_operations, "source")
            if source_manifests
            else ()
        ),
        target_executors=_build_executor_plans(
            target_manifests, bound_operations, "target"
        ),
        pipeline_routes=_build_pipeline_routes(bound_operations),
    )


def plan_runtime_transfer(
    source_manifests: Sequence[RuntimeManifest],
    target_manifests: Sequence[RuntimeManifest],
) -> TransferPlan:
    source_tensors, source_fragments = _collect_manifests(source_manifests, "source")
    target_tensors, target_fragments = _collect_manifests(target_manifests, "target")
    source = source_manifests[0]
    target = target_manifests[0]
    if source.model_id != target.model_id:
        raise ValueError("source and target model_id differ")
    if source.revision != target.revision:
        raise ValueError("source and target revision differ")
    transfer = _plan_transfer(
        source.model_id,
        source.revision,
        source_tensors,
        source_fragments,
        target_tensors,
        target_fragments,
    )
    operations = _deduplicate_target_copies(
        transfer.operations, source_tensors, target_tensors
    )
    return TransferPlan(
        model_id=transfer.model_id,
        revision=transfer.revision,
        operations=operations,
        source_executors=_build_executor_plans(source_manifests, operations, "source"),
        target_executors=_build_executor_plans(target_manifests, operations, "target"),
        pipeline_routes=_build_pipeline_routes(operations),
    )


def plan_runtime_transfer_to_local_target(
    source_manifests: Sequence[RuntimeManifest],
    target_manifest: RuntimeManifest,
) -> TransferPlan:
    """Plan one target executor while retaining the full source snapshot."""

    source_tensors, source_fragments = _collect_manifests(source_manifests, "source")
    target_tensors, target_fragments = _collect_manifests((target_manifest,), "target")
    source = source_manifests[0]
    if source.model_id != target_manifest.model_id:
        raise ValueError("source and target model_id differ")
    if source.revision != target_manifest.revision:
        raise ValueError("source and target revision differ")

    transfer = _plan_transfer(
        source.model_id,
        source.revision,
        source_tensors,
        source_fragments,
        target_tensors,
        target_fragments,
        local_target=True,
    )
    operations = _deduplicate_target_copies(
        transfer.operations, source_tensors, target_tensors
    )
    target_executors = _build_executor_plans((target_manifest,), operations, "target")
    if len(target_executors) != 1:
        raise ValueError("local target manifest must describe exactly one executor")
    return TransferPlan(
        model_id=transfer.model_id,
        revision=transfer.revision,
        operations=operations,
        source_executors=_build_executor_plans(source_manifests, operations, "source"),
        target_executors=target_executors,
        pipeline_routes=_build_pipeline_routes(operations),
    )


def plan_stored_transfer(
    source_manifest: WeightManifest,
    target_manifests: Sequence[RuntimeManifest],
) -> TransferPlan:
    target_tensors, target_fragments = _collect_manifests(target_manifests, "target")
    target = target_manifests[0]
    if source_manifest.model_id != target.model_id:
        raise ValueError("source and target model_id differ")
    if source_manifest.revision != target.revision:
        raise ValueError("source and target revision differ")
    source_tensors = {tensor.tensor_id: tensor for tensor in source_manifest.tensors}
    transfer = _plan_transfer(
        source_manifest.model_id,
        source_manifest.revision,
        source_tensors,
        source_manifest.fragments,
        target_tensors,
        target_fragments,
    )
    operations = _deduplicate_target_copies(
        transfer.operations, source_tensors, target_tensors
    )
    return TransferPlan(
        model_id=transfer.model_id,
        revision=transfer.revision,
        operations=operations,
        target_executors=_build_executor_plans(target_manifests, operations, "target"),
        pipeline_routes=_build_pipeline_routes(operations),
    )
