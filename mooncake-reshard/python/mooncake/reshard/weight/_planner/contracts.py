from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Any, Iterable, Union

from ..._compat import _strict_zip
from ..manifest import (
    ParallelRank,
    PlacementFragment,
    RuntimeBindingFragment,
    TensorDescriptor,
    WeightPlacementManifest,
)
from .geometry import (
    _box_contains,
    _derive_region_geometry,
    _fragment_itemsize,
    _validate_outer_strides,
)
from .attestation import RuntimeBindingAttestation


RuntimeTensorOwner = tuple[tuple[str, int], ...]
_MAX_U64 = (1 << 64) - 1


@dataclass(frozen=True)
class BoundWeightFragment:
    """Planner-local physical view of one logical placement fragment."""

    placement: PlacementFragment
    binding: RuntimeBindingFragment
    instance_id: str
    runtime_lease_id: str
    lease_generation: int
    owner: Any = field(default=None, compare=False, repr=False)
    attestation: RuntimeBindingAttestation | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.placement, PlacementFragment):
            raise ValueError("bound fragment placement is invalid")
        if not isinstance(self.binding, RuntimeBindingFragment):
            raise ValueError("bound fragment runtime binding is invalid")
        if self.placement.placement_fragment_id != self.binding.placement_fragment_id:
            raise ValueError("bound fragment placement identity differs")
        if self.placement.nbytes != self.binding.nbytes:
            raise ValueError("bound fragment byte size differs")
        for name in ("instance_id", "runtime_lease_id"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"bound fragment {name} must be non-empty")
        if (
            type(self.lease_generation) is not int
            or self.lease_generation < 0
            or self.lease_generation > _MAX_U64
        ):
            raise ValueError(
                "bound fragment lease_generation must fit in an unsigned 64-bit integer"
            )
        if self.owner is not self.binding.owner:
            raise ValueError("bound fragment owner differs from runtime binding")
        if self.attestation is not None and not isinstance(
            self.attestation, RuntimeBindingAttestation
        ):
            raise ValueError("bound fragment runtime attestation is invalid")

    @property
    def placement_fragment_id(self) -> str:
        return self.placement.placement_fragment_id

    @property
    def fragment_id(self) -> str:
        return self.binding.fragment_id

    @property
    def tensor_id(self) -> str:
        return self.placement.tensor_id

    @property
    def global_offset(self) -> tuple[int, ...]:
        return self.placement.global_offset

    @property
    def local_shape(self) -> tuple[int, ...]:
        return self.placement.local_shape

    @property
    def nbytes(self) -> int:
        return self.binding.nbytes

    @property
    def rank(self) -> ParallelRank:
        return self.placement.rank

    @property
    def aliases(self) -> tuple[str, ...]:
        return self.placement.aliases

    @property
    def address(self) -> int:
        return self.binding.address

    @property
    def worker_id(self) -> str:
        return self.binding.worker_id

    @property
    def endpoint(self) -> str:
        return self.binding.endpoint

    @property
    def device(self) -> str:
        return self.binding.device


SourceFragment = Union[
    BoundWeightFragment,
    PlacementFragment,
]
TargetFragment = Union[BoundWeightFragment, PlacementFragment]


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
    device: str
    lease_generation: int

    @classmethod
    def from_fragment(cls, fragment: BoundWeightFragment) -> RuntimeLeaseSnapshot:
        return cls(
            fragment_id=fragment.fragment_id,
            tensor_id=fragment.tensor_id,
            global_offset=fragment.global_offset,
            local_shape=fragment.local_shape,
            address=fragment.address,
            nbytes=fragment.nbytes,
            worker_id=fragment.worker_id,
            endpoint=fragment.endpoint,
            device=fragment.device,
            lease_generation=fragment.lease_generation,
        )


@dataclass(frozen=True)
class ExecutorTransferPlan:
    instance_id: str
    placement_id: str
    participant_id: str
    placement_digest: str
    runtime_lease_id: str | None
    worker_id: str
    rank: ParallelRank
    fragment_ids: tuple[str, ...]
    fragment_leases: tuple[RuntimeLeaseSnapshot, ...]
    operation_indices: tuple[int, ...]
    attestation: RuntimeBindingAttestation | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "fragment_ids", tuple(self.fragment_ids))
        object.__setattr__(self, "fragment_leases", tuple(self.fragment_leases))
        object.__setattr__(self, "operation_indices", tuple(self.operation_indices))
        if (
            not self.instance_id
            or not self.placement_id
            or not self.participant_id
            or not self.worker_id
            or not self.fragment_ids
        ):
            raise ValueError("executor plan identifiers must not be empty")
        if len(self.placement_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.placement_digest
        ):
            raise ValueError("executor plan placement digest must be SHA-256")
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
        if self.attestation is not None and not isinstance(
            self.attestation, RuntimeBindingAttestation
        ):
            raise ValueError("executor plan runtime attestation is invalid")


@dataclass(frozen=True)
class PipelineRouteGroup:
    source_pp: int
    target_pp: int
    operation_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_indices", tuple(self.operation_indices))
        if type(self.source_pp) is not int or self.source_pp < 0:
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
                for count, stride in _strict_zip(
                    self.outer_loop_counts, self.source_strides
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
                for count, stride in _strict_zip(
                    self.outer_loop_counts, self.target_strides
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
                    for index, stride in _strict_zip(indices, self.source_strides)
                ),
                self.target_base_offset
                + sum(
                    index * stride
                    for index, stride in _strict_zip(indices, self.target_strides)
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


TransferOperation = Union[CopyRange, TransferRegion]


def _validate_execution_provenance(
    *,
    resource_id: str,
    revision: str,
    weight_generation: int,
    operations: tuple[TransferOperation, ...],
) -> None:
    """Require live execution fragments to come from verified bindings."""

    for operation in operations:
        fragments: tuple[tuple[str, BoundWeightFragment], ...] = (
            ("target", operation.target),
        )
        fragments = (("source", operation.source), *fragments)
        for side, fragment in fragments:
            attestation = fragment.attestation
            if not isinstance(attestation, RuntimeBindingAttestation):
                raise ValueError(
                    f"transfer plan {side} fragment lacks an attested runtime binding"
                )
            placement = attestation.placement
            if (
                placement.resource_id != resource_id
                or placement.revision != revision
                or placement.weight_generation != weight_generation
            ):
                raise ValueError(
                    f"transfer plan identity differs from {side} placement"
                )
            if not attestation.validates(fragment.placement, fragment.binding):
                raise ValueError(
                    f"transfer plan {side} fragment differs from attested runtime binding"
                )


def _validate_executor_provenance(
    *,
    resource_id: str,
    revision: str,
    weight_generation: int,
    operations: tuple[TransferOperation, ...],
    executors: tuple[ExecutorTransferPlan, ...],
    side: str,
) -> None:
    """Validate live executor routing against attested operation fragments."""

    fragments = tuple(
        operation.source if side == "source" else operation.target
        for operation in operations
    )
    if not all(isinstance(fragment, BoundWeightFragment) for fragment in fragments):
        raise ValueError(f"transfer plan {side} fragment is not runtime-bound")
    live_fragments: tuple[BoundWeightFragment, ...] = tuple(fragments)
    if not executors:
        return

    ExecutorKey = tuple[str, str, str, str, str, str, ParallelRank]
    expected_indices: dict[ExecutorKey, list[int]] = {}
    for index, fragment in enumerate(live_fragments):
        attestation = fragment.attestation
        if not isinstance(attestation, RuntimeBindingAttestation):
            raise ValueError(f"transfer plan {side} fragment lacks runtime attestation")
        binding = attestation.binding
        placement = attestation.placement
        key: ExecutorKey = (
            binding.instance_id,
            placement.placement_id,
            binding.participant_id,
            placement.digest,
            binding.lease_id,
            fragment.worker_id,
            fragment.rank,
        )
        expected_indices.setdefault(key, []).append(index)

    active_placement_ids = {
        fragment.attestation.placement.placement_id
        for fragment in live_fragments
        if isinstance(fragment.attestation, RuntimeBindingAttestation)
    }
    actual_indices: dict[ExecutorKey, list[int]] = {}
    for executor in executors:
        if executor.runtime_lease_id is None:
            raise ValueError(f"{side} executor is missing runtime lease provenance")
        key: ExecutorKey = (
            executor.instance_id,
            executor.placement_id,
            executor.participant_id,
            executor.placement_digest,
            executor.runtime_lease_id,
            executor.worker_id,
            executor.rank,
        )
        if key in actual_indices:
            raise ValueError(f"transfer plan has duplicate {side} executor provenance")
        attestation = executor.attestation
        if not isinstance(attestation, RuntimeBindingAttestation):
            raise ValueError(f"{side} executor lacks runtime attestation")
        placement = attestation.placement
        binding = attestation.binding
        if (
            placement.resource_id != resource_id
            or placement.revision != revision
            or placement.weight_generation != weight_generation
            or placement.placement_id not in active_placement_ids
        ):
            raise ValueError(f"{side} executor attestation identity differs")
        expected_key: ExecutorKey = (
            binding.instance_id,
            placement.placement_id,
            binding.participant_id,
            placement.digest,
            binding.lease_id,
            executor.worker_id,
            executor.rank,
        )
        if key != expected_key:
            raise ValueError(f"{side} executor provenance differs from attestation")
        actual_indices[key] = list(executor.operation_indices)
        expected_fragment_leases = tuple(
            sorted(
                (
                    RuntimeLeaseSnapshot(
                        fragment_id=runtime.fragment_id,
                        tensor_id=placement_fragment.tensor_id,
                        global_offset=placement_fragment.global_offset,
                        local_shape=placement_fragment.local_shape,
                        address=runtime.address,
                        nbytes=runtime.nbytes,
                        worker_id=runtime.worker_id,
                        endpoint=runtime.endpoint,
                        device=runtime.device,
                        lease_generation=binding.generation,
                    )
                    for placement_fragment, runtime in attestation.worker_fragment_pairs(
                        executor.worker_id
                    )
                ),
                key=lambda item: item.fragment_id,
            )
        )
        if executor.fragment_leases != expected_fragment_leases:
            raise ValueError(f"{side} executor fragment provenance differs")

    for key, actual in actual_indices.items():
        expected = expected_indices.get(key)
        if expected is None:
            if actual:
                raise ValueError(f"{side} executor provenance differs from operations")
            continue
        if sorted(actual) != sorted(expected):
            raise ValueError(f"{side} executor provenance differs from operations")
    if set(expected_indices) - set(actual_indices):
        raise ValueError(f"{side} executor provenance differs from operations")


@dataclass(frozen=True)
class TransferPlan:
    resource_id: str
    revision: str
    weight_generation: int
    operations: tuple[TransferOperation, ...]
    source_executors: tuple[ExecutorTransferPlan, ...] = ()
    target_executors: tuple[ExecutorTransferPlan, ...] = ()
    pipeline_routes: tuple[PipelineRouteGroup, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(self, "source_executors", tuple(self.source_executors))
        object.__setattr__(self, "target_executors", tuple(self.target_executors))
        object.__setattr__(self, "pipeline_routes", tuple(self.pipeline_routes))
        if not self.resource_id or not self.revision:
            raise ValueError("transfer plan identifiers must not be empty")
        if any(
            not isinstance(operation.target, BoundWeightFragment)
            for operation in self.operations
        ):
            raise ValueError("executable transfer plan target must be runtime-bound")
        if any(
            not isinstance(operation.source, BoundWeightFragment)
            for operation in self.operations
        ):
            raise ValueError("executable transfer plan source must be runtime-bound")
        if type(self.weight_generation) is not int or self.weight_generation < 0:
            raise ValueError("transfer plan weight_generation must be non-negative")
        if not all(
            isinstance(executor, ExecutorTransferPlan)
            for executor in (*self.source_executors, *self.target_executors)
        ):
            raise ValueError("transfer plan has invalid canonical executor metadata")
        _validate_execution_provenance(
            resource_id=self.resource_id,
            revision=self.revision,
            weight_generation=self.weight_generation,
            operations=self.operations,
        )
        _validate_executor_provenance(
            resource_id=self.resource_id,
            revision=self.revision,
            weight_generation=self.weight_generation,
            operations=self.operations,
            executors=self.source_executors,
            side="source",
        )
        _validate_executor_provenance(
            resource_id=self.resource_id,
            revision=self.revision,
            weight_generation=self.weight_generation,
            operations=self.operations,
            executors=self.target_executors,
            side="target",
        )
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
    participant_id: str
    rank: ParallelRank
    placement_fragment_ids: tuple[str, ...]
    operation_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "placement_fragment_ids", tuple(self.placement_fragment_ids)
        )
        object.__setattr__(self, "operation_indices", tuple(self.operation_indices))
        if (
            not self.placement_id
            or not self.participant_id
            or not self.placement_fragment_ids
        ):
            raise ValueError("placement executor identifiers must not be empty")
        if len(self.placement_fragment_ids) != len(set(self.placement_fragment_ids)):
            raise ValueError("placement executor has duplicate fragment IDs")
        if any(type(index) is not int or index < 0 for index in self.operation_indices):
            raise ValueError(
                "placement executor operation indices must be non-negative integers"
            )


@dataclass(frozen=True)
class LogicalTransferPlan:
    resource_id: str
    revision: str
    source_placement: WeightPlacementManifest
    target_placement: WeightPlacementManifest
    source_tensors: tuple[TensorDescriptor, ...]
    target_tensors: tuple[TensorDescriptor, ...]
    operations: tuple[TransferOperation, ...]
    source_executors: tuple[PlacementExecutorPlan, ...] = ()
    target_executors: tuple[PlacementExecutorPlan, ...] = ()
    pipeline_routes: tuple[PipelineRouteGroup, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "source_tensors",
            "target_tensors",
            "operations",
            "source_executors",
            "target_executors",
            "pipeline_routes",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.resource_id or not self.revision:
            raise ValueError("logical transfer plan identifiers must not be empty")
        if not isinstance(self.target_placement, WeightPlacementManifest):
            raise ValueError("logical transfer plan target placement is invalid")
        if not isinstance(self.source_placement, WeightPlacementManifest):
            raise ValueError("logical transfer plan source placement is invalid")
        for side, placement in (
            ("source", self.source_placement),
            ("target", self.target_placement),
        ):
            if (
                placement.resource_id != self.resource_id
                or placement.revision != self.revision
            ):
                raise ValueError(
                    f"logical transfer plan {side} placement identity differs"
                )
        if any(
            not isinstance(operation.source, PlacementFragment)
            for operation in self.operations
        ):
            raise ValueError("logical transfer plan source must be a placement")
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
    def source_placement_id(self) -> str | None:
        return (
            self.source_placement.placement_id
            if self.source_placement is not None
            else None
        )

    @property
    def target_placement_id(self) -> str:
        return self.target_placement.placement_id


__all__ = [
    "BoundWeightFragment",
    "CopyRange",
    "ExecutorTransferPlan",
    "LogicalTransferPlan",
    "PipelineRouteGroup",
    "PlacementExecutorPlan",
    "RuntimeLeaseSnapshot",
    "RuntimeTensorOwner",
    "SourceFragment",
    "TargetFragment",
    "TransferOperation",
    "TransferPlan",
    "TransferRegion",
]
