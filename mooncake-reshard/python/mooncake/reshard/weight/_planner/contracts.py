from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Generic, Iterable, TypeAlias, TypeVar, cast

from ..._compat import _strict_zip
from ...contracts import (
    LeaseId,
    ParticipantId,
    PlacementFragmentId,
    PlacementId,
    ResourceId,
    RevisionId,
    RuntimeFragmentId,
    RuntimeInstanceId,
    TensorId,
)
from ..manifest import (
    ParallelRank,
    PlacementFragment,
    TensorDescriptor,
    WeightPlacementManifest,
)
from ..storage_manifest import StoredFragment
from .geometry import (
    _box_contains,
    _derive_region_geometry,
    _fragment_itemsize,
    _validate_outer_strides,
)
from .attestation import RuntimeBindingAttestation
from .fragments import (
    BoundWeightFragment,
    ExecutableSourceFragment,
    ExecutableTargetFragment,
    GeometryFragment,
    LogicalSourceFragment,
    LogicalTargetFragment,
)


RuntimeTensorOwner = tuple[tuple[str, int], ...]
_SourceFragmentT = TypeVar("_SourceFragmentT", bound=GeometryFragment)
_TargetFragmentT = TypeVar("_TargetFragmentT", bound=GeometryFragment)


@dataclass(frozen=True)
class RuntimeLeaseSnapshot:
    fragment_id: RuntimeFragmentId
    tensor_id: TensorId
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
    instance_id: RuntimeInstanceId
    placement_id: PlacementId
    participant_id: ParticipantId
    placement_digest: str
    runtime_lease_id: LeaseId | None
    worker_id: str
    rank: ParallelRank
    fragment_ids: tuple[RuntimeFragmentId, ...]
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
class CopyRange(Generic[_SourceFragmentT, _TargetFragmentT]):
    tensor_id: TensorId
    source: _SourceFragmentT
    target: _TargetFragmentT
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
class TransferRegion(Generic[_SourceFragmentT, _TargetFragmentT]):
    tensor_id: TensorId
    source: _SourceFragmentT
    target: _TargetFragmentT
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


LogicalCopyRange: TypeAlias = CopyRange[
    LogicalSourceFragment,
    LogicalTargetFragment,
]
LogicalTransferRegion: TypeAlias = TransferRegion[
    LogicalSourceFragment,
    LogicalTargetFragment,
]
LogicalTransferOperation: TypeAlias = LogicalCopyRange | LogicalTransferRegion

ExecutableCopyRange: TypeAlias = CopyRange[
    ExecutableSourceFragment,
    ExecutableTargetFragment,
]
ExecutableTransferRegion: TypeAlias = TransferRegion[
    ExecutableSourceFragment,
    ExecutableTargetFragment,
]
ExecutableTransferOperation: TypeAlias = ExecutableCopyRange | ExecutableTransferRegion

LiveTransferOperation: TypeAlias = (
    CopyRange[BoundWeightFragment, BoundWeightFragment]
    | TransferRegion[BoundWeightFragment, BoundWeightFragment]
)
StoredLoadOperation: TypeAlias = (
    CopyRange[StoredFragment, BoundWeightFragment]
    | TransferRegion[StoredFragment, BoundWeightFragment]
)
ExecutorKey: TypeAlias = tuple[
    RuntimeInstanceId,
    PlacementId,
    ParticipantId,
    str,
    LeaseId,
    str,
    ParallelRank,
]


def _is_canonical_operation(value: object) -> bool:
    return isinstance(value, (CopyRange, TransferRegion))


def _validate_executable_operation(value: object) -> None:
    if not _is_canonical_operation(value):
        raise ValueError("transfer plan requires a canonical transfer operation")
    operation = cast(ExecutableTransferOperation, value)
    if not isinstance(operation.target, BoundWeightFragment):
        raise ValueError("executable transfer plan target must be runtime-bound")
    if not isinstance(operation.source, (BoundWeightFragment, StoredFragment)):
        raise ValueError("executable transfer plan source must be runtime-bound")
    operation.validate_bounds()


def _validate_logical_operation(
    value: object,
    source_has_placement: bool,
) -> None:
    if not _is_canonical_operation(value):
        raise ValueError(
            "logical transfer plan requires a canonical transfer operation"
        )
    operation = cast(LogicalTransferOperation, value)
    if not isinstance(operation.target, PlacementFragment):
        raise ValueError("logical transfer plan target must be a placement")
    if source_has_placement:
        if not isinstance(operation.source, PlacementFragment):
            raise ValueError("logical transfer plan source must be a placement")
    elif not isinstance(operation.source, StoredFragment):
        raise ValueError("logical transfer plan source has no placement")
    operation.validate_bounds()


def _validate_source_target_allocations_do_not_overlap(
    operations: tuple[ExecutableTransferOperation, ...],
) -> None:
    source_allocations: set[tuple[RuntimeInstanceId, str, str, int, int]] = set()
    target_allocations: set[tuple[RuntimeInstanceId, str, str, int, int]] = set()
    for operation in operations:
        if isinstance(operation.source, BoundWeightFragment):
            source_allocations.add(
                (
                    operation.source.instance_id,
                    operation.source.worker_id,
                    operation.source.device,
                    operation.source.binding.storage_address,
                    operation.source.binding.storage_nbytes,
                )
            )
        if isinstance(operation.target, BoundWeightFragment):
            target_allocations.add(
                (
                    operation.target.instance_id,
                    operation.target.worker_id,
                    operation.target.device,
                    operation.target.binding.storage_address,
                    operation.target.binding.storage_nbytes,
                )
            )

    for source in source_allocations:
        source_space = source[:3]
        source_start, source_size = source[3:]
        source_end = source_start + source_size
        for target in target_allocations:
            if target[:3] != source_space:
                continue
            target_start, target_size = target[3:]
            target_end = target_start + target_size
            if source_start < target_end and target_start < source_end:
                raise ValueError(
                    "source and target runtime storage allocations overlap in "
                    f"address space {source_space}; in-place reshard is unsupported"
                )


def _validate_execution_provenance(
    *,
    resource_id: ResourceId,
    revision: RevisionId,
    weight_generation: int,
    operations: tuple[ExecutableTransferOperation, ...],
) -> None:
    """Require live execution fragments to come from verified bindings."""

    for operation in operations:
        fragments: tuple[tuple[str, BoundWeightFragment], ...] = (
            ("target", operation.target),
        )
        if isinstance(operation.source, BoundWeightFragment):
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
    resource_id: ResourceId,
    revision: RevisionId,
    weight_generation: int,
    operations: tuple[ExecutableTransferOperation, ...],
    executors: tuple[ExecutorTransferPlan, ...],
    side: str,
) -> None:
    """Validate live executor routing against attested operation fragments."""

    fragments = tuple(
        operation.source if side == "source" else operation.target
        for operation in operations
    )
    has_stored_source = any(
        isinstance(fragment, StoredFragment) for fragment in fragments
    )
    if has_stored_source:
        if side != "source" or not all(
            isinstance(fragment, StoredFragment) for fragment in fragments
        ):
            raise ValueError("transfer plan mixes stored and live source fragments")
        if executors:
            raise ValueError("stored source must not have live executor provenance")
        return
    live_fragments = tuple(
        fragment for fragment in fragments if isinstance(fragment, BoundWeightFragment)
    )
    if len(live_fragments) != len(fragments):
        raise ValueError(f"transfer plan {side} fragment is not runtime-bound")
    if not executors:
        return

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
    resource_id: ResourceId
    revision: RevisionId
    weight_generation: int
    target_placement: WeightPlacementManifest
    operations: tuple[ExecutableTransferOperation, ...]
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
        if type(self.weight_generation) is not int or self.weight_generation < 0:
            raise ValueError("transfer plan weight_generation must be non-negative")
        for operation in self.operations:
            _validate_executable_operation(operation)
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
        if not isinstance(self.target_placement, WeightPlacementManifest):
            raise ValueError("transfer plan target placement is invalid")
        if (
            self.target_placement.resource_id != self.resource_id
            or self.target_placement.revision != self.revision
            or self.target_placement.weight_generation != self.weight_generation
        ):
            raise ValueError("transfer plan target placement identity differs")
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
        _validate_source_target_allocations_do_not_overlap(self.operations)
        # Re-check the complete target placement at the public executable-plan
        # boundary. Binding output can be serialized or reconstructed, so the
        # executor snapshot alone cannot be trusted as a coverage proof.
        from .validation import (
            _validate_bound_target_coverage,
            _validate_target_physical_ranges,
        )

        _validate_target_physical_ranges(self.operations)
        _validate_bound_target_coverage(self)
        for executor in (*self.source_executors, *self.target_executors):
            if any(
                index >= len(self.operations) for index in executor.operation_indices
            ):
                raise ValueError("executor operation index is out of range")
        route_indices: list[int] = []
        route_keys: set[tuple[int | None, int]] = set()
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
    def regions(self) -> tuple[ExecutableTransferOperation, ...]:
        return self.operations


@dataclass(frozen=True)
class PlacementExecutorPlan:
    placement_id: PlacementId
    participant_id: ParticipantId
    rank: ParallelRank
    placement_fragment_ids: tuple[PlacementFragmentId, ...]
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
        if not self.operation_indices:
            raise ValueError("placement executor operation indices must not be empty")


@dataclass(frozen=True)
class LogicalTransferPlan:
    resource_id: ResourceId
    revision: RevisionId
    source_placement: WeightPlacementManifest | None
    target_placement: WeightPlacementManifest
    source_tensors: tuple[TensorDescriptor, ...]
    target_tensors: tuple[TensorDescriptor, ...]
    operations: tuple[LogicalTransferOperation, ...]
    source_executors: tuple[PlacementExecutorPlan, ...] = ()
    target_executors: tuple[PlacementExecutorPlan, ...] = ()
    pipeline_routes: tuple[PipelineRouteGroup, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_tensors", tuple(self.source_tensors))
        object.__setattr__(self, "target_tensors", tuple(self.target_tensors))
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(self, "source_executors", tuple(self.source_executors))
        object.__setattr__(self, "target_executors", tuple(self.target_executors))
        object.__setattr__(self, "pipeline_routes", tuple(self.pipeline_routes))
        if not self.resource_id or not self.revision:
            raise ValueError("logical transfer plan identifiers must not be empty")
        if not isinstance(self.target_placement, WeightPlacementManifest):
            raise ValueError("logical transfer plan target placement is invalid")
        if self.source_placement is not None and not isinstance(
            self.source_placement, WeightPlacementManifest
        ):
            raise ValueError("logical transfer plan source placement is invalid")
        for side, placement in (
            ("source", self.source_placement),
            ("target", self.target_placement),
        ):
            if placement is not None and (
                placement.resource_id != self.resource_id
                or placement.revision != self.revision
            ):
                raise ValueError(
                    f"logical transfer plan {side} placement identity differs"
                )
        if self.source_placement is not None and (
            self.source_placement.weight_generation
            != self.target_placement.weight_generation
        ):
            raise ValueError(
                "logical transfer plan source and target weight_generation differs"
            )
        for operation in self.operations:
            _validate_logical_operation(operation, self.source_placement is not None)
        if not all(
            isinstance(executor, PlacementExecutorPlan)
            for executor in (*self.source_executors, *self.target_executors)
        ):
            raise ValueError(
                "logical transfer plan has invalid canonical executor metadata"
            )
        if not all(
            isinstance(route, PipelineRouteGroup) for route in self.pipeline_routes
        ):
            raise ValueError(
                "logical transfer plan has invalid pipeline route metadata"
            )
        for executor in (*self.source_executors, *self.target_executors):
            if any(
                index >= len(self.operations) for index in executor.operation_indices
            ):
                raise ValueError("logical executor operation index is out of range")
        route_indices: list[int] = [
            index for route in self.pipeline_routes for index in route.operation_indices
        ]
        route_keys: set[tuple[int | None, int]] = set()
        for route in self.pipeline_routes:
            key = (route.source_pp, route.target_pp)
            if key in route_keys:
                raise ValueError(
                    "logical transfer plan has duplicate pipeline route groups"
                )
            route_keys.add(key)
        if self.pipeline_routes and sorted(route_indices) != list(
            range(len(self.operations))
        ):
            raise ValueError(
                "logical pipeline routes must cover every operation exactly once"
            )
        # Keep construction strict without making the data-contract module own
        # planner geometry validation.
        from .validation import _validate_logical_target_coverage

        _validate_logical_target_coverage(self)

    @property
    def total_bytes(self) -> int:
        return sum(operation.total_bytes for operation in self.operations)

    @property
    def source_placement_id(self) -> PlacementId | None:
        return (
            self.source_placement.placement_id
            if self.source_placement is not None
            else None
        )

    @property
    def target_placement_id(self) -> PlacementId:
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
    "ExecutableCopyRange",
    "ExecutableTransferOperation",
    "ExecutableTransferRegion",
    "LiveTransferOperation",
    "LogicalCopyRange",
    "LogicalTransferOperation",
    "LogicalTransferRegion",
    "StoredLoadOperation",
    "TransferPlan",
    "TransferRegion",
]
