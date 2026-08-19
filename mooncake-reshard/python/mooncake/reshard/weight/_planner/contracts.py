from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from types import MappingProxyType
from typing import (
    ClassVar,
    Generic,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)

from ..._compat import _strict_zip
from ..._typing import TypeAlias
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
from ...geometry import box_contains as _box_contains
from ..manifest import (
    ParallelRank,
    PlacementFragment,
    TensorDescriptor,
    WeightPlacementManifest,
)
from ..storage_manifest import (
    StoredFragment,
    StoredManifestIdentity,
    WeightManifest,
    validate_weight_manifest_snapshot,
)
from .geometry import (
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
    runtime_lease_id: Optional[LeaseId]
    worker_id: str
    rank: ParallelRank
    fragment_ids: tuple[RuntimeFragmentId, ...]
    fragment_leases: tuple[RuntimeLeaseSnapshot, ...]
    attestation: Optional[RuntimeBindingAttestation] = field(
        default=None,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "fragment_ids", tuple(self.fragment_ids))
        object.__setattr__(self, "fragment_leases", tuple(self.fragment_leases))
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
        if self.attestation is not None and not isinstance(
            self.attestation, RuntimeBindingAttestation
        ):
            raise ValueError("executor plan runtime attestation is invalid")


@dataclass(frozen=True)
class PipelineRouteGroup:
    source_pp: Optional[int]
    source_pipeline_stage_id: Optional[int]
    target_pp: int
    target_pipeline_stage_id: Optional[int]
    operation_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_indices", tuple(self.operation_indices))
        if self.source_pp is not None and (
            type(self.source_pp) is not int or self.source_pp < 0
        ):
            raise ValueError("pipeline route source_pp must be non-negative")
        if type(self.target_pp) is not int or self.target_pp < 0:
            raise ValueError("pipeline route target_pp must be non-negative")
        for name in (
            "source_pipeline_stage_id",
            "target_pipeline_stage_id",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"pipeline route {name} must be non-negative")
        if any(type(index) is not int or index < 0 for index in self.operation_indices):
            raise ValueError("pipeline route indices must be non-negative integers")
        if len(self.operation_indices) != len(set(self.operation_indices)):
            raise ValueError("pipeline route has duplicate operation indices")


@dataclass(frozen=True)
class PlanningLimits:
    """Fail-closed limits for canonical N-D planning and lowering."""

    max_transfer_regions: int = 1_000_000
    max_segments_per_region: int = 1_000_000
    max_total_lowered_segments: int = 10_000_000

    def __post_init__(self) -> None:
        for name in (
            "max_transfer_regions",
            "max_segments_per_region",
            "max_total_lowered_segments",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"planning limit {name} must be a positive integer")


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

    def iter_segments(self, *, max_segments: int) -> Iterable[tuple[int, int, int]]:
        if type(max_segments) is not int or max_segments <= 0:
            raise ValueError("max_segments must be a positive integer")
        if self.segment_count > max_segments:
            raise ValueError(
                "transfer region exceeds max_segments: "
                f"{self.segment_count} > {max_segments}"
            )
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


LogicalTransferRegion: TypeAlias = TransferRegion[
    LogicalSourceFragment,
    LogicalTargetFragment,
]
LogicalTransferOperation: TypeAlias = LogicalTransferRegion

ExecutableTransferRegion: TypeAlias = TransferRegion[
    ExecutableSourceFragment,
    ExecutableTargetFragment,
]
ExecutableTransferOperation: TypeAlias = ExecutableTransferRegion

LiveTransferOperation: TypeAlias = TransferRegion[
    BoundWeightFragment, BoundWeightFragment
]
StoredLoadOperation: TypeAlias = TransferRegion[StoredFragment, BoundWeightFragment]
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
    return isinstance(value, TransferRegion)


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
    source_allocations: dict[
        tuple[RuntimeInstanceId, str, str], set[tuple[int, int]]
    ] = {}
    target_allocations: dict[
        tuple[RuntimeInstanceId, str, str], set[tuple[int, int]]
    ] = {}
    for operation in operations:
        if isinstance(operation.source, BoundWeightFragment):
            source_allocations.setdefault(
                (
                    operation.source.instance_id,
                    operation.source.worker_id,
                    operation.source.device,
                ),
                set(),
            ).add(
                (
                    operation.source.binding.storage_address,
                    operation.source.binding.storage_nbytes,
                )
            )
        if isinstance(operation.target, BoundWeightFragment):
            target_allocations.setdefault(
                (
                    operation.target.instance_id,
                    operation.target.worker_id,
                    operation.target.device,
                ),
                set(),
            ).add(
                (
                    operation.target.binding.storage_address,
                    operation.target.binding.storage_nbytes,
                )
            )

    for address_space in sorted(set(source_allocations) & set(target_allocations)):
        sources = sorted(source_allocations[address_space])
        targets = sorted(target_allocations[address_space])
        source_index = 0
        target_index = 0
        while source_index < len(sources) and target_index < len(targets):
            source_start, source_size = sources[source_index]
            target_start, target_size = targets[target_index]
            source_end = source_start + source_size
            target_end = target_start + target_size
            if source_start < target_end and target_start < source_end:
                raise ValueError(
                    "source and target runtime storage allocations overlap in "
                    f"address space {address_space}; in-place reshard is unsupported"
                )
            if source_end <= target_end:
                source_index += 1
            else:
                target_index += 1


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
    operation_views: _OperationViews,
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
        actual_indices[key] = list(
            operation_views.operation_indices_for(executor, side)
        )
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
    _operation_views: ClassVar[_OperationViews]

    resource_id: ResourceId
    revision: RevisionId
    weight_generation: int
    target_placement: WeightPlacementManifest
    operations: tuple[ExecutableTransferOperation, ...]
    source_manifest_identity: Optional[StoredManifestIdentity] = None
    planning_limits: PlanningLimits = field(default_factory=PlanningLimits)
    source_executors: tuple[ExecutorTransferPlan, ...] = ()
    target_executors: tuple[ExecutorTransferPlan, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(self, "source_executors", tuple(self.source_executors))
        object.__setattr__(self, "target_executors", tuple(self.target_executors))
        if not self.resource_id or not self.revision:
            raise ValueError("transfer plan identifiers must not be empty")
        if type(self.weight_generation) is not int or self.weight_generation < 0:
            raise ValueError("transfer plan weight_generation must be non-negative")
        has_stored_source = any(
            isinstance(operation.source, StoredFragment)
            for operation in self.operations
        )
        if has_stored_source:
            if self.source_manifest_identity is None:
                raise ValueError("stored transfer plan lacks source manifest identity")
            if (
                self.source_manifest_identity.resource_id != self.resource_id
                or self.source_manifest_identity.revision != self.revision
                or self.source_manifest_identity.weight_generation
                != self.weight_generation
            ):
                raise ValueError("transfer plan source manifest identity differs")
        elif self.source_manifest_identity is not None:
            raise ValueError("runtime transfer plan has source manifest identity")
        if not isinstance(self.planning_limits, PlanningLimits):
            raise ValueError("transfer plan planning_limits is invalid")
        if len(self.operations) > self.planning_limits.max_transfer_regions:
            raise ValueError("transfer plan exceeds max_transfer_regions")
        total_lowered_segments = 0
        for operation in self.operations:
            _validate_executable_operation(operation)
            if operation.segment_count > self.planning_limits.max_segments_per_region:
                raise ValueError("transfer plan exceeds max_segments_per_region")
            total_lowered_segments += operation.segment_count
            if total_lowered_segments > self.planning_limits.max_total_lowered_segments:
                raise ValueError("transfer plan exceeds max_total_lowered_segments")
        if not all(
            isinstance(executor, ExecutorTransferPlan)
            for executor in (*self.source_executors, *self.target_executors)
        ):
            raise ValueError("transfer plan has invalid canonical executor metadata")
        object.__setattr__(
            self,
            "_operation_views",
            _build_operation_views(
                self.operations,
                self.source_executors,
                self.target_executors,
            ),
        )
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
            operation_views=self._operation_views,
            side="source",
        )
        _validate_executor_provenance(
            resource_id=self.resource_id,
            revision=self.revision,
            weight_generation=self.weight_generation,
            operations=self.operations,
            executors=self.target_executors,
            operation_views=self._operation_views,
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

    @property
    def total_bytes(self) -> int:
        return sum(operation.total_bytes for operation in self.operations)

    @property
    def regions(self) -> tuple[ExecutableTransferOperation, ...]:
        return self.operations

    @property
    def pipeline_routes(self) -> tuple[PipelineRouteGroup, ...]:
        return self._operation_views.pipeline_routes

    def operation_indices_for_executor(
        self,
        executor: ExecutorTransferPlan,
        side: str,
    ) -> tuple[int, ...]:
        executors = self.source_executors if side == "source" else self.target_executors
        if side not in ("source", "target"):
            raise ValueError(f"invalid executor side: {side}")
        if executor not in executors:
            raise ValueError(f"{side} executor is not part of this transfer plan")
        return self._operation_views.operation_indices_for(executor, side)

    def __getstate__(self) -> dict[str, object]:
        return {
            name: value
            for name, value in self.__dict__.items()
            if name != "_operation_views"
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        for name, value in state.items():
            object.__setattr__(self, name, value)
        self.__post_init__()


@dataclass(frozen=True)
class PlacementExecutorPlan:
    placement_id: PlacementId
    participant_id: ParticipantId
    rank: ParallelRank
    placement_fragment_ids: tuple[PlacementFragmentId, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "placement_fragment_ids", tuple(self.placement_fragment_ids)
        )
        if (
            not self.placement_id
            or not self.participant_id
            or not self.placement_fragment_ids
        ):
            raise ValueError("placement executor identifiers must not be empty")
        if len(self.placement_fragment_ids) != len(set(self.placement_fragment_ids)):
            raise ValueError("placement executor has duplicate fragment IDs")


RuntimeExecutorProjectionKey: TypeAlias = tuple[
    RuntimeInstanceId,
    PlacementId,
    ParticipantId,
    str,
    Optional[LeaseId],
    str,
    ParallelRank,
    tuple[RuntimeFragmentId, ...],
]
PlacementExecutorProjectionKey: TypeAlias = tuple[
    PlacementId,
    ParticipantId,
    ParallelRank,
    tuple[PlacementFragmentId, ...],
]
ExecutorProjectionKey: TypeAlias = Union[
    RuntimeExecutorProjectionKey,
    PlacementExecutorProjectionKey,
]
FragmentProjectionKey: TypeAlias = tuple[str, str]


@dataclass(frozen=True)
class _OperationViews:
    source_indices: Mapping[ExecutorProjectionKey, tuple[int, ...]]
    target_indices: Mapping[ExecutorProjectionKey, tuple[int, ...]]
    pipeline_routes: tuple[PipelineRouteGroup, ...]

    def operation_indices_for(
        self,
        executor: Union[ExecutorTransferPlan, PlacementExecutorPlan],
        side: str,
    ) -> tuple[int, ...]:
        if side == "source":
            return self.source_indices.get(_executor_projection_key(executor), ())
        if side == "target":
            return self.target_indices.get(_executor_projection_key(executor), ())
        raise ValueError(f"invalid executor side: {side}")


@dataclass(frozen=True)
class LogicalTransferPlan:
    _operation_views: ClassVar[_OperationViews]

    resource_id: ResourceId
    revision: RevisionId
    source_placement: Optional[WeightPlacementManifest]
    target_placement: WeightPlacementManifest
    source_tensors: tuple[TensorDescriptor, ...]
    target_tensors: tuple[TensorDescriptor, ...]
    operations: tuple[LogicalTransferOperation, ...]
    source_manifest: Optional[WeightManifest] = None
    planning_limits: PlanningLimits = field(default_factory=PlanningLimits)
    source_executors: tuple[PlacementExecutorPlan, ...] = ()
    target_executors: tuple[PlacementExecutorPlan, ...] = ()
    source_manifest_identity: Optional[StoredManifestIdentity] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_tensors", tuple(self.source_tensors))
        object.__setattr__(self, "target_tensors", tuple(self.target_tensors))
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(self, "source_executors", tuple(self.source_executors))
        object.__setattr__(self, "target_executors", tuple(self.target_executors))
        if not self.resource_id or not self.revision:
            raise ValueError("logical transfer plan identifiers must not be empty")
        if not isinstance(self.target_placement, WeightPlacementManifest):
            raise ValueError("logical transfer plan target placement is invalid")
        if not isinstance(self.planning_limits, PlanningLimits):
            raise ValueError("logical transfer plan planning_limits is invalid")
        if len(self.operations) > self.planning_limits.max_transfer_regions:
            raise ValueError("logical transfer plan exceeds max_transfer_regions")
        total_lowered_segments = 0
        if self.source_placement is not None and not isinstance(
            self.source_placement, WeightPlacementManifest
        ):
            raise ValueError("logical transfer plan source placement is invalid")
        if self.source_manifest is not None and not isinstance(
            self.source_manifest, WeightManifest
        ):
            raise ValueError("logical transfer plan source manifest is invalid")
        if (self.source_placement is None) == (self.source_manifest is None):
            raise ValueError(
                "logical transfer plan requires exactly one source provenance"
            )
        if self.source_manifest is not None:
            source_manifest = validate_weight_manifest_snapshot(self.source_manifest)
            object.__setattr__(self, "source_manifest", source_manifest)
            object.__setattr__(
                self,
                "source_manifest_identity",
                source_manifest.manifest_identity,
            )
        else:
            object.__setattr__(self, "source_manifest_identity", None)
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
        if self.source_manifest is not None and (
            self.source_manifest.resource_id != self.resource_id
            or self.source_manifest.revision != self.revision
            or self.source_manifest.weight_generation
            != self.target_placement.weight_generation
        ):
            raise ValueError("logical transfer plan source manifest identity differs")
        for operation in self.operations:
            _validate_logical_operation(operation, self.source_placement is not None)
            if operation.segment_count > self.planning_limits.max_segments_per_region:
                raise ValueError(
                    "logical transfer plan exceeds max_segments_per_region"
                )
            total_lowered_segments += operation.segment_count
            if total_lowered_segments > self.planning_limits.max_total_lowered_segments:
                raise ValueError(
                    "logical transfer plan exceeds max_total_lowered_segments"
                )
        if not all(
            isinstance(executor, PlacementExecutorPlan)
            for executor in (*self.source_executors, *self.target_executors)
        ):
            raise ValueError(
                "logical transfer plan has invalid canonical executor metadata"
            )
        object.__setattr__(
            self,
            "_operation_views",
            _build_operation_views(
                self.operations,
                self.source_executors,
                self.target_executors,
            ),
        )
        # Keep construction strict without making the data-contract module own
        # planner geometry validation.
        from .validation import _validate_logical_target_coverage

        _validate_logical_target_coverage(self)
        self.validate_source_manifest_snapshot()

    @property
    def total_bytes(self) -> int:
        return sum(operation.total_bytes for operation in self.operations)

    @property
    def source_placement_id(self) -> Optional[PlacementId]:
        return (
            self.source_placement.placement_id
            if self.source_placement is not None
            else None
        )

    @property
    def target_placement_id(self) -> PlacementId:
        return self.target_placement.placement_id

    def validate_source_manifest_snapshot(self) -> None:
        """Fail closed if a stored source no longer matches its plan snapshot."""

        if self.source_manifest is None:
            if self.source_manifest_identity is not None:
                raise ValueError("logical plan has unexpected source manifest identity")
            return
        source_manifest = validate_weight_manifest_snapshot(self.source_manifest)
        if self.source_manifest_identity != source_manifest.manifest_identity:
            raise ValueError("logical plan source manifest identity differs")
        source_by_id = {
            fragment.fragment_id: fragment for fragment in source_manifest.fragments
        }
        for operation in self.operations:
            source = operation.source
            if (
                not isinstance(source, StoredFragment)
                or source_by_id.get(source.fragment_id) != source
            ):
                raise ValueError(
                    "logical plan and source manifest fragment snapshots differ"
                )

    @property
    def pipeline_routes(self) -> tuple[PipelineRouteGroup, ...]:
        return self._operation_views.pipeline_routes

    def operation_indices_for_executor(
        self,
        executor: PlacementExecutorPlan,
        side: str,
    ) -> tuple[int, ...]:
        executors = self.source_executors if side == "source" else self.target_executors
        if side not in ("source", "target"):
            raise ValueError(f"invalid executor side: {side}")
        if executor not in executors:
            raise ValueError(f"{side} executor is not part of this logical plan")
        return self._operation_views.operation_indices_for(executor, side)

    def __getstate__(self) -> dict[str, object]:
        return {
            name: value
            for name, value in self.__dict__.items()
            if name != "_operation_views"
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        for name, value in state.items():
            object.__setattr__(self, name, value)
        self.__post_init__()


def _executor_projection_key(
    executor: Union[ExecutorTransferPlan, PlacementExecutorPlan],
) -> ExecutorProjectionKey:
    if isinstance(executor, ExecutorTransferPlan):
        return (
            executor.instance_id,
            executor.placement_id,
            executor.participant_id,
            executor.placement_digest,
            executor.runtime_lease_id,
            executor.worker_id,
            executor.rank,
            executor.fragment_ids,
        )
    return (
        executor.placement_id,
        executor.participant_id,
        executor.rank,
        executor.placement_fragment_ids,
    )


def _fragment_projection_key(
    fragment: GeometryFragment,
) -> Optional[FragmentProjectionKey]:
    if isinstance(fragment, BoundWeightFragment):
        return "runtime", fragment.fragment_id
    if isinstance(fragment, PlacementFragment):
        return "placement", fragment.placement_fragment_id
    return None


def _index_executors_by_fragment(
    executors: Sequence[Union[ExecutorTransferPlan, PlacementExecutorPlan]],
) -> tuple[
    dict[ExecutorProjectionKey, list[int]],
    dict[FragmentProjectionKey, list[ExecutorProjectionKey]],
]:
    indices_by_executor: dict[ExecutorProjectionKey, list[int]] = {}
    executors_by_fragment: dict[FragmentProjectionKey, list[ExecutorProjectionKey]] = {}
    for executor in executors:
        executor_key = _executor_projection_key(executor)
        if executor_key in indices_by_executor:
            raise ValueError("transfer plan has duplicate executor projection key")
        indices_by_executor[executor_key] = []
        fragment_ids: Sequence[str]
        fragment_kind: str
        if isinstance(executor, ExecutorTransferPlan):
            fragment_ids = executor.fragment_ids
            fragment_kind = "runtime"
        else:
            fragment_ids = executor.placement_fragment_ids
            fragment_kind = "placement"
        for fragment_id in fragment_ids:
            executors_by_fragment.setdefault((fragment_kind, fragment_id), []).append(
                executor_key
            )
    return indices_by_executor, executors_by_fragment


def _build_operation_views(
    operations: Sequence[Union[LogicalTransferOperation, ExecutableTransferOperation]],
    source_executors: Sequence[Union[ExecutorTransferPlan, PlacementExecutorPlan]],
    target_executors: Sequence[Union[ExecutorTransferPlan, PlacementExecutorPlan]],
) -> _OperationViews:
    source_indices, source_by_fragment = _index_executors_by_fragment(source_executors)
    target_indices, target_by_fragment = _index_executors_by_fragment(target_executors)
    for index, operation in enumerate(operations):
        source_key = _fragment_projection_key(operation.source)
        if source_key is not None:
            for executor_key in source_by_fragment.get(source_key, ()):
                source_indices[executor_key].append(index)
        target_key = _fragment_projection_key(operation.target)
        if target_key is not None:
            for executor_key in target_by_fragment.get(target_key, ()):
                target_indices[executor_key].append(index)
    return _OperationViews(
        source_indices=MappingProxyType(
            {key: tuple(indices) for key, indices in source_indices.items()}
        ),
        target_indices=MappingProxyType(
            {key: tuple(indices) for key, indices in target_indices.items()}
        ),
        pipeline_routes=_pipeline_routes(operations),
    )


def _pipeline_routes(
    operations: Sequence[Union[LogicalTransferOperation, ExecutableTransferOperation]],
) -> tuple[PipelineRouteGroup, ...]:
    indices_by_route: dict[
        tuple[Optional[int], Optional[int], int, Optional[int]],
        list[int],
    ] = {}
    for index, operation in enumerate(operations):
        source_pp = (
            operation.source.rank.pp
            if isinstance(operation.source, (BoundWeightFragment, PlacementFragment))
            else None
        )
        source_pipeline_stage_id = (
            operation.source.pipeline_stage_id
            if isinstance(operation.source, (BoundWeightFragment, PlacementFragment))
            else None
        )
        indices_by_route.setdefault(
            (
                source_pp,
                source_pipeline_stage_id,
                operation.target.rank.pp,
                operation.target.pipeline_stage_id,
            ),
            [],
        ).append(index)
    return tuple(
        PipelineRouteGroup(
            source_pp=source_pp,
            source_pipeline_stage_id=source_pipeline_stage_id,
            target_pp=target_pp,
            target_pipeline_stage_id=target_pipeline_stage_id,
            operation_indices=tuple(indices),
        )
        for (
            source_pp,
            source_pipeline_stage_id,
            target_pp,
            target_pipeline_stage_id,
        ), indices in sorted(
            indices_by_route.items(),
            key=lambda item: (
                -1 if item[0][0] is None else item[0][0],
                -1 if item[0][1] is None else item[0][1],
                item[0][2],
                -1 if item[0][3] is None else item[0][3],
            ),
        )
    )


__all__ = [
    "BoundWeightFragment",
    "ExecutorTransferPlan",
    "LogicalTransferPlan",
    "PlanningLimits",
    "PipelineRouteGroup",
    "PlacementExecutorPlan",
    "RuntimeLeaseSnapshot",
    "RuntimeTensorOwner",
    "ExecutableTransferOperation",
    "ExecutableTransferRegion",
    "LiveTransferOperation",
    "LogicalTransferOperation",
    "LogicalTransferRegion",
    "StoredLoadOperation",
    "TransferPlan",
    "TransferRegion",
]
