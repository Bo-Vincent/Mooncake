from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from ...contracts import (
    LeaseId,
    ParticipantId,
    PlacementId,
    RuntimeInstanceId,
    StoredFragmentId,
)
from ..manifest import PlacementFragment, RuntimeBindingFragment
from ..planner import (
    BoundWeightFragment,
    ExecutableTransferOperation,
    StoredLoadOperation,
    TransferPlan,
)
from ..storage_manifest import StoredFragment, WeightManifest


_MAX_U64 = (1 << 64) - 1


def _require_nonempty_string(value: object, name: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _require_u64(value: object, name: str) -> None:
    if type(value) is not int or value < 0 or value > _MAX_U64:
        raise ValueError(f"{name} must fit in an unsigned 64-bit integer")


@dataclass(frozen=True)
class UploadOperation:
    source_placement: PlacementFragment
    source_binding: RuntimeBindingFragment
    source_participant_id: ParticipantId
    source_instance_id: RuntimeInstanceId
    target: StoredFragment
    source_generation: int
    source_lease_id: LeaseId

    def __post_init__(self) -> None:
        if not isinstance(self.source_placement, PlacementFragment):
            raise ValueError("upload source placement is invalid")
        if not isinstance(self.source_binding, RuntimeBindingFragment):
            raise ValueError("upload source binding is invalid")
        if not isinstance(self.target, StoredFragment):
            raise ValueError("upload target is invalid")
        for name in (
            "source_participant_id",
            "source_instance_id",
            "source_lease_id",
        ):
            _require_nonempty_string(getattr(self, name), name)
        _require_u64(self.source_generation, "source_generation")
        if (
            self.source_binding.placement_fragment_id
            != self.source_placement.placement_fragment_id
        ):
            raise ValueError("upload source placement and binding differ")
        if (
            self.source_binding.nbytes != self.source_placement.nbytes
            or self.source_binding.local_shape != self.source_placement.local_shape
        ):
            raise ValueError("upload source geometry differs")
        if (
            self.target.tensor_id != self.source_placement.tensor_id
            or self.target.global_offset != self.source_placement.global_offset
            or self.target.local_shape != self.source_placement.local_shape
            or self.target.nbytes != self.source_placement.nbytes
            or self.target.aliases != self.source_placement.aliases
        ):
            raise ValueError("upload source and target geometry differ")
        if self.target.object_offset != 0:
            raise ValueError("upload target object_offset must be zero")


@dataclass(frozen=True)
class WeightUploadPlan:
    manifest: WeightManifest
    source_placement_id: PlacementId
    source_placement_digest: str
    session_group_id: str
    control_key: str
    operations: tuple[UploadOperation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, WeightManifest):
            raise ValueError("upload plan manifest is invalid")
        _require_nonempty_string(self.source_placement_id, "source_placement_id")
        _require_nonempty_string(
            self.source_placement_digest, "source_placement_digest"
        )
        _require_nonempty_string(self.session_group_id, "session_group_id")
        _require_nonempty_string(self.control_key, "control_key")
        operations = tuple(self.operations)
        if not operations or not all(
            isinstance(operation, UploadOperation) for operation in operations
        ):
            raise ValueError("upload plan operations are invalid")
        object.__setattr__(self, "operations", operations)
        session_prefix = f"{self.manifest.group_id}/sessions/"
        if not self.session_group_id.startswith(session_prefix):
            raise ValueError("upload session group does not belong to manifest group")
        upload_id = self.session_group_id[len(session_prefix) :]
        if not upload_id or "/" in upload_id:
            raise ValueError("upload session group has an invalid upload ID")
        if self.control_key != f"{self.session_group_id}/decision":
            raise ValueError("upload control key does not belong to session group")
        operation_targets = tuple(
            sorted(
                (operation.target for operation in operations),
                key=lambda fragment: fragment.fragment_id,
            )
        )
        manifest_fragments = tuple(
            sorted(
                self.manifest.fragments,
                key=lambda fragment: fragment.fragment_id,
            )
        )
        if operation_targets != manifest_fragments:
            raise ValueError("upload plan operations and manifest differ")
        payload_prefix = f"{self.manifest.group_id}/payload/"
        if any(
            operation.target.object_key[len(payload_prefix) :].split("/", 1)[0]
            != upload_id
            for operation in operations
        ):
            raise ValueError("upload session does not own payload members")


@dataclass(frozen=True)
class UploadReceipt:
    fragment_id: StoredFragmentId
    object_key: str
    worker_id: str

    def __post_init__(self) -> None:
        for name in ("fragment_id", "object_key", "worker_id"):
            _require_nonempty_string(getattr(self, name), name)


@dataclass(frozen=True)
class WeightLoadPlan:
    manifest: WeightManifest
    transfer: TransferPlan

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, WeightManifest):
            raise ValueError("load plan manifest is invalid")
        if not isinstance(self.transfer, TransferPlan):
            raise ValueError("load plan transfer is invalid")
        if (
            self.transfer.resource_id != self.manifest.resource_id
            or self.transfer.revision != self.manifest.revision
            or self.transfer.weight_generation != self.manifest.weight_generation
        ):
            raise ValueError("load plan transfer and manifest identity differ")
        if not self.transfer.operations:
            raise ValueError("load plan must transfer stored fragments to runtime")
        stored_operations = tuple(
            _require_stored_load_operation(operation)
            for operation in self.transfer.operations
        )
        manifest_fragments = {
            fragment.fragment_id: fragment for fragment in self.manifest.fragments
        }
        planned_sources: dict[StoredFragmentId, StoredFragment] = {}
        for operation in stored_operations:
            source = operation.source
            previous = planned_sources.setdefault(source.fragment_id, source)
            if previous != source:
                raise ValueError("load plan contains conflicting stored fragments")
        if set(planned_sources) - set(manifest_fragments):
            raise ValueError("load plan sources and manifest differ")
        for fragment_id, manifest_fragment in manifest_fragments.items():
            planned = planned_sources.get(fragment_id)
            if planned == manifest_fragment:
                continue
            if not any(
                _stored_alias_equivalent(manifest_fragment, candidate)
                for candidate in planned_sources.values()
            ):
                raise ValueError("load plan sources and manifest differ")


def _stored_alias_equivalent(left: StoredFragment, right: StoredFragment) -> bool:
    return (
        len(left.aliases) >= 2
        and left.aliases == right.aliases
        and left.tensor_id in left.aliases
        and right.tensor_id in left.aliases
        and left.global_offset == right.global_offset
        and left.local_shape == right.local_shape
        and left.nbytes == right.nbytes
    )


__all__ = [
    "UploadOperation",
    "UploadReceipt",
    "WeightLoadPlan",
    "WeightUploadPlan",
]


def _require_stored_load_operation(
    operation: ExecutableTransferOperation,
) -> StoredLoadOperation:
    if not isinstance(operation.source, StoredFragment) or not isinstance(
        operation.target, BoundWeightFragment
    ):
        raise ValueError("load plan must transfer stored fragments to runtime")
    return cast(StoredLoadOperation, operation)
