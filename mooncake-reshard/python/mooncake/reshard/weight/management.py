"""Typed catalog contracts for Store-managed model-weight revisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional


_MAX_U64 = (1 << 64) - 1


def _require_string(value: object, name: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _require_u64(value: object, name: str, *, nonzero: bool = False) -> None:
    minimum = 1 if nonzero else 0
    if type(value) is not int or value < minimum or value > _MAX_U64:
        qualifier = "non-zero " if nonzero else ""
        raise ValueError(f"{name} must fit in a {qualifier}unsigned 64-bit integer")


def _require_sha256(value: object, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


class WeightAvailabilityState(IntEnum):
    IMPORTING = 0
    READY = 1
    DEGRADED = 2
    DELETING = 3
    DELETED = 4


class WeightResidencyState(IntEnum):
    UNKNOWN = 0
    HOT = 1
    COLD = 2
    MIXED = 3
    ABSENT = 4


class WeightOperationState(IntEnum):
    NONE = 0
    EVICTING = 1
    REHYDRATING = 2
    REPAIRING = 3


class WeightManagementErrorCode(IntEnum):
    INVALID_ARGUMENT = 1
    NOT_FOUND = 2
    CONFLICT = 3
    STALE_GENERATION = 4
    NOT_READY = 5
    BUSY = 6
    LEASE_EXPIRED = 7
    GENERATION_EXHAUSTED = 8
    DURABILITY_FAILED = 9


class WeightManagementError(RuntimeError):
    """A catalog-domain failure returned by the Store Master."""

    def __init__(self, code: WeightManagementErrorCode, message: str = "") -> None:
        self.code = WeightManagementErrorCode(code)
        super().__init__(message or self.code.name.lower().replace("_", " "))


class WeightManagementTransportError(RuntimeError):
    """A transport failure whose mutation outcome may be uncertain."""

    def __init__(self, code: int, message: str = "") -> None:
        self.code = int(code)
        super().__init__(message or f"weight management RPC failed: {self.code}")


@dataclass(frozen=True)
class WeightRevisionIdentity:
    tenant_id: str
    namespace: str
    resource_id: str
    revision: str
    weight_generation: int

    def __post_init__(self) -> None:
        for name in ("tenant_id", "namespace", "resource_id", "revision"):
            _require_string(getattr(self, name), name)
        _require_u64(self.weight_generation, "weight_generation", nonzero=True)


@dataclass(frozen=True)
class WeightManifestReference:
    manifest_key: str
    manifest_sha256: str
    payload_group_id: str
    payload_keys_sha256: str
    payload_count: int
    logical_bytes: int

    def __post_init__(self) -> None:
        _require_string(self.payload_group_id, "payload_group_id")
        _require_u64(self.payload_count, "payload_count", nonzero=True)
        _require_u64(self.logical_bytes, "logical_bytes", nonzero=True)
        unpublished = not (
            self.manifest_key or self.manifest_sha256 or self.payload_keys_sha256
        )
        if unpublished:
            return
        _require_string(self.manifest_key, "manifest_key")
        _require_sha256(self.manifest_sha256, "manifest_sha256")
        _require_sha256(self.payload_keys_sha256, "payload_keys_sha256")


@dataclass(frozen=True)
class WeightRevisionMetadata:
    identity: WeightRevisionIdentity
    manifest: WeightManifestReference
    availability: WeightAvailabilityState
    residency: WeightResidencyState
    operation: WeightOperationState
    operation_id: int
    metadata_generation: int
    created_at_ms: int
    updated_at_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.identity, WeightRevisionIdentity):
            raise ValueError("identity must be a WeightRevisionIdentity")
        if not isinstance(self.manifest, WeightManifestReference):
            raise ValueError("manifest must be a WeightManifestReference")
        object.__setattr__(
            self, "availability", WeightAvailabilityState(self.availability)
        )
        object.__setattr__(self, "residency", WeightResidencyState(self.residency))
        object.__setattr__(self, "operation", WeightOperationState(self.operation))
        _require_u64(self.operation_id, "operation_id")
        _require_u64(self.metadata_generation, "metadata_generation", nonzero=True)
        _require_u64(self.created_at_ms, "created_at_ms")
        _require_u64(self.updated_at_ms, "updated_at_ms")
        if self.updated_at_ms < self.created_at_ms:
            raise ValueError("updated_at_ms must not precede created_at_ms")


@dataclass(frozen=True)
class WeightRevisionLease:
    lease_id: int
    identity: WeightRevisionIdentity
    holder: str
    expires_at_ms: int
    fenced_metadata_generation: int

    def __post_init__(self) -> None:
        _require_u64(self.lease_id, "lease_id", nonzero=True)
        if not isinstance(self.identity, WeightRevisionIdentity):
            raise ValueError("identity must be a WeightRevisionIdentity")
        _require_string(self.holder, "holder")
        _require_u64(self.expires_at_ms, "expires_at_ms", nonzero=True)
        _require_u64(
            self.fenced_metadata_generation,
            "fenced_metadata_generation",
            nonzero=True,
        )


@dataclass(frozen=True)
class WeightResidencyOperation:
    operation_id: int
    identity: WeightRevisionIdentity
    operation: WeightOperationState
    target_residency: WeightResidencyState
    fenced_metadata_generation: int
    started_at_ms: int
    updated_at_ms: int
    processed_members: int
    total_members: int
    cursor: str
    message: str

    def __post_init__(self) -> None:
        _require_u64(self.operation_id, "operation_id", nonzero=True)
        if not isinstance(self.identity, WeightRevisionIdentity):
            raise ValueError("identity must be a WeightRevisionIdentity")
        object.__setattr__(self, "operation", WeightOperationState(self.operation))
        object.__setattr__(
            self, "target_residency", WeightResidencyState(self.target_residency)
        )
        for name in (
            "fenced_metadata_generation",
            "started_at_ms",
            "updated_at_ms",
            "processed_members",
            "total_members",
        ):
            _require_u64(getattr(self, name), name)
        if type(self.cursor) is not str or type(self.message) is not str:
            raise ValueError("operation cursor and message must be strings")


@dataclass(frozen=True)
class WeightRevisionView:
    metadata: WeightRevisionMetadata
    active_lease_count: int
    nearest_lease_expiry_ms: Optional[int]

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, WeightRevisionMetadata):
            raise ValueError("metadata must be a WeightRevisionMetadata")
        _require_u64(self.active_lease_count, "active_lease_count")
        if self.nearest_lease_expiry_ms is not None:
            _require_u64(
                self.nearest_lease_expiry_ms,
                "nearest_lease_expiry_ms",
                nonzero=True,
            )


@dataclass(frozen=True)
class WeightRevisionPage:
    revisions: tuple[WeightRevisionView, ...]
    next_page_token: str

    def __post_init__(self) -> None:
        revisions = tuple(self.revisions)
        if not all(isinstance(item, WeightRevisionView) for item in revisions):
            raise ValueError("revisions must contain WeightRevisionView values")
        if type(self.next_page_token) is not str:
            raise ValueError("next_page_token must be a string")
        object.__setattr__(self, "revisions", revisions)


def identity_from_native(value: Any) -> WeightRevisionIdentity:
    return WeightRevisionIdentity(
        tenant_id=value.tenant_id,
        namespace=value.namespace,
        resource_id=value.resource_id,
        revision=value.revision,
        weight_generation=value.weight_generation,
    )


def manifest_reference_from_native(value: Any) -> WeightManifestReference:
    return WeightManifestReference(
        manifest_key=value.manifest_key,
        manifest_sha256=value.manifest_sha256,
        payload_group_id=value.payload_group_id,
        payload_keys_sha256=value.payload_keys_sha256,
        payload_count=value.payload_count,
        logical_bytes=value.logical_bytes,
    )


def metadata_from_native(value: Any) -> WeightRevisionMetadata:
    if isinstance(value, WeightRevisionMetadata):
        return value
    return WeightRevisionMetadata(
        identity=identity_from_native(value.identity),
        manifest=manifest_reference_from_native(value.manifest),
        availability=WeightAvailabilityState(int(value.availability)),
        residency=WeightResidencyState(int(value.residency)),
        operation=WeightOperationState(int(value.operation)),
        operation_id=value.operation_id,
        metadata_generation=value.metadata_generation,
        created_at_ms=value.created_at_ms,
        updated_at_ms=value.updated_at_ms,
    )


def lease_from_native(value: Any) -> WeightRevisionLease:
    if isinstance(value, WeightRevisionLease):
        return value
    return WeightRevisionLease(
        lease_id=value.lease_id,
        identity=identity_from_native(value.identity),
        holder=value.holder,
        expires_at_ms=value.expires_at_ms,
        fenced_metadata_generation=value.fenced_metadata_generation,
    )


def operation_from_native(value: Any) -> WeightResidencyOperation:
    if isinstance(value, WeightResidencyOperation):
        return value
    return WeightResidencyOperation(
        operation_id=value.operation_id,
        identity=identity_from_native(value.identity),
        operation=WeightOperationState(int(value.operation)),
        target_residency=WeightResidencyState(int(value.target_residency)),
        fenced_metadata_generation=value.fenced_metadata_generation,
        started_at_ms=value.started_at_ms,
        updated_at_ms=value.updated_at_ms,
        processed_members=value.processed_members,
        total_members=value.total_members,
        cursor=value.cursor,
        message=value.message,
    )


def view_from_native(value: Any) -> WeightRevisionView:
    if isinstance(value, WeightRevisionView):
        return value
    return WeightRevisionView(
        metadata=metadata_from_native(value.metadata),
        active_lease_count=value.active_lease_count,
        nearest_lease_expiry_ms=value.nearest_lease_expiry_ms,
    )


__all__ = [
    "WeightAvailabilityState",
    "WeightManagementError",
    "WeightManagementErrorCode",
    "WeightManagementTransportError",
    "WeightManifestReference",
    "WeightOperationState",
    "WeightResidencyState",
    "WeightResidencyOperation",
    "WeightRevisionIdentity",
    "WeightRevisionLease",
    "WeightRevisionMetadata",
    "WeightRevisionPage",
    "WeightRevisionView",
]
