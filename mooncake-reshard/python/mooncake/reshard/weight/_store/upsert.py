"""Generation-aware replacement context for managed weight revisions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

from ..management import (
    WeightRevisionIdentity,
    WeightUpsertMode,
)
from .errors import WeightStoreError
from .snapshot import WeightSnapshotDescriptor


@dataclass(frozen=True)
class WeightUpsertContext:
    request_id: str
    mode: WeightUpsertMode
    replacing: WeightRevisionIdentity
    target: WeightRevisionIdentity
    expected_metadata_generation: int


def _derive_request_id(
    replacing: WeightRevisionIdentity,
    target: WeightRevisionIdentity,
    mode: WeightUpsertMode,
    expected_metadata_generation: int,
) -> str:
    digest = hashlib.sha256()
    fields = (
        "weight-upsert-request-v1",
        replacing.tenant_id,
        replacing.namespace,
        replacing.resource_id,
        replacing.revision,
        str(replacing.weight_generation),
        target.tenant_id,
        target.namespace,
        target.resource_id,
        target.revision,
        str(target.weight_generation),
        str(int(mode)),
        str(expected_metadata_generation),
    )
    for field in fields:
        encoded = field.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def prepare_weight_upsert_context(
    snapshot: WeightSnapshotDescriptor,
    *,
    replacing: WeightRevisionIdentity,
    expected_metadata_generation: int,
    mode: WeightUpsertMode,
    tenant_id: str,
    request_id: Optional[str],
) -> WeightUpsertContext:
    if not isinstance(snapshot, WeightSnapshotDescriptor):
        raise WeightStoreError("snapshot must be a WeightSnapshotDescriptor")
    if not isinstance(replacing, WeightRevisionIdentity):
        raise WeightStoreError("replacing must be a WeightRevisionIdentity")
    if (
        type(expected_metadata_generation) is not int
        or expected_metadata_generation <= 0
    ):
        raise WeightStoreError("expected_metadata_generation must be positive")
    if not isinstance(mode, WeightUpsertMode):
        raise WeightStoreError("mode must be a WeightUpsertMode")
    resolved_mode = mode
    target = WeightRevisionIdentity(
        tenant_id=tenant_id,
        namespace=snapshot.namespace,
        resource_id=snapshot.resource_id,
        revision=snapshot.revision,
        weight_generation=snapshot.weight_generation,
    )
    if (
        replacing.tenant_id != target.tenant_id
        or replacing.namespace != target.namespace
        or replacing.resource_id != target.resource_id
        or replacing.revision != target.revision
    ):
        raise WeightStoreError(
            "upsert target and replacing revision must use same lineage"
        )
    if target.weight_generation <= replacing.weight_generation:
        raise WeightStoreError("upsert target generation must increase")
    if request_id is None:
        request_id = _derive_request_id(
            replacing,
            target,
            resolved_mode,
            expected_metadata_generation,
        )
    elif type(request_id) is not str or not request_id:
        raise WeightStoreError("request_id must be a non-empty string")
    return WeightUpsertContext(
        request_id=request_id,
        mode=resolved_mode,
        replacing=replacing,
        target=target,
        expected_metadata_generation=expected_metadata_generation,
    )


__all__ = ["WeightUpsertContext"]
