from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from mooncake.reshard.weight.management import (
    WeightAvailabilityState,
    WeightManagementError,
    WeightManagementErrorCode,
    WeightManifestReference,
    WeightMigrationMode,
    WeightOperationKind,
    WeightResidencyState,
    WeightResidencyOperation,
    WeightRevisionIdentity,
    WeightRevisionMetadata,
    WeightRevisionPage,
    WeightRevisionView,
    WeightStoragePolicy,
    metadata_from_native,
)
from mooncake.reshard.weight._store.backend import StoreBackend


def _metadata() -> WeightRevisionMetadata:
    identity = WeightRevisionIdentity(
        tenant_id="tenant-a",
        namespace="production",
        resource_id="llama-70b",
        revision="step-100",
        weight_generation=7,
    )
    return WeightRevisionMetadata(
        identity=identity,
        manifest=WeightManifestReference(
            manifest_key="weights/production/llama-70b/step-100/7/manifest",
            manifest_sha256="a" * 64,
            payload_group_id="weight:group",
            payload_keys_sha256="b" * 64,
            payload_count=2,
            logical_bytes=4096,
        ),
        availability=WeightAvailabilityState.READY,
        residency=WeightResidencyState.HOT,
        policy=WeightStoragePolicy(),
        operation_id=None,
        affinity_count=2,
        affinity_digest="c" * 64,
        observed_hot_ratio=1.0,
        metadata_generation=2,
        created_at_ms=10,
        updated_at_ms=20,
    )


def test_management_records_are_immutable_and_have_no_tensor_metadata() -> None:
    metadata = _metadata()
    with pytest.raises(FrozenInstanceError):
        metadata.metadata_generation = 3  # type: ignore[misc]
    assert not hasattr(metadata, "tensors")
    assert not hasattr(metadata.manifest, "fragments")
    assert not hasattr(metadata, "operation")


def test_storage_policy_defaults_to_mixed_auto() -> None:
    policy = WeightStoragePolicy()
    assert policy.preferred_residency is WeightResidencyState.MIXED
    assert policy.mixed_hot_ratio == 0.5
    assert policy.migration_mode is WeightMigrationMode.AUTO


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.1])
def test_mixed_policy_requires_strict_interior_ratio(ratio: float) -> None:
    with pytest.raises(ValueError, match="mixed_hot_ratio"):
        WeightStoragePolicy(mixed_hot_ratio=ratio)


def test_operation_has_kind_target_and_byte_progress() -> None:
    metadata = _metadata()
    operation = WeightResidencyOperation(
        operation_id=7,
        identity=metadata.identity,
        kind=WeightOperationKind.MIGRATING,
        target_residency=WeightResidencyState.COLD,
        target_hot_ratio=None,
        fenced_metadata_generation=metadata.metadata_generation,
        started_at_ms=20,
        updated_at_ms=21,
        processed_units=1,
        total_units=2,
        processed_bytes=1024,
        total_bytes=4096,
        cursor="unit-1",
        message="copying",
    )
    assert operation.kind is WeightOperationKind.MIGRATING
    assert operation.processed_bytes == 1024


def test_mixed_operation_requires_target_ratio() -> None:
    metadata = _metadata()
    with pytest.raises(ValueError, match="target_hot_ratio"):
        WeightResidencyOperation(
            operation_id=7,
            identity=metadata.identity,
            kind=WeightOperationKind.MIGRATING,
            target_residency=WeightResidencyState.MIXED,
            target_hot_ratio=None,
            fenced_metadata_generation=metadata.metadata_generation,
            started_at_ms=20,
            updated_at_ms=21,
            processed_units=0,
            total_units=2,
            processed_bytes=0,
            total_bytes=4096,
            cursor="",
            message="",
        )


def test_native_metadata_conversion_preserves_exact_identity() -> None:
    metadata = _metadata()

    native = SimpleNamespace(
        **metadata.__dict__,
    )
    native.identity = SimpleNamespace(**metadata.identity.__dict__)
    native.manifest = SimpleNamespace(**metadata.manifest.__dict__)

    assert metadata_from_native(native) == metadata


def test_error_mapping_retains_domain_code() -> None:
    error = WeightManagementError(WeightManagementErrorCode.STALE_GENERATION)
    assert error.code is WeightManagementErrorCode.STALE_GENERATION
    assert "stale generation" in str(error)


def test_backend_maps_domain_errors_without_collapsing_transport_errors() -> None:
    class Raw:
        def get_weight_revision(self, *args):
            return None, int(WeightManagementErrorCode.STALE_GENERATION), 0

    with pytest.raises(WeightManagementError) as error:
        StoreBackend(Raw()).weight_get_metadata(_metadata().identity)
    assert error.value.code is WeightManagementErrorCode.STALE_GENERATION


def test_backend_converts_bounded_list_page() -> None:
    view = WeightRevisionView(
        metadata=_metadata(),
        active_lease_count=0,
        nearest_lease_expiry_ms=None,
    )

    class Raw:
        def list_weight_revisions(self, *args):
            assert args[-2:] == ("cursor", 17)
            return WeightRevisionPage((view,), "next"), 0, 0

    page = StoreBackend(Raw()).weight_list(
        tenant_id="tenant-a",
        namespace="production",
        resource_id="llama-70b",
        page_token="cursor",
        limit=17,
    )
    assert page.revisions == (view,)
    assert page.next_page_token == "next"


@pytest.mark.parametrize("value", ["A" * 64, "0" * 63, "g" * 64])
def test_manifest_digest_must_be_canonical_sha256(value: str) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        WeightManifestReference(
            manifest_key="manifest",
            manifest_sha256=value,
            payload_group_id="group",
            payload_keys_sha256="b" * 64,
            payload_count=1,
            logical_bytes=1,
        )
