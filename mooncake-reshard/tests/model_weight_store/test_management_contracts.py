from dataclasses import FrozenInstanceError

import pytest

from mooncake.reshard.weight.management import (
    WeightAvailabilityState,
    WeightManagementError,
    WeightManagementErrorCode,
    WeightManifestReference,
    WeightOperationState,
    WeightResidencyState,
    WeightRevisionIdentity,
    WeightRevisionMetadata,
    WeightRevisionPage,
    WeightRevisionView,
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
        operation=WeightOperationState.NONE,
        operation_id=0,
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


def test_native_metadata_conversion_preserves_exact_identity() -> None:
    metadata = _metadata()

    class Native:
        pass

    native = Native()
    native.__dict__.update(metadata.__dict__)
    native.identity = Native()
    native.identity.__dict__.update(
        {
            "tenant_id": metadata.identity.tenant_id,
            "name_space": metadata.identity.namespace,
            "resource_id": metadata.identity.resource_id,
            "revision": metadata.identity.revision,
            "weight_generation": metadata.identity.weight_generation,
        }
    )
    native.manifest = Native()
    native.manifest.__dict__.update(metadata.manifest.__dict__)

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
        StoreBackend(Raw()).get_weight_revision(_metadata().identity)
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

    page = StoreBackend(Raw()).list_weight_revisions(
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
