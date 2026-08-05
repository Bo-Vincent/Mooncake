from __future__ import annotations

from dataclasses import dataclass

import pytest

from mooncake.reshard import (
    PlacementManifest,
    ResourceAdapter,
    ResourceAdapterRegistry,
    ResourceKind,
    RuntimeBindingFragment,
    RuntimeBindingManifest,
    StoredResourceManifest,
    validate_resource_binding_identity,
)
from mooncake.reshard.weight import WEIGHT_RESHARD_ADAPTER
from mooncake.reshard.weight import RuntimeBindingFragment as WeightRuntimeFragment
from mooncake.reshard.weight.storage_manifest import WeightManifest


@dataclass(frozen=True)
class KvPlacement:
    resource_id: str
    placement_id: str

    @property
    def resource_kind(self) -> ResourceKind:
        return ResourceKind.KV_CACHE

    @property
    def digest(self) -> str:
        return "a" * 64


@dataclass(frozen=True)
class KvBinding:
    resource_id: str
    placement_id: str
    placement_digest: str
    instance_id: str
    generation: int
    lease_id: str

    @property
    def resource_kind(self) -> ResourceKind:
        return ResourceKind.KV_CACHE


class KvAdapter(ResourceAdapter):
    resource_kind = ResourceKind.KV_CACHE
    placement_type = KvPlacement
    binding_type = KvBinding

    def validate_binding(
        self,
        placement: PlacementManifest,
        binding: RuntimeBindingManifest,
    ) -> None:
        validate_resource_binding_identity(placement, binding)


def kv_pair() -> tuple[KvPlacement, KvBinding]:
    placement = KvPlacement(resource_id="request-7", placement_id="kv-placement")
    binding = KvBinding(
        resource_id=placement.resource_id,
        placement_id=placement.placement_id,
        placement_digest=placement.digest,
        instance_id="decode-0",
        generation=3,
        lease_id="kv-lease-3",
    )
    return placement, binding


def test_registry_routes_weight_and_kv_without_model_name_rules() -> None:
    kv_adapter = KvAdapter()
    registry = ResourceAdapterRegistry((WEIGHT_RESHARD_ADAPTER, kv_adapter))

    assert registry.resolve(ResourceKind.MODEL_WEIGHT) is WEIGHT_RESHARD_ADAPTER
    assert WEIGHT_RESHARD_ADAPTER.stored_manifest_type is WeightManifest
    assert registry.resolve(ResourceKind.KV_CACHE) is kv_adapter
    assert set(registry.resource_kinds()) == {
        ResourceKind.MODEL_WEIGHT,
        ResourceKind.KV_CACHE,
    }


def test_registry_rejects_duplicate_resource_kind() -> None:
    with pytest.raises(ValueError, match="duplicate resource adapter"):
        ResourceAdapterRegistry((KvAdapter(), KvAdapter()))


def test_common_identity_fence_accepts_kv_and_rejects_cross_resource_binding() -> None:
    placement, binding = kv_pair()

    validate_resource_binding_identity(placement, binding)

    with pytest.raises(ValueError, match="resource_kind"):
        validate_resource_binding_identity(
            placement,
            WEIGHT_RESHARD_ADAPTER.binding_type(
                resource_id=placement.resource_id,
                revision="step-1",
                placement_id=placement.placement_id,
                placement_digest=placement.digest,
                instance_id=binding.instance_id,
                participant_id="worker-0",
                generation=binding.generation,
                lease_id=binding.lease_id,
                fragments=(),
            ),
        )


def test_weight_storage_manifest_occupies_typed_resource_contract() -> None:
    manifest = WeightManifest(
        namespace="test",
        resource_id="model",
        revision="revision",
        weight_generation=0,
        group_id="weights/test/model/revision",
        manifest_key="weights/test/model/revision/manifest",
        tensors=(),
        fragments=(),
        created_at="2026-08-11T00:00:00Z",
    )
    stored: StoredResourceManifest = manifest

    assert stored.resource_kind is ResourceKind.MODEL_WEIGHT
    assert stored.resource_id == "model"


def test_weight_uses_the_resource_neutral_runtime_fragment() -> None:
    assert WeightRuntimeFragment is RuntimeBindingFragment
