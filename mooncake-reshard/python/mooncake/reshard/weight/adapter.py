"""Weight-specific adapter for the resource-neutral reshard registry."""

from __future__ import annotations

from ..contracts import (
    PlacementManifest,
    ResourceAdapter,
    ResourceKind,
    RuntimeBindingManifest,
)
from .binding import validate_runtime_binding
from .placement import WeightPlacementManifest
from .runtime import WeightRuntimeBindingManifest


class WeightReshardAdapter(ResourceAdapter):
    resource_kind = ResourceKind.MODEL_WEIGHT
    placement_type = WeightPlacementManifest
    binding_type = WeightRuntimeBindingManifest
    stored_manifest_type = None

    def validate_binding(
        self,
        placement: PlacementManifest,
        binding: RuntimeBindingManifest,
    ) -> None:
        if not isinstance(placement, WeightPlacementManifest):
            raise TypeError("weight placement has an invalid concrete type")
        if not isinstance(binding, WeightRuntimeBindingManifest):
            raise TypeError("weight binding has an invalid concrete type")
        validate_runtime_binding(placement, binding)


WEIGHT_RESHARD_ADAPTER = WeightReshardAdapter()


__all__ = ["WEIGHT_RESHARD_ADAPTER", "WeightReshardAdapter"]
