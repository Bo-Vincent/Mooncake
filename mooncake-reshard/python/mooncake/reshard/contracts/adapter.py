"""Resource adapter dispatch without framework or model-name inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from .manifest import (
    PlacementManifest,
    ResourceKind,
    RuntimeBindingManifest,
    StoredResourceManifest,
)


class ResourceAdapter(ABC):
    """Bind one resource kind to its semantic placement validator."""

    resource_kind: ResourceKind
    placement_type: type[PlacementManifest]
    binding_type: type[RuntimeBindingManifest]
    stored_manifest_type: type[StoredResourceManifest] | None = None

    @abstractmethod
    def validate_binding(
        self,
        placement: PlacementManifest,
        binding: RuntimeBindingManifest,
    ) -> None:
        """Validate one concrete binding against its logical placement."""


class ResourceAdapterRegistry:
    """Explicit resource-kind routing shared by weight and future KV adapters."""

    def __init__(self, adapters: Iterable[ResourceAdapter] = ()) -> None:
        self._adapters: dict[ResourceKind, ResourceAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: ResourceAdapter) -> None:
        if not isinstance(adapter, ResourceAdapter):
            raise TypeError("resource adapter must implement ResourceAdapter")
        if not isinstance(adapter.resource_kind, ResourceKind):
            raise TypeError("resource adapter resource_kind is invalid")
        if not isinstance(adapter.placement_type, type):
            raise TypeError("resource adapter placement_type is invalid")
        if not isinstance(adapter.binding_type, type):
            raise TypeError("resource adapter binding_type is invalid")
        if adapter.stored_manifest_type is not None and not isinstance(
            adapter.stored_manifest_type, type
        ):
            raise TypeError("resource adapter stored_manifest_type is invalid")
        if adapter.resource_kind in self._adapters:
            raise ValueError(
                f"duplicate resource adapter: {adapter.resource_kind.value}"
            )
        self._adapters[adapter.resource_kind] = adapter

    def resolve(self, resource_kind: ResourceKind) -> ResourceAdapter:
        if not isinstance(resource_kind, ResourceKind):
            raise TypeError("resource_kind must be a ResourceKind")
        try:
            return self._adapters[resource_kind]
        except KeyError as error:
            raise KeyError(
                f"resource adapter is not registered: {resource_kind.value}"
            ) from error

    def resource_kinds(self) -> tuple[ResourceKind, ...]:
        return tuple(sorted(self._adapters, key=lambda item: item.value))
