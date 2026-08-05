"""Framework-neutral contracts for reusable model runtime resources."""

from .contracts import (
    PlacementManifest,
    ResourceAdapter,
    ResourceAdapterRegistry,
    ResourceKind,
    ResourceManifest,
    RuntimeBindingFragment,
    RuntimeBindingManifest,
    StoredResourceManifest,
    validate_resource_binding_identity,
)

__all__ = [
    "PlacementManifest",
    "ResourceAdapter",
    "ResourceAdapterRegistry",
    "ResourceKind",
    "ResourceManifest",
    "RuntimeBindingFragment",
    "RuntimeBindingManifest",
    "StoredResourceManifest",
    "validate_resource_binding_identity",
]
