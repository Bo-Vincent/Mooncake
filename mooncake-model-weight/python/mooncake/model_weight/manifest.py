"""Stable public façade for model-weight manifest contracts."""

from .binding import (
    bind_runtime_manifest,
    placement_manifest_from_runtime_manifest,
    runtime_binding_from_runtime_manifest,
)
from .placement import PlacementManifest
from .runtime import RuntimeBindingManifest, RuntimeManifest
from .types import (
    ParallelRank,
    PlacementFragment,
    RuntimeBindingFragment,
    RuntimeFragment,
    TensorDescriptor,
)


__all__ = [
    "ParallelRank",
    "PlacementFragment",
    "PlacementManifest",
    "RuntimeBindingFragment",
    "RuntimeBindingManifest",
    "RuntimeFragment",
    "RuntimeManifest",
    "TensorDescriptor",
    "bind_runtime_manifest",
    "placement_manifest_from_runtime_manifest",
    "runtime_binding_from_runtime_manifest",
]
