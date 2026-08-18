from __future__ import annotations

from typing import Sequence, Union

from ..manifest import (
    RuntimeBindingFragment,
    WeightPlacementManifest,
    WeightRuntimeBindingManifest,
    validate_runtime_bindings,
    validate_runtime_binding,
)
from ..planner import BoundWeightFragment
from .errors import WeightStoreError


def same_runtime_snapshot(
    current: RuntimeBindingFragment,
    planned: RuntimeBindingFragment,
) -> bool:
    return current == planned


def validate_manifest_pair(
    placement: WeightPlacementManifest,
    binding: WeightRuntimeBindingManifest,
    label: str,
) -> None:
    try:
        validate_runtime_binding(placement, binding)
    except ValueError as error:
        raise WeightStoreError(f"invalid {label} runtime binding: {error}") from error


def validate_manifest_set(
    placement: WeightPlacementManifest,
    bindings: Sequence[WeightRuntimeBindingManifest],
    label: str,
) -> None:
    try:
        validate_runtime_bindings(placement, bindings)
    except (TypeError, ValueError) as error:
        raise WeightStoreError(f"invalid {label} runtime bindings: {error}") from error


def pair_manifests(
    placement: WeightPlacementManifest,
    bindings: Sequence[WeightRuntimeBindingManifest],
    label: str,
) -> tuple[tuple[WeightPlacementManifest, WeightRuntimeBindingManifest], ...]:
    if not isinstance(placement, WeightPlacementManifest):
        raise ValueError(f"{label} placement must be a WeightPlacementManifest")
    items = tuple(bindings)
    if not items:
        raise WeightStoreError(f"{label} runtime bindings must not be empty")
    if not all(isinstance(item, WeightRuntimeBindingManifest) for item in items):
        raise WeightStoreError(
            f"{label} runtime bindings must contain WeightRuntimeBindingManifest"
        )
    participant_ids = [item.participant_id for item in items]
    if len(participant_ids) != len(set(participant_ids)):
        raise WeightStoreError(f"duplicate {label} runtime binding participant")
    for binding in items:
        validate_manifest_pair(placement, binding, label)
    return tuple((placement, binding) for binding in items)


def runtime_binding_fragment(
    fragment: Union[RuntimeBindingFragment, BoundWeightFragment],
) -> RuntimeBindingFragment:
    if isinstance(fragment, RuntimeBindingFragment):
        return fragment
    if isinstance(fragment, BoundWeightFragment):
        return fragment.binding
    raise WeightStoreError(
        "transfer plan physical fragment must be a RuntimeBindingFragment "
        "or expose one as .binding"
    )
