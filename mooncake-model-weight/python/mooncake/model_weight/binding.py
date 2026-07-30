"""Projection and binding between logical placement and runtime locations."""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from .placement import (
    PlacementManifest,
    _canonical_json_digest,
    _logical_placement_id,
)
from .runtime import RuntimeBindingManifest, RuntimeManifest
from .types import (
    PlacementFragment,
    RuntimeBindingFragment,
    RuntimeFragment,
    _require_nonempty_string,
)


def placement_manifest_from_runtime_manifest(
    manifest: RuntimeManifest,
    *,
    placement_id: Optional[str] = None,
) -> PlacementManifest:
    """Project a runtime snapshot into address-free placement semantics."""

    fragments = tuple(
        PlacementFragment(
            placement_fragment_id=_logical_fragment_id(fragment),
            tensor_id=fragment.tensor_id,
            global_offset=fragment.global_offset,
            local_shape=fragment.local_shape,
            nbytes=fragment.nbytes,
            rank=fragment.rank,
            aliases=fragment.aliases,
        )
        for fragment in manifest.fragments
    )
    if placement_id is not None:
        _require_nonempty_string(placement_id, "placement_id")
    effective_placement_id = placement_id or manifest.placement_id
    if effective_placement_id is None:
        effective_placement_id = _logical_placement_id(
            model_id=manifest.model_id,
            revision=manifest.revision,
            tensors=manifest.tensors,
            fragments=fragments,
        )
    return PlacementManifest(
        model_id=manifest.model_id,
        revision=manifest.revision,
        placement_id=effective_placement_id,
        tensors=manifest.tensors,
        fragments=fragments,
    )


def runtime_binding_from_runtime_manifest(
    manifest: RuntimeManifest,
    *,
    placement_id: Optional[str] = None,
) -> RuntimeBindingManifest:
    """Project runtime locations into the ephemeral binding contract."""

    if manifest.lease_id is None:
        raise ValueError("runtime manifest has no lease_id for runtime binding")
    if manifest.generation is None:
        raise ValueError("runtime manifest has no generation for runtime binding")
    placement = placement_manifest_from_runtime_manifest(
        manifest,
        placement_id=placement_id,
    )
    return RuntimeBindingManifest(
        model_id=manifest.model_id,
        revision=manifest.revision,
        placement_id=placement.placement_id,
        placement_digest=placement.digest,
        instance_id=manifest.instance_id,
        generation=manifest.generation,
        lease_id=manifest.lease_id,
        fragments=tuple(
            RuntimeBindingFragment(
                placement_fragment_id=_logical_fragment_id(fragment),
                fragment_id=fragment.fragment_id,
                address=fragment.address,
                nbytes=fragment.nbytes,
                worker_id=fragment.worker_id,
                endpoint=fragment.endpoint,
                device=fragment.device,
                owner=fragment.owner,
            )
            for fragment in manifest.fragments
        ),
    )


def bind_runtime_manifest(
    placement: PlacementManifest,
    binding: RuntimeBindingManifest,
) -> RuntimeManifest:
    """Bind physical runtime locations to an exact logical placement."""

    if placement.model_id != binding.model_id:
        raise ValueError("placement and runtime binding model_id differ")
    if placement.revision != binding.revision:
        raise ValueError("placement and runtime binding revision differ")
    if placement.placement_id != binding.placement_id:
        raise ValueError("placement_id and runtime binding placement_id differ")
    if placement.digest != binding.placement_digest:
        raise ValueError("placement digest and runtime binding placement digest differ")

    placement_by_id = {
        fragment.placement_fragment_id: fragment for fragment in placement.fragments
    }
    binding_by_id = {
        fragment.placement_fragment_id: fragment for fragment in binding.fragments
    }
    unknown = sorted(binding_by_id.keys() - placement_by_id.keys())
    if unknown:
        raise ValueError(f"unknown placement fragment in runtime binding: {unknown[0]}")
    missing = sorted(placement_by_id.keys() - binding_by_id.keys())
    if missing:
        raise ValueError(f"missing placement fragment in runtime binding: {missing[0]}")

    fragments = []
    for placement_fragment in placement.fragments:
        runtime = binding_by_id[placement_fragment.placement_fragment_id]
        if runtime.nbytes != placement_fragment.nbytes:
            raise ValueError(
                "runtime binding byte size does not match placement: "
                f"{placement_fragment.placement_fragment_id}"
            )
        fragments.append(
            RuntimeFragment(
                fragment_id=runtime.fragment_id,
                tensor_id=placement_fragment.tensor_id,
                global_offset=placement_fragment.global_offset,
                local_shape=placement_fragment.local_shape,
                address=runtime.address,
                nbytes=runtime.nbytes,
                worker_id=runtime.worker_id,
                endpoint=runtime.endpoint,
                device=runtime.device,
                rank=placement_fragment.rank,
                lease_generation=binding.generation,
                owner=runtime.owner,
                aliases=placement_fragment.aliases,
                placement_fragment_id=placement_fragment.placement_fragment_id,
            )
        )
    return RuntimeManifest(
        model_id=placement.model_id,
        revision=placement.revision,
        instance_id=binding.instance_id,
        tensors=placement.tensors,
        fragments=tuple(fragments),
        lease_id=binding.lease_id,
        placement_id=placement.placement_id,
        generation=binding.generation,
    )


def _logical_fragment_id(fragment: RuntimeFragment) -> str:
    if fragment.placement_fragment_id is not None:
        return fragment.placement_fragment_id
    content = {
        "tensor_id": fragment.tensor_id,
        "global_offset": fragment.global_offset,
        "local_shape": fragment.local_shape,
        "nbytes": fragment.nbytes,
        "rank": asdict(fragment.rank),
        "aliases": fragment.aliases,
    }
    return f"logical:{_canonical_json_digest(content)}"
