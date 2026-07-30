"""Cross-fragment logical and physical validation."""

from __future__ import annotations

from math import prod
from typing import Mapping, Sequence

from .types import (
    ManifestFragment,
    ParallelRank,
    RuntimeFragment,
    TensorDescriptor,
    _canonical_tensor_descriptor,
)


def _validate_fragments(
    tensors: Sequence[TensorDescriptor],
    fragments: Sequence[ManifestFragment],
) -> None:
    tensor_by_id: dict[str, TensorDescriptor] = {}
    for tensor in tensors:
        if tensor.tensor_id in tensor_by_id:
            raise ValueError(f"duplicate tensor_id: {tensor.tensor_id}")
        tensor_by_id[tensor.tensor_id] = tensor

    fragment_ids: set[str] = set()
    logical_fragments: set[tuple] = set()
    for fragment in fragments:
        if fragment.fragment_id in fragment_ids:
            raise ValueError(f"duplicate fragment_id: {fragment.fragment_id}")
        fragment_ids.add(fragment.fragment_id)
        logical_fragment = (
            fragment.tensor_id,
            fragment.rank,
            fragment.global_offset,
            fragment.local_shape,
        )
        if logical_fragment in logical_fragments:
            raise ValueError(
                "duplicate logical fragment for tensor and parallel rank: "
                f"{fragment.fragment_id}"
            )
        logical_fragments.add(logical_fragment)
        tensor = tensor_by_id.get(fragment.tensor_id)
        if tensor is None:
            raise ValueError(f"unknown tensor_id: {fragment.tensor_id}")
        _validate_fragment_geometry(tensor, fragment)
    _validate_logical_fragment_overlaps(fragments)


def _validate_logical_fragment_overlaps(
    fragments: Sequence[ManifestFragment],
) -> None:
    by_tensor_and_rank: dict[tuple[str, ParallelRank], list[ManifestFragment]] = {}
    for fragment in fragments:
        by_tensor_and_rank.setdefault((fragment.tensor_id, fragment.rank), []).append(
            fragment
        )

    for owner_fragments in by_tensor_and_rank.values():
        if len(owner_fragments) < 2:
            continue
        ndim = len(owner_fragments[0].global_offset)
        sweep_dim = max(
            range(ndim),
            key=lambda dim: len(
                {
                    (
                        fragment.global_offset[dim],
                        fragment.global_offset[dim] + fragment.local_shape[dim],
                    )
                    for fragment in owner_fragments
                }
            ),
        )
        ordered = sorted(
            owner_fragments,
            key=lambda fragment: fragment.global_offset[sweep_dim],
        )
        active: list[ManifestFragment] = []
        for current in ordered:
            current_begin = current.global_offset[sweep_dim]
            active = [
                previous
                for previous in active
                if previous.global_offset[sweep_dim] + previous.local_shape[sweep_dim]
                > current_begin
            ]
            for previous in active:
                if all(
                    previous_offset < current_offset + current_extent
                    and current_offset < previous_offset + previous_extent
                    for (
                        previous_offset,
                        previous_extent,
                        current_offset,
                        current_extent,
                    ) in zip(
                        previous.global_offset,
                        previous.local_shape,
                        current.global_offset,
                        current.local_shape,
                    )
                ):
                    raise ValueError(
                        "logical fragment boxes overlap for tensor and "
                        "parallel rank: "
                        f"{previous.fragment_id} and {current.fragment_id}"
                    )
            active.append(current)


def _validate_fragment_geometry(
    tensor: TensorDescriptor,
    fragment: ManifestFragment,
) -> None:
    ndim = len(tensor.global_shape)
    if len(fragment.global_offset) != ndim or len(fragment.local_shape) != ndim:
        raise ValueError(f"fragment rank mismatch: {fragment.fragment_id}")
    for offset, extent, total in zip(
        fragment.global_offset,
        fragment.local_shape,
        tensor.global_shape,
    ):
        if offset + extent > total:
            raise ValueError(f"fragment is out of bounds: {fragment.fragment_id}")

    shard_dims = frozenset(tensor.effective_shard_dims)
    if not shard_dims:
        if fragment.global_offset != (0,) * ndim:
            raise ValueError(
                f"replicated fragment has an offset: {fragment.fragment_id}"
            )
        if fragment.local_shape != tensor.global_shape:
            raise ValueError(
                f"replicated fragment is incomplete: {fragment.fragment_id}"
            )
    else:
        for dim in range(ndim):
            if dim in shard_dims:
                continue
            if fragment.global_offset[dim] != 0:
                raise ValueError(f"fragment offset uses a non-shard axis: {dim}")
            if fragment.local_shape[dim] != tensor.global_shape[dim]:
                raise ValueError(f"fragment shape uses a non-shard axis: {dim}")

    expected_nbytes = prod(fragment.local_shape) * tensor.itemsize
    if fragment.nbytes != expected_nbytes:
        raise ValueError(
            f"fragment byte size mismatch: {fragment.fragment_id}: "
            f"expected {expected_nbytes}, got {fragment.nbytes}"
        )


def _runtime_alias_descriptor_key(tensor: TensorDescriptor) -> tuple:
    tensor = _canonical_tensor_descriptor(tensor)
    return (
        tensor.global_shape,
        tensor.dtype,
        tensor.itemsize,
        tensor.partition_dim,
        tensor.effective_shard_dims,
        tensor.layer_id,
        tensor.expert_id,
        tensor.layout_fingerprint,
    )


def _is_exact_declared_runtime_alias(
    left: RuntimeFragment,
    right: RuntimeFragment,
    tensors: Mapping[str, TensorDescriptor],
) -> bool:
    return (
        left.address == right.address
        and left.nbytes == right.nbytes
        and left.lease_generation == right.lease_generation
        and len(left.aliases) >= 2
        and left.aliases == right.aliases
        and left.global_offset == right.global_offset
        and left.local_shape == right.local_shape
        and _runtime_alias_descriptor_key(tensors[left.tensor_id])
        == _runtime_alias_descriptor_key(tensors[right.tensor_id])
    )


def _validate_runtime_address_ranges(
    *,
    instance_id: str,
    tensors: Sequence[TensorDescriptor],
    fragments: Sequence[RuntimeFragment],
) -> None:
    tensor_by_id = {tensor.tensor_id: tensor for tensor in tensors}
    by_address_space: dict[tuple[str, str, str], list[RuntimeFragment]] = {}
    for fragment in fragments:
        address_space = (instance_id, fragment.worker_id, fragment.device)
        by_address_space.setdefault(address_space, []).append(fragment)

    for address_space, address_fragments in by_address_space.items():
        ordered = sorted(address_fragments, key=lambda item: item.address)
        active: list[RuntimeFragment] = []
        for current in ordered:
            active = [
                previous
                for previous in active
                if previous.address + previous.nbytes > current.address
            ]
            for previous in active:
                if _is_exact_declared_runtime_alias(
                    previous,
                    current,
                    tensor_by_id,
                ):
                    continue
                raise ValueError(
                    "runtime manifest address ranges overlap: "
                    f"{previous.fragment_id} and {current.fragment_id} "
                    f"in {address_space}"
                )
            active.append(current)
