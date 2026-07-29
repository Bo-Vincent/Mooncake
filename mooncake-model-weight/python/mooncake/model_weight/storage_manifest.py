"""Persisted model-weight fragments and manifest contracts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import prod
from typing import Any, Mapping, Sequence

from .types import (
    TensorDescriptor,
    _require_integer,
    _require_integer_tuple,
    _require_manifest_items,
    _require_nonempty_string,
)
from .validation import _validate_fragment_geometry


@dataclass(frozen=True)
class StoredFragment:
    fragment_id: str
    tensor_id: str
    global_offset: tuple[int, ...]
    local_shape: tuple[int, ...]
    object_key: str
    object_offset: int
    nbytes: int
    checksum: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "global_offset",
            _require_integer_tuple(self.global_offset, "global_offset", minimum=0),
        )
        object.__setattr__(
            self,
            "local_shape",
            _require_integer_tuple(self.local_shape, "local_shape", minimum=1),
        )
        for name in ("fragment_id", "tensor_id", "object_key"):
            _require_nonempty_string(getattr(self, name), name)
        _require_integer(self.object_offset, "object_offset", minimum=0)
        _require_integer(self.nbytes, "nbytes", minimum=1)
        if self.checksum is not None:
            _require_nonempty_string(self.checksum, "checksum")


@dataclass(frozen=True)
class WeightManifest:
    namespace: str
    model_id: str
    revision: str
    group_id: str
    manifest_key: str
    tensors: tuple[TensorDescriptor, ...]
    fragments: tuple[StoredFragment, ...]
    created_at: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tensors",
            _require_manifest_items(
                self.tensors,
                "WeightManifest tensors",
                TensorDescriptor,
            ),
        )
        object.__setattr__(
            self,
            "fragments",
            _require_manifest_items(
                self.fragments,
                "WeightManifest fragments",
                StoredFragment,
            ),
        )
        for name in (
            "namespace",
            "model_id",
            "revision",
            "group_id",
            "manifest_key",
            "created_at",
        ):
            _require_nonempty_string(getattr(self, name), name)
        if self.manifest_key != f"{self.group_id}/manifest":
            raise ValueError("manifest_key does not belong to manifest group")
        payload_prefix = f"{self.group_id}/payload/"
        if any(
            not fragment.object_key.startswith(payload_prefix)
            for fragment in self.fragments
        ):
            raise ValueError("payload object_key does not belong to manifest group")
        _validate_stored_fragments(self.tensors, self.fragments)
        _validate_stored_coverage(self.tensors, self.fragments)
        _validate_stored_object_ranges(self.fragments)

    def to_json(self) -> str:
        tensors = []
        for tensor in self.tensors:
            tensors.append(
                {
                    "tensor_id": tensor.tensor_id,
                    "global_shape": tensor.global_shape,
                    "dtype": tensor.dtype,
                    "itemsize": tensor.itemsize,
                    "partition_dim": tensor.partition_dim,
                    "layer_id": tensor.layer_id,
                    "expert_id": tensor.expert_id,
                    "layout_fingerprint": tensor.layout_fingerprint,
                    "shard_dims": tensor.shard_dims,
                }
            )
        raw = {
            "namespace": self.namespace,
            "model_id": self.model_id,
            "revision": self.revision,
            "group_id": self.group_id,
            "manifest_key": self.manifest_key,
            "tensors": tensors,
            "fragments": [asdict(fragment) for fragment in self.fragments],
            "created_at": self.created_at,
        }
        return json.dumps(
            raw,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> WeightManifest:
        def reject_constant(constant: str) -> None:
            raise ValueError(f"non-finite JSON number is unsupported: {constant}")

        try:
            raw = json.loads(value, parse_constant=reject_constant)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("weight manifest is not valid JSON") from error
        if not isinstance(raw, Mapping):
            raise ValueError("weight manifest must be a JSON object")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> WeightManifest:
        raw = _require_exact_fields(
            raw,
            frozenset(
                {
                    "namespace",
                    "model_id",
                    "revision",
                    "group_id",
                    "manifest_key",
                    "tensors",
                    "fragments",
                    "created_at",
                }
            ),
            "weight manifest",
        )
        tensor_fields = frozenset(
            {
                "tensor_id",
                "global_shape",
                "dtype",
                "itemsize",
                "partition_dim",
                "layer_id",
                "expert_id",
                "layout_fingerprint",
                "shard_dims",
            }
        )
        fragment_fields = frozenset(
            {
                "fragment_id",
                "tensor_id",
                "global_offset",
                "local_shape",
                "object_key",
                "object_offset",
                "nbytes",
                "checksum",
            }
        )
        tensors = tuple(
            TensorDescriptor(
                **_require_exact_fields(item, tensor_fields, "tensor descriptor")
            )
            for item in raw["tensors"]
        )
        fragments = tuple(
            StoredFragment(
                **_require_exact_fields(item, fragment_fields, "stored fragment")
            )
            for item in raw["fragments"]
        )
        return cls(
            namespace=raw["namespace"],
            model_id=raw["model_id"],
            revision=raw["revision"],
            group_id=raw["group_id"],
            manifest_key=raw["manifest_key"],
            tensors=tensors,
            fragments=fragments,
            created_at=raw["created_at"],
        )


def _require_exact_fields(
    value: Any,
    expected: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} schema fields do not match contract")
    return value


def _validate_stored_fragments(
    tensors: Sequence[TensorDescriptor],
    fragments: Sequence[StoredFragment],
) -> None:
    tensor_by_id: dict[str, TensorDescriptor] = {}
    for tensor in tensors:
        if tensor.tensor_id in tensor_by_id:
            raise ValueError(f"duplicate tensor_id: {tensor.tensor_id}")
        tensor_by_id[tensor.tensor_id] = tensor

    fragment_ids: set[str] = set()
    for fragment in fragments:
        if fragment.fragment_id in fragment_ids:
            raise ValueError(f"duplicate fragment_id: {fragment.fragment_id}")
        fragment_ids.add(fragment.fragment_id)
        tensor = tensor_by_id.get(fragment.tensor_id)
        if tensor is None:
            raise ValueError(f"unknown tensor_id: {fragment.tensor_id}")
        _validate_fragment_geometry(tensor, fragment)


def _validate_stored_coverage(
    tensors: Sequence[TensorDescriptor],
    fragments: Sequence[StoredFragment],
) -> None:
    by_tensor: dict[str, list[StoredFragment]] = {}
    geometries: set[tuple] = set()
    for fragment in fragments:
        geometry = (
            fragment.tensor_id,
            fragment.global_offset,
            fragment.local_shape,
        )
        if geometry in geometries:
            raise ValueError(f"duplicate fragment geometry: {fragment.tensor_id}")
        geometries.add(geometry)
        by_tensor.setdefault(fragment.tensor_id, []).append(fragment)

    for tensor in tensors:
        tensor_fragments = by_tensor.get(tensor.tensor_id, [])
        covered_volume = sum(
            prod(fragment.local_shape) for fragment in tensor_fragments
        )
        if covered_volume != prod(tensor.global_shape) or _has_overlapping_boxes(
            tensor_fragments
        ):
            raise ValueError(f"tensor is not fully covered: {tensor.tensor_id}")


def _validate_stored_object_ranges(
    fragments: Sequence[StoredFragment],
) -> None:
    by_object: dict[str, list[StoredFragment]] = {}
    for fragment in fragments:
        by_object.setdefault(fragment.object_key, []).append(fragment)

    for object_key, object_fragments in by_object.items():
        ordered = sorted(object_fragments, key=lambda item: item.object_offset)
        for previous, current in zip(ordered, ordered[1:]):
            if current.object_offset < previous.object_offset + previous.nbytes:
                raise ValueError(
                    "stored fragment object ranges overlap: "
                    f"{previous.fragment_id} and {current.fragment_id} "
                    f"in {object_key}"
                )


def _has_overlapping_boxes(fragments: Sequence[StoredFragment]) -> bool:
    if len(fragments) < 2:
        return False

    ndim = len(fragments[0].global_offset)
    sweep_dim = max(
        range(ndim),
        key=lambda dim: len(
            {
                (
                    fragment.global_offset[dim],
                    fragment.global_offset[dim] + fragment.local_shape[dim],
                )
                for fragment in fragments
            }
        ),
    )
    ordered = sorted(fragments, key=lambda item: item.global_offset[sweep_dim])
    active: list[StoredFragment] = []
    for fragment in ordered:
        begin = fragment.global_offset[sweep_dim]
        active = [
            candidate
            for candidate in active
            if candidate.global_offset[sweep_dim] + candidate.local_shape[sweep_dim]
            > begin
        ]
        for candidate in active:
            if all(
                left_offset < right_offset + right_extent
                and right_offset < left_offset + left_extent
                for left_offset, left_extent, right_offset, right_extent in zip(
                    candidate.global_offset,
                    candidate.local_shape,
                    fragment.global_offset,
                    fragment.local_shape,
                    strict=True,
                )
            ):
                return True
        active.append(fragment)
    return False
