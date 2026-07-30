"""Serializable logical placement and canonical identity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

from .types import (
    ParallelRank,
    PlacementFragment,
    TensorDescriptor,
    _canonical_tensor_descriptor,
    _read_aliases,
    _read_field,
    _read_optional_field,
    _require_manifest_items,
    _require_nonempty_string,
    _require_sequence,
)
from .validation import _validate_fragments


@dataclass(frozen=True)
class PlacementManifest:
    """Serializable address-free placement with a canonical content ID."""

    model_id: str
    revision: str
    tensors: tuple[TensorDescriptor, ...]
    fragments: tuple[PlacementFragment, ...]
    placement_id: Optional[str] = None

    def __post_init__(self) -> None:
        tensors = _require_manifest_items(
            self.tensors, "PlacementManifest tensors", TensorDescriptor
        )
        tensors = tuple(_canonical_tensor_descriptor(tensor) for tensor in tensors)
        fragments = _require_manifest_items(
            self.fragments, "PlacementManifest fragments", PlacementFragment
        )
        object.__setattr__(
            self,
            "tensors",
            tuple(sorted(tensors, key=lambda item: item.tensor_id)),
        )
        object.__setattr__(
            self,
            "fragments",
            tuple(
                sorted(
                    fragments,
                    key=lambda item: item.placement_fragment_id,
                )
            ),
        )
        for name in ("model_id", "revision"):
            _require_nonempty_string(getattr(self, name), name)
        _validate_fragments(self.tensors, self.fragments)
        canonical_placement_id = _logical_placement_id(
            model_id=self.model_id,
            revision=self.revision,
            tensors=self.tensors,
            fragments=self.fragments,
        )
        if self.placement_id is None:
            object.__setattr__(self, "placement_id", canonical_placement_id)
        else:
            _require_nonempty_string(self.placement_id, "placement_id")
            if self.placement_id != canonical_placement_id:
                raise ValueError(
                    "placement_id does not match canonical logical content"
                )

    @property
    def digest(self) -> str:
        """Return the stable SHA-256 digest of the canonical JSON form."""

        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def to_json(self) -> str:
        """Serialize logical placement without any runtime location."""

        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> PlacementManifest:
        """Parse a strict placement manifest."""

        manifest = _require_exact_fields(
            _load_json_object(value, "placement manifest"),
            frozenset(
                {
                    "model_id",
                    "revision",
                    "placement_id",
                    "tensors",
                    "fragments",
                }
            ),
            "placement manifest",
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
        tensors = []
        for index, item in enumerate(
            _require_sequence(manifest["tensors"], "placement tensors")
        ):
            tensor = _require_exact_fields(
                item, tensor_fields, f"placement tensor {index}"
            )
            tensors.append(
                TensorDescriptor(
                    tensor_id=tensor["tensor_id"],
                    global_shape=tensor["global_shape"],
                    dtype=tensor["dtype"],
                    itemsize=tensor["itemsize"],
                    partition_dim=tensor["partition_dim"],
                    layer_id=tensor["layer_id"],
                    expert_id=tensor["expert_id"],
                    layout_fingerprint=tensor["layout_fingerprint"],
                    shard_dims=(
                        tensor["shard_dims"]
                        if tensor["shard_dims"] is not None
                        else None
                    ),
                )
            )

        fragment_fields = frozenset(
            {
                "placement_fragment_id",
                "tensor_id",
                "global_offset",
                "local_shape",
                "nbytes",
                "rank",
                "aliases",
            }
        )
        rank_fields = frozenset({"dp", "tp", "pp", "ep"})
        fragments = []
        for index, item in enumerate(
            _require_sequence(manifest["fragments"], "placement fragments")
        ):
            fragment = _require_exact_fields(
                item, fragment_fields, f"placement fragment {index}"
            )
            rank = _require_exact_fields(
                fragment["rank"], rank_fields, f"placement rank {index}"
            )
            fragments.append(
                PlacementFragment(
                    placement_fragment_id=fragment["placement_fragment_id"],
                    tensor_id=fragment["tensor_id"],
                    global_offset=fragment["global_offset"],
                    local_shape=fragment["local_shape"],
                    nbytes=fragment["nbytes"],
                    rank=ParallelRank(**rank),
                    aliases=fragment["aliases"],
                )
            )
        return cls(
            model_id=manifest["model_id"],
            revision=manifest["revision"],
            placement_id=manifest["placement_id"],
            tensors=tuple(tensors),
            fragments=tuple(fragments),
        )

    @classmethod
    def from_runtime_inventory(cls, inventory: Any) -> PlacementManifest:
        """Import framework-provided logical placement records."""

        tensors: dict[str, TensorDescriptor] = {}
        fragments = []
        for record in _require_sequence(
            _read_field(inventory, "tensors"), "placement inventory tensors"
        ):
            shard_dims = _read_optional_field(record, "shard_dims")
            descriptor = TensorDescriptor(
                tensor_id=_read_field(record, "tensor_id"),
                global_shape=_read_field(record, "global_shape"),
                dtype=_read_field(record, "dtype"),
                itemsize=_read_field(record, "itemsize"),
                partition_dim=_read_field(record, "partition_dim"),
                layer_id=_read_optional_field(record, "layer_id"),
                expert_id=_read_optional_field(record, "expert_id"),
                layout_fingerprint=_read_field(record, "layout_fingerprint"),
                shard_dims=shard_dims,
            )
            descriptor = _canonical_tensor_descriptor(descriptor)
            previous = tensors.setdefault(descriptor.tensor_id, descriptor)
            if previous != descriptor:
                raise ValueError(
                    f"placement descriptor mismatch: {descriptor.tensor_id}"
                )
            rank = _read_field(record, "rank")
            fragments.append(
                PlacementFragment(
                    placement_fragment_id=_read_field(record, "placement_fragment_id"),
                    tensor_id=descriptor.tensor_id,
                    global_offset=_read_field(record, "global_offset"),
                    local_shape=_read_field(record, "local_shape"),
                    nbytes=_read_field(record, "nbytes"),
                    rank=ParallelRank(
                        dp=_read_field(rank, "dp"),
                        tp=_read_field(rank, "tp"),
                        pp=_read_field(rank, "pp"),
                        ep=_read_field(rank, "ep"),
                    ),
                    aliases=_read_aliases(record),
                )
            )
        return cls(
            model_id=_read_field(inventory, "model_id"),
            revision=_read_field(inventory, "revision"),
            tensors=tuple(sorted(tensors.values(), key=lambda item: item.tensor_id)),
            fragments=tuple(fragments),
            placement_id=_read_optional_field(inventory, "placement_id"),
        )


def _canonical_json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _logical_placement_id(
    *,
    model_id: str,
    revision: str,
    tensors: Sequence[TensorDescriptor],
    fragments: Sequence[PlacementFragment],
) -> str:
    content = {
        "schema": "weight-placement",
        "model_id": model_id,
        "revision": revision,
        "tensors": [
            asdict(_canonical_tensor_descriptor(tensor))
            for tensor in sorted(tensors, key=lambda item: item.tensor_id)
        ],
        "fragments": [
            asdict(fragment)
            for fragment in sorted(
                fragments,
                key=lambda item: item.placement_fragment_id,
            )
        ],
    }
    return f"sha256:{_canonical_json_digest(content)}"


def _require_exact_fields(
    value: Any, expected: frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} schema fields do not match contract")
    return value


def _load_json_object(value: str, label: str) -> Mapping[str, Any]:
    def reject_constant(constant: str) -> None:
        raise ValueError(f"non-finite JSON number is unsupported: {constant}")

    def reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = item
        return result

    try:
        raw = json.loads(
            value,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_fields,
        )
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return raw
