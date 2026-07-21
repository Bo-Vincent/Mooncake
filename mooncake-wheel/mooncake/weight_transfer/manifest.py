from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from math import prod
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class ParallelRank:
    dp: int = 0
    tp: int = 0
    pp: int = 0
    ep: int = 0

    def __post_init__(self) -> None:
        for name in ("dp", "tp", "pp", "ep"):
            _require_integer(getattr(self, name), f"parallel rank {name}", minimum=0)


@dataclass(frozen=True)
class TensorDescriptor:
    tensor_id: str
    global_shape: tuple[int, ...]
    dtype: str
    itemsize: int
    partition_dim: int | None
    layer_id: int | None = None
    expert_id: int | None = None
    layout_fingerprint: str = "contiguous"
    shard_dims: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        shape = _require_integer_tuple(self.global_shape, "global_shape", minimum=1)
        if not shape:
            raise ValueError("global_shape must not be empty")
        object.__setattr__(self, "global_shape", shape)
        _require_nonempty_string(self.tensor_id, "tensor_id")
        _require_nonempty_string(self.dtype, "dtype")
        _require_integer(self.itemsize, "itemsize", minimum=1)
        if self.partition_dim is not None:
            _require_integer(self.partition_dim, "partition_dim", minimum=0)
            if self.partition_dim >= len(shape):
                raise ValueError("partition_dim is out of range")
        if self.shard_dims is not None:
            shard_dims = _require_integer_tuple(
                self.shard_dims, "shard_dims", minimum=0
            )
            if len(shard_dims) != len(set(shard_dims)):
                raise ValueError("shard_dims must not contain duplicates")
            if any(dim >= len(shape) for dim in shard_dims):
                raise ValueError("shard_dims contains an out-of-range dimension")
            if tuple(sorted(shard_dims)) != shard_dims:
                raise ValueError("shard_dims must be sorted")
            if self.partition_dim is not None and shard_dims != (self.partition_dim,):
                raise ValueError("partition_dim conflicts with shard_dims")
            object.__setattr__(self, "shard_dims", shard_dims)
        for name in ("layer_id", "expert_id"):
            value = getattr(self, name)
            if value is not None:
                _require_integer(value, name, minimum=0)
        _require_nonempty_string(self.layout_fingerprint, "layout_fingerprint")

    @property
    def effective_shard_dims(self) -> tuple[int, ...]:
        if self.shard_dims is not None:
            return self.shard_dims
        if self.partition_dim is None:
            return ()
        return (self.partition_dim,)


@dataclass(frozen=True)
class RuntimeFragment:
    fragment_id: str
    tensor_id: str
    global_offset: tuple[int, ...]
    local_shape: tuple[int, ...]
    address: int
    nbytes: int
    worker_id: str
    endpoint: str
    rank: ParallelRank
    lease_generation: int
    owner: Any = field(default=None, compare=False, repr=False)
    aliases: tuple[str, ...] = ()

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
        for name in ("fragment_id", "tensor_id", "worker_id", "endpoint"):
            _require_nonempty_string(getattr(self, name), name)
        _require_integer(self.address, "address", minimum=1)
        _require_integer(self.nbytes, "nbytes", minimum=1)
        _require_integer(self.lease_generation, "lease_generation", minimum=0)
        if isinstance(self.aliases, (str, bytes, bytearray)):
            raise ValueError("aliases must contain non-empty strings")
        try:
            aliases = tuple(self.aliases)
        except TypeError as error:
            raise ValueError("aliases must contain non-empty strings") from error
        if any(type(alias) is not str or not alias for alias in aliases):
            raise ValueError("aliases must contain non-empty strings")
        if len(aliases) != len(set(aliases)):
            raise ValueError("aliases must not contain duplicates")
        object.__setattr__(self, "aliases", tuple(sorted(aliases)))
        if not isinstance(self.rank, ParallelRank):
            raise ValueError("rank must be a ParallelRank")


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


Fragment = RuntimeFragment | StoredFragment


def _require_nonempty_string(value: Any, name: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _require_integer(
    value: Any,
    name: str,
    *,
    minimum: int | None = None,
) -> None:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _require_integer_tuple(
    value: Any,
    name: str,
    *,
    minimum: int,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must contain integers")
    try:
        result = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must contain integers") from error
    for item in result:
        _require_integer(item, name, minimum=minimum)
    return result


def _read_field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value[name]
    return getattr(value, name)


def _read_optional_field(value: Any, name: str) -> Any | None:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _read_aliases(value: Any) -> tuple[str, ...]:
    aliases = _read_optional_field(value, "aliases")
    if aliases is None:
        return ()
    if isinstance(aliases, (str, bytes, bytearray)) or not isinstance(
        aliases, Sequence
    ):
        raise ValueError("aliases must be a sequence of non-empty strings")
    return tuple(aliases)


def _canonical_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = []
    running = 1
    for extent in reversed(shape):
        stride.append(running)
        running *= extent
    return tuple(reversed(stride))


def _validate_runtime_view(record: Any, descriptor: TensorDescriptor) -> None:
    is_contiguous = _read_field(record, "is_contiguous")
    if type(is_contiguous) is not bool or not is_contiguous:
        raise ValueError("runtime inventory tensor must be contiguous")

    local_shape = _require_integer_tuple(
        _read_field(record, "local_shape"), "local_shape", minimum=1
    )
    stride = _require_integer_tuple(_read_field(record, "stride"), "stride", minimum=0)
    canonical_stride = _canonical_stride(local_shape)
    if any(
        extent != 1 and actual != canonical
        for extent, actual, canonical in zip(
            local_shape, stride, canonical_stride, strict=True
        )
    ):
        raise ValueError("runtime inventory tensor must use canonical stride")

    storage_offset = _read_field(record, "storage_offset")
    byte_offset = _read_field(record, "byte_offset")
    _require_integer(storage_offset, "storage_offset", minimum=0)
    _require_integer(byte_offset, "byte_offset", minimum=0)
    if byte_offset != storage_offset * descriptor.itemsize:
        raise ValueError("runtime inventory byte_offset does not match storage_offset")


def _require_exact_fields(
    value: Any, expected: frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} schema fields do not match format_version")
    return value


def _validate_fragments(
    tensors: Sequence[TensorDescriptor], fragments: Sequence[Fragment]
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


def _validate_fragment_geometry(tensor: TensorDescriptor, fragment: Fragment) -> None:
    ndim = len(tensor.global_shape)
    if len(fragment.global_offset) != ndim or len(fragment.local_shape) != ndim:
        raise ValueError(f"fragment rank mismatch: {fragment.fragment_id}")
    for offset, extent, total in zip(
        fragment.global_offset, fragment.local_shape, tensor.global_shape
    ):
        if offset < 0 or extent < 0 or offset + extent > total:
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


def _validate_stored_coverage(
    tensors: Sequence[TensorDescriptor], fragments: Sequence[StoredFragment]
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


@dataclass(frozen=True)
class RuntimeManifest:
    model_id: str
    revision: str
    instance_id: str
    tensors: tuple[TensorDescriptor, ...]
    fragments: tuple[RuntimeFragment, ...]
    lease_id: str | None = None
    format_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensors", tuple(self.tensors))
        object.__setattr__(self, "fragments", tuple(self.fragments))
        for name in ("model_id", "revision", "instance_id"):
            _require_nonempty_string(getattr(self, name), name)
        _require_integer(self.format_version, "format_version", minimum=1)
        if self.format_version not in (1, 2):
            raise ValueError(
                f"unsupported runtime inventory format_version: {self.format_version}"
            )
        if self.lease_id is not None:
            _require_nonempty_string(self.lease_id, "lease_id")
        if not all(isinstance(item, TensorDescriptor) for item in self.tensors):
            raise ValueError("RuntimeManifest tensors must be TensorDescriptor")
        if not all(isinstance(item, RuntimeFragment) for item in self.fragments):
            raise ValueError("RuntimeManifest fragments must be RuntimeFragment")
        if self.format_version == 1 and any(
            tensor.shard_dims is not None for tensor in self.tensors
        ):
            raise ValueError("runtime manifest v1 cannot contain shard_dims")
        _validate_fragments(self.tensors, self.fragments)

    @classmethod
    def from_runtime_inventory(
        cls,
        inventory: Any,
        *,
        owner_resolver: Callable[[Any], Any] | None = None,
    ) -> RuntimeManifest:
        format_version = _read_field(inventory, "format_version")
        _require_integer(format_version, "format_version", minimum=1)
        if format_version not in (1, 2):
            raise ValueError(
                f"unsupported runtime inventory format_version: {format_version}"
            )
        generation = _read_field(inventory, "generation")
        _require_integer(generation, "generation", minimum=0)
        tensors: dict[str, TensorDescriptor] = {}
        fragments = []
        for record in _read_field(inventory, "tensors"):
            descriptor = TensorDescriptor(
                tensor_id=_read_field(record, "tensor_id"),
                global_shape=tuple(_read_field(record, "global_shape")),
                dtype=_read_field(record, "dtype"),
                itemsize=_read_field(record, "itemsize"),
                partition_dim=_read_field(record, "partition_dim"),
                layer_id=_read_optional_field(record, "layer_id"),
                expert_id=_read_optional_field(record, "expert_id"),
                layout_fingerprint=_read_field(record, "layout_fingerprint"),
                shard_dims=(
                    _read_optional_field(record, "shard_dims")
                    if format_version == 2
                    else None
                ),
            )
            previous = tensors.setdefault(descriptor.tensor_id, descriptor)
            if previous != descriptor:
                raise ValueError(
                    f"runtime inventory descriptor mismatch: {descriptor.tensor_id}"
                )
            _validate_runtime_view(record, descriptor)
            rank = _read_field(record, "rank")
            lease_generation = _read_field(record, "lease_generation")
            _require_integer(lease_generation, "lease_generation", minimum=0)
            if lease_generation != generation:
                raise ValueError(
                    f"runtime inventory lease generation mismatch: "
                    f"{lease_generation} != {generation}"
                )
            fragments.append(
                RuntimeFragment(
                    fragment_id=_read_field(record, "fragment_id"),
                    tensor_id=descriptor.tensor_id,
                    global_offset=tuple(_read_field(record, "global_offset")),
                    local_shape=tuple(_read_field(record, "local_shape")),
                    address=_read_field(record, "address"),
                    nbytes=_read_field(record, "nbytes"),
                    worker_id=_read_field(record, "worker_id"),
                    endpoint=_read_field(record, "endpoint"),
                    rank=ParallelRank(
                        dp=_read_field(rank, "dp"),
                        tp=_read_field(rank, "tp"),
                        pp=_read_field(rank, "pp"),
                        ep=_read_field(rank, "ep"),
                    ),
                    lease_generation=lease_generation,
                    aliases=_read_aliases(record),
                    owner=(
                        owner_resolver(record) if owner_resolver is not None else None
                    ),
                )
            )
        return cls(
            model_id=_read_field(inventory, "model_id"),
            revision=_read_field(inventory, "revision"),
            instance_id=_read_field(inventory, "instance_id"),
            tensors=tuple(sorted(tensors.values(), key=lambda item: item.tensor_id)),
            fragments=tuple(fragments),
            lease_id=_read_optional_field(inventory, "lease_id"),
            format_version=format_version,
        )


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
    format_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensors", tuple(self.tensors))
        object.__setattr__(self, "fragments", tuple(self.fragments))
        _require_integer(self.format_version, "format_version", minimum=1)
        if self.format_version not in (1, 2):
            raise ValueError(f"unsupported format_version: {self.format_version}")
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
        if not all(isinstance(item, TensorDescriptor) for item in self.tensors):
            raise ValueError("WeightManifest tensors must be TensorDescriptor")
        if not all(isinstance(item, StoredFragment) for item in self.fragments):
            raise ValueError("WeightManifest fragments must be StoredFragment")
        if self.format_version == 1 and any(
            tensor.shard_dims is not None for tensor in self.tensors
        ):
            raise ValueError("weight manifest v1 cannot contain shard_dims")
        payload_prefix = f"{self.group_id}/payload/"
        if any(
            not fragment.object_key.startswith(payload_prefix)
            for fragment in self.fragments
        ):
            raise ValueError("payload object_key does not belong to manifest group")
        _validate_fragments(self.tensors, self.fragments)
        _validate_stored_coverage(self.tensors, self.fragments)

    def to_json(self) -> str:
        tensors = []
        for tensor in self.tensors:
            item = {
                "tensor_id": tensor.tensor_id,
                "global_shape": tensor.global_shape,
                "dtype": tensor.dtype,
                "itemsize": tensor.itemsize,
                "partition_dim": tensor.partition_dim,
                "layer_id": tensor.layer_id,
                "expert_id": tensor.expert_id,
                "layout_fingerprint": tensor.layout_fingerprint,
            }
            if self.format_version == 2:
                item["shard_dims"] = tensor.effective_shard_dims
            tensors.append(item)
        raw = {
            "namespace": self.namespace,
            "model_id": self.model_id,
            "revision": self.revision,
            "group_id": self.group_id,
            "manifest_key": self.manifest_key,
            "tensors": tensors,
            "fragments": [asdict(fragment) for fragment in self.fragments],
            "created_at": self.created_at,
            "format_version": self.format_version,
        }
        return json.dumps(
            raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> WeightManifest:
        def reject_constant(constant: str) -> None:
            raise ValueError(f"non-finite JSON number is unsupported: {constant}")

        raw = json.loads(value, parse_constant=reject_constant)
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
                    "format_version",
                }
            ),
            "weight manifest",
        )
        format_version = raw["format_version"]
        _require_integer(format_version, "format_version", minimum=1)
        if format_version not in (1, 2):
            raise ValueError(f"unsupported format_version: {format_version}")
        tensor_fields = {
            "tensor_id",
            "global_shape",
            "dtype",
            "itemsize",
            "partition_dim",
            "layer_id",
            "expert_id",
            "layout_fingerprint",
        }
        if format_version == 2:
            tensor_fields.add("shard_dims")
        tensor_fields = frozenset(tensor_fields)
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
            format_version=format_version,
        )
