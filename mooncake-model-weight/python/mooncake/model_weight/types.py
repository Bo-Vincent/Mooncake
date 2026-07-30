"""Public model-weight tensor and fragment contracts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence, Union


_MAX_U64 = (1 << 64) - 1


@dataclass(frozen=True)
class ParallelRank:
    """Framework-provided owner coordinates used only for routing.

    Logical sharding is defined by ``global_offset``, ``local_shape``, and
    ``shard_dims``. This coordinate is not a replacement for the topology and
    axis metadata used to synthesize a target placement.
    """

    dp: int = 0
    tp: int = 0
    pp: int = 0
    ep: int = 0

    def __post_init__(self) -> None:
        for name in ("dp", "tp", "pp", "ep"):
            _require_integer(getattr(self, name), f"parallel rank {name}", minimum=0)


@dataclass(frozen=True)
class TensorDescriptor:
    """Logical tensor identity, shape, dtype, and framework-supplied semantics."""

    tensor_id: str
    global_shape: tuple[int, ...]
    dtype: str
    itemsize: int
    partition_dim: Optional[int]
    layout_fingerprint: str
    layer_id: Optional[int] = None
    expert_id: Optional[int] = None
    shard_dims: Optional[tuple[int, ...]] = None

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
            if tuple(sorted(shard_dims)) != shard_dims:
                raise ValueError("shard_dims must be sorted")
            if any(dim >= len(shape) for dim in shard_dims):
                raise ValueError("shard_dims contains an out-of-range dimension")
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
        """Return normalized shard dimensions for single-axis and N-D inputs."""

        if self.shard_dims is not None:
            return self.shard_dims
        if self.partition_dim is None:
            return ()
        return (self.partition_dim,)


@dataclass(frozen=True)
class RuntimeFragment:
    """One contiguous runtime view backing a logical tensor box.

    ``address`` always points at the first byte of this view, not at the base
    of the underlying framework storage.
    """

    fragment_id: str
    tensor_id: str
    global_offset: tuple[int, ...]
    local_shape: tuple[int, ...]
    address: int
    nbytes: int
    worker_id: str
    endpoint: str
    device: str
    rank: ParallelRank
    lease_generation: int
    owner: Any = field(default=None, compare=False, repr=False)
    aliases: tuple[str, ...] = ()
    placement_fragment_id: Optional[str] = None

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
        for name in ("fragment_id", "tensor_id", "worker_id", "endpoint", "device"):
            _require_nonempty_string(getattr(self, name), name)
        _require_address_range(self.address, self.nbytes)
        _require_u64(self.lease_generation, "lease_generation")
        if not isinstance(self.rank, ParallelRank):
            raise ValueError("rank must be a ParallelRank")
        object.__setattr__(self, "aliases", _normalize_aliases(self.aliases))
        if self.placement_fragment_id is not None:
            _require_nonempty_string(
                self.placement_fragment_id, "placement_fragment_id"
            )


@dataclass(frozen=True)
class PlacementFragment:
    """An address-free logical tensor box assigned to one parallel rank."""

    placement_fragment_id: str
    tensor_id: str
    global_offset: tuple[int, ...]
    local_shape: tuple[int, ...]
    nbytes: int
    rank: ParallelRank
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
        for name in ("placement_fragment_id", "tensor_id"):
            _require_nonempty_string(getattr(self, name), name)
        _require_u64(self.nbytes, "nbytes", minimum=1)
        if not isinstance(self.rank, ParallelRank):
            raise ValueError("rank must be a ParallelRank")
        object.__setattr__(self, "aliases", _normalize_aliases(self.aliases))

    @property
    def fragment_id(self) -> str:
        """Expose the common fragment identifier used by future planners."""

        return self.placement_fragment_id


@dataclass(frozen=True)
class RuntimeBindingFragment:
    """Contiguous physical runtime view for one placement fragment."""

    placement_fragment_id: str
    fragment_id: str
    address: int
    nbytes: int
    worker_id: str
    endpoint: str
    device: str
    owner: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "placement_fragment_id",
            "fragment_id",
            "worker_id",
            "endpoint",
            "device",
        ):
            _require_nonempty_string(getattr(self, name), name)
        _require_address_range(self.address, self.nbytes)


ManifestFragment = Union[RuntimeFragment, PlacementFragment]


def _require_nonempty_string(value: Any, name: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _require_integer(
    value: Any,
    name: str,
    *,
    minimum: Optional[int] = None,
) -> None:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _require_u64(value: Any, name: str, *, minimum: int = 0) -> None:
    _require_integer(value, name, minimum=minimum)
    if value > _MAX_U64:
        raise ValueError(f"{name} must fit in an unsigned 64-bit integer")


def _require_address_range(address: Any, nbytes: Any) -> None:
    _require_u64(address, "address", minimum=1)
    _require_u64(nbytes, "nbytes", minimum=1)
    if nbytes > _MAX_U64 - address:
        raise ValueError("address range must fit in an unsigned 64-bit integer")


def _require_sha256_digest(value: Any, name: str) -> None:
    _require_nonempty_string(value, name)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_integer_tuple(
    value: Any,
    name: str,
    *,
    minimum: int,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must contain integers")
    result = tuple(value)
    for item in result:
        _require_integer(item, name, minimum=minimum)
    return result


def _require_sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return value


def _require_manifest_items(
    value: Any,
    name: str,
    item_type: type,
) -> tuple[Any, ...]:
    items = tuple(_require_sequence(value, name))
    if not all(isinstance(item, item_type) for item in items):
        raise ValueError(f"{name} must contain {item_type.__name__}")
    return items


def _read_field(value: Any, name: str) -> Any:
    try:
        if isinstance(value, Mapping):
            return value[name]
        return getattr(value, name)
    except (KeyError, AttributeError) as error:
        raise ValueError(f"missing required field: {name}") from error


def _read_optional_field(value: Any, name: str) -> Optional[Any]:
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


def _normalize_aliases(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError("aliases must contain non-empty strings")
    aliases = tuple(value)
    if any(type(alias) is not str or not alias for alias in aliases):
        raise ValueError("aliases must contain non-empty strings")
    if len(aliases) != len(set(aliases)):
        raise ValueError("aliases must not contain duplicates")
    return tuple(sorted(aliases))


def _canonical_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = []
    running = 1
    for extent in reversed(shape):
        stride.append(running)
        running *= extent
    return tuple(reversed(stride))


def _validate_address_semantics(
    address_semantics: Optional[str],
    *,
    has_nonzero_offset: bool,
) -> None:
    if address_semantics is not None and address_semantics != "view":
        raise ValueError("address_semantics must be 'view'")
    if has_nonzero_offset and address_semantics != "view":
        raise ValueError("non-zero runtime offsets require address_semantics='view'")


def _canonical_tensor_descriptor(tensor: TensorDescriptor) -> TensorDescriptor:
    shard_dims = tensor.effective_shard_dims
    partition_dim = shard_dims[0] if len(shard_dims) == 1 else None
    if tensor.shard_dims == shard_dims and tensor.partition_dim == partition_dim:
        return tensor
    return replace(
        tensor,
        partition_dim=partition_dim,
        shard_dims=shard_dims,
    )
