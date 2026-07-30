"""Runtime inventory adapters and ephemeral binding contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .types import (
    ParallelRank,
    RuntimeBindingFragment,
    RuntimeFragment,
    TensorDescriptor,
    _canonical_stride,
    _canonical_tensor_descriptor,
    _read_aliases,
    _read_field,
    _read_optional_field,
    _require_integer,
    _require_integer_tuple,
    _require_manifest_items,
    _require_nonempty_string,
    _require_sequence,
    _require_sha256_digest,
    _require_u64,
    _validate_address_semantics,
)
from .validation import _validate_fragments, _validate_runtime_address_ranges


@dataclass(frozen=True)
class RuntimeBindingManifest:
    """Ephemeral physical locations and lifetime fence for one placement."""

    model_id: str
    revision: str
    placement_id: str
    placement_digest: str
    instance_id: str
    generation: int
    lease_id: str
    fragments: tuple[RuntimeBindingFragment, ...]

    def __post_init__(self) -> None:
        fragments = _require_manifest_items(
            self.fragments,
            "RuntimeBindingManifest fragments",
            RuntimeBindingFragment,
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
        for name in (
            "model_id",
            "revision",
            "placement_id",
            "instance_id",
            "lease_id",
        ):
            _require_nonempty_string(getattr(self, name), name)
        _require_sha256_digest(self.placement_digest, "placement_digest")
        _require_u64(self.generation, "generation")
        placement_ids = [item.placement_fragment_id for item in self.fragments]
        if len(placement_ids) != len(set(placement_ids)):
            raise ValueError("duplicate placement fragment in runtime binding")
        fragment_ids = [item.fragment_id for item in self.fragments]
        if len(fragment_ids) != len(set(fragment_ids)):
            raise ValueError("duplicate runtime fragment_id in runtime binding")

    @classmethod
    def from_runtime_inventory(
        cls,
        inventory: Any,
        *,
        owner_resolver: Optional[Callable[[Any], Any]] = None,
        address_semantics: Optional[str] = None,
    ) -> RuntimeBindingManifest:
        """Import framework runtime locations without importing the framework.

        A non-zero framework storage or byte offset requires
        ``address_semantics="view"`` to declare that each address is already
        normalized to the first byte of the transferable view.
        """

        _validate_address_semantics(
            address_semantics,
            has_nonzero_offset=False,
        )
        generation = _read_field(inventory, "generation")
        _require_u64(generation, "generation")
        return cls(
            model_id=_read_field(inventory, "model_id"),
            revision=_read_field(inventory, "revision"),
            placement_id=_read_field(inventory, "placement_id"),
            placement_digest=_read_field(inventory, "placement_digest"),
            instance_id=_read_field(inventory, "instance_id"),
            generation=generation,
            lease_id=_read_field(inventory, "lease_id"),
            fragments=tuple(
                _runtime_binding_fragment_from_record(
                    record,
                    owner_resolver,
                    address_semantics,
                    generation,
                )
                for record in _require_sequence(
                    _read_field(inventory, "fragments"),
                    "runtime binding fragments",
                )
            ),
        )


@dataclass(frozen=True)
class RuntimeManifest:
    """A validated runtime snapshot containing logical and physical facts."""

    model_id: str
    revision: str
    instance_id: str
    tensors: tuple[TensorDescriptor, ...]
    fragments: tuple[RuntimeFragment, ...]
    lease_id: Optional[str] = None
    placement_id: Optional[str] = None
    generation: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tensors",
            _require_manifest_items(
                self.tensors,
                "RuntimeManifest tensors",
                TensorDescriptor,
            ),
        )
        object.__setattr__(
            self,
            "fragments",
            _require_manifest_items(
                self.fragments,
                "RuntimeManifest fragments",
                RuntimeFragment,
            ),
        )
        for name in ("model_id", "revision", "instance_id"):
            _require_nonempty_string(getattr(self, name), name)
        if self.lease_id is not None:
            _require_nonempty_string(self.lease_id, "lease_id")
        if self.placement_id is not None:
            _require_nonempty_string(self.placement_id, "placement_id")
        if self.generation is not None:
            _require_u64(self.generation, "generation")
        fragment_generations = {
            fragment.lease_generation for fragment in self.fragments
        }
        if len(fragment_generations) > 1:
            raise ValueError("runtime manifest has inconsistent lease generations")
        if fragment_generations:
            fragment_generation = next(iter(fragment_generations))
            if self.generation is None:
                object.__setattr__(self, "generation", fragment_generation)
            elif self.generation != fragment_generation:
                raise ValueError(
                    "runtime manifest generation does not match fragment generation"
                )

        _validate_fragments(self.tensors, self.fragments)
        _validate_runtime_address_ranges(
            instance_id=self.instance_id,
            tensors=self.tensors,
            fragments=self.fragments,
        )

    @classmethod
    def from_runtime_inventory(
        cls,
        inventory: Any,
        *,
        owner_resolver: Optional[Callable[[Any], Any]] = None,
        address_semantics: Optional[str] = None,
    ) -> RuntimeManifest:
        """Import and validate a framework runtime tensor inventory.

        A non-zero framework view offset requires
        ``address_semantics="view"`` to declare that ``address`` is already
        normalized to the first byte of that view.
        """

        _validate_address_semantics(
            address_semantics,
            has_nonzero_offset=False,
        )
        generation = _read_field(inventory, "generation")
        _require_integer(generation, "generation", minimum=0)
        tensors: dict[str, TensorDescriptor] = {}
        fragments = []
        for record in _require_sequence(
            _read_field(inventory, "tensors"), "runtime inventory tensors"
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
                    f"runtime inventory descriptor mismatch: {descriptor.tensor_id}"
                )
            _validate_runtime_view(record, descriptor, address_semantics)
            rank = _read_field(record, "rank")
            lease_generation = _read_field(record, "lease_generation")
            _require_integer(lease_generation, "lease_generation", minimum=0)
            if lease_generation != generation:
                raise ValueError(
                    "runtime inventory lease generation mismatch: "
                    f"{lease_generation} != {generation}"
                )
            fragments.append(
                RuntimeFragment(
                    fragment_id=_read_field(record, "fragment_id"),
                    tensor_id=descriptor.tensor_id,
                    global_offset=_read_field(record, "global_offset"),
                    local_shape=_read_field(record, "local_shape"),
                    address=_read_field(record, "address"),
                    nbytes=_read_field(record, "nbytes"),
                    worker_id=_read_field(record, "worker_id"),
                    endpoint=_read_field(record, "endpoint"),
                    device=_read_field(record, "device"),
                    rank=ParallelRank(
                        dp=_read_field(rank, "dp"),
                        tp=_read_field(rank, "tp"),
                        pp=_read_field(rank, "pp"),
                        ep=_read_field(rank, "ep"),
                    ),
                    lease_generation=lease_generation,
                    owner=(
                        owner_resolver(record) if owner_resolver is not None else None
                    ),
                    aliases=_read_aliases(record),
                    placement_fragment_id=_read_optional_field(
                        record, "placement_fragment_id"
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
            placement_id=_read_optional_field(inventory, "placement_id"),
            generation=generation,
        )


def _validate_runtime_view(
    record: Any,
    descriptor: TensorDescriptor,
    address_semantics: Optional[str],
) -> None:
    is_contiguous = _read_field(record, "is_contiguous")
    if type(is_contiguous) is not bool or not is_contiguous:
        raise ValueError("runtime inventory tensor must be contiguous")

    local_shape = _require_integer_tuple(
        _read_field(record, "local_shape"), "local_shape", minimum=1
    )
    stride = _require_integer_tuple(_read_field(record, "stride"), "stride", minimum=0)
    if len(stride) != len(local_shape):
        raise ValueError("runtime inventory stride rank mismatch")
    canonical_stride = _canonical_stride(local_shape)
    if any(
        extent != 1 and actual != canonical
        for extent, actual, canonical in zip(local_shape, stride, canonical_stride)
    ):
        raise ValueError("runtime inventory tensor must use canonical stride")

    storage_offset = _read_field(record, "storage_offset")
    byte_offset = _read_field(record, "byte_offset")
    _require_integer(storage_offset, "storage_offset", minimum=0)
    _require_integer(byte_offset, "byte_offset", minimum=0)
    if byte_offset % descriptor.itemsize != 0:
        raise ValueError("runtime inventory byte_offset must be item-aligned")
    _validate_address_semantics(
        address_semantics,
        has_nonzero_offset=storage_offset != 0 or byte_offset != 0,
    )


def _runtime_binding_fragment_from_record(
    record: Any,
    owner_resolver: Optional[Callable[[Any], Any]],
    address_semantics: Optional[str],
    expected_generation: int,
) -> RuntimeBindingFragment:
    is_contiguous = _read_field(record, "is_contiguous")
    if type(is_contiguous) is not bool or not is_contiguous:
        raise ValueError("runtime binding allocation must be contiguous")
    fragment_generation = _read_optional_field(record, "lease_generation")
    if fragment_generation is not None:
        _require_u64(fragment_generation, "lease_generation")
        if fragment_generation != expected_generation:
            raise ValueError("runtime binding fragment lease generation mismatch")
    storage_offset = _read_optional_field(record, "storage_offset")
    byte_offset = _read_optional_field(record, "byte_offset")
    if storage_offset is not None:
        _require_integer(storage_offset, "storage_offset", minimum=0)
    if byte_offset is not None:
        _require_integer(byte_offset, "byte_offset", minimum=0)
    _validate_address_semantics(
        address_semantics,
        has_nonzero_offset=(
            storage_offset not in (None, 0) or byte_offset not in (None, 0)
        ),
    )
    return RuntimeBindingFragment(
        placement_fragment_id=_read_field(record, "placement_fragment_id"),
        fragment_id=_read_field(record, "fragment_id"),
        address=_read_field(record, "address"),
        nbytes=_read_field(record, "nbytes"),
        worker_id=_read_field(record, "worker_id"),
        endpoint=_read_field(record, "endpoint"),
        device=_read_field(record, "device"),
        owner=(owner_resolver(record) if owner_resolver is not None else None),
    )
