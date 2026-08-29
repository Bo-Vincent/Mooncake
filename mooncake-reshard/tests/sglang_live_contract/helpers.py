from __future__ import annotations

import ctypes
import importlib
import importlib.util
import os
import sys
from contextlib import contextmanager
from itertools import product
from math import prod
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from mooncake.reshard._compat import _strict_zip
from mooncake.reshard.weight import (
    OwnershipAxis,
    ParallelRank,
    ParallelTopology,
    PlacementFragment,
    ReplicatedAxis,
    RuntimeBindingFragment,
    SplitAxis,
    TensorDescriptor,
    TopologyParticipant,
    WeightPlacementManifest,
    WeightPlacementPart,
    WeightRuntimeBindingManifest,
)


_MISSING = object()


def _parallel_axes_for_tensor(
    tensor,
    *,
    tp_size: int,
    pp_size: int,
    ep_size: int,
    dp_size: int,
):
    axes = []
    if dp_size > 1:
        axes.append(ReplicatedAxis("dp"))
    if pp_size > 1:
        axes.append(OwnershipAxis("pp"))
    shard_dims = tuple(tensor.get("shard_dims") or ())
    ep_dim = tensor.get("expert_axis")
    if tensor.get("expert_id") is not None:
        ep_dim = None
        axes.append(OwnershipAxis("ep"))
    elif ep_dim is not None or (ep_size > 1 and 0 in shard_dims):
        ep_dim = 0 if ep_dim is None else ep_dim
        axes.append(SplitAxis("ep", dim=ep_dim))
    tp_dims = tuple(dim for dim in shard_dims if dim != ep_dim)
    if tp_dims:
        if len(tp_dims) != 1:
            raise ValueError("SGLang tensor has unsupported TP shard semantics")
        axes.append(SplitAxis("tp", dim=tp_dims[0]))
    return tuple(axes)


def _axis_to_fixture(axis):
    if isinstance(axis, SplitAxis):
        return {"semantics": "split", "kind": axis.kind, "dim": axis.dim}
    if isinstance(axis, ReplicatedAxis):
        return {"semantics": "replicated", "kind": axis.kind}
    return {"semantics": "ownership", "kind": axis.kind}


def _contiguous_strides_bytes(local_shape, itemsize: int):
    stride = itemsize
    result = []
    for extent in reversed(local_shape):
        result.append(stride)
        stride *= extent
    return list(reversed(result))


def _normalize_aliases(tensor) -> None:
    aliases = tuple(tensor.get("aliases") or ())
    tensor["aliases"] = list(aliases) if len(aliases) > 1 else []


def _normalize_sglang_runtime_fixture(
    fixture,
    *,
    tp_size: int,
    pp_size: int,
    ep_size: int,
    dp_size: int,
):
    tensors = fixture["tensors"]
    allocation_ends = {}
    for tensor in tensors:
        _normalize_aliases(tensor)
        axes = _parallel_axes_for_tensor(
            tensor, tp_size=tp_size, pp_size=pp_size, ep_size=ep_size, dp_size=dp_size
        )
        tensor["parallel_axes"] = [_axis_to_fixture(axis) for axis in axes]
        tensor.pop("partition_dim", None)
        tensor["rank"].setdefault("moe_dp", 0)
        itemsize = tensor["itemsize"]
        offset_bytes = tensor.get("storage_offset_bytes")
        if offset_bytes is None:
            offset_bytes = tensor.get("storage_offset", 0) * itemsize
        tensor["storage_offset_bytes"] = offset_bytes
        storage_address = tensor.get(
            "storage_address", tensor["address"] - offset_bytes
        )
        tensor["storage_address"] = storage_address
        tensor["strides_bytes"] = tensor.get(
            "strides_bytes",
            [stride * itemsize for stride in tensor["stride"]],
        )
        allocation_ends[storage_address] = max(
            allocation_ends.get(storage_address, 0),
            offset_bytes + tensor["nbytes"],
        )
    for tensor in tensors:
        tensor["storage_nbytes"] = allocation_ends[tensor["storage_address"]]
    return fixture


def _contract_required() -> bool:
    return os.environ.get("SGLANG_CONTRACT_REQUIRED") == "1"


def _load_msgspec():
    try:
        return importlib.import_module("msgspec")
    except ModuleNotFoundError as error:
        if error.name != "msgspec":
            raise
        if _contract_required():
            pytest.fail("msgspec is required for the live SGLang contract")
        pytest.skip("msgspec is required for the live SGLang contract")


def _adapt_sglang_parts(
    runtime_parts,
    msgspec,
    *,
    placement_set_id: str,
    tp_size: int,
    pp_size: int,
    ep_size: int,
    dp_size: int,
    weight_generation: int = 0,
):
    """Explicit SGLang-side adapter into Mooncake canonical typed contracts."""

    local = []
    for parts in runtime_parts:
        placement_inventory = msgspec.to_builtins(parts.placement)
        participant_id = placement_inventory["placement_id"]
        placement_inventory.pop("format_version", None)
        placement_inventory.pop("placement_id", None)
        binding_inventory = msgspec.to_builtins(parts.binding)
        binding_inventory.pop("format_version", None)
        tensors = placement_inventory["tensors"]
        if not tensors:
            raise ValueError("SGLang placement part must contain tensors")
        ranks = {
            tuple(tensor["rank"][axis] for axis in ("dp", "tp", "pp", "ep"))
            for tensor in tensors
        }
        if len(ranks) != 1:
            raise ValueError("SGLang placement part spans multiple parallel ranks")
        dp, tp, pp, ep = next(iter(ranks))
        rank = ParallelRank(dp=dp, tp=tp, pp=pp, ep=ep)
        descriptors = {}
        placement_fragments = []
        placement_by_id = {}
        for tensor in tensors:
            _normalize_aliases(tensor)
            axes = _parallel_axes_for_tensor(
                tensor,
                tp_size=tp_size,
                pp_size=pp_size,
                ep_size=ep_size,
                dp_size=dp_size,
            )
            placement_by_id[tensor["placement_fragment_id"]] = tensor
            descriptor = TensorDescriptor(
                tensor_id=tensor["tensor_id"],
                global_shape=tuple(tensor["global_shape"]),
                dtype=tensor["dtype"],
                itemsize=tensor["itemsize"],
                shard_dims=tuple(tensor.get("shard_dims") or ()),
                layout_fingerprint=tensor["layout_fingerprint"],
                parallel_axes=axes,
                layer_id=tensor.get("layer_id"),
                expert_id=tensor.get("expert_id"),
            )
            previous = descriptors.setdefault(descriptor.tensor_id, descriptor)
            if previous != descriptor:
                raise ValueError("SGLang tensor descriptors disagree")
            placement_fragments.append(
                PlacementFragment(
                    placement_fragment_id=tensor["placement_fragment_id"],
                    tensor_id=tensor["tensor_id"],
                    global_offset=tuple(tensor["global_offset"]),
                    local_shape=tuple(tensor["local_shape"]),
                    nbytes=tensor["nbytes"],
                    rank=rank,
                    aliases=tuple(tensor["aliases"]),
                )
            )
        storage_ends = {}
        for fragment in binding_inventory["fragments"]:
            tensor = placement_by_id[fragment["placement_fragment_id"]]
            itemsize = tensor["itemsize"]
            offset_bytes = fragment.get("storage_offset_bytes")
            if offset_bytes is None:
                offset_bytes = fragment.get("storage_offset", 0) * itemsize
            storage_address = fragment.get(
                "storage_address", fragment["address"] - offset_bytes
            )
            fragment.update(
                itemsize=itemsize,
                local_shape=tensor["local_shape"],
                strides_bytes=_contiguous_strides_bytes(
                    tensor["local_shape"], itemsize
                ),
                storage_address=storage_address,
                storage_offset_bytes=offset_bytes,
            )
            storage_ends[storage_address] = max(
                storage_ends.get(storage_address, 0),
                offset_bytes + fragment["nbytes"],
            )
        runtime_fragments = []
        for fragment in binding_inventory["fragments"]:
            tensor = placement_by_id[fragment["placement_fragment_id"]]
            fragment["storage_nbytes"] = storage_ends[fragment["storage_address"]]
            runtime_fragments.append(
                RuntimeBindingFragment(
                    placement_fragment_id=fragment["placement_fragment_id"],
                    fragment_id=fragment["fragment_id"],
                    address=fragment["address"],
                    nbytes=fragment["nbytes"],
                    worker_id=fragment["worker_id"],
                    endpoint=fragment["endpoint"],
                    device=fragment["device"],
                    itemsize=tensor["itemsize"],
                    local_shape=tuple(tensor["local_shape"]),
                    strides_bytes=tuple(fragment["strides_bytes"]),
                    storage_address=fragment["storage_address"],
                    storage_nbytes=fragment["storage_nbytes"],
                    storage_offset_bytes=fragment["storage_offset_bytes"],
                )
            )
        local.append(
            (
                placement_inventory,
                binding_inventory,
                participant_id,
                rank,
                tuple(descriptors.values()),
                tuple(placement_fragments),
                tuple(runtime_fragments),
            )
        )

    topology = ParallelTopology(
        tp_size=tp_size,
        pp_size=pp_size,
        ep_size=ep_size,
        dp_size=dp_size,
        participants=tuple(
            TopologyParticipant(participant_id, rank)
            for _, _, participant_id, rank, _, _, _ in local
        ),
    )
    part_manifests = []
    for placement_inventory, _, participant_id, rank, tensors, fragments, _ in local:
        part_manifests.append(
            WeightPlacementPart(
                resource_id=placement_inventory["model_id"],
                revision=placement_inventory["revision"],
                weight_generation=weight_generation,
                placement_set_id=placement_set_id,
                topology_id=topology.topology_id,
                participant_id=participant_id,
                rank=rank,
                tensors=tensors,
                fragments=fragments,
            )
        )

    first = part_manifests[0]
    placement = WeightPlacementManifest(
        resource_id=first.resource_id,
        revision=first.revision,
        weight_generation=weight_generation,
        placement_set_id=placement_set_id,
        topology=topology,
        parts=tuple(part_manifests),
    )
    bindings = []
    for _, binding_inventory, participant_id, _, _, _, fragments in local:
        bindings.append(
            WeightRuntimeBindingManifest(
                resource_id=placement.resource_id,
                revision=placement.revision,
                placement_id=placement.placement_id,
                placement_digest=placement.digest,
                instance_id=binding_inventory["instance_id"],
                participant_id=participant_id,
                generation=binding_inventory["generation"],
                lease_id=binding_inventory["lease_id"],
                fragments=fragments,
            )
        )
    return placement, tuple(bindings)


def _sglang_source_root() -> Path:
    source_root = os.environ.get("SGLANG_SOURCE_ROOT")
    if not source_root:
        if _contract_required():
            pytest.fail(
                "SGLANG_SOURCE_ROOT is required when SGLANG_CONTRACT_REQUIRED=1"
            )
        pytest.skip("SGLANG_SOURCE_ROOT is required for the live contract test")
    return Path(source_root)


def _load_source_module(module_name: str, module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        pytest.fail(f"cannot load SGLang module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def _load_sglang_contract_modules():
    source_root = _sglang_source_root()
    package_root = source_root / "python/sglang"
    module_path = package_root / "srt/model_executor/weight_runtime_manifest.py"
    if not module_path.is_file():
        pytest.fail(f"SGLang manifest module does not exist: {module_path}")

    _load_msgspec()
    package_paths = {
        "sglang": package_root,
        "sglang.srt": package_root / "srt",
        "sglang.srt.model_executor": package_root / "srt/model_executor",
        "sglang.srt.model_executor.weight_semantics": (
            package_root / "srt/model_executor/weight_semantics"
        ),
    }
    loaded_names = (
        *package_paths,
        "sglang.srt.model_executor.weight_runtime_manifest",
        "sglang.srt.model_executor.weight_semantics.qwen3_5",
        "sglang.srt.model_executor.weight_semantics.qwen3",
        "sglang.srt.model_executor.weight_semantics.qwen3_next",
        "sglang.srt.model_executor.weight_semantics.fp8_block",
    )
    previous = {
        module_name: sys.modules.get(module_name, _MISSING)
        for module_name in loaded_names
    }
    try:
        for package_name, package_path in package_paths.items():
            package = ModuleType(package_name)
            package.__path__ = [str(package_path)]
            sys.modules[package_name] = package
        manifest = _load_source_module(
            "sglang.srt.model_executor.weight_runtime_manifest",
            module_path,
        )
        qwen = _load_source_module(
            "sglang.srt.model_executor.weight_semantics.qwen3_5",
            package_paths["sglang.srt.model_executor.weight_semantics"] / "qwen3_5.py",
        )
        _load_source_module(
            "sglang.srt.model_executor.weight_semantics.qwen3",
            package_paths["sglang.srt.model_executor.weight_semantics"] / "qwen3.py",
        )
        _load_source_module(
            "sglang.srt.model_executor.weight_semantics.qwen3_next",
            package_paths["sglang.srt.model_executor.weight_semantics"]
            / "qwen3_next.py",
        )
        fp8 = _load_source_module(
            "sglang.srt.model_executor.weight_semantics.fp8_block",
            package_paths["sglang.srt.model_executor.weight_semantics"]
            / "fp8_block.py",
        )
        yield SimpleNamespace(manifest=manifest, qwen=qwen, fp8=fp8)
    finally:
        for module_name in reversed(loaded_names):
            old_module = previous[module_name]
            if old_module is _MISSING:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = old_module


class _Storage:
    def __init__(self, address: int) -> None:
        self._address = address

    def data_ptr(self) -> int:
        return self._address


class _Parameter:
    def __init__(
        self,
        *,
        address: int,
        shape: tuple[int, ...],
        dtype: str = "float16",
        itemsize: int = 2,
    ) -> None:
        self._address = address
        self._itemsize = itemsize
        self.shape = shape
        self.dtype = dtype
        self.device = SimpleNamespace(type="cpu")
        self.layout = "strided"
        self.is_sparse = False

    def untyped_storage(self) -> _Storage:
        return _Storage(self._address)

    def storage_offset(self) -> int:
        return 0

    def stride(self) -> tuple[int, ...]:
        result = []
        value = 1
        for extent in reversed(self.shape):
            result.append(value)
            value *= extent
        return tuple(reversed(result))

    def is_contiguous(self) -> bool:
        return True

    def element_size(self) -> int:
        return self._itemsize

    def numel(self) -> int:
        value = 1
        for extent in self.shape:
            value *= extent
        return value

    def data_ptr(self) -> int:
        return self._address


class _BufferParameter(_Parameter):
    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        dtype: str = "float16",
        itemsize: int = 2,
    ) -> None:
        self.buffer = ctypes.create_string_buffer(prod(shape) * itemsize)
        super().__init__(
            address=ctypes.addressof(self.buffer),
            shape=shape,
            dtype=dtype,
            itemsize=itemsize,
        )

    def fill(self, value: int) -> None:
        ctypes.memset(self.data_ptr(), value, self.numel() * self.element_size())


class _Model:
    def __init__(self, parameter: _Parameter) -> None:
        self._parameter = parameter

    def named_parameters(self, *, remove_duplicate: bool):
        assert not remove_duplicate
        return (("layers.0.mlp.weight", self._parameter),)


class _NamedParameterModel:
    def __init__(self, parameters) -> None:
        self._parameters = tuple(parameters)

    def named_parameters(self, *, remove_duplicate: bool):
        assert not remove_duplicate
        return iter(self._parameters)


class _Fp8RuntimeModule:
    def __init__(
        self,
        *,
        weight: _Parameter,
        scale: _Parameter,
        up_first: bool,
    ) -> None:
        self.w13_weight = weight
        self.w13_weight_scale_inv = scale
        self.block_quant = True
        self.weight_block_size = [128, 128]
        quant_config = SimpleNamespace(
            activation_scheme="dynamic",
            is_checkpoint_fp8_serialized=True,
            use_mxfp8=False,
            weight_block_size=[128, 128],
        )
        self.quant_method = SimpleNamespace(
            block_quant=True,
            is_checkpoint_fp8_serialized=True,
            load_up_proj_weight_first=up_first,
            quant_config=quant_config,
            use_marlin=False,
            use_mxfp8=False,
            weight_block_size=[128, 128],
        )


class _Fp8Model(_NamedParameterModel):
    def __init__(self, parameters, *, runtime_module: _Fp8RuntimeModule) -> None:
        super().__init__(parameters)
        self._runtime_module = runtime_module

    def modules(self):
        return iter((self, self._runtime_module))


class _CopyingReadEngine:
    def __init__(self) -> None:
        self.calls = []

    def batch_transfer_sync_read(
        self,
        endpoint,
        target_addresses,
        source_addresses,
        sizes,
    ) -> int:
        self.calls.append(
            (
                endpoint,
                tuple(target_addresses),
                tuple(source_addresses),
                tuple(sizes),
            )
        )
        for target, source, nbytes in _strict_zip(
            target_addresses,
            source_addresses,
            sizes,
        ):
            ctypes.memmove(target, source, nbytes)
        return 0


def _fragment_payload(tensor, fragment) -> bytes:
    seed = sum(
        (index + 1) * value for index, value in enumerate(tensor.tensor_id.encode())
    )
    payload = bytearray()
    for local_coordinate in product(
        *(range(extent) for extent in fragment.local_shape)
    ):
        global_coordinate = tuple(
            offset + coordinate
            for offset, coordinate in _strict_zip(
                fragment.global_offset,
                local_coordinate,
            )
        )
        value = (
            seed
            + sum(
                (dimension + 1) * 257 * coordinate
                for dimension, coordinate in enumerate(global_coordinate)
            )
        ) & ((1 << (8 * tensor.itemsize)) - 1)
        payload.extend(value.to_bytes(tensor.itemsize, byteorder="little"))
    assert len(payload) == fragment.nbytes
    return bytes(payload)


def _write_logical_payloads(manifest) -> None:
    tensors = {tensor.tensor_id: tensor for tensor in manifest.tensors}
    for fragment in manifest.fragments:
        payload = _fragment_payload(tensors[fragment.tensor_id], fragment)
        ctypes.memmove(fragment.address, payload, len(payload))


def _assert_logical_payloads(manifest) -> None:
    tensors = {tensor.tensor_id: tensor for tensor in manifest.tensors}
    for fragment in manifest.fragments:
        expected = _fragment_payload(tensors[fragment.tensor_id], fragment)
        assert ctypes.string_at(fragment.address, fragment.nbytes) == expected
