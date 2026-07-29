from __future__ import annotations

import ctypes
import importlib
import importlib.util
import json
import os
import sys
from contextlib import contextmanager
from itertools import product
from math import prod
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from mooncake.model_weight import (
    MemoryRegistrationLease,
    MooncakeTransferEngineReader,
    PlacementManifest,
    RuntimeBindingManifest,
    SourcePlacementManifest,
    TargetPlacementManifest,
    bind_logical_transfer_plan,
    bind_runtime_manifest,
    plan_placement_transfer_to_local_target,
)


_MISSING = object()


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


def test_required_contract_fails_when_msgspec_is_missing(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_CONTRACT_REQUIRED", "1")

    def missing_msgspec(module_name: str):
        raise ModuleNotFoundError(
            f"No module named {module_name!r}",
            name=module_name,
        )

    monkeypatch.setattr(importlib, "import_module", missing_msgspec)

    with pytest.raises(pytest.fail.Exception, match="msgspec is required"):
        _load_msgspec()


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
        for target, source, nbytes in zip(
            target_addresses,
            source_addresses,
            sizes,
            strict=True,
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
            for offset, coordinate in zip(
                fragment.global_offset,
                local_coordinate,
                strict=True,
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


def test_sglang_runtime_output_binds_with_mooncake_parser() -> None:
    with _load_sglang_contract_modules() as modules:
        sglang = modules.manifest
        msgspec = _load_msgspec()
        parameter = _Parameter(address=0x100000, shape=(4, 8))

        class Adapter:
            def describe_parameter(self, *, names, parameter, topology):
                assert names == ("layers.0.mlp.weight",)
                assert topology.tp_size == 2
                return (
                    sglang.LogicalTensorView(
                        tensor_id="model.layers.0.mlp.weight",
                        global_shape=(8, 8),
                        global_offset=(0, 0),
                        local_shape=(4, 8),
                        partition_dim=0,
                        shard_dims=(0,),
                        byte_offset=0,
                        layer_id=0,
                        expert_id=None,
                        layout_fingerprint="row-major-f16",
                    ),
                )

        manager = sglang.WeightRuntimeManifestManager(
            model=_Model(parameter),
            adapter=Adapter(),
            topology=sglang.WeightParallelTopology(tp_rank=0, tp_size=2),
            allowed_devices=("cpu",),
        )
        parts = manager.snapshot_parts(
            model_id="contract-model",
            revision="step-1",
            instance_id="target-0",
            worker_id="worker-0",
            endpoint="127.0.0.1:12345",
            lease_timeout_sec=30,
        )
        try:
            placement = PlacementManifest.from_runtime_inventory(
                msgspec.to_builtins(parts.placement)
            )
            binding = RuntimeBindingManifest.from_runtime_inventory(
                msgspec.to_builtins(parts.binding)
            )
            runtime = bind_runtime_manifest(placement, binding)

            assert runtime.model_id == "contract-model"
            assert runtime.revision == "step-1"
            assert runtime.placement_id == parts.placement.placement_id
            assert runtime.lease_id == parts.binding.lease_id
            assert runtime.fragments[0].address == parameter.data_ptr()
            assert runtime.fragments[0].global_offset == (0, 0)
            assert runtime.fragments[0].local_shape == (4, 8)
            assert runtime.fragments[0].rank.tp == 0
        finally:
            manager.release(parts.binding.lease_id)


def test_sglang_qwen_split_contract_plans_binds_and_executes() -> None:
    with _load_sglang_contract_modules() as modules:
        sglang = modules.manifest
        qwen = modules.qwen
        msgspec = _load_msgspec()
        source_root = _sglang_source_root()
        local_fixture = json.loads(
            (
                Path(__file__).parent / "fixtures/qwen3_5_moe_runtime_manifest.json"
            ).read_text()
        )
        sglang_fixture = json.loads(
            (
                source_root / "test/registered/unit/model_executor/fixtures/"
                "qwen3_5_moe_runtime_manifest.json"
            ).read_text()
        )
        assert local_fixture == sglang_fixture

        config = SimpleNamespace(
            num_experts=8,
            moe_intermediate_size=8,
            hidden_size=8,
        )
        source_managers = []
        source_parts = []
        target_manager = None
        target_parts = None
        try:
            for ep_rank in range(4):
                for tp_rank in range(4):
                    rank_name = f"source-p1-e{ep_rank}-t{tp_rank}"
                    w13 = _BufferParameter(shape=(2, 4, 8))
                    w2 = _BufferParameter(shape=(2, 8, 2))
                    manager = sglang.WeightRuntimeManifestManager(
                        model=_NamedParameterModel(
                            (
                                (
                                    "layers.10.mlp.experts.w13_weight",
                                    w13,
                                ),
                                (
                                    "layers.10.mlp.experts.w2_weight",
                                    w2,
                                ),
                            )
                        ),
                        adapter=qwen.Qwen35WeightSemanticsAdapter(config=config),
                        topology=sglang.WeightParallelTopology(
                            dp_rank=0,
                            dp_size=2,
                            tp_rank=tp_rank,
                            tp_size=4,
                            pp_rank=1,
                            pp_size=2,
                            ep_rank=ep_rank,
                            ep_size=4,
                            moe_tp_rank=tp_rank,
                            moe_tp_size=4,
                        ),
                        allowed_devices=("cpu",),
                    )
                    parts = manager.snapshot_parts(
                        model_id="qwen3.5-moe",
                        revision="step-42",
                        instance_id=rank_name,
                        worker_id=rank_name,
                        endpoint=f"{rank_name}:12345",
                        lease_timeout_sec=30,
                    )
                    source_managers.append(manager)
                    source_parts.append(parts)

            target_w13 = _BufferParameter(shape=(4, 8, 8))
            target_w2 = _BufferParameter(shape=(4, 8, 4))
            target_w13.fill(0xFF)
            target_w2.fill(0xFF)
            target_manager = sglang.WeightRuntimeManifestManager(
                model=_NamedParameterModel(
                    (
                        ("layers.10.mlp.experts.w13_weight", target_w13),
                        ("layers.10.mlp.experts.w2_weight", target_w2),
                    )
                ),
                adapter=qwen.Qwen35WeightSemanticsAdapter(config=config),
                topology=sglang.WeightParallelTopology(
                    dp_rank=3,
                    dp_size=4,
                    tp_rank=0,
                    tp_size=2,
                    pp_rank=2,
                    pp_size=4,
                    ep_rank=0,
                    ep_size=2,
                    moe_tp_rank=0,
                    moe_tp_size=2,
                ),
                allowed_devices=("cpu",),
            )
            target_parts = target_manager.snapshot_parts(
                model_id="qwen3.5-moe",
                revision="step-42",
                instance_id="target-p2-e0-t0",
                worker_id="target-p2-e0-t0",
                endpoint="target-p2-e0-t0:12345",
                lease_timeout_sec=30,
            )

            source_placements = tuple(
                SourcePlacementManifest.from_runtime_inventory(
                    msgspec.to_builtins(parts.placement)
                )
                for parts in source_parts
            )
            source_bindings = tuple(
                RuntimeBindingManifest.from_runtime_inventory(
                    msgspec.to_builtins(parts.binding)
                )
                for parts in source_parts
            )
            target_placement = TargetPlacementManifest.from_runtime_inventory(
                msgspec.to_builtins(target_parts.placement)
            )
            target_binding = RuntimeBindingManifest.from_runtime_inventory(
                msgspec.to_builtins(target_parts.binding)
            )

            logical = plan_placement_transfer_to_local_target(
                source_placements,
                target_placement,
            )
            transfer = bind_logical_transfer_plan(
                logical,
                (target_binding,),
                source_bindings=source_bindings,
            )
            source_runtime = tuple(
                bind_runtime_manifest(placement, binding)
                for placement, binding in zip(
                    source_placements,
                    source_bindings,
                    strict=True,
                )
            )
            target_runtime = bind_runtime_manifest(
                target_placement,
                target_binding,
            )
            for manifest in source_runtime:
                _write_logical_payloads(manifest)

            engine = _CopyingReadEngine()
            receipts = MooncakeTransferEngineReader(engine).execute(
                transfer,
                source_runtime,
                target_runtime,
                source_registrations=tuple(
                    MemoryRegistrationLease.from_fragment(
                        fragment,
                        runtime_lease_id=manifest.lease_id,
                    )
                    for manifest in source_runtime
                    for fragment in manifest.fragments
                ),
                target_pre_registered=True,
                target_registrations=tuple(
                    MemoryRegistrationLease.from_fragment(
                        fragment,
                        runtime_lease_id=target_runtime.lease_id,
                    )
                    for fragment in target_runtime.fragments
                ),
            )

            assert len(transfer.operations) == 24
            assert {
                (route.source_pp, route.target_pp) for route in transfer.pipeline_routes
            } == {(1, 2)}
            assert sum(receipt.nbytes for receipt in receipts) == 768
            assert max(operation.repeat for operation in transfer.operations) == 8
            assert sum(receipt.operation_count for receipt in receipts) == 80
            assert len(engine.calls) == 4
            _assert_logical_payloads(target_runtime)
        finally:
            for manager, parts in zip(
                source_managers,
                source_parts,
                strict=True,
            ):
                if manager.has_lease(parts.binding.lease_id):
                    manager.release(parts.binding.lease_id)
            if (
                target_manager is not None
                and target_parts is not None
                and target_manager.has_lease(target_parts.binding.lease_id)
            ):
                target_manager.release(target_parts.binding.lease_id)


def test_sglang_fp8_w31_to_w13_contract_plans_binds_and_executes() -> None:
    with _load_sglang_contract_modules() as modules:
        sglang = modules.manifest
        msgspec = _load_msgspec()
        config = SimpleNamespace(
            model_type="qwen3_5_moe_text",
            num_experts=2,
            moe_intermediate_size=256,
            hidden_size=128,
        )
        source_managers = []
        source_parts = []
        target_manager = None
        target_parts = None
        try:
            for ep_rank in range(2):
                for tp_rank in range(2):
                    rank_name = f"source-p1-e{ep_rank}-t{tp_rank}"
                    weight = _BufferParameter(
                        shape=(1, 256, 128),
                        dtype="torch.float8_e4m3fn",
                        itemsize=1,
                    )
                    scale = _BufferParameter(
                        shape=(1, 2, 1),
                        dtype="torch.float32",
                        itemsize=4,
                    )
                    runtime_module = _Fp8RuntimeModule(
                        weight=weight,
                        scale=scale,
                        up_first=True,
                    )
                    manager = sglang.create_weight_runtime_manifest_manager(
                        model=_Fp8Model(
                            (
                                ("layers.2.mlp.experts.w13_weight", weight),
                                (
                                    "layers.2.mlp.experts.w13_weight_scale_inv",
                                    scale,
                                ),
                            ),
                            runtime_module=runtime_module,
                        ),
                        config=config,
                        topology=sglang.WeightParallelTopology(
                            dp_rank=0,
                            dp_size=2,
                            tp_rank=tp_rank,
                            tp_size=2,
                            pp_rank=1,
                            pp_size=2,
                            ep_rank=ep_rank,
                            ep_size=2,
                            moe_tp_rank=tp_rank,
                            moe_tp_size=2,
                        ),
                        allowed_devices=("cpu",),
                        quantization="fp8",
                        fp8_gemm_backend="triton",
                        moe_runner_backend="triton",
                    )
                    parts = manager.snapshot_parts(
                        model_id="qwen3.5-moe-fp8",
                        revision="step-43",
                        instance_id=rank_name,
                        worker_id=rank_name,
                        endpoint=f"{rank_name}:12345",
                        lease_timeout_sec=30,
                    )
                    source_managers.append(manager)
                    source_parts.append(parts)

            target_weight = _BufferParameter(
                shape=(2, 512, 128),
                dtype="torch.float8_e4m3fn",
                itemsize=1,
            )
            target_scale = _BufferParameter(
                shape=(2, 4, 1),
                dtype="torch.float32",
                itemsize=4,
            )
            target_weight.fill(0xFF)
            target_scale.fill(0xFF)
            target_runtime_module = _Fp8RuntimeModule(
                weight=target_weight,
                scale=target_scale,
                up_first=False,
            )
            target_manager = sglang.create_weight_runtime_manifest_manager(
                model=_Fp8Model(
                    (
                        ("layers.2.mlp.experts.w13_weight", target_weight),
                        (
                            "layers.2.mlp.experts.w13_weight_scale_inv",
                            target_scale,
                        ),
                    ),
                    runtime_module=target_runtime_module,
                ),
                config=config,
                topology=sglang.WeightParallelTopology(
                    dp_rank=3,
                    dp_size=4,
                    tp_rank=0,
                    tp_size=1,
                    pp_rank=2,
                    pp_size=4,
                    ep_rank=0,
                    ep_size=1,
                    moe_tp_rank=0,
                    moe_tp_size=1,
                ),
                allowed_devices=("cpu",),
                quantization="fp8",
                fp8_gemm_backend="triton",
                moe_runner_backend="triton",
            )
            target_parts = target_manager.snapshot_parts(
                model_id="qwen3.5-moe-fp8",
                revision="step-43",
                instance_id="target-p2-e0-t0",
                worker_id="target-p2-e0-t0",
                endpoint="target-p2-e0-t0:12345",
                lease_timeout_sec=30,
            )

            source_placements = tuple(
                SourcePlacementManifest.from_runtime_inventory(
                    msgspec.to_builtins(parts.placement)
                )
                for parts in source_parts
            )
            source_bindings = tuple(
                RuntimeBindingManifest.from_runtime_inventory(
                    msgspec.to_builtins(parts.binding)
                )
                for parts in source_parts
            )
            target_placement = TargetPlacementManifest.from_runtime_inventory(
                msgspec.to_builtins(target_parts.placement)
            )
            target_binding = RuntimeBindingManifest.from_runtime_inventory(
                msgspec.to_builtins(target_parts.binding)
            )
            logical = plan_placement_transfer_to_local_target(
                source_placements,
                target_placement,
            )
            transfer = bind_logical_transfer_plan(
                logical,
                (target_binding,),
                source_bindings=source_bindings,
            )
            source_runtime = tuple(
                bind_runtime_manifest(placement, binding)
                for placement, binding in zip(
                    source_placements,
                    source_bindings,
                    strict=True,
                )
            )
            target_runtime = bind_runtime_manifest(
                target_placement,
                target_binding,
            )
            for manifest in source_runtime:
                _write_logical_payloads(manifest)

            engine = _CopyingReadEngine()
            receipts = MooncakeTransferEngineReader(engine).execute(
                transfer,
                source_runtime,
                target_runtime,
                source_registrations=tuple(
                    MemoryRegistrationLease.from_fragment(
                        fragment,
                        runtime_lease_id=manifest.lease_id,
                    )
                    for manifest in source_runtime
                    for fragment in manifest.fragments
                ),
                target_pre_registered=True,
                target_registrations=tuple(
                    MemoryRegistrationLease.from_fragment(
                        fragment,
                        runtime_lease_id=target_runtime.lease_id,
                    )
                    for fragment in target_runtime.fragments
                ),
            )

            tensor_ids = {tensor.tensor_id for tensor in target_runtime.tensors}
            assert tensor_ids == {
                "layers.2.mlp.experts.gate_proj.weight",
                "layers.2.mlp.experts.gate_proj.weight_scale_inv",
                "layers.2.mlp.experts.up_proj.weight",
                "layers.2.mlp.experts.up_proj.weight_scale_inv",
            }
            assert {
                (route.source_pp, route.target_pp) for route in transfer.pipeline_routes
            } == {(1, 2)}
            assert len(transfer.operations) <= 32
            assert len(engine.calls) <= 8
            assert sum(receipt.nbytes for receipt in receipts) == 131104
            _assert_logical_payloads(target_runtime)
        finally:
            for manager, parts in zip(
                source_managers,
                source_parts,
                strict=True,
            ):
                if manager.has_lease(parts.binding.lease_id):
                    manager.release(parts.binding.lease_id)
            if (
                target_manager is not None
                and target_parts is not None
                and target_manager.has_lease(target_parts.binding.lease_id)
            ):
                target_manager.release(target_parts.binding.lease_id)
