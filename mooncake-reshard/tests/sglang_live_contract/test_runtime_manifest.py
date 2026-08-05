from __future__ import annotations

import importlib

import pytest

from mooncake.reshard._compat import _strict_zip
from mooncake.reshard.weight import (
    SplitAxis,
    validate_runtime_binding,
)

from .helpers import (
    _Model,
    _Parameter,
    _adapt_sglang_parts,
    _load_msgspec,
    _load_sglang_contract_modules,
)


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


def test_sglang_runtime_output_binds_with_mooncake_parser() -> None:
    with _load_sglang_contract_modules() as modules:
        sglang = modules.manifest
        msgspec = _load_msgspec()

        class Adapter:
            def describe_parameter(self, *, names, parameter, topology):
                assert names == ("layers.0.mlp.weight",)
                assert topology.tp_size == 2
                return (
                    sglang.LogicalTensorView(
                        tensor_id="model.layers.0.mlp.weight",
                        global_shape=(8, 8),
                        global_offset=(topology.tp_rank * 4, 0),
                        local_shape=(4, 8),
                        shard_dims=(0,),
                        byte_offset=0,
                        layer_id=0,
                        expert_id=None,
                        layout_fingerprint="row-major-f16",
                    ),
                )

        managers = []
        runtime_parts = []
        parameters = []
        try:
            for tp_rank in range(2):
                parameter = _Parameter(
                    address=0x100000 + tp_rank * 0x1000,
                    shape=(4, 8),
                )
                manager = sglang.WeightRuntimeManifestManager(
                    model=_Model(parameter),
                    adapter=Adapter(),
                    topology=sglang.WeightParallelTopology(
                        tp_rank=tp_rank,
                        tp_size=2,
                    ),
                    allowed_devices=("cpu",),
                )
                parts = manager.snapshot_parts(
                    model_id="contract-model",
                    revision="step-1",
                    instance_id=f"target-{tp_rank}",
                    worker_id=f"worker-{tp_rank}",
                    endpoint="127.0.0.1:12345",
                    lease_timeout_sec=30,
                )
                managers.append(manager)
                runtime_parts.append(parts)
                parameters.append(parameter)

            placement, bindings = _adapt_sglang_parts(
                runtime_parts,
                msgspec,
                placement_set_id="contract-target",
                tp_size=2,
                pp_size=1,
                ep_size=1,
                dp_size=1,
            )
            for binding in bindings:
                validate_runtime_binding(placement, binding)

            assert placement.resource_id == "contract-model"
            assert placement.revision == "step-1"
            assert all(
                placement.placement_id == binding.placement_id for binding in bindings
            )
            assert all(
                placement.digest == binding.placement_digest for binding in bindings
            )
            assert [binding.participant_id for binding in bindings] == [
                parts.placement.placement_id for parts in runtime_parts
            ]
            assert all(
                binding.participant_id != binding.instance_id for binding in bindings
            )
            assert bindings[0].lease_id == runtime_parts[0].binding.lease_id
            assert bindings[0].fragments[0].address == parameters[0].data_ptr()
            assert placement.fragments[0].global_offset == (0, 0)
            assert placement.fragments[0].local_shape == (4, 8)
            assert placement.fragments[0].rank.tp == 0
            assert placement.tensors[0].parallel_axes == (SplitAxis("tp", dim=0),)
            runtime = bindings[0].fragments[0]
            assert runtime.itemsize == 2
            assert runtime.local_shape == (4, 8)
            assert runtime.strides_bytes == (16, 2)
            assert runtime.storage_address == parameters[0].data_ptr()
            assert runtime.storage_nbytes == runtime.nbytes
            assert runtime.storage_offset_bytes == 0
        finally:
            for manager, parts in _strict_zip(managers, runtime_parts):
                manager.release(parts.binding.lease_id)
