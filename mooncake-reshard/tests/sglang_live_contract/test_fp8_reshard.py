from __future__ import annotations

import ctypes
from types import SimpleNamespace

from mooncake.reshard._compat import _strict_zip
from mooncake.reshard.weight import (
    MemoryRegistrationLease,
    MooncakeTransferEngineReader,
    bind_logical_transfer_plan,
    plan_placement_transfer_to_local_target,
)

from .helpers import (
    _BufferParameter,
    _CopyingReadEngine,
    _Fp8Model,
    _Fp8RuntimeModule,
    _adapt_sglang_parts,
    _fragment_payload,
    _load_msgspec,
    _load_sglang_contract_modules,
)


def _write_logical_payloads(placement, binding) -> None:
    tensors = {tensor.tensor_id: tensor for tensor in placement.tensors}
    runtime_by_id = {
        fragment.placement_fragment_id: fragment for fragment in binding.fragments
    }
    part = next(
        part
        for part in placement.parts
        if part.participant_id == binding.participant_id
    )
    for fragment in part.fragments:
        payload = _fragment_payload(tensors[fragment.tensor_id], fragment)
        ctypes.memmove(
            runtime_by_id[fragment.placement_fragment_id].address,
            payload,
            len(payload),
        )


def _assert_logical_payloads(placement, binding) -> None:
    tensors = {tensor.tensor_id: tensor for tensor in placement.tensors}
    runtime_by_id = {
        fragment.placement_fragment_id: fragment for fragment in binding.fragments
    }
    part = next(
        part
        for part in placement.parts
        if part.participant_id == binding.participant_id
    )
    for fragment in part.fragments:
        runtime = runtime_by_id[fragment.placement_fragment_id]
        expected = _fragment_payload(tensors[fragment.tensor_id], fragment)
        assert ctypes.string_at(runtime.address, runtime.nbytes) == expected


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

            source_placement, source_bindings = _adapt_sglang_parts(
                source_parts,
                msgspec,
                placement_set_id="fp8-source",
                tp_size=2,
                pp_size=2,
                ep_size=2,
                dp_size=2,
            )
            target_placement, target_bindings = _adapt_sglang_parts(
                (target_parts,),
                msgspec,
                placement_set_id="fp8-target",
                tp_size=1,
                pp_size=4,
                ep_size=1,
                dp_size=4,
            )
            target_binding = target_bindings[0]
            logical = plan_placement_transfer_to_local_target(
                source_placement,
                target_placement,
                target_participant_id=target_binding.participant_id,
            )
            transfer = bind_logical_transfer_plan(
                logical,
                (target_binding,),
                source_bindings=source_bindings,
            )
            for binding in source_bindings:
                _write_logical_payloads(source_placement, binding)

            engine = _CopyingReadEngine()
            receipts = MooncakeTransferEngineReader(engine).execute(
                transfer,
                source_placement,
                source_bindings,
                target_placement,
                target_binding,
                source_registrations=tuple(
                    MemoryRegistrationLease.from_fragment(
                        fragment,
                        lease_generation=binding.generation,
                        runtime_lease_id=binding.lease_id,
                    )
                    for binding in source_bindings
                    for fragment in binding.fragments
                ),
                target_pre_registered=True,
                target_registrations=tuple(
                    MemoryRegistrationLease.from_fragment(
                        fragment,
                        lease_generation=target_binding.generation,
                        runtime_lease_id=target_binding.lease_id,
                    )
                    for fragment in target_binding.fragments
                ),
            )

            tensor_ids = {tensor.tensor_id for tensor in target_placement.tensors}
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
            _assert_logical_payloads(target_placement, target_binding)
        finally:
            for manager, parts in _strict_zip(
                source_managers,
                source_parts,
            ):
                if manager.has_lease(parts.binding.lease_id):
                    manager.release(parts.binding.lease_id)
            if (
                target_manager is not None
                and target_parts is not None
                and target_manager.has_lease(target_parts.binding.lease_id)
            ):
                target_manager.release(target_parts.binding.lease_id)
