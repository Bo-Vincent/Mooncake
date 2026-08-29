from __future__ import annotations

import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

from mooncake.reshard._compat import _strict_zip
from mooncake.reshard.weight import (
    MemoryRegistrationLease,
    MooncakeTransferEngineReader,
    ParallelRank,
    bind_logical_transfer_plan,
    plan_placement_transfer_to_local_target,
)
from weight_gpu_e2e.lifetime import allocation_guards_for_bindings

from .helpers import (
    _BufferParameter,
    _CopyingReadEngine,
    _NamedParameterModel,
    _adapt_sglang_parts,
    _fragment_payload,
    _load_msgspec,
    _load_sglang_contract_modules,
    _normalize_sglang_runtime_fixture,
    _sglang_source_root,
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


def test_sglang_qwen_split_contract_plans_binds_and_executes() -> None:
    with _load_sglang_contract_modules() as modules:
        sglang = modules.manifest
        qwen = modules.qwen
        msgspec = _load_msgspec()
        source_root = _sglang_source_root()
        local_fixture = _normalize_sglang_runtime_fixture(
            json.loads(
                (
                    Path(__file__).parents[1]
                    / "fixtures/qwen3_5_moe_runtime_manifest.json"
                ).read_text()
            ),
            tp_size=4,
            pp_size=2,
            ep_size=4,
            dp_size=2,
        )
        sglang_fixture = _normalize_sglang_runtime_fixture(
            json.loads(
                (
                    source_root / "test/registered/unit/model_executor/fixtures/"
                    "qwen3_5_moe_runtime_manifest.json"
                ).read_text()
            ),
            tp_size=4,
            pp_size=2,
            ep_size=4,
            dp_size=2,
        )
        sglang_fixture.pop("format_version", None)
        assert local_fixture == sglang_fixture

        config = SimpleNamespace(
            num_experts=8,
            moe_intermediate_size=8,
            hidden_size=8,
        )
        source_managers = []
        source_parts = []
        target_managers = []
        target_parts = []
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

            for ep_rank in range(2):
                for tp_rank in range(2):
                    rank_name = f"target-p2-e{ep_rank}-t{tp_rank}"
                    target_w13 = _BufferParameter(shape=(4, 8, 8))
                    target_w2 = _BufferParameter(shape=(4, 8, 4))
                    target_w13.fill(0xFF)
                    target_w2.fill(0xFF)
                    target_manager = sglang.WeightRuntimeManifestManager(
                        model=_NamedParameterModel(
                            (
                                (
                                    "layers.10.mlp.experts.w13_weight",
                                    target_w13,
                                ),
                                (
                                    "layers.10.mlp.experts.w2_weight",
                                    target_w2,
                                ),
                            )
                        ),
                        adapter=qwen.Qwen35WeightSemanticsAdapter(config=config),
                        topology=sglang.WeightParallelTopology(
                            dp_rank=3,
                            dp_size=4,
                            tp_rank=tp_rank,
                            tp_size=2,
                            pp_rank=2,
                            pp_size=4,
                            ep_rank=ep_rank,
                            ep_size=2,
                            moe_tp_rank=tp_rank,
                            moe_tp_size=2,
                        ),
                        allowed_devices=("cpu",),
                    )
                    parts = target_manager.snapshot_parts(
                        model_id="qwen3.5-moe",
                        revision="step-42",
                        instance_id=rank_name,
                        worker_id=rank_name,
                        endpoint=f"{rank_name}:12345",
                        lease_timeout_sec=30,
                    )
                    target_managers.append(target_manager)
                    target_parts.append(parts)

            source_placement, source_bindings = _adapt_sglang_parts(
                source_parts,
                msgspec,
                placement_set_id="qwen-source",
                tp_size=4,
                pp_size=2,
                ep_size=4,
                dp_size=2,
            )
            target_placement, target_bindings = _adapt_sglang_parts(
                target_parts,
                msgspec,
                placement_set_id="qwen-target",
                tp_size=2,
                pp_size=4,
                ep_size=2,
                dp_size=4,
            )
            target_participant_id = next(
                part.participant_id
                for part in target_placement.parts
                if part.rank == ParallelRank(dp=3, tp=0, pp=2, ep=0)
            )
            target_binding = next(
                binding
                for binding in target_bindings
                if binding.participant_id == target_participant_id
            )

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
                source_allocation_guards=allocation_guards_for_bindings(
                    source_bindings
                ),
                target_allocation_guards=allocation_guards_for_bindings(
                    (target_binding,)
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
            _assert_logical_payloads(target_placement, target_binding)
        finally:
            for manager, parts in _strict_zip(
                source_managers,
                source_parts,
            ):
                if manager.has_lease(parts.binding.lease_id):
                    manager.release(parts.binding.lease_id)
            for manager, parts in _strict_zip(
                target_managers,
                target_parts,
            ):
                if manager.has_lease(parts.binding.lease_id):
                    manager.release(parts.binding.lease_id)
