from __future__ import annotations

import time
from uuid import uuid4

from mooncake.reshard.weight import (
    MemoryRegistrationLease,
    MooncakeTransferEngineSink,
    TensorDescriptor,
    WeightStore,
    bind_logical_transfer_plan,
    plan_placement_transfer,
)

from .buffers import TransferBuffer
from .lifetime import allocation_guards_for_bindings
from .manifests import (
    _rank_manifests,
    _verify_tp_buffers,
)


def _cleanup_store_upload(store, upload_plan) -> None:
    keys = [
        upload_plan.manifest.manifest_key,
        *(operation.target.object_key for operation in upload_plan.operations),
        upload_plan.control_key,
    ]
    for key in dict.fromkeys(keys):
        result = store.remove(key, force=True)
        if result not in (0, -704):
            raise AssertionError(f"failed to remove benchmark object {key}: {result}")
        if store.is_exist(key) != 0:
            raise AssertionError(f"benchmark object remains after cleanup: {key}")


def _run_store_iteration(
    *,
    store,
    weight_store: WeightStore,
    tensor: TensorDescriptor,
    source_buffers: list[TransferBuffer],
    target_buffers: list[TransferBuffer],
    namespace: str,
) -> dict[str, float]:
    source_tp = len(source_buffers)
    target_tp = len(target_buffers)
    total_bytes = tensor.global_shape[0]
    for buffer in target_buffers:
        buffer.zero()

    revision = uuid4().hex
    sources = _rank_manifests(
        tensor=tensor,
        revision=revision,
        prefix="source",
        buffers=source_buffers,
    )
    target_ranks = _rank_manifests(
        tensor=tensor,
        revision=revision,
        prefix="target",
        buffers=target_buffers,
    )
    upload_plan = None
    durations = {}
    e2e_start = time.perf_counter()
    try:
        started = time.perf_counter()
        upload_plan = weight_store.prepare_upload(
            sources.placement,
            sources.bindings,
            namespace=namespace,
        )
        durations["prepare"] = time.perf_counter() - started

        started = time.perf_counter()
        receipts = tuple(
            receipt
            for source_binding in sources.bindings
            for receipt in weight_store.upload(
                upload_plan,
                sources.placement,
                source_binding,
                source_allocation_guards=allocation_guards_for_bindings(
                    (source_binding,)
                ),
            )
        )
        durations["upload"] = time.perf_counter() - started

        started = time.perf_counter()
        manifest = weight_store.commit(upload_plan, receipts)
        durations["commit"] = time.perf_counter() - started

        started = time.perf_counter()
        loaded = weight_store.load_manifest(manifest.manifest_key)
        durations["manifest_get"] = time.perf_counter() - started

        started = time.perf_counter()
        load_plan = weight_store.plan_load(
            loaded,
            target_ranks.placement,
            target_ranks.bindings,
        )
        durations["plan_load"] = time.perf_counter() - started

        started = time.perf_counter()
        for target_binding in target_ranks.bindings:
            weight_store.load(
                load_plan,
                target_ranks.placement,
                target_binding,
                target_allocation_guards=allocation_guards_for_bindings(
                    (target_binding,)
                ),
            )
        durations["load"] = time.perf_counter() - started
        durations["e2e"] = time.perf_counter() - e2e_start

        _verify_tp_buffers(
            target_buffers,
            total_bytes=total_bytes,
            source_tp=source_tp,
            target_tp=target_tp,
        )
        return durations
    finally:
        if upload_plan is not None:
            _cleanup_store_upload(store, upload_plan)


def _run_te_iteration(
    *,
    source_engine,
    target_endpoint: str,
    tensor: TensorDescriptor,
    source_buffers: list[TransferBuffer],
    target_buffers: list[TransferBuffer],
) -> dict[str, float]:
    source_tp = len(source_buffers)
    target_tp = len(target_buffers)
    total_bytes = tensor.global_shape[0]
    for buffer in target_buffers:
        buffer.zero()

    revision = uuid4().hex
    sources = _rank_manifests(
        tensor=tensor,
        revision=revision,
        prefix="source",
        buffers=source_buffers,
    )
    targets = _rank_manifests(
        tensor=tensor,
        revision=revision,
        prefix="target",
        buffers=target_buffers,
        endpoint=target_endpoint,
    )
    e2e_start = time.perf_counter()
    started = time.perf_counter()
    logical = plan_placement_transfer(sources.placement, targets.placement)
    plan = bind_logical_transfer_plan(
        logical,
        targets.bindings,
        source_bindings=sources.bindings,
    )
    plan_seconds = time.perf_counter() - started

    sink = MooncakeTransferEngineSink(source_engine)
    source_registrations = tuple(
        MemoryRegistrationLease.from_fragment(
            fragment,
            lease_generation=binding.generation,
            runtime_lease_id=binding.lease_id,
        )
        for binding in sources.bindings
        for fragment in binding.fragments
    )
    target_registrations = tuple(
        MemoryRegistrationLease.from_fragment(
            fragment,
            lease_generation=binding.generation,
            runtime_lease_id=binding.lease_id,
        )
        for binding in targets.bindings
        for fragment in binding.fragments
    )
    started = time.perf_counter()
    receipts = tuple(
        receipt
        for source_binding in sources.bindings
        for receipt in sink.execute(
            plan,
            sources.placement,
            source_binding,
            targets.placement,
            targets.bindings,
            target_registrations=target_registrations,
            source_pre_registered=True,
            source_registrations=source_registrations,
            source_allocation_guards=allocation_guards_for_bindings(sources.bindings),
            target_allocation_guards=allocation_guards_for_bindings(targets.bindings),
        )
    )
    transfer_seconds = time.perf_counter() - started
    e2e_seconds = time.perf_counter() - e2e_start

    if sum(receipt.nbytes for receipt in receipts) != total_bytes:
        raise AssertionError("Transfer Engine receipt bytes are incomplete")
    _verify_tp_buffers(
        target_buffers,
        total_bytes=total_bytes,
        source_tp=source_tp,
        target_tp=target_tp,
    )
    return {
        "plan": plan_seconds,
        "transfer": transfer_seconds,
        "e2e": e2e_seconds,
    }
