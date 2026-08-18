from __future__ import annotations

import os
import socket
from contextlib import ExitStack
from uuid import uuid4

import pytest

from mooncake.reshard.weight import (
    WeightLoadPlan,
    WeightStore,
    bind_logical_transfer_plan,
    plan_stored_transfer_to_target_placement,
)
from weight_all_axes_e2e.builders import (
    _build_cross_dim_reader_fixture,
    _build_fixture,
)
from weight_all_axes_e2e.fixtures import (
    _native_store_config_factory,
)
from weight_gpu_e2e.buffers import (
    CudaBuffer,
    CudaRuntime,
    _parse_cuda_devices,
)
from weight_gpu_e2e.execution import _cleanup_store_upload
from weight_gpu_e2e.lifetime import allocation_guards_for_bindings


@pytest.mark.skipif(
    os.getenv("MOONCAKE_WEIGHT_GPU_STORE_E2E") != "1",
    reason="set MOONCAKE_WEIGHT_GPU_STORE_E2E=1 to run the CUDA Store test",
)
def test_gpu_store_moves_weights_across_dp_tp_pp_ep_together() -> None:
    from mooncake.store import MooncakeDistributedStore

    source_devices = _parse_cuda_devices(
        os.environ, "MOONCAKE_WEIGHT_SOURCE_CUDA_DEVICES", default="0"
    )
    target_devices = _parse_cuda_devices(
        os.environ, "MOONCAKE_WEIGHT_TARGET_CUDA_DEVICES", default="0"
    )
    runtimes = {
        cuda_device: CudaRuntime(cuda_device)
        for cuda_device in sorted(set(source_devices) | set(target_devices))
    }
    source_index = 0
    target_index = 0
    with ExitStack() as stack:

        def allocate_source(size: int):
            nonlocal source_index
            device = source_devices[source_index % len(source_devices)]
            source_index += 1
            return stack.enter_context(CudaBuffer(runtimes[device], size))

        def allocate_target(size: int):
            nonlocal target_index
            device = target_devices[target_index % len(target_devices)]
            target_index += 1
            return stack.enter_context(CudaBuffer(runtimes[device], size))

        fixture = _build_fixture(
            revision=uuid4().hex,
            source_tp=2,
            target_tp=4,
            allocate_source=allocate_source,
            allocate_target=allocate_target,
            target_endpoint="store-target:12345",
        )
        store = MooncakeDistributedStore()
        result = store.setup(
            os.getenv(
                "MOONCAKE_WEIGHT_LOCAL_HOSTNAME",
                socket.gethostbyname(socket.gethostname()),
            ),
            os.getenv(
                "MOONCAKE_WEIGHT_METADATA_SERVER",
                "http://127.0.0.1:8080/metadata",
            ),
            128 * 1024 * 1024,
            64 * 1024 * 1024,
            os.getenv("MOONCAKE_WEIGHT_PROTOCOL", "tcp"),
            os.getenv("MOONCAKE_WEIGHT_DEVICE", "eth0"),
            os.getenv("MOONCAKE_WEIGHT_MASTER", "127.0.0.1:50051"),
        )
        assert result == 0
        upload_plan = None
        try:
            weight_store = WeightStore(
                store, config_factory=_native_store_config_factory
            )
            upload_plan = weight_store.prepare_upload(
                fixture.sources.placement,
                fixture.sources.bindings,
                namespace="all-axes-native-e2e",
            )
            receipts = tuple(
                receipt
                for source_placement, source_binding in fixture.sources.active_pairs()
                for receipt in weight_store.upload(
                    upload_plan,
                    source_placement,
                    source_binding,
                    source_allocation_guards=allocation_guards_for_bindings(
                        (source_binding,)
                    ),
                )
            )
            persisted = weight_store.commit(upload_plan, receipts)
            weight_store.finalize_upload_session(upload_plan)
            loaded = weight_store.load_manifest(persisted.manifest_key)
            logical = plan_stored_transfer_to_target_placement(
                loaded,
                fixture.targets.placement,
            )
            assert all(
                not hasattr(operation.target, "address")
                for operation in logical.operations
            )
            load_plan = WeightLoadPlan(
                manifest=loaded,
                transfer=bind_logical_transfer_plan(
                    logical,
                    fixture.targets.bindings,
                    source_manifest=loaded,
                ),
            )
            for target_placement, target_binding in fixture.targets.active_pairs():
                weight_store.load(
                    load_plan,
                    target_placement,
                    target_binding,
                    target_allocation_guards=allocation_guards_for_bindings(
                        (target_binding,)
                    ),
                )
            fixture.verify()
        finally:
            if upload_plan is not None:
                _cleanup_store_upload(store, upload_plan)
            close = getattr(store, "close", None)
            if callable(close):
                assert close() == 0


@pytest.mark.skipif(
    os.getenv("MOONCAKE_WEIGHT_GPU_STORE_E2E") != "1",
    reason="set MOONCAKE_WEIGHT_GPU_STORE_E2E=1 to run the CUDA Store test",
)
def test_gpu_store_reshards_independent_experts_across_dimensions() -> None:
    from mooncake.store import MooncakeDistributedStore

    source_devices = _parse_cuda_devices(
        os.environ, "MOONCAKE_WEIGHT_SOURCE_CUDA_DEVICES", default="0"
    )
    target_devices = _parse_cuda_devices(
        os.environ, "MOONCAKE_WEIGHT_TARGET_CUDA_DEVICES", default="0"
    )
    runtimes = {
        cuda_device: CudaRuntime(cuda_device)
        for cuda_device in sorted(set(source_devices) | set(target_devices))
    }
    source_index = 0
    target_index = 0
    with ExitStack() as stack:

        def allocate_source(size: int):
            nonlocal source_index
            device = source_devices[source_index % len(source_devices)]
            source_index += 1
            return stack.enter_context(CudaBuffer(runtimes[device], size))

        def allocate_target(size: int):
            nonlocal target_index
            device = target_devices[target_index % len(target_devices)]
            target_index += 1
            return stack.enter_context(CudaBuffer(runtimes[device], size))

        fixture = _build_cross_dim_reader_fixture(
            revision=uuid4().hex,
            source_endpoint="store-source:12345",
            target_endpoint="store-target:12345",
            allocate_source=allocate_source,
            allocate_target=allocate_target,
        )
        store = MooncakeDistributedStore()
        result = store.setup(
            os.getenv(
                "MOONCAKE_WEIGHT_LOCAL_HOSTNAME",
                socket.gethostbyname(socket.gethostname()),
            ),
            os.getenv(
                "MOONCAKE_WEIGHT_METADATA_SERVER",
                "http://127.0.0.1:8080/metadata",
            ),
            128 * 1024 * 1024,
            64 * 1024 * 1024,
            os.getenv("MOONCAKE_WEIGHT_PROTOCOL", "tcp"),
            os.getenv("MOONCAKE_WEIGHT_DEVICE", "eth0"),
            os.getenv("MOONCAKE_WEIGHT_MASTER", "127.0.0.1:50051"),
        )
        assert result == 0
        upload_plan = None
        try:
            weight_store = WeightStore(
                store, config_factory=_native_store_config_factory
            )
            upload_plan = weight_store.prepare_upload(
                fixture.sources.placement,
                fixture.sources.bindings,
                namespace="cross-dim-native-e2e",
            )
            assert len(upload_plan.operations) == 8
            assert (
                len(
                    {
                        operation.target.object_key
                        for operation in upload_plan.operations
                    }
                )
                == 8
            )
            receipts = tuple(
                receipt
                for source_placement, source_binding in fixture.sources.active_pairs()
                for receipt in weight_store.upload(
                    upload_plan,
                    source_placement,
                    source_binding,
                    source_allocation_guards=allocation_guards_for_bindings(
                        (source_binding,)
                    ),
                )
            )
            persisted = weight_store.commit(upload_plan, receipts)
            weight_store.finalize_upload_session(upload_plan)
            loaded = weight_store.load_manifest(persisted.manifest_key)
            assert loaded == persisted
            logical = plan_stored_transfer_to_target_placement(
                loaded,
                fixture.targets.placement,
            )
            assert all(
                not hasattr(operation.target, "address")
                for operation in logical.operations
            )
            load_plan = WeightLoadPlan(
                manifest=loaded,
                transfer=bind_logical_transfer_plan(
                    logical,
                    fixture.targets.bindings,
                    source_manifest=loaded,
                ),
            )
            for target_placement, target_binding in fixture.targets.active_pairs():
                weight_store.load(
                    load_plan,
                    target_placement,
                    target_binding,
                    target_allocation_guards=allocation_guards_for_bindings(
                        (target_binding,)
                    ),
                )
            fixture.verify()
        finally:
            if upload_plan is not None:
                _cleanup_store_upload(store, upload_plan)
            close = getattr(store, "close", None)
            if callable(close):
                assert close() == 0
