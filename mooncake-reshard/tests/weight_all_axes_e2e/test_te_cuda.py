from __future__ import annotations

import os
import socket
from contextlib import ExitStack
from uuid import uuid4

import pytest

from weight_all_axes_e2e.buffers import (
    _registered_engine_buffers,
    _remote_cuda_target,
)
from weight_all_axes_e2e.builders import (
    _build_cross_dim_reader_fixture,
    _build_fixture,
    _build_packed_reader_fixture,
)
from weight_all_axes_e2e.execution import (
    _execute_cross_dim_reader_fixture,
    _execute_cross_dim_sink_fixture,
    _execute_reader_fixture,
    _execute_te_fixture,
)
from weight_gpu_e2e.buffers import (
    CudaBuffer,
    CudaRuntime,
    _parse_cuda_devices,
)


@pytest.mark.skipif(
    os.getenv("MOONCAKE_WEIGHT_TE_GPU_E2E") != "1",
    reason="set MOONCAKE_WEIGHT_TE_GPU_E2E=1 to run the CUDA TE test",
)
@pytest.mark.parametrize(
    ("source_tp", "target_tp"),
    (
        pytest.param(4, 8, id="tp4-to-tp8"),
        pytest.param(8, 4, id="tp8-to-tp4"),
    ),
)
def test_te_cuda_moves_weights_across_dp_tp_pp_ep_together(
    source_tp: int,
    target_tp: int,
) -> None:
    from mooncake.engine import TransferEngine

    source_devices = _parse_cuda_devices(
        os.environ, "MOONCAKE_WEIGHT_SOURCE_CUDA_DEVICES", default="0"
    )
    target_devices = _parse_cuda_devices(
        os.environ, "MOONCAKE_WEIGHT_TARGET_CUDA_DEVICES", default="0"
    )
    runtimes = {
        device: CudaRuntime(device)
        for device in sorted(set(source_devices) | set(target_devices))
    }
    local_hostname = os.getenv(
        "MOONCAKE_WEIGHT_TE_HOSTNAME",
        os.getenv(
            "MOONCAKE_WEIGHT_LOCAL_HOSTNAME",
            socket.gethostbyname(socket.gethostname()),
        ),
    )
    protocol = os.getenv("MOONCAKE_WEIGHT_TE_PROTOCOL", "rdma")
    transport_device = os.getenv("MOONCAKE_WEIGHT_TE_DEVICE", "")
    source_engine = TransferEngine()
    assert (
        source_engine.initialize(
            local_hostname,
            "P2PHANDSHAKE",
            protocol,
            transport_device,
        )
        == 0
    )

    source_index = 0
    with (
        ExitStack() as stack,
        _remote_cuda_target(
            local_hostname=local_hostname,
            protocol=protocol,
            device=transport_device,
            target_devices=target_devices,
        ) as (target_endpoint, allocate_target, _execute_reader),
    ):

        def allocate_source(size: int):
            nonlocal source_index
            cuda_device = source_devices[source_index % len(source_devices)]
            source_index += 1
            return stack.enter_context(CudaBuffer(runtimes[cuda_device], size))

        fixture = _build_fixture(
            revision=uuid4().hex,
            source_tp=source_tp,
            target_tp=target_tp,
            allocate_source=allocate_source,
            allocate_target=allocate_target,
            target_endpoint=target_endpoint,
        )
        with _registered_engine_buffers(source_engine, fixture.source_buffers.values()):
            _execute_te_fixture(source_engine, fixture)


@pytest.mark.skipif(
    os.getenv("MOONCAKE_WEIGHT_TE_GPU_E2E") != "1",
    reason="set MOONCAKE_WEIGHT_TE_GPU_E2E=1 to run the CUDA TE test",
)
def test_te_reader_cuda_pulls_packed_tp1_weights_into_tp2_targets() -> None:
    from mooncake.engine import TransferEngine

    source_devices = _parse_cuda_devices(
        os.environ, "MOONCAKE_WEIGHT_SOURCE_CUDA_DEVICES", default="0"
    )
    target_devices = _parse_cuda_devices(
        os.environ, "MOONCAKE_WEIGHT_TARGET_CUDA_DEVICES", default="0"
    )
    runtimes = {device: CudaRuntime(device) for device in sorted(set(source_devices))}
    local_hostname = os.getenv(
        "MOONCAKE_WEIGHT_TE_HOSTNAME",
        os.getenv(
            "MOONCAKE_WEIGHT_LOCAL_HOSTNAME",
            socket.gethostbyname(socket.gethostname()),
        ),
    )
    protocol = os.getenv("MOONCAKE_WEIGHT_TE_PROTOCOL", "rdma")
    transport_device = os.getenv("MOONCAKE_WEIGHT_TE_DEVICE", "")
    source_engine = TransferEngine()
    assert (
        source_engine.initialize(
            local_hostname,
            "P2PHANDSHAKE",
            protocol,
            transport_device,
        )
        == 0
    )
    source_endpoint = f"{local_hostname}:{source_engine.get_rpc_port()}"

    source_index = 0
    with (
        ExitStack() as stack,
        _remote_cuda_target(
            local_hostname=local_hostname,
            protocol=protocol,
            device=transport_device,
            target_devices=target_devices,
        ) as (target_endpoint, allocate_target, execute_reader),
    ):

        def allocate_source(size: int):
            nonlocal source_index
            cuda_device = source_devices[source_index % len(source_devices)]
            source_index += 1
            return stack.enter_context(CudaBuffer(runtimes[cuda_device], size))

        fixture = _build_packed_reader_fixture(
            revision=uuid4().hex,
            source_endpoint=source_endpoint,
            target_endpoint=target_endpoint,
            allocate_source=allocate_source,
            allocate_target=allocate_target,
        )
        with _registered_engine_buffers(source_engine, (fixture.source_buffer,)):
            _execute_reader_fixture(fixture, execute_reader)


@pytest.mark.skipif(
    os.getenv("MOONCAKE_WEIGHT_TE_GPU_E2E") != "1",
    reason="set MOONCAKE_WEIGHT_TE_GPU_E2E=1 to run the CUDA TE test",
)
def test_te_sink_cuda_reshards_independent_experts_across_dimensions() -> None:
    from mooncake.engine import TransferEngine

    source_devices = _parse_cuda_devices(
        os.environ, "MOONCAKE_WEIGHT_SOURCE_CUDA_DEVICES", default="0"
    )
    target_devices = _parse_cuda_devices(
        os.environ, "MOONCAKE_WEIGHT_TARGET_CUDA_DEVICES", default="0"
    )
    runtimes = {device: CudaRuntime(device) for device in sorted(set(source_devices))}
    local_hostname = os.getenv(
        "MOONCAKE_WEIGHT_TE_HOSTNAME",
        os.getenv(
            "MOONCAKE_WEIGHT_LOCAL_HOSTNAME",
            socket.gethostbyname(socket.gethostname()),
        ),
    )
    protocol = os.getenv("MOONCAKE_WEIGHT_TE_PROTOCOL", "rdma")
    transport_device = os.getenv("MOONCAKE_WEIGHT_TE_DEVICE", "")
    source_engine = TransferEngine()
    assert (
        source_engine.initialize(
            local_hostname,
            "P2PHANDSHAKE",
            protocol,
            transport_device,
        )
        == 0
    )

    source_index = 0
    with (
        ExitStack() as stack,
        _remote_cuda_target(
            local_hostname=local_hostname,
            protocol=protocol,
            device=transport_device,
            target_devices=target_devices,
        ) as (target_endpoint, allocate_target, _execute_reader),
    ):

        def allocate_source(size: int):
            nonlocal source_index
            cuda_device = source_devices[source_index % len(source_devices)]
            source_index += 1
            return stack.enter_context(CudaBuffer(runtimes[cuda_device], size))

        fixture = _build_cross_dim_reader_fixture(
            revision=uuid4().hex,
            source_endpoint=(f"{local_hostname}:{source_engine.get_rpc_port()}"),
            target_endpoint=target_endpoint,
            allocate_source=allocate_source,
            allocate_target=allocate_target,
        )
        with _registered_engine_buffers(source_engine, fixture.source_buffers):
            _execute_cross_dim_sink_fixture(fixture, source_engine)


@pytest.mark.skipif(
    os.getenv("MOONCAKE_WEIGHT_TE_GPU_E2E") != "1",
    reason="set MOONCAKE_WEIGHT_TE_GPU_E2E=1 to run the CUDA TE test",
)
def test_te_reader_cuda_reshards_independent_experts_across_dimensions() -> None:
    from mooncake.engine import TransferEngine

    source_devices = _parse_cuda_devices(
        os.environ, "MOONCAKE_WEIGHT_SOURCE_CUDA_DEVICES", default="0"
    )
    target_devices = _parse_cuda_devices(
        os.environ, "MOONCAKE_WEIGHT_TARGET_CUDA_DEVICES", default="0"
    )
    runtimes = {device: CudaRuntime(device) for device in sorted(set(source_devices))}
    local_hostname = os.getenv(
        "MOONCAKE_WEIGHT_TE_HOSTNAME",
        os.getenv(
            "MOONCAKE_WEIGHT_LOCAL_HOSTNAME",
            socket.gethostbyname(socket.gethostname()),
        ),
    )
    protocol = os.getenv("MOONCAKE_WEIGHT_TE_PROTOCOL", "rdma")
    transport_device = os.getenv("MOONCAKE_WEIGHT_TE_DEVICE", "")
    source_engine = TransferEngine()
    assert (
        source_engine.initialize(
            local_hostname,
            "P2PHANDSHAKE",
            protocol,
            transport_device,
        )
        == 0
    )
    source_endpoint = f"{local_hostname}:{source_engine.get_rpc_port()}"

    source_index = 0
    with (
        ExitStack() as stack,
        _remote_cuda_target(
            local_hostname=local_hostname,
            protocol=protocol,
            device=transport_device,
            target_devices=target_devices,
        ) as (target_endpoint, allocate_target, execute_reader),
    ):

        def allocate_source(size: int):
            nonlocal source_index
            cuda_device = source_devices[source_index % len(source_devices)]
            source_index += 1
            return stack.enter_context(CudaBuffer(runtimes[cuda_device], size))

        fixture = _build_cross_dim_reader_fixture(
            revision=uuid4().hex,
            source_endpoint=source_endpoint,
            target_endpoint=target_endpoint,
            allocate_source=allocate_source,
            allocate_target=allocate_target,
        )
        with _registered_engine_buffers(source_engine, fixture.source_buffers):
            _execute_cross_dim_reader_fixture(fixture, execute_reader)
