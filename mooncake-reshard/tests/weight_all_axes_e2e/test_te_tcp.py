from __future__ import annotations

import os
import socket
from contextlib import ExitStack
from uuid import uuid4

import pytest

from mooncake.reshard.weight import (
    MemoryRegistrationLease,
    MooncakeTransferEngineReader,
)
from weight_all_axes_e2e.builders import (
    _build_fixture,
    _build_packed_reader_fixture,
)
from weight_all_axes_e2e.execution import (
    _execute_reader_fixture,
    _execute_te_fixture,
)
from weight_gpu_e2e.buffers import ManagedBuffer


@pytest.mark.skipif(
    os.getenv("MOONCAKE_WEIGHT_TE_E2E") != "1",
    reason="set MOONCAKE_WEIGHT_TE_E2E=1 to run the Transfer Engine test",
)
@pytest.mark.parametrize(("source_tp", "target_tp"), ((4, 8), (8, 4)))
def test_te_tcp_moves_weights_across_dp_tp_pp_ep_together(
    source_tp: int,
    target_tp: int,
) -> None:
    from mooncake.engine import TransferEngine

    local_hostname = os.getenv(
        "MOONCAKE_WEIGHT_LOCAL_HOSTNAME",
        socket.gethostbyname(socket.gethostname()),
    )
    source_engine = TransferEngine()
    target_engine = TransferEngine()
    assert source_engine.initialize(local_hostname, "P2PHANDSHAKE", "tcp", "") == 0
    assert target_engine.initialize(local_hostname, "P2PHANDSHAKE", "tcp", "") == 0
    target_endpoint = f"{local_hostname}:{target_engine.get_rpc_port()}"

    with ExitStack() as stack:
        fixture = _build_fixture(
            revision=uuid4().hex,
            source_tp=source_tp,
            target_tp=target_tp,
            allocate_source=lambda size: stack.enter_context(
                ManagedBuffer(source_engine, size)
            ),
            allocate_target=lambda size: stack.enter_context(
                ManagedBuffer(target_engine, size)
            ),
            target_endpoint=target_endpoint,
        )
        _execute_te_fixture(source_engine, fixture)


@pytest.mark.skipif(
    os.getenv("MOONCAKE_WEIGHT_TE_E2E") != "1",
    reason="set MOONCAKE_WEIGHT_TE_E2E=1 to run the Transfer Engine test",
)
def test_te_reader_tcp_pulls_packed_tp1_weights_into_tp2_targets() -> None:
    from mooncake.engine import TransferEngine

    local_hostname = os.getenv(
        "MOONCAKE_WEIGHT_LOCAL_HOSTNAME",
        socket.gethostbyname(socket.gethostname()),
    )
    source_engine = TransferEngine()
    target_engine = TransferEngine()
    assert source_engine.initialize(local_hostname, "P2PHANDSHAKE", "tcp", "") == 0
    assert target_engine.initialize(local_hostname, "P2PHANDSHAKE", "tcp", "") == 0
    source_endpoint = f"{local_hostname}:{source_engine.get_rpc_port()}"
    target_endpoint = f"{local_hostname}:{target_engine.get_rpc_port()}"

    with ExitStack() as stack:
        fixture = _build_packed_reader_fixture(
            revision=uuid4().hex,
            source_endpoint=source_endpoint,
            target_endpoint=target_endpoint,
            allocate_source=lambda size: stack.enter_context(
                ManagedBuffer(source_engine, size)
            ),
            allocate_target=lambda size: stack.enter_context(
                ManagedBuffer(target_engine, size)
            ),
        )

        def execute_reader(plan, sources, target):
            target_placement, target_binding = target.single()
            return MooncakeTransferEngineReader(target_engine).execute(
                plan,
                sources.placement,
                sources.bindings,
                target_placement,
                target_binding,
                source_registrations=tuple(
                    MemoryRegistrationLease.from_fragment(
                        fragment,
                        lease_generation=binding.generation,
                        runtime_lease_id=binding.lease_id,
                    )
                    for binding in sources.bindings
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

        _execute_reader_fixture(fixture, execute_reader)
