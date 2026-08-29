from __future__ import annotations

import os
import socket
from contextlib import ExitStack

import pytest

from .buffers import ManagedBuffer
from .execution import _run_te_iteration
from .manifests import _tensor


@pytest.mark.skipif(
    os.getenv("MOONCAKE_WEIGHT_TE_E2E") != "1",
    reason="set MOONCAKE_WEIGHT_TE_E2E=1 to run the Transfer Engine test",
)
def test_te_tcp_round_trip_reshards_tp_split_and_merge() -> None:
    from mooncake.engine import TransferEngine

    total_bytes = 8 * 1024 * 1024
    source_engine = TransferEngine()
    target_engine = TransferEngine()
    local_hostname = os.getenv(
        "MOONCAKE_WEIGHT_LOCAL_HOSTNAME",
        socket.gethostbyname(socket.gethostname()),
    )
    assert source_engine.initialize(local_hostname, "P2PHANDSHAKE", "tcp", "") == 0
    assert target_engine.initialize(local_hostname, "P2PHANDSHAKE", "tcp", "") == 0
    assert source_engine.get_rpc_port() != target_engine.get_rpc_port()
    target_endpoint = f"{local_hostname}:{target_engine.get_rpc_port()}"

    for source_tp, target_tp in ((2, 4), (4, 2)):
        with ExitStack() as stack:
            source_buffers = [
                stack.enter_context(
                    ManagedBuffer(source_engine, total_bytes // source_tp)
                )
                for _ in range(source_tp)
            ]
            target_buffers = [
                stack.enter_context(
                    ManagedBuffer(target_engine, total_bytes // target_tp)
                )
                for _ in range(target_tp)
            ]
            for rank, buffer in enumerate(source_buffers):
                buffer.fill(rank + 1)

            durations = _run_te_iteration(
                source_engine=source_engine,
                target_endpoint=target_endpoint,
                tensor=_tensor(total_bytes),
                source_buffers=source_buffers,
                target_buffers=target_buffers,
            )
            assert set(durations) == {"plan", "transfer", "e2e"}
