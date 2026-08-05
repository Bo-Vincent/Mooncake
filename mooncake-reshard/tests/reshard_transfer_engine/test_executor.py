from __future__ import annotations

import pytest

from mooncake.reshard.transfer_engine import (
    BufferRegistrationLease,
    MooncakeTransferEngineExecutor,
    TransferBatch,
    TransferDirection,
    TransferCompletionUnknownError,
    TransferEngineError,
)
from mooncake.reshard.weight import MemoryRegistrationLease


class CompletedTicket:
    status = "COMPLETED"


class FakeEngine:
    def __init__(self) -> None:
        self.calls = []

    def get_engine_ptr(self) -> int:
        return id(self)

    def batch_transfer_sync_read_with_ticket(self, *arguments):
        self.calls.append(("read", arguments))
        return CompletedTicket()

    def batch_transfer_sync_write_with_ticket(self, *arguments):
        self.calls.append(("write", arguments))
        return CompletedTicket()


def batch() -> TransferBatch:
    return TransferBatch(
        endpoint="worker-1:12345",
        source_addresses=(0x1000, 0x2000),
        target_addresses=(0x3000, 0x4000),
        sizes=(64, 128),
    )


def test_resource_neutral_executor_submits_read_and_write_batches() -> None:
    engine = FakeEngine()
    executor = MooncakeTransferEngineExecutor(engine)

    read = executor.execute_batch(batch(), TransferDirection.READ)
    write = executor.execute_batch(batch(), TransferDirection.WRITE)

    assert [call[0] for call in engine.calls] == ["read", "write"]
    assert read.operation_count == write.operation_count == 2
    assert read.nbytes == write.nbytes == 192
    assert read.endpoint == write.endpoint == "worker-1:12345"


def test_transfer_batch_rejects_mismatched_or_invalid_ranges() -> None:
    for values in (
        {"source_addresses": (0x1000,), "target_addresses": (), "sizes": (1,)},
        {
            "source_addresses": (0x1000,),
            "target_addresses": (0x2000,),
            "sizes": (0,),
        },
    ):
        try:
            TransferBatch(endpoint="worker-1:12345", **values)
        except ValueError:
            continue
        raise AssertionError("invalid transfer batch was accepted")


def test_weight_registration_name_reexports_common_buffer_lease() -> None:
    assert MemoryRegistrationLease is BufferRegistrationLease


class UnknownTicket:
    status = "COMPLETION_UNKNOWN"

    def drain(self, timeout_ms: int) -> str:
        return self.status


class SharedEngine(FakeEngine):
    def __init__(self, engine_ptr: int, ticket: UnknownTicket) -> None:
        super().__init__()
        self.engine_ptr = engine_ptr
        self.ticket = ticket

    def get_engine_ptr(self) -> int:
        return self.engine_ptr

    def batch_transfer_sync_write_with_ticket(self, *arguments):
        self.calls.append(("write", arguments))
        return self.ticket


def test_pending_engine_fence_is_shared_across_resource_executors() -> None:
    ticket = UnknownTicket()
    weight = MooncakeTransferEngineExecutor(SharedEngine(0xCAFE, ticket))
    kv = MooncakeTransferEngineExecutor(SharedEngine(0xCAFE, CompletedTicket()))

    with pytest.raises(TransferCompletionUnknownError) as raised:
        weight.execute_batch(batch(), TransferDirection.WRITE)
    weight.retain_pending_resources(
        raised.value.pending_transfer_id,
        registrations=(),
        resources=(ticket,),
    )

    with pytest.raises(TransferEngineError, match="pending transfer"):
        kv.execute_batch(batch(), TransferDirection.WRITE)

    ticket.status = "COMPLETED"
    assert (
        weight.drain_pending_transfer(raised.value.pending_transfer_id) == "COMPLETED"
    )
    assert kv.execute_batch(batch(), TransferDirection.WRITE).operation_count == 2
