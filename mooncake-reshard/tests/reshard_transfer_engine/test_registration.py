from __future__ import annotations

import gc
import weakref
from contextlib import AbstractContextManager

import pytest

from mooncake.reshard.contracts import RuntimeBindingFragment
from mooncake.reshard.transfer_engine import (
    AllocationFence,
    MooncakeTransferEngineExecutor,
    TerminalTransferState,
    TransferEngineError,
    TransferRegistrationCleanupPendingError,
)
from mooncake.reshard.transfer_engine.lifetime import AllocationTokenSet
from mooncake.reshard.transfer_engine.registration import (
    registered_sources,
    registered_targets,
)


class FakeRegistrationEngine:
    def __init__(
        self,
        register_results: list[int | BaseException] | None = None,
        unregister_results: list[int | BaseException] | None = None,
    ) -> None:
        self.register_calls: list[tuple[int, int]] = []
        self.unregister_calls: list[int] = []
        self.register_results = list(register_results or ())
        self.unregister_results = list(unregister_results or ())

    def get_engine_ptr(self) -> int:
        return id(self)

    def register_memory(self, address: int, nbytes: int) -> int:
        self.register_calls.append((address, nbytes))
        if self.register_results:
            result = self.register_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        return 0

    def unregister_memory(self, address: int) -> int:
        self.unregister_calls.append(address)
        if self.unregister_results:
            result = self.unregister_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        return 0


class RecordingAllocationToken:
    def __init__(
        self,
        fence: AllocationFence,
        released_states: list[TerminalTransferState],
    ) -> None:
        self.fence = fence
        self.released_states = released_states

    def release_after_terminal(self, terminal_state: TerminalTransferState) -> None:
        self.released_states.append(terminal_state)


def _fragment(
    fragment_id: str,
    *,
    storage_address: int = 0x10000,
    storage_nbytes: int = 64,
    storage_offset_bytes: int = 0,
) -> RuntimeBindingFragment:
    return RuntimeBindingFragment(
        placement_fragment_id=f"placement-{fragment_id}",
        fragment_id=fragment_id,
        address=storage_address + storage_offset_bytes,
        nbytes=16,
        worker_id="worker-0",
        endpoint="worker-0:12345",
        device="cuda:0",
        itemsize=4,
        local_shape=(4,),
        strides_bytes=(4,),
        storage_address=storage_address,
        storage_nbytes=storage_nbytes,
        storage_offset_bytes=storage_offset_bytes,
    )


def _token(
    token_id: str,
    released_states: list[TerminalTransferState],
) -> RecordingAllocationToken:
    return RecordingAllocationToken(
        AllocationFence(
            resource_id="resource",
            revision="revision",
            placement_id="placement",
            placement_digest="placement-digest",
            instance_id="instance",
            participant_id="participant",
            runtime_lease_id="lease",
            runtime_generation=1,
            binding_digest="binding-digest",
            fragment_ids=("runtime-0",),
            token_id=token_id,
        ),
        released_states,
    )


def _registered(
    label: str,
    engine: FakeRegistrationEngine,
    executor: MooncakeTransferEngineExecutor,
    fragments: tuple[RuntimeBindingFragment, ...],
    *,
    resources: tuple[object, ...] = (),
    lifetime_tokens: AllocationTokenSet | None = None,
) -> AbstractContextManager[None]:
    if label == "source":
        return registered_sources(
            engine,
            executor,
            fragments,
            pre_registered=False,
            registrations=None,
            lease_generation=1,
            runtime_lease_id="lease",
            resources=resources,
            lifetime_tokens=lifetime_tokens,
        )
    if label == "target":
        return registered_targets(
            engine,
            executor,
            fragments,
            pre_registered=False,
            resources=resources,
            lifetime_tokens=lifetime_tokens,
        )
    raise AssertionError(f"unknown registration label: {label}")


@pytest.mark.parametrize("label", ("source", "target"))
def test_registration_registers_and_unregisters_each_allocation_once(
    label: str,
) -> None:
    engine = FakeRegistrationEngine()
    executor = MooncakeTransferEngineExecutor(engine)
    fragments = (
        _fragment("runtime-0"),
        _fragment("runtime-1", storage_offset_bytes=16),
    )

    with _registered(label, engine, executor, fragments):
        assert engine.register_calls == [(0x10000, 64)]
        assert engine.unregister_calls == []

    assert engine.unregister_calls == [0x10000]


@pytest.mark.parametrize("label", ("source", "target"))
def test_unregister_failure_retains_cleanup_until_retry_succeeds(
    label: str,
) -> None:
    engine = FakeRegistrationEngine(unregister_results=[-7, -8, 0])
    executor = MooncakeTransferEngineExecutor(engine)
    released_states: list[TerminalTransferState] = []
    token = _token(f"{label}-token", released_states)
    token_ref = weakref.ref(token)
    token_set = AllocationTokenSet((token,))
    resource = object()

    with pytest.raises(TransferRegistrationCleanupPendingError) as raised:
        with _registered(
            label,
            engine,
            executor,
            (_fragment("runtime-0"),),
            resources=(resource,),
            lifetime_tokens=token_set,
        ):
            pass

    pending_transfer_id = raised.value.pending_transfer_id
    assert engine.register_calls == [(0x10000, 64)]
    assert engine.unregister_calls == [0x10000]
    assert executor.pending_transfer_ids() == (pending_transfer_id,)
    assert executor.pending_transfer_status(pending_transfer_id) == "COMPLETED"
    assert token_set.pending
    assert released_states == []

    del raised
    del token
    del token_set
    gc.collect()
    assert token_ref() is not None

    with pytest.raises(TransferEngineError, match="registration cleanup failed"):
        executor.drain_pending_transfer(pending_transfer_id)

    assert executor.pending_transfer_ids() == (pending_transfer_id,)
    assert engine.unregister_calls == [0x10000, 0x10000]
    assert released_states == []
    gc.collect()
    assert token_ref() is not None

    assert executor.drain_pending_transfer(pending_transfer_id) == "COMPLETED"
    assert executor.pending_transfer_ids() == ()
    assert engine.unregister_calls == [0x10000, 0x10000, 0x10000]
    assert released_states == [TerminalTransferState.COMPLETED]
    gc.collect()
    assert token_ref() is None


@pytest.mark.parametrize("label", ("source", "target"))
def test_pending_unregister_exception_requires_restart_without_replay(
    label: str,
) -> None:
    engine = FakeRegistrationEngine(unregister_results=[-7])
    executor = MooncakeTransferEngineExecutor(engine)

    with pytest.raises(TransferRegistrationCleanupPendingError) as raised:
        with _registered(
            label,
            engine,
            executor,
            (_fragment("runtime-0"),),
        ):
            pass

    pending_transfer_id = raised.value.pending_transfer_id
    engine.unregister_results.append(RuntimeError("cleanup outcome unknown"))
    with pytest.raises(TransferEngineError, match="restart required"):
        executor.drain_pending_transfer(pending_transfer_id)

    assert executor.pending_transfer_status(pending_transfer_id) == (
        "COMPLETION_UNKNOWN_RESTART_REQUIRED"
    )
    assert executor.drain_pending_transfer(pending_transfer_id) == (
        "COMPLETION_UNKNOWN"
    )
    assert engine.unregister_calls == [0x10000, 0x10000]


@pytest.mark.parametrize("label", ("source", "target"))
def test_registration_rejects_duplicate_allocation_capacity_mismatch(
    label: str,
) -> None:
    engine = FakeRegistrationEngine()
    executor = MooncakeTransferEngineExecutor(engine)
    fragments = (
        _fragment("runtime-0", storage_nbytes=64),
        _fragment(
            "runtime-1",
            storage_nbytes=32,
            storage_offset_bytes=16,
        ),
    )

    with pytest.raises(
        TransferEngineError,
        match=rf"{label} allocation capacity mismatch",
    ):
        with _registered(label, engine, executor, fragments):
            pass

    assert engine.register_calls == []
    assert engine.unregister_calls == []


@pytest.mark.parametrize("label", ("source", "target"))
def test_cleanup_interruption_quarantines_registration_until_restart(
    label: str,
) -> None:
    engine = FakeRegistrationEngine(
        unregister_results=[KeyboardInterrupt("cleanup interrupted")]
    )
    executor = MooncakeTransferEngineExecutor(engine)
    released_states: list[TerminalTransferState] = []
    token_set = AllocationTokenSet((_token(f"{label}-token", released_states),))

    with pytest.raises(KeyboardInterrupt, match="cleanup interrupted"):
        with _registered(
            label,
            engine,
            executor,
            (_fragment("runtime-0"),),
            resources=(object(),),
            lifetime_tokens=token_set,
        ):
            pass

    (pending_transfer_id,) = executor.pending_transfer_ids()
    assert executor.pending_transfer_status(pending_transfer_id) == (
        "COMPLETION_UNKNOWN_RESTART_REQUIRED"
    )
    assert executor.drain_pending_transfer(pending_transfer_id) == (
        "COMPLETION_UNKNOWN"
    )
    assert engine.unregister_calls == [0x10000]
    assert token_set.pending
    assert released_states == []
    with pytest.raises(TransferEngineError, match="restart-required"):
        with executor.submission():
            pass


@pytest.mark.parametrize("label", ("source", "target"))
def test_registration_interruption_quarantines_unknown_allocation_until_restart(
    label: str,
) -> None:
    engine = FakeRegistrationEngine(
        register_results=[KeyboardInterrupt("registration interrupted")]
    )
    executor = MooncakeTransferEngineExecutor(engine)

    with pytest.raises(KeyboardInterrupt, match="registration interrupted"):
        with _registered(
            label,
            engine,
            executor,
            (_fragment("runtime-0"),),
            resources=(object(),),
        ):
            pass

    (pending_transfer_id,) = executor.pending_transfer_ids()
    assert executor.pending_transfer_status(pending_transfer_id) == (
        "COMPLETION_UNKNOWN_RESTART_REQUIRED"
    )
    assert engine.register_calls == [(0x10000, 64)]
    assert engine.unregister_calls == []
