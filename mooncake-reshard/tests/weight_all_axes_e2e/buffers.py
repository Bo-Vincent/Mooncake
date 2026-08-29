from __future__ import annotations

import ctypes
import multiprocessing
import os
import sys
from contextlib import contextmanager
from dataclasses import replace
from multiprocessing.connection import Connection
from threading import Thread
from typing import Callable, Iterator, Optional, Sequence

from mooncake.reshard._compat import _strict_zip
from mooncake.reshard.transfer_engine.lifetime import TerminalTransferState
from mooncake.reshard.weight import (
    DirectReadReceipt,
    MemoryRegistrationLease,
    MooncakeTransferEngineReader,
    TransferPlan,
    WeightRuntimeBindingManifest,
)
from mooncake.reshard.weight.lifetime import (
    AcquiredWeightBinding,
    weight_allocation_fence,
)
from weight_all_axes_e2e.fixtures import RuntimeInputs
from weight_gpu_e2e.buffers import CudaBuffer, CudaRuntime
from weight_gpu_e2e.lifetime import allocation_guards_for_bindings


def _e2e_debug(event: str) -> None:
    if os.getenv("MOONCAKE_WEIGHT_E2E_DEBUG") == "1":
        print(
            f"[weight-e2e pid={os.getpid()}] {event}",
            file=sys.stderr,
            flush=True,
        )


class HostBuffer:
    def __init__(self, size: int) -> None:
        self.size = size
        self._storage = ctypes.create_string_buffer(size)

    @property
    def pointer(self) -> int:
        return ctypes.addressof(self._storage)

    def activate(self) -> None:
        return None

    def fill(self, value: int) -> None:
        ctypes.memset(self.pointer, value, self.size)

    def write(self, data: bytes) -> None:
        if len(data) != self.size:
            raise ValueError("host buffer write size mismatch")
        ctypes.memmove(self.pointer, data, self.size)

    def zero(self) -> None:
        self.fill(0)

    def read_range(self, offset: int, nbytes: int) -> bytes:
        return ctypes.string_at(self.pointer + offset, nbytes)


class HostTransferEngine:
    def register_memory(self, address: int, nbytes: int) -> int:
        del address, nbytes
        return 0

    def unregister_memory(self, address: int) -> int:
        del address
        return 0

    def batch_transfer_sync_write(
        self, endpoint, source_addresses, target_addresses, sizes
    ) -> int:
        del endpoint
        for source, target, size in _strict_zip(
            source_addresses, target_addresses, sizes
        ):
            ctypes.memmove(target, source, size)
        return 0

    def batch_transfer_sync_read(
        self, endpoint, target_addresses, source_addresses, sizes
    ) -> int:
        del endpoint
        for target, source, size in _strict_zip(
            target_addresses, source_addresses, sizes
        ):
            ctypes.memmove(target, source, size)
        return 0


class RemoteCudaBuffer:
    def __init__(
        self,
        connection: Connection,
        index: int,
        pointer: int,
        size: int,
    ) -> None:
        self._connection = connection
        self._index = index
        self._pointer = pointer
        self.size = size

    @property
    def pointer(self) -> int:
        return self._pointer

    def activate(self) -> None:
        return None

    def zero(self) -> None:
        return None

    def write(self, data: bytes) -> None:
        if len(data) != self.size:
            raise ValueError("remote CUDA buffer write size mismatch")
        self._connection.send(("write", self._index, data))
        response = self._connection.recv()
        if response[0] == "error":
            raise RuntimeError(response[1])
        assert response[0] == "written"

    def read_range(self, offset: int, nbytes: int) -> bytes:
        self._connection.send(("read", self._index, offset, nbytes))
        response = self._connection.recv()
        if response[0] == "error":
            raise RuntimeError(response[1])
        assert response[0] == "data"
        return response[1]


def _wire_safe_reader_payload(
    plan: TransferPlan,
    sources: RuntimeInputs,
    target: RuntimeInputs,
) -> tuple[TransferPlan, RuntimeInputs, RuntimeInputs]:
    """Return the normal payload; fragment pickle reducers drop local owners."""

    return plan, sources, target


class _RemoteSourceAllocationToken:
    """Target-side token backed by the source framework's guard RPC."""

    def __init__(
        self,
        connection: Connection,
        binding: WeightRuntimeBindingManifest,
        fragment_ids: Sequence[str],
        token_id: str,
    ) -> None:
        self._connection = connection
        self._token_id = token_id
        self._fence = weight_allocation_fence(
            binding,
            fragment_ids,
            token_id=token_id,
        )
        self._released = False

    @property
    def fence(self):
        return self._fence

    def release_after_terminal(self, terminal_state: TerminalTransferState) -> None:
        if self._released:
            return
        _e2e_debug(f"source-guard release send token={self._token_id}")
        self._connection.send(("release", self._token_id, terminal_state.value))
        response = self._connection.recv()
        if response[0] == "error":
            raise RuntimeError(response[1])
        if response != ("released", self._token_id):
            raise RuntimeError("source allocation guard release response is invalid")
        _e2e_debug(f"source-guard release received token={self._token_id}")
        self._released = True


class _RemoteSourceBindingGuard:
    """Framework adapter proxy; the source process owns the real allocation."""

    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def acquire(
        self,
        *,
        transfer_id: str,
        expected_binding: WeightRuntimeBindingManifest,
        required_fragment_ids: Sequence[str],
    ) -> AcquiredWeightBinding:
        _e2e_debug(
            f"source-guard acquire send participant={expected_binding.participant_id}"
        )
        self._connection.send(
            (
                "acquire",
                transfer_id,
                expected_binding,
                tuple(required_fragment_ids),
            )
        )
        response = self._connection.recv()
        if response[0] == "error":
            raise RuntimeError(response[1])
        if response[0] != "acquired" or type(response[1]) is not str:
            raise RuntimeError("source allocation guard acquire response is invalid")
        _e2e_debug(
            "source-guard acquire received "
            f"participant={expected_binding.participant_id}"
        )
        return AcquiredWeightBinding(
            binding=expected_binding,
            token=_RemoteSourceAllocationToken(
                self._connection,
                expected_binding,
                required_fragment_ids,
                response[1],
            ),
        )


def _binding_with_local_owners(
    binding: WeightRuntimeBindingManifest,
    buffers: Sequence[CudaBuffer],
) -> WeightRuntimeBindingManifest:
    buffers_by_address = {buffer.pointer: buffer for buffer in buffers}
    fragments = []
    for fragment in binding.fragments:
        owner = buffers_by_address.get(fragment.storage_address)
        if owner is None:
            raise RuntimeError(
                f"target allocation is missing for storage {fragment.storage_address}"
            )
        fragments.append(replace(fragment, owner=owner))
    return replace(binding, fragments=tuple(fragments))


def _serve_source_allocation_guards(
    connection: Connection,
    owner_bindings: Sequence[WeightRuntimeBindingManifest],
) -> None:
    tokens = {}
    try:
        _e2e_debug("source-guard server started")
        while True:
            try:
                request = connection.recv()
            except (EOFError, OSError):
                return
            if request[0] == "stop":
                _e2e_debug("source-guard server stopped")
                connection.send(("stopped",))
                return
            try:
                if request[0] == "acquire":
                    _, transfer_id, expected_binding, fragment_ids = request
                    key = (
                        expected_binding.instance_id,
                        expected_binding.participant_id,
                    )
                    provider = allocation_guards_for_bindings(
                        (expected_binding,),
                        owner_bindings=owner_bindings,
                    ).get(key)
                    if provider is None:
                        raise RuntimeError(
                            "source allocation guard provider is missing"
                        )
                    acquired = provider.acquire(
                        transfer_id=transfer_id,
                        expected_binding=expected_binding,
                        required_fragment_ids=fragment_ids,
                    )
                    token_id = acquired.token.fence.token_id
                    if token_id in tokens:
                        raise RuntimeError(
                            "source allocation guard token is duplicated"
                        )
                    tokens[token_id] = acquired.token
                    _e2e_debug(f"source-guard server acquired token={token_id}")
                    connection.send(("acquired", token_id))
                elif request[0] == "release":
                    _, token_id, terminal_state = request
                    token = tokens.pop(token_id, None)
                    if token is None:
                        raise RuntimeError("source allocation guard token is unknown")
                    token.release_after_terminal(TerminalTransferState(terminal_state))
                    _e2e_debug(f"source-guard server released token={token_id}")
                    connection.send(("released", token_id))
                else:
                    raise RuntimeError(
                        f"unknown source allocation guard command: {request[0]}"
                    )
            except BaseException as error:
                connection.send(("error", repr(error)))
    finally:
        for token in tokens.values():
            token.release_after_terminal(TerminalTransferState.ABORTED)
        connection.close()


def _cuda_target_worker(
    connection: Connection,
    source_guard_connection: Connection,
    local_hostname: str,
    protocol: str,
    device: str,
    target_devices: tuple[int, ...],
) -> None:
    from mooncake.engine import TransferEngine

    engine = TransferEngine()
    buffers = []
    source_guard_active = False
    runtimes = {target: CudaRuntime(target) for target in target_devices}
    try:
        result = engine.initialize(
            local_hostname,
            "P2PHANDSHAKE",
            protocol,
            device,
        )
        if result != 0:
            raise RuntimeError(f"target TransferEngine initialize failed: {result}")
        connection.send(("ready", f"{local_hostname}:{engine.get_rpc_port()}"))
        _e2e_debug("target worker ready")
        while True:
            command = connection.recv()
            if command[0] == "allocate":
                index = len(buffers)
                runtime = runtimes[target_devices[index % len(target_devices)]]
                buffer = CudaBuffer(runtime, command[1])
                buffer.zero()
                buffer.activate()
                result = engine.register_memory(buffer.pointer, buffer.size)
                if result != 0:
                    buffer.close()
                    raise RuntimeError(f"target register_memory failed: {result}")
                buffers.append(buffer)
                connection.send(("allocated", index, buffer.pointer, buffer.size))
            elif command[0] == "write":
                _, index, data = command
                buffers[index].write(data)
                connection.send(("written",))
            elif command[0] == "read":
                _, index, offset, nbytes = command
                connection.send(("data", buffers[index].read_range(offset, nbytes)))
            elif command[0] == "execute_reader":
                _e2e_debug("target worker received execute_reader")
                _, plan, sources, target = command
                target_placement, target_binding = target.single()
                target_binding = _binding_with_local_owners(target_binding, buffers)
                _e2e_debug("target worker rebound local target owners")
                _e2e_debug("target worker entering reader.execute")
                source_guard_active = True
                receipts = MooncakeTransferEngineReader(engine).execute(
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
                    source_allocation_guards={
                        (binding.instance_id, binding.participant_id): (
                            _RemoteSourceBindingGuard(source_guard_connection)
                        )
                        for binding in sources.bindings
                    },
                    target_allocation_guards=allocation_guards_for_bindings(
                        (target_binding,)
                    ),
                )
                _e2e_debug("target worker reader.execute completed")
                connection.send(("executed", receipts))
            elif command[0] == "shutdown":
                if source_guard_active:
                    source_guard_connection.send(("stop",))
                    stopped = source_guard_connection.recv()
                    if stopped != ("stopped",):
                        raise RuntimeError(
                            "source allocation guard server did not stop"
                        )
                break
            else:
                raise RuntimeError(f"unknown target command: {command[0]}")

        for buffer in reversed(buffers):
            buffer.activate()
            result = engine.unregister_memory(buffer.pointer)
            if result != 0:
                raise RuntimeError(f"target unregister_memory failed: {result}")
        for buffer in reversed(buffers):
            buffer.close()
        connection.send(("stopped",))
    except BaseException as exc:
        connection.send(("error", repr(exc)))
    finally:
        source_guard_connection.close()
        connection.close()


@contextmanager
def _remote_cuda_target(
    *,
    local_hostname: str,
    protocol: str,
    device: str,
    target_devices: tuple[int, ...],
) -> Iterator[
    tuple[
        str,
        Callable[[int], RemoteCudaBuffer],
        Callable[
            [TransferPlan, RuntimeInputs, RuntimeInputs],
            tuple[DirectReadReceipt, ...],
        ],
    ]
]:
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe()
    source_guard_parent, source_guard_child = context.Pipe()
    process = context.Process(
        target=_cuda_target_worker,
        args=(
            child_connection,
            source_guard_child,
            local_hostname,
            protocol,
            device,
            target_devices,
        ),
    )
    process.start()
    child_connection.close()
    source_guard_child.close()
    response = parent_connection.recv()
    if response[0] == "error":
        process.join(timeout=10)
        raise RuntimeError(response[1])
    assert response[0] == "ready"
    target_endpoint = response[1]
    guard_server: Optional[Thread] = None
    source_binding_identity: Optional[tuple[tuple[str, str, int, str], ...]] = None

    def allocate(size: int) -> RemoteCudaBuffer:
        parent_connection.send(("allocate", size))
        allocation = parent_connection.recv()
        if allocation[0] == "error":
            raise RuntimeError(allocation[1])
        assert allocation[0] == "allocated"
        return RemoteCudaBuffer(
            parent_connection,
            allocation[1],
            allocation[2],
            allocation[3],
        )

    def execute_reader(
        plan: TransferPlan,
        sources: RuntimeInputs,
        target: RuntimeInputs,
    ) -> tuple[DirectReadReceipt, ...]:
        nonlocal guard_server, source_binding_identity
        current_identity = tuple(
            sorted(
                (
                    binding.instance_id,
                    binding.participant_id,
                    binding.generation,
                    binding.lease_id,
                )
                for binding in sources.bindings
            )
        )
        if guard_server is None:
            source_binding_identity = current_identity
            guard_server = Thread(
                target=_serve_source_allocation_guards,
                args=(source_guard_parent, sources.bindings),
                daemon=True,
            )
            guard_server.start()
        elif source_binding_identity != current_identity:
            raise RuntimeError("remote target cannot change source guard binding")
        _e2e_debug("parent sending execute_reader")
        parent_connection.send(
            ("execute_reader", *_wire_safe_reader_payload(plan, sources, target))
        )
        _e2e_debug("parent waiting for execute_reader result")
        response = parent_connection.recv()
        if response[0] == "error":
            raise RuntimeError(response[1])
        assert response[0] == "executed"
        _e2e_debug("parent received execute_reader result")
        return response[1]

    try:
        yield target_endpoint, allocate, execute_reader
    finally:
        if process.is_alive():
            parent_connection.send(("shutdown",))
            stopped = parent_connection.recv()
            if stopped[0] == "error":
                raise RuntimeError(stopped[1])
            assert stopped[0] == "stopped"
        if guard_server is not None:
            guard_server.join(timeout=10)
            if guard_server.is_alive():
                raise RuntimeError("source allocation guard server did not stop")
        parent_connection.close()
        source_guard_parent.close()
        process.join(timeout=20)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
            raise RuntimeError("target TransferEngine process did not stop")
        if process.exitcode != 0:
            raise RuntimeError(
                f"target TransferEngine process exited with {process.exitcode}"
            )


@contextmanager
def _registered_engine_buffers(engine, buffers) -> Iterator[None]:
    registered = []
    try:
        for buffer in buffers:
            buffer.activate()
            result = engine.register_memory(buffer.pointer, buffer.size)
            if result != 0:
                raise RuntimeError(f"register_memory failed: {result}")
            registered.append(buffer)
        yield
    finally:
        failures = []
        for buffer in reversed(registered):
            buffer.activate()
            result = engine.unregister_memory(buffer.pointer)
            if result != 0:
                failures.append((buffer.pointer, result))
        if failures:
            raise RuntimeError(f"unregister_memory failed: {failures}")
