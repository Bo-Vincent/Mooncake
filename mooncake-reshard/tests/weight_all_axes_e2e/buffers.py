from __future__ import annotations

import ctypes
import multiprocessing
from contextlib import contextmanager
from dataclasses import replace
from multiprocessing.connection import Connection
from typing import Callable, Iterator

from mooncake.reshard._compat import _strict_zip
from mooncake.reshard.weight import (
    BoundWeightFragment,
    DirectReadReceipt,
    MemoryRegistrationLease,
    MooncakeTransferEngineReader,
    TransferPlan,
)
from weight_all_axes_e2e.fixtures import RuntimeInputs
from weight_gpu_e2e.buffers import CudaBuffer, CudaRuntime


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
    """Drop process-local owners while retaining runtime address evidence."""

    def wire_binding(binding):
        return replace(
            binding,
            fragments=tuple(
                replace(fragment, owner=None) for fragment in binding.fragments
            ),
        )

    def wire_fragment(fragment):
        if not isinstance(fragment, BoundWeightFragment):
            return fragment
        binding = replace(fragment.binding, owner=None)
        return replace(fragment, binding=binding, owner=None)

    return (
        replace(
            plan,
            operations=tuple(
                replace(
                    operation,
                    source=wire_fragment(operation.source),
                    target=wire_fragment(operation.target),
                )
                for operation in plan.operations
            ),
        ),
        replace(
            sources,
            bindings=tuple(wire_binding(binding) for binding in sources.bindings),
        ),
        replace(
            target,
            bindings=tuple(wire_binding(binding) for binding in target.bindings),
        ),
    )


def _cuda_target_worker(
    connection: Connection,
    local_hostname: str,
    protocol: str,
    device: str,
    target_devices: tuple[int, ...],
) -> None:
    from mooncake.engine import TransferEngine

    engine = TransferEngine()
    buffers = []
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
                _, plan, sources, target = command
                target_placement, target_binding = target.single()
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
                )
                connection.send(("executed", receipts))
            elif command[0] == "shutdown":
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
    process = context.Process(
        target=_cuda_target_worker,
        args=(
            child_connection,
            local_hostname,
            protocol,
            device,
            target_devices,
        ),
    )
    process.start()
    child_connection.close()
    response = parent_connection.recv()
    if response[0] == "error":
        process.join(timeout=10)
        raise RuntimeError(response[1])
    assert response[0] == "ready"
    target_endpoint = response[1]

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
        parent_connection.send(
            ("execute_reader", *_wire_safe_reader_payload(plan, sources, target))
        )
        response = parent_connection.recv()
        if response[0] == "error":
            raise RuntimeError(response[1])
        assert response[0] == "executed"
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
        parent_connection.close()
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
