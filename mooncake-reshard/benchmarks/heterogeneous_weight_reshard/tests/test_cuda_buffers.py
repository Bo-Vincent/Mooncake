from __future__ import annotations

import ctypes
from contextlib import ExitStack
from typing import Callable

import pytest

from benchmarks.heterogeneous_weight_reshard.cuda_buffers import (
    CudaBuffer,
    CudaRuntime,
    registered_engine_buffers,
)


def _pointer_value(value) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, ctypes.c_void_p):
        return value.value or 0
    return ctypes.cast(value, ctypes.c_void_p).value or 0


class FakeCudaFunction:
    def __init__(self, callback: Callable):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class FakeCudaLibrary:
    def __init__(self, events: list[tuple] | None = None) -> None:
        self.events = events if events is not None else []
        self.allocations: dict[int, ctypes.Array] = {}
        self.cudaMalloc = FakeCudaFunction(self._malloc)
        self.cudaFree = FakeCudaFunction(self._free)
        self.cudaMemcpy = FakeCudaFunction(self._memcpy)
        self.cudaMemset = FakeCudaFunction(self._memset)
        self.cudaSetDevice = FakeCudaFunction(self._set_device)
        self.cudaGetErrorString = FakeCudaFunction(self._error_string)

    def _malloc(self, address, size: int) -> int:
        storage = ctypes.create_string_buffer(size)
        pointer = ctypes.addressof(storage)
        ctypes.cast(address, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(
            pointer
        )
        self.allocations[pointer] = storage
        self.events.append(("malloc", pointer, size))
        return 0

    def _free(self, address) -> int:
        pointer = _pointer_value(address)
        self.events.append(("free", pointer))
        self.allocations.pop(pointer)
        return 0

    def _memcpy(self, target, source, size: int, kind: int) -> int:
        target_pointer = _pointer_value(target)
        source_pointer = _pointer_value(source)
        self.events.append(("memcpy", target_pointer, source_pointer, size, kind))
        ctypes.memmove(target_pointer, source_pointer, size)
        return 0

    def _memset(self, address, value: int, size: int) -> int:
        pointer = _pointer_value(address)
        self.events.append(("memset", pointer, value, size))
        ctypes.memset(pointer, value, size)
        return 0

    def _set_device(self, device: int) -> int:
        self.events.append(("set_device", device))
        return 0

    @staticmethod
    def _error_string(result: int) -> bytes:
        return f"fake error {result}".encode()


class FakeEngine:
    def __init__(
        self,
        events: list[tuple],
        *,
        register_results: tuple[int | BaseException, ...] = (),
        unregister_results: dict[int, int | BaseException] | None = None,
    ) -> None:
        self.events = events
        self.register_results = list(register_results)
        self.unregister_results = unregister_results or {}

    def register_memory(self, address: int, nbytes: int) -> int:
        self.events.append(("register", address, nbytes))
        outcome = self.register_results.pop(0) if self.register_results else 0
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def unregister_memory(self, address: int) -> int:
        self.events.append(("unregister", address))
        outcome = self.unregister_results.get(address, 0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _event_names(events: list[tuple]) -> list[str]:
    return [event[0] for event in events]


def test_cuda_runtime_configures_injected_library_and_reports_errors() -> None:
    library = FakeCudaLibrary()

    runtime = CudaRuntime(3, library=library)

    assert library.events == [("set_device", 3)]
    assert library.cudaMalloc.argtypes == [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
    ]
    assert library.cudaFree.argtypes == [ctypes.c_void_p]
    assert library.cudaMemcpy.argtypes == [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    assert library.cudaMemset.argtypes == [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
    ]
    assert library.cudaSetDevice.argtypes == [ctypes.c_int]
    assert library.cudaGetErrorString.argtypes == [ctypes.c_int]
    assert library.cudaGetErrorString.restype is ctypes.c_char_p
    for function in (
        library.cudaMalloc,
        library.cudaFree,
        library.cudaMemcpy,
        library.cudaMemset,
        library.cudaSetDevice,
    ):
        assert function.restype is ctypes.c_int

    runtime.activate()
    assert library.events[-1] == ("set_device", 3)
    with pytest.raises(RuntimeError, match="CUDA call failed: 7: fake error 7"):
        runtime.check(7)


def test_cuda_runtime_rejects_negative_device() -> None:
    with pytest.raises(ValueError, match="CUDA device must be non-negative"):
        CudaRuntime(-1, library=FakeCudaLibrary())


def test_cuda_buffer_activates_before_each_operation_and_closes_once() -> None:
    events: list[tuple] = []
    library = FakeCudaLibrary(events)
    runtime = CudaRuntime(2, library=library)
    events.clear()

    buffer = CudaBuffer(runtime, 8)
    pointer = buffer.pointer
    assert buffer.size == 8
    assert buffer.device == 2
    assert pointer > 0
    assert _event_names(events) == ["set_device", "malloc"]

    events.clear()
    buffer.fill(0xAB)
    assert _event_names(events) == ["set_device", "memset"]
    assert buffer.read_range(2, 4) == bytes([0xAB]) * 4
    assert _event_names(events) == [
        "set_device",
        "memset",
        "set_device",
        "memcpy",
    ]

    buffer.zero()
    assert buffer.read_range(0, buffer.size) == bytes(buffer.size)

    events.clear()
    buffer.close()
    buffer.close()
    assert events == [("set_device", 2), ("free", pointer)]


@pytest.mark.parametrize(
    ("offset", "nbytes"),
    ((-1, 1), (0, -1), (7, 2)),
)
def test_cuda_buffer_rejects_invalid_read_range(offset: int, nbytes: int) -> None:
    with CudaBuffer(CudaRuntime(0, library=FakeCudaLibrary()), 8) as buffer:
        with pytest.raises(ValueError, match="CUDA buffer read range is invalid"):
            buffer.read_range(offset, nbytes)


@pytest.mark.parametrize("value", (-1, 256))
def test_cuda_buffer_rejects_invalid_fill(value: int) -> None:
    with CudaBuffer(CudaRuntime(0, library=FakeCudaLibrary()), 8) as buffer:
        with pytest.raises(ValueError, match="CUDA fill value must fit in one byte"):
            buffer.fill(value)


def test_registration_precedes_reverse_unregister_and_buffer_free() -> None:
    events: list[tuple] = []
    library = FakeCudaLibrary(events)
    runtime = CudaRuntime(1, library=library)
    engine = FakeEngine(events)

    with ExitStack() as stack:
        first = stack.enter_context(CudaBuffer(runtime, 8))
        second = stack.enter_context(CudaBuffer(runtime, 16))
        first_pointer = first.pointer
        second_pointer = second.pointer
        events.clear()

        with registered_engine_buffers(engine, (first, second)):
            events.append(("body",))

        assert events == [
            ("set_device", 1),
            ("register", first_pointer, first.size),
            ("set_device", 1),
            ("register", second_pointer, second.size),
            ("body",),
            ("set_device", 1),
            ("unregister", second_pointer),
            ("set_device", 1),
            ("unregister", first_pointer),
        ]
        assert "free" not in _event_names(events)

    assert events[-4:] == [
        ("set_device", 1),
        ("free", second_pointer),
        ("set_device", 1),
        ("free", first_pointer),
    ]


def test_registration_failure_unregisters_only_successful_buffers() -> None:
    events: list[tuple] = []
    runtime = CudaRuntime(0, library=FakeCudaLibrary(events))
    engine = FakeEngine(events, register_results=(0, 17))

    with ExitStack() as stack:
        first = stack.enter_context(CudaBuffer(runtime, 8))
        second = stack.enter_context(CudaBuffer(runtime, 8))
        events.clear()

        with pytest.raises(
            RuntimeError,
            match=rf"register_memory failed for {second.pointer}: 17",
        ):
            with registered_engine_buffers(engine, (first, second)):
                pytest.fail("registration context must not be entered")

        assert [event for event in events if event[0] == "unregister"] == [
            ("unregister", first.pointer)
        ]
        assert "free" not in _event_names(events)


def test_business_exception_is_preserved_when_cleanup_succeeds() -> None:
    events: list[tuple] = []
    runtime = CudaRuntime(0, library=FakeCudaLibrary(events))
    engine = FakeEngine(events)
    business_error = ValueError("body failed")

    with CudaBuffer(runtime, 8) as buffer:
        with pytest.raises(ValueError) as caught:
            with registered_engine_buffers(engine, (buffer,)):
                raise business_error

    assert caught.value is business_error


def test_cleanup_failures_are_aggregated_with_business_error_as_cause() -> None:
    events: list[tuple] = []
    runtime = CudaRuntime(0, library=FakeCudaLibrary(events))
    business_error = ValueError("body failed")

    with ExitStack() as stack:
        first = stack.enter_context(CudaBuffer(runtime, 8))
        second = stack.enter_context(CudaBuffer(runtime, 8))
        first_pointer = first.pointer
        second_pointer = second.pointer
        engine = FakeEngine(
            events,
            unregister_results={
                second_pointer: 23,
                first_pointer: RuntimeError("unregister exploded"),
            },
        )

        with pytest.raises(RuntimeError) as caught:
            with registered_engine_buffers(engine, (first, second)):
                raise business_error

    assert caught.value.__cause__ is business_error
    assert "body failed; unregister_memory failed" in str(caught.value)
    assert str(second_pointer) in str(caught.value)
    assert str(first_pointer) in str(caught.value)
    assert [event for event in events if event[0] == "unregister"] == [
        ("unregister", second_pointer),
        ("unregister", first_pointer),
    ]


def test_cleanup_failure_without_business_error_is_reported() -> None:
    events: list[tuple] = []
    runtime = CudaRuntime(0, library=FakeCudaLibrary(events))

    with CudaBuffer(runtime, 8) as buffer:
        engine = FakeEngine(events, unregister_results={buffer.pointer: 31})
        with pytest.raises(RuntimeError, match="unregister_memory failed") as caught:
            with registered_engine_buffers(engine, (buffer,)):
                pass

    assert caught.value.__cause__ is None
