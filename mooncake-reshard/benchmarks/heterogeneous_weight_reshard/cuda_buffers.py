from __future__ import annotations

import ctypes
from contextlib import contextmanager
from typing import Any, Iterable, Iterator


__all__ = ["CudaBuffer", "CudaRuntime", "registered_engine_buffers"]


def _configure_function(function: Any, argtypes: list[Any], restype: Any) -> None:
    try:
        function.argtypes = argtypes
        function.restype = restype
    except (AttributeError, TypeError):
        return


class CudaRuntime:
    def __init__(self, device: int = 0, library: Any | None = None) -> None:
        if type(device) is not int or device < 0:
            raise ValueError("CUDA device must be non-negative")
        self.device = device
        self.library = library if library is not None else ctypes.CDLL("libcudart.so")
        _configure_function(
            self.library.cudaMalloc,
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
            ctypes.c_int,
        )
        _configure_function(
            self.library.cudaFree,
            [ctypes.c_void_p],
            ctypes.c_int,
        )
        _configure_function(
            self.library.cudaMemcpy,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ],
            ctypes.c_int,
        )
        _configure_function(
            self.library.cudaMemset,
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t],
            ctypes.c_int,
        )
        _configure_function(
            self.library.cudaSetDevice,
            [ctypes.c_int],
            ctypes.c_int,
        )
        _configure_function(
            self.library.cudaGetErrorString,
            [ctypes.c_int],
            ctypes.c_char_p,
        )
        self.activate()

    def activate(self) -> None:
        self.check(self.library.cudaSetDevice(self.device))

    def check(self, result: int) -> None:
        if result == 0:
            return
        raw_message = self.library.cudaGetErrorString(result)
        if isinstance(raw_message, bytes):
            message = raw_message.decode(errors="replace")
        elif raw_message is None:
            message = "unknown CUDA error"
        else:
            message = str(raw_message)
        raise RuntimeError(f"CUDA call failed: {result}: {message}")


class CudaBuffer:
    def __init__(self, runtime: CudaRuntime, size: int) -> None:
        if type(size) is not int or size <= 0:
            raise ValueError("CUDA buffer size must be positive")
        self._runtime = runtime
        self.size = size
        self.address = ctypes.c_void_p()
        self.activate()
        self._runtime.check(
            self._runtime.library.cudaMalloc(
                ctypes.byref(self.address),
                self.size,
            )
        )
        if self.address.value is None:
            raise RuntimeError("cudaMalloc returned a null pointer")

    @property
    def pointer(self) -> int:
        return self.address.value or 0

    @property
    def device(self) -> int:
        return self._runtime.device

    def activate(self) -> None:
        self._runtime.activate()

    def fill(self, value: int) -> None:
        if type(value) is not int or not 0 <= value <= 255:
            raise ValueError("CUDA fill value must fit in one byte")
        self.activate()
        self._runtime.check(
            self._runtime.library.cudaMemset(self.address, value, self.size)
        )

    def zero(self) -> None:
        self.fill(0)

    def read_range(self, offset: int, nbytes: int) -> bytes:
        if (
            type(offset) is not int
            or type(nbytes) is not int
            or offset < 0
            or nbytes < 0
            or offset + nbytes > self.size
        ):
            raise ValueError("CUDA buffer read range is invalid")
        target = ctypes.create_string_buffer(nbytes)
        self.activate()
        self._runtime.check(
            self._runtime.library.cudaMemcpy(
                ctypes.cast(target, ctypes.c_void_p),
                ctypes.c_void_p(self.pointer + offset),
                nbytes,
                2,
            )
        )
        return target.raw

    def close(self) -> None:
        if self.address.value is None:
            return
        self.activate()
        self._runtime.check(self._runtime.library.cudaFree(self.address))
        self.address = ctypes.c_void_p()

    def __enter__(self) -> CudaBuffer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@contextmanager
def registered_engine_buffers(
    engine: Any,
    buffers: Iterable[CudaBuffer],
) -> Iterator[None]:
    registered: list[tuple[CudaBuffer, int]] = []
    primary_error: BaseException | None = None
    try:
        for buffer in buffers:
            address = buffer.pointer
            buffer.activate()
            result = engine.register_memory(address, buffer.size)
            if result != 0:
                raise RuntimeError(f"register_memory failed for {address}: {result}")
            registered.append((buffer, address))
        yield
    except BaseException as error:
        primary_error = error

    failures: list[tuple[int, object]] = []
    for buffer, address in reversed(registered):
        try:
            buffer.activate()
            result = engine.unregister_memory(address)
        except BaseException as error:
            failures.append((address, repr(error)))
            continue
        if result != 0:
            failures.append((address, result))

    if failures:
        detail = f"unregister_memory failed: {failures}"
        if primary_error is not None:
            raise RuntimeError(f"{primary_error}; {detail}") from primary_error
        raise RuntimeError(detail)
    if primary_error is not None:
        raise primary_error
