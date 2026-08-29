from __future__ import annotations

import ctypes
from contextlib import ExitStack
from typing import Mapping, Protocol


def _parse_cuda_devices(
    environ: Mapping[str, str],
    name: str,
    *,
    default: str,
) -> tuple[int, ...]:
    raw = environ.get(name, default)
    if raw != raw.strip():
        raise ValueError(f"{name} must contain unique non-negative device IDs")
    devices = []
    try:
        for item in raw.split(","):
            device = int(item)
            if str(device) != item or device < 0 or device in devices:
                raise ValueError
            devices.append(device)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must contain unique non-negative device IDs"
        ) from error
    if not devices:
        raise ValueError(f"{name} must not be empty")
    return tuple(devices)


def _cuda_rank_devices(devices: tuple[int, ...], ranks: int) -> tuple[int, ...]:
    if not devices or ranks <= 0 or any(device < 0 for device in devices):
        raise ValueError("CUDA rank placement is invalid")
    return tuple(devices[rank % len(devices)] for rank in range(ranks))


class TransferBuffer(Protocol):
    size: int

    @property
    def pointer(self) -> int: ...

    def activate(self) -> None: ...

    def fill(self, value: int) -> None: ...

    def read_range(self, offset: int, nbytes: int) -> bytes: ...


class CudaBuffer:
    def __init__(self, runtime, size: int) -> None:
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

    @property
    def pointer(self) -> int:
        return self.address.value

    def activate(self) -> None:
        self._runtime.activate()

    def write(self, data: bytes) -> None:
        if len(data) != self.size:
            raise ValueError("CUDA buffer write size mismatch")
        source = ctypes.create_string_buffer(data)
        self.activate()
        self._runtime.check(
            self._runtime.library.cudaMemcpy(
                self.address,
                ctypes.cast(source, ctypes.c_void_p),
                self.size,
                1,
            )
        )

    def fill(self, value: int) -> None:
        if not 0 <= value <= 255:
            raise ValueError("CUDA fill value must fit in one byte")
        self.activate()
        self._runtime.check(
            self._runtime.library.cudaMemset(self.address, value, self.size)
        )

    def read_range(self, offset: int, nbytes: int) -> bytes:
        if offset < 0 or nbytes < 0 or offset + nbytes > self.size:
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

    def read(self) -> bytes:
        return self.read_range(0, self.size)

    def zero(self) -> None:
        self.fill(0)

    def close(self) -> None:
        if self.address.value is not None:
            self.activate()
            self._runtime.check(self._runtime.library.cudaFree(self.address))
            self.address = ctypes.c_void_p()

    def __enter__(self) -> CudaBuffer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ManagedBuffer:
    def __init__(self, engine, size: int) -> None:
        self._engine = engine
        self.size = size
        self.address = engine.allocate_managed_buffer(size)
        if self.address == 0:
            raise RuntimeError("failed to allocate managed transfer buffer")

    @property
    def pointer(self) -> int:
        return self.address

    def activate(self) -> None:
        return None

    def write(self, data: bytes) -> None:
        if len(data) != self.size:
            raise ValueError("managed buffer write size mismatch")
        result = self._engine.write_bytes_to_buffer(
            self.address,
            data,
            self.size,
        )
        if result != 0:
            raise RuntimeError(f"managed buffer write failed: {result}")

    def read(self) -> bytes:
        return self.read_range(0, self.size)

    def fill(self, value: int) -> None:
        if not 0 <= value <= 255:
            raise ValueError("managed fill value must fit in one byte")
        ctypes.memset(self.address, value, self.size)

    def read_range(self, offset: int, nbytes: int) -> bytes:
        if offset < 0 or nbytes < 0 or offset + nbytes > self.size:
            raise ValueError("managed buffer read range is invalid")
        return ctypes.string_at(self.address + offset, nbytes)

    def zero(self) -> None:
        self.fill(0)

    def close(self) -> None:
        if self.address != 0:
            self._engine.free_managed_buffer(self.address, self.size)
            self.address = 0

    def __enter__(self) -> ManagedBuffer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class CudaRuntime:
    def __init__(self, device: int = 0) -> None:
        if device < 0:
            raise ValueError("CUDA device must be non-negative")
        self.device = device
        self.library = ctypes.CDLL("libcudart.so")
        self.library.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self.library.cudaFree.argtypes = [ctypes.c_void_p]
        self.library.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.library.cudaMemset.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
        ]
        self.library.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.library.cudaGetErrorString.restype = ctypes.c_char_p
        self.library.cudaSetDevice.argtypes = [ctypes.c_int]
        self.activate()

    def activate(self) -> None:
        self.check(self.library.cudaSetDevice(self.device))

    def check(self, result: int) -> None:
        if result == 0:
            return
        message = self.library.cudaGetErrorString(result).decode()
        raise RuntimeError(f"CUDA call failed: {result}: {message}")


def _cuda_rank_buffers(
    stack: ExitStack,
    runtimes: Mapping[int, CudaRuntime],
    devices: tuple[int, ...],
    *,
    ranks: int,
    size: int,
) -> tuple[list[CudaBuffer], tuple[int, ...]]:
    rank_devices = _cuda_rank_devices(devices, ranks)
    buffers = [
        stack.enter_context(CudaBuffer(runtimes[device], size))
        for device in rank_devices
    ]
    return buffers, rank_devices
