from __future__ import annotations

import ctypes

import pytest

from .buffers import (
    CudaBuffer,
    CudaRuntime,
    _cuda_rank_devices,
    _parse_cuda_devices,
)


def test_cuda_devices_parse_and_assign_round_robin() -> None:
    devices = _parse_cuda_devices(
        {"MOONCAKE_WEIGHT_SOURCE_CUDA_DEVICES": "0,2,3"},
        "MOONCAKE_WEIGHT_SOURCE_CUDA_DEVICES",
        default="0",
    )

    assert devices == (0, 2, 3)
    assert _cuda_rank_devices(devices, 5) == (0, 2, 3, 0, 2)


@pytest.mark.parametrize("raw", ["", "0,", "-1", "0,0", "gpu0", " 0, 1 "])
def test_cuda_devices_reject_invalid_or_duplicate_values(raw: str) -> None:
    with pytest.raises(ValueError, match="MOONCAKE_WEIGHT_SOURCE_CUDA_DEVICES"):
        _parse_cuda_devices(
            {"MOONCAKE_WEIGHT_SOURCE_CUDA_DEVICES": raw},
            "MOONCAKE_WEIGHT_SOURCE_CUDA_DEVICES",
            default="0",
        )


def test_cuda_runtime_activates_configured_device(monkeypatch) -> None:
    calls = []

    class FakeFunction:
        def __init__(self, name: str, result=0) -> None:
            self.name = name
            self.result = result
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            calls.append((self.name, *args))
            return self.result

    class FakeLibrary:
        cudaMalloc = FakeFunction("malloc")
        cudaFree = FakeFunction("free")
        cudaMemcpy = FakeFunction("memcpy")
        cudaMemset = FakeFunction("memset")
        cudaGetErrorString = FakeFunction("error_string", b"error")
        cudaSetDevice = FakeFunction("set_device")

    monkeypatch.setattr(ctypes, "CDLL", lambda _: FakeLibrary())

    runtime = CudaRuntime(device=3)
    runtime.activate()

    assert calls == [("set_device", 3), ("set_device", 3)]


def test_cuda_buffer_activates_its_runtime_before_alloc_and_free() -> None:
    events = []

    class FakeLibrary:
        @staticmethod
        def cudaMalloc(address, size: int) -> int:
            events.append(("malloc", size))
            ctypes.cast(address, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(
                1234
            )
            return 0

        @staticmethod
        def cudaFree(address) -> int:
            events.append(("free", address.value))
            return 0

    class FakeRuntime:
        library = FakeLibrary()

        @staticmethod
        def activate() -> None:
            events.append(("activate",))

        @staticmethod
        def check(result: int) -> None:
            assert result == 0

    buffer = CudaBuffer(FakeRuntime(), 16)
    buffer.close()

    assert events == [
        ("activate",),
        ("malloc", 16),
        ("activate",),
        ("free", 1234),
    ]
