from __future__ import annotations

import os
import socket
from contextlib import ExitStack

import pytest

from mooncake.reshard.weight import WeightStore

from .buffers import (
    CudaRuntime,
    _cuda_rank_buffers,
    _parse_cuda_devices,
)
from .execution import _run_store_iteration
from .manifests import (
    _expected_tp_segments,
    _expected_tp_shard,
    _tensor,
    _verify_tp_buffers,
)


@pytest.mark.skipif(
    os.getenv("MOONCAKE_WEIGHT_GPU_STORE_E2E") != "1",
    reason="set MOONCAKE_WEIGHT_GPU_STORE_E2E=1 to run the CUDA Store test",
)
@pytest.mark.parametrize(
    ("source_tp", "target_tp"),
    (
        pytest.param(4, 8, id="tp4-to-tp8"),
        pytest.param(8, 4, id="tp8-to-tp4"),
    ),
)
def test_gpu_store_round_trip_reshards_tp_split_and_merge(
    source_tp: int,
    target_tp: int,
) -> None:
    from mooncake.store import MooncakeDistributedStore

    source_devices = _parse_cuda_devices(
        os.environ,
        "MOONCAKE_WEIGHT_SOURCE_CUDA_DEVICES",
        default="0",
    )
    target_devices = _parse_cuda_devices(
        os.environ,
        "MOONCAKE_WEIGHT_TARGET_CUDA_DEVICES",
        default="0",
    )
    runtimes = {
        device: CudaRuntime(device)
        for device in sorted(set(source_devices) | set(target_devices))
    }
    total_bytes = 8 * 1024 * 1024
    store = MooncakeDistributedStore()
    result = store.setup(
        os.getenv(
            "MOONCAKE_WEIGHT_LOCAL_HOSTNAME", socket.gethostbyname(socket.gethostname())
        ),
        os.getenv(
            "MOONCAKE_WEIGHT_METADATA_SERVER",
            "http://127.0.0.1:8080/metadata",
        ),
        128 * 1024 * 1024,
        64 * 1024 * 1024,
        os.getenv("MOONCAKE_WEIGHT_PROTOCOL", "tcp"),
        os.getenv("MOONCAKE_WEIGHT_DEVICE", "eth0"),
        os.getenv("MOONCAKE_WEIGHT_MASTER", "127.0.0.1:50051"),
    )
    assert result == 0
    try:
        with ExitStack() as stack:
            source_buffers, _ = _cuda_rank_buffers(
                stack,
                runtimes,
                source_devices,
                ranks=source_tp,
                size=total_bytes // source_tp,
            )
            target_buffers, _ = _cuda_rank_buffers(
                stack,
                runtimes,
                target_devices,
                ranks=target_tp,
                size=total_bytes // target_tp,
            )
            for rank, buffer in enumerate(source_buffers):
                buffer.fill(rank + 1)

            durations = _run_store_iteration(
                store=store,
                weight_store=WeightStore(store),
                tensor=_tensor(total_bytes),
                source_buffers=source_buffers,
                target_buffers=target_buffers,
                namespace=f"gpu-e2e-tp{source_tp}-to-tp{target_tp}",
            )
            assert set(durations) == {
                "prepare",
                "upload",
                "commit",
                "manifest_get",
                "plan_load",
                "load",
                "e2e",
            }
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            assert close() == 0


@pytest.mark.parametrize(
    "source_tp,target_tp,target_rank,expected",
    [
        (4, 2, 0, bytes([1]) * 4 + bytes([2]) * 4),
        (4, 2, 1, bytes([3]) * 4 + bytes([4]) * 4),
        (2, 4, 0, bytes([1]) * 4),
        (2, 4, 1, bytes([1]) * 4),
        (2, 4, 2, bytes([2]) * 4),
        (2, 4, 3, bytes([2]) * 4),
    ],
)
def test_expected_tp_shard_handles_split_and_merge(
    source_tp: int,
    target_tp: int,
    target_rank: int,
    expected: bytes,
) -> None:
    assert (
        _expected_tp_shard(
            total_bytes=16,
            source_tp=source_tp,
            target_tp=target_tp,
            target_rank=target_rank,
        )
        == expected
    )


@pytest.mark.parametrize(
    "source_tp,target_tp,target_rank,expected",
    [
        (4, 2, 0, ((0, 4, 1), (4, 4, 2))),
        (4, 2, 1, ((0, 4, 3), (4, 4, 4))),
        (2, 4, 0, ((0, 4, 1),)),
        (2, 4, 1, ((0, 4, 1),)),
        (2, 4, 2, ((0, 4, 2),)),
        (2, 4, 3, ((0, 4, 2),)),
    ],
)
def test_expected_tp_segments_handle_split_and_merge(
    source_tp: int,
    target_tp: int,
    target_rank: int,
    expected: tuple[tuple[int, int, int], ...],
) -> None:
    assert (
        _expected_tp_segments(
            total_bytes=16,
            source_tp=source_tp,
            target_tp=target_tp,
            target_rank=target_rank,
        )
        == expected
    )


def test_verify_tp_buffers_checks_every_chunk() -> None:
    class FakeBuffer:
        def __init__(self, data: bytes) -> None:
            self.data = data
            self.size = len(data)
            self.reads = []

        def read_range(self, offset: int, nbytes: int) -> bytes:
            self.reads.append((offset, nbytes))
            return self.data[offset : offset + nbytes]

    buffers = [
        FakeBuffer(bytes([1]) * 4 + bytes([2]) * 4),
        FakeBuffer(bytes([3]) * 4 + bytes([4]) * 4),
    ]

    _verify_tp_buffers(
        buffers,
        total_bytes=16,
        source_tp=4,
        target_tp=2,
        chunk_bytes=3,
    )

    assert buffers[0].reads == [(0, 3), (3, 1), (4, 3), (7, 1)]
    assert buffers[1].reads == [(0, 3), (3, 1), (4, 3), (7, 1)]


def test_verify_tp_buffers_rejects_corruption() -> None:
    class FakeBuffer:
        size = 8

        def read_range(self, offset: int, nbytes: int) -> bytes:
            data = bytes([1]) * 4 + bytes([2]) * 4
            if offset <= 3 < offset + nbytes:
                data = data[:3] + b"\xff" + data[4:]
            return data[offset : offset + nbytes]

    with pytest.raises(AssertionError, match="target TP rank 0"):
        _verify_tp_buffers(
            [FakeBuffer(), FakeBuffer()],
            total_bytes=16,
            source_tp=4,
            target_tp=2,
            chunk_bytes=4,
        )
