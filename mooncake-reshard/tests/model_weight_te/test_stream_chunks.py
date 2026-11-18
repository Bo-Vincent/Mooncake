from dataclasses import dataclass

import pytest
import torch
from mooncake.reshard.weight.stream_chunks import (
    chunk_nbytes,
    chunk_table,
    iter_packed_chunks,
    pack_bucket,
    plan_chunks,
)


@dataclass(frozen=True)
class Shard:
    tensor_id: str
    tensor: object
    global_shape: tuple
    global_offset: tuple


def test_framework_neutral_chunk_roundtrip():
    values = [
        torch.arange(21, dtype=torch.float64).view(3, 7),
        torch.arange(13, dtype=torch.int16),
    ]
    chunks = plan_chunks(
        [Shard(str(i), t, tuple(t.shape), (0,) * t.ndim) for i, t in enumerate(values)],
        16,
    )
    restored = [torch.empty_like(t) for t in values]
    for chunk in chunks:
        assert chunk_nbytes(chunk) <= 16
        packed = pack_bucket(chunk, torch.empty(16, dtype=torch.uint8), torch)
        cursor = 0
        for spec in chunk:
            size = spec.tensor.numel() * spec.tensor.element_size()
            index = tuple(
                slice(o, o + n) for o, n in zip(spec.global_offset, spec.tensor.shape)
            )
            restored[int(spec.tensor_id)][index] = (
                packed[cursor : cursor + size]
                .view(spec.tensor.dtype)
                .view(spec.tensor.shape)
            )
            cursor += size
    assert all(torch.equal(a, b) for a, b in zip(values, restored))
    from mooncake.reshard.weight import TiledCOOLayout

    assert TiledCOOLayout.from_dict(chunk_table(chunks[0])).records


@pytest.mark.parametrize("slots_count", [1, 2, 4])
def test_packed_stream_waits_before_copy_and_drains(slots_count):
    source = torch.arange(18, dtype=torch.float32)
    chunks = plan_chunks([Shard("weight", source, (18,), (0,))], 16)
    slots = tuple(torch.empty(16, dtype=torch.uint8) for _ in range(slots_count))
    events, received = [], []

    def acquire(ordinal):
        events.append(("wait", ordinal))

    def ready():
        events.append(("ready",))

    for ordinal, chunk, used in iter_packed_chunks(
        chunks, slots, acquire=acquire, ready=ready, torch_module=torch
    ):
        assert events[-2:] == [("wait", ordinal), ("ready",)]
        assert used.data_ptr() == slots[ordinal % slots_count].data_ptr()
        received.append(used.clone().view(torch.float32))
    assert torch.equal(torch.cat(received), source)
    assert events[-min(len(chunks), slots_count) :] == [
        ("wait", i + slots_count)
        for i in range(max(0, len(chunks) - slots_count), len(chunks))
    ]


def test_stream_failed_publication_does_not_drain_or_advance():
    source = torch.arange(12, dtype=torch.float32)
    chunks = plan_chunks([Shard("weight", source, (12,), (0,))], 16)
    waits = []
    iterator = iter_packed_chunks(
        chunks,
        (torch.empty(16, dtype=torch.uint8),),
        acquire=waits.append,
        ready=lambda: None,
        torch_module=torch,
    )
    next(iterator)
    iterator.close()
    assert waits == [0]
    with pytest.raises(ValueError, match="slot"):
        next(
            iter_packed_chunks(
                chunks, (), acquire=waits.append, ready=lambda: None, torch_module=torch
            )
        )


def test_empty_writer_needs_no_slots_or_callbacks():
    def unexpected(*args):
        raise AssertionError("Empty stream must not wait or copy")

    assert (
        list(
            iter_packed_chunks(
                (), (), acquire=unexpected, ready=unexpected, torch_module=None
            )
        )
        == []
    )
