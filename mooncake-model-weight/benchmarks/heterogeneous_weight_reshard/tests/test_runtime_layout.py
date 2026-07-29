from __future__ import annotations

from dataclasses import dataclass

import pytest

from benchmarks.heterogeneous_weight_reshard.case_spec import BenchmarkCase, MeshSpec
from benchmarks.heterogeneous_weight_reshard.runtime_layout import (
    build_runtime_manifests,
    registration_leases,
)
from mooncake.model_weight import ParallelRank, plan_runtime_transfer


@dataclass
class FakeBuffer:
    pointer: int
    size: int


def _case() -> BenchmarkCase:
    return BenchmarkCase(
        id="dp2_tp4_to_tp8_cross_dim",
        category="physical",
        source=MeshSpec(replicas=2, shards=4, shard_dim=0),
        target=MeshSpec(replicas=1, shards=8, shard_dim=1),
        global_shape=(8, 8),
        required_ranks=16,
    )


def _buffers(count: int, size: int, base: int) -> list[FakeBuffer]:
    return [
        FakeBuffer(pointer=base + rank * 0x1000, size=size) for rank in range(count)
    ]


def test_runtime_layout_uses_explicit_mesh_geometry_for_source_and_target() -> None:
    case = _case()
    sources = build_runtime_manifests(
        case,
        side="source",
        buffers=_buffers(8, 16, 0x100000),
        endpoint="172.16.1.107:12000",
        revision="revision-1",
    )
    targets = build_runtime_manifests(
        case,
        side="target",
        buffers=_buffers(8, 8, 0x200000),
        endpoint="172.16.1.108:13000",
        revision="revision-1",
    )

    assert len(sources) == 8
    assert len(targets) == 8
    assert sources[5].fragments[0].rank == ParallelRank(dp=1, tp=1)
    assert sources[5].fragments[0].global_offset == (2, 0)
    assert sources[5].fragments[0].local_shape == (2, 8)
    assert targets[5].fragments[0].rank == ParallelRank(dp=0, tp=5)
    assert targets[5].fragments[0].global_offset == (0, 5)
    assert targets[5].fragments[0].local_shape == (8, 1)
    assert sources[0].tensors[0].shard_dims == (0,)
    assert targets[0].tensors[0].shard_dims == (1,)
    assert sources[0].fragments[0].owner.pointer == 0x100000
    assert all(manifest.lease_id for manifest in (*sources, *targets))

    plan = plan_runtime_transfer(sources, targets)
    assert plan.total_bytes == 64
    assert {region.source.rank.dp for region in plan.operations} == {0}


def test_runtime_layout_rejects_buffer_count_and_size_mismatch() -> None:
    case = _case()

    with pytest.raises(ValueError, match="source requires 8 buffers"):
        build_runtime_manifests(
            case,
            side="source",
            buffers=_buffers(7, 16, 0x100000),
            endpoint="172.16.1.107:12000",
            revision="revision-1",
        )

    buffers = _buffers(8, 16, 0x100000)
    buffers[3].size = 15
    with pytest.raises(ValueError, match="source buffer 3 must contain 16 bytes"):
        build_runtime_manifests(
            case,
            side="source",
            buffers=buffers,
            endpoint="172.16.1.107:12000",
            revision="revision-1",
        )


def test_registration_leases_preserve_runtime_lease_identity() -> None:
    manifests = build_runtime_manifests(
        _case(),
        side="target",
        buffers=_buffers(8, 8, 0x200000),
        endpoint="172.16.1.108:13000",
        revision="revision-1",
    )

    leases = registration_leases(manifests)

    assert len(leases) == 8
    assert leases[0].runtime_lease_id == manifests[0].lease_id
    assert leases[0].address == manifests[0].fragments[0].address
    assert leases[0].lease_generation == 1
