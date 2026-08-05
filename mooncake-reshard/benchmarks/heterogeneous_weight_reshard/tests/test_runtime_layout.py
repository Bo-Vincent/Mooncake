from __future__ import annotations

from dataclasses import dataclass

import pytest

from benchmarks.heterogeneous_weight_reshard.case_spec import BenchmarkCase, MeshSpec
from benchmarks.heterogeneous_weight_reshard.runtime_layout import (
    RuntimeTopology,
    build_runtime_topology,
    registration_leases,
)
from mooncake.reshard.weight import ParallelRank, ReplicatedAxis, SplitAxis
from mooncake.reshard.weight.planner import (
    bind_logical_transfer_plan,
    plan_placement_transfer,
)


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
    sources = build_runtime_topology(
        case,
        side="source",
        buffers=_buffers(8, 16, 0x100000),
        endpoint="172.16.1.107:12000",
        revision="revision-1",
    )
    targets = build_runtime_topology(
        case,
        side="target",
        buffers=_buffers(8, 8, 0x200000),
        endpoint="172.16.1.108:13000",
        revision="revision-1",
    )

    assert isinstance(sources, RuntimeTopology)
    assert not hasattr(sources, "resource_id")
    assert not hasattr(sources, "placements")
    assert len(sources.placement.parts) == 8
    assert len(sources.bindings) == 8
    assert len(targets.placement.parts) == 8
    assert len(targets.bindings) == 8
    assert sources.placement.parts[5].fragments[0].rank == ParallelRank(dp=1, tp=1)
    assert sources.placement.parts[5].fragments[0].global_offset == (2, 0)
    assert sources.placement.parts[5].fragments[0].local_shape == (2, 8)
    assert targets.placement.parts[5].fragments[0].rank == ParallelRank(dp=0, tp=5)
    assert targets.placement.parts[5].fragments[0].global_offset == (0, 5)
    assert targets.placement.parts[5].fragments[0].local_shape == (8, 1)
    assert sources.placement.tensors[0].shard_dims == (0,)
    assert targets.placement.tensors[0].shard_dims == (1,)
    assert sources.placement.tensors[0].parallel_axes == (
        ReplicatedAxis(kind="dp"),
        SplitAxis(kind="tp", dim=0),
    )
    assert targets.placement.tensors[0].parallel_axes == (
        ReplicatedAxis(kind="dp"),
        SplitAxis(kind="tp", dim=1),
    )
    assert sources.placement.topology.dp_size == 2
    assert sources.placement.topology.tp_size == 4
    assert targets.placement.topology.dp_size == 1
    assert targets.placement.topology.tp_size == 8
    assert sources.bindings[0].fragments[0].owner.pointer == 0x100000
    assert sources.bindings[0].fragments[0].itemsize == 1
    assert sources.bindings[0].fragments[0].local_shape == (2, 8)
    assert sources.bindings[0].fragments[0].strides_bytes == (8, 1)
    assert sources.bindings[0].fragments[0].storage_address == 0x100000
    assert sources.bindings[0].fragments[0].storage_nbytes == 16
    assert sources.bindings[0].fragments[0].storage_offset_bytes == 0
    assert all(binding.lease_id for binding in (*sources.bindings, *targets.bindings))

    logical = plan_placement_transfer(sources.placement, targets.placement)
    plan = bind_logical_transfer_plan(
        logical,
        targets.bindings,
        source_bindings=sources.bindings,
    )
    assert plan.total_bytes == 64
    assert {region.source.rank.dp for region in logical.operations} == {0}


def test_runtime_layout_rejects_buffer_count_and_size_mismatch() -> None:
    case = _case()

    with pytest.raises(ValueError, match="source requires 8 buffers"):
        build_runtime_topology(
            case,
            side="source",
            buffers=_buffers(7, 16, 0x100000),
            endpoint="172.16.1.107:12000",
            revision="revision-1",
        )

    buffers = _buffers(8, 16, 0x100000)
    buffers[3].size = 15
    with pytest.raises(ValueError, match="source buffer 3 must contain 16 bytes"):
        build_runtime_topology(
            case,
            side="source",
            buffers=buffers,
            endpoint="172.16.1.107:12000",
            revision="revision-1",
        )


def test_registration_leases_preserve_runtime_lease_identity() -> None:
    topology = build_runtime_topology(
        _case(),
        side="target",
        buffers=_buffers(8, 8, 0x200000),
        endpoint="172.16.1.108:13000",
        revision="revision-1",
    )

    leases = registration_leases(topology.bindings)

    assert len(leases) == 8
    assert leases[0].runtime_lease_id == topology.bindings[0].lease_id
    assert leases[0].address == topology.bindings[0].fragments[0].address
    assert leases[0].lease_generation == 1
