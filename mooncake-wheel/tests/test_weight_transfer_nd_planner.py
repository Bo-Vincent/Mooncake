from __future__ import annotations

from itertools import product
from math import prod

import pytest

from mooncake.weight_transfer.manifest import (
    ParallelRank,
    RuntimeFragment,
    RuntimeManifest,
    TensorDescriptor,
)
from mooncake.weight_transfer.planner import TransferRegion, plan_runtime_transfer


MODEL_ID = "qwen-family-moe"
REVISION = "step-42"


def tensor_descriptor(
    tensor_id: str,
    *,
    global_shape: tuple[int, ...],
    shard_dims: tuple[int, ...],
    layer_id: int | None = 0,
) -> TensorDescriptor:
    return TensorDescriptor(
        tensor_id=tensor_id,
        global_shape=global_shape,
        dtype="bfloat16",
        itemsize=2,
        partition_dim=None,
        layer_id=layer_id,
        expert_id=None,
        layout_fingerprint="framework:logical-contiguous:v2",
        shard_dims=shard_dims,
    )


def build_manifests(
    side: str,
    placements: list[
        tuple[
            TensorDescriptor,
            ParallelRank,
            tuple[int, ...],
            tuple[int, ...],
        ]
    ],
    *,
    address_base: int,
) -> tuple[RuntimeManifest, ...]:
    grouped: dict[
        ParallelRank,
        list[tuple[TensorDescriptor, tuple[int, ...], tuple[int, ...]]],
    ] = {}
    for tensor, rank, offset, shape in placements:
        grouped.setdefault(rank, []).append((tensor, offset, shape))

    manifests = []
    address = address_base
    for rank in sorted(grouped, key=lambda item: (item.dp, item.pp, item.ep, item.tp)):
        worker_id = f"{side}-d{rank.dp}-p{rank.pp}-e{rank.ep}-t{rank.tp}"
        fragments = []
        tensors: dict[str, TensorDescriptor] = {}
        for tensor, offset, shape in sorted(
            grouped[rank], key=lambda item: item[0].tensor_id
        ):
            nbytes = prod(shape) * tensor.itemsize
            fragments.append(
                RuntimeFragment(
                    fragment_id=f"{worker_id}-{tensor.tensor_id}",
                    tensor_id=tensor.tensor_id,
                    global_offset=offset,
                    local_shape=shape,
                    address=address,
                    nbytes=nbytes,
                    worker_id=worker_id,
                    endpoint=f"{worker_id}:12345",
                    rank=rank,
                    lease_generation=1,
                )
            )
            address += nbytes + 4096
            tensors[tensor.tensor_id] = tensor
        manifests.append(
            RuntimeManifest(
                model_id=MODEL_ID,
                revision=REVISION,
                instance_id=worker_id,
                tensors=tuple(
                    sorted(tensors.values(), key=lambda item: item.tensor_id)
                ),
                fragments=tuple(fragments),
                format_version=2,
            )
        )
    return tuple(manifests)


def ep_tp_placements(
    tensors: tuple[TensorDescriptor, ...],
    *,
    dp: int,
    pp_owner: dict[str, int],
    ep: int,
    tp: int,
    tp_dim: int,
) -> list[tuple[TensorDescriptor, ParallelRank, tuple[int, ...], tuple[int, ...]]]:
    placements = []
    for tensor in tensors:
        assert tensor.global_shape[0] % ep == 0
        assert tensor.global_shape[tp_dim] % tp == 0
        expert_extent = tensor.global_shape[0] // ep
        tp_extent = tensor.global_shape[tp_dim] // tp
        for dp_rank, ep_rank, tp_rank in product(range(dp), range(ep), range(tp)):
            shape = list(tensor.global_shape)
            offset = [0] * len(shape)
            shape[0] = expert_extent
            offset[0] = ep_rank * expert_extent
            shape[tp_dim] = tp_extent
            offset[tp_dim] = tp_rank * tp_extent
            placements.append(
                (
                    tensor,
                    ParallelRank(
                        dp=dp_rank,
                        pp=pp_owner[tensor.tensor_id],
                        ep=ep_rank,
                        tp=tp_rank,
                    ),
                    tuple(offset),
                    tuple(shape),
                )
            )
    return placements


def fragment_payload(
    descriptor: TensorDescriptor, fragment: RuntimeFragment
) -> bytearray:
    global_strides = []
    running = 1
    for extent in reversed(descriptor.global_shape):
        global_strides.append(running)
        running *= extent
    global_strides.reverse()
    payload = bytearray()
    for local_coordinate in product(
        *(range(extent) for extent in fragment.local_shape)
    ):
        global_coordinate = tuple(
            begin + local
            for begin, local in zip(
                fragment.global_offset, local_coordinate, strict=True
            )
        )
        value = 1 + sum(
            coordinate * stride
            for coordinate, stride in zip(
                global_coordinate, global_strides, strict=True
            )
        )
        payload.extend(value.to_bytes(descriptor.itemsize, "little"))
    return payload


def assert_plan_copies_logical_contents(
    plan,
    source_manifests: tuple[RuntimeManifest, ...],
    target_manifests: tuple[RuntimeManifest, ...],
) -> None:
    descriptors = {
        tensor.tensor_id: tensor
        for manifest in source_manifests
        for tensor in manifest.tensors
    }
    source_payloads = {
        fragment.fragment_id: fragment_payload(
            descriptors[fragment.tensor_id], fragment
        )
        for manifest in source_manifests
        for fragment in manifest.fragments
    }
    target_payloads = {
        fragment.fragment_id: bytearray(fragment.nbytes)
        for manifest in target_manifests
        for fragment in manifest.fragments
    }

    for operation in plan.operations:
        source = source_payloads[operation.source.fragment_id]
        target = target_payloads[operation.target.fragment_id]
        for source_offset, target_offset, nbytes in operation.iter_segments():
            target[target_offset : target_offset + nbytes] = source[
                source_offset : source_offset + nbytes
            ]

    for manifest in target_manifests:
        for fragment in manifest.fragments:
            assert target_payloads[fragment.fragment_id] == fragment_payload(
                descriptors[fragment.tensor_id], fragment
            )


def pp_placements(
    tensors: tuple[TensorDescriptor, ...],
    owners: dict[str, int],
) -> list[tuple[TensorDescriptor, ParallelRank, tuple[int, ...], tuple[int, ...]]]:
    return [
        (
            tensor,
            ParallelRank(pp=owners[tensor.tensor_id]),
            (0,),
            tensor.global_shape,
        )
        for tensor in tensors
    ]


@pytest.mark.parametrize(
    ("source_owners", "target_owners", "expected_routes"),
    [
        (
            {f"layers.{layer}.weight": layer // 2 for layer in range(4)},
            {f"layers.{layer}.weight": layer for layer in range(4)},
            {(0, 0), (0, 1), (1, 2), (1, 3)},
        ),
        (
            {f"layers.{layer}.weight": layer for layer in range(4)},
            {f"layers.{layer}.weight": layer // 2 for layer in range(4)},
            {(0, 0), (1, 0), (2, 1), (3, 1)},
        ),
    ],
)
def test_pp_ownership_routes_are_manifest_derived(
    source_owners: dict[str, int],
    target_owners: dict[str, int],
    expected_routes: set[tuple[int, int]],
) -> None:
    tensors = tuple(
        tensor_descriptor(
            f"layers.{layer}.weight",
            global_shape=(8,),
            shard_dims=(),
            layer_id=layer,
        )
        for layer in range(4)
    )
    sources = build_manifests(
        "source",
        pp_placements(tensors, source_owners),
        address_base=0x100000,
    )
    targets = build_manifests(
        "target",
        pp_placements(tensors, target_owners),
        address_base=0x200000,
    )

    plan = plan_runtime_transfer(sources, targets)

    assert {
        (route.source_pp, route.target_pp) for route in plan.pipeline_routes
    } == expected_routes
    assert sorted(
        index for route in plan.pipeline_routes for index in route.operation_indices
    ) == list(range(len(plan.operations)))
    assert_plan_copies_logical_contents(plan, sources, targets)


@pytest.mark.parametrize(("source_ep", "target_ep"), [(8, 2), (2, 8)])
def test_ep_reshard_uses_leading_expert_coordinate(
    source_ep: int, target_ep: int
) -> None:
    source_tensor = tensor_descriptor(
        "layers.0.experts.w1",
        global_shape=(8, 4, 2),
        shard_dims=(0,),
    )
    target_tensor = tensor_descriptor(
        source_tensor.tensor_id,
        global_shape=source_tensor.global_shape,
        shard_dims=(0,),
    )
    sources = build_manifests(
        "source",
        ep_tp_placements(
            (source_tensor,),
            dp=1,
            pp_owner={source_tensor.tensor_id: 0},
            ep=source_ep,
            tp=1,
            tp_dim=1,
        ),
        address_base=0x100000,
    )
    targets = build_manifests(
        "target",
        ep_tp_placements(
            (target_tensor,),
            dp=1,
            pp_owner={target_tensor.tensor_id: 0},
            ep=target_ep,
            tp=1,
            tp_dim=1,
        ),
        address_base=0x200000,
    )

    plan = plan_runtime_transfer(sources, targets)

    assert all(operation.overlap_shape[0] > 0 for operation in plan.operations)
    assert {operation.source.rank.ep for operation in plan.operations} == set(
        range(source_ep)
    )
    assert {operation.target.rank.ep for operation in plan.operations} == set(
        range(target_ep)
    )
    assert_plan_copies_logical_contents(plan, sources, targets)


@pytest.mark.parametrize("target_dim", [1, 2])
def test_ep_tp_cross_dim_reshard(target_dim: int) -> None:
    source_tensor = tensor_descriptor(
        "layers.0.experts.w1",
        global_shape=(4, 6, 8),
        shard_dims=(0,),
    )
    target_tensor = tensor_descriptor(
        source_tensor.tensor_id,
        global_shape=source_tensor.global_shape,
        shard_dims=(target_dim,),
    )
    source_placements = []
    for ep_rank in range(2):
        source_placements.append(
            (
                source_tensor,
                ParallelRank(ep=ep_rank),
                (ep_rank * 2, 0, 0),
                (2, 6, 8),
            )
        )
    target_placements = []
    for tp_rank in range(2):
        shape = list(target_tensor.global_shape)
        offset = [0, 0, 0]
        shape[target_dim] //= 2
        offset[target_dim] = tp_rank * shape[target_dim]
        target_placements.append(
            (
                target_tensor,
                ParallelRank(tp=tp_rank),
                tuple(offset),
                tuple(shape),
            )
        )
    sources = build_manifests("source", source_placements, address_base=0x100000)
    targets = build_manifests("target", target_placements, address_base=0x200000)

    plan = plan_runtime_transfer(sources, targets)

    assert len(plan.operations) == 4
    assert all(isinstance(operation, TransferRegion) for operation in plan.operations)
    selected = next(
        operation
        for operation in plan.operations
        if operation.source.rank.ep == 0 and operation.target.rank.tp == 1
    )
    if target_dim == 1:
        assert selected.overlap_offset == (0, 3, 0)
        assert selected.overlap_shape == (2, 3, 8)
        assert selected.source_base_offset == 48
        assert selected.target_base_offset == 0
        assert selected.inner_bytes == 48
        assert selected.outer_loop_counts == (2,)
        assert selected.source_strides == (96,)
        assert selected.target_strides == (48,)
    else:
        assert selected.overlap_offset == (0, 0, 4)
        assert selected.overlap_shape == (2, 6, 4)
        assert selected.source_base_offset == 8
        assert selected.target_base_offset == 0
        assert selected.inner_bytes == 8
        assert selected.outer_loop_counts == (2, 6)
        assert selected.source_strides == (96, 16)
        assert selected.target_strides == (48, 8)
    assert_plan_copies_logical_contents(plan, sources, targets)


def test_four_axis_reshard_has_complete_content_and_routes() -> None:
    source_tensors = tuple(
        tensor_descriptor(
            f"layers.{layer}.experts.w1",
            global_shape=(8, 8, 2),
            shard_dims=(0, 1),
            layer_id=layer,
        )
        for layer in range(4)
    )
    target_tensors = tuple(
        tensor_descriptor(
            tensor.tensor_id,
            global_shape=tensor.global_shape,
            shard_dims=(0, 1),
            layer_id=tensor.layer_id,
        )
        for tensor in source_tensors
    )
    source_owners = {
        tensor.tensor_id: tensor.layer_id // 2 for tensor in source_tensors
    }
    target_owners = {tensor.tensor_id: tensor.layer_id for tensor in target_tensors}
    sources = build_manifests(
        "source",
        ep_tp_placements(
            source_tensors,
            dp=2,
            pp_owner=source_owners,
            ep=8,
            tp=4,
            tp_dim=1,
        ),
        address_base=0x10000000,
    )
    targets = build_manifests(
        "target",
        ep_tp_placements(
            target_tensors,
            dp=4,
            pp_owner=target_owners,
            ep=2,
            tp=8,
            tp_dim=1,
        ),
        address_base=0x20000000,
    )

    plan = plan_runtime_transfer(sources, targets)

    assert plan.total_bytes == 4 * 8 * 8 * 2 * 2 * 4
    assert {operation.source.rank.dp for operation in plan.operations} == {0, 1}
    assert {operation.target.rank.dp for operation in plan.operations} == {
        0,
        1,
        2,
        3,
    }
    assert {operation.source.rank.tp for operation in plan.operations} == set(range(4))
    assert {operation.target.rank.tp for operation in plan.operations} == set(range(8))
    assert {operation.source.rank.ep for operation in plan.operations} == set(range(8))
    assert {operation.target.rank.ep for operation in plan.operations} == {0, 1}
    assert {(route.source_pp, route.target_pp) for route in plan.pipeline_routes} == {
        (0, 0),
        (0, 1),
        (1, 2),
        (1, 3),
    }
    assert_plan_copies_logical_contents(plan, sources, targets)


def test_cross_dim_planner_keeps_operation_count_at_region_granularity() -> None:
    source_tensor = tensor_descriptor(
        "layers.0.experts.w1",
        global_shape=(8, 8192, 8192),
        shard_dims=(0,),
    )
    target_tensor = tensor_descriptor(
        source_tensor.tensor_id,
        global_shape=source_tensor.global_shape,
        shard_dims=(2,),
    )
    source_placements = [
        (
            source_tensor,
            ParallelRank(ep=rank),
            (rank, 0, 0),
            (1, 8192, 8192),
        )
        for rank in range(8)
    ]
    target_placements = [
        (
            target_tensor,
            ParallelRank(tp=rank),
            (0, 0, rank * 1024),
            (8, 8192, 1024),
        )
        for rank in range(8)
    ]
    sources = build_manifests("source", source_placements, address_base=0x100000000)
    targets = build_manifests("target", target_placements, address_base=0x300000000)

    plan = plan_runtime_transfer(sources, targets)

    assert len(plan.operations) == 64
    assert {operation.segment_count for operation in plan.operations} == {8192}
    assert all(isinstance(operation, TransferRegion) for operation in plan.operations)
