from __future__ import annotations

import unittest
from dataclasses import dataclass, replace
from typing import Iterable

from mooncake.reshard.weight import (
    ParallelRank,
    PlacementFragment,
    StoredFragment,
    TensorDescriptor,
    OwnershipAxis,
    ReplicatedAxis,
    SplitAxis,
    WeightManifest,
    WeightPlacementManifest,
    WeightRuntimeBindingManifest,
    WeightStore,
    bind_logical_transfer_plan,
    plan_placement_transfer,
    plan_stored_transfer_to_target_placement,
)

from global_placement_helpers import (
    global_placement,
    runtime_bindings,
    runtime_fragment,
)


MODEL_ID = "matrix-model"
REVISION = "matrix-revision"


@dataclass(frozen=True)
class RuntimeInputs:
    """Test-only carrier; it is deliberately not a manifest."""

    placement: WeightPlacementManifest
    bindings: tuple[WeightRuntimeBindingManifest, ...]


@dataclass(frozen=True)
class PlacementShard:
    shard_id: str
    rank: ParallelRank
    tensors: tuple[TensorDescriptor, ...]
    fragments: tuple[PlacementFragment, ...]
    address_base: int = 0


def tensor(
    *,
    tensor_id: str = "layers.0.weight",
    global_shape: tuple[int, ...] = (8,),
    shard_dims: tuple[int, ...] = (0,),
    layer_id: int | None = 0,
    parallel_axes: tuple[SplitAxis | ReplicatedAxis | OwnershipAxis, ...] | None = None,
) -> TensorDescriptor:
    if parallel_axes is None:
        if len(shard_dims) > 1:
            raise ValueError("multi-dimensional test tensors require explicit axes")
        parallel_axes = tuple(SplitAxis("tp", dim=dim) for dim in shard_dims)
    return TensorDescriptor(
        tensor_id=tensor_id,
        global_shape=global_shape,
        dtype="uint8",
        itemsize=1,
        shard_dims=shard_dims,
        layer_id=layer_id,
        layout_fingerprint="matrix:uint8:v2",
        parallel_axes=parallel_axes,
    )


def runtime_inputs(
    *,
    instance_id: str,
    rank: ParallelRank,
    tensors: tuple[TensorDescriptor, ...],
    boxes: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...],
    address_base: int,
) -> PlacementShard:
    fragments = tuple(
        PlacementFragment(
            tensor_id=tensor_id,
            global_offset=offset,
            local_shape=shape,
            nbytes=_volume(shape),
            rank=rank,
        )
        for tensor_id, offset, shape in boxes
    )
    return PlacementShard(
        shard_id=instance_id,
        rank=rank,
        tensors=tensors,
        fragments=fragments,
        address_base=address_base,
    )


def combine_runtime_inputs(inputs: Iterable[PlacementShard]) -> RuntimeInputs:
    items = tuple(inputs)
    placement = _combine_placement_shards(items, "source")
    address_by_id = {
        fragment.placement_fragment_id: item.address_base + index * 0x100
        for item in items
        for index, fragment in enumerate(item.fragments)
    }
    return RuntimeInputs(
        placement,
        runtime_bindings(
            placement,
            instance_prefix="source",
            generation=5,
            address_for_fragment=lambda index, fragment: address_by_id[
                fragment.placement_fragment_id
            ],
        ),
    )


def placement_manifest(
    *,
    placement_id: str,
    rank: ParallelRank,
    tensors: tuple[TensorDescriptor, ...],
    boxes: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...],
) -> PlacementShard:
    return PlacementShard(
        shard_id=placement_id,
        rank=rank,
        tensors=tensors,
        fragments=tuple(
            PlacementFragment(
                tensor_id=tensor_id,
                global_offset=offset,
                local_shape=shape,
                nbytes=_volume(shape),
                rank=rank,
            )
            for tensor_id, offset, shape in boxes
        ),
    )


def _combine_placement_shards(
    shards: Iterable[PlacementShard],
    placement_set_id: str,
) -> WeightPlacementManifest:
    items = tuple(shards)
    tensors = {tensor.tensor_id: tensor for item in items for tensor in item.tensors}
    return global_placement(
        resource_id=MODEL_ID,
        revision=REVISION,
        placement_set_id=placement_set_id,
        tensors=tuple(tensors.values()),
        fragments=tuple(fragment for item in items for fragment in item.fragments),
        ranks=tuple(item.rank for item in items),
        weight_generation=5,
    )


def bind_all(
    placement: WeightPlacementManifest,
) -> tuple[WeightRuntimeBindingManifest, ...]:
    return runtime_bindings(
        placement,
        instance_prefix="target",
        generation=9,
        address_for_fragment=lambda index, fragment: 0x1000000 + index * 0x100,
    )


def _volume(shape: tuple[int, ...]) -> int:
    result = 1
    for extent in shape:
        result *= extent
    return result


def assert_placement_path_binds(
    case: unittest.TestCase,
    sources: RuntimeInputs,
    placement_shards: tuple[PlacementShard, ...],
) -> None:
    placement = _combine_placement_shards(placement_shards, "target")
    logical = plan_placement_transfer(sources.placement, placement)
    targets = bind_all(placement)

    case.assertTrue(logical.operations)
    case.assertTrue(
        all(
            not hasattr(operation.source, "address")
            and not hasattr(operation.target, "address")
            for operation in logical.operations
        )
    )
    bound = bind_logical_transfer_plan(
        logical,
        targets,
        source_bindings=sources.bindings,
    )
    case.assertEqual(len(bound.operations), len(logical.operations))
    case.assertTrue(
        all(
            hasattr(operation.source, "address")
            and hasattr(operation.target, "address")
            for operation in bound.operations
        )
    )


class PlacementReshardMatrixTest(unittest.TestCase):
    def test_full_planner_rejects_incomplete_placement_coverage(self) -> None:
        descriptor = tensor()
        full_source = _combine_placement_shards(
            (
                placement_manifest(
                    placement_id="full-source",
                    rank=ParallelRank(),
                    tensors=(descriptor,),
                    boxes=((descriptor.tensor_id, (0,), (8,)),),
                ),
            ),
            "full-source",
        )
        incomplete_source = placement_manifest(
            placement_id="incomplete-source",
            rank=ParallelRank(),
            tensors=(descriptor,),
            boxes=((descriptor.tensor_id, (0,), (4,)),),
        )
        full_target = _combine_placement_shards(
            (
                placement_manifest(
                    placement_id="full-target",
                    rank=ParallelRank(),
                    tensors=(descriptor,),
                    boxes=((descriptor.tensor_id, (0,), (8,)),),
                ),
            ),
            "full-target",
        )
        incomplete_target = placement_manifest(
            placement_id="incomplete-target",
            rank=ParallelRank(),
            tensors=(descriptor,),
            boxes=((descriptor.tensor_id, (0,), (4,)),),
        )

        with self.assertRaisesRegex(ValueError, "not fully covered"):
            _combine_placement_shards((incomplete_source,), "incomplete-source")
        with self.assertRaisesRegex(ValueError, "not fully covered"):
            _combine_placement_shards((incomplete_target,), "incomplete-target")
        self.assertTrue(plan_placement_transfer(full_source, full_target).operations)

    def test_tp4_to_tp8_and_tp8_to_tp4(self) -> None:
        for source_tp, target_tp in ((4, 8), (8, 4)):
            descriptor = tensor()
            sources = combine_runtime_inputs(
                runtime_inputs(
                    instance_id=f"source-tp{rank}",
                    rank=ParallelRank(tp=rank),
                    tensors=(descriptor,),
                    boxes=(
                        (
                            descriptor.tensor_id,
                            (rank * (8 // source_tp),),
                            (8 // source_tp,),
                        ),
                    ),
                    address_base=0x1000 + rank * 0x1000,
                )
                for rank in range(source_tp)
            )
            placements = tuple(
                placement_manifest(
                    placement_id=f"target-tp{rank}",
                    rank=ParallelRank(tp=rank),
                    tensors=(descriptor,),
                    boxes=(
                        (
                            descriptor.tensor_id,
                            (rank * (8 // target_tp),),
                            (8 // target_tp,),
                        ),
                    ),
                )
                for rank in range(target_tp)
            )

            with self.subTest(source_tp=source_tp, target_tp=target_tp):
                assert_placement_path_binds(self, sources, placements)

    def test_ep8_to_ep2_and_ep2_to_ep8(self) -> None:
        descriptor = tensor(
            tensor_id="layers.0.experts.weight",
            global_shape=(8, 4),
            shard_dims=(0,),
            parallel_axes=(SplitAxis("ep", dim=0),),
        )
        for source_ep, target_ep in ((8, 2), (2, 8)):
            sources = combine_runtime_inputs(
                runtime_inputs(
                    instance_id=f"source-ep{rank}",
                    rank=ParallelRank(ep=rank),
                    tensors=(descriptor,),
                    boxes=(
                        (
                            descriptor.tensor_id,
                            (rank * (8 // source_ep), 0),
                            (8 // source_ep, 4),
                        ),
                    ),
                    address_base=0x20000 + rank * 0x1000,
                )
                for rank in range(source_ep)
            )
            placements = tuple(
                placement_manifest(
                    placement_id=f"target-ep{rank}",
                    rank=ParallelRank(ep=rank),
                    tensors=(descriptor,),
                    boxes=(
                        (
                            descriptor.tensor_id,
                            (rank * (8 // target_ep), 0),
                            (8 // target_ep, 4),
                        ),
                    ),
                )
                for rank in range(target_ep)
            )

            with self.subTest(source_ep=source_ep, target_ep=target_ep):
                assert_placement_path_binds(self, sources, placements)

    def test_pp2_to_pp4_and_pp4_to_pp2_routes_layer_owners(self) -> None:
        descriptors = tuple(
            tensor(
                tensor_id=f"layers.{layer}.weight",
                global_shape=(4,),
                shard_dims=(),
                layer_id=layer,
                parallel_axes=(OwnershipAxis("pp"),),
            )
            for layer in range(4)
        )
        for source_pp, target_pp in ((2, 4), (4, 2)):
            sources = combine_runtime_inputs(
                runtime_inputs(
                    instance_id=f"source-pp{pp_rank}",
                    rank=ParallelRank(pp=pp_rank),
                    tensors=tuple(
                        item
                        for item in descriptors
                        if item.layer_id % source_pp == pp_rank
                    ),
                    boxes=tuple(
                        (item.tensor_id, (0,), item.global_shape)
                        for item in descriptors
                        if item.layer_id % source_pp == pp_rank
                    ),
                    address_base=0x40000 + pp_rank * 0x10000,
                )
                for pp_rank in range(source_pp)
            )
            placements = tuple(
                placement_manifest(
                    placement_id=f"target-pp{pp_rank}",
                    rank=ParallelRank(pp=pp_rank),
                    tensors=tuple(
                        item
                        for item in descriptors
                        if item.layer_id % target_pp == pp_rank
                    ),
                    boxes=tuple(
                        (item.tensor_id, (0,), item.global_shape)
                        for item in descriptors
                        if item.layer_id % target_pp == pp_rank
                    ),
                )
                for pp_rank in range(target_pp)
            )

            target = _combine_placement_shards(placements, "target-pp")
            logical = plan_placement_transfer(sources.placement, target)
            expected_routes = {
                (layer % source_pp, layer % target_pp) for layer in range(4)
            }
            self.assertEqual(
                {
                    (route.source_pp, route.target_pp)
                    for route in logical.pipeline_routes
                },
                expected_routes,
            )
            assert_placement_path_binds(self, sources, placements)

    def test_ep_tp_cross_dim_dim0_to_dim1_and_dim2(self) -> None:
        source_descriptor = tensor(
            tensor_id="layers.0.experts.weight",
            global_shape=(4, 8, 8),
            shard_dims=(0,),
            parallel_axes=(SplitAxis("ep", dim=0),),
        )
        sources = combine_runtime_inputs(
            runtime_inputs(
                instance_id=f"source-ep{rank}",
                rank=ParallelRank(ep=rank),
                tensors=(source_descriptor,),
                boxes=((source_descriptor.tensor_id, (rank, 0, 0), (1, 8, 8)),),
                address_base=0x80000 + rank * 0x10000,
            )
            for rank in range(4)
        )
        for target_dim in (1, 2):
            target_descriptor = tensor(
                tensor_id=source_descriptor.tensor_id,
                global_shape=source_descriptor.global_shape,
                shard_dims=(target_dim,),
                parallel_axes=(SplitAxis("tp", dim=target_dim),),
            )
            placements = []
            for rank in range(4):
                offset = [0, 0, 0]
                shape = [4, 8, 8]
                offset[target_dim] = rank * 2
                shape[target_dim] = 2
                placements.append(
                    placement_manifest(
                        placement_id=f"target-dim{target_dim}-tp{rank}",
                        rank=ParallelRank(tp=rank),
                        tensors=(target_descriptor,),
                        boxes=(
                            (target_descriptor.tensor_id, tuple(offset), tuple(shape)),
                        ),
                    )
                )

            with self.subTest(target_dim=target_dim):
                assert_placement_path_binds(self, sources, tuple(placements))

    def test_combined_dp_tp_pp_ep_reshard(self) -> None:
        descriptors = tuple(
            tensor(
                tensor_id=f"layers.{layer}.experts.weight",
                global_shape=(8, 8),
                shard_dims=(0, 1),
                layer_id=layer,
                parallel_axes=(
                    ReplicatedAxis("dp"),
                    OwnershipAxis("pp"),
                    SplitAxis("ep", dim=0),
                    SplitAxis("tp", dim=1),
                ),
            )
            for layer in range(4)
        )
        source_items = []
        for dp in range(2):
            for pp in range(2):
                local_tensors = tuple(
                    item for item in descriptors if item.layer_id % 2 == pp
                )
                for ep in range(8):
                    for tp in range(4):
                        rank = ParallelRank(dp=dp, tp=tp, pp=pp, ep=ep)
                        source_items.append(
                            runtime_inputs(
                                instance_id=f"source-d{dp}-p{pp}-e{ep}-t{tp}",
                                rank=rank,
                                tensors=local_tensors,
                                boxes=tuple(
                                    (item.tensor_id, (ep, tp * 2), (1, 2))
                                    for item in local_tensors
                                ),
                                address_base=(
                                    0x100000
                                    + dp * 0x100000
                                    + pp * 0x80000
                                    + ep * 0x8000
                                    + tp * 0x1000
                                ),
                            )
                        )
        sources = combine_runtime_inputs(source_items)

        placements = []
        for dp in range(4):
            for pp in range(4):
                local_tensors = tuple(
                    item for item in descriptors if item.layer_id % 4 == pp
                )
                for ep in range(2):
                    for tp in range(8):
                        rank = ParallelRank(dp=dp, tp=tp, pp=pp, ep=ep)
                        placements.append(
                            placement_manifest(
                                placement_id=f"target-d{dp}-p{pp}-e{ep}-t{tp}",
                                rank=rank,
                                tensors=local_tensors,
                                boxes=tuple(
                                    (item.tensor_id, (ep * 4, tp), (4, 1))
                                    for item in local_tensors
                                ),
                            )
                        )

        target = _combine_placement_shards(placements, "target-all-axis")
        logical = plan_placement_transfer(sources.placement, target)
        self.assertEqual(
            {operation.target.rank.dp for operation in logical.operations},
            {0, 1, 2, 3},
        )
        self.assertEqual(
            {operation.source.rank.dp for operation in logical.operations},
            {0, 1},
        )
        self.assertLessEqual(len(logical.operations), 4096)
        assert_placement_path_binds(self, sources, tuple(placements))


class PlacementStoredSourceTest(unittest.TestCase):
    def test_store_source_binds_to_the_existing_store_plan(self) -> None:
        descriptor = tensor()
        source = WeightManifest(
            namespace="default",
            resource_id=MODEL_ID,
            revision=REVISION,
            weight_generation=5,
            group_id="weights/default/model/revision/5",
            manifest_key="weights/default/model/revision/5/manifest",
            created_at="2026-07-23T00:00:00Z",
            tensors=(descriptor,),
            fragments=(
                StoredFragment(
                    fragment_id="stored-full",
                    tensor_id=descriptor.tensor_id,
                    global_offset=(0,),
                    local_shape=(8,),
                    object_key="weights/default/model/revision/5/payload/0",
                    object_offset=0,
                    nbytes=8,
                ),
            ),
        )
        placements = tuple(
            placement_manifest(
                placement_id=f"store-target-tp{rank}",
                rank=ParallelRank(tp=rank),
                tensors=(descriptor,),
                boxes=((descriptor.tensor_id, (rank * 2,), (2,)),),
            )
            for rank in range(4)
        )
        target = _combine_placement_shards(placements, "store-target")
        targets = bind_all(target)
        logical = plan_stored_transfer_to_target_placement(source, target)
        bound = bind_logical_transfer_plan(
            logical,
            targets,
            source_manifest=source,
        )

        self.assertEqual(
            bound,
            WeightStore(object()).plan_load(source, target, targets).transfer,
        )

    def test_store_source_rejects_a_forged_payload_fragment_on_bind(self) -> None:
        descriptor = tensor()
        source = WeightManifest(
            namespace="default",
            resource_id=MODEL_ID,
            revision=REVISION,
            weight_generation=5,
            group_id="weights/default/model/revision/5",
            manifest_key="weights/default/model/revision/5/manifest",
            created_at="2026-07-23T00:00:00Z",
            tensors=(descriptor,),
            fragments=(
                StoredFragment(
                    fragment_id="stored-full",
                    tensor_id=descriptor.tensor_id,
                    global_offset=(0,),
                    local_shape=(8,),
                    object_key="weights/default/model/revision/5/payload/0",
                    object_offset=0,
                    nbytes=8,
                ),
            ),
        )
        placements = tuple(
            placement_manifest(
                placement_id=f"store-target-tp{rank}",
                rank=ParallelRank(tp=rank),
                tensors=(descriptor,),
                boxes=((descriptor.tensor_id, (rank * 2,), (2,)),),
            )
            for rank in range(4)
        )
        target = _combine_placement_shards(placements, "store-target")
        logical = plan_stored_transfer_to_target_placement(source, target)

        self.assertEqual(logical.source_manifest, source)
        self.assertEqual(logical.source_manifest_identity, source.manifest_identity)
        foreign_source = WeightManifest(
            namespace="default",
            resource_id=MODEL_ID,
            revision=REVISION,
            weight_generation=5,
            group_id="weights/default/other/revision/5",
            manifest_key="weights/default/other/revision/5/manifest",
            created_at="2026-07-23T00:00:00Z",
            tensors=(descriptor,),
            fragments=(
                StoredFragment(
                    fragment_id="stored-full",
                    tensor_id=descriptor.tensor_id,
                    global_offset=(0,),
                    local_shape=(8,),
                    object_key="weights/default/other/revision/5/payload/0",
                    object_offset=0,
                    nbytes=8,
                ),
            ),
        )
        with self.assertRaisesRegex(
            ValueError,
            "stored source plan requires a source manifest",
        ):
            bind_logical_transfer_plan(logical, bind_all(target))
        with self.assertRaisesRegex(
            ValueError,
            "source manifest identity differs",
        ):
            bind_logical_transfer_plan(
                logical,
                bind_all(target),
                source_manifest=foreign_source,
            )
        forged_source = replace(
            logical.operations[0].source,
            object_key="another-tenant/private-object",
        )
        forged_operation = replace(logical.operations[0], source=forged_source)

        with self.assertRaisesRegex(
            ValueError,
            "source manifest fragment snapshots differ",
        ):
            replace(
                logical,
                operations=(forged_operation, *logical.operations[1:]),
            )
        object.__setattr__(
            logical,
            "operations",
            (forged_operation, *logical.operations[1:]),
        )

        with self.assertRaisesRegex(
            ValueError,
            "source manifest fragment snapshots differ",
        ):
            bind_logical_transfer_plan(
                logical,
                bind_all(target),
                source_manifest=source,
            )

    def test_store_source_cannot_authorize_target_alias_deduplication(self) -> None:
        aliases = ("alias.a", "alias.b")
        descriptors = tuple(tensor(tensor_id=tensor_id) for tensor_id in aliases)
        source = WeightManifest(
            namespace="default",
            resource_id=MODEL_ID,
            revision=REVISION,
            weight_generation=5,
            group_id="weights/default/model/revision/5",
            manifest_key="weights/default/model/revision/5/manifest",
            created_at="2026-07-23T00:00:00Z",
            tensors=descriptors,
            fragments=tuple(
                StoredFragment(
                    fragment_id=f"stored-{tensor_id}",
                    tensor_id=tensor_id,
                    global_offset=(0,),
                    local_shape=(8,),
                    object_key=(
                        f"weights/default/model/revision/5/payload/{tensor_id}"
                    ),
                    object_offset=0,
                    nbytes=8,
                )
                for tensor_id in aliases
            ),
        )
        target_placement = global_placement(
            resource_id=MODEL_ID,
            revision=REVISION,
            weight_generation=5,
            placement_set_id="alias-target",
            tensors=descriptors,
            fragments=tuple(
                PlacementFragment(
                    placement_fragment_id=f"placement-target-{tensor_id}",
                    tensor_id=tensor_id,
                    global_offset=(0,),
                    local_shape=(8,),
                    nbytes=8,
                    rank=ParallelRank(),
                    aliases=aliases,
                )
                for tensor_id in aliases
            ),
        )
        target_tensors = {tensor.tensor_id: tensor for tensor in descriptors}
        target_part = target_placement.parts[0]
        target_binding = WeightRuntimeBindingManifest(
            resource_id=target_placement.resource_id,
            revision=target_placement.revision,
            placement_id=target_placement.placement_id,
            placement_digest=target_placement.digest,
            participant_id=target_part.participant_id,
            instance_id="target",
            generation=9,
            lease_id="target-lease",
            fragments=tuple(
                runtime_fragment(
                    placement=fragment,
                    tensor=target_tensors[fragment.tensor_id],
                    fragment_id=f"target-{fragment.tensor_id}",
                    address=0x1000000,
                    worker_id="target-worker",
                    endpoint="target-endpoint",
                )
                for fragment in target_placement.fragments
            ),
        )
        logical = plan_stored_transfer_to_target_placement(
            source,
            target_placement,
        )

        with self.assertRaisesRegex(ValueError, "conflicting target physical range"):
            bind_logical_transfer_plan(
                logical,
                (target_binding,),
                source_manifest=source,
            )


if __name__ == "__main__":
    unittest.main()
