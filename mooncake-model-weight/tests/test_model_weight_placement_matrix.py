from __future__ import annotations

import unittest

from mooncake.model_weight import (
    ParallelRank,
    PlacementFragment,
    RuntimeBindingFragment,
    RuntimeBindingManifest,
    RuntimeFragment,
    RuntimeManifest,
    StoredFragment,
    TargetPlacementManifest,
    TensorDescriptor,
    WeightManifest,
    bind_logical_transfer_plan,
    bind_runtime_manifest,
    placement_manifest_from_runtime_manifest,
    plan_placement_transfer,
    plan_runtime_transfer,
    plan_runtime_transfer_to_target_placements,
    plan_stored_transfer,
    plan_stored_transfer_to_target_placements,
)


MODEL_ID = "matrix-model"
REVISION = "matrix-revision"


def tensor(
    *,
    tensor_id: str = "layers.0.weight",
    global_shape: tuple[int, ...] = (8,),
    shard_dims: tuple[int, ...] = (0,),
    layer_id: int | None = 0,
) -> TensorDescriptor:
    return TensorDescriptor(
        tensor_id=tensor_id,
        global_shape=global_shape,
        dtype="uint8",
        itemsize=1,
        partition_dim=None,
        shard_dims=shard_dims,
        layer_id=layer_id,
        layout_fingerprint="matrix:uint8:v2",
    )


def runtime_manifest(
    *,
    instance_id: str,
    rank: ParallelRank,
    tensors: tuple[TensorDescriptor, ...],
    boxes: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...],
    address_base: int,
) -> RuntimeManifest:
    return RuntimeManifest(
        model_id=MODEL_ID,
        revision=REVISION,
        instance_id=instance_id,
        lease_id=f"lease-{instance_id}",
        tensors=tensors,
        fragments=tuple(
            RuntimeFragment(
                fragment_id=f"runtime-{instance_id}-{index}",
                tensor_id=tensor_id,
                global_offset=offset,
                local_shape=shape,
                address=address_base + index * 0x100,
                nbytes=_volume(shape),
                worker_id=f"worker-{instance_id}",
                endpoint=f"endpoint-{instance_id}",
                device="cuda:0",
                rank=rank,
                lease_generation=5,
            )
            for index, (tensor_id, offset, shape) in enumerate(boxes)
        ),
    )


def placement_manifest(
    *,
    placement_id: str,
    rank: ParallelRank,
    tensors: tuple[TensorDescriptor, ...],
    boxes: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...],
) -> TargetPlacementManifest:
    return TargetPlacementManifest(
        model_id=MODEL_ID,
        revision=REVISION,
        placement_id=None,
        tensors=tensors,
        fragments=tuple(
            PlacementFragment(
                placement_fragment_id=f"placement-{placement_id}-{index}",
                tensor_id=tensor_id,
                global_offset=offset,
                local_shape=shape,
                nbytes=_volume(shape),
                rank=rank,
            )
            for index, (tensor_id, offset, shape) in enumerate(boxes)
        ),
    )


def bind_all(
    placements: tuple[TargetPlacementManifest, ...],
) -> tuple[RuntimeManifest, ...]:
    result = []
    for manifest_index, placement in enumerate(placements):
        binding = RuntimeBindingManifest(
            model_id=MODEL_ID,
            revision=REVISION,
            placement_id=placement.placement_id,
            placement_digest=placement.digest,
            instance_id=f"target-{manifest_index}",
            generation=9,
            lease_id=f"target-lease-{manifest_index}",
            fragments=tuple(
                RuntimeBindingFragment(
                    placement_fragment_id=fragment.placement_fragment_id,
                    fragment_id=f"bound-{manifest_index}-{fragment_index}",
                    address=0x1000000
                    + manifest_index * 0x10000
                    + fragment_index * 0x100,
                    nbytes=fragment.nbytes,
                    worker_id=f"target-worker-{manifest_index}",
                    endpoint=f"target-endpoint-{manifest_index}",
                    device="cuda:0",
                )
                for fragment_index, fragment in enumerate(placement.fragments)
            ),
        )
        result.append(bind_runtime_manifest(placement, binding))
    return tuple(result)


def _volume(shape: tuple[int, ...]) -> int:
    result = 1
    for extent in shape:
        result *= extent
    return result


def assert_runtime_path_equivalent(
    case: unittest.TestCase,
    sources: tuple[RuntimeManifest, ...],
    placements: tuple[TargetPlacementManifest, ...],
) -> None:
    source_placements = tuple(
        placement_manifest_from_runtime_manifest(source) for source in sources
    )
    logical = plan_placement_transfer(source_placements, placements)
    targets = bind_all(placements)

    case.assertTrue(logical.operations)
    case.assertTrue(
        all(
            not hasattr(operation.source, "address")
            and not hasattr(operation.target, "address")
            for operation in logical.operations
        )
    )
    case.assertEqual(
        bind_logical_transfer_plan(logical, targets, source_bindings=sources),
        plan_runtime_transfer(sources, targets),
    )


class PlacementReshardMatrixTest(unittest.TestCase):
    def test_full_planner_rejects_incomplete_placement_coverage(self) -> None:
        descriptor = tensor()
        full_source = placement_manifest(
            placement_id="full-source",
            rank=ParallelRank(),
            tensors=(descriptor,),
            boxes=((descriptor.tensor_id, (0,), (8,)),),
        )
        incomplete_source = placement_manifest(
            placement_id="incomplete-source",
            rank=ParallelRank(),
            tensors=(descriptor,),
            boxes=((descriptor.tensor_id, (0,), (4,)),),
        )
        full_target = placement_manifest(
            placement_id="full-target",
            rank=ParallelRank(),
            tensors=(descriptor,),
            boxes=((descriptor.tensor_id, (0,), (8,)),),
        )
        incomplete_target = placement_manifest(
            placement_id="incomplete-target",
            rank=ParallelRank(),
            tensors=(descriptor,),
            boxes=((descriptor.tensor_id, (0,), (4,)),),
        )

        with self.assertRaisesRegex(ValueError, "no complete DP replica"):
            plan_placement_transfer((incomplete_source,), (full_target,))
        with self.assertRaisesRegex(ValueError, "target tensor is not fully covered"):
            plan_placement_transfer((full_source,), (incomplete_target,))

    def test_tp4_to_tp8_and_tp8_to_tp4(self) -> None:
        for source_tp, target_tp in ((4, 8), (8, 4)):
            descriptor = tensor()
            sources = tuple(
                runtime_manifest(
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
                assert_runtime_path_equivalent(self, sources, placements)

    def test_ep8_to_ep2_and_ep2_to_ep8(self) -> None:
        descriptor = tensor(
            tensor_id="layers.0.experts.weight",
            global_shape=(8, 4),
            shard_dims=(0,),
        )
        for source_ep, target_ep in ((8, 2), (2, 8)):
            sources = tuple(
                runtime_manifest(
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
                assert_runtime_path_equivalent(self, sources, placements)

    def test_pp2_to_pp4_and_pp4_to_pp2_routes_layer_owners(self) -> None:
        descriptors = tuple(
            tensor(
                tensor_id=f"layers.{layer}.weight",
                global_shape=(4,),
                shard_dims=(),
                layer_id=layer,
            )
            for layer in range(4)
        )
        for source_pp, target_pp in ((2, 4), (4, 2)):
            sources = []
            for pp_rank in range(source_pp):
                local = tuple(
                    item for item in descriptors if item.layer_id % source_pp == pp_rank
                )
                sources.append(
                    runtime_manifest(
                        instance_id=f"source-pp{pp_rank}",
                        rank=ParallelRank(pp=pp_rank),
                        tensors=local,
                        boxes=tuple(
                            (item.tensor_id, (0,), item.global_shape) for item in local
                        ),
                        address_base=0x40000 + pp_rank * 0x10000,
                    )
                )
            placements = []
            for pp_rank in range(target_pp):
                local = tuple(
                    item for item in descriptors if item.layer_id % target_pp == pp_rank
                )
                placements.append(
                    placement_manifest(
                        placement_id=f"target-pp{pp_rank}",
                        rank=ParallelRank(pp=pp_rank),
                        tensors=local,
                        boxes=tuple(
                            (item.tensor_id, (0,), item.global_shape) for item in local
                        ),
                    )
                )

            logical = plan_runtime_transfer_to_target_placements(
                tuple(sources), tuple(placements)
            )
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
            assert_runtime_path_equivalent(self, tuple(sources), tuple(placements))

    def test_ep_tp_cross_dim_dim0_to_dim1_and_dim2(self) -> None:
        source_descriptor = tensor(
            tensor_id="layers.0.experts.weight",
            global_shape=(4, 8, 8),
            shard_dims=(0,),
        )
        sources = tuple(
            runtime_manifest(
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
                assert_runtime_path_equivalent(self, sources, tuple(placements))

    def test_combined_dp_tp_pp_ep_reshard(self) -> None:
        descriptors = tuple(
            tensor(
                tensor_id=f"layers.{layer}.experts.weight",
                global_shape=(8, 8),
                shard_dims=(0, 1),
                layer_id=layer,
            )
            for layer in range(4)
        )
        sources = []
        for dp in range(2):
            for pp in range(2):
                local_tensors = tuple(
                    item for item in descriptors if item.layer_id % 2 == pp
                )
                for ep in range(8):
                    for tp in range(4):
                        rank = ParallelRank(dp=dp, tp=tp, pp=pp, ep=ep)
                        sources.append(
                            runtime_manifest(
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

        logical = plan_runtime_transfer_to_target_placements(
            tuple(sources), tuple(placements)
        )
        self.assertEqual(
            {operation.target.rank.dp for operation in logical.operations}, {0, 1, 2, 3}
        )
        self.assertEqual(
            {operation.source.rank.dp for operation in logical.operations}, {0, 1}
        )
        self.assertLessEqual(len(logical.operations), 4096)
        assert_runtime_path_equivalent(self, tuple(sources), tuple(placements))


class PlacementStoredSourceTest(unittest.TestCase):
    def test_store_source_binds_to_the_existing_store_plan(self) -> None:
        descriptor = tensor()
        source = WeightManifest(
            namespace="default",
            model_id=MODEL_ID,
            revision=REVISION,
            group_id="weights/default/model/revision",
            manifest_key="weights/default/model/revision/manifest",
            created_at="2026-07-23T00:00:00Z",
            tensors=(descriptor,),
            fragments=(
                StoredFragment(
                    fragment_id="stored-full",
                    tensor_id=descriptor.tensor_id,
                    global_offset=(0,),
                    local_shape=(8,),
                    object_key="weights/default/model/revision/payload/0",
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
        logical = plan_stored_transfer_to_target_placements(source, placements)
        targets = bind_all(placements)

        self.assertEqual(
            bind_logical_transfer_plan(logical, targets),
            plan_stored_transfer(source, targets),
        )

    def test_store_source_cannot_authorize_target_alias_deduplication(self) -> None:
        aliases = ("alias.a", "alias.b")
        descriptors = tuple(tensor(tensor_id=tensor_id) for tensor_id in aliases)
        source = WeightManifest(
            namespace="default",
            model_id=MODEL_ID,
            revision=REVISION,
            group_id="weights/default/model/revision",
            manifest_key="weights/default/model/revision/manifest",
            created_at="2026-07-23T00:00:00Z",
            tensors=descriptors,
            fragments=tuple(
                StoredFragment(
                    fragment_id=f"stored-{tensor_id}",
                    tensor_id=tensor_id,
                    global_offset=(0,),
                    local_shape=(8,),
                    object_key=(f"weights/default/model/revision/payload/{tensor_id}"),
                    object_offset=0,
                    nbytes=8,
                )
                for tensor_id in aliases
            ),
        )
        target = RuntimeManifest(
            model_id=MODEL_ID,
            revision=REVISION,
            instance_id="target",
            lease_id="target-lease",
            tensors=descriptors,
            fragments=tuple(
                RuntimeFragment(
                    fragment_id=f"target-{tensor_id}",
                    tensor_id=tensor_id,
                    global_offset=(0,),
                    local_shape=(8,),
                    address=0x1000000,
                    nbytes=8,
                    worker_id="target-worker",
                    endpoint="target-endpoint",
                    device="cuda:0",
                    rank=ParallelRank(),
                    lease_generation=9,
                    aliases=aliases,
                )
                for tensor_id in aliases
            ),
        )

        with self.assertRaisesRegex(ValueError, "conflicting target physical range"):
            plan_stored_transfer(source, (target,))


if __name__ == "__main__":
    unittest.main()
