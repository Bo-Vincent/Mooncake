from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from types import SimpleNamespace

from mooncake.model_weight import (
    ParallelRank,
    PlacementManifest,
    PlacementFragment,
    RuntimeBindingFragment,
    RuntimeBindingManifest,
    RuntimeFragment,
    RuntimeManifest,
    TargetPlacementManifest,
    TensorDescriptor,
    bind_logical_transfer_plan,
    bind_runtime_manifest,
    plan_placement_transfer_to_local_target,
    plan_runtime_transfer_to_local_target,
    plan_runtime_transfer_to_local_target_placement,
)


MODEL_ID = "model"
REVISION = "revision"


def descriptor(
    *,
    tensor_id: str = "layers.0.weight",
    global_shape: tuple[int, ...] = (8,),
    shard_dims: tuple[int, ...] = (0,),
    layer_id: int | None = 0,
    expert_id: int | None = None,
) -> TensorDescriptor:
    return TensorDescriptor(
        tensor_id=tensor_id,
        global_shape=global_shape,
        dtype="uint8",
        itemsize=1,
        partition_dim=None,
        shard_dims=shard_dims,
        layer_id=layer_id,
        expert_id=expert_id,
        layout_fingerprint="test:logical-box:v2",
    )


def source_manifest() -> RuntimeManifest:
    tensor = descriptor()
    return RuntimeManifest(
        model_id=MODEL_ID,
        revision=REVISION,
        instance_id="source-instance",
        lease_id="source-lease",
        tensors=(tensor,),
        fragments=(
            RuntimeFragment(
                fragment_id="source-fragment",
                tensor_id=tensor.tensor_id,
                global_offset=(0,),
                local_shape=(8,),
                address=0x1000,
                nbytes=8,
                worker_id="source-worker",
                endpoint="source-endpoint",
                device="cuda:0",
                rank=ParallelRank(tp=0),
                lease_generation=3,
            ),
        ),
    )


def source_placement(
    *,
    fragment_id: str = "source-placement-fragment",
    dp: int = 0,
) -> PlacementManifest:
    tensor = descriptor()
    return PlacementManifest(
        model_id=MODEL_ID,
        revision=REVISION,
        placement_id=None,
        tensors=(tensor,),
        fragments=(
            PlacementFragment(
                placement_fragment_id=fragment_id,
                tensor_id=tensor.tensor_id,
                global_offset=(0,),
                local_shape=(8,),
                nbytes=8,
                rank=ParallelRank(dp=dp, tp=0),
            ),
        ),
    )


def source_binding(
    *,
    placement: PlacementManifest | None = None,
    placement_id: str | None = None,
    placement_fragment_id: str = "source-placement-fragment",
    dp: int = 0,
    instance_id: str = "source-instance",
    generation: int = 3,
    lease_id: str = "source-lease",
    address: int = 0x1000,
    nbytes: int = 8,
    worker_id: str = "source-worker",
    endpoint: str = "source-endpoint",
) -> RuntimeBindingManifest:
    placement = placement or source_placement(
        fragment_id=placement_fragment_id,
        dp=dp,
    )
    return RuntimeBindingManifest(
        model_id=MODEL_ID,
        revision=REVISION,
        placement_id=placement_id or placement.placement_id,
        placement_digest=placement.digest,
        instance_id=instance_id,
        generation=generation,
        lease_id=lease_id,
        fragments=(
            RuntimeBindingFragment(
                placement_fragment_id=placement_fragment_id,
                fragment_id=f"{instance_id}-fragment",
                address=address,
                nbytes=nbytes,
                worker_id=worker_id,
                endpoint=endpoint,
                device="cuda:0",
            ),
        ),
    )


def target_placement(
    *,
    fragment_id: str = "placement-fragment",
) -> TargetPlacementManifest:
    tensor = descriptor()
    return TargetPlacementManifest(
        model_id=MODEL_ID,
        revision=REVISION,
        placement_id=None,
        tensors=(tensor,),
        fragments=(
            PlacementFragment(
                placement_fragment_id=fragment_id,
                tensor_id=tensor.tensor_id,
                global_offset=(4,),
                local_shape=(4,),
                nbytes=4,
                rank=ParallelRank(tp=1),
                aliases=("model.layers.0.weight",),
            ),
        ),
    )


def target_binding(
    *,
    placement: PlacementManifest | None = None,
    placement_id: str | None = None,
    placement_fragment_id: str = "placement-fragment",
    nbytes: int = 4,
) -> RuntimeBindingManifest:
    placement = placement or target_placement(fragment_id=placement_fragment_id)
    return RuntimeBindingManifest(
        model_id=MODEL_ID,
        revision=REVISION,
        placement_id=placement_id or placement.placement_id,
        placement_digest=placement.digest,
        instance_id="target-instance",
        generation=7,
        lease_id="target-lease",
        fragments=(
            RuntimeBindingFragment(
                placement_fragment_id=placement_fragment_id,
                fragment_id="target-runtime-fragment",
                address=0x9000,
                nbytes=nbytes,
                worker_id="target-worker",
                endpoint="target-endpoint",
                device="cuda:0",
            ),
        ),
    )


def split_target_placement() -> TargetPlacementManifest:
    tensor = descriptor()
    return TargetPlacementManifest(
        model_id=MODEL_ID,
        revision=REVISION,
        placement_id=None,
        tensors=(tensor,),
        fragments=(
            PlacementFragment(
                placement_fragment_id="placement-left",
                tensor_id=tensor.tensor_id,
                global_offset=(0,),
                local_shape=(4,),
                nbytes=4,
                rank=ParallelRank(tp=0),
            ),
            PlacementFragment(
                placement_fragment_id="placement-right",
                tensor_id=tensor.tensor_id,
                global_offset=(4,),
                local_shape=(4,),
                nbytes=4,
                rank=ParallelRank(tp=1),
            ),
        ),
    )


def split_target_binding(
    *,
    right_address: int,
    right_worker_id: str = "target-worker",
    right_endpoint: str = "target-endpoint",
) -> RuntimeBindingManifest:
    placement = split_target_placement()
    return RuntimeBindingManifest(
        model_id=MODEL_ID,
        revision=REVISION,
        placement_id=placement.placement_id,
        placement_digest=placement.digest,
        instance_id="target-instance",
        generation=7,
        lease_id="target-lease",
        fragments=(
            RuntimeBindingFragment(
                placement_fragment_id="placement-left",
                fragment_id="runtime-left",
                address=0x9000,
                nbytes=4,
                worker_id="target-worker",
                endpoint="target-endpoint",
                device="cuda:0",
            ),
            RuntimeBindingFragment(
                placement_fragment_id="placement-right",
                fragment_id="runtime-right",
                address=right_address,
                nbytes=4,
                worker_id=right_worker_id,
                endpoint=right_endpoint,
                device="cuda:0",
            ),
        ),
    )


class TargetPlacementManifestTest(unittest.TestCase):
    def test_json_round_trip_is_stable_and_contains_no_runtime_location(self) -> None:
        placement = target_placement()

        encoded = placement.to_json()
        payload = json.loads(encoded)

        self.assertEqual(TargetPlacementManifest.from_json(encoded), placement)
        self.assertEqual(
            TargetPlacementManifest.from_json(encoded).digest,
            placement.digest,
        )
        self.assertNotIn("address", encoded)
        self.assertNotIn("endpoint", encoded)
        self.assertNotIn("worker_id", encoded)
        self.assertNotIn("instance_id", encoded)
        self.assertNotIn("generation", encoded)
        self.assertNotIn("lease", encoded)
        self.assertNotIn("owner", encoded)
        self.assertNotIn("fragment_leases", encoded)
        self.assertEqual(payload["placement_id"], placement.placement_id)

    def test_json_round_trip_accepts_partition_dim_descriptor(self) -> None:
        tensor = TensorDescriptor(
            tensor_id="layers.0.single_axis.weight",
            global_shape=(8,),
            dtype="uint8",
            itemsize=1,
            partition_dim=0,
            layout_fingerprint="test:partition-dim",
        )
        placement = TargetPlacementManifest(
            model_id=MODEL_ID,
            revision=REVISION,
            placement_id=None,
            tensors=(tensor,),
            fragments=(
                PlacementFragment(
                    placement_fragment_id="single-axis-placement-fragment",
                    tensor_id=tensor.tensor_id,
                    global_offset=(0,),
                    local_shape=(4,),
                    nbytes=4,
                    rank=ParallelRank(tp=0),
                ),
            ),
        )

        self.assertEqual(
            TargetPlacementManifest.from_json(placement.to_json()), placement
        )

    def test_runtime_inventory_accepts_partition_dim_descriptor(self) -> None:
        inventory = SimpleNamespace(
            model_id=MODEL_ID,
            revision=REVISION,
            placement_id=None,
            tensors=(
                SimpleNamespace(
                    placement_fragment_id="single-axis-runtime-fragment",
                    tensor_id="layers.0.single_axis.weight",
                    runtime_name="model.layers.0.single_axis.weight",
                    aliases=("model.layers.0.single_axis.weight",),
                    global_shape=(8,),
                    global_offset=(0,),
                    local_shape=(4,),
                    dtype="uint8",
                    itemsize=1,
                    partition_dim=0,
                    shard_dims=None,
                    layer_id=0,
                    expert_id=None,
                    layout_fingerprint="test:partition-dim",
                    nbytes=4,
                    rank=SimpleNamespace(dp=0, tp=0, pp=0, ep=0),
                ),
            ),
        )

        placement = TargetPlacementManifest.from_runtime_inventory(inventory)

        self.assertEqual(placement.tensors[0].shard_dims, (0,))
        self.assertEqual(placement.tensors[0].effective_shard_dims, (0,))

    def test_binding_builds_runtime_manifest_without_losing_fencing(
        self,
    ) -> None:
        placement = target_placement()

        runtime = bind_runtime_manifest(placement, target_binding())

        self.assertEqual(runtime.model_id, MODEL_ID)
        self.assertEqual(runtime.revision, REVISION)
        self.assertEqual(runtime.instance_id, "target-instance")
        self.assertEqual(runtime.lease_id, "target-lease")
        self.assertEqual(runtime.placement_id, placement.placement_id)
        self.assertEqual(runtime.fragments[0].address, 0x9000)
        self.assertEqual(runtime.fragments[0].lease_generation, 7)
        self.assertEqual(
            runtime.fragments[0].placement_fragment_id,
            "placement-fragment",
        )
        self.assertEqual(runtime.fragments[0].rank, ParallelRank(tp=1))
        self.assertEqual(runtime.fragments[0].global_offset, (4,))
        self.assertEqual(
            runtime.fragments[0].aliases,
            ("model.layers.0.weight",),
        )
        self.assertEqual(len(runtime.fragments), 1)

    def test_binding_rejects_overlapping_independent_runtime_ranges(self) -> None:
        placement = split_target_placement()

        for right_address in (0x9000, 0x9002):
            with self.subTest(right_address=right_address):
                with self.assertRaisesRegex(
                    ValueError, "runtime (binding|manifest) address ranges overlap"
                ):
                    bind_runtime_manifest(
                        placement,
                        split_target_binding(right_address=right_address),
                    )

    def test_binding_allows_exact_runtime_alias_ranges_declared_by_placement(
        self,
    ) -> None:
        aliases = ("lm_head.weight", "model.embed_tokens.weight")
        tensors = (
            TensorDescriptor(
                tensor_id=tensor_id,
                global_shape=(4,),
                dtype="uint8",
                itemsize=1,
                layout_fingerprint="test:contiguous:v1",
                partition_dim=None,
            )
            for tensor_id in aliases
        )
        placement = TargetPlacementManifest(
            model_id=MODEL_ID,
            revision=REVISION,
            placement_id=None,
            tensors=tuple(tensors),
            fragments=tuple(
                PlacementFragment(
                    placement_fragment_id=f"placement-{tensor_id}",
                    tensor_id=tensor_id,
                    global_offset=(0,),
                    local_shape=(4,),
                    nbytes=4,
                    rank=ParallelRank(),
                    aliases=aliases,
                )
                for tensor_id in aliases
            ),
        )
        binding = RuntimeBindingManifest(
            model_id=MODEL_ID,
            revision=REVISION,
            placement_id=placement.placement_id,
            placement_digest=placement.digest,
            instance_id="target-instance",
            generation=7,
            lease_id="target-lease",
            fragments=tuple(
                RuntimeBindingFragment(
                    placement_fragment_id=fragment.placement_fragment_id,
                    fragment_id=f"runtime-{fragment.tensor_id}",
                    address=0x9000,
                    nbytes=4,
                    worker_id="target-worker",
                    endpoint="target-endpoint",
                    device="cuda:0",
                )
                for fragment in placement.fragments
            ),
        )

        runtime = bind_runtime_manifest(placement, binding)

        self.assertEqual(
            tuple(fragment.address for fragment in runtime.fragments),
            (0x9000, 0x9000),
        )
        self.assertTrue(
            all(fragment.aliases == aliases for fragment in runtime.fragments)
        )

    def test_binding_rejects_partial_runtime_alias_overlap(self) -> None:
        aliases = ("alias.a", "alias.b")
        placement = TargetPlacementManifest(
            model_id=MODEL_ID,
            revision=REVISION,
            placement_id=None,
            tensors=tuple(
                TensorDescriptor(
                    tensor_id=tensor_id,
                    global_shape=(4,),
                    dtype="uint8",
                    itemsize=1,
                    layout_fingerprint="test:contiguous:v1",
                    partition_dim=None,
                )
                for tensor_id in aliases
            ),
            fragments=tuple(
                PlacementFragment(
                    placement_fragment_id=f"placement-{tensor_id}",
                    tensor_id=tensor_id,
                    global_offset=(0,),
                    local_shape=(4,),
                    nbytes=4,
                    rank=ParallelRank(),
                    aliases=aliases,
                )
                for tensor_id in aliases
            ),
        )
        binding = RuntimeBindingManifest(
            model_id=MODEL_ID,
            revision=REVISION,
            placement_id=placement.placement_id,
            placement_digest=placement.digest,
            instance_id="target-instance",
            generation=7,
            lease_id="target-lease",
            fragments=(
                RuntimeBindingFragment(
                    placement_fragment_id="placement-alias.a",
                    fragment_id="runtime-alias.a",
                    address=0x9000,
                    nbytes=4,
                    worker_id="target-worker",
                    endpoint="target-endpoint",
                    device="cuda:0",
                ),
                RuntimeBindingFragment(
                    placement_fragment_id="placement-alias.b",
                    fragment_id="runtime-alias.b",
                    address=0x9002,
                    nbytes=4,
                    worker_id="target-worker",
                    endpoint="target-endpoint",
                    device="cuda:0",
                ),
            ),
        )

        with self.assertRaisesRegex(
            ValueError, "runtime (binding|manifest) address ranges overlap"
        ):
            bind_runtime_manifest(placement, binding)

    def test_binding_allows_adjacent_runtime_ranges(self) -> None:
        runtime = bind_runtime_manifest(
            split_target_placement(),
            split_target_binding(right_address=0x9004),
        )

        self.assertEqual(
            tuple(fragment.address for fragment in runtime.fragments),
            (0x9000, 0x9004),
        )

    def test_binding_allows_same_address_on_different_workers(self) -> None:
        runtime = bind_runtime_manifest(
            split_target_placement(),
            split_target_binding(
                right_address=0x9000,
                right_worker_id="target-worker-1",
            ),
        )

        self.assertEqual(
            tuple(fragment.address for fragment in runtime.fragments),
            (0x9000, 0x9000),
        )

    def test_binding_rejects_wrong_identity_and_fragment_sets(self) -> None:
        placement = target_placement()

        with self.assertRaisesRegex(ValueError, "placement_id"):
            bind_runtime_manifest(
                placement,
                target_binding(placement_id="different-placement"),
            )
        with self.assertRaisesRegex(ValueError, "unknown placement fragment"):
            bind_runtime_manifest(
                placement,
                target_binding(
                    placement=placement,
                    placement_fragment_id="unknown-fragment",
                ),
            )
        with self.assertRaisesRegex(ValueError, "byte size"):
            bind_runtime_manifest(placement, target_binding(nbytes=8))

    def test_binding_rejects_missing_or_duplicate_fragments(self) -> None:
        placement = target_placement()
        empty = RuntimeBindingManifest(
            model_id=MODEL_ID,
            revision=REVISION,
            placement_id=placement.placement_id,
            placement_digest=placement.digest,
            instance_id="target-instance",
            generation=7,
            lease_id="target-lease",
            fragments=(),
        )

        with self.assertRaisesRegex(ValueError, "missing placement fragment"):
            bind_runtime_manifest(placement, empty)

        fragment = target_binding().fragments[0]
        with self.assertRaisesRegex(ValueError, "duplicate placement fragment"):
            RuntimeBindingManifest(
                model_id=MODEL_ID,
                revision=REVISION,
                placement_id=placement.placement_id,
                placement_digest=placement.digest,
                instance_id="target-instance",
                generation=7,
                lease_id="target-lease",
                fragments=(fragment, fragment),
            )

    def test_binding_allows_one_rank_spanning_multiple_runtime_locations(self) -> None:
        tensor = descriptor()
        placement = TargetPlacementManifest(
            model_id=MODEL_ID,
            revision=REVISION,
            placement_id=None,
            tensors=(tensor,),
            fragments=(
                PlacementFragment(
                    placement_fragment_id="placement-left",
                    tensor_id=tensor.tensor_id,
                    global_offset=(0,),
                    local_shape=(4,),
                    nbytes=4,
                    rank=ParallelRank(tp=0),
                ),
                PlacementFragment(
                    placement_fragment_id="placement-right",
                    tensor_id=tensor.tensor_id,
                    global_offset=(4,),
                    local_shape=(4,),
                    nbytes=4,
                    rank=ParallelRank(tp=0),
                ),
            ),
        )
        binding = RuntimeBindingManifest(
            model_id=MODEL_ID,
            revision=REVISION,
            placement_id=placement.placement_id,
            placement_digest=placement.digest,
            instance_id="target-instance",
            generation=7,
            lease_id="target-lease",
            fragments=(
                RuntimeBindingFragment(
                    placement_fragment_id="placement-left",
                    fragment_id="runtime-left",
                    address=0x9000,
                    nbytes=4,
                    worker_id="target-worker-0",
                    endpoint="target-endpoint-0",
                    device="cuda:0",
                ),
                RuntimeBindingFragment(
                    placement_fragment_id="placement-right",
                    fragment_id="runtime-right",
                    address=0xA000,
                    nbytes=4,
                    worker_id="target-worker-1",
                    endpoint="target-endpoint-1",
                    device="cuda:0",
                ),
            ),
        )

        runtime = bind_runtime_manifest(placement, binding)

        self.assertEqual(
            tuple(fragment.worker_id for fragment in runtime.fragments),
            ("target-worker-0", "target-worker-1"),
        )

    def test_imports_split_sglang_inventory_without_framework_dependency(self) -> None:
        rank = SimpleNamespace(dp=0, tp=1, pp=2, ep=3)
        placement_inventory = SimpleNamespace(
            model_id=MODEL_ID,
            revision=REVISION,
            placement_id=None,
            tensors=(
                SimpleNamespace(
                    placement_fragment_id="sglang-placement-fragment",
                    tensor_id="layers.0.weight",
                    runtime_name="model.layers.0.weight",
                    aliases=("model.layers.0.weight",),
                    global_shape=(8,),
                    global_offset=(4,),
                    local_shape=(4,),
                    dtype="uint8",
                    itemsize=1,
                    partition_dim=None,
                    shard_dims=(0,),
                    layer_id=0,
                    expert_id=None,
                    layout_fingerprint="test:logical-box:v2",
                    nbytes=4,
                    byte_offset=0,
                    rank=rank,
                ),
            ),
        )
        placement = TargetPlacementManifest.from_runtime_inventory(placement_inventory)
        binding_inventory = SimpleNamespace(
            model_id=MODEL_ID,
            revision=REVISION,
            placement_id=placement.placement_id,
            placement_digest=placement.digest,
            instance_id="sglang-target",
            generation=11,
            lease_id="sglang-lease",
            fragments=(
                SimpleNamespace(
                    placement_fragment_id="sglang-placement-fragment",
                    fragment_id="sglang-runtime-fragment",
                    address=0xA000,
                    nbytes=4,
                    worker_id="sglang-worker",
                    endpoint="sglang-endpoint",
                    device="cuda:0",
                    is_contiguous=True,
                ),
            ),
        )

        binding = RuntimeBindingManifest.from_runtime_inventory(binding_inventory)
        runtime = bind_runtime_manifest(placement, binding)

        self.assertEqual(runtime.fragments[0].address, 0xA000)
        self.assertEqual(runtime.fragments[0].rank, ParallelRank(tp=1, pp=2, ep=3))
        self.assertEqual(runtime.fragments[0].lease_generation, 11)


class PlacementPlannerTest(unittest.TestCase):
    def test_plans_from_source_and_target_placements_then_binds_both_sides(
        self,
    ) -> None:
        source = source_placement()
        target = target_placement()

        logical = plan_placement_transfer_to_local_target((source,), target)

        encoded = json.dumps(asdict(logical), sort_keys=True)
        self.assertEqual(logical.source_placement_ids, (source.placement_id,))
        self.assertEqual(logical.target_placement_ids, (target.placement_id,))
        self.assertIsInstance(logical.operations[0].source, PlacementFragment)
        self.assertIsInstance(logical.operations[0].target, PlacementFragment)
        self.assertNotIn("address", encoded)
        self.assertNotIn("endpoint", encoded)
        self.assertNotIn("worker_id", encoded)
        self.assertNotIn("instance_id", encoded)
        self.assertNotIn("generation", encoded)
        self.assertNotIn("lease", encoded)
        self.assertNotIn("owner", encoded)
        self.assertNotIn("fragment_leases", encoded)
        self.assertFalse(hasattr(logical.source_executors[0], "fragment_leases"))

        bound = bind_logical_transfer_plan(
            logical,
            (target_binding(),),
            source_bindings=(source_binding(),),
        )
        direct = plan_runtime_transfer_to_local_target(
            (bind_runtime_manifest(source, source_binding()),),
            bind_runtime_manifest(target, target_binding()),
        )

        self.assertEqual(bound, direct)
        self.assertEqual(bound.operations[0].source.address, 0x1000)
        self.assertEqual(bound.operations[0].target.address, 0x9000)
        self.assertEqual(
            bound.source_executors[0].fragment_leases[0].lease_generation,
            3,
        )

    def test_source_binding_is_required_and_validated_fail_closed(self) -> None:
        logical = plan_placement_transfer_to_local_target(
            (source_placement(),), target_placement()
        )

        with self.assertRaisesRegex(ValueError, "requires source runtime bindings"):
            bind_logical_transfer_plan(logical, (target_binding(),))
        with self.assertRaisesRegex(ValueError, "source placement IDs differ"):
            bind_logical_transfer_plan(
                logical,
                (target_binding(),),
                source_bindings=(target_binding(),),
            )
        with self.assertRaisesRegex(ValueError, "byte size"):
            bind_logical_transfer_plan(
                logical,
                (target_binding(),),
                source_bindings=(source_binding(nbytes=4),),
            )

    def test_source_runtime_manifest_cannot_change_logical_ownership(self) -> None:
        source = source_placement()
        logical = plan_placement_transfer_to_local_target((source,), target_placement())
        runtime = bind_runtime_manifest(source, source_binding())
        fragment = runtime.fragments[0]
        forged_source = RuntimeManifest(
            model_id=runtime.model_id,
            revision=runtime.revision,
            instance_id=runtime.instance_id,
            lease_id=runtime.lease_id,
            placement_id=runtime.placement_id,
            tensors=runtime.tensors,
            fragments=(
                RuntimeFragment(
                    fragment_id=fragment.fragment_id,
                    tensor_id=fragment.tensor_id,
                    global_offset=fragment.global_offset,
                    local_shape=fragment.local_shape,
                    address=fragment.address,
                    nbytes=fragment.nbytes,
                    worker_id=fragment.worker_id,
                    endpoint=fragment.endpoint,
                    device="cuda:0",
                    rank=ParallelRank(tp=1),
                    lease_generation=fragment.lease_generation,
                    aliases=fragment.aliases,
                    placement_fragment_id=fragment.placement_fragment_id,
                ),
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "does not match source placement|placement_id does not match",
        ):
            bind_logical_transfer_plan(
                logical,
                (target_binding(),),
                source_bindings=(forged_source,),
            )

    def test_source_binding_rejects_cross_dp_generation_mismatch(self) -> None:
        source_dp0 = source_placement(
            fragment_id="source-dp0-fragment",
            dp=0,
        )
        source_dp1 = source_placement(
            fragment_id="source-dp1-fragment",
            dp=1,
        )
        logical = plan_placement_transfer_to_local_target(
            (source_dp0, source_dp1), target_placement()
        )

        with self.assertRaisesRegex(ValueError, "inconsistent lease generations"):
            bind_logical_transfer_plan(
                logical,
                (target_binding(),),
                source_bindings=(
                    source_binding(
                        placement=source_dp0,
                        placement_fragment_id="source-dp0-fragment",
                        instance_id="source-dp0-instance",
                        generation=3,
                        address=0x1000,
                    ),
                    source_binding(
                        placement=source_dp1,
                        placement_fragment_id="source-dp1-fragment",
                        instance_id="source-dp1-instance",
                        generation=4,
                        address=0x2000,
                    ),
                ),
            )

    def test_source_address_change_rebinds_without_replanning(self) -> None:
        logical = plan_placement_transfer_to_local_target(
            (source_placement(),), target_placement()
        )

        first = bind_logical_transfer_plan(
            logical,
            (target_binding(),),
            source_bindings=(source_binding(address=0x1000),),
        )
        second = bind_logical_transfer_plan(
            logical,
            (target_binding(),),
            source_bindings=(
                source_binding(
                    instance_id="source-instance-2",
                    generation=4,
                    lease_id="source-lease-2",
                    address=0x2000,
                    worker_id="source-worker-2",
                    endpoint="source-endpoint-2",
                ),
            ),
        )

        self.assertEqual(first.operations[0].source.address, 0x1000)
        self.assertEqual(second.operations[0].source.address, 0x2000)
        self.assertEqual(second.source_executors[0].instance_id, "source-instance-2")
        self.assertEqual(second.source_executors[0].worker_id, "source-worker-2")
        self.assertEqual(
            second.source_executors[0].fragment_leases[0].lease_generation,
            4,
        )
        self.assertEqual(
            first.operations[0].overlap_offset,
            second.operations[0].overlap_offset,
        )
        self.assertEqual(
            first.operations[0].overlap_shape,
            second.operations[0].overlap_shape,
        )

    def test_runtime_source_logical_wrapper_contains_no_runtime_location(self) -> None:
        logical = plan_runtime_transfer_to_local_target_placement(
            (source_manifest(),), target_placement()
        )

        encoded = json.dumps(asdict(logical), sort_keys=True)
        self.assertIsInstance(logical.operations[0].source, PlacementFragment)
        self.assertNotIn("address", encoded)
        self.assertNotIn("endpoint", encoded)
        self.assertNotIn("lease", encoded)

    def test_plans_before_target_address_exists_then_binds_to_direct_plan(self) -> None:
        source = source_manifest()
        placement = target_placement()

        logical = plan_runtime_transfer_to_local_target_placement((source,), placement)

        self.assertEqual(logical.target_placement_ids, (placement.placement_id,))
        self.assertEqual(len(logical.operations), 1)
        self.assertFalse(hasattr(logical.operations[0].target, "address"))
        self.assertEqual(logical.operations[0].overlap_offset, (4,))
        self.assertEqual(logical.operations[0].overlap_shape, (4,))

        target = bind_runtime_manifest(placement, target_binding())
        bound = bind_logical_transfer_plan(
            logical, (target,), source_bindings=(source,)
        )
        direct = plan_runtime_transfer_to_local_target((source,), target)

        self.assertEqual(bound, direct)
        self.assertEqual(bound.operations[0].target.address, 0x9000)
        self.assertEqual(bound.target_executors[0].worker_id, "target-worker")

    def test_logical_plan_rejects_binding_from_another_placement(self) -> None:
        logical = plan_runtime_transfer_to_local_target_placement(
            (source_manifest(),), target_placement()
        )
        other_placement = target_placement(
            fragment_id="other-fragment",
        )
        other_target = bind_runtime_manifest(
            other_placement,
            target_binding(
                placement=other_placement,
                placement_fragment_id="other-fragment",
            ),
        )

        with self.assertRaisesRegex(ValueError, "placement"):
            bind_logical_transfer_plan(
                logical,
                (other_target,),
                source_bindings=(source_manifest(),),
            )

    def test_logical_plan_rejects_runtime_fragment_that_changes_ownership(self) -> None:
        placement = target_placement()
        logical = plan_runtime_transfer_to_local_target_placement(
            (source_manifest(),), placement
        )
        target = bind_runtime_manifest(placement, target_binding())
        fragment = target.fragments[0]
        forged_target = RuntimeManifest(
            model_id=target.model_id,
            revision=target.revision,
            instance_id=target.instance_id,
            lease_id=target.lease_id,
            placement_id=target.placement_id,
            tensors=target.tensors,
            fragments=(
                RuntimeFragment(
                    fragment_id=fragment.fragment_id,
                    tensor_id=fragment.tensor_id,
                    global_offset=fragment.global_offset,
                    local_shape=fragment.local_shape,
                    address=fragment.address,
                    nbytes=fragment.nbytes,
                    worker_id=fragment.worker_id,
                    endpoint=fragment.endpoint,
                    device="cuda:0",
                    rank=ParallelRank(tp=0),
                    lease_generation=fragment.lease_generation,
                    aliases=fragment.aliases,
                    placement_fragment_id=fragment.placement_fragment_id,
                ),
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "does not match target placement|placement_id does not match",
        ):
            bind_logical_transfer_plan(
                logical,
                (forged_target,),
                source_bindings=(source_manifest(),),
            )


if __name__ == "__main__":
    unittest.main()
