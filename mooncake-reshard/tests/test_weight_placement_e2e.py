from __future__ import annotations

import ctypes
import unittest

from mooncake.reshard._compat import _strict_zip
from mooncake.reshard.weight import (
    MemoryRegistrationLease,
    MooncakeTransferEngineReader,
    ParallelRank,
    PlacementFragment,
    StoredFragment,
    TensorDescriptor,
    OwnershipAxis,
    ReplicatedAxis,
    SplitAxis,
    WeightLoadPlan,
    WeightManifest,
    WeightPlacementManifest,
    WeightRuntimeBindingManifest,
    WeightStore,
    bind_logical_transfer_plan,
    plan_placement_transfer,
    plan_placement_transfer_to_local_target,
    plan_stored_transfer_to_target_placement,
)

from global_placement_helpers import global_placement, runtime_fragment


class Buffer:
    def __init__(self, size: int) -> None:
        self.size = size
        self.storage = ctypes.create_string_buffer(size)

    @property
    def address(self) -> int:
        return ctypes.addressof(self.storage)

    def write(self, value: bytes) -> None:
        if len(value) != self.size:
            raise ValueError("buffer size mismatch")
        ctypes.memmove(self.address, value, self.size)

    def read(self) -> bytes:
        return ctypes.string_at(self.address, self.size)


class ReadOnlyTransferEngine:
    def __init__(self) -> None:
        self.read_calls = 0

    def register_memory(self, address: int, nbytes: int) -> int:
        del address, nbytes
        return 0

    def unregister_memory(self, address: int) -> int:
        del address
        return 0

    def batch_transfer_sync_read(
        self, endpoint, target_addresses, source_addresses, sizes
    ) -> int:
        del endpoint
        self.read_calls += 1
        for target, source, size in _strict_zip(
            target_addresses, source_addresses, sizes
        ):
            ctypes.memmove(target, source, size)
        return 0


class RangeStore:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.get_into_ranges_calls = 0

    def register_buffer(self, address: int, nbytes: int) -> int:
        del address, nbytes
        return 0

    def unregister_buffer(self, address: int) -> int:
        del address
        return 0

    def get(self, key: str) -> bytes:
        return self.objects[key]

    def get_into_ranges(
        self,
        addresses,
        all_keys,
        all_target_offsets,
        all_source_offsets,
        all_sizes,
    ):
        self.get_into_ranges_calls += 1
        results = []
        for address, keys, target_groups, source_groups, size_groups in _strict_zip(
            addresses,
            all_keys,
            all_target_offsets,
            all_source_offsets,
            all_sizes,
        ):
            object_results = []
            for key, target_offsets, source_offsets, sizes in _strict_zip(
                keys,
                target_groups,
                source_groups,
                size_groups,
            ):
                payload = self.objects[key]
                for target_offset, source_offset, size in _strict_zip(
                    target_offsets, source_offsets, sizes
                ):
                    ctypes.memmove(
                        address + target_offset,
                        payload[source_offset : source_offset + size],
                        size,
                    )
                object_results.append(list(sizes))
            results.append(object_results)
        return results


def descriptor() -> TensorDescriptor:
    return TensorDescriptor(
        tensor_id="layers.0.weight",
        global_shape=(8,),
        dtype="uint8",
        itemsize=1,
        shard_dims=(0,),
        layer_id=0,
        layout_fingerprint="placement-e2e:uint8:v2",
        parallel_axes=(SplitAxis("tp", dim=0),),
    )


def placement(
    *, shape: tuple[int, ...], offset: tuple[int, ...]
) -> WeightPlacementManifest:
    tensor = descriptor()
    fragments = [
        PlacementFragment(
            placement_fragment_id="target-placement-fragment",
            tensor_id=tensor.tensor_id,
            global_offset=offset,
            local_shape=shape,
            nbytes=shape[0],
            rank=ParallelRank(tp=1 if offset != (0,) else 0),
        )
    ]
    ranks = [fragments[0].rank]
    if offset != (0,):
        fragments.insert(
            0,
            PlacementFragment(
                placement_fragment_id="target-companion-fragment",
                tensor_id=tensor.tensor_id,
                global_offset=(0,),
                local_shape=offset,
                nbytes=offset[0],
                rank=ParallelRank(tp=0),
            ),
        )
        ranks.insert(0, ParallelRank(tp=0))
    return global_placement(
        resource_id="model",
        revision="revision",
        weight_generation=4,
        placement_set_id="target-placement",
        tensors=(tensor,),
        fragments=tuple(fragments),
        ranks=tuple(ranks),
    )


def binding(
    placement_manifest: WeightPlacementManifest,
    target: Buffer,
) -> WeightRuntimeBindingManifest:
    target_part = next(
        part
        for part in placement_manifest.parts
        if any(
            fragment.placement_fragment_id == "target-placement-fragment"
            for fragment in part.fragments
        )
    )
    return WeightRuntimeBindingManifest(
        resource_id="model",
        revision="revision",
        placement_id=placement_manifest.placement_id,
        placement_digest=placement_manifest.digest,
        participant_id=target_part.participant_id,
        instance_id="target",
        generation=4,
        lease_id="target-lease",
        fragments=(
            runtime_fragment(
                placement=next(
                    fragment
                    for fragment in target_part.fragments
                    if fragment.placement_fragment_id == "target-placement-fragment"
                ),
                tensor=placement_manifest.tensors[0],
                fragment_id="target-runtime-fragment",
                address=target.address,
                worker_id="target-worker",
                endpoint="target-endpoint",
            ),
        ),
    )


class PlacementExecutionE2ETest(unittest.TestCase):
    def test_direct_te_reader_executes_bound_plan_without_store_api(self) -> None:
        source_buffer = Buffer(8)
        source_buffer.write(bytes(range(8)))
        target_buffer = Buffer(4)
        target_buffer.write(b"\xff" * 4)
        tensor = descriptor()
        source_placement = global_placement(
            resource_id="model",
            revision="revision",
            weight_generation=4,
            placement_set_id="source-placement",
            tensors=(tensor,),
            fragments=(
                PlacementFragment(
                    placement_fragment_id="source-placement-fragment",
                    tensor_id=tensor.tensor_id,
                    global_offset=(0,),
                    local_shape=(8,),
                    nbytes=8,
                    rank=ParallelRank(),
                ),
            ),
        )
        source_binding = WeightRuntimeBindingManifest(
            resource_id=source_placement.resource_id,
            revision=source_placement.revision,
            placement_id=source_placement.placement_id,
            placement_digest=source_placement.digest,
            participant_id=source_placement.parts[0].participant_id,
            instance_id="source",
            generation=2,
            lease_id="source-lease",
            fragments=(
                runtime_fragment(
                    placement=source_placement.fragments[0],
                    tensor=tensor,
                    fragment_id="source-fragment",
                    address=source_buffer.address,
                    worker_id="source-worker",
                    endpoint="source-endpoint",
                ),
            ),
        )
        target_placement = placement(shape=(4,), offset=(4,))
        target_binding = binding(target_placement, target_buffer)
        logical = plan_placement_transfer_to_local_target(
            source_placement,
            target_placement,
            target_binding.participant_id,
        )
        plan = bind_logical_transfer_plan(
            logical,
            (target_binding,),
            source_bindings=(source_binding,),
        )
        engine = ReadOnlyTransferEngine()

        receipts = MooncakeTransferEngineReader(engine).execute(
            plan,
            source_placement,
            (source_binding,),
            target_placement,
            target_binding,
            source_registrations=(
                MemoryRegistrationLease.from_fragment(
                    source_binding.fragments[0],
                    lease_generation=source_binding.generation,
                    runtime_lease_id=source_binding.lease_id,
                ),
            ),
        )

        self.assertEqual(target_buffer.read(), bytes(range(4, 8)))
        self.assertEqual(engine.read_calls, 1)
        self.assertEqual(sum(receipt.nbytes for receipt in receipts), 4)
        self.assertFalse(hasattr(engine, "get_into_ranges"))

    def test_store_executes_bound_plan_through_get_into_ranges(self) -> None:
        payload = bytes(range(8))
        target_buffer = Buffer(8)
        target_buffer.write(b"\xff" * 8)
        tensor = descriptor()
        stored = StoredFragment(
            fragment_id="stored-fragment",
            tensor_id=tensor.tensor_id,
            global_offset=(0,),
            local_shape=(8,),
            object_key="weights/model/payload/0",
            object_offset=0,
            nbytes=8,
        )
        source = WeightManifest(
            namespace="default",
            resource_id="model",
            revision="revision",
            weight_generation=4,
            group_id="weights/model",
            manifest_key="weights/model/manifest",
            tensors=(tensor,),
            fragments=(stored,),
            created_at="2026-07-23T00:00:00Z",
        )
        target_placement = placement(shape=(8,), offset=(0,))
        target_binding = binding(target_placement, target_buffer)
        logical = plan_stored_transfer_to_target_placement(
            source,
            target_placement,
        )
        transfer = bind_logical_transfer_plan(logical, (target_binding,))
        store = RangeStore(
            {
                stored.object_key: payload,
                source.manifest_key: source.to_json().encode(),
            }
        )

        WeightStore(store).load(
            WeightLoadPlan(manifest=source, transfer=transfer),
            target_placement,
            target_binding,
        )

        self.assertEqual(target_buffer.read(), payload)
        self.assertEqual(store.get_into_ranges_calls, 1)
        self.assertFalse(hasattr(store, "batch_transfer_sync_read"))

    def test_store_planner_rejects_a_different_weight_generation(self) -> None:
        tensor = descriptor()
        source = WeightManifest(
            namespace="default",
            resource_id="model",
            revision="revision",
            weight_generation=4,
            group_id="weights/model/4",
            manifest_key="weights/model/4/manifest",
            tensors=(tensor,),
            fragments=(
                StoredFragment(
                    fragment_id="stored-fragment",
                    tensor_id=tensor.tensor_id,
                    global_offset=(0,),
                    local_shape=(8,),
                    object_key="weights/model/4/payload/0",
                    object_offset=0,
                    nbytes=8,
                ),
            ),
            created_at="2026-07-23T00:00:00Z",
        )
        target = global_placement(
            resource_id="model",
            revision="revision",
            weight_generation=5,
            placement_set_id="target-placement-generation-5",
            tensors=(tensor,),
            fragments=(
                PlacementFragment(
                    placement_fragment_id="target-placement-fragment",
                    tensor_id=tensor.tensor_id,
                    global_offset=(0,),
                    local_shape=(8,),
                    nbytes=8,
                    rank=ParallelRank(),
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "weight_generation"):
            plan_stored_transfer_to_target_placement(source, target)

    def test_four_axis_placement_plan_fills_every_target_logical_box(self) -> None:
        tensors = tuple(
            TensorDescriptor(
                tensor_id=f"layers.{layer}.experts.weight",
                global_shape=(8, 8),
                dtype="uint8",
                itemsize=1,
                shard_dims=(0, 1),
                layer_id=layer,
                layout_fingerprint="placement-e2e:four-axis:uint8:v2",
                parallel_axes=(
                    ReplicatedAxis("dp"),
                    OwnershipAxis("pp"),
                    SplitAxis("ep", dim=0),
                    SplitAxis("tp", dim=1),
                ),
            )
            for layer in range(4)
        )
        source_records = []
        source_buffers = []
        for dp_rank in range(2):
            for pp_rank in range(2):
                local_tensors = tuple(
                    tensor for tensor in tensors if tensor.layer_id % 2 == pp_rank
                )
                for ep_rank in range(8):
                    for tp_rank in range(4):
                        worker_id = (
                            f"source-d{dp_rank}-p{pp_rank}-e{ep_rank}-t{tp_rank}"
                        )
                        placement_fragments = []
                        binding_fragments = []
                        for tensor in local_tensors:
                            assert tensor.layer_id is not None
                            payload = bytes(
                                tensor.layer_id * 64 + ep_rank * 8 + column
                                for column in range(tp_rank * 2, tp_rank * 2 + 2)
                            )
                            buffer = Buffer(len(payload))
                            buffer.write(payload)
                            source_buffers.append(buffer)
                            placement_fragment = PlacementFragment(
                                tensor_id=tensor.tensor_id,
                                global_offset=(ep_rank, tp_rank * 2),
                                local_shape=(1, 2),
                                nbytes=buffer.size,
                                rank=ParallelRank(
                                    dp=dp_rank,
                                    tp=tp_rank,
                                    pp=pp_rank,
                                    ep=ep_rank,
                                ),
                            )
                            placement_fragments.append(placement_fragment)
                            binding_fragments.append(
                                runtime_fragment(
                                    placement=placement_fragment,
                                    tensor=tensor,
                                    fragment_id=f"{worker_id}:{tensor.tensor_id}",
                                    address=buffer.address,
                                    worker_id=worker_id,
                                    endpoint="source-endpoint",
                                    owner=buffer,
                                )
                            )
                        source_records.append(
                            (
                                ParallelRank(
                                    dp=dp_rank,
                                    tp=tp_rank,
                                    pp=pp_rank,
                                    ep=ep_rank,
                                ),
                                worker_id,
                                tuple(placement_fragments),
                                tuple(binding_fragments),
                            )
                        )

        source_placement = global_placement(
            resource_id="four-axis-model",
            revision="four-axis-revision",
            weight_generation=11,
            placement_set_id="four-axis-source",
            tensors=tensors,
            fragments=tuple(
                fragment
                for _, _, placement_fragments, _ in source_records
                for fragment in placement_fragments
            ),
            ranks=tuple(rank for rank, _, _, _ in source_records),
        )
        source_bindings = tuple(
            WeightRuntimeBindingManifest(
                resource_id=source_placement.resource_id,
                revision=source_placement.revision,
                placement_id=source_placement.placement_id,
                placement_digest=source_placement.digest,
                participant_id=next(
                    participant.participant_id
                    for participant in source_placement.topology.participants
                    if participant.rank == rank
                ),
                instance_id=worker_id,
                generation=3,
                lease_id=f"lease:{worker_id}",
                fragments=binding_fragments,
            )
            for rank, worker_id, _, binding_fragments in source_records
        )

        target_records = []
        for dp_rank in range(4):
            for pp_rank in range(4):
                tensor = tensors[pp_rank]
                for ep_rank in range(2):
                    for tp_rank in range(8):
                        rank = ParallelRank(
                            dp=dp_rank,
                            tp=tp_rank,
                            pp=pp_rank,
                            ep=ep_rank,
                        )
                        instance_id = (
                            f"target-d{dp_rank}-p{pp_rank}-e{ep_rank}-t{tp_rank}"
                        )
                        placement_fragment = PlacementFragment(
                            tensor_id=tensor.tensor_id,
                            global_offset=(ep_rank * 4, tp_rank),
                            local_shape=(4, 1),
                            nbytes=4,
                            rank=rank,
                        )
                        buffer = Buffer(4)
                        buffer.write(b"\xff" * 4)
                        binding_fragment = runtime_fragment(
                            placement=placement_fragment,
                            tensor=tensor,
                            fragment_id=f"runtime:{instance_id}",
                            address=buffer.address,
                            worker_id=instance_id,
                            endpoint="target-endpoint",
                            owner=buffer,
                        )
                        target_records.append(
                            (
                                rank,
                                instance_id,
                                placement_fragment,
                                binding_fragment,
                                buffer,
                                bytes(
                                    pp_rank * 64 + row * 8 + tp_rank
                                    for row in range(ep_rank * 4, ep_rank * 4 + 4)
                                ),
                            )
                        )

        target_placement = global_placement(
            resource_id="four-axis-model",
            revision="four-axis-revision",
            weight_generation=11,
            placement_set_id="four-axis-target",
            tensors=tensors,
            fragments=tuple(record[2] for record in target_records),
            ranks=tuple(record[0] for record in target_records),
        )
        target_bindings = tuple(
            WeightRuntimeBindingManifest(
                resource_id=target_placement.resource_id,
                revision=target_placement.revision,
                placement_id=target_placement.placement_id,
                placement_digest=target_placement.digest,
                participant_id=next(
                    participant.participant_id
                    for participant in target_placement.topology.participants
                    if participant.rank == rank
                ),
                instance_id=instance_id,
                generation=7,
                lease_id=f"lease:{instance_id}",
                fragments=(binding_fragment,),
            )
            for rank, instance_id, _, binding_fragment, _, _ in target_records
        )
        expected_targets = tuple(
            (
                target_placement,
                binding,
                buffer,
                expected,
            )
            for binding, (_, _, _, _, buffer, expected) in _strict_zip(
                target_bindings,
                target_records,
            )
        )

        logical = plan_placement_transfer(source_placement, target_placement)
        self.assertTrue(
            all(
                not hasattr(operation.target, "address")
                for operation in logical.operations
            )
        )
        self.assertLessEqual(len(logical.operations), 4096)
        transfer = bind_logical_transfer_plan(
            logical,
            target_bindings,
            source_bindings=source_bindings,
        )
        source_registrations = tuple(
            MemoryRegistrationLease.from_fragment(
                fragment,
                lease_generation=source_binding.generation,
                runtime_lease_id=source_binding.lease_id,
            )
            for source_binding in source_bindings
            for fragment in source_binding.fragments
        )
        reader = MooncakeTransferEngineReader(ReadOnlyTransferEngine())

        for target_placement, target_binding, buffer, expected in expected_targets:
            receipts = reader.execute(
                transfer,
                source_placement,
                source_bindings,
                target_placement,
                target_binding,
                source_registrations=source_registrations,
            )
            self.assertEqual(sum(receipt.nbytes for receipt in receipts), buffer.size)
            self.assertEqual(buffer.read(), expected)


if __name__ == "__main__":
    unittest.main()
