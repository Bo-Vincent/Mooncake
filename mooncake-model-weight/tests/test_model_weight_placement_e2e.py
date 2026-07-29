from __future__ import annotations

import ctypes
import unittest

from mooncake.model_weight import (
    MemoryRegistrationLease,
    MooncakeTransferEngineReader,
    ParallelRank,
    PlacementFragment,
    RuntimeBindingFragment,
    RuntimeBindingManifest,
    RuntimeFragment,
    RuntimeManifest,
    StoredFragment,
    TargetPlacementManifest,
    TensorDescriptor,
    WeightLoadPlan,
    WeightManifest,
    WeightStore,
    bind_logical_transfer_plan,
    bind_runtime_manifest,
    plan_runtime_transfer_to_local_target_placement,
    plan_runtime_transfer_to_target_placements,
    plan_stored_transfer_to_target_placements,
)


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
        for target, source, size in zip(
            target_addresses, source_addresses, sizes, strict=True
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
        for address, keys, target_groups, source_groups, size_groups in zip(
            addresses,
            all_keys,
            all_target_offsets,
            all_source_offsets,
            all_sizes,
            strict=True,
        ):
            object_results = []
            for key, target_offsets, source_offsets, sizes in zip(
                keys,
                target_groups,
                source_groups,
                size_groups,
                strict=True,
            ):
                payload = self.objects[key]
                for target_offset, source_offset, size in zip(
                    target_offsets, source_offsets, sizes, strict=True
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
        partition_dim=None,
        shard_dims=(0,),
        layer_id=0,
        layout_fingerprint="placement-e2e:uint8:v2",
    )


def placement(
    *, shape: tuple[int, ...], offset: tuple[int, ...]
) -> TargetPlacementManifest:
    tensor = descriptor()
    return TargetPlacementManifest(
        model_id="model",
        revision="revision",
        placement_id=None,
        tensors=(tensor,),
        fragments=(
            PlacementFragment(
                placement_fragment_id="target-placement-fragment",
                tensor_id=tensor.tensor_id,
                global_offset=offset,
                local_shape=shape,
                nbytes=shape[0],
                rank=ParallelRank(tp=1),
            ),
        ),
    )


def bind(
    placement_manifest: TargetPlacementManifest, target: Buffer
) -> RuntimeManifest:
    return bind_runtime_manifest(
        placement_manifest,
        RuntimeBindingManifest(
            model_id="model",
            revision="revision",
            placement_id=placement_manifest.placement_id,
            placement_digest=placement_manifest.digest,
            instance_id="target",
            generation=4,
            lease_id="target-lease",
            fragments=(
                RuntimeBindingFragment(
                    placement_fragment_id="target-placement-fragment",
                    fragment_id="target-runtime-fragment",
                    address=target.address,
                    nbytes=target.size,
                    worker_id="target-worker",
                    endpoint="target-endpoint",
                    device="cuda:0",
                ),
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
        source = RuntimeManifest(
            model_id="model",
            revision="revision",
            instance_id="source",
            lease_id="source-lease",
            tensors=(tensor,),
            fragments=(
                RuntimeFragment(
                    fragment_id="source-fragment",
                    tensor_id=tensor.tensor_id,
                    global_offset=(0,),
                    local_shape=(8,),
                    address=source_buffer.address,
                    nbytes=8,
                    worker_id="source-worker",
                    endpoint="source-endpoint",
                    device="cuda:0",
                    rank=ParallelRank(),
                    lease_generation=2,
                ),
            ),
        )
        target_placement = placement(shape=(4,), offset=(4,))
        logical = plan_runtime_transfer_to_local_target_placement(
            (source,), target_placement
        )
        target = bind(target_placement, target_buffer)
        plan = bind_logical_transfer_plan(logical, (target,), source_bindings=(source,))
        engine = ReadOnlyTransferEngine()

        receipts = MooncakeTransferEngineReader(engine).execute(
            plan,
            (source,),
            target,
            source_registrations=(
                MemoryRegistrationLease.from_fragment(
                    source.fragments[0], runtime_lease_id=source.lease_id
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
            model_id="model",
            revision="revision",
            group_id="weights/model",
            manifest_key="weights/model/manifest",
            tensors=(tensor,),
            fragments=(stored,),
            created_at="2026-07-23T00:00:00Z",
        )
        target_placement = placement(shape=(8,), offset=(0,))
        logical = plan_stored_transfer_to_target_placements(source, (target_placement,))
        target = bind(target_placement, target_buffer)
        transfer = bind_logical_transfer_plan(logical, (target,))
        store = RangeStore({stored.object_key: payload})

        WeightStore(store).load(
            WeightLoadPlan(manifest=source, transfer=transfer), target
        )

        self.assertEqual(target_buffer.read(), payload)
        self.assertEqual(store.get_into_ranges_calls, 1)
        self.assertFalse(hasattr(store, "batch_transfer_sync_read"))

    def test_four_axis_placement_plan_fills_every_target_logical_box(self) -> None:
        tensors = tuple(
            TensorDescriptor(
                tensor_id=f"layers.{layer}.experts.weight",
                global_shape=(8, 8),
                dtype="uint8",
                itemsize=1,
                partition_dim=None,
                shard_dims=(0, 1),
                layer_id=layer,
                layout_fingerprint="placement-e2e:four-axis:uint8:v2",
            )
            for layer in range(4)
        )
        sources = []
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
                        fragments = []
                        for tensor in local_tensors:
                            assert tensor.layer_id is not None
                            payload = bytes(
                                tensor.layer_id * 64 + ep_rank * 8 + column
                                for column in range(tp_rank * 2, tp_rank * 2 + 2)
                            )
                            buffer = Buffer(len(payload))
                            buffer.write(payload)
                            source_buffers.append(buffer)
                            fragments.append(
                                RuntimeFragment(
                                    fragment_id=f"{worker_id}:{tensor.tensor_id}",
                                    tensor_id=tensor.tensor_id,
                                    global_offset=(ep_rank, tp_rank * 2),
                                    local_shape=(1, 2),
                                    address=buffer.address,
                                    nbytes=buffer.size,
                                    worker_id=worker_id,
                                    endpoint="source-endpoint",
                                    device="cuda:0",
                                    rank=ParallelRank(
                                        dp=dp_rank,
                                        tp=tp_rank,
                                        pp=pp_rank,
                                        ep=ep_rank,
                                    ),
                                    lease_generation=3,
                                    owner=buffer,
                                )
                            )
                        sources.append(
                            RuntimeManifest(
                                model_id="four-axis-model",
                                revision="four-axis-revision",
                                instance_id=worker_id,
                                lease_id=f"lease:{worker_id}",
                                tensors=local_tensors,
                                fragments=tuple(fragments),
                            )
                        )

        placements = []
        targets = []
        expected_targets = []
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
                        placement_id = (
                            f"target-d{dp_rank}-p{pp_rank}-e{ep_rank}-t{tp_rank}"
                        )
                        placement_fragment_id = f"placement:{placement_id}"
                        placement_manifest = TargetPlacementManifest(
                            model_id="four-axis-model",
                            revision="four-axis-revision",
                            placement_id=None,
                            tensors=(tensor,),
                            fragments=(
                                PlacementFragment(
                                    placement_fragment_id=placement_fragment_id,
                                    tensor_id=tensor.tensor_id,
                                    global_offset=(ep_rank * 4, tp_rank),
                                    local_shape=(4, 1),
                                    nbytes=4,
                                    rank=rank,
                                ),
                            ),
                        )
                        buffer = Buffer(4)
                        buffer.write(b"\xff" * 4)
                        binding = RuntimeBindingManifest(
                            model_id="four-axis-model",
                            revision="four-axis-revision",
                            placement_id=placement_manifest.placement_id,
                            placement_digest=placement_manifest.digest,
                            instance_id=placement_id,
                            generation=7,
                            lease_id=f"lease:{placement_id}",
                            fragments=(
                                RuntimeBindingFragment(
                                    placement_fragment_id=placement_fragment_id,
                                    fragment_id=f"runtime:{placement_id}",
                                    address=buffer.address,
                                    nbytes=buffer.size,
                                    worker_id=placement_id,
                                    endpoint="target-endpoint",
                                    device="cuda:0",
                                    owner=buffer,
                                ),
                            ),
                        )
                        placements.append(placement_manifest)
                        target = bind_runtime_manifest(placement_manifest, binding)
                        targets.append(target)
                        expected_targets.append(
                            (
                                target,
                                buffer,
                                bytes(
                                    pp_rank * 64 + row * 8 + tp_rank
                                    for row in range(ep_rank * 4, ep_rank * 4 + 4)
                                ),
                            )
                        )

        logical = plan_runtime_transfer_to_target_placements(sources, placements)
        self.assertTrue(
            all(
                not hasattr(operation.target, "address")
                for operation in logical.operations
            )
        )
        self.assertLessEqual(len(logical.operations), 4096)
        transfer = bind_logical_transfer_plan(logical, targets, source_bindings=sources)
        source_registrations = tuple(
            MemoryRegistrationLease.from_fragment(
                fragment, runtime_lease_id=source.lease_id
            )
            for source in sources
            for fragment in source.fragments
        )
        reader = MooncakeTransferEngineReader(ReadOnlyTransferEngine())

        for target, buffer, expected in expected_targets:
            receipts = reader.execute(
                transfer,
                sources,
                target,
                source_registrations=source_registrations,
            )
            self.assertEqual(sum(receipt.nbytes for receipt in receipts), buffer.size)
            self.assertEqual(buffer.read(), expected)


if __name__ == "__main__":
    unittest.main()
