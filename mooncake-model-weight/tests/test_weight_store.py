from __future__ import annotations

import ctypes
from dataclasses import dataclass, replace
from itertools import product
from math import prod

import pytest

from mooncake.model_weight.manifest import (
    ParallelRank,
    RuntimeFragment,
    RuntimeManifest,
    TensorDescriptor,
    WeightManifest,
)
from mooncake.model_weight.planner import plan_runtime_transfer
from mooncake.model_weight.store import (
    UploadReceipt,
    WeightStore,
    WeightStoreError,
)


@dataclass
class FakeReplicateConfig:
    group_ids: list[str]
    data_type: str
    with_hard_pin: bool


class InMemoryStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.group_ids: dict[str, str] = {}
        self.configs: dict[str, tuple[str, bool]] = {}
        self.calls: list[str] = []
        self.put_batches: list[tuple[str, ...]] = []
        self.registered: set[int] = set()
        self.register_calls = 0
        self.register_args: list[tuple[int, int]] = []
        self.unregister_calls = 0
        self.unregister_addresses: list[int] = []
        self.fail_key: str | None = None
        self.processing_keys: set[str] = set()
        self.range_get_calls = 0
        self.range_sizes: list[int] = []
        self.range_batch_sizes: list[int] = []
        self.exist_batch_sizes: list[int] = []
        self.register_result = 0
        self.removed_keys: list[str] = []
        self.remove_forces: list[bool] = []
        self.unregister_results: dict[int, int] = {}
        self.unregister_exceptions: dict[int, Exception] = {}
        self.fail_after_write_key: str | None = None
        self.manifest_race_value: bytes | None = None
        self.manifest_race_key: str | None = None
        self.after_batch_is_exist = None

    def register_buffer(self, address: int, nbytes: int) -> int:
        self.calls.append("register_buffer")
        self.register_calls += 1
        self.register_args.append((address, nbytes))
        if self.register_result != 0:
            return self.register_result
        self.registered.add(address)
        return 0

    def unregister_buffer(self, address: int) -> int:
        self.calls.append("unregister_buffer")
        self.unregister_calls += 1
        self.unregister_addresses.append(address)
        if address in self.unregister_exceptions:
            raise self.unregister_exceptions[address]
        result = self.unregister_results.get(address, 0)
        if result == 0:
            self.registered.remove(address)
        return result

    def batch_put_from(
        self,
        keys: list[str],
        addresses: list[int],
        sizes: list[int],
        config: FakeReplicateConfig,
    ) -> list[int]:
        self.calls.append("batch_put_from")
        self.put_batches.append(tuple(keys))
        results = []
        for key, address, size, group_id in zip(
            keys, addresses, sizes, config.group_ids
        ):
            if key == self.fail_key:
                results.append(-1)
                continue
            if key in self.processing_keys:
                results.append(0)
                continue
            self.objects[key] = ctypes.string_at(address, size)
            self.group_ids[key] = group_id
            self.configs[key] = (config.data_type, config.with_hard_pin)
            results.append(0)
        return results

    def put(self, key: str, value, config: FakeReplicateConfig) -> int:
        self.calls.append("put")
        if self.manifest_race_value is not None and key == self.manifest_race_key:
            self.objects[key] = self.manifest_race_value
            self.group_ids[key] = config.group_ids[0]
            self.configs[key] = (config.data_type, config.with_hard_pin)
            self.manifest_race_value = None
            return 0
        if key == self.fail_key:
            return -1
        if key in self.processing_keys:
            return 0
        if key in self.objects:
            return 0
        self.objects[key] = bytes(value)
        self.group_ids[key] = config.group_ids[0]
        self.configs[key] = (config.data_type, config.with_hard_pin)
        if key == self.fail_after_write_key:
            return -1
        return 0

    def get(self, key: str) -> bytes:
        self.calls.append("get")
        return self.objects[key]

    def is_exist(self, key: str) -> int:
        self.calls.append("is_exist")
        return int(key in self.objects)

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        self.calls.append("batch_is_exist")
        self.exist_batch_sizes.append(len(keys))
        results = [self.is_exist(key) for key in keys]
        callback = self.after_batch_is_exist
        self.after_batch_is_exist = None
        if callback is not None:
            callback()
        return results

    def remove(self, key: str, force: bool = False) -> int:
        self.calls.append("remove")
        self.removed_keys.append(key)
        self.remove_forces.append(force)
        self.objects.pop(key, None)
        self.processing_keys.discard(key)
        self.group_ids.pop(key, None)
        return 0

    def get_into_ranges(
        self,
        addresses: list[int],
        all_keys: list[list[str]],
        all_dst_offsets: list[list[list[int]]],
        all_src_offsets: list[list[list[int]]],
        all_sizes: list[list[list[int]]],
    ) -> list[list[list[int]]]:
        self.calls.append("get_into_ranges")
        self.range_get_calls += 1
        self.range_batch_sizes.append(
            sum(len(sizes) for buffer in all_sizes for sizes in buffer)
        )
        results = []
        for address, keys, dst_offsets, src_offsets, sizes in zip(
            addresses,
            all_keys,
            all_dst_offsets,
            all_src_offsets,
            all_sizes,
        ):
            buffer_results = []
            for key, dst_group, src_group, size_group in zip(
                keys, dst_offsets, src_offsets, sizes
            ):
                object_data = self.objects[key]
                range_results = []
                for dst, src, size in zip(dst_group, src_group, size_group):
                    self.range_sizes.append(size)
                    ctypes.memmove(address + dst, object_data[src : src + size], size)
                    range_results.append(size)
                buffer_results.append(range_results)
            results.append(buffer_results)
        return results


def tensor_descriptor() -> TensorDescriptor:
    return TensorDescriptor(
        tensor_id="layers.2.experts.3.w1",
        global_shape=(8,),
        dtype="uint8",
        itemsize=1,
        partition_dim=0,
        layer_id=2,
        expert_id=3,
        layout_fingerprint="sglang:qwen3.5:uint8:test",
    )


def source_manifests(dp: int = 2, tp: int = 2) -> tuple[RuntimeManifest, ...]:
    tensor = tensor_descriptor()
    extent = tensor.global_shape[0] // tp
    manifests = []
    for dp_rank in range(dp):
        for tp_rank in range(tp):
            start = tp_rank * extent
            owner = (ctypes.c_ubyte * extent)(*range(start, start + extent))
            worker_id = f"source-d{dp_rank}-t{tp_rank}"
            fragment = RuntimeFragment(
                fragment_id=f"{worker_id}-fragment",
                tensor_id=tensor.tensor_id,
                global_offset=(start,),
                local_shape=(extent,),
                address=ctypes.addressof(owner),
                nbytes=extent,
                worker_id=worker_id,
                endpoint=f"{worker_id}:12345",
                device="cuda:0",
                rank=ParallelRank(dp=dp_rank, tp=tp_rank, pp=1, ep=1),
                lease_generation=1,
                owner=owner,
            )
            manifests.append(
                RuntimeManifest(
                    model_id="qwen3.5-0.8b",
                    revision="step-42",
                    instance_id=worker_id,
                    tensors=(tensor,),
                    fragments=(fragment,),
                )
            )
    return tuple(manifests)


def target_manifests(
    dp: int = 3, tp: int = 4, pp_rank: int = 3, ep_rank: int = 7
) -> tuple[RuntimeManifest, ...]:
    tensor = tensor_descriptor()
    extent = tensor.global_shape[0] // tp
    manifests = []
    for dp_rank in range(dp):
        for tp_rank in range(tp):
            owner = (ctypes.c_ubyte * extent)(*[255] * extent)
            worker_id = f"target-d{dp_rank}-t{tp_rank}"
            fragment = RuntimeFragment(
                fragment_id=f"{worker_id}-fragment",
                tensor_id=tensor.tensor_id,
                global_offset=(tp_rank * extent,),
                local_shape=(extent,),
                address=ctypes.addressof(owner),
                nbytes=extent,
                worker_id=worker_id,
                endpoint=f"{worker_id}:12345",
                device="cuda:0",
                rank=ParallelRank(
                    dp=dp_rank,
                    tp=tp_rank,
                    pp=pp_rank,
                    ep=ep_rank,
                ),
                lease_generation=1,
                owner=owner,
            )
            manifests.append(
                RuntimeManifest(
                    model_id="qwen3.5-0.8b",
                    revision="step-42",
                    instance_id=worker_id,
                    tensors=(tensor,),
                    fragments=(fragment,),
                )
            )
    return tuple(manifests)


def nd_store_manifests(
    prefix: str,
    *,
    source: bool,
    target_dim: int = 2,
) -> tuple[RuntimeManifest, ...]:
    tensor = TensorDescriptor(
        tensor_id="layers.0.experts.w1",
        global_shape=(4, 6, 8),
        dtype="uint8",
        itemsize=1,
        partition_dim=None,
        layer_id=0,
        expert_id=None,
        layout_fingerprint="framework:logical-contiguous:v2",
        shard_dims=(0,) if source else (target_dim,),
    )
    manifests = []
    rank_count = 4 if source else 2
    for rank in range(rank_count):
        if source:
            offset = (rank, 0, 0)
            shape = (1, 6, 8)
            parallel_rank = ParallelRank(ep=rank)
        else:
            shape_list = list(tensor.global_shape)
            shape_list[target_dim] //= rank_count
            offset_list = [0, 0, 0]
            offset_list[target_dim] = rank * shape_list[target_dim]
            offset = tuple(offset_list)
            shape = tuple(shape_list)
            parallel_rank = ParallelRank(tp=rank)
        values = []
        for coordinate in product(*(range(extent) for extent in shape)):
            global_coordinate = tuple(
                begin + local for begin, local in zip(offset, coordinate, strict=True)
            )
            values.append(
                global_coordinate[0] * 48
                + global_coordinate[1] * 8
                + global_coordinate[2]
                if source
                else 255
            )
        owner = (ctypes.c_ubyte * prod(shape))(*values)
        worker_id = f"{prefix}-{rank}"
        fragment = RuntimeFragment(
            fragment_id=f"{worker_id}-fragment",
            tensor_id=tensor.tensor_id,
            global_offset=offset,
            local_shape=shape,
            address=ctypes.addressof(owner),
            nbytes=prod(shape),
            worker_id=worker_id,
            endpoint=f"{worker_id}:12345",
            device="cuda:0",
            rank=parallel_rank,
            lease_generation=1,
            owner=owner,
        )
        manifests.append(
            RuntimeManifest(
                model_id="qwen-family-moe",
                revision="step-42",
                instance_id=worker_id,
                tensors=(tensor,),
                fragments=(fragment,),
            )
        )
    return tuple(manifests)


def expected_nd_fragment(fragment: RuntimeFragment) -> bytes:
    values = []
    for coordinate in product(*(range(extent) for extent in fragment.local_shape)):
        global_coordinate = tuple(
            begin + local
            for begin, local in zip(fragment.global_offset, coordinate, strict=True)
        )
        values.append(
            global_coordinate[0] * 48 + global_coordinate[1] * 8 + global_coordinate[2]
        )
    return bytes(values)


def make_weight_store(
    store: InMemoryStore | None = None,
    *,
    max_range_bytes: int = 64 * 1024 * 1024,
    max_ranges_per_request: int = 1024,
    max_region_segments: int = 1_000_000,
):
    current = store or InMemoryStore()
    return current, WeightStore(
        current,
        config_factory=lambda group_ids, record_type: FakeReplicateConfig(
            list(group_ids),
            data_type=("WEIGHT" if record_type == "payload" else "METADATA"),
            with_hard_pin=True,
        ),
        max_range_bytes=max_range_bytes,
        max_ranges_per_request=max_ranges_per_request,
        max_region_segments=max_region_segments,
    )


def upload_all(weight_store: WeightStore, plan, manifests):
    receipts = []
    try:
        for manifest in manifests:
            receipts.extend(weight_store.upload(plan, manifest))
    except Exception:
        weight_store.abort_upload(plan, receipts)
        raise
    return receipts


@pytest.mark.parametrize("target_dim", [1, 2])
def test_store_preserves_expert_boxes_and_loads_cross_dim(
    target_dim: int,
) -> None:
    store, weight_store = make_weight_store(max_ranges_per_request=5)
    sources = nd_store_manifests("source", source=True)
    targets = nd_store_manifests("target", source=False, target_dim=target_dim)

    upload_plan = weight_store.prepare_upload(sources)

    assert upload_plan.manifest.tensors[0].effective_shard_dims == (0,)
    assert len(upload_plan.operations) == 4
    assert len({item.target.object_key for item in upload_plan.operations}) == 4
    assert {item.target.global_offset for item in upload_plan.operations} == {
        (rank, 0, 0) for rank in range(4)
    }

    manifest = weight_store.commit(
        upload_plan, upload_all(weight_store, upload_plan, sources)
    )
    loaded = weight_store.load_manifest(manifest.manifest_key)
    load_plan = weight_store.plan_load(loaded, targets)
    for target in targets:
        weight_store.load(load_plan, target)

    assert loaded == manifest
    assert all(route.source_pp is None for route in load_plan.transfer.pipeline_routes)
    assert max(store.range_batch_sizes) <= 5
    for target in targets:
        assert bytes(target.fragments[0].owner) == expected_nd_fragment(
            target.fragments[0]
        )


def test_store_commit_preserves_mixed_single_axis_descriptor() -> None:
    single_axis = TensorDescriptor(
        tensor_id="layers.0.attn.qkv",
        global_shape=(4,),
        dtype="uint8",
        itemsize=1,
        partition_dim=0,
        layer_id=0,
        expert_id=None,
        layout_fingerprint="framework:single-axis-contiguous",
    )
    sources = []
    for rank, manifest in enumerate(nd_store_manifests("source", source=True)):
        owner = (ctypes.c_ubyte * 1)(rank)
        single_axis_fragment = RuntimeFragment(
            fragment_id=f"source-{rank}-single-axis",
            tensor_id=single_axis.tensor_id,
            global_offset=(rank,),
            local_shape=(1,),
            address=ctypes.addressof(owner),
            nbytes=1,
            worker_id=manifest.fragments[0].worker_id,
            endpoint=manifest.fragments[0].endpoint,
            device="cuda:0",
            rank=manifest.fragments[0].rank,
            lease_generation=1,
            owner=owner,
        )
        sources.append(
            replace(
                manifest,
                tensors=(*manifest.tensors, single_axis),
                fragments=(*manifest.fragments, single_axis_fragment),
            )
        )

    _store, weight_store = make_weight_store()
    upload_plan = weight_store.prepare_upload(tuple(sources))
    persisted = weight_store.commit(
        upload_plan,
        upload_all(weight_store, upload_plan, tuple(sources)),
    )
    loaded = weight_store.load_manifest(persisted.manifest_key)

    assert loaded == persisted == upload_plan.manifest
    loaded_single_axis = next(
        tensor for tensor in loaded.tensors if tensor.tensor_id == single_axis.tensor_id
    )
    assert loaded_single_axis.partition_dim == 0
    assert loaded_single_axis.shard_dims is None


def test_store_nd_lowering_limit_fails_before_registration_or_read() -> None:
    store, weight_store = make_weight_store(max_region_segments=5)
    sources = nd_store_manifests("source", source=True)
    targets = nd_store_manifests("target", source=False, target_dim=2)
    upload_plan = weight_store.prepare_upload(sources)
    manifest = weight_store.commit(
        upload_plan, upload_all(weight_store, upload_plan, sources)
    )
    load_plan = weight_store.plan_load(manifest, targets)
    register_calls = store.register_calls
    range_get_calls = store.range_get_calls

    with pytest.raises(WeightStoreError, match="max_region_segments"):
        weight_store.load(load_plan, targets[0])

    assert store.register_calls == register_calls
    assert store.range_get_calls == range_get_calls


def test_upload_deduplicates_dp_and_commits_manifest_last() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests()

    plan = weight_store.prepare_upload(sources, namespace="default")
    receipts = upload_all(weight_store, plan, sources)

    assert len(plan.operations) == 2
    assert len(receipts) == 2
    assert plan.manifest.manifest_key not in store.objects
    assert len(store.objects) == 2

    manifest = weight_store.commit(plan, receipts)

    assert WeightManifest.from_json(store.objects[manifest.manifest_key]) == manifest
    assert len(store.objects) == 4
    revision_keys = {
        manifest.manifest_key,
        *(operation.target.object_key for operation in plan.operations),
    }
    assert {store.group_ids[key] for key in revision_keys} == {manifest.group_id}
    assert store.group_ids[plan.control_key] == plan.session_group_id
    assert plan.session_group_id != manifest.group_id
    assert store.register_calls == 2
    assert store.unregister_calls == 2
    assert store.registered == set()


def test_prepare_upload_selects_one_complete_generation_consistent_dp_replica() -> None:
    sources = []
    for manifest in source_manifests(dp=2, tp=2):
        fragment = manifest.fragments[0]
        if fragment.rank.dp == 0 and fragment.rank.tp == 1:
            continue
        if fragment.rank.dp == 1:
            fragment = replace(fragment, lease_generation=2)
            manifest = replace(
                manifest,
                fragments=(fragment,),
                generation=None,
            )
        sources.append(manifest)

    _, weight_store = make_weight_store()
    plan = weight_store.prepare_upload(tuple(sources))

    assert {operation.source.rank.dp for operation in plan.operations} == {1}
    assert {operation.source.lease_generation for operation in plan.operations} == {2}


def test_prepare_upload_rejects_complete_dp_replicas_at_different_generations() -> None:
    sources = tuple(
        replace(
            manifest,
            generation=None,
            fragments=(
                replace(
                    manifest.fragments[0],
                    lease_generation=manifest.fragments[0].rank.dp + 1,
                ),
            ),
        )
        for manifest in source_manifests(dp=2, tp=2)
    )
    _, weight_store = make_weight_store()

    with pytest.raises(ValueError, match="inconsistent lease generations"):
        weight_store.prepare_upload(sources)


def test_prepare_upload_rejects_mixed_generations_within_one_dp_replica() -> None:
    sources = list(source_manifests(dp=1, tp=2))
    sources[1] = replace(
        sources[1],
        generation=None,
        fragments=(replace(sources[1].fragments[0], lease_generation=2),),
    )
    _, weight_store = make_weight_store()

    with pytest.raises(ValueError, match="generation-consistent DP replica"):
        weight_store.prepare_upload(tuple(sources))


def test_prepare_upload_matches_te_planner_owner_selection() -> None:
    higher_owner = tuple(
        replace(
            manifest,
            fragments=(
                replace(
                    manifest.fragments[0],
                    rank=replace(manifest.fragments[0].rank, pp=2, ep=2),
                ),
            ),
        )
        for manifest in source_manifests(dp=1, tp=2)
    )
    lower_owner = tuple(
        replace(
            manifest,
            instance_id=f"{manifest.instance_id}-owner-1",
            fragments=(
                replace(
                    manifest.fragments[0],
                    fragment_id=f"{manifest.fragments[0].fragment_id}-owner-1",
                    worker_id=f"{manifest.fragments[0].worker_id}-owner-1",
                    endpoint=f"{manifest.fragments[0].worker_id}-owner-1:12345",
                    rank=replace(manifest.fragments[0].rank, pp=1, ep=1),
                ),
            ),
        )
        for manifest in higher_owner
    )
    sources = (*higher_owner, *lower_owner)
    targets = target_manifests(dp=1, tp=1)
    _, weight_store = make_weight_store()

    te_plan = plan_runtime_transfer(sources, targets)
    store_plan = weight_store.prepare_upload(sources)

    assert {
        (operation.source.rank.pp, operation.source.rank.ep)
        for operation in te_plan.operations
    } == {(1, 1)}
    assert {
        (operation.source.rank.pp, operation.source.rank.ep)
        for operation in store_plan.operations
    } == {(1, 1)}


def test_prepare_upload_matches_te_planner_rejecting_cross_owner_coverage() -> None:
    sources = tuple(
        replace(
            manifest,
            fragments=(
                replace(
                    manifest.fragments[0],
                    rank=replace(
                        manifest.fragments[0].rank,
                        pp=manifest.fragments[0].rank.tp,
                        ep=manifest.fragments[0].rank.tp,
                    ),
                ),
            ),
        )
        for manifest in source_manifests(dp=1, tp=2)
    )
    targets = target_manifests(dp=1, tp=1)
    _, weight_store = make_weight_store()

    with pytest.raises(ValueError, match="not fully covered"):
        plan_runtime_transfer(sources, targets)
    with pytest.raises(ValueError, match="generation-consistent DP replica"):
        weight_store.prepare_upload(sources)


def test_prepare_upload_collects_dense_tp_shards_across_ep_ranks() -> None:
    tensor = replace(
        tensor_descriptor(),
        tensor_id="layers.2.self_attn.q_proj.weight",
        expert_id=None,
    )
    sources = tuple(
        replace(
            manifest,
            tensors=(tensor,),
            fragments=(
                replace(
                    manifest.fragments[0],
                    tensor_id=tensor.tensor_id,
                    rank=replace(
                        manifest.fragments[0].rank,
                        ep=manifest.fragments[0].rank.tp,
                    ),
                ),
            ),
        )
        for manifest in source_manifests(dp=1, tp=2)
    )
    _, weight_store = make_weight_store()

    plan = weight_store.prepare_upload(sources)

    assert len(plan.operations) == 2
    assert {operation.source.rank.ep for operation in plan.operations} == {0, 1}


def test_weight_group_objects_are_hard_pinned_and_typed() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)

    plan = weight_store.prepare_upload(sources)
    receipts = upload_all(weight_store, plan, sources)
    weight_store.commit(plan, receipts)

    payload_keys = {operation.target.object_key for operation in plan.operations}
    assert {store.configs[key] for key in payload_keys} == {("WEIGHT", True)}
    assert store.configs[plan.control_key] == ("METADATA", True)
    assert store.configs[plan.manifest.manifest_key] == ("METADATA", True)


def test_finalize_upload_session_keeps_committed_revision() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)
    plan = weight_store.prepare_upload(sources)
    receipts = upload_all(weight_store, plan, sources)
    manifest = weight_store.commit(plan, receipts)

    weight_store.finalize_upload_session(plan)
    weight_store.finalize_upload_session(plan)

    assert plan.control_key in store.objects
    assert manifest.manifest_key in store.objects
    assert all(
        operation.target.object_key in store.objects for operation in plan.operations
    )


def test_commit_rejects_incomplete_receipts() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests()
    plan = weight_store.prepare_upload(sources, namespace="default")
    receipts = upload_all(weight_store, plan, sources)

    with pytest.raises(WeightStoreError, match="missing upload receipts"):
        weight_store.commit(plan, receipts[:-1])

    assert plan.manifest.manifest_key not in store.objects


def test_upload_waits_for_complete_payload_before_returning_receipt() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=1)
    plan = weight_store.prepare_upload(sources)
    key = plan.operations[0].target.object_key
    store.processing_keys.add(key)

    with pytest.raises(WeightStoreError, match="payload is not complete"):
        weight_store.upload(plan, sources[0])

    assert key in store.processing_keys
    assert store.removed_keys == []


def test_commit_rechecks_every_payload_after_receipts_are_issued() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)
    plan = weight_store.prepare_upload(sources)
    receipts = upload_all(weight_store, plan, sources)
    store.objects.pop(receipts[0].object_key)

    with pytest.raises(WeightStoreError, match="payload is not complete"):
        weight_store.commit(plan, receipts)

    assert plan.manifest.manifest_key not in store.objects


def test_incomplete_payload_does_not_lock_upload_into_commit_decision() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)
    plan = weight_store.prepare_upload(sources)
    receipts = upload_all(weight_store, plan, sources)
    store.objects.pop(receipts[0].object_key)

    with pytest.raises(WeightStoreError, match="payload is not complete"):
        weight_store.commit(plan, receipts)

    weight_store.abort_upload(plan, receipts)
    assert all(
        operation.target.object_key not in store.objects
        for operation in plan.operations
    )


def test_payload_completion_queries_are_bounded() -> None:
    store, weight_store = make_weight_store(max_ranges_per_request=1)
    sources = source_manifests(dp=1, tp=2)
    plan = weight_store.prepare_upload(sources)
    receipts = upload_all(weight_store, plan, sources)
    store.exist_batch_sizes.clear()

    weight_store.commit(plan, receipts)

    assert store.exist_batch_sizes == [1, 1, 1, 1]


def test_upload_batches_payload_puts_by_range_limit() -> None:
    store, weight_store = make_weight_store(max_ranges_per_request=2)
    sources = source_manifests(dp=1, tp=4)
    combined = RuntimeManifest(
        model_id=sources[0].model_id,
        revision=sources[0].revision,
        instance_id="combined-source",
        tensors=sources[0].tensors,
        fragments=tuple(
            fragment for manifest in sources for fragment in manifest.fragments
        ),
    )
    plan = weight_store.prepare_upload((combined,))

    receipts = weight_store.upload(plan, combined)

    expected_keys = [operation.target.object_key for operation in plan.operations]
    assert max(map(len, store.put_batches)) <= 2
    assert [key for batch in store.put_batches for key in batch] == expected_keys
    assert [receipt.object_key for receipt in receipts] == expected_keys


def test_abort_cleans_the_whole_plan_when_a_receipt_was_lost() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)
    plan = weight_store.prepare_upload(sources)
    receipts = upload_all(weight_store, plan, sources)

    weight_store.abort_upload(plan, receipts[:-1])

    assert not any(
        operation.target.object_key in store.objects for operation in plan.operations
    )
    assert set(store.removed_keys) == {
        operation.target.object_key for operation in plan.operations
    }


def test_abort_does_not_delete_payload_while_manifest_commit_is_processing() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)
    plan = weight_store.prepare_upload(sources)
    receipts = upload_all(weight_store, plan, sources)
    store.processing_keys.add(plan.control_key)

    with pytest.raises(WeightStoreError, match="not complete"):
        weight_store.abort_upload(plan, receipts)

    assert all(
        operation.target.object_key in store.objects for operation in plan.operations
    )
    assert store.removed_keys == []


def test_abort_loses_after_commit_claims_plan_while_manifest_is_processing() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)
    plan = weight_store.prepare_upload(sources)
    receipts = upload_all(weight_store, plan, sources)
    store.processing_keys.add(plan.manifest.manifest_key)

    with pytest.raises(WeightStoreError, match="manifest put failed"):
        weight_store.commit(plan, receipts)
    with pytest.raises(WeightStoreError, match="already chose commit"):
        weight_store.abort_upload(plan, receipts)

    assert all(
        operation.target.object_key in store.objects for operation in plan.operations
    )
    assert store.removed_keys == []


def test_upload_fails_and_cleans_up_when_abort_wins_after_complete_check() -> None:
    store, weight_store = make_weight_store()
    source = source_manifests(dp=1, tp=1)[0]
    plan = weight_store.prepare_upload((source,))
    store.after_batch_is_exist = lambda: weight_store.abort_upload(plan, ())

    with pytest.raises(WeightStoreError, match="already chose abort"):
        weight_store.upload(plan, source)

    assert all(
        operation.target.object_key not in store.objects
        for operation in plan.operations
    )


def test_payload_failure_does_not_publish_manifest() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests()
    plan = weight_store.prepare_upload(sources, namespace="default")
    store.fail_key = plan.operations[1].target.object_key

    with pytest.raises(WeightStoreError, match="batch_put_from"):
        upload_all(weight_store, plan, sources)

    assert plan.manifest.manifest_key not in store.objects
    assert not any("/payload/" in key for key in store.objects)
    assert set(store.removed_keys) == {
        operation.target.object_key for operation in plan.operations
    }
    assert all(store.remove_forces)


def test_upload_surfaces_scalar_batch_put_from_error() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=1)
    plan = weight_store.prepare_upload(sources)
    store.batch_put_from = lambda *args, **kwargs: -17

    with pytest.raises(WeightStoreError, match="batch_put_from failed: -17"):
        upload_all(weight_store, plan, sources)


def test_upload_surfaces_scalar_batch_is_exist_error() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=1)
    plan = weight_store.prepare_upload(sources)
    store.batch_is_exist = lambda *args, **kwargs: -18

    with pytest.raises(WeightStoreError, match="existence check failed: -18"):
        upload_all(weight_store, plan, sources)


def test_commit_is_idempotent_for_the_same_upload_plan() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests()
    plan = weight_store.prepare_upload(sources, namespace="default")
    receipts = upload_all(weight_store, plan, sources)
    first = weight_store.commit(plan, receipts)

    assert weight_store.commit(plan, receipts) == first
    assert all(
        operation.target.object_key in store.objects for operation in plan.operations
    )


def test_commit_conflict_keeps_winner_and_force_cleans_loser_payloads() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)
    winner = weight_store.prepare_upload(sources)
    winner_receipts = upload_all(weight_store, winner, sources)
    weight_store.commit(winner, winner_receipts)
    loser = weight_store.prepare_upload(sources)
    loser_receipts = upload_all(weight_store, loser, sources)

    with pytest.raises(WeightStoreError, match="conflicting weight revision"):
        weight_store.commit(loser, loser_receipts)

    assert weight_store.load_manifest(winner.manifest.manifest_key) == winner.manifest
    assert all(
        operation.target.object_key not in store.objects
        for operation in loser.operations
    )
    assert store.remove_forces[-len(loser.operations) :] == [True] * len(
        loser.operations
    )


def test_finalize_conflicting_commit_keeps_terminal_decision() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)
    winner = weight_store.prepare_upload(sources)
    winner_receipts = upload_all(weight_store, winner, sources)
    winner_manifest = weight_store.commit(winner, winner_receipts)
    loser = weight_store.prepare_upload(sources)
    loser_receipts = upload_all(weight_store, loser, sources)

    with pytest.raises(WeightStoreError, match="conflicting weight revision"):
        weight_store.commit(loser, loser_receipts)
    assert loser.control_key in store.objects

    weight_store.finalize_upload_session(loser)

    assert loser.control_key in store.objects
    assert weight_store.load_manifest(winner_manifest.manifest_key) == winner_manifest
    assert all(
        fragment.object_key in store.objects for fragment in winner_manifest.fragments
    )


def test_conflict_cleanup_preserves_payloads_referenced_by_winner() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)
    plan = weight_store.prepare_upload(sources)
    receipts = upload_all(weight_store, plan, sources)
    winner = replace(plan.manifest, created_at="2026-07-19T00:00:00Z")
    store.objects[plan.manifest.manifest_key] = winner.to_json().encode()

    with pytest.raises(WeightStoreError, match="conflicting weight revision"):
        weight_store.commit(plan, receipts)
    weight_store.finalize_upload_session(plan)

    assert all(fragment.object_key in store.objects for fragment in winner.fragments)
    assert plan.control_key in store.objects


def test_commit_finalize_rejects_late_abort_and_preserves_ready_revision() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)
    plan = weight_store.prepare_upload(sources)
    receipts = upload_all(weight_store, plan, sources)
    manifest = weight_store.commit(plan, receipts)
    weight_store.finalize_upload_session(plan)

    with pytest.raises(WeightStoreError, match="published weight revision"):
        weight_store.abort_upload(plan, receipts)

    assert weight_store.load_manifest(manifest.manifest_key) == manifest
    assert all(fragment.object_key in store.objects for fragment in manifest.fragments)
    assert plan.control_key in store.objects


def test_abort_finalize_rejects_late_upload_and_commit() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)
    plan = weight_store.prepare_upload(sources)
    receipts = upload_all(weight_store, plan, sources)
    weight_store.abort_upload(plan, receipts)
    weight_store.finalize_upload_session(plan)

    with pytest.raises(WeightStoreError, match="already chose abort"):
        weight_store.upload(plan, sources[0])
    with pytest.raises(WeightStoreError, match="already chose abort"):
        weight_store.commit(plan, receipts)

    assert plan.manifest.manifest_key not in store.objects
    assert all(
        operation.target.object_key not in store.objects
        for operation in plan.operations
    )
    assert plan.control_key in store.objects


def test_abort_checks_ready_manifest_when_commit_tombstone_was_lost() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)
    plan = weight_store.prepare_upload(sources)
    receipts = upload_all(weight_store, plan, sources)
    manifest = weight_store.commit(plan, receipts)
    store.objects.pop(plan.control_key)

    with pytest.raises(WeightStoreError, match="published weight revision"):
        weight_store.abort_upload(plan, receipts)

    assert weight_store.load_manifest(manifest.manifest_key) == manifest
    assert all(fragment.object_key in store.objects for fragment in manifest.fragments)


def test_commit_detects_a_concurrent_winner_after_manifest_preflight() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)
    winner = weight_store.prepare_upload(sources)
    loser = weight_store.prepare_upload(sources)
    winner_receipts = upload_all(weight_store, winner, sources)
    loser_receipts = upload_all(weight_store, loser, sources)
    assert len(winner_receipts) == len(loser_receipts)
    store.manifest_race_key = loser.manifest.manifest_key
    store.manifest_race_value = winner.manifest.to_json().encode()

    with pytest.raises(WeightStoreError, match="conflicting weight revision"):
        weight_store.commit(loser, loser_receipts)

    assert weight_store.load_manifest(winner.manifest.manifest_key) == winner.manifest
    assert all(
        operation.target.object_key in store.objects for operation in winner.operations
    )
    assert all(
        operation.target.object_key not in store.objects
        for operation in loser.operations
    )


def test_upload_rejects_stale_runtime_fragment() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)
    plan = weight_store.prepare_upload(sources)
    stale_fragment = RuntimeFragment(
        **{
            **sources[0].fragments[0].__dict__,
            "lease_generation": 2,
        }
    )
    stale_manifest = RuntimeManifest(
        model_id=sources[0].model_id,
        revision=sources[0].revision,
        instance_id=sources[0].instance_id,
        tensors=sources[0].tensors,
        fragments=(stale_fragment,),
    )

    with pytest.raises(WeightStoreError, match="stale source fragment"):
        weight_store.upload(plan, stale_manifest)


def test_upload_rejects_manifest_lease_rollover_before_store_io() -> None:
    store, weight_store = make_weight_store()
    source = replace(
        source_manifests(dp=1, tp=1)[0],
        lease_id="source-lease-1",
    )
    plan = weight_store.prepare_upload((source,))
    current = replace(source, lease_id="source-lease-2")
    store.calls.clear()

    with pytest.raises(WeightStoreError, match="stale source lease"):
        weight_store.upload(plan, current)

    assert store.calls == []


def test_upload_rejects_generation_scoped_fragment_id_rollover() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)
    plan = weight_store.prepare_upload(sources)
    replacement = RuntimeFragment(
        **{
            **sources[0].fragments[0].__dict__,
            "fragment_id": "replacement-fragment",
            "lease_generation": 2,
        }
    )
    current = RuntimeManifest(
        model_id=sources[0].model_id,
        revision=sources[0].revision,
        instance_id=sources[0].instance_id,
        tensors=sources[0].tensors,
        fragments=(replacement,),
    )

    with pytest.raises(WeightStoreError, match="missing planned source fragment"):
        weight_store.upload(plan, current)


def test_register_invalid_params_is_not_treated_as_already_registered() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=1)
    plan = weight_store.prepare_upload(sources)
    store.register_result = -600

    with pytest.raises(WeightStoreError, match="register_buffer failed"):
        weight_store.upload(plan, sources[0])


def test_registration_deduplicates_exact_aliases_with_same_address() -> None:
    aliases = ("a.weight", "b.weight")
    tensors = tuple(
        TensorDescriptor(
            tensor_id=tensor_id,
            global_shape=(8,),
            dtype="uint8",
            itemsize=1,
            layout_fingerprint="test:contiguous:v1",
            partition_dim=None,
        )
        for tensor_id in aliases
    )
    owner = (ctypes.c_ubyte * 8)(*range(8))
    address = ctypes.addressof(owner)
    fragments = tuple(
        RuntimeFragment(
            fragment_id=f"fragment-{tensor.tensor_id}",
            tensor_id=tensor.tensor_id,
            global_offset=(0,),
            local_shape=tensor.global_shape,
            address=address,
            nbytes=8,
            worker_id="source",
            endpoint="source:12345",
            device="cuda:0",
            rank=ParallelRank(),
            lease_generation=1,
            owner=owner,
            aliases=aliases,
        )
        for tensor in tensors
    )
    source = RuntimeManifest(
        model_id="qwen",
        revision="rev",
        instance_id="source",
        tensors=tensors,
        fragments=fragments,
    )
    store, weight_store = make_weight_store()
    plan = weight_store.prepare_upload((source,))

    weight_store.upload(plan, source)

    assert store.register_args == [(address, 8)]


def test_commit_rejects_duplicate_or_forged_receipts() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)
    plan = weight_store.prepare_upload(sources)
    receipts = upload_all(weight_store, plan, sources)

    with pytest.raises(WeightStoreError, match="duplicate upload receipt"):
        weight_store.commit(plan, [*receipts, receipts[0]])

    forged = UploadReceipt(
        fragment_id=receipts[0].fragment_id,
        object_key="forged",
        worker_id=receipts[0].worker_id,
    )
    with pytest.raises(WeightStoreError, match="invalid upload receipt"):
        weight_store.commit(plan, [forged, receipts[1]])


def test_manifest_put_failure_keeps_payloads_for_an_idempotent_retry() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)
    plan = weight_store.prepare_upload(sources)
    receipts = upload_all(weight_store, plan, sources)
    store.fail_key = plan.manifest.manifest_key

    with pytest.raises(WeightStoreError, match="manifest put failed"):
        weight_store.commit(plan, receipts)

    assert all(receipt.object_key in store.objects for receipt in receipts)
    store.fail_key = None
    assert weight_store.commit(plan, receipts) == plan.manifest


def test_commit_recovers_when_manifest_response_is_lost_after_write() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)
    plan = weight_store.prepare_upload(sources)
    receipts = upload_all(weight_store, plan, sources)
    store.fail_after_write_key = plan.manifest.manifest_key

    assert weight_store.commit(plan, receipts) == plan.manifest
    assert weight_store.load_manifest(plan.manifest.manifest_key) == plan.manifest
    assert all(receipt.object_key in store.objects for receipt in receipts)


def test_load_reshards_tp_and_fans_out_dp_with_new_pp_ep_owners() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=2, tp=2)
    upload_plan = weight_store.prepare_upload(sources, namespace="default")
    manifest = weight_store.commit(
        upload_plan, upload_all(weight_store, upload_plan, sources)
    )
    targets = target_manifests(dp=3, tp=4, pp_rank=3, ep_rank=7)

    loaded_manifest = weight_store.load_manifest(manifest.manifest_key)
    load_plan = weight_store.plan_load(loaded_manifest, targets)
    for target in targets:
        weight_store.load(load_plan, target)

    assert load_plan.transfer.total_bytes == 3 * 8
    assert store.range_get_calls == len(targets)
    for target in targets:
        fragment = target.fragments[0]
        start = fragment.global_offset[0]
        assert bytes(fragment.owner) == bytes(range(start, start + fragment.nbytes))
    assert store.register_calls == 2 + len(targets)
    assert store.unregister_calls == 2 + len(targets)
    assert store.registered == set()


def test_store_round_trip_moves_layers_and_experts_across_all_parallel_axes() -> None:
    def make_manifests(*, dp: int, tp: int, source: bool):
        manifests = []
        for layer_id, expert_id, dp_rank, tp_rank in product(
            range(2), range(2), range(dp), range(tp)
        ):
            tensor_index = layer_id * 2 + expert_id
            tensor = TensorDescriptor(
                tensor_id=f"layers.{layer_id}.experts.{expert_id}.w1",
                global_shape=(8,),
                dtype="uint8",
                itemsize=1,
                partition_dim=0,
                layer_id=layer_id,
                expert_id=expert_id,
                layout_fingerprint="sglang:qwen3.5:uint8:test",
            )
            extent = 8 // tp
            offset = tp_rank * extent
            values = (
                range(tensor_index * 16 + offset, tensor_index * 16 + offset + extent)
                if source
                else [255] * extent
            )
            owner = (ctypes.c_ubyte * extent)(*values)
            prefix = "source" if source else "target"
            worker_id = f"{prefix}-l{layer_id}-e{expert_id}-d{dp_rank}-t{tp_rank}"
            fragment = RuntimeFragment(
                fragment_id=f"{worker_id}-fragment",
                tensor_id=tensor.tensor_id,
                global_offset=(offset,),
                local_shape=(extent,),
                address=ctypes.addressof(owner),
                nbytes=extent,
                worker_id=worker_id,
                endpoint=f"{worker_id}:12345",
                device="cuda:0",
                rank=ParallelRank(
                    dp=dp_rank,
                    tp=tp_rank,
                    pp=layer_id if source else 1 - layer_id,
                    ep=expert_id if source else 1 - expert_id,
                ),
                lease_generation=1,
                owner=owner,
            )
            manifests.append(
                RuntimeManifest(
                    model_id="qwen3.5-moe",
                    revision="step-42",
                    instance_id=worker_id,
                    tensors=(tensor,),
                    fragments=(fragment,),
                )
            )
        return tuple(manifests)

    sources = make_manifests(dp=2, tp=2, source=True)
    targets = make_manifests(dp=3, tp=4, source=False)
    store, weight_store = make_weight_store()
    upload_plan = weight_store.prepare_upload(sources)
    manifest = weight_store.commit(
        upload_plan, upload_all(weight_store, upload_plan, sources)
    )
    load_plan = weight_store.plan_load(manifest, targets)

    for target in targets:
        weight_store.load(load_plan, target)

    assert len(upload_plan.operations) == 4 * 2
    assert load_plan.transfer.total_bytes == 4 * 3 * 8
    for target in targets:
        fragment = target.fragments[0]
        tensor = next(
            item for item in target.tensors if item.tensor_id == fragment.tensor_id
        )
        tensor_index = tensor.layer_id * 2 + tensor.expert_id
        begin = tensor_index * 16 + fragment.global_offset[0]
        assert bytes(fragment.owner) == bytes(range(begin, begin + fragment.nbytes))


def test_load_merges_store_fragments_for_larger_target_shards() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=4)
    upload_plan = weight_store.prepare_upload(sources, namespace="default")
    manifest = weight_store.commit(
        upload_plan, upload_all(weight_store, upload_plan, sources)
    )
    targets = target_manifests(dp=1, tp=2, pp_rank=0, ep_rank=0)

    load_plan = weight_store.plan_load(manifest, targets)
    for target in targets:
        weight_store.load(load_plan, target)

    assert len(load_plan.transfer.operations) == 4
    assert bytes(targets[0].fragments[0].owner) == bytes(range(4))
    assert bytes(targets[1].fragments[0].owner) == bytes(range(4, 8))


def test_load_rejects_partial_range_result() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)
    upload_plan = weight_store.prepare_upload(sources)
    manifest = weight_store.commit(
        upload_plan, upload_all(weight_store, upload_plan, sources)
    )
    targets = target_manifests(dp=1, tp=4)
    load_plan = weight_store.plan_load(manifest, targets)
    original = store.get_into_ranges

    def partial(*args, **kwargs):
        results = original(*args, **kwargs)
        results[0][0][0] -= 1
        return results

    store.get_into_ranges = partial
    with pytest.raises(WeightStoreError, match="get_into_ranges failed"):
        weight_store.load(load_plan, targets[0])


def test_load_surfaces_scalar_get_into_ranges_error() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)
    upload_plan = weight_store.prepare_upload(sources)
    manifest = weight_store.commit(
        upload_plan, upload_all(weight_store, upload_plan, sources)
    )
    targets = target_manifests(dp=1, tp=4)
    load_plan = weight_store.plan_load(manifest, targets)
    store.get_into_ranges = lambda *args, **kwargs: -19

    with pytest.raises(WeightStoreError, match="get_into_ranges failed: -19"):
        weight_store.load(load_plan, targets[0])


def test_load_rejects_stale_target_generation() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)
    upload_plan = weight_store.prepare_upload(sources)
    manifest = weight_store.commit(
        upload_plan, upload_all(weight_store, upload_plan, sources)
    )
    targets = target_manifests(dp=1, tp=4)
    load_plan = weight_store.plan_load(manifest, targets)
    stale_fragment = RuntimeFragment(
        **{
            **targets[0].fragments[0].__dict__,
            "lease_generation": 2,
        }
    )
    stale_manifest = RuntimeManifest(
        model_id=targets[0].model_id,
        revision=targets[0].revision,
        instance_id=targets[0].instance_id,
        tensors=targets[0].tensors,
        fragments=(stale_fragment,),
    )

    with pytest.raises(WeightStoreError, match="target executor snapshot mismatch"):
        weight_store.load(load_plan, stale_manifest)


def test_load_rejects_generation_scoped_target_id_rollover() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)
    upload_plan = weight_store.prepare_upload(sources)
    manifest = weight_store.commit(
        upload_plan, upload_all(weight_store, upload_plan, sources)
    )
    targets = target_manifests(dp=1, tp=4)
    load_plan = weight_store.plan_load(manifest, targets)
    replacement = RuntimeFragment(
        **{
            **targets[0].fragments[0].__dict__,
            "fragment_id": "replacement-target-fragment",
            "lease_generation": 2,
        }
    )
    current = RuntimeManifest(
        model_id=targets[0].model_id,
        revision=targets[0].revision,
        instance_id=targets[0].instance_id,
        tensors=targets[0].tensors,
        fragments=(replacement,),
    )

    with pytest.raises(WeightStoreError, match="target executor snapshot mismatch"):
        weight_store.load(load_plan, current)


def test_load_rejects_worker_and_generation_rollover_instead_of_succeeding_noop() -> (
    None
):
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)
    upload_plan = weight_store.prepare_upload(sources)
    manifest = weight_store.commit(
        upload_plan, upload_all(weight_store, upload_plan, sources)
    )
    targets = target_manifests(dp=1, tp=4)
    load_plan = weight_store.plan_load(manifest, targets)
    replacement = replace(
        targets[0].fragments[0],
        fragment_id="replacement-target-fragment",
        worker_id="replacement-target-worker",
        lease_generation=2,
    )
    current = replace(
        targets[0],
        instance_id="replacement-target-instance",
        generation=None,
        fragments=(replacement,),
    )

    with pytest.raises(WeightStoreError, match="target executor snapshot mismatch"):
        weight_store.load(load_plan, current)


def test_upload_unregisters_every_buffer_without_deleting_unowned_payloads() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)
    combined = RuntimeManifest(
        model_id=sources[0].model_id,
        revision=sources[0].revision,
        instance_id="combined-source",
        tensors=sources[0].tensors,
        fragments=tuple(
            fragment for manifest in sources for fragment in manifest.fragments
        ),
    )
    plan = weight_store.prepare_upload((combined,))
    first_address = combined.fragments[0].address
    store.unregister_results[first_address] = -9

    with pytest.raises(WeightStoreError, match="unregister_buffer failed"):
        weight_store.upload(plan, combined)

    assert store.unregister_calls == 2
    assert all(
        operation.target.object_key in store.objects for operation in plan.operations
    )
    assert store.remove_forces == []


def test_unregister_failure_does_not_mask_payload_transfer_failure() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)
    combined = RuntimeManifest(
        model_id=sources[0].model_id,
        revision=sources[0].revision,
        instance_id="combined-source",
        tensors=sources[0].tensors,
        fragments=tuple(
            fragment for manifest in sources for fragment in manifest.fragments
        ),
    )
    plan = weight_store.prepare_upload((combined,))
    store.fail_key = plan.operations[1].target.object_key
    store.unregister_results[combined.fragments[0].address] = -9

    with pytest.raises(WeightStoreError) as error:
        weight_store.upload(plan, combined)

    assert "batch_put_from failed" in str(error.value)
    assert "unregister_buffer failed" in str(error.value)
    assert store.unregister_calls == 2
    assert plan.operations[0].target.object_key in store.objects
    assert store.remove_forces == []


def test_unregister_attempts_every_buffer_when_cleanup_raises() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)
    combined = RuntimeManifest(
        model_id=sources[0].model_id,
        revision=sources[0].revision,
        instance_id="combined-source",
        tensors=sources[0].tensors,
        fragments=tuple(
            fragment for manifest in sources for fragment in manifest.fragments
        ),
    )
    plan = weight_store.prepare_upload((combined,))
    first, second = combined.fragments
    store.fail_key = plan.operations[1].target.object_key
    store.unregister_exceptions[first.address] = RuntimeError("unregister exploded")
    store.unregister_results[second.address] = -9

    with pytest.raises(WeightStoreError) as error:
        weight_store.upload(plan, combined)

    assert "batch_put_from failed" in str(error.value)
    assert "unregister exploded" in str(error.value)
    assert "-9" in str(error.value)
    assert store.unregister_addresses == [second.address, first.address]


def test_load_chunks_large_ranges_to_bound_host_staging() -> None:
    """GPU range GET uses a host temporary per range, so each range is capped."""
    store, weight_store = make_weight_store(max_range_bytes=2)
    sources = source_manifests(dp=1, tp=1)
    upload_plan = weight_store.prepare_upload(sources)
    manifest = weight_store.commit(
        upload_plan, upload_all(weight_store, upload_plan, sources)
    )
    targets = target_manifests(dp=1, tp=1)

    load_plan = weight_store.plan_load(manifest, targets)
    weight_store.load(load_plan, targets[0])

    assert store.range_sizes == [2, 2, 2, 2]
    assert bytes(targets[0].fragments[0].owner) == bytes(range(8))


def test_load_expands_strided_ranges_in_bounded_requests() -> None:
    tensor = TensorDescriptor(
        tensor_id="layers.0.mlp.down_proj.weight",
        global_shape=(5, 8),
        dtype="uint8",
        itemsize=1,
        partition_dim=1,
        layer_id=0,
        layout_fingerprint="sglang:qwen3.5:uint8:test",
    )

    def make_manifests(tp: int, prefix: str, *, source: bool):
        manifests = []
        extent = tensor.global_shape[1] // tp
        for tp_rank in range(tp):
            values = []
            for row in range(tensor.global_shape[0]):
                begin = row * tensor.global_shape[1] + tp_rank * extent
                values.extend(range(begin, begin + extent))
            if not source:
                values = [255] * len(values)
            owner = (ctypes.c_ubyte * len(values))(*values)
            worker_id = f"{prefix}-t{tp_rank}"
            fragment = RuntimeFragment(
                fragment_id=f"{worker_id}-fragment",
                tensor_id=tensor.tensor_id,
                global_offset=(0, tp_rank * extent),
                local_shape=(tensor.global_shape[0], extent),
                address=ctypes.addressof(owner),
                nbytes=prod((tensor.global_shape[0], extent)),
                worker_id=worker_id,
                endpoint=f"{worker_id}:12345",
                device="cuda:0",
                rank=ParallelRank(tp=tp_rank),
                lease_generation=1,
                owner=owner,
            )
            manifests.append(
                RuntimeManifest(
                    model_id="qwen3.5-0.8b",
                    revision="step-42",
                    instance_id=worker_id,
                    tensors=(tensor,),
                    fragments=(fragment,),
                )
            )
        return tuple(manifests)

    sources = make_manifests(2, "source", source=True)
    targets = make_manifests(4, "target", source=False)
    store, weight_store = make_weight_store(max_ranges_per_request=2)
    upload_plan = weight_store.prepare_upload(sources)
    manifest = weight_store.commit(
        upload_plan, upload_all(weight_store, upload_plan, sources)
    )
    load_plan = weight_store.plan_load(manifest, targets)

    for target in targets:
        weight_store.load(load_plan, target)

    for tp_rank, target in enumerate(targets):
        expected = []
        for row in range(tensor.global_shape[0]):
            begin = row * tensor.global_shape[1] + tp_rank * 2
            expected.extend(range(begin, begin + 2))
        assert bytes(target.fragments[0].owner) == bytes(expected)
    assert max(store.range_batch_sizes) <= 2
    assert store.range_get_calls == 12


def test_load_batches_multiple_local_target_buffers_in_one_request() -> None:
    tensors = tuple(
        TensorDescriptor(
            tensor_id=f"layers.{index}.norm.weight",
            global_shape=(4,),
            dtype="uint8",
            itemsize=1,
            partition_dim=None,
            layer_id=index,
            layout_fingerprint="sglang:qwen3.5:uint8:test",
        )
        for index in range(4)
    )

    def make_manifest(prefix: str, *, source: bool) -> RuntimeManifest:
        fragments = []
        for index, tensor in enumerate(tensors):
            values = range(index * 4, index * 4 + 4) if source else [255] * 4
            owner = (ctypes.c_ubyte * 4)(*values)
            fragments.append(
                RuntimeFragment(
                    fragment_id=f"{prefix}-{index}",
                    tensor_id=tensor.tensor_id,
                    global_offset=(0,),
                    local_shape=(4,),
                    address=ctypes.addressof(owner),
                    nbytes=4,
                    worker_id=prefix,
                    endpoint=f"{prefix}:12345",
                    device="cuda:0",
                    rank=ParallelRank(),
                    lease_generation=1,
                    owner=owner,
                )
            )
        return RuntimeManifest(
            model_id="qwen3.5-0.8b",
            revision="step-42",
            instance_id=prefix,
            tensors=tensors,
            fragments=tuple(fragments),
        )

    source = make_manifest("source", source=True)
    target = make_manifest("target", source=False)
    store, weight_store = make_weight_store()
    upload_plan = weight_store.prepare_upload((source,))
    manifest = weight_store.commit(
        upload_plan, upload_all(weight_store, upload_plan, (source,))
    )
    load_plan = weight_store.plan_load(manifest, (target,))

    weight_store.load(load_plan, target)

    assert store.range_get_calls == 1
    assert store.range_batch_sizes == [4]
    for index, fragment in enumerate(target.fragments):
        assert bytes(fragment.owner) == bytes(range(index * 4, index * 4 + 4))


def test_one_runtime_manifest_can_execute_multiple_rank_subplans() -> None:
    store, weight_store = make_weight_store()
    sources = source_manifests(dp=1, tp=2)
    upload_plan = weight_store.prepare_upload(sources)
    manifest = weight_store.commit(
        upload_plan, upload_all(weight_store, upload_plan, sources)
    )
    target_ranks = target_manifests(dp=1, tp=2)
    target = RuntimeManifest(
        model_id=target_ranks[0].model_id,
        revision=target_ranks[0].revision,
        instance_id="combined-target",
        tensors=target_ranks[0].tensors,
        fragments=tuple(
            replace(fragment, worker_id="combined-target")
            for rank_manifest in target_ranks
            for fragment in rank_manifest.fragments
        ),
    )

    load_plan = weight_store.plan_load(manifest, (target,))
    weight_store.load(load_plan, target)

    assert store.range_get_calls == 1
    assert [bytes(fragment.owner) for fragment in target.fragments] == [
        bytes(range(4)),
        bytes(range(4, 8)),
    ]
