from __future__ import annotations

from dataclasses import replace
from itertools import product
from math import prod

import pytest

from mooncake.weight_transfer.manifest import (
    ParallelRank,
    RuntimeFragment,
    RuntimeManifest,
    TensorDescriptor,
)
from mooncake.weight_transfer.planner import (
    CopyRange,
    TransferPlan,
    TransferRegion,
    _complete_runtime_source_replicas,
    plan_runtime_transfer,
    plan_runtime_transfer_to_local_target,
    resolve_executor_plans,
)


def descriptor(
    *,
    global_shape: tuple[int, ...] = (8, 4),
    partition_dim: int = 0,
    fingerprint: str = "sglang:qwen3.5:bf16:v1",
) -> TensorDescriptor:
    return TensorDescriptor(
        tensor_id="layers.2.experts.3.w1",
        global_shape=global_shape,
        dtype="bfloat16",
        itemsize=2,
        partition_dim=partition_dim,
        layer_id=2,
        expert_id=3,
        layout_fingerprint=fingerprint,
    )


def tp_manifests(
    *,
    tp: int,
    dp: int = 1,
    pp_rank: int,
    ep_rank: int,
    address_base: int,
    worker_prefix: str,
    tensor: TensorDescriptor | None = None,
) -> tuple[RuntimeManifest, ...]:
    tensor = tensor or descriptor()
    dim = tensor.partition_dim
    assert dim is not None
    assert tensor.global_shape[dim] % tp == 0
    extent = tensor.global_shape[dim] // tp
    manifests = []
    for dp_rank, tp_rank in product(range(dp), range(tp)):
        local_shape = list(tensor.global_shape)
        local_shape[dim] = extent
        global_offset = [0] * len(tensor.global_shape)
        global_offset[dim] = tp_rank * extent
        worker_id = f"{worker_prefix}-d{dp_rank}-t{tp_rank}"
        fragment = RuntimeFragment(
            fragment_id=f"{worker_id}-fragment",
            tensor_id=tensor.tensor_id,
            global_offset=tuple(global_offset),
            local_shape=tuple(local_shape),
            address=address_base + (dp_rank * tp + tp_rank) * 0x1000,
            nbytes=extent
            * (tensor.global_shape[1 - dim] if len(tensor.global_shape) == 2 else 1)
            * tensor.itemsize,
            worker_id=worker_id,
            endpoint=f"{worker_id}:12345",
            rank=ParallelRank(dp=dp_rank, tp=tp_rank, pp=pp_rank, ep=ep_rank),
            lease_generation=1,
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


def operation_for_target(plan, tp_rank: int, dp_rank: int = 0):
    return [
        operation
        for operation in plan.operations
        if operation.target.rank.tp == tp_rank and operation.target.rank.dp == dp_rank
    ]


def distribute_tp_shards_across_ep_ranks(
    manifests: tuple[RuntimeManifest, ...],
) -> tuple[RuntimeManifest, ...]:
    return tuple(
        replace(
            manifest,
            fragments=tuple(
                replace(
                    fragment,
                    rank=replace(fragment.rank, ep=fragment.rank.tp),
                )
                for fragment in manifest.fragments
            ),
        )
        for manifest in manifests
    )


class CountingFragment:
    def __init__(self, fragment: RuntimeFragment, accesses: list[int]) -> None:
        self._fragment = fragment
        self._accesses = accesses

    @property
    def tensor_id(self) -> str:
        self._accesses[0] += 1
        return self._fragment.tensor_id

    def __getattr__(self, name: str):
        return getattr(self._fragment, name)


def test_copy_range_rejects_fragment_overflow_and_non_integer_offsets() -> None:
    source = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
    )[0].fragments[0]
    target = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x20000,
        worker_prefix="target",
    )[0].fragments[0]

    with pytest.raises(ValueError, match="source fragment"):
        CopyRange(
            tensor_id=source.tensor_id,
            source=source,
            target=target,
            source_offset=1,
            target_offset=0,
            nbytes=source.nbytes,
        )

    with pytest.raises(ValueError, match="target fragment"):
        CopyRange(
            tensor_id=source.tensor_id,
            source=source,
            target=target,
            source_offset=0,
            target_offset=0,
            nbytes=1,
            repeat=2,
            source_stride=1,
            target_stride=target.nbytes,
        )

    with pytest.raises(ValueError, match="integer"):
        CopyRange(
            tensor_id=source.tensor_id,
            source=source,
            target=target,
            source_offset=False,
            target_offset=0,
            nbytes=1,
        )

    with pytest.raises(ValueError, match="tensor mismatch"):
        CopyRange(
            tensor_id="different.tensor",
            source=source,
            target=target,
            source_offset=0,
            target_offset=0,
            nbytes=1,
        )


def nd_fragment(
    *,
    fragment_id: str,
    global_offset: tuple[int, ...],
    local_shape: tuple[int, ...],
    address: int,
) -> RuntimeFragment:
    return RuntimeFragment(
        fragment_id=fragment_id,
        tensor_id="layers.2.experts.w1",
        global_offset=global_offset,
        local_shape=local_shape,
        address=address,
        nbytes=2 * prod(local_shape),
        worker_id=fragment_id,
        endpoint=f"{fragment_id}:12345",
        rank=ParallelRank(),
        lease_generation=1,
    )


@pytest.mark.parametrize(
    (
        "target_offset",
        "target_shape",
        "overlap_offset",
        "overlap_shape",
        "source_base_offset",
        "target_base_offset",
        "outer_loop_counts",
        "source_strides",
        "target_strides",
        "inner_bytes",
    ),
    [
        (
            (0, 2, 0),
            (4, 3, 8),
            (1, 2, 0),
            (2, 3, 8),
            32,
            48,
            (2, 3),
            (96, 16),
            (48, 16),
            16,
        ),
        (
            (0, 0, 4),
            (4, 6, 4),
            (1, 0, 4),
            (2, 6, 4),
            8,
            48,
            (2, 6),
            (96, 16),
            (48, 8),
            8,
        ),
    ],
)
def test_transfer_region_describes_cross_dim_logical_overlap(
    target_offset: tuple[int, ...],
    target_shape: tuple[int, ...],
    overlap_offset: tuple[int, ...],
    overlap_shape: tuple[int, ...],
    source_base_offset: int,
    target_base_offset: int,
    outer_loop_counts: tuple[int, ...],
    source_strides: tuple[int, ...],
    target_strides: tuple[int, ...],
    inner_bytes: int,
) -> None:
    source = nd_fragment(
        fragment_id="source",
        global_offset=(1, 0, 0),
        local_shape=(2, 6, 8),
        address=0x10000,
    )
    target = nd_fragment(
        fragment_id="target",
        global_offset=target_offset,
        local_shape=target_shape,
        address=0x20000,
    )

    region = TransferRegion(
        tensor_id=source.tensor_id,
        source=source,
        target=target,
        overlap_offset=overlap_offset,
        overlap_shape=overlap_shape,
        source_base_offset=source_base_offset,
        target_base_offset=target_base_offset,
        inner_bytes=inner_bytes,
        outer_loop_counts=outer_loop_counts,
        source_strides=source_strides,
        target_strides=target_strides,
    )

    assert region.overlap_offset == overlap_offset
    assert region.overlap_shape == overlap_shape
    assert region.source_base_offset == source_base_offset
    assert region.target_base_offset == target_base_offset
    assert region.inner_bytes == inner_bytes
    assert region.outer_loop_counts == outer_loop_counts
    assert region.source_strides == source_strides
    assert region.target_strides == target_strides
    assert region.segment_count == prod(outer_loop_counts)
    assert region.total_bytes == prod(overlap_shape) * 2
    assert len(tuple(region.iter_segments())) == region.segment_count

    plan = TransferPlan(
        model_id="qwen3.5-moe",
        revision="step-42",
        operations=(region,),
    )
    assert plan.regions == (region,)


def test_transfer_region_mixed_radix_iteration_and_nd_bounds() -> None:
    source = nd_fragment(
        fragment_id="source",
        global_offset=(1, 0, 0),
        local_shape=(2, 6, 8),
        address=0x10000,
    )
    target = nd_fragment(
        fragment_id="target",
        global_offset=(0, 2, 0),
        local_shape=(4, 3, 8),
        address=0x20000,
    )
    region = TransferRegion(
        tensor_id=source.tensor_id,
        source=source,
        target=target,
        overlap_offset=(1, 2, 0),
        overlap_shape=(2, 3, 8),
        source_base_offset=32,
        target_base_offset=48,
        inner_bytes=16,
        outer_loop_counts=(2, 3),
        source_strides=(96, 16),
        target_strides=(48, 16),
    )

    assert tuple(region.iter_segments()) == (
        (32, 48, 16),
        (48, 64, 16),
        (64, 80, 16),
        (128, 96, 16),
        (144, 112, 16),
        (160, 128, 16),
    )

    with pytest.raises(ValueError, match="source fragment"):
        replace(region, source_strides=(193, 16))


def test_runtime_plan_records_explicit_noop_source_executors() -> None:
    sources = tp_manifests(
        tp=2,
        dp=2,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
    )
    targets = tp_manifests(
        tp=4,
        dp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x40000,
        worker_prefix="target",
    )

    plan = plan_runtime_transfer(sources, targets)
    noop = [executor for executor in plan.source_executors if executor.rank.dp == 1]

    assert len(plan.source_executors) == len(sources)
    assert len(plan.target_executors) == len(targets)
    assert len(noop) == 2
    assert all(not executor.operation_indices for executor in noop)


def test_runtime_plan_deduplicates_identical_physical_alias_copies() -> None:
    tensors = tuple(
        replace(
            descriptor(),
            tensor_id=tensor_id,
            global_shape=(32,),
            partition_dim=None,
            layer_id=None,
            expert_id=None,
            layout_fingerprint="sglang:qwen3.5:vocab-parallel:v1",
        )
        for tensor_id in ("embed_tokens.weight", "lm_head.weight")
    )

    def manifest(prefix: str, address: int) -> RuntimeManifest:
        worker_id = f"{prefix}-tp0"
        return RuntimeManifest(
            model_id="qwen3.5-0.8b",
            revision="step-42",
            instance_id=worker_id,
            tensors=tensors,
            fragments=tuple(
                RuntimeFragment(
                    fragment_id=f"{worker_id}-{tensor.tensor_id}",
                    tensor_id=tensor.tensor_id,
                    global_offset=(0,),
                    local_shape=(32,),
                    address=address,
                    nbytes=64,
                    worker_id=worker_id,
                    endpoint=f"{worker_id}:12345",
                    rank=ParallelRank(),
                    lease_generation=1,
                )
                for tensor in tensors
            ),
        )

    plan = plan_runtime_transfer(
        (manifest("source", 0x10000),),
        (manifest("target", 0x20000),),
    )

    assert len(plan.operations) == 1
    assert plan.total_bytes == 64


def test_runtime_plan_rejects_target_aliases_with_distinct_source_storage() -> None:
    tensors = tuple(
        replace(
            descriptor(),
            tensor_id=tensor_id,
            global_shape=(32,),
            partition_dim=None,
            layer_id=None,
            expert_id=None,
            layout_fingerprint="sglang:qwen3.5:vocab-parallel:v1",
        )
        for tensor_id in ("embed_tokens.weight", "lm_head.weight")
    )
    source_worker = "source-tp0"
    target_worker = "target-tp0"
    source = RuntimeManifest(
        model_id="qwen3.5-0.8b",
        revision="step-42",
        instance_id=source_worker,
        tensors=tensors,
        fragments=tuple(
            RuntimeFragment(
                fragment_id=f"{source_worker}-{tensor.tensor_id}",
                tensor_id=tensor.tensor_id,
                global_offset=(0,),
                local_shape=(32,),
                address=0x10000 + index * 0x1000,
                nbytes=64,
                worker_id=source_worker,
                endpoint=f"{source_worker}:12345",
                rank=ParallelRank(),
                lease_generation=1,
            )
            for index, tensor in enumerate(tensors)
        ),
    )
    target = RuntimeManifest(
        model_id="qwen3.5-0.8b",
        revision="step-42",
        instance_id=target_worker,
        tensors=tensors,
        fragments=tuple(
            RuntimeFragment(
                fragment_id=f"{target_worker}-{tensor.tensor_id}",
                tensor_id=tensor.tensor_id,
                global_offset=(0,),
                local_shape=(32,),
                address=0x20000,
                nbytes=64,
                worker_id=target_worker,
                endpoint=f"{target_worker}:12345",
                rank=ParallelRank(),
                lease_generation=1,
            )
            for tensor in tensors
        ),
    )

    with pytest.raises(ValueError, match="conflicting target physical range"):
        plan_runtime_transfer((source,), (target,))


def test_runtime_plan_deduplicates_declared_target_alias_across_pp_sources() -> None:
    tensors = tuple(
        replace(
            descriptor(),
            tensor_id=tensor_id,
            global_shape=(32,),
            partition_dim=None,
            layer_id=None,
            expert_id=None,
            layout_fingerprint="sglang:qwen3.5:vocab-parallel:v1",
        )
        for tensor_id in ("embed_tokens.weight", "lm_head.weight")
    )
    aliases = ("lm_head.weight", "model.embed_tokens.weight")
    sources = tuple(
        RuntimeManifest(
            model_id="qwen3.5-0.8b",
            revision="step-42",
            instance_id=f"source-pp{pp_rank}",
            tensors=(tensor,),
            fragments=(
                RuntimeFragment(
                    fragment_id=f"source-pp{pp_rank}-{tensor.tensor_id}",
                    tensor_id=tensor.tensor_id,
                    global_offset=(0,),
                    local_shape=(32,),
                    address=0x10000 + pp_rank * 0x1000,
                    nbytes=64,
                    worker_id=f"source-pp{pp_rank}",
                    endpoint=f"source-pp{pp_rank}:12345",
                    rank=ParallelRank(pp=pp_rank),
                    lease_generation=1,
                ),
            ),
        )
        for pp_rank, tensor in enumerate(tensors)
    )
    target_worker = "target-pp0"
    target = RuntimeManifest(
        model_id="qwen3.5-0.8b",
        revision="step-42",
        instance_id=target_worker,
        tensors=tensors,
        fragments=tuple(
            RuntimeFragment(
                fragment_id=f"{target_worker}-{tensor.tensor_id}",
                tensor_id=tensor.tensor_id,
                global_offset=(0,),
                local_shape=(32,),
                address=0x20000,
                nbytes=64,
                worker_id=target_worker,
                endpoint=f"{target_worker}:12345",
                rank=ParallelRank(),
                lease_generation=1,
                aliases=aliases,
            )
            for tensor in tensors
        ),
    )

    plan = plan_runtime_transfer_to_local_target(sources, target)

    assert len(plan.operations) == 1
    assert plan.operations[0].tensor_id == "embed_tokens.weight"
    assert plan.operations[0].source.worker_id == "source-pp0"


def test_runtime_plan_rejects_partially_overlapping_target_physical_ranges() -> None:
    tensors = tuple(
        replace(
            descriptor(),
            tensor_id=tensor_id,
            global_shape=(32,),
            partition_dim=None,
            layer_id=None,
            expert_id=None,
            layout_fingerprint="sglang:qwen3.5:vocab-parallel:v1",
        )
        for tensor_id in ("embed_tokens.weight", "lm_head.weight")
    )

    def manifest(
        prefix: str, addresses: tuple[int, int], *, target: bool
    ) -> RuntimeManifest:
        worker_id = f"{prefix}-tp0"
        return RuntimeManifest(
            model_id="qwen3.5-0.8b",
            revision="step-42",
            instance_id=worker_id,
            tensors=tensors,
            fragments=tuple(
                RuntimeFragment(
                    fragment_id=f"{worker_id}-{tensor.tensor_id}",
                    tensor_id=tensor.tensor_id,
                    global_offset=(0,),
                    local_shape=(32,),
                    address=addresses[index],
                    nbytes=64,
                    worker_id=worker_id,
                    endpoint=f"{worker_id}:12345",
                    rank=ParallelRank(),
                    lease_generation=1,
                )
                for index, tensor in enumerate(tensors)
            ),
        )

    source = manifest("source", (0x10000, 0x11000), target=False)
    target = manifest("target", (0x20000, 0x20020), target=True)

    with pytest.raises(ValueError, match="conflicting target physical range"):
        plan_runtime_transfer((source,), (target,))


def test_runtime_plan_rejects_fragment_ids_reused_by_different_workers() -> None:
    sources = list(
        tp_manifests(
            tp=2,
            pp_rank=0,
            ep_rank=0,
            address_base=0x10000,
            worker_prefix="source",
        )
    )
    duplicate = replace(
        sources[1].fragments[0],
        fragment_id=sources[0].fragments[0].fragment_id,
    )
    sources[1] = replace(sources[1], fragments=(duplicate,))
    targets = tp_manifests(
        tp=2,
        pp_rank=0,
        ep_rank=0,
        address_base=0x40000,
        worker_prefix="target",
    )

    with pytest.raises(ValueError, match="duplicate source fragment_id"):
        plan_runtime_transfer(tuple(sources), targets)


def test_runtime_plan_rejects_target_omitting_a_source_tensor() -> None:
    primary = descriptor()
    secondary = replace(
        primary,
        tensor_id="layers.3.experts.4.w1",
        layer_id=3,
        expert_id=4,
    )
    source = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
        tensor=primary,
    )[0]
    secondary_fragment = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x20000,
        worker_prefix="secondary",
        tensor=secondary,
    )[0].fragments[0]
    secondary_fragment = replace(
        secondary_fragment,
        worker_id=source.fragments[0].worker_id,
        endpoint=source.fragments[0].endpoint,
    )
    source = replace(
        source,
        tensors=(primary, secondary),
        fragments=(*source.fragments, secondary_fragment),
    )
    targets = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x30000,
        worker_prefix="target",
        tensor=primary,
    )

    with pytest.raises(ValueError, match="target manifests are missing tensors"):
        plan_runtime_transfer((source,), targets)


def test_runtime_plan_rejects_tensor_missing_from_one_target_dp_replica() -> None:
    primary = descriptor()
    secondary = replace(
        primary,
        tensor_id="layers.3.experts.4.w1",
        layer_id=3,
        expert_id=4,
    )

    def combined_manifest(*, dp_rank: int, include_secondary: bool, prefix: str):
        primary_manifest = tp_manifests(
            tp=1,
            dp=dp_rank + 1,
            pp_rank=0,
            ep_rank=0,
            address_base=0x10000 + dp_rank * 0x10000,
            worker_prefix=prefix,
            tensor=primary,
        )[dp_rank]
        if not include_secondary:
            return primary_manifest
        secondary_fragment = tp_manifests(
            tp=1,
            pp_rank=0,
            ep_rank=0,
            address_base=0x50000 + dp_rank * 0x10000,
            worker_prefix=f"{prefix}-secondary",
            tensor=secondary,
        )[0].fragments[0]
        secondary_fragment = replace(
            secondary_fragment,
            worker_id=primary_manifest.fragments[0].worker_id,
            endpoint=primary_manifest.fragments[0].endpoint,
            rank=primary_manifest.fragments[0].rank,
        )
        return replace(
            primary_manifest,
            tensors=(primary, secondary),
            fragments=(*primary_manifest.fragments, secondary_fragment),
        )

    sources = (combined_manifest(dp_rank=0, include_secondary=True, prefix="source"),)
    targets = (
        combined_manifest(dp_rank=0, include_secondary=True, prefix="target"),
        combined_manifest(dp_rank=1, include_secondary=False, prefix="target"),
    )

    with pytest.raises(ValueError, match="target tensor is not fully covered"):
        plan_runtime_transfer(sources, targets)


def test_runtime_plan_rejects_target_tp_gap() -> None:
    sources = tp_manifests(
        tp=2,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
    )
    targets = tp_manifests(
        tp=2,
        pp_rank=0,
        ep_rank=0,
        address_base=0x30000,
        worker_prefix="target",
    )

    with pytest.raises(ValueError, match="target tensor is not fully covered"):
        plan_runtime_transfer(sources, targets[:1])


def test_executor_snapshot_rejects_an_entire_missing_rank() -> None:
    sources = tp_manifests(
        tp=2,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
    )
    target_ranks = tp_manifests(
        tp=2,
        pp_rank=0,
        ep_rank=0,
        address_base=0x30000,
        worker_prefix="target",
    )
    worker_id = "target-worker"
    combined_target = RuntimeManifest(
        model_id=target_ranks[0].model_id,
        revision=target_ranks[0].revision,
        instance_id=worker_id,
        tensors=target_ranks[0].tensors,
        fragments=tuple(
            replace(
                manifest.fragments[0],
                worker_id=worker_id,
                endpoint=f"{worker_id}:12345",
            )
            for manifest in target_ranks
        ),
    )
    plan = plan_runtime_transfer(sources, (combined_target,))
    missing_rank = replace(combined_target, fragments=combined_target.fragments[:1])

    with pytest.raises(ValueError, match="target executor snapshot mismatch"):
        resolve_executor_plans(plan, missing_rank, "target")


def test_tp2_to_tp4_splits_source_fragments_without_full_tensor() -> None:
    source = tp_manifests(
        tp=2,
        pp_rank=1,
        ep_rank=1,
        address_base=0x10000,
        worker_prefix="source",
    )
    target = tp_manifests(
        tp=4,
        pp_rank=2,
        ep_rank=3,
        address_base=0x20000,
        worker_prefix="target",
    )

    plan = plan_runtime_transfer(source, target)

    assert len(plan.operations) == 4
    assert [
        (op.source_offset, op.target_offset, op.nbytes)
        for op in operation_for_target(plan, 0)
    ] == [(0, 0, 16)]
    assert [
        (op.source_offset, op.target_offset, op.nbytes)
        for op in operation_for_target(plan, 1)
    ] == [(16, 0, 16)]
    assert operation_for_target(plan, 0)[0].source.rank.pp == 1
    assert operation_for_target(plan, 0)[0].target.rank.pp == 2
    assert operation_for_target(plan, 0)[0].source.rank.ep == 1
    assert operation_for_target(plan, 0)[0].target.rank.ep == 3


def test_tp4_to_tp2_merges_into_non_overlapping_target_offsets() -> None:
    source = tp_manifests(
        tp=4,
        pp_rank=2,
        ep_rank=3,
        address_base=0x10000,
        worker_prefix="source",
    )
    target = tp_manifests(
        tp=2,
        pp_rank=1,
        ep_rank=1,
        address_base=0x20000,
        worker_prefix="target",
    )

    plan = plan_runtime_transfer(source, target)

    assert [
        (op.source_offset, op.target_offset, op.nbytes)
        for op in operation_for_target(plan, 0)
    ] == [
        (0, 0, 16),
        (0, 16, 16),
    ]


def test_partition_dim_one_generates_row_ranges() -> None:
    tensor = descriptor(global_shape=(2, 8), partition_dim=1)
    source = tp_manifests(
        tp=2,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
        tensor=tensor,
    )
    target = tp_manifests(
        tp=4,
        pp_rank=0,
        ep_rank=0,
        address_base=0x20000,
        worker_prefix="target",
        tensor=tensor,
    )

    plan = plan_runtime_transfer(source, target)

    operations = operation_for_target(plan, 0)
    assert len(operations) == 1
    assert operations[0].repeat == 2
    assert operations[0].source_stride == 8
    assert operations[0].target_stride == 4
    assert list(operations[0].iter_segments()) == [
        (0, 0, 4),
        (8, 4, 4),
    ]


def test_large_inner_axis_partition_uses_compact_strided_ranges() -> None:
    tensor = descriptor(global_shape=(8192, 8192), partition_dim=1)
    source = tp_manifests(
        tp=2,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
        tensor=tensor,
    )
    target = tp_manifests(
        tp=4,
        pp_rank=0,
        ep_rank=0,
        address_base=0x20000,
        worker_prefix="target",
        tensor=tensor,
    )

    plan = plan_runtime_transfer(source, target)

    assert len(plan.operations) == 4
    assert {operation.repeat for operation in plan.operations} == {8192}
    assert plan.total_bytes == 8192 * 8192 * 2


def test_all_parallel_axes_change_in_one_plan() -> None:
    source = tp_manifests(
        tp=2,
        dp=2,
        pp_rank=1,
        ep_rank=1,
        address_base=0x10000,
        worker_prefix="source",
    )
    target = tp_manifests(
        tp=4,
        dp=3,
        pp_rank=2,
        ep_rank=3,
        address_base=0x40000,
        worker_prefix="target",
    )

    plan = plan_runtime_transfer(source, target)

    assert len(plan.operations) == 12
    assert {op.target.rank.dp for op in plan.operations} == {0, 1, 2}
    assert {op.target.rank.pp for op in plan.operations} == {2}
    assert {op.target.rank.ep for op in plan.operations} == {3}
    assert {op.source.rank.pp for op in plan.operations} == {1}
    assert {op.source.rank.ep for op in plan.operations} == {1}
    assert {op.source.rank.dp for op in operation_for_target(plan, 0, 0)} == {0}
    assert {op.source.rank.dp for op in operation_for_target(plan, 0, 1)} == {1}
    assert {op.source.rank.dp for op in operation_for_target(plan, 0, 2)} == {0}


def test_dense_source_tp_coverage_can_span_ep_ranks() -> None:
    tensor = replace(
        descriptor(),
        tensor_id="layers.2.self_attn.q_proj.weight",
        expert_id=None,
    )
    source = distribute_tp_shards_across_ep_ranks(
        tp_manifests(
            tp=4,
            pp_rank=0,
            ep_rank=0,
            address_base=0x10000,
            worker_prefix="source",
            tensor=tensor,
        )
    )
    target = tp_manifests(
        tp=2,
        pp_rank=0,
        ep_rank=0,
        address_base=0x40000,
        worker_prefix="target",
        tensor=tensor,
    )

    plan = plan_runtime_transfer(source, target)

    assert len(plan.operations) == 4
    assert {operation.source.rank.ep for operation in plan.operations} == {
        0,
        1,
        2,
        3,
    }


def test_source_replica_validation_indexes_fragments_by_tensor_once() -> None:
    tensor_count = 64
    tensors = tuple(
        replace(
            descriptor(),
            tensor_id=f"layers.{index}.self_attn.q_proj.weight",
            layer_id=index,
            expert_id=None,
        )
        for index in range(tensor_count)
    )
    fragments = tuple(
        tp_manifests(
            tp=1,
            pp_rank=0,
            ep_rank=0,
            address_base=0x10000 + index * 0x1000,
            worker_prefix=f"source-{index}",
            tensor=tensor,
        )[0].fragments[0]
        for index, tensor in enumerate(tensors)
    )
    accesses = [0]

    replicas = _complete_runtime_source_replicas(
        {tensor.tensor_id: tensor for tensor in tensors},
        tuple(CountingFragment(fragment, accesses) for fragment in fragments),
    )

    assert set(replicas) == {0}
    assert accesses[0] <= tensor_count * 3


def test_dense_target_tp_coverage_can_span_ep_ranks() -> None:
    tensor = replace(
        descriptor(),
        tensor_id="layers.2.self_attn.q_proj.weight",
        expert_id=None,
    )
    source = tp_manifests(
        tp=4,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
        tensor=tensor,
    )
    target = distribute_tp_shards_across_ep_ranks(
        tp_manifests(
            tp=2,
            pp_rank=0,
            ep_rank=0,
            address_base=0x40000,
            worker_prefix="target",
            tensor=tensor,
        )
    )

    plan = plan_runtime_transfer(source, target)

    assert len(plan.operations) == 4
    assert {operation.target.rank.ep for operation in plan.operations} == {0, 1}


def test_expert_tp_coverage_cannot_span_ep_owners() -> None:
    source = distribute_tp_shards_across_ep_ranks(
        tp_manifests(
            tp=2,
            pp_rank=0,
            ep_rank=0,
            address_base=0x10000,
            worker_prefix="source",
        )
    )
    target = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x40000,
        worker_prefix="target",
    )

    with pytest.raises(ValueError, match="no complete DP replica"):
        plan_runtime_transfer(source, target)


def test_local_plan_uses_one_complete_source_dp_replica() -> None:
    sources = list(
        tp_manifests(
            tp=2,
            dp=2,
            pp_rank=0,
            ep_rank=0,
            address_base=0x10000,
            worker_prefix="source",
        )
    )
    sources = [
        manifest
        for manifest in sources
        if not (
            manifest.fragments[0].rank.dp == 1 and manifest.fragments[0].rank.tp == 1
        )
    ]
    target = tp_manifests(
        tp=1,
        dp=2,
        pp_rank=0,
        ep_rank=0,
        address_base=0x40000,
        worker_prefix="target",
    )[1]

    plan = plan_runtime_transfer_to_local_target(tuple(sources), target)

    assert {operation.source.rank.dp for operation in plan.operations} == {0}


def test_runtime_plan_rejects_target_tp_coverage_split_across_pp_owners() -> None:
    sources = tp_manifests(
        tp=2,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
    )
    targets = list(
        tp_manifests(
            tp=2,
            pp_rank=0,
            ep_rank=0,
            address_base=0x40000,
            worker_prefix="target",
        )
    )
    moved = replace(
        targets[1].fragments[0],
        rank=replace(targets[1].fragments[0].rank, pp=1),
    )
    targets[1] = replace(targets[1], fragments=(moved,))

    with pytest.raises(ValueError, match="target tensor is not fully covered"):
        plan_runtime_transfer(sources, tuple(targets))


def test_multiple_layers_and_experts_move_pp_ep_tp_dp_ownership_together() -> None:
    source_manifests = []
    target_manifests = []
    for layer_id, expert_id in product(range(2), range(2)):
        tensor = replace(
            descriptor(),
            tensor_id=f"layers.{layer_id}.experts.{expert_id}.w1",
            layer_id=layer_id,
            expert_id=expert_id,
        )
        source_manifests.extend(
            tp_manifests(
                tp=2,
                dp=2,
                pp_rank=layer_id,
                ep_rank=expert_id,
                address_base=0x100000 + (layer_id * 2 + expert_id) * 0x10000,
                worker_prefix=f"source-l{layer_id}-e{expert_id}",
                tensor=tensor,
            )
        )
        target_manifests.extend(
            tp_manifests(
                tp=4,
                dp=3,
                pp_rank=1 - layer_id,
                ep_rank=1 - expert_id,
                address_base=0x400000 + (layer_id * 2 + expert_id) * 0x10000,
                worker_prefix=f"target-l{layer_id}-e{expert_id}",
                tensor=tensor,
            )
        )

    plan = plan_runtime_transfer(source_manifests, target_manifests)

    assert plan.model_id == "qwen3.5-0.8b"
    assert plan.revision == "step-42"
    assert plan.total_bytes == 4 * 3 * 8 * 4 * 2
    assert len(plan.operations) == 4 * 3 * 4
    for operation in plan.operations:
        layer_id = operation.target.tensor_id.split(".")[1]
        expert_id = operation.target.tensor_id.split(".")[3]
        assert operation.source.tensor_id == operation.target.tensor_id
        assert operation.source.rank.pp == int(layer_id)
        assert operation.target.rank.pp == 1 - int(layer_id)
        assert operation.source.rank.ep == int(expert_id)
        assert operation.target.rank.ep == 1 - int(expert_id)


def test_planner_rejects_missing_source_coverage() -> None:
    source = tp_manifests(
        tp=2,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
    )[:1]
    target = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x20000,
        worker_prefix="target",
    )

    with pytest.raises(ValueError, match="not fully covered"):
        plan_runtime_transfer(source, target)


def test_planner_rejects_incompatible_layout() -> None:
    source = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
    )
    target = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x20000,
        worker_prefix="target",
        tensor=descriptor(fingerprint="sglang:qwen3.5:fp8:v1"),
    )

    with pytest.raises(ValueError, match="layout mismatch"):
        plan_runtime_transfer(source, target)


@pytest.mark.parametrize("field", ["layer_id", "expert_id"])
def test_planner_rejects_incompatible_tensor_semantics(field: str) -> None:
    source = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
    )
    target_tensor = replace(descriptor(), **{field: 99})
    target = tp_manifests(
        tp=1,
        pp_rank=0,
        ep_rank=0,
        address_base=0x20000,
        worker_prefix="target",
        tensor=target_tensor,
    )

    with pytest.raises(ValueError, match="tensor descriptor mismatch"):
        plan_runtime_transfer(source, target)


def test_local_target_plan_supports_independent_tp_rank_startup() -> None:
    sources = tp_manifests(
        tp=2,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
    )
    targets = tp_manifests(
        tp=4,
        pp_rank=2,
        ep_rank=3,
        address_base=0x40000,
        worker_prefix="target",
    )

    plan = plan_runtime_transfer_to_local_target(sources, targets[1])

    assert len(plan.target_executors) == 1
    assert plan.target_executors[0].rank == targets[1].fragments[0].rank
    assert len(plan.operations) == 1
    operation = plan.operations[0]
    assert operation.source.rank.tp == 0
    assert operation.target.rank.tp == 1
    assert operation.source_offset == 2 * 4 * 2
    assert operation.target_offset == 0
    assert operation.nbytes == 2 * 4 * 2
    assert len(plan.source_executors) == 2
    assert not plan.source_executors[1].operation_indices


def test_local_target_plan_allows_explicit_pp_ep_tensor_subset() -> None:
    first = descriptor()
    second = replace(
        first,
        tensor_id="layers.7.experts.5.w1",
        layer_id=7,
        expert_id=5,
    )
    sources = (
        *tp_manifests(
            tp=2,
            pp_rank=0,
            ep_rank=0,
            address_base=0x10000,
            worker_prefix="source-first",
            tensor=first,
        ),
        *tp_manifests(
            tp=2,
            pp_rank=1,
            ep_rank=1,
            address_base=0x30000,
            worker_prefix="source-second",
            tensor=second,
        ),
    )
    local_target = tp_manifests(
        tp=4,
        pp_rank=3,
        ep_rank=2,
        address_base=0x50000,
        worker_prefix="target",
        tensor=second,
    )[2]

    plan = plan_runtime_transfer_to_local_target(sources, local_target)

    assert {operation.tensor_id for operation in plan.operations} == {second.tensor_id}
    assert {operation.source.rank.pp for operation in plan.operations} == {1}
    assert {operation.source.rank.ep for operation in plan.operations} == {1}
    assert {operation.target.rank.pp for operation in plan.operations} == {3}
    assert {operation.target.rank.ep for operation in plan.operations} == {2}


def test_local_target_plan_rejects_unknown_target_tensor() -> None:
    sources = tp_manifests(
        tp=2,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
    )
    target = tp_manifests(
        tp=4,
        pp_rank=0,
        ep_rank=0,
        address_base=0x40000,
        worker_prefix="target",
        tensor=replace(descriptor(), tensor_id="unknown.weight"),
    )[0]

    with pytest.raises(ValueError, match="unknown tensors"):
        plan_runtime_transfer_to_local_target(sources, target)


def test_local_target_plan_rejects_tensor_without_local_fragment() -> None:
    sources = tp_manifests(
        tp=2,
        pp_rank=0,
        ep_rank=0,
        address_base=0x10000,
        worker_prefix="source",
    )
    target = tp_manifests(
        tp=4,
        pp_rank=0,
        ep_rank=0,
        address_base=0x40000,
        worker_prefix="target",
    )[0]
    target = replace(target, fragments=())

    with pytest.raises(ValueError, match="no fragments"):
        plan_runtime_transfer_to_local_target(sources, target)
