from __future__ import annotations

from dataclasses import replace
from math import prod
from types import SimpleNamespace
import pytest

from mooncake.model_weight import (
    ParallelRank,
    RuntimeFragment,
    RuntimeManifest,
    runtime_binding_from_runtime_manifest,
)

from .helpers import (
    MODEL_ID,
    REVISION,
    descriptor,
    runtime_fragment,
    runtime_inventory,
    runtime_inventory_tensor,
)


def test_runtime_inventory_is_framework_neutral_and_retains_owner() -> None:
    inventory = runtime_inventory(
        tensors=(runtime_inventory_tensor(device="cuda:0"),),
    )
    owner = object()

    manifest = RuntimeManifest.from_runtime_inventory(
        inventory,
        owner_resolver=lambda record: owner if record["fragment_id"] else None,
    )

    assert manifest.model_id == MODEL_ID
    assert manifest.revision == REVISION
    assert manifest.instance_id == "instance"
    assert manifest.placement_id is None
    assert manifest.generation == 7
    assert manifest.lease_id == "lease-7"
    assert manifest.fragments[0].owner is owner
    assert manifest.fragments[0].device == "cuda:0"
    assert manifest.fragments[0].rank == ParallelRank(dp=0, tp=0, pp=1, ep=1)
    assert not hasattr(manifest, "to_json")


def test_runtime_inventory_accepts_object_records_and_optional_semantics() -> None:
    tensor = runtime_inventory_tensor()
    tensor.pop("layer_id")
    tensor.pop("expert_id")
    tensor.pop("aliases")
    inventory = SimpleNamespace(
        **{
            **runtime_inventory(),
            "tensors": (SimpleNamespace(**tensor),),
        }
    )

    manifest = RuntimeManifest.from_runtime_inventory(inventory)

    assert manifest.tensors[0].layer_id is None
    assert manifest.tensors[0].expert_id is None
    assert manifest.fragments[0].aliases == ()


def test_runtime_inventory_imports_multi_axis_logical_box() -> None:
    tensor = runtime_inventory_tensor(
        tensor_id="layers.2.experts.w1",
        global_shape=(8, 16, 32),
        global_offset=(3, 8, 0),
        local_shape=(1, 8, 32),
        partition_dim=None,
        shard_dims=(0, 1),
        expert_id=None,
        nbytes=512,
        stride=(256, 32, 1),
    )

    manifest = RuntimeManifest.from_runtime_inventory(
        runtime_inventory(tensors=(tensor,))
    )

    assert manifest.tensors[0].effective_shard_dims == (0, 1)
    assert manifest.fragments[0].global_offset == (3, 8, 0)
    assert manifest.fragments[0].local_shape == (1, 8, 32)


def test_runtime_inventory_normalizes_equivalent_single_axis_descriptors() -> None:
    first = runtime_inventory_tensor(
        fragment_id="runtime-0",
        placement_fragment_id="placement-0",
        global_offset=(0, 0),
        address=0x1000,
        worker_id="worker-0",
        endpoint="worker-0:12345",
        rank={"dp": 0, "tp": 0, "pp": 1, "ep": 1},
    )
    second = runtime_inventory_tensor(
        fragment_id="runtime-1",
        placement_fragment_id="placement-1",
        shard_dims=(0,),
        global_offset=(4, 0),
        address=0x2000,
        worker_id="worker-1",
        endpoint="worker-1:12345",
        rank={"dp": 0, "tp": 1, "pp": 1, "ep": 1},
    )

    manifest = RuntimeManifest.from_runtime_inventory(
        runtime_inventory(tensors=(first, second))
    )

    assert manifest.tensors[0].partition_dim == 0
    assert manifest.tensors[0].shard_dims == (0,)
    assert len(manifest.fragments) == 2


@pytest.mark.parametrize(
    "tensor_overrides, message",
    [
        ({"is_contiguous": False}, "contiguous"),
        ({"is_contiguous": 1}, "contiguous"),
        ({"stride": (1, 4)}, "canonical stride"),
        ({"stride": (4, True)}, "stride"),
        ({"storage_offset": -1}, "storage_offset"),
        ({"storage_offset": 0.0}, "storage_offset"),
        ({"byte_offset": 1}, "item-aligned"),
        ({"byte_offset": 0.0}, "byte_offset"),
        ({"nbytes": 31}, "byte size mismatch"),
    ],
)
def test_runtime_inventory_rejects_unsafe_physical_views(
    tensor_overrides: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RuntimeManifest.from_runtime_inventory(
            runtime_inventory(tensors=(runtime_inventory_tensor(**tensor_overrides),))
        )


def test_runtime_inventory_requires_explicit_view_address_semantics() -> None:
    inventory = runtime_inventory(
        tensors=(
            runtime_inventory_tensor(
                storage_offset=7,
                byte_offset=14,
            ),
        )
    )

    with pytest.raises(ValueError, match="address_semantics"):
        RuntimeManifest.from_runtime_inventory(inventory)

    manifest = RuntimeManifest.from_runtime_inventory(
        inventory,
        address_semantics="view",
    )

    assert manifest.fragments[0].address == 0x1000


def test_runtime_inventory_rejects_unknown_address_semantics() -> None:
    with pytest.raises(ValueError, match="address_semantics"):
        RuntimeManifest.from_runtime_inventory(
            runtime_inventory(),
            address_semantics="storage",
        )


def test_runtime_inventory_accepts_arbitrary_singleton_stride() -> None:
    tensor = runtime_inventory_tensor(
        global_shape=(8, 1, 4),
        global_offset=(0, 0, 0),
        local_shape=(4, 1, 4),
        stride=(4, 99, 1),
        nbytes=32,
    )

    manifest = RuntimeManifest.from_runtime_inventory(
        runtime_inventory(tensors=(tensor,))
    )

    assert manifest.fragments[0].local_shape == (4, 1, 4)


def test_runtime_inventory_requires_snapshot_generation_match() -> None:
    with pytest.raises(ValueError, match="lease generation mismatch"):
        RuntimeManifest.from_runtime_inventory(
            runtime_inventory(
                generation=8,
                tensors=(runtime_inventory_tensor(lease_generation=7),),
            )
        )


def test_direct_runtime_manifest_requires_one_generation() -> None:
    with pytest.raises(ValueError, match="inconsistent lease generations"):
        RuntimeManifest(
            model_id=MODEL_ID,
            revision=REVISION,
            instance_id="instance",
            tensors=(descriptor(),),
            fragments=(
                runtime_fragment(fragment_id="runtime-0", lease_generation=7),
                runtime_fragment(
                    fragment_id="runtime-1",
                    global_offset=(4, 0),
                    address=0x2000,
                    lease_generation=8,
                ),
            ),
        )


def test_runtime_manifest_derives_generation_for_direct_construction() -> None:
    manifest = RuntimeManifest(
        model_id=MODEL_ID,
        revision=REVISION,
        instance_id="instance",
        tensors=(descriptor(),),
        fragments=(runtime_fragment(),),
    )

    assert manifest.generation == 7


def test_empty_runtime_manifest_retains_explicit_generation() -> None:
    manifest = RuntimeManifest(
        model_id=MODEL_ID,
        revision=REVISION,
        instance_id="instance",
        tensors=(),
        fragments=(),
        generation=9,
        lease_id="lease-9",
    )

    assert runtime_binding_from_runtime_manifest(manifest).generation == 9


@pytest.mark.parametrize("right_address", [0x1000, 0x1010])
def test_runtime_manifest_rejects_overlapping_independent_ranges(
    right_address: int,
) -> None:
    tensors = (
        descriptor(tensor_id="a.weight"),
        descriptor(tensor_id="b.weight"),
    )

    with pytest.raises(ValueError, match="address ranges overlap"):
        RuntimeManifest(
            model_id=MODEL_ID,
            revision=REVISION,
            instance_id="instance",
            tensors=tensors,
            fragments=(
                runtime_fragment(fragment_id="runtime-a", tensor_id="a.weight"),
                runtime_fragment(
                    fragment_id="runtime-b",
                    tensor_id="b.weight",
                    address=right_address,
                ),
            ),
        )


def test_runtime_manifest_rejects_nested_overlap() -> None:
    tensors = (
        descriptor(
            tensor_id="large.weight",
            global_shape=(16,),
            dtype="uint8",
            itemsize=1,
            partition_dim=None,
        ),
        descriptor(
            tensor_id="small.weight",
            global_shape=(4,),
            dtype="uint8",
            itemsize=1,
            partition_dim=None,
        ),
        descriptor(
            tensor_id="nested.weight",
            global_shape=(4,),
            dtype="uint8",
            itemsize=1,
            partition_dim=None,
        ),
    )

    with pytest.raises(ValueError, match="address ranges overlap"):
        RuntimeManifest(
            model_id=MODEL_ID,
            revision=REVISION,
            instance_id="instance",
            tensors=tensors,
            fragments=(
                runtime_fragment(
                    fragment_id="large",
                    tensor_id="large.weight",
                    global_offset=(0,),
                    local_shape=(16,),
                    nbytes=16,
                ),
                runtime_fragment(
                    fragment_id="small",
                    tensor_id="small.weight",
                    global_offset=(0,),
                    local_shape=(4,),
                    address=0x1000,
                    nbytes=4,
                    aliases=("large.weight", "small.weight"),
                ),
                runtime_fragment(
                    fragment_id="nested",
                    tensor_id="nested.weight",
                    global_offset=(0,),
                    local_shape=(4,),
                    address=0x1008,
                    nbytes=4,
                ),
            ),
        )


def test_runtime_manifest_treats_endpoint_as_routing_not_address_space() -> None:
    tensors = (
        descriptor(tensor_id="a.weight"),
        descriptor(tensor_id="b.weight"),
    )

    with pytest.raises(ValueError, match="address ranges overlap"):
        RuntimeManifest(
            model_id=MODEL_ID,
            revision=REVISION,
            instance_id="instance",
            tensors=tensors,
            fragments=(
                runtime_fragment(fragment_id="a", tensor_id="a.weight"),
                runtime_fragment(
                    fragment_id="b",
                    tensor_id="b.weight",
                    endpoint="worker-0:54321",
                ),
            ),
        )


def test_runtime_manifest_allows_same_address_on_different_workers() -> None:
    tensors = (
        descriptor(tensor_id="a.weight"),
        descriptor(tensor_id="b.weight"),
    )

    manifest = RuntimeManifest(
        model_id=MODEL_ID,
        revision=REVISION,
        instance_id="instance",
        tensors=tensors,
        fragments=(
            runtime_fragment(fragment_id="a", tensor_id="a.weight"),
            runtime_fragment(
                fragment_id="b",
                tensor_id="b.weight",
                worker_id="worker-1",
                endpoint="worker-1:12345",
            ),
        ),
    )

    assert len(manifest.fragments) == 2


def test_runtime_manifest_treats_device_as_address_space() -> None:
    tensors = (
        descriptor(tensor_id="a.weight"),
        descriptor(tensor_id="b.weight"),
    )

    manifest = RuntimeManifest(
        model_id=MODEL_ID,
        revision=REVISION,
        instance_id="instance",
        tensors=tensors,
        fragments=(
            runtime_fragment(fragment_id="a", tensor_id="a.weight"),
            runtime_fragment(
                fragment_id="b",
                tensor_id="b.weight",
                device="cuda:1",
            ),
        ),
    )

    assert len(manifest.fragments) == 2


def test_runtime_manifest_rejects_duplicate_logical_box_for_same_rank() -> None:
    with pytest.raises(ValueError, match="duplicate logical fragment"):
        RuntimeManifest(
            model_id=MODEL_ID,
            revision=REVISION,
            instance_id="instance",
            generation=7,
            tensors=(descriptor(),),
            fragments=(
                runtime_fragment(fragment_id="runtime-0", address=0x1000),
                runtime_fragment(fragment_id="runtime-1", address=0x2000),
            ),
        )


@pytest.mark.parametrize(
    "second_offset, second_shape",
    [
        ((2, 0), (4, 4)),
        ((1, 1), (2, 2)),
    ],
)
def test_runtime_manifest_rejects_overlapping_logical_boxes_for_same_rank(
    second_offset: tuple[int, ...],
    second_shape: tuple[int, ...],
) -> None:
    tensor = descriptor(
        global_shape=(8, 4),
        partition_dim=None,
        shard_dims=(0, 1),
    )
    first = runtime_fragment(
        fragment_id="runtime-0",
        global_offset=(0, 0),
        local_shape=(4, 4),
        address=0x1000,
        nbytes=32,
    )
    second = runtime_fragment(
        fragment_id="runtime-1",
        global_offset=second_offset,
        local_shape=second_shape,
        address=0x2000,
        nbytes=prod(second_shape) * tensor.itemsize,
    )

    with pytest.raises(ValueError, match="logical fragment boxes overlap"):
        RuntimeManifest(
            model_id=MODEL_ID,
            revision=REVISION,
            instance_id="instance",
            generation=7,
            tensors=(tensor,),
            fragments=(first, second),
        )


def test_runtime_manifest_allows_adjacent_logical_boxes_for_same_rank() -> None:
    manifest = RuntimeManifest(
        model_id=MODEL_ID,
        revision=REVISION,
        instance_id="instance",
        generation=7,
        tensors=(
            descriptor(
                global_shape=(8, 4),
                partition_dim=None,
                shard_dims=(0, 1),
            ),
        ),
        fragments=(
            runtime_fragment(
                fragment_id="runtime-0",
                global_offset=(0, 0),
                local_shape=(4, 4),
                address=0x1000,
                nbytes=32,
            ),
            runtime_fragment(
                fragment_id="runtime-1",
                global_offset=(4, 0),
                local_shape=(4, 4),
                address=0x2000,
                nbytes=32,
            ),
        ),
    )

    assert len(manifest.fragments) == 2


def test_runtime_manifest_allows_same_logical_box_on_different_ranks() -> None:
    manifest = RuntimeManifest(
        model_id=MODEL_ID,
        revision=REVISION,
        instance_id="instance",
        generation=7,
        tensors=(descriptor(),),
        fragments=(
            runtime_fragment(fragment_id="runtime-0", address=0x1000),
            runtime_fragment(
                fragment_id="runtime-1",
                address=0x2000,
                rank=ParallelRank(dp=1, tp=0, pp=1, ep=1),
            ),
        ),
    )

    assert len(manifest.fragments) == 2


def test_runtime_manifest_allows_only_exact_compatible_declared_aliases() -> None:
    aliases = ("lm_head.weight", "model.embed_tokens.weight")
    tensors = (
        descriptor(tensor_id="embed.weight"),
        descriptor(tensor_id="head.weight"),
    )
    fragments = tuple(
        runtime_fragment(
            fragment_id=f"runtime-{tensor.tensor_id}",
            tensor_id=tensor.tensor_id,
            aliases=aliases,
        )
        for tensor in tensors
    )

    manifest = RuntimeManifest(
        model_id=MODEL_ID,
        revision=REVISION,
        instance_id="instance",
        tensors=tensors,
        fragments=fragments,
    )

    assert len(manifest.fragments) == 2

    with pytest.raises(ValueError, match="address ranges overlap"):
        RuntimeManifest(
            model_id=MODEL_ID,
            revision=REVISION,
            instance_id="instance",
            tensors=(
                tensors[0],
                replace(tensors[1], layout_fingerprint="different"),
            ),
            fragments=fragments,
        )


@pytest.mark.parametrize(
    "fragment",
    [
        runtime_fragment(tensor_id="missing"),
        runtime_fragment(global_offset=(6, 0)),
        runtime_fragment(nbytes=31),
    ],
)
def test_runtime_manifest_rejects_invalid_fragment(fragment: RuntimeFragment) -> None:
    with pytest.raises(ValueError):
        RuntimeManifest(
            model_id=MODEL_ID,
            revision=REVISION,
            instance_id="instance",
            tensors=(descriptor(),),
            fragments=(fragment,),
        )


def test_runtime_fragment_requires_device() -> None:
    with pytest.raises(ValueError, match="device"):
        runtime_fragment(device="")
