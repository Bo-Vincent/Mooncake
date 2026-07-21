from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mooncake.weight_transfer.manifest import (
    ParallelRank,
    RuntimeFragment,
    RuntimeManifest,
    StoredFragment,
    TensorDescriptor,
    WeightManifest,
)


def tensor_descriptor(**overrides) -> TensorDescriptor:
    values = {
        "tensor_id": "layers.2.experts.3.w1",
        "global_shape": (8, 4),
        "dtype": "bfloat16",
        "itemsize": 2,
        "partition_dim": 0,
        "layer_id": 2,
        "expert_id": 3,
        "layout_fingerprint": "sglang:qwen3.5:bf16:v1",
    }
    values.update(overrides)
    return TensorDescriptor(**values)


def runtime_fragment(**overrides) -> RuntimeFragment:
    values = {
        "fragment_id": "runtime-0",
        "tensor_id": "layers.2.experts.3.w1",
        "global_offset": (0, 0),
        "local_shape": (4, 4),
        "address": 0x1000,
        "nbytes": 32,
        "worker_id": "source-0",
        "endpoint": "source-0:12345",
        "rank": ParallelRank(dp=0, tp=0, pp=1, ep=1),
        "lease_generation": 7,
    }
    values.update(overrides)
    return RuntimeFragment(**values)


def framework_tensor(**overrides) -> SimpleNamespace:
    values = {
        "fragment_id": "sglang-fragment",
        "tensor_id": "layers.2.experts.3.w1",
        "global_shape": (8, 4),
        "global_offset": (4, 0),
        "local_shape": (4, 4),
        "dtype": "bfloat16",
        "itemsize": 2,
        "partition_dim": 0,
        "layer_id": 2,
        "expert_id": 3,
        "layout_fingerprint": "sglang:qwen3.5:moe-w13:v1",
        "address": 0x9000,
        "nbytes": 32,
        "worker_id": "sglang-worker",
        "endpoint": "sglang-worker:12345",
        "rank": SimpleNamespace(dp=1, tp=2, pp=3, ep=4),
        "lease_generation": 9,
        "is_contiguous": True,
        "stride": (4, 1),
        "storage_offset": 16,
        "byte_offset": 32,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def stored_fragment(**overrides) -> StoredFragment:
    values = {
        "fragment_id": "stored-0",
        "tensor_id": "layers.2.experts.3.w1",
        "global_offset": (0, 0),
        "local_shape": (8, 4),
        "object_key": "weights/default/qwen/rev/payload/0",
        "object_offset": 0,
        "nbytes": 64,
    }
    values.update(overrides)
    return StoredFragment(**values)


def test_weight_manifest_round_trip_is_stable_and_has_no_runtime_address() -> None:
    manifest = WeightManifest(
        namespace="default",
        model_id="qwen3.5-0.8b",
        revision="step-42",
        group_id="weights/default/qwen3.5-0.8b/step-42",
        manifest_key="weights/default/qwen3.5-0.8b/step-42/manifest",
        tensors=(tensor_descriptor(),),
        fragments=(
            stored_fragment(
                object_key=("weights/default/qwen3.5-0.8b/step-42/payload/0")
            ),
        ),
        created_at="2026-07-17T00:00:00Z",
    )

    encoded = manifest.to_json()

    assert WeightManifest.from_json(encoded) == manifest
    assert encoded == manifest.to_json()
    assert "4096" not in encoded
    assert "address" not in json.loads(encoded)["fragments"][0]


def test_runtime_manifest_keeps_addresses_ephemeral() -> None:
    manifest = RuntimeManifest(
        model_id="qwen3.5-0.8b",
        revision="step-42",
        instance_id="source",
        tensors=(tensor_descriptor(),),
        fragments=(runtime_fragment(),),
    )

    assert manifest.fragments[0].address == 0x1000
    assert not hasattr(manifest, "to_json")


def test_runtime_manifest_imports_framework_inventory_without_framework_dependency() -> (
    None
):
    tensor = framework_tensor(aliases=("model.embed_tokens.weight", "lm_head.weight"))
    inventory = SimpleNamespace(
        model_id="qwen3.5-0.8b",
        revision="step-42",
        instance_id="sglang-instance",
        generation=9,
        tensors=(tensor,),
        format_version=1,
    )

    manifest = RuntimeManifest.from_runtime_inventory(
        inventory,
        owner_resolver=lambda record: ("owner", record.fragment_id),
    )

    assert manifest.tensors[0].tensor_id == tensor.tensor_id
    assert manifest.fragments[0].address == 0x9000
    assert manifest.fragments[0].rank == ParallelRank(dp=1, tp=2, pp=3, ep=4)
    assert manifest.fragments[0].aliases == (
        "lm_head.weight",
        "model.embed_tokens.weight",
    )
    assert manifest.fragments[0].owner == ("owner", "sglang-fragment")


def test_runtime_manifest_v2_imports_multi_axis_logical_box() -> None:
    tensor = framework_tensor(
        tensor_id="layers.2.experts.w1",
        global_shape=(8, 16, 32),
        global_offset=(3, 8, 0),
        local_shape=(1, 8, 32),
        partition_dim=None,
        shard_dims=(0, 1),
        expert_id=None,
        nbytes=512,
        stride=(256, 32, 1),
        storage_offset=0,
        byte_offset=0,
    )
    inventory = SimpleNamespace(
        model_id="qwen3.5-moe",
        revision="step-42",
        instance_id="sglang-instance",
        generation=9,
        tensors=(tensor,),
        format_version=2,
    )

    manifest = RuntimeManifest.from_runtime_inventory(inventory)

    assert manifest.format_version == 2
    assert manifest.tensors[0].effective_shard_dims == (0, 1)
    assert manifest.fragments[0].global_offset == (3, 8, 0)
    assert manifest.fragments[0].local_shape == (1, 8, 32)


def test_runtime_manifest_v2_rejects_partial_non_shard_dimension() -> None:
    tensor = framework_tensor(
        tensor_id="layers.2.experts.w1",
        global_shape=(8, 16, 32),
        global_offset=(3, 8, 4),
        local_shape=(1, 8, 28),
        partition_dim=None,
        shard_dims=(0, 1),
        expert_id=None,
        nbytes=448,
        stride=(224, 28, 1),
        storage_offset=0,
        byte_offset=0,
    )
    inventory = SimpleNamespace(
        model_id="qwen3.5-moe",
        revision="step-42",
        instance_id="sglang-instance",
        generation=9,
        tensors=(tensor,),
        format_version=2,
    )

    with pytest.raises(ValueError, match="non-shard axis"):
        RuntimeManifest.from_runtime_inventory(inventory)


def test_tensor_descriptor_rejects_conflicting_legacy_and_v2_shard_dims() -> None:
    with pytest.raises(ValueError, match="conflicts with shard_dims"):
        tensor_descriptor(partition_dim=0, shard_dims=(1,))


@pytest.mark.parametrize(
    "shard_dims",
    [
        (0, 0),
        (1, 0),
        (2,),
        (True,),
        "01",
    ],
)
def test_tensor_descriptor_rejects_invalid_shard_dims(shard_dims) -> None:
    with pytest.raises(ValueError, match="shard_dims"):
        tensor_descriptor(partition_dim=None, shard_dims=shard_dims)


def test_weight_manifest_v2_round_trip_preserves_shard_dims() -> None:
    tensor = tensor_descriptor(
        tensor_id="layers.2.experts.w1",
        global_shape=(2, 8, 4),
        partition_dim=None,
        shard_dims=(0, 1),
        expert_id=None,
    )
    group_id = "weights/default/qwen/rev"
    manifest = WeightManifest(
        namespace="default",
        model_id="qwen",
        revision="rev",
        group_id=group_id,
        manifest_key=f"{group_id}/manifest",
        tensors=(tensor,),
        fragments=tuple(
            StoredFragment(
                fragment_id=f"stored-e{expert}-o{out_shard}",
                tensor_id=tensor.tensor_id,
                global_offset=(expert, out_shard * 4, 0),
                local_shape=(1, 4, 4),
                object_key=f"{group_id}/payload/e{expert}-o{out_shard}",
                object_offset=0,
                nbytes=32,
            )
            for expert in range(2)
            for out_shard in range(2)
        ),
        created_at="2026-07-22T00:00:00Z",
        format_version=2,
    )

    encoded = manifest.to_json()
    decoded = WeightManifest.from_json(encoded)

    assert decoded == manifest
    assert json.loads(encoded)["tensors"][0]["shard_dims"] == [0, 1]


def test_weight_manifest_v2_round_trip_preserves_legacy_shard_representation() -> None:
    legacy = tensor_descriptor(tensor_id="layers.0.attn.qkv", expert_id=None)
    nd_tensor = tensor_descriptor(
        tensor_id="layers.2.experts.w1",
        global_shape=(2, 8, 4),
        partition_dim=None,
        shard_dims=(0, 1),
        expert_id=None,
    )
    group_id = "weights/default/qwen/rev"
    manifest = WeightManifest(
        namespace="default",
        model_id="qwen",
        revision="rev",
        group_id=group_id,
        manifest_key=f"{group_id}/manifest",
        tensors=(legacy, nd_tensor),
        fragments=(
            StoredFragment(
                fragment_id="legacy",
                tensor_id=legacy.tensor_id,
                global_offset=(0, 0),
                local_shape=legacy.global_shape,
                object_key=f"{group_id}/payload/legacy",
                object_offset=0,
                nbytes=64,
            ),
            *(
                StoredFragment(
                    fragment_id=f"nd-e{expert}-o{out_shard}",
                    tensor_id=nd_tensor.tensor_id,
                    global_offset=(expert, out_shard * 4, 0),
                    local_shape=(1, 4, 4),
                    object_key=f"{group_id}/payload/nd-e{expert}-o{out_shard}",
                    object_offset=0,
                    nbytes=32,
                )
                for expert in range(2)
                for out_shard in range(2)
            ),
        ),
        created_at="2026-07-22T00:00:00Z",
        format_version=2,
    )

    encoded = manifest.to_json()
    decoded = WeightManifest.from_json(encoded)

    assert decoded == manifest
    raw_legacy = next(
        tensor
        for tensor in json.loads(encoded)["tensors"]
        if tensor["tensor_id"] == legacy.tensor_id
    )
    assert raw_legacy["shard_dims"] is None


def test_weight_manifest_v1_json_schema_does_not_emit_shard_dims() -> None:
    manifest = WeightManifest(
        namespace="default",
        model_id="qwen",
        revision="rev",
        group_id="weights/default/qwen/rev",
        manifest_key="weights/default/qwen/rev/manifest",
        tensors=(tensor_descriptor(),),
        fragments=(stored_fragment(),),
        created_at="2026-07-17T00:00:00Z",
    )

    raw = json.loads(manifest.to_json())

    assert manifest.format_version == 1
    assert "shard_dims" not in raw["tensors"][0]


@pytest.mark.parametrize("aliases", ["ab", b"ab"])
def test_runtime_manifest_rejects_scalar_alias_sequences(aliases) -> None:
    inventory = SimpleNamespace(
        model_id="qwen3.5-0.8b",
        revision="step-42",
        instance_id="sglang-instance",
        generation=9,
        tensors=(framework_tensor(aliases=aliases),),
        format_version=1,
    )

    with pytest.raises(ValueError, match="aliases must be a sequence"):
        RuntimeManifest.from_runtime_inventory(inventory)


def test_runtime_manifest_accepts_alias_lists() -> None:
    inventory = SimpleNamespace(
        model_id="qwen3.5-0.8b",
        revision="step-42",
        instance_id="sglang-instance",
        generation=9,
        tensors=(framework_tensor(aliases=["model.weight", "model.alias"]),),
        format_version=1,
    )

    manifest = RuntimeManifest.from_runtime_inventory(inventory)

    assert manifest.fragments[0].aliases == ("model.alias", "model.weight")


@pytest.mark.parametrize("mapping_record", [False, True])
def test_runtime_manifest_defaults_missing_optional_tensor_semantics(
    mapping_record: bool,
) -> None:
    values = vars(framework_tensor()).copy()
    values.pop("layer_id")
    values.pop("expert_id")
    tensor = values if mapping_record else SimpleNamespace(**values)
    inventory = SimpleNamespace(
        model_id="qwen3.5-0.8b",
        revision="step-42",
        instance_id="sglang-instance",
        generation=9,
        tensors=(tensor,),
        format_version=1,
    )

    manifest = RuntimeManifest.from_runtime_inventory(inventory)

    assert manifest.tensors[0].layer_id is None
    assert manifest.tensors[0].expert_id is None


def test_runtime_manifest_accepts_contiguous_singleton_stride() -> None:
    tensor = framework_tensor(
        global_shape=(8, 1, 4),
        global_offset=(4, 0, 0),
        local_shape=(4, 1, 4),
        stride=(4, 99, 1),
    )
    inventory = SimpleNamespace(
        model_id="qwen3.5-0.8b",
        revision="step-42",
        instance_id="sglang-instance",
        generation=9,
        tensors=(tensor,),
        format_version=1,
    )

    manifest = RuntimeManifest.from_runtime_inventory(inventory)

    assert manifest.fragments[0].local_shape == (4, 1, 4)


def test_runtime_fragment_preserves_positional_owner_argument() -> None:
    owner = object()

    fragment = RuntimeFragment(
        "runtime-0",
        "layers.2.experts.3.w1",
        (0, 0),
        (4, 4),
        0x1000,
        32,
        "source-0",
        "source-0:12345",
        ParallelRank(dp=0, tp=0, pp=1, ep=1),
        7,
        owner,
    )

    assert fragment.owner is owner
    assert fragment.aliases == ()


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"is_contiguous": False}, "contiguous"),
        ({"is_contiguous": 1}, "contiguous"),
        ({"stride": (1, 4)}, "canonical stride"),
        ({"stride": (4, True)}, "stride"),
        ({"storage_offset": 16, "byte_offset": 30}, "byte_offset"),
        ({"storage_offset": 15, "byte_offset": 32}, "byte_offset"),
        ({"storage_offset": 16.0}, "storage_offset"),
        ({"storage_offset": -1, "byte_offset": -2}, "storage_offset"),
        ({"byte_offset": 32.0}, "byte_offset"),
        ({"nbytes": 30}, "byte size mismatch"),
    ],
)
def test_runtime_manifest_rejects_unexplainable_runtime_views(
    overrides: dict, message: str
) -> None:
    tensor = framework_tensor(**overrides)
    inventory = SimpleNamespace(
        model_id="qwen3.5-0.8b",
        revision="step-42",
        instance_id="sglang-instance",
        generation=9,
        tensors=(tensor,),
        format_version=1,
    )

    with pytest.raises(ValueError, match=message):
        RuntimeManifest.from_runtime_inventory(inventory)


@pytest.mark.parametrize(
    "fragment",
    [
        stored_fragment(tensor_id="missing"),
        stored_fragment(global_offset=(6, 0), local_shape=(4, 4)),
        stored_fragment(nbytes=31),
    ],
)
def test_weight_manifest_rejects_invalid_fragment(fragment: StoredFragment) -> None:
    with pytest.raises(ValueError):
        WeightManifest(
            namespace="default",
            model_id="qwen",
            revision="rev",
            group_id="weights/default/qwen/rev",
            manifest_key="weights/default/qwen/rev/manifest",
            tensors=(tensor_descriptor(),),
            fragments=(fragment,),
            created_at="2026-07-17T00:00:00Z",
        )


def test_weight_manifest_rejects_missing_tensor_coverage() -> None:
    with pytest.raises(ValueError, match="not fully covered"):
        WeightManifest(
            namespace="default",
            model_id="qwen",
            revision="rev",
            group_id="weights/default/qwen/rev",
            manifest_key="weights/default/qwen/rev/manifest",
            tensors=(tensor_descriptor(),),
            fragments=(stored_fragment(local_shape=(4, 4), nbytes=32),),
            created_at="2026-07-17T00:00:00Z",
        )


def test_weight_manifest_rejects_duplicate_fragment_geometry() -> None:
    with pytest.raises(ValueError, match="duplicate fragment geometry"):
        WeightManifest(
            namespace="default",
            model_id="qwen",
            revision="rev",
            group_id="weights/default/qwen/rev",
            manifest_key="weights/default/qwen/rev/manifest",
            tensors=(tensor_descriptor(),),
            fragments=(
                stored_fragment(),
                stored_fragment(
                    fragment_id="stored-1",
                    object_key="weights/default/qwen/rev/payload/1",
                ),
            ),
            created_at="2026-07-17T00:00:00Z",
        )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"format_version": 3}, "unsupported runtime inventory format_version"),
        ({"generation": 8}, "lease generation mismatch"),
    ],
)
def test_runtime_manifest_rejects_incompatible_framework_inventory(
    overrides: dict, message: str
) -> None:
    tensor = framework_tensor(
        global_offset=(0, 0),
        rank=SimpleNamespace(dp=0, tp=0, pp=0, ep=0),
    )
    values = dict(
        model_id="qwen3.5-0.8b",
        revision="step-42",
        instance_id="sglang-instance",
        generation=9,
        tensors=(tensor,),
        format_version=1,
    )
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        RuntimeManifest.from_runtime_inventory(SimpleNamespace(**values))


def test_runtime_manifest_rejects_duplicate_fragment_ids() -> None:
    fragment = runtime_fragment()

    with pytest.raises(ValueError, match="duplicate fragment_id"):
        RuntimeManifest(
            model_id="qwen",
            revision="rev",
            instance_id="source",
            tensors=(tensor_descriptor(),),
            fragments=(fragment, fragment),
        )


def test_manifest_types_cannot_cross_the_runtime_storage_boundary() -> None:
    with pytest.raises(ValueError, match="RuntimeManifest fragments"):
        RuntimeManifest(
            model_id="qwen",
            revision="rev",
            instance_id="source",
            tensors=(tensor_descriptor(),),
            fragments=(stored_fragment(),),
        )

    with pytest.raises(ValueError, match="WeightManifest fragments"):
        WeightManifest(
            namespace="default",
            model_id="qwen",
            revision="rev",
            group_id="weights/default/qwen/rev",
            manifest_key="weights/default/qwen/rev/manifest",
            tensors=(tensor_descriptor(),),
            fragments=(runtime_fragment(),),
            created_at="2026-07-17T00:00:00Z",
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ParallelRank(dp=True),
        lambda: tensor_descriptor(global_shape=(8.0, 4)),
        lambda: tensor_descriptor(itemsize=2.0),
        lambda: tensor_descriptor(partition_dim=0.0),
        lambda: runtime_fragment(global_offset=(0.0, 0)),
        lambda: runtime_fragment(address=4096.0),
        lambda: runtime_fragment(lease_generation=False),
        lambda: stored_fragment(object_offset=0.0),
        lambda: stored_fragment(nbytes=64.0),
    ],
)
def test_manifest_schema_rejects_non_integer_numeric_fields(factory) -> None:
    with pytest.raises(ValueError, match="integer"):
        factory()


@pytest.mark.parametrize(
    "aliases, message",
    [
        ("model.weight", "non-empty strings"),
        (("model.weight", "model.weight"), "duplicates"),
        (("model.weight", ""), "non-empty strings"),
    ],
)
def test_runtime_fragment_rejects_invalid_aliases(aliases, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        runtime_fragment(aliases=aliases)


def test_weight_manifest_json_rejects_non_finite_and_non_mapping_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        WeightManifest.from_json('{"format_version":NaN}')

    with pytest.raises(ValueError, match="JSON object"):
        WeightManifest.from_json("[]")


def test_weight_manifest_json_rejects_float_geometry() -> None:
    manifest = WeightManifest(
        namespace="default",
        model_id="qwen",
        revision="rev",
        group_id="weights/default/qwen/rev",
        manifest_key="weights/default/qwen/rev/manifest",
        tensors=(tensor_descriptor(),),
        fragments=(stored_fragment(),),
        created_at="2026-07-17T00:00:00Z",
    )
    raw = json.loads(manifest.to_json())
    raw["tensors"][0]["global_shape"][0] = 8.0

    with pytest.raises(ValueError, match="integer"):
        WeightManifest.from_json(json.dumps(raw))


@pytest.mark.parametrize(
    "manifest_key, object_key",
    [
        (
            "weights/default/other/rev/manifest",
            "weights/default/qwen/rev/payload/0",
        ),
        (
            "weights/default/qwen/rev/manifest",
            "weights/default/other/rev/payload/0",
        ),
    ],
)
def test_weight_manifest_binds_manifest_and_payload_keys_to_group(
    manifest_key: str, object_key: str
) -> None:
    with pytest.raises(ValueError, match="group"):
        WeightManifest(
            namespace="default",
            model_id="qwen",
            revision="rev",
            group_id="weights/default/qwen/rev",
            manifest_key=manifest_key,
            tensors=(tensor_descriptor(),),
            fragments=(stored_fragment(object_key=object_key),),
            created_at="2026-07-17T00:00:00Z",
        )


@pytest.mark.parametrize("mutation", ["missing-version", "unknown-field"])
def test_weight_manifest_json_requires_exact_top_level_schema(mutation: str) -> None:
    manifest = WeightManifest(
        namespace="default",
        model_id="qwen",
        revision="rev",
        group_id="weights/default/qwen/rev",
        manifest_key="weights/default/qwen/rev/manifest",
        tensors=(tensor_descriptor(),),
        fragments=(stored_fragment(),),
        created_at="2026-07-17T00:00:00Z",
    )
    raw = json.loads(manifest.to_json())
    if mutation == "missing-version":
        del raw["format_version"]
    else:
        raw["future_semantics"] = "required"

    with pytest.raises(ValueError, match="schema"):
        WeightManifest.from_json(json.dumps(raw))


@pytest.mark.parametrize(
    "overrides",
    [
        {"global_shape": ()},
        {"partition_dim": 2},
        {"itemsize": 0},
        {"tensor_id": ""},
    ],
)
def test_tensor_descriptor_rejects_invalid_schema(overrides: dict) -> None:
    with pytest.raises(ValueError):
        tensor_descriptor(**overrides)
