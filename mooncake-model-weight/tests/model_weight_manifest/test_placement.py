from __future__ import annotations

import json
import pytest

from mooncake.model_weight import (
    ParallelRank,
    PlacementManifest,
)

from .helpers import (
    MODEL_ID,
    REVISION,
    descriptor,
    placement_fragment,
    placement_manifest,
    runtime_inventory_tensor,
)


def test_placement_round_trip_is_stable_and_address_free() -> None:
    placement = placement_manifest()

    encoded = placement.to_json()
    decoded = PlacementManifest.from_json(encoded)

    assert decoded == placement
    assert decoded.digest == placement.digest
    assert placement.placement_id == (
        "sha256:c4f3bc2feed99a64ff156fd57cb5cf626ae8b38c1551c0fe2225e1921e14d73a"
    )
    assert placement.digest == (
        "7320a5090dc88556c57e5d9e8cb316d05b3bf99b00f7a0c0da04198ca995e915"
    )
    assert encoded == placement.to_json()
    for forbidden in (
        "address",
        "endpoint",
        "worker_id",
        "instance_id",
        "generation",
        "lease_id",
        "owner",
    ):
        assert forbidden not in encoded


def test_placement_digest_is_independent_of_inventory_order() -> None:
    tensors = (
        descriptor(tensor_id="b.weight"),
        descriptor(tensor_id="a.weight"),
    )
    fragments = (
        placement_fragment(
            placement_fragment_id="b",
            tensor_id="b.weight",
            rank=ParallelRank(tp=1),
        ),
        placement_fragment(
            placement_fragment_id="a",
            tensor_id="a.weight",
            rank=ParallelRank(tp=0),
        ),
    )

    first = placement_manifest(tensors=tensors, fragments=fragments)
    second = placement_manifest(
        tensors=tuple(reversed(tensors)),
        fragments=tuple(reversed(fragments)),
    )

    assert first == second
    assert first.to_json() == second.to_json()
    assert first.digest == second.digest


def test_placement_normalizes_partition_dim_and_shard_dims() -> None:
    partitioned = placement_manifest(
        tensors=(descriptor(shard_dims=None),),
    )
    multidim_placement = placement_manifest(
        tensors=(descriptor(shard_dims=(0,)),),
    )

    assert partitioned == multidim_placement
    assert partitioned.tensors[0].shard_dims == (0,)
    assert partitioned.digest == multidim_placement.digest


@pytest.mark.parametrize("mutation", ["missing", "unknown", "nan"])
def test_placement_json_requires_strict_schema(mutation: str) -> None:
    raw = json.loads(placement_manifest().to_json())
    if mutation == "missing":
        del raw["revision"]
    elif mutation == "unknown":
        raw["future_semantics"] = "required"
    else:
        raw["model_id"] = float("nan")

    with pytest.raises(ValueError):
        PlacementManifest.from_json(json.dumps(raw))


@pytest.mark.parametrize(
    ("path", "mutation"),
    [
        (("tensors", 0), ("pop", "dtype")),
        (("tensors", 0), ("set", "future_semantics")),
        (("fragments", 0), ("pop", "nbytes")),
        (("fragments", 0), ("set", "future_semantics")),
        (("fragments", 0, "rank"), ("pop", "tp")),
        (("fragments", 0, "rank"), ("set", "future_semantics")),
    ],
)
def test_placement_json_requires_strict_nested_schema(
    path: tuple, mutation: tuple[str, str]
) -> None:
    raw = json.loads(placement_manifest().to_json())
    target = raw
    for component in path:
        target = target[component]
    operation, field = mutation
    if operation == "pop":
        target.pop(field)
    else:
        target[field] = "unsupported"

    with pytest.raises(ValueError, match="schema"):
        PlacementManifest.from_json(json.dumps(raw))


@pytest.mark.parametrize("value", ["not-json", "[]", '"placement"'])
def test_placement_json_rejects_invalid_document(value: str) -> None:
    with pytest.raises(ValueError):
        PlacementManifest.from_json(value)


def test_placement_json_rejects_duplicate_object_keys() -> None:
    encoded = placement_manifest().to_json()
    duplicated = encoded.replace(
        '"model_id":"model"',
        '"model_id":"model","model_id":"other"',
        1,
    )

    with pytest.raises(ValueError, match="duplicate JSON field"):
        PlacementManifest.from_json(duplicated)


@pytest.mark.parametrize("aliases", ["alias", {"alias": 1}, ["alias", "alias"]])
def test_placement_json_rejects_invalid_aliases(aliases) -> None:
    raw = json.loads(placement_manifest().to_json())
    raw["fragments"][0]["aliases"] = aliases

    with pytest.raises(ValueError, match="aliases"):
        PlacementManifest.from_json(json.dumps(raw))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("tensors",), {}),
        (("fragments",), 1),
        (("tensors", 0, "global_shape"), 8),
        (("tensors", 0, "shard_dims"), "0"),
        (("fragments", 0, "global_offset"), 0),
        (("fragments", 0, "local_shape"), None),
        (("fragments", 0, "rank"), []),
    ],
)
def test_placement_json_rejects_wrong_container_types(
    path: tuple, value: object
) -> None:
    raw = json.loads(placement_manifest().to_json())
    target = raw
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value

    with pytest.raises(ValueError):
        PlacementManifest.from_json(json.dumps(raw))


def test_placement_inventory_is_framework_neutral() -> None:
    tensor = runtime_inventory_tensor()
    inventory = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "tensors": (
            {
                key: value
                for key, value in tensor.items()
                if key
                in {
                    "placement_fragment_id",
                    "tensor_id",
                    "global_shape",
                    "global_offset",
                    "local_shape",
                    "dtype",
                    "itemsize",
                    "partition_dim",
                    "layer_id",
                    "expert_id",
                    "layout_fingerprint",
                    "nbytes",
                    "rank",
                    "aliases",
                }
            },
        ),
    }

    placement = PlacementManifest.from_runtime_inventory(inventory)

    assert placement.fragments[0].rank == ParallelRank(dp=0, tp=0, pp=1, ep=1)
    assert placement.fragments[0].global_offset == (0, 0)


def test_placement_inventory_normalizes_equivalent_single_axis_descriptors() -> None:
    fields = {
        "placement_fragment_id",
        "tensor_id",
        "global_shape",
        "global_offset",
        "local_shape",
        "dtype",
        "itemsize",
        "partition_dim",
        "layer_id",
        "expert_id",
        "layout_fingerprint",
        "shard_dims",
        "nbytes",
        "rank",
        "aliases",
    }
    first = runtime_inventory_tensor(
        placement_fragment_id="placement-0",
        global_offset=(0, 0),
        rank={"dp": 0, "tp": 0, "pp": 1, "ep": 1},
    )
    second = runtime_inventory_tensor(
        placement_fragment_id="placement-1",
        shard_dims=(0,),
        global_offset=(4, 0),
        rank={"dp": 0, "tp": 1, "pp": 1, "ep": 1},
    )
    inventory = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "tensors": tuple(
            {key: value for key, value in tensor.items() if key in fields}
            for tensor in (first, second)
        ),
    }

    placement = PlacementManifest.from_runtime_inventory(inventory)

    assert placement.tensors[0].partition_dim == 0
    assert placement.tensors[0].shard_dims == (0,)
    assert len(placement.fragments) == 2


def test_placement_id_must_match_canonical_logical_content() -> None:
    with pytest.raises(ValueError, match="canonical logical content"):
        placement_manifest(placement_id="opaque-placement-id")
