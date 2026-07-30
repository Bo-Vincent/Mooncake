from __future__ import annotations

from dataclasses import replace
import pytest

from mooncake.model_weight import (
    ParallelRank,
    RuntimeBindingManifest,
    bind_runtime_manifest,
)

from .helpers import (
    MODEL_ID,
    REVISION,
    binding_fragment,
    binding_manifest,
    descriptor,
    placement_fragment,
    placement_manifest,
)


def test_runtime_binding_inventory_retains_owner() -> None:
    placement = placement_manifest()
    inventory = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "placement_id": placement.placement_id,
        "placement_digest": placement.digest,
        "instance_id": "instance",
        "generation": 7,
        "lease_id": "lease-7",
        "fragments": (
            {
                "placement_fragment_id": "placement-0",
                "fragment_id": "runtime-0",
                "address": 0x1000,
                "nbytes": 32,
                "worker_id": "worker-0",
                "endpoint": "worker-0:12345",
                "device": "cuda:0",
                "is_contiguous": True,
            },
        ),
    }
    owner = object()

    binding = RuntimeBindingManifest.from_runtime_inventory(
        inventory,
        owner_resolver=lambda record: owner if record["fragment_id"] else None,
    )

    assert binding.fragments[0].owner is owner
    assert binding.fragments[0].device == "cuda:0"


def test_runtime_binding_inventory_requires_contiguous_proof() -> None:
    placement = placement_manifest()
    inventory = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "placement_id": placement.placement_id,
        "placement_digest": placement.digest,
        "instance_id": "instance",
        "generation": 7,
        "lease_id": "lease-7",
        "fragments": (
            {
                "placement_fragment_id": "placement-0",
                "fragment_id": "runtime-0",
                "address": 0x1000,
                "nbytes": 32,
                "worker_id": "worker-0",
                "endpoint": "worker-0:12345",
                "device": "cuda:0",
            },
        ),
    }

    with pytest.raises(ValueError, match="is_contiguous"):
        RuntimeBindingManifest.from_runtime_inventory(inventory)


def test_runtime_binding_inventory_rejects_fragment_generation_mismatch() -> None:
    placement = placement_manifest()
    inventory = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "placement_id": placement.placement_id,
        "placement_digest": placement.digest,
        "instance_id": "instance",
        "generation": 7,
        "lease_id": "lease-7",
        "fragments": (
            {
                "placement_fragment_id": "placement-0",
                "fragment_id": "runtime-0",
                "address": 0x1000,
                "nbytes": 32,
                "worker_id": "worker-0",
                "endpoint": "worker-0:12345",
                "device": "cuda:0",
                "is_contiguous": True,
                "lease_generation": 6,
            },
        ),
    }

    with pytest.raises(ValueError, match="lease generation"):
        RuntimeBindingManifest.from_runtime_inventory(inventory)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("is_contiguous", False, "contiguous"),
        ("storage_offset", -1, "storage_offset"),
        ("device", "", "device"),
    ],
)
def test_runtime_binding_inventory_rejects_unsafe_views(
    field: str, value: object, message: str
) -> None:
    placement = placement_manifest()
    fragment = {
        "placement_fragment_id": "placement-0",
        "fragment_id": "runtime-0",
        "address": 0x1000,
        "nbytes": 32,
        "worker_id": "worker-0",
        "endpoint": "worker-0:12345",
        "device": "cuda:0",
        "is_contiguous": True,
        field: value,
    }
    inventory = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "placement_id": placement.placement_id,
        "placement_digest": placement.digest,
        "instance_id": "instance",
        "generation": 7,
        "lease_id": "lease-7",
        "fragments": (fragment,),
    }

    with pytest.raises(ValueError, match=message):
        RuntimeBindingManifest.from_runtime_inventory(inventory)


@pytest.mark.parametrize("offset_field", ["storage_offset", "byte_offset"])
def test_runtime_binding_inventory_requires_explicit_view_address_semantics(
    offset_field: str,
) -> None:
    placement = placement_manifest()
    fragment = {
        "placement_fragment_id": "placement-0",
        "fragment_id": "runtime-0",
        "address": 0x1000,
        "nbytes": 32,
        "worker_id": "worker-0",
        "endpoint": "worker-0:12345",
        "device": "cuda:0",
        "is_contiguous": True,
        offset_field: 7,
    }
    inventory = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "placement_id": placement.placement_id,
        "placement_digest": placement.digest,
        "instance_id": "instance",
        "generation": 7,
        "lease_id": "lease-7",
        "fragments": (fragment,),
    }

    with pytest.raises(ValueError, match="address_semantics"):
        RuntimeBindingManifest.from_runtime_inventory(inventory)

    binding = RuntimeBindingManifest.from_runtime_inventory(
        inventory,
        address_semantics="view",
    )

    assert binding.fragments[0].address == 0x1000


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"placement_digest": ""}, "placement_digest"),
        ({"placement_digest": "g" * 64}, "SHA-256"),
        ({"placement_digest": "a" * 63}, "SHA-256"),
    ],
)
def test_runtime_binding_requires_content_attestation(
    overrides: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        binding_manifest(**overrides)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"model_id": "other"}, "model_id"),
        ({"revision": "other"}, "revision"),
        ({"placement_id": "other"}, "placement_id"),
    ],
)
def test_binding_rejects_identity_mismatch(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        bind_runtime_manifest(placement_manifest(), binding_manifest(**overrides))


def test_binding_requires_exact_fragment_set_and_size() -> None:
    placement = placement_manifest()

    with pytest.raises(ValueError, match="missing placement fragment"):
        bind_runtime_manifest(
            placement,
            binding_manifest(placement=placement, fragments=()),
        )
    with pytest.raises(ValueError, match="unknown placement fragment"):
        bind_runtime_manifest(
            placement,
            binding_manifest(
                placement=placement,
                fragments=(binding_fragment(placement_fragment_id="unknown"),),
            ),
        )
    with pytest.raises(ValueError, match="byte size"):
        bind_runtime_manifest(
            placement,
            binding_manifest(
                placement=placement,
                fragments=(binding_fragment(nbytes=64),),
            ),
        )


def test_binding_rejects_duplicate_fragment_ids() -> None:
    fragment = binding_fragment()

    with pytest.raises(ValueError, match="duplicate placement fragment"):
        binding_manifest(fragments=(fragment, replace(fragment, fragment_id="other")))
    with pytest.raises(ValueError, match="duplicate runtime fragment_id"):
        binding_manifest(
            fragments=(
                fragment,
                replace(
                    fragment,
                    placement_fragment_id="placement-other",
                ),
            )
        )


def test_binding_allows_one_rank_to_span_runtime_locations() -> None:
    placement = placement_manifest(
        fragments=(
            placement_fragment(placement_fragment_id="left"),
            placement_fragment(
                placement_fragment_id="right",
                global_offset=(4, 0),
            ),
        )
    )
    binding = binding_manifest(
        placement=placement,
        fragments=(
            binding_fragment(placement_fragment_id="left"),
            binding_fragment(
                placement_fragment_id="right",
                fragment_id="runtime-right",
                address=0x2000,
                worker_id="worker-1",
                endpoint="worker-1:12345",
            ),
        ),
    )

    runtime = bind_runtime_manifest(placement, binding)

    assert {fragment.worker_id for fragment in runtime.fragments} == {
        "worker-0",
        "worker-1",
    }


def test_binding_rejects_overlapping_runtime_ranges() -> None:
    placement = placement_manifest(
        tensors=(
            descriptor(tensor_id="a.weight"),
            descriptor(tensor_id="b.weight"),
        ),
        fragments=(
            placement_fragment(
                placement_fragment_id="a",
                tensor_id="a.weight",
                rank=ParallelRank(tp=0),
            ),
            placement_fragment(
                placement_fragment_id="b",
                tensor_id="b.weight",
                rank=ParallelRank(tp=1),
            ),
        ),
    )
    binding = binding_manifest(
        placement=placement,
        fragments=(
            binding_fragment(placement_fragment_id="a", fragment_id="runtime-a"),
            binding_fragment(
                placement_fragment_id="b",
                fragment_id="runtime-b",
                endpoint="worker-0:54321",
            ),
        ),
    )

    with pytest.raises(ValueError, match="address ranges overlap"):
        bind_runtime_manifest(placement, binding)


def test_binding_preserves_logical_and_physical_halves() -> None:
    placement = placement_manifest()
    binding = binding_manifest(placement=placement)

    runtime = bind_runtime_manifest(placement, binding)

    assert runtime.model_id == placement.model_id
    assert runtime.revision == placement.revision
    assert runtime.placement_id == placement.placement_id
    assert runtime.instance_id == binding.instance_id
    assert runtime.generation == binding.generation
    assert runtime.lease_id == binding.lease_id
    assert runtime.fragments[0].global_offset == placement.fragments[0].global_offset
    assert runtime.fragments[0].address == binding.fragments[0].address
    assert runtime.fragments[0].placement_fragment_id == "placement-0"


def test_binding_order_does_not_change_runtime_manifest() -> None:
    tensors = (
        descriptor(tensor_id="a.weight"),
        descriptor(tensor_id="b.weight"),
    )
    fragments = (
        placement_fragment(
            placement_fragment_id="a",
            tensor_id="a.weight",
            rank=ParallelRank(tp=0),
        ),
        placement_fragment(
            placement_fragment_id="b",
            tensor_id="b.weight",
            rank=ParallelRank(tp=1),
        ),
    )
    placement = placement_manifest(tensors=tensors, fragments=fragments)
    bindings = (
        binding_fragment(
            placement_fragment_id="a",
            fragment_id="runtime-a",
            address=0x1000,
        ),
        binding_fragment(
            placement_fragment_id="b",
            fragment_id="runtime-b",
            address=0x2000,
        ),
    )

    first = bind_runtime_manifest(
        placement,
        binding_manifest(placement=placement, fragments=bindings),
    )
    second = bind_runtime_manifest(
        placement,
        binding_manifest(
            placement=placement,
            fragments=tuple(reversed(bindings)),
        ),
    )

    assert first == second


def test_empty_placement_binds_to_generation_scoped_empty_runtime() -> None:
    placement = placement_manifest(tensors=(), fragments=())
    binding = binding_manifest(
        placement=placement,
        fragments=(),
        generation=11,
        lease_id="lease-11",
    )

    runtime = bind_runtime_manifest(placement, binding)

    assert runtime.fragments == ()
    assert runtime.generation == 11
    assert runtime.lease_id == "lease-11"
