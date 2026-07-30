from __future__ import annotations

import pytest

from mooncake.model_weight import (
    ParallelRank,
    RuntimeManifest,
)

from .helpers import (
    MODEL_ID,
    REVISION,
    binding_fragment,
    binding_manifest,
    placement_manifest,
    runtime_fragment,
    runtime_inventory,
)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ParallelRank(dp=True),
        lambda: runtime_fragment(address=4096.0),
        lambda: runtime_fragment(lease_generation=False),
        lambda: binding_fragment(nbytes=32.0),
        lambda: binding_manifest(generation=True),
    ],
)
def test_contract_rejects_bool_and_float_integer_fields(factory) -> None:
    with pytest.raises(ValueError, match="integer"):
        factory()


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: placement_manifest(tensors=(object(),)), "tensors"),
        (lambda: placement_manifest(fragments=(object(),)), "fragments"),
        (lambda: binding_manifest(fragments=(object(),)), "fragments"),
        (
            lambda: RuntimeManifest(
                model_id=MODEL_ID,
                revision=REVISION,
                instance_id="instance",
                tensors=(object(),),
                fragments=(),
            ),
            "tensors",
        ),
    ],
)
def test_manifest_collections_reject_wrong_element_types(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: placement_manifest(tensors=None), "tensors"),
        (lambda: placement_manifest(fragments=None), "fragments"),
        (lambda: binding_manifest(fragments=None), "fragments"),
        (
            lambda: RuntimeManifest(
                model_id=MODEL_ID,
                revision=REVISION,
                instance_id="instance",
                tensors=None,
                fragments=(),
            ),
            "tensors",
        ),
        (
            lambda: RuntimeManifest(
                model_id=MODEL_ID,
                revision=REVISION,
                instance_id="instance",
                tensors=(),
                fragments=None,
            ),
            "fragments",
        ),
    ],
)
def test_manifest_collections_reject_wrong_container_types(
    factory, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: runtime_fragment(address=2**64),
        lambda: runtime_fragment(address=2**64 - 16, nbytes=32),
        lambda: runtime_fragment(nbytes=2**64),
        lambda: binding_fragment(address=2**64),
        lambda: binding_fragment(address=2**64 - 16, nbytes=32),
        lambda: binding_manifest(generation=2**64),
    ],
)
def test_physical_contract_rejects_values_outside_u64_abi(factory) -> None:
    with pytest.raises(ValueError, match="64-bit"):
        factory()


def test_physical_contract_rejects_unrepresentable_exclusive_end() -> None:
    with pytest.raises(ValueError, match="64-bit"):
        runtime_fragment(address=2**64 - 4, nbytes=4)


def test_inventory_missing_required_field_is_a_contract_error() -> None:
    inventory = runtime_inventory()
    del inventory["model_id"]

    with pytest.raises(ValueError, match="missing required field: model_id"):
        RuntimeManifest.from_runtime_inventory(inventory)
