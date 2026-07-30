from __future__ import annotations

from dataclasses import replace
import pytest

from mooncake.model_weight import (
    ParallelRank,
    RuntimeManifest,
    bind_runtime_manifest,
    placement_manifest_from_runtime_manifest,
    runtime_binding_from_runtime_manifest,
)

from .helpers import (
    MODEL_ID,
    REVISION,
    descriptor,
    placement_manifest,
    runtime_fragment,
)


def test_runtime_projection_round_trip_supports_rebinding() -> None:
    owner = object()
    runtime = RuntimeManifest(
        model_id=MODEL_ID,
        revision=REVISION,
        instance_id="instance",
        generation=7,
        lease_id="lease-7",
        tensors=(descriptor(),),
        fragments=(runtime_fragment(owner=owner),),
    )

    placement = placement_manifest_from_runtime_manifest(runtime)
    binding = runtime_binding_from_runtime_manifest(runtime)
    rebound = bind_runtime_manifest(
        placement,
        replace(
            binding,
            instance_id="instance-2",
            generation=8,
            lease_id="lease-8",
            fragments=(
                replace(
                    binding.fragments[0],
                    address=0x2000,
                    worker_id="worker-2",
                    endpoint="worker-2:12345",
                ),
            ),
        ),
    )

    assert binding.fragments[0].owner is owner
    assert rebound.fragments[0].owner is owner
    assert rebound.fragments[0].address == 0x2000
    assert rebound.generation == 8
    assert placement.digest == placement_manifest_from_runtime_manifest(rebound).digest


@pytest.mark.parametrize(
    "mutate",
    [
        lambda placement: replace(
            placement,
            fragments=(replace(placement.fragments[0], global_offset=(4, 0)),),
        ),
        lambda placement: replace(
            placement,
            fragments=(replace(placement.fragments[0], rank=ParallelRank(tp=1)),),
        ),
        lambda placement: replace(
            placement,
            fragments=(
                replace(
                    placement.fragments[0],
                    aliases=("alias-a", "alias-b"),
                ),
            ),
        ),
        lambda placement: replace(
            placement,
            tensors=(replace(placement.tensors[0], dtype="float16"),),
        ),
        lambda placement: replace(
            placement,
            tensors=(
                replace(
                    placement.tensors[0],
                    layout_fingerprint="test:qwen:packed:v2",
                ),
            ),
        ),
        lambda placement: replace(
            placement,
            tensors=(
                replace(
                    placement.tensors[0],
                    partition_dim=None,
                    shard_dims=(0, 1),
                ),
            ),
            fragments=(
                replace(
                    placement.fragments[0],
                    local_shape=(8, 2),
                ),
            ),
        ),
    ],
)
def test_placement_identity_attests_exact_logical_content(mutate) -> None:
    placement = placement_manifest()

    with pytest.raises(ValueError, match="canonical logical content"):
        mutate(placement)


def test_projection_identity_is_stable_across_runtime_restarts() -> None:
    first = RuntimeManifest(
        model_id=MODEL_ID,
        revision=REVISION,
        instance_id="instance-a",
        generation=7,
        lease_id="lease-7",
        tensors=(descriptor(),),
        fragments=(runtime_fragment(fragment_id="runtime-a"),),
    )
    second = replace(
        first,
        instance_id="instance-b",
        generation=8,
        lease_id="lease-8",
        fragments=(
            replace(
                first.fragments[0],
                fragment_id="runtime-b",
                address=0x2000,
                worker_id="worker-b",
                endpoint="worker-b:12345",
                lease_generation=8,
            ),
        ),
    )

    first_placement = placement_manifest_from_runtime_manifest(first)
    second_placement = placement_manifest_from_runtime_manifest(second)

    assert first_placement.placement_id == second_placement.placement_id
    assert first_placement.digest == second_placement.digest
    assert (
        first_placement.fragments[0].placement_fragment_id
        == second_placement.fragments[0].placement_fragment_id
    )


def test_projection_identity_normalizes_single_axis_shard_representations() -> None:
    partitioned = RuntimeManifest(
        model_id=MODEL_ID,
        revision=REVISION,
        instance_id="instance",
        generation=7,
        lease_id="lease-7",
        tensors=(descriptor(shard_dims=None),),
        fragments=(runtime_fragment(),),
    )
    multidim_runtime = replace(
        partitioned,
        tensors=(descriptor(shard_dims=(0,)),),
    )

    partitioned_placement = placement_manifest_from_runtime_manifest(partitioned)
    multidim_placement = placement_manifest_from_runtime_manifest(multidim_runtime)

    assert partitioned_placement.placement_id == multidim_placement.placement_id
    assert partitioned_placement.digest == multidim_placement.digest


def test_runtime_projection_requires_lease_and_known_generation() -> None:
    without_lease = RuntimeManifest(
        model_id=MODEL_ID,
        revision=REVISION,
        instance_id="instance",
        generation=7,
        tensors=(descriptor(),),
        fragments=(runtime_fragment(),),
    )
    without_generation = RuntimeManifest(
        model_id=MODEL_ID,
        revision=REVISION,
        instance_id="instance",
        lease_id="lease",
        tensors=(),
        fragments=(),
    )

    with pytest.raises(ValueError, match="lease_id"):
        runtime_binding_from_runtime_manifest(without_lease)
    with pytest.raises(ValueError, match="generation"):
        runtime_binding_from_runtime_manifest(without_generation)


@pytest.mark.parametrize(
    "project",
    [
        placement_manifest_from_runtime_manifest,
        runtime_binding_from_runtime_manifest,
    ],
)
def test_runtime_projection_rejects_explicit_empty_placement_id(project) -> None:
    runtime = RuntimeManifest(
        model_id=MODEL_ID,
        revision=REVISION,
        instance_id="instance",
        generation=7,
        lease_id="lease-7",
        tensors=(descriptor(),),
        fragments=(runtime_fragment(),),
    )

    with pytest.raises(ValueError, match="placement_id"):
        project(runtime, placement_id="")
