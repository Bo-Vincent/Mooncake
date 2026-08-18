from __future__ import annotations

from dataclasses import replace
from inspect import signature

import pytest

from mooncake.reshard.weight import store
from mooncake.reshard.weight._store import (
    PayloadStoreOperations as ExportedPayloadStoreOperations,
)
from mooncake.reshard.weight._store import (
    WeightUploadService as ExportedWeightUploadService,
)
from mooncake.reshard.weight._store import (
    WeightUploadSession as ExportedWeightUploadSession,
)
from mooncake.reshard.weight._store.client import WeightStore, WeightStoreError
from mooncake.reshard.weight._store.contracts import (
    UploadOperation,
    UploadReceipt,
    WeightLoadPlan,
    WeightUploadPlan,
)
from mooncake.reshard.weight._store.payload import PayloadStoreOperations
from mooncake.reshard.weight._store.session import WeightUploadSession
from mooncake.reshard.weight._store.upload import WeightUploadService
from mooncake.reshard.weight.manifest import (
    ParallelRank,
    ParallelTopology,
    PlacementFragment,
    RuntimeBindingFragment,
    TensorDescriptor,
    SplitAxis,
    TopologyParticipant,
    WeightPlacementManifest,
    WeightRuntimeBindingManifest,
)


def manifest_pair():
    tensor = TensorDescriptor(
        tensor_id="layers.0.weight",
        global_shape=(4,),
        dtype="uint8",
        itemsize=1,
        shard_dims=(0,),
        layout_fingerprint="test:contiguous:v1",
        parallel_axes=(SplitAxis(kind="tp", dim=0),),
    )
    rank = ParallelRank()
    placement = WeightPlacementManifest.from_fragments(
        resource_id="qwen",
        revision="step-1",
        weight_generation=3,
        placement_set_id="module-layout-test",
        topology=ParallelTopology(
            tp_size=1,
            pp_size=1,
            ep_size=1,
            dp_size=1,
            participants=(TopologyParticipant(participant_id="worker-0", rank=rank),),
        ),
        tensors=(tensor,),
        fragments=(
            PlacementFragment(
                placement_fragment_id="placement-0",
                tensor_id=tensor.tensor_id,
                global_offset=(0,),
                local_shape=(4,),
                nbytes=4,
                rank=rank,
            ),
        ),
    )
    binding = WeightRuntimeBindingManifest(
        resource_id=placement.resource_id,
        revision=placement.revision,
        placement_id=placement.placement_id,
        placement_digest=placement.digest,
        instance_id="worker-0",
        participant_id="worker-0",
        generation=7,
        lease_id="lease-7",
        fragments=(
            RuntimeBindingFragment(
                placement_fragment_id="placement-0",
                fragment_id="runtime-0",
                address=0x1000,
                nbytes=4,
                worker_id="worker-0",
                endpoint="worker-0:12345",
                device="cuda:0",
                itemsize=1,
                local_shape=(4,),
                strides_bytes=(1,),
                storage_address=0x1000,
                storage_nbytes=4,
                storage_offset_bytes=0,
            ),
        ),
    )
    return placement, binding


def test_store_responsibility_modules_preserve_public_identity() -> None:
    assert store.WeightStore is WeightStore
    assert store.WeightStoreError is WeightStoreError
    assert store.UploadOperation is UploadOperation
    assert store.UploadReceipt is UploadReceipt
    assert store.WeightLoadPlan is WeightLoadPlan
    assert store.WeightUploadPlan is WeightUploadPlan


def test_store_internal_services_have_one_definition() -> None:
    assert ExportedPayloadStoreOperations is PayloadStoreOperations
    assert ExportedWeightUploadSession is WeightUploadSession
    assert ExportedWeightUploadService is WeightUploadService


def test_store_contract_uses_explicit_placement_and_binding() -> None:
    prepare_parameters = tuple(signature(WeightStore.prepare_upload).parameters)
    assert prepare_parameters[:4] == (
        "self",
        "source_placement",
        "source_bindings",
        "namespace",
    )

    upload_parameters = tuple(signature(WeightStore.upload).parameters)
    assert upload_parameters[:7] == (
        "self",
        "plan",
        "source_placement",
        "source_binding",
        "source_worker_id",
        "source_allocation_guards",
        "registration_lease",
    )

    plan_load_parameters = tuple(signature(WeightStore.plan_load).parameters)
    assert plan_load_parameters[:4] == (
        "self",
        "manifest",
        "target_placement",
        "target_bindings",
    )

    load_parameters = tuple(signature(WeightStore.load).parameters)
    assert load_parameters[:7] == (
        "self",
        "plan",
        "target_placement",
        "target_binding",
        "target_worker_id",
        "target_allocation_guards",
        "registration_lease",
    )


def test_store_rejects_binding_for_different_placement_digest() -> None:
    placement, binding = manifest_pair()

    with pytest.raises(WeightStoreError, match="placement digest"):
        WeightStore(object()).prepare_upload(
            placement,
            (replace(binding, placement_digest="0" * 64),),
        )
