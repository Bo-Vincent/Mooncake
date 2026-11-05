from __future__ import annotations

from dataclasses import replace
from inspect import signature
from types import SimpleNamespace

import pytest

import mooncake.reshard.weight as weight
import mooncake.reshard.weight._store as internal_store
from mooncake.reshard.weight import store
from mooncake.reshard.weight._store import (
    PayloadStoreOperations as ExportedPayloadStoreOperations,
)
from mooncake.reshard.weight._store import (
    WeightUploadService as ExportedWeightUploadService,
)
from mooncake.reshard.weight._store import (
    WeightUploadTransaction as ExportedWeightUploadTransaction,
)
from mooncake.reshard.weight._store.store import WeightStore, WeightStoreError
from mooncake.reshard.weight._store.snapshot import (
    WeightSnapshotAdapter,
    WeightSnapshotDescriptor,
)
from mooncake.reshard.weight._store.writer import WeightStoreWriter
from mooncake.reshard.weight._store.backend import default_config_factory
from mooncake.reshard.weight._store.payload import PayloadStoreOperations
from mooncake.reshard.weight._store.transaction import WeightUploadTransaction
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
    for internal_name in (
        "UploadOperation",
        "UploadReceipt",
        "StoreRegistrationLease",
        "WeightLoadPlan",
        "WeightUploadPlan",
    ):
        assert not hasattr(store, internal_name)
        assert not hasattr(weight, internal_name)


def test_store_internal_modules_match_their_responsibilities() -> None:
    assert WeightStore.__module__.endswith("._store.store")
    assert WeightSnapshotDescriptor.__module__.endswith("._store.snapshot")
    assert WeightSnapshotAdapter.__module__.endswith("._store.snapshot")
    assert WeightStoreWriter.__module__.endswith("._store.writer")


def test_store_internal_services_have_one_definition() -> None:
    assert ExportedPayloadStoreOperations is PayloadStoreOperations
    assert ExportedWeightUploadTransaction is WeightUploadTransaction
    assert ExportedWeightUploadService is WeightUploadService


def test_store_contract_keeps_payload_operations_private() -> None:
    for name in (
        "_weight_put_plan",
        "_weight_put_payload",
        "_weight_put_commit",
        "_weight_put_abort",
        "_weight_put_finalize",
        "_weight_get_plan",
        "_weight_get_payload",
    ):
        assert callable(getattr(WeightStore, name))
    for unmanaged_name in (
        "weight_put_plan",
        "weight_put_payload",
        "weight_put_commit",
        "weight_put_abort",
        "weight_put_finalize",
        "weight_get_plan",
        "weight_get_payload",
        "plan_upload",
        "begin_weight_snapshot",
        "upload",
        "abort_upload",
        "finalize_upload_transaction",
        "commit_upload",
        "load_manifest",
        "plan_load",
        "load",
        "prepare_upload",
        "commit",
        "finalize_upload_session",
    ):
        assert not hasattr(WeightStore, unmanaged_name)

    plan_parameters = tuple(signature(WeightStore._weight_put_plan).parameters)
    assert plan_parameters[:4] == (
        "self",
        "source_placement",
        "source_bindings",
        "namespace",
    )

    upload_parameters = tuple(signature(WeightStore._weight_put_payload).parameters)
    assert upload_parameters[:7] == (
        "self",
        "plan",
        "source_placement",
        "source_binding",
        "source_worker_id",
        "source_allocation_guards",
        "registration_lease",
    )

    plan_load_parameters = tuple(signature(WeightStore._weight_get_plan).parameters)
    assert plan_load_parameters[:4] == (
        "self",
        "manifest",
        "target_placement",
        "target_bindings",
    )

    load_parameters = tuple(signature(WeightStore._weight_get_payload).parameters)
    assert load_parameters[:7] == (
        "self",
        "plan",
        "target_placement",
        "target_binding",
        "target_worker_id",
        "target_allocation_guards",
        "registration_lease",
    )


def test_unmanaged_entrypoints_are_not_exported() -> None:
    for module in (weight, store, internal_store):
        assert not hasattr(module, "begin_weight_snapshot")
        assert not hasattr(module, "plan_weight_upload")

    assert "managed" not in signature(WeightStoreWriter).parameters
    assert "upsert" not in signature(WeightStoreWriter).parameters
    assert not hasattr(WeightStoreWriter, "write_tensor")
    assert not hasattr(WeightStoreWriter, "plan")
    assert isinstance(WeightStoreWriter.identity, property)
    assert isinstance(WeightStoreWriter.request_id, property)


def test_weight_upsert_public_signature_is_generation_aware() -> None:
    assert tuple(signature(WeightStore.weight_upsert).parameters) == (
        "self",
        "snapshot",
        "adapter",
        "replacing",
        "expected_metadata_generation",
        "mode",
        "tenant_id",
        "policy",
        "request_id",
    )


def test_store_rejects_binding_for_different_placement_digest() -> None:
    placement, binding = manifest_pair()

    with pytest.raises(WeightStoreError, match="placement digest"):
        WeightStore(object())._weight_put_plan(
            placement,
            (replace(binding, placement_digest="0" * 64),),
        )


def test_native_store_requires_group_semantics_binding(monkeypatch) -> None:
    class ReplicateConfigWithoutGroups:
        __slots__ = ("with_hard_pin", "data_type")

        def __init__(self) -> None:
            self.with_hard_pin = False
            self.data_type = None

    class ObjectDataType:
        WEIGHT = "weight"
        METADATA = "metadata"

    native_store_module = SimpleNamespace(
        ReplicateConfig=ReplicateConfigWithoutGroups,
        ObjectDataType=ObjectDataType,
    )

    monkeypatch.setattr(
        "mooncake.reshard.weight._store.backend.import_module",
        lambda _: native_store_module,
    )

    with pytest.raises(WeightStoreError, match="ReplicateConfig.group_ids"):
        default_config_factory(("weight-group",), "payload")
