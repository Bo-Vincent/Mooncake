from __future__ import annotations

import inspect
import typing
import pytest

import mooncake.model_weight as model_weight

from mooncake.model_weight import (
    TensorDescriptor,
)

from .helpers import (
    descriptor,
)


def test_public_api_is_minimal_and_explicit() -> None:
    assert model_weight.__all__ == [
        "MODEL_WEIGHT_CAPABILITIES",
        "supports_model_weight_capability",
        "ParallelRank",
        "PlacementFragment",
        "PlacementManifest",
        "RuntimeBindingFragment",
        "RuntimeBindingManifest",
        "RuntimeFragment",
        "RuntimeManifest",
        "SourcePlacementManifest",
        "StoredFragment",
        "TargetPlacementManifest",
        "TensorDescriptor",
        "WeightManifest",
        "bind_runtime_manifest",
        "placement_manifest_from_runtime_manifest",
        "runtime_binding_from_runtime_manifest",
        "CopyRange",
        "ExecutorTransferPlan",
        "LogicalTransferPlan",
        "PipelineRouteGroup",
        "PlacementExecutorPlan",
        "TransferPlan",
        "TransferRegion",
        "bind_logical_transfer_plan",
        "plan_placement_transfer",
        "plan_placement_transfer_to_local_target",
        "plan_runtime_transfer",
        "plan_runtime_transfer_to_local_target",
        "plan_runtime_transfer_to_local_target_placement",
        "plan_runtime_transfer_to_target_placements",
        "plan_stored_transfer",
        "plan_stored_transfer_to_target_placements",
        "UploadOperation",
        "UploadReceipt",
        "WeightLoadPlan",
        "WeightStore",
        "WeightStoreError",
        "WeightUploadPlan",
        "DirectReadReceipt",
        "DirectTransferReceipt",
        "MemoryRegistrationLease",
        "MooncakeTransferEngineReader",
        "MooncakeTransferEngineSink",
        "TransferCompletionUnknownError",
        "TransferEngineError",
    ]


def test_public_type_hints_resolve() -> None:
    for name in model_weight.__all__:
        value = getattr(model_weight, name)
        if inspect.isclass(value):
            targets = (value, value.__init__)
        elif inspect.isfunction(value):
            targets = (value,)
        else:
            continue
        for target in targets:
            typing.get_type_hints(target)


def test_partition_dim_and_single_axis_shard_dims_are_normalized() -> None:
    single_axis = descriptor()
    multidim_descriptor = descriptor(
        partition_dim=None,
        shard_dims=(0, 1),
        global_shape=(8, 16, 32),
    )

    assert single_axis.effective_shard_dims == (0,)
    assert multidim_descriptor.effective_shard_dims == (0, 1)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"global_shape": ()}, "global_shape"),
        ({"global_shape": (8.0, 4)}, "integer"),
        ({"itemsize": True}, "integer"),
        ({"partition_dim": 2}, "out of range"),
        ({"partition_dim": 0, "shard_dims": (1,)}, "conflicts"),
        ({"partition_dim": None, "shard_dims": (0, 0)}, "duplicates"),
        ({"partition_dim": None, "shard_dims": (1, 0)}, "sorted"),
        ({"partition_dim": None, "shard_dims": (2,)}, "out-of-range"),
        ({"partition_dim": None, "shard_dims": (True,)}, "integer"),
        ({"layout_fingerprint": ""}, "layout_fingerprint"),
    ],
)
def test_tensor_descriptor_rejects_invalid_schema(
    overrides: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        descriptor(**overrides)


@pytest.mark.parametrize(
    "shape",
    [
        {8: None, 4: None},
        {8, 4},
        frozenset((8, 4)),
        (extent for extent in (8, 4)),
    ],
)
def test_tensor_descriptor_rejects_unordered_or_one_shot_shape(shape) -> None:
    with pytest.raises(ValueError, match="global_shape must contain integers"):
        descriptor(global_shape=shape)


def test_tensor_descriptor_accepts_ordered_shape_sequence() -> None:
    assert descriptor(global_shape=[8, 4]).global_shape == (8, 4)


def test_tensor_descriptor_requires_explicit_layout_fingerprint() -> None:
    values = {
        "tensor_id": "weight",
        "global_shape": (4, 4),
        "dtype": "bfloat16",
        "itemsize": 2,
        "partition_dim": 0,
    }

    with pytest.raises(TypeError, match="layout_fingerprint"):
        TensorDescriptor(**values)
