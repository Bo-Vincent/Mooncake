from __future__ import annotations

from mooncake.model_weight import (
    ParallelRank,
    PlacementFragment,
    PlacementManifest,
    RuntimeBindingFragment,
    RuntimeBindingManifest,
    RuntimeFragment,
    TensorDescriptor,
)


MODEL_ID = "model"
REVISION = "revision"


def descriptor(**overrides) -> TensorDescriptor:
    values = {
        "tensor_id": "layers.2.experts.3.w1",
        "global_shape": (8, 4),
        "dtype": "bfloat16",
        "itemsize": 2,
        "partition_dim": 0,
        "layer_id": 2,
        "expert_id": 3,
        "layout_fingerprint": "test:qwen:bf16:v1",
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
        "worker_id": "worker-0",
        "endpoint": "worker-0:12345",
        "device": "cuda:0",
        "rank": ParallelRank(dp=0, tp=0, pp=1, ep=1),
        "lease_generation": 7,
    }
    values.update(overrides)
    return RuntimeFragment(**values)


def placement_fragment(**overrides) -> PlacementFragment:
    values = {
        "placement_fragment_id": "placement-0",
        "tensor_id": "layers.2.experts.3.w1",
        "global_offset": (0, 0),
        "local_shape": (4, 4),
        "nbytes": 32,
        "rank": ParallelRank(dp=0, tp=0, pp=1, ep=1),
    }
    values.update(overrides)
    return PlacementFragment(**values)


def placement_manifest(**overrides) -> PlacementManifest:
    values = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "placement_id": None,
        "tensors": (descriptor(),),
        "fragments": (placement_fragment(),),
    }
    values.update(overrides)
    return PlacementManifest(**values)


def binding_fragment(**overrides) -> RuntimeBindingFragment:
    values = {
        "placement_fragment_id": "placement-0",
        "fragment_id": "runtime-0",
        "address": 0x1000,
        "nbytes": 32,
        "worker_id": "worker-0",
        "endpoint": "worker-0:12345",
        "device": "cuda:0",
    }
    values.update(overrides)
    return RuntimeBindingFragment(**values)


def binding_manifest(
    *,
    placement: PlacementManifest | None = None,
    **overrides,
) -> RuntimeBindingManifest:
    logical = placement or placement_manifest()
    values = {
        "model_id": logical.model_id,
        "revision": logical.revision,
        "placement_id": logical.placement_id,
        "placement_digest": logical.digest,
        "instance_id": "instance",
        "generation": 7,
        "lease_id": "lease-7",
        "fragments": (binding_fragment(),),
    }
    values.update(overrides)
    return RuntimeBindingManifest(**values)


def runtime_inventory_tensor(**overrides) -> dict:
    values = {
        "fragment_id": "runtime-0",
        "placement_fragment_id": "placement-0",
        "tensor_id": "layers.2.experts.3.w1",
        "global_shape": (8, 4),
        "global_offset": (0, 0),
        "local_shape": (4, 4),
        "dtype": "bfloat16",
        "itemsize": 2,
        "partition_dim": 0,
        "layer_id": 2,
        "expert_id": 3,
        "layout_fingerprint": "test:qwen:bf16:v1",
        "address": 0x1000,
        "nbytes": 32,
        "worker_id": "worker-0",
        "endpoint": "worker-0:12345",
        "device": "cuda:0",
        "rank": {"dp": 0, "tp": 0, "pp": 1, "ep": 1},
        "lease_generation": 7,
        "aliases": (),
        "is_contiguous": True,
        "stride": (4, 1),
        "storage_offset": 0,
        "byte_offset": 0,
    }
    values.update(overrides)
    return values


def runtime_inventory(**overrides) -> dict:
    values = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "instance_id": "instance",
        "placement_id": None,
        "generation": 7,
        "lease_id": "lease-7",
        "tensors": (runtime_inventory_tensor(),),
    }
    values.update(overrides)
    return values
