from __future__ import annotations

import socket
import struct

import pytest

from benchmarks.heterogeneous_weight_reshard.wire_protocol import (
    WireProtocolError,
    placement_manifest_from_wire,
    placement_manifest_to_wire,
    receive_message,
    runtime_binding_from_wire,
    runtime_binding_to_wire,
    send_message,
)
from mooncake.reshard.weight import (
    ParallelRank,
    ParallelTopology,
    PlacementFragment,
    RuntimeBindingFragment,
    TensorDescriptor,
    OwnershipAxis,
    TopologyParticipant,
    WeightPlacementManifest,
    WeightPlacementPart,
    WeightRuntimeBindingManifest,
)


def _runtime_topology() -> (
    tuple[
        WeightPlacementManifest,
        WeightRuntimeBindingManifest,
    ]
):
    owner = object()
    participant_id = "target-d0-t1"
    rank = ParallelRank(dp=0, tp=1, pp=2, ep=3)
    topology = ParallelTopology(
        tp_size=2,
        pp_size=3,
        ep_size=4,
        dp_size=1,
        participants=(TopologyParticipant(participant_id=participant_id, rank=rank),),
    )
    descriptor = TensorDescriptor(
        tensor_id="benchmark-tensor",
        global_shape=(8, 2, 2),
        dtype="uint8",
        itemsize=1,
        layer_id=3,
        expert_id=7,
        layout_fingerprint="benchmark:contiguous:v1",
        parallel_axes=(
            OwnershipAxis(kind="pp"),
            OwnershipAxis(kind="ep"),
            OwnershipAxis(kind="tp"),
        ),
        shard_dims=(),
    )
    fragment = PlacementFragment(
        placement_fragment_id="target-d0-t1-placement-fragment",
        tensor_id="benchmark-tensor",
        global_offset=(0, 0, 0),
        local_shape=(8, 2, 2),
        nbytes=32,
        rank=rank,
        aliases=(),
    )
    placement_set_id = "wire-placement"
    placement = WeightPlacementManifest(
        resource_id="benchmark-model",
        revision="revision-1",
        weight_generation=0,
        placement_set_id=placement_set_id,
        topology=topology,
        parts=(
            WeightPlacementPart(
                resource_id="benchmark-model",
                revision="revision-1",
                weight_generation=0,
                placement_set_id=placement_set_id,
                topology_id=topology.topology_id,
                participant_id=participant_id,
                rank=rank,
                tensors=(descriptor,),
                fragments=(fragment,),
            ),
        ),
    )
    binding = WeightRuntimeBindingManifest(
        resource_id="benchmark-model",
        revision="revision-1",
        placement_id=placement.placement_id,
        placement_digest=placement.digest,
        instance_id="target-d0-t1",
        participant_id=participant_id,
        generation=11,
        lease_id="target-runtime-lease",
        fragments=(
            RuntimeBindingFragment(
                placement_fragment_id="target-d0-t1-placement-fragment",
                fragment_id="target-d0-t1-runtime-fragment",
                address=0x100020,
                nbytes=32,
                worker_id="target-d0-t1",
                endpoint="172.16.1.108:12345",
                device="cuda:0",
                itemsize=1,
                local_shape=(8, 2, 2),
                strides_bytes=(4, 2, 1),
                storage_address=0x100000,
                storage_nbytes=96,
                storage_offset_bytes=32,
                owner=owner,
            ),
        ),
    )
    return placement, binding


def test_message_round_trip_uses_length_prefixed_json() -> None:
    sender, receiver = socket.socketpair()
    try:
        send_message(sender, {"type": "ready", "sequence": 7})
        assert receive_message(receiver) == {"sequence": 7, "type": "ready"}
    finally:
        sender.close()
        receiver.close()


def test_receive_rejects_message_larger_than_limit_before_reading_payload() -> None:
    sender, receiver = socket.socketpair()
    try:
        sender.sendall(struct.pack("!I", 1025))
        with pytest.raises(WireProtocolError, match="exceeds limit"):
            receive_message(receiver, max_bytes=1024)
    finally:
        sender.close()
        receiver.close()


def test_receive_rejects_truncated_message() -> None:
    sender, receiver = socket.socketpair()
    sender.sendall(struct.pack("!I", 4) + b"{}")
    sender.close()
    try:
        with pytest.raises(WireProtocolError, match="connection closed"):
            receive_message(receiver)
    finally:
        receiver.close()


def test_placement_and_binding_round_trip_remain_independent() -> None:
    placement, binding = _runtime_topology()

    restored_placement = placement_manifest_from_wire(
        placement_manifest_to_wire(placement)
    )
    restored_binding = runtime_binding_from_wire(runtime_binding_to_wire(binding))

    assert restored_placement == placement
    assert restored_placement.fragments[0].rank == ParallelRank(dp=0, tp=1, pp=2, ep=3)
    assert restored_binding == binding
    assert restored_binding.fragments[0].owner is None
    assert restored_binding.fragments[0].endpoint == "172.16.1.108:12345"
    assert restored_binding.fragments[0].address == 0x100020
    assert restored_binding.fragments[0].itemsize == 1
    assert restored_binding.fragments[0].local_shape == (8, 2, 2)
    assert restored_binding.fragments[0].strides_bytes == (4, 2, 1)
    assert restored_binding.fragments[0].storage_address == 0x100000
    assert restored_binding.fragments[0].storage_nbytes == 96
    assert restored_binding.fragments[0].storage_offset_bytes == 32
    assert restored_binding.generation == 11
    assert restored_binding.lease_id == "target-runtime-lease"


def test_runtime_binding_wire_schema_rejects_unknown_fields() -> None:
    _, binding = _runtime_topology()
    raw = runtime_binding_to_wire(binding)
    raw["unexpected"] = True

    with pytest.raises(WireProtocolError, match="runtime binding fields"):
        runtime_binding_from_wire(raw)


@pytest.mark.parametrize(
    "field",
    (
        "itemsize",
        "local_shape",
        "strides_bytes",
        "storage_address",
        "storage_nbytes",
        "storage_offset_bytes",
    ),
)
def test_runtime_binding_wire_schema_rejects_missing_storage_evidence(
    field: str,
) -> None:
    _, binding = _runtime_topology()
    raw = runtime_binding_to_wire(binding)
    del raw["fragments"][0][field]

    with pytest.raises(WireProtocolError, match="runtime binding fragment fields"):
        runtime_binding_from_wire(raw)


def test_runtime_binding_wire_schema_rejects_invalid_runtime_fragment() -> None:
    _, binding = _runtime_topology()
    raw = runtime_binding_to_wire(binding)
    raw["fragments"][0]["address"] = 0

    with pytest.raises(WireProtocolError, match="invalid runtime binding"):
        runtime_binding_from_wire(raw)
