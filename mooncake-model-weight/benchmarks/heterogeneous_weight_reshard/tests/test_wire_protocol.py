from __future__ import annotations

import socket
import struct

import pytest

from benchmarks.heterogeneous_weight_reshard.wire_protocol import (
    WireProtocolError,
    receive_message,
    runtime_manifest_from_wire,
    runtime_manifest_to_wire,
    send_message,
)
from mooncake.model_weight import (
    ParallelRank,
    RuntimeFragment,
    RuntimeManifest,
    TensorDescriptor,
)


def _runtime_manifest() -> RuntimeManifest:
    owner = object()
    return RuntimeManifest(
        model_id="benchmark-model",
        revision="revision-1",
        instance_id="target-d0-t1",
        tensors=(
            TensorDescriptor(
                tensor_id="benchmark-tensor",
                global_shape=(8, 4, 2),
                dtype="uint8",
                itemsize=1,
                partition_dim=None,
                layer_id=3,
                expert_id=7,
                layout_fingerprint="benchmark:contiguous:v1",
                shard_dims=(1,),
            ),
        ),
        fragments=(
            RuntimeFragment(
                fragment_id="target-d0-t1-fragment",
                tensor_id="benchmark-tensor",
                global_offset=(0, 2, 0),
                local_shape=(8, 2, 2),
                address=0x100000,
                nbytes=32,
                worker_id="target-d0-t1",
                endpoint="172.16.1.108:12345",
                device="cuda:0",
                rank=ParallelRank(dp=0, tp=1, pp=2, ep=3),
                lease_generation=11,
                owner=owner,
                aliases=("legacy-name",),
            ),
        ),
        lease_id="target-runtime-lease",
    )


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


def test_runtime_manifest_round_trip_preserves_runtime_facts_without_owner() -> None:
    original = _runtime_manifest()

    restored = runtime_manifest_from_wire(runtime_manifest_to_wire(original))

    assert restored == original
    assert restored.fragments[0].owner is None
    assert restored.fragments[0].endpoint == "172.16.1.108:12345"
    assert restored.fragments[0].address == 0x100000
    assert restored.fragments[0].rank == ParallelRank(dp=0, tp=1, pp=2, ep=3)
    assert restored.fragments[0].lease_generation == 11
    assert restored.lease_id == "target-runtime-lease"


def test_runtime_manifest_wire_schema_rejects_unknown_fields() -> None:
    raw = runtime_manifest_to_wire(_runtime_manifest())
    raw["unexpected"] = True

    with pytest.raises(WireProtocolError, match="runtime manifest fields"):
        runtime_manifest_from_wire(raw)


def test_runtime_manifest_wire_schema_rejects_non_runtime_fragments() -> None:
    raw = runtime_manifest_to_wire(_runtime_manifest())
    raw["fragments"][0]["address"] = 0

    with pytest.raises(WireProtocolError, match="invalid runtime manifest"):
        runtime_manifest_from_wire(raw)
