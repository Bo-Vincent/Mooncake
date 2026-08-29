"""Bounded control-channel protocol for cross-host benchmark roles."""

from __future__ import annotations

import json
import socket
import struct
from typing import Mapping, Sequence

from mooncake.reshard.weight import (
    RuntimeBindingFragment,
    WeightPlacementManifest,
    WeightRuntimeBindingManifest,
    weight_placement_from_json,
    weight_placement_to_json,
)


DEFAULT_MAX_MESSAGE_BYTES = 16 * 1024 * 1024
_LENGTH = struct.Struct("!I")
_PLACEMENT_FIELDS = frozenset(
    {
        "resource_kind",
        "resource_id",
        "revision",
        "weight_generation",
        "placement_set_id",
        "placement_id",
        "topology",
        "tensors",
        "parts",
    }
)
_BINDING_FIELDS = frozenset(
    {
        "resource_id",
        "revision",
        "placement_id",
        "placement_digest",
        "instance_id",
        "participant_id",
        "generation",
        "lease_id",
        "fragments",
    }
)
_BINDING_FRAGMENT_FIELDS = frozenset(
    {
        "placement_fragment_id",
        "fragment_id",
        "address",
        "nbytes",
        "worker_id",
        "endpoint",
        "device",
        "itemsize",
        "local_shape",
        "strides_bytes",
        "storage_address",
        "storage_nbytes",
        "storage_offset_bytes",
    }
)


class WireProtocolError(ValueError):
    """The control peer sent a malformed or incomplete message."""


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        try:
            chunk = connection.recv(size - len(chunks))
        except OSError as error:
            raise WireProtocolError(f"control receive failed: {error}") from error
        if not chunk:
            raise WireProtocolError(
                "control connection closed before message completed"
            )
        chunks.extend(chunk)
    return bytes(chunks)


def send_message(
    connection: socket.socket,
    message: Mapping[str, object],
    *,
    max_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
) -> None:
    if not isinstance(message, Mapping):
        raise WireProtocolError("control message must be an object")
    try:
        payload = json.dumps(
            message,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise WireProtocolError(
            f"control message is not JSON serializable: {error}"
        ) from error
    if not payload or len(payload) > max_bytes:
        raise WireProtocolError(
            f"control message size {len(payload)} exceeds limit {max_bytes}"
        )
    try:
        connection.sendall(_LENGTH.pack(len(payload)) + payload)
    except OSError as error:
        raise WireProtocolError(f"control send failed: {error}") from error


def receive_message(
    connection: socket.socket,
    *,
    max_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
) -> dict[str, object]:
    payload_size = _LENGTH.unpack(_receive_exact(connection, _LENGTH.size))[0]
    if payload_size == 0 or payload_size > max_bytes:
        raise WireProtocolError(
            f"control message size {payload_size} exceeds limit {max_bytes}"
        )
    payload = _receive_exact(connection, payload_size)
    try:
        message = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WireProtocolError(
            f"control message is not valid JSON: {error}"
        ) from error
    if not isinstance(message, dict) or not all(
        isinstance(key, str) for key in message
    ):
        raise WireProtocolError("control message must be an object")
    return message


def _object(value: object, fields: frozenset[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise WireProtocolError(f"{label} fields do not match wire schema")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise WireProtocolError(f"{label} must be an array")
    return value


def placement_manifest_to_wire(
    placement: WeightPlacementManifest,
) -> dict[str, object]:
    if not isinstance(placement, WeightPlacementManifest):
        raise WireProtocolError("expected WeightPlacementManifest")
    return json.loads(weight_placement_to_json(placement))


def placement_manifest_from_wire(value: object) -> WeightPlacementManifest:
    raw = _object(value, _PLACEMENT_FIELDS, "placement manifest")
    try:
        return weight_placement_from_json(
            json.dumps(raw, sort_keys=True, separators=(",", ":"))
        )
    except (TypeError, ValueError) as error:
        raise WireProtocolError(f"invalid placement manifest: {error}") from error


def runtime_binding_to_wire(
    binding: WeightRuntimeBindingManifest,
) -> dict[str, object]:
    if not isinstance(binding, WeightRuntimeBindingManifest):
        raise WireProtocolError("expected WeightRuntimeBindingManifest")
    return {
        "resource_id": binding.resource_id,
        "revision": binding.revision,
        "placement_id": binding.placement_id,
        "placement_digest": binding.placement_digest,
        "instance_id": binding.instance_id,
        "participant_id": binding.participant_id,
        "generation": binding.generation,
        "lease_id": binding.lease_id,
        "fragments": [
            {
                "placement_fragment_id": fragment.placement_fragment_id,
                "fragment_id": fragment.fragment_id,
                "address": fragment.address,
                "nbytes": fragment.nbytes,
                "worker_id": fragment.worker_id,
                "endpoint": fragment.endpoint,
                "device": fragment.device,
                "itemsize": fragment.itemsize,
                "local_shape": list(fragment.local_shape),
                "strides_bytes": list(fragment.strides_bytes),
                "storage_address": fragment.storage_address,
                "storage_nbytes": fragment.storage_nbytes,
                "storage_offset_bytes": fragment.storage_offset_bytes,
            }
            for fragment in binding.fragments
        ],
    }


def runtime_binding_from_wire(value: object) -> WeightRuntimeBindingManifest:
    raw = _object(value, _BINDING_FIELDS, "runtime binding")
    try:
        fragments = tuple(
            RuntimeBindingFragment(
                placement_fragment_id=fragment["placement_fragment_id"],
                fragment_id=fragment["fragment_id"],
                address=fragment["address"],
                nbytes=fragment["nbytes"],
                worker_id=fragment["worker_id"],
                endpoint=fragment["endpoint"],
                device=fragment["device"],
                itemsize=fragment["itemsize"],
                local_shape=tuple(
                    _sequence(fragment["local_shape"], "runtime local shape")
                ),
                strides_bytes=tuple(
                    _sequence(fragment["strides_bytes"], "runtime strides")
                ),
                storage_address=fragment["storage_address"],
                storage_nbytes=fragment["storage_nbytes"],
                storage_offset_bytes=fragment["storage_offset_bytes"],
            )
            for fragment in (
                _object(item, _BINDING_FRAGMENT_FIELDS, "runtime binding fragment")
                for item in _sequence(raw["fragments"], "runtime binding fragments")
            )
        )
        return WeightRuntimeBindingManifest(
            resource_id=raw["resource_id"],
            revision=raw["revision"],
            placement_id=raw["placement_id"],
            placement_digest=raw["placement_digest"],
            instance_id=raw["instance_id"],
            participant_id=raw["participant_id"],
            generation=raw["generation"],
            lease_id=raw["lease_id"],
            fragments=fragments,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise WireProtocolError(f"invalid runtime binding: {error}") from error
