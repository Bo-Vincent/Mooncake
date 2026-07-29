"""Bounded control-channel protocol for cross-host benchmark roles."""

from __future__ import annotations

import json
import socket
import struct
from typing import Mapping, Sequence

from mooncake.model_weight import (
    ParallelRank,
    RuntimeFragment,
    RuntimeManifest,
    TensorDescriptor,
)


DEFAULT_MAX_MESSAGE_BYTES = 16 * 1024 * 1024
_LENGTH = struct.Struct("!I")
_MANIFEST_FIELDS = frozenset(
    {
        "model_id",
        "revision",
        "instance_id",
        "tensors",
        "fragments",
        "lease_id",
    }
)
_TENSOR_FIELDS = frozenset(
    {
        "tensor_id",
        "global_shape",
        "dtype",
        "itemsize",
        "partition_dim",
        "layer_id",
        "expert_id",
        "layout_fingerprint",
        "shard_dims",
    }
)
_FRAGMENT_FIELDS = frozenset(
    {
        "fragment_id",
        "tensor_id",
        "global_offset",
        "local_shape",
        "address",
        "nbytes",
        "worker_id",
        "endpoint",
        "rank",
        "lease_generation",
        "aliases",
    }
)
_RANK_FIELDS = frozenset({"dp", "tp", "pp", "ep"})


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


def runtime_manifest_to_wire(manifest: RuntimeManifest) -> dict[str, object]:
    if not isinstance(manifest, RuntimeManifest):
        raise WireProtocolError("expected RuntimeManifest")
    return {
        "model_id": manifest.model_id,
        "revision": manifest.revision,
        "instance_id": manifest.instance_id,
        "tensors": [
            {
                "tensor_id": tensor.tensor_id,
                "global_shape": list(tensor.global_shape),
                "dtype": tensor.dtype,
                "itemsize": tensor.itemsize,
                "partition_dim": tensor.partition_dim,
                "layer_id": tensor.layer_id,
                "expert_id": tensor.expert_id,
                "layout_fingerprint": tensor.layout_fingerprint,
                "shard_dims": (
                    None if tensor.shard_dims is None else list(tensor.shard_dims)
                ),
            }
            for tensor in manifest.tensors
        ],
        "fragments": [
            {
                "fragment_id": fragment.fragment_id,
                "tensor_id": fragment.tensor_id,
                "global_offset": list(fragment.global_offset),
                "local_shape": list(fragment.local_shape),
                "address": fragment.address,
                "nbytes": fragment.nbytes,
                "worker_id": fragment.worker_id,
                "endpoint": fragment.endpoint,
                "rank": {
                    "dp": fragment.rank.dp,
                    "tp": fragment.rank.tp,
                    "pp": fragment.rank.pp,
                    "ep": fragment.rank.ep,
                },
                "lease_generation": fragment.lease_generation,
                "aliases": list(fragment.aliases),
            }
            for fragment in manifest.fragments
        ],
        "lease_id": manifest.lease_id,
    }


def _object(value: object, fields: frozenset[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise WireProtocolError(f"{label} fields do not match wire schema")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise WireProtocolError(f"{label} must be an array")
    return value


def runtime_manifest_from_wire(value: object) -> RuntimeManifest:
    raw = _object(value, _MANIFEST_FIELDS, "runtime manifest")
    try:
        tensors = []
        for value_tensor in _sequence(raw["tensors"], "runtime manifest tensors"):
            tensor = _object(value_tensor, _TENSOR_FIELDS, "tensor descriptor")
            shard_dims = tensor["shard_dims"]
            tensors.append(
                TensorDescriptor(
                    tensor_id=tensor["tensor_id"],
                    global_shape=tuple(
                        _sequence(tensor["global_shape"], "global_shape")
                    ),
                    dtype=tensor["dtype"],
                    itemsize=tensor["itemsize"],
                    partition_dim=tensor["partition_dim"],
                    layer_id=tensor["layer_id"],
                    expert_id=tensor["expert_id"],
                    layout_fingerprint=tensor["layout_fingerprint"],
                    shard_dims=(
                        None
                        if shard_dims is None
                        else tuple(_sequence(shard_dims, "shard_dims"))
                    ),
                )
            )

        fragments = []
        for value_fragment in _sequence(raw["fragments"], "runtime manifest fragments"):
            fragment = _object(value_fragment, _FRAGMENT_FIELDS, "runtime fragment")
            rank = _object(fragment["rank"], _RANK_FIELDS, "parallel rank")
            fragments.append(
                RuntimeFragment(
                    fragment_id=fragment["fragment_id"],
                    tensor_id=fragment["tensor_id"],
                    global_offset=tuple(
                        _sequence(fragment["global_offset"], "global_offset")
                    ),
                    local_shape=tuple(
                        _sequence(fragment["local_shape"], "local_shape")
                    ),
                    address=fragment["address"],
                    nbytes=fragment["nbytes"],
                    worker_id=fragment["worker_id"],
                    endpoint=fragment["endpoint"],
                    device="cuda:0",
                    rank=ParallelRank(
                        dp=rank["dp"],
                        tp=rank["tp"],
                        pp=rank["pp"],
                        ep=rank["ep"],
                    ),
                    lease_generation=fragment["lease_generation"],
                    aliases=tuple(_sequence(fragment["aliases"], "aliases")),
                )
            )
        return RuntimeManifest(
            model_id=raw["model_id"],
            revision=raw["revision"],
            instance_id=raw["instance_id"],
            tensors=tuple(tensors),
            fragments=tuple(fragments),
            lease_id=raw["lease_id"],
        )
    except WireProtocolError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise WireProtocolError(f"invalid runtime manifest: {error}") from error
