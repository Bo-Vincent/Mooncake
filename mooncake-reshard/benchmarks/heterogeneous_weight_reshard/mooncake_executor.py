"""Cross-host Mooncake benchmark contracts and execution helpers."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from math import prod
from statistics import fmean
from typing import Callable, Iterator, Mapping, Sequence

from mooncake.reshard.weight import (
    PlacementFragment,
    WeightPlacementManifest,
    WeightRuntimeBindingManifest,
    validate_runtime_bindings,
)
from mooncake.reshard.weight.planner import (
    TransferPlan,
)
from mooncake.reshard.weight.te import MemoryRegistrationLease

from .case_spec import BenchmarkCase, MeshSpec
from .wire_protocol import (
    placement_manifest_from_wire,
    placement_manifest_to_wire,
    runtime_binding_from_wire,
    runtime_binding_to_wire,
)


_CONTROL_SCHEMA_VERSION = 1
_CASE_FIELDS = frozenset(
    {
        "id",
        "category",
        "source",
        "target",
        "global_shape",
        "required_ranks",
    }
)
_MESH_FIELDS = frozenset({"replicas", "shards", "shard_dim"})
_PREPARE_FIELDS = frozenset(
    {
        "type",
        "schema_version",
        "session_id",
        "generation",
        "case",
        "revision",
    }
)
_READY_FIELDS = frozenset(
    {
        "type",
        "schema_version",
        "session_id",
        "generation",
        "endpoint",
        "placement",
        "bindings",
        "registrations",
        "metrics",
    }
)
_READY_METRIC_FIELDS = frozenset(
    {"transport_init_ms", "allocation_ms", "registration_ms"}
)
_REGISTRATION_FIELDS = frozenset(
    {
        "allocation_id",
        "fragment_id",
        "device_id",
        "base_address",
        "nbytes",
        "session_id",
        "generation",
    }
)


def _exact_object(
    value: object, fields: frozenset[str], label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields do not match schema")
    return value


def _positive_integer(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _non_negative_integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _nonempty_string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _mesh_to_wire(mesh: MeshSpec) -> dict[str, int]:
    return {
        "replicas": mesh.replicas,
        "shards": mesh.shards,
        "shard_dim": mesh.shard_dim,
    }


def _mesh_from_wire(value: object, label: str) -> MeshSpec:
    raw = _exact_object(value, _MESH_FIELDS, label)
    return MeshSpec(
        replicas=_positive_integer(raw["replicas"], f"{label}.replicas"),
        shards=_positive_integer(raw["shards"], f"{label}.shards"),
        shard_dim=_non_negative_integer(raw["shard_dim"], f"{label}.shard_dim"),
    )


def case_to_wire(case: BenchmarkCase) -> dict[str, object]:
    if case.source is None or case.target is None or case.global_shape is None:
        raise ValueError(f"{case.id}: executable case geometry is incomplete")
    if case.required_ranks is None:
        raise ValueError(f"{case.id}: required_ranks is missing")
    return {
        "id": case.id,
        "category": case.category,
        "source": _mesh_to_wire(case.source),
        "target": _mesh_to_wire(case.target),
        "global_shape": list(case.global_shape),
        "required_ranks": case.required_ranks,
    }


def case_from_wire(value: object) -> BenchmarkCase:
    raw = _exact_object(value, _CASE_FIELDS, "benchmark case")
    global_shape = raw["global_shape"]
    if isinstance(global_shape, (str, bytes, bytearray)) or not isinstance(
        global_shape, Sequence
    ):
        raise ValueError("benchmark case global_shape must be an array")
    shape = tuple(
        _positive_integer(extent, "benchmark case global_shape extent")
        for extent in global_shape
    )
    case = BenchmarkCase(
        id=_nonempty_string(raw["id"], "benchmark case id"),
        category=_nonempty_string(raw["category"], "benchmark case category"),
        source=_mesh_from_wire(raw["source"], "source mesh"),
        target=_mesh_from_wire(raw["target"], "target mesh"),
        global_shape=shape,
        required_ranks=_positive_integer(
            raw["required_ranks"], "benchmark case required_ranks"
        ),
    )
    assert case.source is not None and case.target is not None
    if case.required_ranks != case.source.total_ranks + case.target.total_ranks:
        raise ValueError("benchmark case required_ranks does not match meshes")
    for label, mesh in (("source", case.source), ("target", case.target)):
        if mesh.shard_dim >= len(shape):
            raise ValueError(f"{label} shard_dim is out of range")
        if shape[mesh.shard_dim] % mesh.shards:
            raise ValueError(f"{label} shard dimension is not divisible")
    return case


@dataclass(frozen=True)
class PreparedRequest:
    session_id: str
    generation: int
    case: BenchmarkCase
    revision: str


def prepare_message(
    *,
    session_id: str,
    generation: int,
    case: BenchmarkCase,
    revision: str,
) -> dict[str, object]:
    return {
        "type": "prepare",
        "schema_version": _CONTROL_SCHEMA_VERSION,
        "session_id": session_id,
        "generation": generation,
        "case": case_to_wire(case),
        "revision": revision,
    }


def parse_prepare_message(value: object) -> PreparedRequest:
    raw = _exact_object(value, _PREPARE_FIELDS, "prepare message")
    if raw["type"] != "prepare" or raw["schema_version"] != _CONTROL_SCHEMA_VERSION:
        raise ValueError("prepare message type or schema version mismatch")
    return PreparedRequest(
        session_id=_nonempty_string(raw["session_id"], "session_id"),
        generation=_non_negative_integer(raw["generation"], "generation"),
        case=case_from_wire(raw["case"]),
        revision=_nonempty_string(raw["revision"], "revision"),
    )


@dataclass(frozen=True)
class RegistrationEnvelope:
    allocation_id: str
    fragment_id: str
    device_id: int
    base_address: int
    nbytes: int
    session_id: str
    generation: int

    def __post_init__(self) -> None:
        for name in ("allocation_id", "fragment_id", "session_id"):
            _nonempty_string(getattr(self, name), name)
        _non_negative_integer(self.device_id, "device_id")
        _positive_integer(self.base_address, "base_address")
        _positive_integer(self.nbytes, "nbytes")
        _non_negative_integer(self.generation, "generation")

    def to_wire(self) -> dict[str, object]:
        return {
            "allocation_id": self.allocation_id,
            "fragment_id": self.fragment_id,
            "device_id": self.device_id,
            "base_address": self.base_address,
            "nbytes": self.nbytes,
            "session_id": self.session_id,
            "generation": self.generation,
        }

    @classmethod
    def from_wire(cls, value: object) -> "RegistrationEnvelope":
        raw = _exact_object(value, _REGISTRATION_FIELDS, "registration envelope")
        return cls(
            allocation_id=raw["allocation_id"],
            fragment_id=raw["fragment_id"],
            device_id=raw["device_id"],
            base_address=raw["base_address"],
            nbytes=raw["nbytes"],
            session_id=raw["session_id"],
            generation=raw["generation"],
        )


@dataclass(frozen=True)
class TargetReady:
    endpoint: str
    placement: WeightPlacementManifest
    bindings: tuple[WeightRuntimeBindingManifest, ...]
    registrations: tuple[RegistrationEnvelope, ...]
    transport_init_ms: float
    allocation_ms: float
    registration_ms: float


def _metric(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def ready_message(
    *,
    session_id: str,
    generation: int,
    endpoint: str,
    placement: WeightPlacementManifest,
    bindings: Sequence[WeightRuntimeBindingManifest],
    registrations: Sequence[RegistrationEnvelope],
    transport_init_ms: float,
    allocation_ms: float,
    registration_ms: float,
) -> dict[str, object]:
    return {
        "type": "ready",
        "schema_version": _CONTROL_SCHEMA_VERSION,
        "session_id": session_id,
        "generation": generation,
        "endpoint": endpoint,
        "placement": placement_manifest_to_wire(placement),
        "bindings": [runtime_binding_to_wire(item) for item in bindings],
        "registrations": [item.to_wire() for item in registrations],
        "metrics": {
            "transport_init_ms": transport_init_ms,
            "allocation_ms": allocation_ms,
            "registration_ms": registration_ms,
        },
    }


def parse_ready_message(
    value: object,
    *,
    expected_session_id: str,
    expected_generation: int,
) -> TargetReady:
    raw = _exact_object(value, _READY_FIELDS, "ready message")
    if raw["type"] != "ready" or raw["schema_version"] != _CONTROL_SCHEMA_VERSION:
        raise ValueError("ready message type or schema version mismatch")
    if raw["session_id"] != expected_session_id:
        raise ValueError("ready message session mismatch")
    if raw["generation"] != expected_generation:
        raise ValueError("ready message generation mismatch")
    endpoint = _nonempty_string(raw["endpoint"], "ready endpoint")
    placement = placement_manifest_from_wire(raw["placement"])
    bindings = tuple(runtime_binding_from_wire(item) for item in raw["bindings"])
    registrations = tuple(
        RegistrationEnvelope.from_wire(item) for item in raw["registrations"]
    )
    metrics = _exact_object(raw["metrics"], _READY_METRIC_FIELDS, "ready metrics")
    if not bindings:
        raise ValueError("ready message must contain target bindings")
    validate_runtime_bindings(placement, bindings)
    if any(binding.generation != expected_generation for binding in bindings):
        raise ValueError("binding generation mismatch")

    fragments = {
        fragment.fragment_id: fragment
        for binding in bindings
        for fragment in binding.fragments
    }
    if len(fragments) != sum(len(binding.fragments) for binding in bindings):
        raise ValueError("ready message contains duplicate fragments")
    envelopes = {item.fragment_id: item for item in registrations}
    if len(envelopes) != len(registrations) or set(envelopes) != set(fragments):
        raise ValueError("ready registration set does not match fragments")
    for fragment_id, fragment in fragments.items():
        envelope = envelopes[fragment_id]
        if envelope.session_id != expected_session_id:
            raise ValueError("registration session mismatch")
        if envelope.generation != expected_generation:
            raise ValueError("registration generation mismatch")
        if fragment.endpoint != endpoint:
            raise ValueError("fragment endpoint mismatch")
        if not (
            envelope.base_address <= fragment.address
            and fragment.address + fragment.nbytes
            <= envelope.base_address + envelope.nbytes
        ):
            raise ValueError(f"registration bounds mismatch: {fragment_id}")

    return TargetReady(
        endpoint=endpoint,
        placement=placement,
        bindings=bindings,
        registrations=registrations,
        transport_init_ms=_metric(metrics["transport_init_ms"], "transport_init_ms"),
        allocation_ms=_metric(metrics["allocation_ms"], "allocation_ms"),
        registration_ms=_metric(metrics["registration_ms"], "registration_ms"),
    )


def expected_fragment_segments(
    case: BenchmarkCase,
    fragment: PlacementFragment,
    *,
    selected_source_replica: int = 0,
) -> Iterator[tuple[int, int, int]]:
    if case.source is None or case.global_shape is None or case.logical_bytes is None:
        raise ValueError("source geometry is unavailable")
    source = case.source
    if selected_source_replica < 0 or selected_source_replica >= source.replicas:
        raise ValueError("selected source replica is out of range")
    itemsize = case.logical_bytes // prod(case.global_shape)
    axis = source.shard_dim
    shard_extent = case.global_shape[axis] // source.shards
    inner_bytes = prod(fragment.local_shape[axis + 1 :]) * itemsize
    outer_count = prod(fragment.local_shape[:axis])
    local_axis_extent = fragment.local_shape[axis]
    global_axis_start = fragment.global_offset[axis]
    pending: tuple[int, int, int] | None = None

    for outer in range(outer_count):
        local_axis = 0
        outer_base = outer * local_axis_extent * inner_bytes
        while local_axis < local_axis_extent:
            global_axis = global_axis_start + local_axis
            shard = global_axis // shard_extent
            next_boundary = min(
                global_axis_start + local_axis_extent,
                (shard + 1) * shard_extent,
            )
            axis_count = next_boundary - global_axis
            segment = (
                outer_base + local_axis * inner_bytes,
                axis_count * inner_bytes,
                1 + selected_source_replica * source.shards + shard,
            )
            if segment[2] > 255:
                raise ValueError("source fill pattern exceeds uint8")
            if (
                pending is not None
                and pending[0] + pending[1] == segment[0]
                and pending[2] == segment[2]
            ):
                pending = (pending[0], pending[1] + segment[1], pending[2])
            else:
                if pending is not None:
                    yield pending
                pending = segment
            local_axis += axis_count
    if pending is not None:
        yield pending


class TimedTransferEngine:
    """Measure native blocking TE calls while preserving the engine API."""

    def __init__(
        self,
        engine: object,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self._engine = engine
        self._clock = clock
        self.reset_measurements()

    def __getattr__(self, name: str) -> object:
        if name == "batch_transfer_sync_write_with_ticket":
            method = getattr(self._engine, name, None)
            if not callable(method):
                raise AttributeError(name)
            return self._batch_transfer_sync_write_with_ticket
        return getattr(self._engine, name)

    def reset_measurements(self) -> None:
        self.native_transfer_seconds = 0.0
        self.batch_count = 0
        self.operation_count = 0
        self.wire_bytes = 0

    def batch_transfer_sync_write(
        self,
        endpoint: str,
        source_addresses: Sequence[int],
        target_addresses: Sequence[int],
        sizes: Sequence[int],
    ) -> int:
        started = self._clock()
        try:
            return self._engine.batch_transfer_sync_write(
                endpoint,
                source_addresses,
                target_addresses,
                sizes,
            )
        finally:
            self.native_transfer_seconds += self._clock() - started
            self.batch_count += 1
            self.operation_count += len(sizes)
            self.wire_bytes += sum(sizes)

    def _batch_transfer_sync_write_with_ticket(
        self,
        endpoint: str,
        source_addresses: Sequence[int],
        target_addresses: Sequence[int],
        sizes: Sequence[int],
    ) -> object:
        started = self._clock()
        try:
            return self._engine.batch_transfer_sync_write_with_ticket(
                endpoint,
                source_addresses,
                target_addresses,
                sizes,
            )
        finally:
            self.native_transfer_seconds += self._clock() - started
            self.batch_count += 1
            self.operation_count += len(sizes)
            self.wire_bytes += sum(sizes)


@dataclass(frozen=True)
class UpdateSample:
    total_seconds: float
    native_transfer_seconds: float
    host_dispatch_seconds: float
    receipt_bytes: int
    wire_bytes: int
    operation_count: int
    batch_count: int


def execute_update(
    *,
    sink: object,
    timed_engine: TimedTransferEngine,
    plan: TransferPlan,
    source_placement: WeightPlacementManifest,
    source_bindings: Sequence[WeightRuntimeBindingManifest],
    target_placement: WeightPlacementManifest,
    target_bindings: Sequence[WeightRuntimeBindingManifest],
    source_registrations: Sequence[MemoryRegistrationLease],
    target_registrations: Sequence[MemoryRegistrationLease],
    clock: Callable[[], float] = time.perf_counter,
) -> UpdateSample:
    timed_engine.reset_measurements()
    started = clock()
    selected_source_participants = {
        executor.participant_id for executor in plan.source_executors
    }
    receipts = tuple(
        receipt
        for source_binding in source_bindings
        if source_binding.participant_id in selected_source_participants
        for receipt in sink.execute(
            plan=plan,
            source_placement=source_placement,
            source_binding=source_binding,
            target_placement=target_placement,
            target_bindings=target_bindings,
            target_registrations=target_registrations,
            source_pre_registered=True,
            source_registrations=source_registrations,
        )
    )
    total_seconds = clock() - started
    receipt_bytes = sum(receipt.nbytes for receipt in receipts)
    if receipt_bytes != plan.total_bytes:
        raise RuntimeError(
            f"Transfer Engine receipts cover {receipt_bytes} bytes, "
            f"expected {plan.total_bytes}"
        )
    if timed_engine.wire_bytes != receipt_bytes:
        raise RuntimeError(
            f"Transfer Engine native calls cover {timed_engine.wire_bytes} bytes, "
            f"expected {receipt_bytes}"
        )
    host_dispatch_seconds = total_seconds - timed_engine.native_transfer_seconds
    if host_dispatch_seconds < 0 and abs(host_dispatch_seconds) < 1e-12:
        host_dispatch_seconds = 0.0
    if host_dispatch_seconds < 0:
        raise RuntimeError("native transfer time exceeds full update time")
    return UpdateSample(
        total_seconds=total_seconds,
        native_transfer_seconds=timed_engine.native_transfer_seconds,
        host_dispatch_seconds=host_dispatch_seconds,
        receipt_bytes=receipt_bytes,
        wire_bytes=timed_engine.wire_bytes,
        operation_count=timed_engine.operation_count,
        batch_count=timed_engine.batch_count,
    )


def _percentile(sorted_samples: Sequence[float], percentile: float) -> float:
    position = (len(sorted_samples) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(sorted_samples) - 1)
    fraction = position - lower
    return (
        sorted_samples[lower]
        + (sorted_samples[upper] - sorted_samples[lower]) * fraction
    )


def summarize_samples(
    samples: Sequence[float], *, logical_bytes: int
) -> dict[str, float | int]:
    if not samples or any(not math.isfinite(item) or item <= 0 for item in samples):
        raise ValueError("samples must contain positive finite seconds")
    if type(logical_bytes) is not int or logical_bytes <= 0:
        raise ValueError("logical_bytes must be a positive integer")
    ordered = sorted(samples)
    p50 = _percentile(ordered, 0.50)
    p95 = _percentile(ordered, 0.95)
    return {
        "count": len(ordered),
        "mean_ms": fmean(ordered) * 1000.0,
        "p50_ms": p50 * 1000.0,
        "p95_ms": p95 * 1000.0,
        "p50_logical_gibps": logical_bytes / (2**30) / p50,
    }
