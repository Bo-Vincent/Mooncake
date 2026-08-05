"""Live CUDA allocation lifecycle for the Mooncake benchmark executor."""

from __future__ import annotations

import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from mooncake.reshard._compat import _strict_zip
from mooncake.reshard.weight import PlacementFragment
from mooncake.reshard.weight.planner import (
    bind_logical_transfer_plan,
    plan_placement_transfer,
)
from mooncake.reshard.weight.te import MooncakeTransferEngineSink

from .case_spec import BenchmarkCase
from .cuda_buffers import CudaBuffer, CudaRuntime, registered_engine_buffers
from .mooncake_executor import (
    RegistrationEnvelope,
    TimedTransferEngine,
    execute_update,
    expected_fragment_segments,
    parse_prepare_message,
    parse_ready_message,
    prepare_message,
    ready_message,
    summarize_samples,
)
from .runtime_layout import (
    RuntimeBuffer,
    RuntimeTopology,
    build_runtime_topology,
    registration_leases,
)
from .wire_protocol import receive_message, send_message


@dataclass(frozen=True)
class ValidationResult:
    checked_bytes: int
    fragment_count: int
    validation_seconds: float


_CONTROL_SCHEMA_VERSION = 1
_IDENTITY_FIELDS = frozenset({"type", "schema_version", "session_id", "generation"})
_ERROR_FIELDS = _IDENTITY_FIELDS | {"detail"}
_VALIDATE_FIELDS = _IDENTITY_FIELDS | {"selected_source_replica"}
_VALIDATION_FIELDS = _IDENTITY_FIELDS | {
    "passed",
    "checked_bytes",
    "fragment_count",
    "validation_ms",
}
_STOPPED_FIELDS = _IDENTITY_FIELDS | {"target_protocol_wall_ms"}


class LiveMeshAllocation:
    def __init__(
        self,
        *,
        case: BenchmarkCase,
        side: str,
        buffers: tuple[RuntimeBuffer, ...],
        topology: RuntimeTopology,
        registrations: tuple[RegistrationEnvelope, ...],
        allocation_seconds: float,
        registration_seconds: float,
        stack: ExitStack,
    ) -> None:
        self.case = case
        self.side = side
        self.buffers = buffers
        self.topology = topology
        self.registrations = registrations
        self.allocation_seconds = allocation_seconds
        self.registration_seconds = registration_seconds
        self._stack = stack
        self._closed = False

    @classmethod
    def open(
        cls,
        *,
        case: BenchmarkCase,
        side: str,
        engine: object,
        endpoint: str,
        cuda_devices: Sequence[int],
        revision: str,
        session_id: str,
        generation: int,
        runtime_factory: Callable[[int], object] = CudaRuntime,
        buffer_factory: Callable[[object, int], RuntimeBuffer] = CudaBuffer,
        clock: Callable[[], float] = time.perf_counter,
    ) -> "LiveMeshAllocation":
        mesh = case.source if side == "source" else case.target
        if side not in ("source", "target") or mesh is None:
            raise ValueError("live allocation side or mesh is invalid")
        devices = tuple(cuda_devices)
        if (
            len(devices) < mesh.total_ranks
            or any(type(device) is not int or device < 0 for device in devices)
            or len(devices) != len(set(devices))
        ):
            raise ValueError(
                f"{side} requires {mesh.total_ranks} CUDA devices with unique IDs"
            )
        if case.logical_bytes is None or case.logical_bytes % mesh.shards:
            raise ValueError("logical bytes are not divisible by mesh shards")
        shard_bytes = case.logical_bytes // mesh.shards
        stack = ExitStack()
        try:
            started = clock()
            runtimes = {
                device: runtime_factory(device)
                for device in devices[: mesh.total_ranks]
            }
            buffers = tuple(
                stack.enter_context(buffer_factory(runtimes[device], shard_bytes))
                for device in devices[: mesh.total_ranks]
            )
            if side == "target":
                for buffer in buffers:
                    buffer.zero()
            allocation_seconds = clock() - started

            started = clock()
            stack.enter_context(registered_engine_buffers(engine, buffers))
            registration_seconds = clock() - started

            topology = build_runtime_topology(
                case,
                side=side,
                buffers=buffers,
                endpoint=endpoint,
                revision=revision,
                lease_generation=generation,
            )
            registrations = tuple(
                RegistrationEnvelope(
                    allocation_id=f"{session_id}:{side}:{index}",
                    fragment_id=binding.fragments[0].fragment_id,
                    device_id=buffer.device,
                    base_address=buffer.pointer,
                    nbytes=buffer.size,
                    session_id=session_id,
                    generation=generation,
                )
                for index, (binding, buffer) in enumerate(
                    _strict_zip(topology.bindings, buffers)
                )
            )
            return cls(
                case=case,
                side=side,
                buffers=buffers,
                topology=topology,
                registrations=registrations,
                allocation_seconds=allocation_seconds,
                registration_seconds=registration_seconds,
                stack=stack,
            )
        except BaseException:
            stack.close()
            raise

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def placement(self):
        return self.topology.placement

    @property
    def bindings(self):
        return self.topology.bindings

    def fill_source_pattern(self) -> None:
        if self.side != "source" or self.case.source is None or self._closed:
            raise RuntimeError("source allocation is not active")
        for index, buffer in enumerate(self.buffers):
            replica, shard = divmod(index, self.case.source.shards)
            value = 1 + replica * self.case.source.shards + shard
            if value > 255:
                raise ValueError("source fill pattern exceeds uint8")
            buffer.fill(value)

    def zero(self) -> None:
        if self._closed:
            raise RuntimeError("allocation is closed")
        for buffer in self.buffers:
            buffer.zero()

    def expected_segments(
        self,
        fragment: PlacementFragment,
        *,
        selected_source_replica: int,
    ):
        return expected_fragment_segments(
            self.case,
            fragment,
            selected_source_replica=selected_source_replica,
        )

    def close(self) -> None:
        if not self._closed:
            self._stack.close()
            self._closed = True

    def __enter__(self) -> "LiveMeshAllocation":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def verify_target_allocation(
    allocation: LiveMeshAllocation,
    *,
    selected_source_replica: int,
    chunk_bytes: int = 8 * 1024 * 1024,
    clock: Callable[[], float] = time.perf_counter,
) -> ValidationResult:
    if allocation.side != "target" or allocation.closed:
        raise RuntimeError("target allocation is not active")
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be a positive integer")
    started = clock()
    checked_bytes = 0
    part_by_participant = {
        part.participant_id: part for part in allocation.placement.parts
    }
    for binding, buffer in _strict_zip(allocation.bindings, allocation.buffers):
        fragment = part_by_participant[binding.participant_id].fragments[0]
        fragment_checked = 0
        for offset, nbytes, value in allocation.expected_segments(
            fragment,
            selected_source_replica=selected_source_replica,
        ):
            for chunk_offset in range(0, nbytes, chunk_bytes):
                current = min(chunk_bytes, nbytes - chunk_offset)
                actual = buffer.read_range(offset + chunk_offset, current)
                if actual != bytes([value]) * current:
                    raise RuntimeError(
                        f"target content mismatch: {fragment.fragment_id} "
                        f"at byte {offset + chunk_offset}"
                    )
                fragment_checked += current
        if fragment_checked != fragment.nbytes:
            raise RuntimeError(
                f"target validation coverage mismatch: {fragment.fragment_id}"
            )
        checked_bytes += fragment_checked
    return ValidationResult(
        checked_bytes=checked_bytes,
        fragment_count=len(allocation.placement.fragments),
        validation_seconds=clock() - started,
    )


def _control_message(
    message_type: str,
    *,
    session_id: str,
    generation: int,
    **fields: object,
) -> dict[str, object]:
    return {
        "type": message_type,
        "schema_version": _CONTROL_SCHEMA_VERSION,
        "session_id": session_id,
        "generation": generation,
        **fields,
    }


def _expect_control(
    value: object,
    *,
    message_type: str,
    session_id: str,
    generation: int,
    fields: frozenset[str] = _IDENTITY_FIELDS,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{message_type} control fields do not match schema")
    if value.get("type") == "error":
        if set(value) != _ERROR_FIELDS:
            raise RuntimeError("error control fields do not match schema")
        if value.get("schema_version") != _CONTROL_SCHEMA_VERSION:
            raise RuntimeError("error control schema version mismatch")
        if value.get("session_id") != session_id:
            raise RuntimeError("error control session mismatch")
        if value.get("generation") != generation:
            raise RuntimeError("error control generation mismatch")
        if not isinstance(value.get("detail"), str) or not value["detail"]:
            raise RuntimeError("error control detail is invalid")
        raise RuntimeError(f"target error: {value.get('detail')}")
    if set(value) != fields:
        raise RuntimeError(f"{message_type} control fields do not match schema")
    if (
        value.get("type") != message_type
        or value.get("schema_version") != _CONTROL_SCHEMA_VERSION
    ):
        raise RuntimeError(f"expected {message_type} control response")
    if value.get("session_id") != session_id:
        raise RuntimeError("control session mismatch")
    if value.get("generation") != generation:
        raise RuntimeError("control generation mismatch")
    return value


def handle_target_connection(
    connection,
    *,
    engine: object,
    endpoint: str,
    cuda_devices: Sequence[int],
    transport_init_seconds: float,
    runtime_factory: Callable[[int], object] = CudaRuntime,
    buffer_factory: Callable[[object, int], RuntimeBuffer] = CudaBuffer,
    timeout_s: float = 300.0,
    process_started: float | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> None:
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    connection.settimeout(timeout_s)
    if process_started is None:
        process_started = clock()
    allocation: LiveMeshAllocation | None = None
    prepared = None
    try:
        prepared = parse_prepare_message(receive_message(connection))
        allocation = LiveMeshAllocation.open(
            case=prepared.case,
            side="target",
            engine=engine,
            endpoint=endpoint,
            cuda_devices=cuda_devices,
            revision=prepared.revision,
            session_id=prepared.session_id,
            generation=prepared.generation,
            runtime_factory=runtime_factory,
            buffer_factory=buffer_factory,
            clock=clock,
        )
        send_message(
            connection,
            ready_message(
                session_id=prepared.session_id,
                generation=prepared.generation,
                endpoint=endpoint,
                placement=allocation.placement,
                bindings=allocation.bindings,
                registrations=allocation.registrations,
                transport_init_ms=transport_init_seconds * 1000.0,
                allocation_ms=allocation.allocation_seconds * 1000.0,
                registration_ms=allocation.registration_seconds * 1000.0,
            ),
        )

        while True:
            command = receive_message(connection)
            command_type = command.get("type")
            if command_type == "reset":
                _expect_control(
                    command,
                    message_type="reset",
                    session_id=prepared.session_id,
                    generation=prepared.generation,
                )
                allocation.zero()
                send_message(
                    connection,
                    _control_message(
                        "reset",
                        session_id=prepared.session_id,
                        generation=prepared.generation,
                    ),
                )
            elif command_type == "validate":
                parsed = _expect_control(
                    command,
                    message_type="validate",
                    session_id=prepared.session_id,
                    generation=prepared.generation,
                    fields=_VALIDATE_FIELDS,
                )
                selected = parsed["selected_source_replica"]
                if type(selected) is not int or selected < 0:
                    raise RuntimeError("selected source replica is invalid")
                validation = verify_target_allocation(
                    allocation,
                    selected_source_replica=selected,
                    clock=clock,
                )
                send_message(
                    connection,
                    _control_message(
                        "validation",
                        session_id=prepared.session_id,
                        generation=prepared.generation,
                        passed=True,
                        checked_bytes=validation.checked_bytes,
                        fragment_count=validation.fragment_count,
                        validation_ms=validation.validation_seconds * 1000.0,
                    ),
                )
            elif command_type == "release":
                _expect_control(
                    command,
                    message_type="release",
                    session_id=prepared.session_id,
                    generation=prepared.generation,
                )
                allocation.close()
                send_message(
                    connection,
                    _control_message(
                        "released",
                        session_id=prepared.session_id,
                        generation=prepared.generation,
                    ),
                )
            elif command_type == "shutdown":
                _expect_control(
                    command,
                    message_type="shutdown",
                    session_id=prepared.session_id,
                    generation=prepared.generation,
                )
                if not allocation.closed:
                    raise RuntimeError("target allocation must be released first")
                send_message(
                    connection,
                    _control_message(
                        "stopped",
                        session_id=prepared.session_id,
                        generation=prepared.generation,
                        target_protocol_wall_ms=(clock() - process_started) * 1000.0,
                    ),
                )
                return
            else:
                raise RuntimeError(f"unsupported control command: {command_type}")
    except BaseException as error:
        if prepared is not None:
            try:
                send_message(
                    connection,
                    _control_message(
                        "error",
                        session_id=prepared.session_id,
                        generation=prepared.generation,
                        detail=repr(error),
                    ),
                )
            except BaseException:
                pass
        raise
    finally:
        if allocation is not None:
            allocation.close()


def _sample_to_dict(sample, logical_bytes: int) -> dict[str, object]:
    return {
        "total_ms": sample.total_seconds * 1000.0,
        "native_transfer_ms": sample.native_transfer_seconds * 1000.0,
        "host_dispatch_ms": sample.host_dispatch_seconds * 1000.0,
        "receipt_bytes": sample.receipt_bytes,
        "wire_bytes": sample.wire_bytes,
        "operation_count": sample.operation_count,
        "batch_count": sample.batch_count,
        "logical_gibps": logical_bytes / (2**30) / sample.total_seconds,
    }


def _latency_summary(samples: Sequence[float]) -> dict[str, float | int]:
    if not samples or any(value < 0 for value in samples):
        raise ValueError("latency samples must be non-negative")
    ordered = sorted(samples)
    middle = len(ordered) // 2
    p50 = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    return {
        "count": len(ordered),
        "mean_ms": sum(ordered) / len(ordered) * 1000.0,
        "p50_ms": p50 * 1000.0,
    }


def run_source_session(
    connection,
    *,
    case: BenchmarkCase,
    revision: str,
    session_id: str,
    generation: int,
    engine: object,
    source_allocation: LiveMeshAllocation,
    source_transport_init_seconds: float,
    warmups: int,
    iterations: int,
    timeout_s: float = 300.0,
    one_shot: bool = False,
    process_started: float | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    if timeout_s <= 0:
        raise ValueError("timeout must be positive")
    if one_shot:
        if warmups != 1 or iterations != 0:
            raise ValueError("one-shot requires one warmup and zero iterations")
    elif warmups <= 0 or iterations <= 0:
        raise ValueError("warmups and iterations must be positive")
    if source_allocation.side != "source" or source_allocation.closed:
        raise ValueError("source allocation must be active")
    if process_started is None:
        process_started = clock()
    connection.settimeout(timeout_s)
    send_message(
        connection,
        prepare_message(
            session_id=session_id,
            generation=generation,
            case=case,
            revision=revision,
        ),
    )
    target_ready = parse_ready_message(
        receive_message(connection),
        expected_session_id=session_id,
        expected_generation=generation,
    )

    plan_started = clock()
    logical_plan = plan_placement_transfer(
        source_allocation.placement,
        target_ready.placement,
    )
    plan = bind_logical_transfer_plan(
        logical_plan,
        target_ready.bindings,
        source_bindings=source_allocation.bindings,
    )
    plan_seconds = clock() - plan_started
    selected_replicas = {
        operation.source.rank.dp for operation in logical_plan.operations
    }
    if len(selected_replicas) != 1:
        raise RuntimeError(
            "physical runner requires one selected source replica per target set"
        )
    selected_source_replica = next(iter(selected_replicas))

    timed_engine = TimedTransferEngine(engine, clock=clock)
    sink = MooncakeTransferEngineSink(timed_engine)
    source_leases = registration_leases(source_allocation.bindings)
    target_leases = registration_leases(target_ready.bindings)

    first = execute_update(
        sink=sink,
        timed_engine=timed_engine,
        plan=plan,
        source_placement=source_allocation.placement,
        source_bindings=source_allocation.bindings,
        target_placement=target_ready.placement,
        target_bindings=target_ready.bindings,
        source_registrations=source_leases,
        target_registrations=target_leases,
        clock=clock,
    )
    cold_e2e_seconds = clock() - process_started
    steady = ()
    validation_update = None
    validation: dict[str, object] = {"passed": None}
    if not one_shot:
        for _ in range(warmups - 1):
            execute_update(
                sink=sink,
                timed_engine=timed_engine,
                plan=plan,
                source_placement=source_allocation.placement,
                source_bindings=source_allocation.bindings,
                target_placement=target_ready.placement,
                target_bindings=target_ready.bindings,
                source_registrations=source_leases,
                target_registrations=target_leases,
                clock=clock,
            )
        steady = tuple(
            execute_update(
                sink=sink,
                timed_engine=timed_engine,
                plan=plan,
                source_placement=source_allocation.placement,
                source_bindings=source_allocation.bindings,
                target_placement=target_ready.placement,
                target_bindings=target_ready.bindings,
                source_registrations=source_leases,
                target_registrations=target_leases,
                clock=clock,
            )
            for _ in range(iterations)
        )

        send_message(
            connection,
            _control_message("reset", session_id=session_id, generation=generation),
        )
        _expect_control(
            receive_message(connection),
            message_type="reset",
            session_id=session_id,
            generation=generation,
        )
        validation_update = execute_update(
            sink=sink,
            timed_engine=timed_engine,
            plan=plan,
            source_placement=source_allocation.placement,
            source_bindings=source_allocation.bindings,
            target_placement=target_ready.placement,
            target_bindings=target_ready.bindings,
            source_registrations=source_leases,
            target_registrations=target_leases,
            clock=clock,
        )
        send_message(
            connection,
            _control_message(
                "validate",
                session_id=session_id,
                generation=generation,
                selected_source_replica=selected_source_replica,
            ),
        )
        validation = _expect_control(
            receive_message(connection),
            message_type="validation",
            session_id=session_id,
            generation=generation,
            fields=_VALIDATION_FIELDS,
        )
        if validation["passed"] is not True:
            raise RuntimeError("target validation failed")
        if validation["checked_bytes"] != plan.total_bytes:
            raise RuntimeError("target validation byte count is incomplete")

    send_message(
        connection,
        _control_message("release", session_id=session_id, generation=generation),
    )
    _expect_control(
        receive_message(connection),
        message_type="released",
        session_id=session_id,
        generation=generation,
    )
    send_message(
        connection,
        _control_message("shutdown", session_id=session_id, generation=generation),
    )
    stopped = _expect_control(
        receive_message(connection),
        message_type="stopped",
        session_id=session_id,
        generation=generation,
        fields=_STOPPED_FIELDS,
    )
    session_seconds = clock() - process_started
    target_protocol_wall_ms = stopped["target_protocol_wall_ms"]
    if (
        isinstance(target_protocol_wall_ms, bool)
        or not isinstance(target_protocol_wall_ms, (int, float))
        or target_protocol_wall_ms <= 0
    ):
        raise RuntimeError("target protocol wall time is invalid")
    source_protocol_wall_ms = session_seconds * 1000.0

    return {
        "schema_version": 1,
        "backend": "mooncake-te-placement-binding",
        "phase": "cold" if one_shot else "steady",
        "case_id": case.id,
        "logical_bytes": case.logical_bytes,
        "topology": {
            "source_replicas": case.source.replicas if case.source else None,
            "source_shards": case.source.shards if case.source else None,
            "source_shard_dim": case.source.shard_dim if case.source else None,
            "target_replicas": case.target.replicas if case.target else None,
            "target_shards": case.target.shards if case.target else None,
            "target_shard_dim": case.target.shard_dim if case.target else None,
        },
        "warmups": warmups,
        "iterations": iterations,
        "protocol_wall_ms": max(source_protocol_wall_ms, target_protocol_wall_ms),
        "source_protocol_wall_ms": source_protocol_wall_ms,
        "target_protocol_wall_ms": target_protocol_wall_ms,
        "first_update_ready_ms": cold_e2e_seconds * 1000.0,
        "control_plane_ms": {
            "source_transport_init": source_transport_init_seconds * 1000.0,
            "target_transport_init": target_ready.transport_init_ms,
            "source_allocation": source_allocation.allocation_seconds * 1000.0,
            "target_allocation": target_ready.allocation_ms,
            "source_registration": source_allocation.registration_seconds * 1000.0,
            "target_registration": target_ready.registration_ms,
            "plan": plan_seconds * 1000.0,
        },
        "first_update": _sample_to_dict(first, case.logical_bytes),
        "steady_update": (
            None
            if one_shot
            else {
                "total": summarize_samples(
                    tuple(item.total_seconds for item in steady),
                    logical_bytes=case.logical_bytes,
                ),
                "native_transfer": summarize_samples(
                    tuple(item.native_transfer_seconds for item in steady),
                    logical_bytes=case.logical_bytes,
                ),
                "host_dispatch": _latency_summary(
                    tuple(item.host_dispatch_seconds for item in steady)
                ),
                "samples": [
                    _sample_to_dict(item, case.logical_bytes) for item in steady
                ],
            }
        ),
        "validation_update": (
            None
            if validation_update is None
            else _sample_to_dict(validation_update, case.logical_bytes)
        ),
        "validation": (
            validation
            if one_shot
            else {
                "passed": True,
                "checked_bytes": validation["checked_bytes"],
                "fragment_count": validation["fragment_count"],
                "validation_ms": validation["validation_ms"],
            }
        ),
        "lowering_ms": None,
        "lowering_included_in": "update total minus native blocking TE calls",
    }
