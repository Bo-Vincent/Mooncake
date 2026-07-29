from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Any, Iterator, Sequence
from uuid import uuid4

from .manifest import RuntimeFragment, RuntimeManifest
from .planner import TransferOperation, TransferPlan, resolve_executor_plans


class TransferEngineError(RuntimeError):
    pass


class TransferCompletionUnknownError(TransferEngineError):
    def __init__(self, message: str, *, pending_transfer_id: str) -> None:
        super().__init__(message)
        self.pending_transfer_id = pending_transfer_id


class _CompletionUnknown(RuntimeError):
    def __init__(self, ticket: Any) -> None:
        super().__init__("transfer completion remains unknown")
        self.ticket = ticket


class _CompletionWaitInterrupted(BaseException):
    def __init__(self, ticket: Any, interruption: BaseException) -> None:
        super().__init__(str(interruption))
        self.ticket = ticket
        self.interruption = interruption


class _PendingCompletionWaitInterrupted(BaseException):
    def __init__(
        self,
        pending_transfer_id: str,
        interruption: BaseException,
    ) -> None:
        super().__init__(str(interruption))
        self.pending_transfer_id = pending_transfer_id
        self.interruption = interruption


class _UndrainableCompletionUnknownTicket:
    status = "COMPLETION_UNKNOWN"
    restart_required = True

    def drain(self, timeout_ms: int) -> str:
        return self.status


def _canonical_engine_identity(engine: Any) -> tuple[str, int]:
    try:
        get_engine_ptr = getattr(engine, "get_engine_ptr", None)
    except Exception as error:
        raise TransferEngineError(
            "failed to resolve canonical engine identity"
        ) from error
    if not callable(get_engine_ptr):
        return ("python", id(engine))
    try:
        engine_ptr = get_engine_ptr()
    except Exception as error:
        raise TransferEngineError(
            "failed to resolve canonical engine identity"
        ) from error
    if type(engine_ptr) is not int or engine_ptr <= 0:
        raise TransferEngineError(
            "get_engine_ptr returned an invalid canonical engine identity"
        )
    return ("native", engine_ptr)


@dataclass
class _PendingTransfer:
    engine: Any
    engine_identity: tuple[str, int]
    ticket: Any
    registrations: set[int]
    resources: tuple[Any, ...]
    restart_required: bool
    resources_handed_off: bool = False
    draining: bool = False


_pending_transfer_lock = Lock()
_pending_transfers: dict[str, _PendingTransfer] = {}
_active_submission_engine_ids: set[tuple[str, int]] = set()


def _completion_status_name(status: Any) -> str:
    name = getattr(status, "name", None)
    if isinstance(name, str):
        return name
    try:
        value = int(status)
    except (TypeError, ValueError):
        value = None
    by_value = {
        0: "COMPLETED",
        -1: "FAILED_DRAINED",
        -2: "COMPLETION_UNKNOWN",
    }
    if value in by_value:
        return by_value[value]
    text = str(status)
    for candidate in by_value.values():
        if candidate in text:
            return candidate
    return text


def _batch_transfer_with_completion_fence(
    engine: Any,
    *,
    ticket_method_name: str,
    legacy_method_name: str,
    arguments: tuple[Any, ...],
    max_drain_attempts: int,
    drain_timeout_ms: int,
) -> int | str:
    ticket_method = getattr(engine, ticket_method_name, None)
    if not callable(ticket_method):
        result = getattr(engine, legacy_method_name)(*arguments)
        if _completion_status_name(result) == "COMPLETION_UNKNOWN":
            raise _CompletionUnknown(_UndrainableCompletionUnknownTicket())
        return result

    ticket = ticket_method(*arguments)
    status_name = _completion_status_name(ticket.status)
    for _ in range(max_drain_attempts):
        if status_name != "COMPLETION_UNKNOWN":
            break
        try:
            status = ticket.drain(drain_timeout_ms)
        except Exception:
            # UNKNOWN means native DMA may still reference caller buffers.
            continue
        except BaseException as error:
            raise _CompletionWaitInterrupted(ticket, error) from error
        status_name = _completion_status_name(status)
    if status_name == "COMPLETION_UNKNOWN":
        raise _CompletionUnknown(ticket)
    return 0 if status_name == "COMPLETED" else status_name


@dataclass(frozen=True)
class DirectTransferReceipt:
    source_worker_id: str
    target_endpoint: str
    operation_count: int
    nbytes: int


@dataclass(frozen=True)
class DirectReadReceipt:
    source_endpoint: str
    target_worker_id: str
    operation_count: int
    nbytes: int


@dataclass(frozen=True)
class MemoryRegistrationLease:
    fragment_id: str
    worker_id: str
    address: int
    nbytes: int
    lease_generation: int
    runtime_lease_id: str | None = None

    def __post_init__(self) -> None:
        if not self.fragment_id or not self.worker_id:
            raise ValueError("registration lease identifiers must not be empty")
        for name in ("address", "nbytes", "lease_generation"):
            value = getattr(self, name)
            if type(value) is not int:
                raise ValueError(f"registration lease {name} must be an integer")
        if self.address <= 0 or self.nbytes <= 0 or self.lease_generation < 0:
            raise ValueError("registration lease values are invalid")
        if self.runtime_lease_id is not None and (
            type(self.runtime_lease_id) is not str or not self.runtime_lease_id
        ):
            raise ValueError("registration runtime_lease_id must be a non-empty string")

    @classmethod
    def from_fragment(
        cls,
        fragment: RuntimeFragment,
        *,
        runtime_lease_id: str | None = None,
    ) -> MemoryRegistrationLease:
        return cls(
            fragment_id=fragment.fragment_id,
            worker_id=fragment.worker_id,
            address=fragment.address,
            nbytes=fragment.nbytes,
            lease_generation=fragment.lease_generation,
            runtime_lease_id=runtime_lease_id,
        )


def _same_snapshot(current: RuntimeFragment, planned: RuntimeFragment) -> bool:
    return (
        current.tensor_id == planned.tensor_id
        and current.global_offset == planned.global_offset
        and current.local_shape == planned.local_shape
        and current.address == planned.address
        and current.nbytes == planned.nbytes
        and current.worker_id == planned.worker_id
        and current.endpoint == planned.endpoint
        and current.lease_generation == planned.lease_generation
    )


class MooncakeTransferEngineSink:
    def __init__(
        self,
        engine: Any,
        *,
        max_batch_operations: int = 1024,
        max_region_segments: int = 1_000_000,
        max_completion_drain_attempts: int = 3,
        completion_drain_timeout_ms: int = 1000,
    ) -> None:
        if (
            max_batch_operations <= 0
            or max_region_segments <= 0
            or max_completion_drain_attempts <= 0
            or completion_drain_timeout_ms < 0
        ):
            raise ValueError("transfer lowering limits must be positive")
        self.engine = engine
        self._engine_identity = _canonical_engine_identity(engine)
        self.max_batch_operations = max_batch_operations
        self.max_region_segments = max_region_segments
        self.max_completion_drain_attempts = max_completion_drain_attempts
        self.completion_drain_timeout_ms = completion_drain_timeout_ms

    def execute(
        self,
        plan: TransferPlan,
        source_manifest: RuntimeManifest,
        target_manifests: Sequence[RuntimeManifest],
        *,
        target_registrations: Sequence[MemoryRegistrationLease] | None = None,
        source_pre_registered: bool = False,
        source_registrations: Sequence[MemoryRegistrationLease] | None = None,
    ) -> tuple[DirectTransferReceipt, ...]:
        self._reserve_submission()
        try:
            return self._execute_reserved(
                plan,
                source_manifest,
                target_manifests,
                target_registrations=target_registrations,
                source_pre_registered=source_pre_registered,
                source_registrations=source_registrations,
            )
        finally:
            self._release_submission()

    def _execute_reserved(
        self,
        plan: TransferPlan,
        source_manifest: RuntimeManifest,
        target_manifests: Sequence[RuntimeManifest],
        *,
        target_registrations: Sequence[MemoryRegistrationLease] | None = None,
        source_pre_registered: bool = False,
        source_registrations: Sequence[MemoryRegistrationLease] | None = None,
    ) -> tuple[DirectTransferReceipt, ...]:
        self._validate_plan_identity(plan, source_manifest, "source")
        if not target_manifests:
            raise TransferEngineError("target manifests must not be empty")
        try:
            source_executors = resolve_executor_plans(plan, source_manifest, "source")
        except ValueError as error:
            raise TransferEngineError(str(error)) from error
        source_workers = {executor.worker_id for executor in source_executors}
        if len(source_workers) != 1:
            raise TransferEngineError("source manifest spans multiple workers")
        source_worker_id = next(iter(source_workers))

        targets: dict[str, RuntimeFragment] = {}
        target_runtime_lease_ids: dict[str, str | None] = {}
        target_ranks = set()
        for manifest in target_manifests:
            self._validate_plan_identity(plan, manifest, "target")
            try:
                executors = resolve_executor_plans(plan, manifest, "target")
            except ValueError as error:
                raise TransferEngineError(str(error)) from error
            for executor in executors:
                if executor.rank in target_ranks:
                    raise TransferEngineError(
                        f"duplicate target executor rank: {executor.rank}"
                    )
                target_ranks.add(executor.rank)
            for fragment in manifest.fragments:
                if fragment.fragment_id in targets:
                    raise TransferEngineError(
                        f"duplicate target fragment: {fragment.fragment_id}"
                    )
                targets[fragment.fragment_id] = fragment
                target_runtime_lease_ids[fragment.fragment_id] = manifest.lease_id
        expected_target_ranks = {executor.rank for executor in plan.target_executors}
        if target_ranks != expected_target_ranks:
            raise TransferEngineError("target executor set is incomplete")

        local_operations = [
            plan.operations[index]
            for executor in source_executors
            for index in executor.operation_indices
        ]
        if not local_operations:
            return ()

        local = {
            fragment.fragment_id: fragment for fragment in source_manifest.fragments
        }
        target_registration_by_id = self._registration_map(
            target_registrations, "target"
        )

        operations_by_endpoint: dict[
            str, list[tuple[TransferOperation, RuntimeFragment, RuntimeFragment]]
        ] = {}
        used_sources: dict[str, RuntimeFragment] = {}
        for operation in local_operations:
            if not isinstance(operation.source, RuntimeFragment):
                raise TransferEngineError(
                    "MooncakeTransferEngineSink requires runtime sources"
                )
            current = local[operation.source.fragment_id]
            if not _same_snapshot(current, operation.source):
                raise TransferEngineError(
                    f"stale source fragment: {operation.source.fragment_id}"
                )
            target = targets.get(operation.target.fragment_id)
            if target is None:
                raise TransferEngineError(
                    f"missing planned target fragment: {operation.target.fragment_id}"
                )
            if not _same_snapshot(target, operation.target):
                raise TransferEngineError(
                    f"stale target fragment: {operation.target.fragment_id}"
                )
            if not target.endpoint:
                raise TransferEngineError(
                    f"target endpoint is empty: {operation.target.fragment_id}"
                )
            try:
                operation.validate_bounds()
            except ValueError as error:
                raise TransferEngineError(
                    f"invalid copy range for {operation.tensor_id}: {error}"
                ) from error
            if operation.repeat > self.max_region_segments:
                raise TransferEngineError(
                    f"transfer region exceeds max_region_segments: "
                    f"{operation.tensor_id}: {operation.repeat} > "
                    f"{self.max_region_segments}"
                )
            self._validate_registration(
                target,
                target_registration_by_id,
                "target",
                runtime_lease_id=target_runtime_lease_ids[target.fragment_id],
            )
            used_sources[current.fragment_id] = current
            operations_by_endpoint.setdefault(target.endpoint, []).append(
                (operation, current, target)
            )

        with self._registered_sources(
            tuple(used_sources.values()),
            pre_registered=source_pre_registered,
            registrations=source_registrations,
            runtime_lease_id=source_manifest.lease_id,
            resources=(
                source_manifest,
                tuple(target_manifests),
                source_registrations,
                target_registrations,
            ),
        ):
            receipts = []
            for endpoint in sorted(operations_by_endpoint):
                operations = sorted(
                    operations_by_endpoint[endpoint],
                    key=lambda item: (
                        item[2].address + item[0].target_offset,
                        item[1].address + item[0].source_offset,
                    ),
                )
                source_addresses = []
                target_addresses = []
                sizes = []
                operation_count = 0
                total_bytes = 0
                for operation, source, target in operations:
                    for (
                        source_offset,
                        target_offset,
                        nbytes,
                    ) in operation.iter_segments():
                        source_addresses.append(source.address + source_offset)
                        target_addresses.append(target.address + target_offset)
                        sizes.append(nbytes)
                        operation_count += 1
                        total_bytes += nbytes
                        if len(sizes) == self.max_batch_operations:
                            self._transfer_batch(
                                endpoint,
                                source_addresses,
                                target_addresses,
                                sizes,
                            )
                            source_addresses = []
                            target_addresses = []
                            sizes = []
                if sizes:
                    self._transfer_batch(
                        endpoint, source_addresses, target_addresses, sizes
                    )
                receipts.append(
                    DirectTransferReceipt(
                        source_worker_id=source_worker_id,
                        target_endpoint=endpoint,
                        operation_count=operation_count,
                        nbytes=total_bytes,
                    )
                )
        return tuple(receipts)

    @staticmethod
    def _validate_plan_identity(
        plan: TransferPlan, manifest: RuntimeManifest, label: str
    ) -> None:
        if manifest.model_id != plan.model_id:
            raise TransferEngineError(f"{label} model_id mismatch")
        if manifest.revision != plan.revision:
            raise TransferEngineError(f"{label} revision mismatch")

    @staticmethod
    def _registration_map(
        registrations: Sequence[MemoryRegistrationLease] | None,
        label: str,
    ) -> dict[str, MemoryRegistrationLease]:
        if registrations is None:
            raise TransferEngineError(f"{label} registration leases are required")
        result = {}
        for registration in registrations:
            if registration.fragment_id in result:
                raise TransferEngineError(
                    f"duplicate {label} registration lease: {registration.fragment_id}"
                )
            result[registration.fragment_id] = registration
        return result

    @staticmethod
    def _validate_registration(
        fragment: RuntimeFragment,
        registrations: dict[str, MemoryRegistrationLease],
        label: str,
        *,
        runtime_lease_id: str | None,
    ) -> None:
        registration = registrations.get(fragment.fragment_id)
        if registration is None or (
            registration.worker_id != fragment.worker_id
            or registration.address != fragment.address
            or registration.nbytes != fragment.nbytes
            or registration.lease_generation != fragment.lease_generation
            or registration.runtime_lease_id != runtime_lease_id
        ):
            raise TransferEngineError(
                f"{label} registration lease mismatch: {fragment.fragment_id}"
            )

    @contextmanager
    def _registered_sources(
        self,
        fragments: Sequence[RuntimeFragment],
        *,
        pre_registered: bool,
        registrations: Sequence[MemoryRegistrationLease] | None,
        runtime_lease_id: str | None,
        resources: Sequence[Any],
    ) -> Iterator[None]:
        if pre_registered:
            registration_by_id = self._registration_map(registrations, "source")
            for fragment in fragments:
                self._validate_registration(
                    fragment,
                    registration_by_id,
                    "source",
                    runtime_lease_id=runtime_lease_id,
                )
            try:
                yield
            except (
                TransferCompletionUnknownError,
                _PendingCompletionWaitInterrupted,
            ) as error:
                self._retain_pending_resources(
                    error.pending_transfer_id,
                    registrations=(),
                    resources=resources,
                )
                if isinstance(error, _PendingCompletionWaitInterrupted):
                    raise error.interruption from error
                raise
            return
        if registrations is not None:
            raise TransferEngineError(
                "source registration leases require source_pre_registered=True"
            )

        sizes_by_address: dict[int, int] = {}
        for fragment in fragments:
            sizes_by_address[fragment.address] = max(
                sizes_by_address.get(fragment.address, 0), fragment.nbytes
            )
        owned = []
        primary_error: BaseException | None = None
        try:
            for address, nbytes in sizes_by_address.items():
                try:
                    result = self.engine.register_memory(address, nbytes)
                except Exception as error:
                    raise TransferEngineError(
                        f"source register_memory failed for {address}: {error}"
                    ) from error
                if result != 0:
                    raise TransferEngineError(
                        f"source register_memory failed for {address}: {result}"
                    )
                owned.append(address)
            yield
        except BaseException as error:
            primary_error = error

        if isinstance(
            primary_error,
            (
                TransferCompletionUnknownError,
                _PendingCompletionWaitInterrupted,
            ),
        ):
            self._retain_pending_resources(
                primary_error.pending_transfer_id,
                registrations=owned,
                resources=resources,
            )
            if isinstance(primary_error, _PendingCompletionWaitInterrupted):
                raise primary_error.interruption from primary_error
            raise primary_error

        failures = []
        for address in reversed(owned):
            try:
                result = self.engine.unregister_memory(address)
            except Exception as error:
                failures.append((address, repr(error)))
                continue
            if result != 0:
                failures.append((address, result))
        if failures:
            detail = f"source unregister_memory failed: {failures}"
            if primary_error is not None:
                raise TransferEngineError(
                    f"{primary_error}; {detail}"
                ) from primary_error
            raise TransferEngineError(detail)
        if primary_error is not None:
            raise primary_error

    def _transfer_batch(
        self,
        endpoint: str,
        source_addresses: list[int],
        target_addresses: list[int],
        sizes: list[int],
    ) -> None:
        try:
            result = _batch_transfer_with_completion_fence(
                self.engine,
                ticket_method_name="batch_transfer_sync_write_with_ticket",
                legacy_method_name="batch_transfer_sync_write",
                arguments=(
                    endpoint,
                    source_addresses,
                    target_addresses,
                    sizes,
                ),
                max_drain_attempts=self.max_completion_drain_attempts,
                drain_timeout_ms=self.completion_drain_timeout_ms,
            )
        except _CompletionUnknown as error:
            pending_transfer_id = self._retain_pending_ticket(error.ticket)
            restart_required = getattr(error.ticket, "restart_required", False)
            suffix = (
                "; legacy API exposes no drainable ticket, so this engine is "
                "restart-required"
                if restart_required
                else ""
            )
            raise TransferCompletionUnknownError(
                "batch transfer completion is unknown; registrations remain "
                f"quarantined as {pending_transfer_id}{suffix}",
                pending_transfer_id=pending_transfer_id,
            ) from error
        except _CompletionWaitInterrupted as error:
            pending_transfer_id = self._retain_pending_ticket(error.ticket)
            raise _PendingCompletionWaitInterrupted(
                pending_transfer_id,
                error.interruption,
            ) from error
        except Exception as error:
            raise TransferEngineError(
                f"batch transfer to {endpoint} failed: {error}"
            ) from error
        if result != 0:
            raise TransferEngineError(f"batch transfer to {endpoint} failed: {result}")

    def _retain_pending_ticket(self, ticket: Any) -> str:
        pending_transfer_id = uuid4().hex
        with _pending_transfer_lock:
            _pending_transfers[pending_transfer_id] = _PendingTransfer(
                engine=self.engine,
                engine_identity=self._engine_identity,
                ticket=ticket,
                registrations=set(),
                resources=(),
                restart_required=getattr(ticket, "restart_required", False),
            )
        return pending_transfer_id

    def _retain_pending_resources(
        self,
        pending_transfer_id: str,
        *,
        registrations: Sequence[int],
        resources: Sequence[Any],
    ) -> None:
        with _pending_transfer_lock:
            pending = _pending_transfers.get(pending_transfer_id)
            if pending is None:
                raise TransferEngineError(
                    f"pending transfer does not exist: {pending_transfer_id}"
                )
            if pending.engine_identity != self._engine_identity:
                raise TransferEngineError(
                    f"pending transfer does not exist: {pending_transfer_id}"
                )
            if pending.resources_handed_off:
                raise TransferEngineError(
                    "pending transfer resources were already handed off: "
                    f"{pending_transfer_id}"
                )
            pending.registrations.update(registrations)
            pending.resources += tuple(resources)
            pending.resources_handed_off = True

    def _reserve_submission(self) -> None:
        with _pending_transfer_lock:
            pending = sorted(
                (
                    transfer_id,
                    transfer,
                )
                for transfer_id, transfer in _pending_transfers.items()
                if transfer.engine_identity == self._engine_identity
            )
            if pending:
                pending_transfer_id, transfer = pending[0]
                if transfer.restart_required:
                    raise TransferEngineError(
                        "transfer engine is blocked by restart-required pending "
                        f"transfer {pending_transfer_id}; restart before submitting "
                        "another transfer"
                    )
                raise TransferEngineError(
                    f"transfer engine has pending transfer {pending_transfer_id}; "
                    "call drain_pending_transfer before submitting another transfer, "
                    "or restart if it cannot be drained"
                )
            if self._engine_identity in _active_submission_engine_ids:
                raise TransferEngineError(
                    "transfer engine already has an active weight transfer submission"
                )
            _active_submission_engine_ids.add(self._engine_identity)

    def _release_submission(self) -> None:
        with _pending_transfer_lock:
            _active_submission_engine_ids.remove(self._engine_identity)

    def _get_pending_transfer(
        self,
        pending_transfer_id: str,
    ) -> _PendingTransfer:
        pending = _pending_transfers.get(pending_transfer_id)
        if pending is None or pending.engine_identity != self._engine_identity:
            raise TransferEngineError(
                f"pending transfer does not exist: {pending_transfer_id}"
            )
        return pending

    def pending_transfer_ids(self) -> tuple[str, ...]:
        with _pending_transfer_lock:
            return tuple(
                sorted(
                    transfer_id
                    for transfer_id, pending in _pending_transfers.items()
                    if pending.engine_identity == self._engine_identity
                )
            )

    def pending_transfer_status(self, pending_transfer_id: str) -> str:
        with _pending_transfer_lock:
            pending = self._get_pending_transfer(pending_transfer_id)
            if not pending.resources_handed_off:
                raise TransferEngineError(
                    "pending transfer resource handoff is incomplete: "
                    f"{pending_transfer_id}"
                )
            status = _completion_status_name(pending.ticket.status)
            if pending.restart_required and status == "COMPLETION_UNKNOWN":
                return "COMPLETION_UNKNOWN_RESTART_REQUIRED"
            return status

    def drain_pending_transfer(
        self,
        pending_transfer_id: str,
        *,
        timeout_ms: int = 1000,
    ) -> str:
        if timeout_ms < 0:
            raise ValueError("timeout_ms must be non-negative")
        with _pending_transfer_lock:
            pending = self._get_pending_transfer(pending_transfer_id)
            if not pending.resources_handed_off:
                raise TransferEngineError(
                    "pending transfer resource handoff is incomplete: "
                    f"{pending_transfer_id}"
                )
            if pending.draining:
                raise TransferEngineError(
                    f"pending transfer is already being drained: {pending_transfer_id}"
                )
            pending.draining = True
            ticket = pending.ticket
            cleanup_engine = pending.engine
        try:
            try:
                status_name = _completion_status_name(ticket.drain(timeout_ms))
            except Exception:
                return "COMPLETION_UNKNOWN"
            if status_name == "COMPLETION_UNKNOWN":
                return status_name

            with _pending_transfer_lock:
                current = _pending_transfers.get(pending_transfer_id)
                if current is None:
                    raise TransferEngineError(
                        f"pending transfer does not exist: {pending_transfer_id}"
                    )
                owned = set(current.registrations)
            failures = []
            failed_addresses = set()
            for address in sorted(owned, reverse=True):
                try:
                    result = cleanup_engine.unregister_memory(address)
                except Exception as error:
                    failures.append((address, repr(error)))
                    failed_addresses.add(address)
                    continue
                if result != 0:
                    failures.append((address, result))
                    failed_addresses.add(address)
            with _pending_transfer_lock:
                if failures:
                    current = _pending_transfers.get(pending_transfer_id)
                    if current is not None:
                        current.registrations = failed_addresses
                else:
                    _pending_transfers.pop(pending_transfer_id, None)
            if failures:
                raise TransferEngineError(
                    f"pending transfer registration cleanup failed: {failures}"
                )
            return status_name
        finally:
            with _pending_transfer_lock:
                pending = _pending_transfers.get(pending_transfer_id)
                if pending is not None:
                    pending.draining = False


class MooncakeTransferEngineReader:
    """Execute a local target plan with target-initiated zero-copy reads."""

    def __init__(
        self,
        engine: Any,
        *,
        max_batch_operations: int = 1024,
        max_region_segments: int = 1_000_000,
        max_completion_drain_attempts: int = 3,
        completion_drain_timeout_ms: int = 1000,
    ) -> None:
        if (
            max_batch_operations <= 0
            or max_region_segments <= 0
            or max_completion_drain_attempts <= 0
            or completion_drain_timeout_ms < 0
        ):
            raise ValueError("transfer lowering limits must be positive")
        self.engine = engine
        self.max_batch_operations = max_batch_operations
        self.max_region_segments = max_region_segments
        self.max_completion_drain_attempts = max_completion_drain_attempts
        self.completion_drain_timeout_ms = completion_drain_timeout_ms
        self._pending = MooncakeTransferEngineSink(
            engine,
            max_batch_operations=max_batch_operations,
            max_region_segments=max_region_segments,
            max_completion_drain_attempts=max_completion_drain_attempts,
            completion_drain_timeout_ms=completion_drain_timeout_ms,
        )

    def execute(
        self,
        plan: TransferPlan,
        source_manifests: Sequence[RuntimeManifest],
        target_manifest: RuntimeManifest,
        *,
        source_pre_registered: bool = True,
        source_registrations: Sequence[MemoryRegistrationLease] | None = None,
        target_pre_registered: bool = False,
        target_registrations: Sequence[MemoryRegistrationLease] | None = None,
    ) -> tuple[DirectReadReceipt, ...]:
        self._pending._reserve_submission()
        try:
            return self._execute_reserved(
                plan,
                source_manifests,
                target_manifest,
                source_pre_registered=source_pre_registered,
                source_registrations=source_registrations,
                target_pre_registered=target_pre_registered,
                target_registrations=target_registrations,
            )
        finally:
            self._pending._release_submission()

    def _execute_reserved(
        self,
        plan: TransferPlan,
        source_manifests: Sequence[RuntimeManifest],
        target_manifest: RuntimeManifest,
        *,
        source_pre_registered: bool = True,
        source_registrations: Sequence[MemoryRegistrationLease] | None = None,
        target_pre_registered: bool = False,
        target_registrations: Sequence[MemoryRegistrationLease] | None = None,
    ) -> tuple[DirectReadReceipt, ...]:
        if not source_pre_registered:
            raise TransferEngineError("remote source memory must be pre-registered")
        source_registration_by_id = MooncakeTransferEngineSink._registration_map(
            source_registrations, "source"
        )
        if target_registrations is not None and not target_pre_registered:
            raise TransferEngineError(
                "target_registrations require target_pre_registered=True"
            )
        MooncakeTransferEngineSink._validate_plan_identity(
            plan, target_manifest, "target"
        )
        try:
            target_executors = resolve_executor_plans(plan, target_manifest, "target")
        except ValueError as error:
            raise TransferEngineError(str(error)) from error
        if len(target_executors) != 1:
            raise TransferEngineError(
                "target manifest must describe one local executor"
            )
        target_executor = target_executors[0]

        sources: dict[str, RuntimeFragment] = {}
        source_runtime_lease_ids: dict[str, str | None] = {}
        source_ranks = set()
        for manifest in source_manifests:
            MooncakeTransferEngineSink._validate_plan_identity(plan, manifest, "source")
            try:
                executors = resolve_executor_plans(plan, manifest, "source")
            except ValueError as error:
                raise TransferEngineError(str(error)) from error
            for executor in executors:
                if executor.rank in source_ranks:
                    raise TransferEngineError(
                        f"duplicate source executor rank: {executor.rank}"
                    )
                source_ranks.add(executor.rank)
            for fragment in manifest.fragments:
                if fragment.fragment_id in sources:
                    raise TransferEngineError(
                        f"duplicate source fragment: {fragment.fragment_id}"
                    )
                sources[fragment.fragment_id] = fragment
                source_runtime_lease_ids[fragment.fragment_id] = manifest.lease_id
        expected_source_ranks = {executor.rank for executor in plan.source_executors}
        if source_ranks != expected_source_ranks:
            raise TransferEngineError("source executor set is incomplete")

        targets = {
            fragment.fragment_id: fragment for fragment in target_manifest.fragments
        }
        registration_by_id = (
            MooncakeTransferEngineSink._registration_map(target_registrations, "target")
            if target_pre_registered
            else {}
        )
        operations_by_endpoint: dict[
            str, list[tuple[TransferOperation, RuntimeFragment, RuntimeFragment]]
        ] = {}
        used_targets: dict[str, RuntimeFragment] = {}
        for index in target_executor.operation_indices:
            operation = plan.operations[index]
            if not isinstance(operation.source, RuntimeFragment):
                raise TransferEngineError(
                    "MooncakeTransferEngineReader requires runtime sources"
                )
            source = sources.get(operation.source.fragment_id)
            target = targets.get(operation.target.fragment_id)
            if source is None or not _same_snapshot(source, operation.source):
                raise TransferEngineError(
                    f"stale source fragment: {operation.source.fragment_id}"
                )
            runtime_lease_id = source_runtime_lease_ids[source.fragment_id]
            MooncakeTransferEngineSink._validate_registration(
                source,
                source_registration_by_id,
                "source",
                runtime_lease_id=runtime_lease_id,
            )
            if target is None or not _same_snapshot(target, operation.target):
                raise TransferEngineError(
                    f"stale target fragment: {operation.target.fragment_id}"
                )
            if target_pre_registered:
                MooncakeTransferEngineSink._validate_registration(
                    target,
                    registration_by_id,
                    "target",
                    runtime_lease_id=target_manifest.lease_id,
                )
            operation.validate_bounds()
            if operation.repeat > self.max_region_segments:
                raise TransferEngineError(
                    f"transfer region exceeds max_region_segments: "
                    f"{operation.tensor_id}: {operation.repeat} > "
                    f"{self.max_region_segments}"
                )
            used_targets[target.fragment_id] = target
            operations_by_endpoint.setdefault(source.endpoint, []).append(
                (operation, source, target)
            )

        with self._registered_targets(
            tuple(used_targets.values()),
            pre_registered=target_pre_registered,
            resources=(
                tuple(source_manifests),
                target_manifest,
                source_registrations,
                target_registrations,
            ),
        ):
            receipts = []
            for endpoint in sorted(operations_by_endpoint):
                operations = sorted(
                    operations_by_endpoint[endpoint],
                    key=lambda item: (
                        item[2].address + item[0].target_offset,
                        item[1].address + item[0].source_offset,
                    ),
                )
                target_addresses = []
                source_addresses = []
                sizes = []
                operation_count = 0
                total_bytes = 0
                for operation, source, target in operations:
                    for (
                        source_offset,
                        target_offset,
                        nbytes,
                    ) in operation.iter_segments():
                        source_addresses.append(source.address + source_offset)
                        target_addresses.append(target.address + target_offset)
                        sizes.append(nbytes)
                        operation_count += 1
                        total_bytes += nbytes
                        if len(sizes) == self.max_batch_operations:
                            self._transfer_batch(
                                endpoint, target_addresses, source_addresses, sizes
                            )
                            target_addresses = []
                            source_addresses = []
                            sizes = []
                if sizes:
                    self._transfer_batch(
                        endpoint, target_addresses, source_addresses, sizes
                    )
                receipts.append(
                    DirectReadReceipt(
                        source_endpoint=endpoint,
                        target_worker_id=target_executor.worker_id,
                        operation_count=operation_count,
                        nbytes=total_bytes,
                    )
                )
        return tuple(receipts)

    @contextmanager
    def _registered_targets(
        self,
        fragments: Sequence[RuntimeFragment],
        *,
        pre_registered: bool,
        resources: Sequence[Any],
    ) -> Iterator[None]:
        if pre_registered:
            try:
                yield
            except (
                TransferCompletionUnknownError,
                _PendingCompletionWaitInterrupted,
            ) as error:
                self._pending._retain_pending_resources(
                    error.pending_transfer_id,
                    registrations=(),
                    resources=resources,
                )
                if isinstance(error, _PendingCompletionWaitInterrupted):
                    raise error.interruption from error
                raise
            return
        sizes_by_address: dict[int, int] = {}
        for fragment in fragments:
            sizes_by_address[fragment.address] = max(
                sizes_by_address.get(fragment.address, 0), fragment.nbytes
            )
        owned = []
        primary_error: BaseException | None = None
        try:
            for address, nbytes in sizes_by_address.items():
                try:
                    result = self.engine.register_memory(address, nbytes)
                except Exception as error:
                    raise TransferEngineError(
                        f"target register_memory failed for {address}: {error}"
                    ) from error
                if result != 0:
                    raise TransferEngineError(
                        f"target register_memory failed for {address}: {result}"
                    )
                owned.append(address)
            yield
        except BaseException as error:
            primary_error = error

        if isinstance(
            primary_error,
            (
                TransferCompletionUnknownError,
                _PendingCompletionWaitInterrupted,
            ),
        ):
            self._pending._retain_pending_resources(
                primary_error.pending_transfer_id,
                registrations=owned,
                resources=resources,
            )
            if isinstance(primary_error, _PendingCompletionWaitInterrupted):
                raise primary_error.interruption from primary_error
            raise primary_error

        failures = []
        for address in reversed(owned):
            try:
                result = self.engine.unregister_memory(address)
            except Exception as error:
                failures.append((address, repr(error)))
                continue
            if result != 0:
                failures.append((address, result))
        if failures:
            detail = f"target unregister_memory failed: {failures}"
            if primary_error is not None:
                raise TransferEngineError(
                    f"{primary_error}; {detail}"
                ) from primary_error
            raise TransferEngineError(detail)
        if primary_error is not None:
            raise primary_error

    def _transfer_batch(
        self,
        endpoint: str,
        target_addresses: list[int],
        source_addresses: list[int],
        sizes: list[int],
    ) -> None:
        try:
            result = _batch_transfer_with_completion_fence(
                self.engine,
                ticket_method_name="batch_transfer_sync_read_with_ticket",
                legacy_method_name="batch_transfer_sync_read",
                arguments=(
                    endpoint,
                    target_addresses,
                    source_addresses,
                    sizes,
                ),
                max_drain_attempts=self.max_completion_drain_attempts,
                drain_timeout_ms=self.completion_drain_timeout_ms,
            )
        except _CompletionUnknown as error:
            pending_transfer_id = self._pending._retain_pending_ticket(error.ticket)
            restart_required = getattr(error.ticket, "restart_required", False)
            suffix = (
                "; legacy API exposes no drainable ticket, so this engine is "
                "restart-required"
                if restart_required
                else ""
            )
            raise TransferCompletionUnknownError(
                "batch transfer completion is unknown; registrations remain "
                f"quarantined as {pending_transfer_id}{suffix}",
                pending_transfer_id=pending_transfer_id,
            ) from error
        except _CompletionWaitInterrupted as error:
            pending_transfer_id = self._pending._retain_pending_ticket(error.ticket)
            raise _PendingCompletionWaitInterrupted(
                pending_transfer_id,
                error.interruption,
            ) from error
        except Exception as error:
            raise TransferEngineError(
                f"batch transfer from {endpoint} failed: {error}"
            ) from error
        if result != 0:
            raise TransferEngineError(
                f"batch transfer from {endpoint} failed: {result}"
            )

    def pending_transfer_ids(self) -> tuple[str, ...]:
        return self._pending.pending_transfer_ids()

    def pending_transfer_status(self, pending_transfer_id: str) -> str:
        return self._pending.pending_transfer_status(pending_transfer_id)

    def drain_pending_transfer(
        self,
        pending_transfer_id: str,
        *,
        timeout_ms: int = 1000,
    ) -> str:
        return self._pending.drain_pending_transfer(
            pending_transfer_id,
            timeout_ms=timeout_ms,
        )
