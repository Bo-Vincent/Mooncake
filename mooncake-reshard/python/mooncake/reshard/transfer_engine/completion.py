from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Sequence
from uuid import uuid4


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
        try:
            result = getattr(engine, legacy_method_name)(*arguments)
        except Exception:
            raise
        except BaseException as error:
            raise _CompletionWaitInterrupted(
                _UndrainableCompletionUnknownTicket(),
                error,
            ) from error
        if _completion_status_name(result) == "COMPLETION_UNKNOWN":
            raise _CompletionUnknown(_UndrainableCompletionUnknownTicket())
        return result

    try:
        ticket = ticket_method(*arguments)
    except Exception:
        raise
    except BaseException as error:
        # The call may have crossed the native submission boundary before the
        # interruption. Without a returned ticket, only restart is safe.
        raise _CompletionWaitInterrupted(
            _UndrainableCompletionUnknownTicket(),
            error,
        ) from error
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


class PendingTransferManager:
    """Own completion quarantine and submission state for one engine identity."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.engine_identity = _canonical_engine_identity(engine)

    def _retain_pending_ticket(self, ticket: Any) -> str:
        pending_transfer_id = uuid4().hex
        with _pending_transfer_lock:
            _pending_transfers[pending_transfer_id] = _PendingTransfer(
                engine=self.engine,
                engine_identity=self.engine_identity,
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
            if pending.engine_identity != self.engine_identity:
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
                if transfer.engine_identity == self.engine_identity
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
            if self.engine_identity in _active_submission_engine_ids:
                raise TransferEngineError(
                    "transfer engine already has an active resource transfer submission"
                )
            _active_submission_engine_ids.add(self.engine_identity)

    def _release_submission(self) -> None:
        with _pending_transfer_lock:
            _active_submission_engine_ids.remove(self.engine_identity)

    def _get_pending_transfer(
        self,
        pending_transfer_id: str,
    ) -> _PendingTransfer:
        pending = _pending_transfers.get(pending_transfer_id)
        if pending is None or pending.engine_identity != self.engine_identity:
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
                    if pending.engine_identity == self.engine_identity
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
