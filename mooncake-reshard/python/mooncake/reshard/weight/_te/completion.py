"""Compatibility exports for the resource-neutral completion coordinator."""

from ...transfer_engine.completion import (
    PendingTransferManager,
    TransferCompletionUnknownError,
    TransferEngineError,
)

__all__ = [
    "PendingTransferManager",
    "TransferCompletionUnknownError",
    "TransferEngineError",
]
