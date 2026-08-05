"""Resource-neutral physical transfer batches."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


_MAX_U64 = (1 << 64) - 1


class TransferDirection(str, Enum):
    READ = "read"
    WRITE = "write"


@dataclass(frozen=True)
class TransferBatch:
    endpoint: str
    source_addresses: tuple[int, ...]
    target_addresses: tuple[int, ...]
    sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.endpoint) is not str or not self.endpoint:
            raise ValueError("transfer endpoint must be a non-empty string")
        object.__setattr__(self, "source_addresses", tuple(self.source_addresses))
        object.__setattr__(self, "target_addresses", tuple(self.target_addresses))
        object.__setattr__(self, "sizes", tuple(self.sizes))
        lengths = {
            len(self.source_addresses),
            len(self.target_addresses),
            len(self.sizes),
        }
        if lengths != {len(self.sizes)} or not self.sizes:
            raise ValueError("transfer batch ranges must be non-empty and aligned")
        for name, values in (
            ("source address", self.source_addresses),
            ("target address", self.target_addresses),
            ("size", self.sizes),
        ):
            for value in values:
                if type(value) is not int or value <= 0 or value > _MAX_U64:
                    raise ValueError(f"transfer batch {name} is invalid")

    @property
    def operation_count(self) -> int:
        return len(self.sizes)

    @property
    def nbytes(self) -> int:
        return sum(self.sizes)


@dataclass(frozen=True)
class TransferBatchReceipt:
    endpoint: str
    direction: TransferDirection
    operation_count: int
    nbytes: int
