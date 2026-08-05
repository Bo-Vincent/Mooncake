"""Compatibility helpers for supported Python runtimes."""

from __future__ import annotations

from itertools import zip_longest
from typing import Iterable, Iterator


def _strict_zip(*iterables: Iterable[object]) -> Iterator[tuple[object, ...]]:
    """Provide ``zip(strict=True)`` semantics on Python 3.9."""

    sentinel = object()
    for values in zip_longest(*iterables, fillvalue=sentinel):
        if any(value is sentinel for value in values):
            raise ValueError("zip() arguments have different lengths")
        yield values
