from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Generator, Sequence

from ..manifest import RuntimeBindingFragment
from .backend import StoreBackend
from .errors import WeightStoreError


class StoreBufferRegistration:
    def __init__(self, store: StoreBackend) -> None:
        self.store = store

    @contextmanager
    def registered(
        self,
        fragments: Sequence[RuntimeBindingFragment],
        *,
        pre_registered: bool,
    ) -> Generator[None, None, None]:
        if pre_registered:
            yield
            return

        requests: dict[int, tuple[int, str]] = {}
        for fragment in fragments:
            previous = requests.get(fragment.address)
            if previous is None or fragment.nbytes > previous[0]:
                requests[fragment.address] = (fragment.nbytes, fragment.fragment_id)

        owned: list[int] = []
        primary_error: BaseException | None = None
        try:
            for address, (nbytes, fragment_id) in requests.items():
                result = self.store.register_buffer(address, nbytes)
                if result == 0:
                    owned.append(address)
                else:
                    raise WeightStoreError(
                        f"register_buffer failed for {fragment_id}: {result}"
                    )
            yield
        except BaseException as error:
            primary_error = error

        failures: list[tuple[int, str | int]] = []
        for address in reversed(owned):
            try:
                result = self.store.unregister_buffer(address)
            except Exception as error:
                failures.append((address, repr(error)))
                continue
            if result != 0:
                failures.append((address, result))
        if failures:
            detail = f"unregister_buffer failed: {failures}"
            if primary_error is not None:
                raise WeightStoreError(f"{primary_error}; {detail}") from primary_error
            raise WeightStoreError(detail)
        if primary_error is not None:
            raise primary_error
