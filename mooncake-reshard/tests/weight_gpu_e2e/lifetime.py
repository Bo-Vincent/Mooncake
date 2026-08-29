"""Framework-side allocation guard used by CUDA test runtimes.

The adapter deliberately lives with the test runtime rather than in Mooncake
core. A production framework adapter pins its tensor/storage object under the
framework allocator lock; this test adapter keeps the same owner alive and
activates the owning CUDA context before Store or TE sees the address.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from mooncake.reshard.transfer_engine.lifetime import TerminalTransferState
from mooncake.reshard.weight.lifetime import (
    AcquiredWeightBinding,
    weight_allocation_fence,
)
from mooncake.reshard.weight.manifest import WeightRuntimeBindingManifest


class _CudaAllocationToken:
    def __init__(self, fence, owners: tuple[object, ...]) -> None:
        self._fence = fence
        self._owners = owners
        self.released_states = []

    @property
    def fence(self):
        return self._fence

    def release_after_terminal(self, terminal_state: TerminalTransferState) -> None:
        self.released_states.append(terminal_state)
        self._owners = ()


class _CudaBindingGuard:
    def __init__(
        self,
        binding: WeightRuntimeBindingManifest,
        owners_by_fragment_id: Mapping[str, object],
    ) -> None:
        self._binding = binding
        self._owners_by_fragment_id = owners_by_fragment_id

    def acquire(
        self,
        *,
        transfer_id: str,
        expected_binding: WeightRuntimeBindingManifest,
        required_fragment_ids: Sequence[str],
    ) -> AcquiredWeightBinding:
        if not transfer_id or expected_binding != self._binding:
            raise RuntimeError("CUDA allocation guard binding differs")
        fragments_by_id = {
            fragment.fragment_id: fragment for fragment in self._binding.fragments
        }
        owners = []
        for fragment_id in required_fragment_ids:
            fragment = fragments_by_id.get(fragment_id)
            if fragment is None:
                raise RuntimeError(
                    f"CUDA allocation guard fragment is missing: {fragment_id}"
                )
            owner = self._owners_by_fragment_id.get(fragment_id)
            if owner is None:
                raise RuntimeError(
                    f"CUDA allocation guard has no local owner: {fragment_id}"
                )
            activate = getattr(owner, "activate", None)
            if callable(activate):
                activate()
            owners.append(owner)
        token = _CudaAllocationToken(
            weight_allocation_fence(
                self._binding,
                required_fragment_ids,
                token_id=(
                    f"cuda-{self._binding.instance_id}-"
                    f"{self._binding.participant_id}-{transfer_id}"
                ),
            ),
            tuple(owners),
        )
        return AcquiredWeightBinding(binding=self._binding, token=token)


def allocation_guards_for_bindings(
    bindings: Sequence[WeightRuntimeBindingManifest],
    *,
    owner_bindings: Optional[Sequence[WeightRuntimeBindingManifest]] = None,
) -> dict[tuple[str, str], _CudaBindingGuard]:
    owner_bindings = bindings if owner_bindings is None else owner_bindings
    owners_by_key = {
        (binding.instance_id, binding.participant_id): {
            fragment.fragment_id: fragment.owner for fragment in binding.fragments
        }
        for binding in owner_bindings
    }
    return {
        (binding.instance_id, binding.participant_id): _CudaBindingGuard(
            binding,
            owners_by_key.get((binding.instance_id, binding.participant_id), {}),
        )
        for binding in bindings
    }
