from __future__ import annotations

from dataclasses import replace

import pytest

from .lifetime import allocation_guards_for_bindings
from .manifests import _rank_manifests, _tensor


class _Buffer:
    def __init__(self, pointer: int, size: int) -> None:
        self.pointer = pointer
        self.size = size


def test_rank_manifests_build_one_complete_global_placement() -> None:
    buffers = [_Buffer(0x1000, 8), _Buffer(0x2000, 8)]

    runtime = _rank_manifests(
        tensor=_tensor(16),
        revision="revision-1",
        prefix="source",
        buffers=buffers,
    )

    assert not hasattr(runtime, "placements")
    assert len(runtime.placement.parts) == 2
    assert {part.rank.tp for part in runtime.placement.parts} == {0, 1}
    assert {part.participant_id for part in runtime.placement.parts} == {
        binding.participant_id for binding in runtime.bindings
    }
    placement_ids = {
        fragment.placement_fragment_id for fragment in runtime.placement.fragments
    }
    binding_ids = {
        fragment.placement_fragment_id
        for binding in runtime.bindings
        for fragment in binding.fragments
    }
    assert len(placement_ids) == 2
    assert all(fragment_id.startswith("sha256:") for fragment_id in placement_ids)
    assert binding_ids == placement_ids


def test_cuda_guard_requires_owner_on_the_attested_binding() -> None:
    runtime = _rank_manifests(
        tensor=_tensor(8),
        revision="revision-1",
        prefix="source",
        buffers=[_Buffer(0x1000, 8)],
    )
    binding = runtime.bindings[0]
    ownerless = replace(
        binding,
        fragments=tuple(
            replace(fragment, owner=None) for fragment in binding.fragments
        ),
    )
    guard = allocation_guards_for_bindings((ownerless,))[
        (ownerless.instance_id, ownerless.participant_id)
    ]

    with pytest.raises(RuntimeError, match="no local owner"):
        guard.acquire(
            transfer_id="transfer-1",
            expected_binding=ownerless,
            required_fragment_ids=tuple(
                fragment.fragment_id for fragment in ownerless.fragments
            ),
        )


def test_cuda_guard_rejects_owner_binding_with_different_allocation() -> None:
    runtime = _rank_manifests(
        tensor=_tensor(8),
        revision="revision-1",
        prefix="source",
        buffers=[_Buffer(0x1000, 8)],
    )
    binding = runtime.bindings[0]
    fragment = binding.fragments[0]
    other = replace(
        binding,
        fragments=(
            replace(
                fragment,
                address=0x2000,
                storage_address=0x2000,
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="owner binding differs"):
        allocation_guards_for_bindings(
            (other,),
            owner_bindings=(binding,),
        )
