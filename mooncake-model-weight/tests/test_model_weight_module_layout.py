from __future__ import annotations

import mooncake.model_weight as model_weight
from mooncake.model_weight.binding import (
    bind_runtime_manifest,
    placement_manifest_from_runtime_manifest,
    runtime_binding_from_runtime_manifest,
)
from mooncake.model_weight.manifest import (
    ParallelRank,
    PlacementFragment,
    PlacementManifest,
    RuntimeBindingFragment,
    RuntimeBindingManifest,
    RuntimeFragment,
    RuntimeManifest,
    TensorDescriptor,
)
from mooncake.model_weight.placement import PlacementManifest as PlacementContract
from mooncake.model_weight.runtime import (
    RuntimeBindingManifest as RuntimeBindingContract,
)
from mooncake.model_weight.runtime import RuntimeManifest as RuntimeContract
from mooncake.model_weight.types import (
    ParallelRank as ParallelRankContract,
)
from mooncake.model_weight.types import (
    PlacementFragment as PlacementFragmentContract,
)
from mooncake.model_weight.types import (
    RuntimeBindingFragment as RuntimeBindingFragmentContract,
)
from mooncake.model_weight.types import RuntimeFragment as RuntimeFragmentContract
from mooncake.model_weight.types import TensorDescriptor as TensorContract


def test_responsibility_modules_preserve_public_contract_identity() -> None:
    assert model_weight.ParallelRank is ParallelRank is ParallelRankContract
    assert model_weight.TensorDescriptor is TensorDescriptor is TensorContract
    assert (
        model_weight.PlacementFragment is PlacementFragment is PlacementFragmentContract
    )
    assert model_weight.RuntimeFragment is RuntimeFragment is RuntimeFragmentContract
    assert (
        model_weight.RuntimeBindingFragment
        is RuntimeBindingFragment
        is RuntimeBindingFragmentContract
    )
    assert model_weight.PlacementManifest is PlacementManifest is PlacementContract
    assert model_weight.RuntimeManifest is RuntimeManifest is RuntimeContract
    assert (
        model_weight.RuntimeBindingManifest
        is RuntimeBindingManifest
        is RuntimeBindingContract
    )
    assert model_weight.bind_runtime_manifest is bind_runtime_manifest
    assert (
        model_weight.placement_manifest_from_runtime_manifest
        is placement_manifest_from_runtime_manifest
    )
    assert (
        model_weight.runtime_binding_from_runtime_manifest
        is runtime_binding_from_runtime_manifest
    )
