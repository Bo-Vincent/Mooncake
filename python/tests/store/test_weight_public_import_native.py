"""Verify the public weight API against the built C++ Store extension."""

from importlib.util import find_spec

import pytest


def test_weight_types_are_available_through_public_store_facade() -> None:
    if find_spec("mooncake._store") is None:
        pytest.skip("C++ Store native extension is not built")

    import mooncake.store as store
    from mooncake.store import (
        WeightAffinitySummary,
        WeightAvailabilityState,
        WeightLineageIdentity,
        WeightLineageMetadata,
        WeightManagementError,
        WeightManifestReference,
        WeightMigrationMode,
        WeightOperationKind,
        WeightResidencyOperation,
        WeightResidencyState,
        WeightRevisionIdentity,
        WeightRevisionLease,
        WeightRevisionMetadata,
        WeightRevisionView,
        WeightStoragePolicy,
        WeightUpsertClaim,
        WeightUpsertImportSummary,
        WeightUpsertMode,
        WeightUpsertPhase,
    )

    public_types = (
        WeightAffinitySummary,
        WeightAvailabilityState,
        WeightLineageIdentity,
        WeightLineageMetadata,
        WeightManagementError,
        WeightManifestReference,
        WeightMigrationMode,
        WeightOperationKind,
        WeightResidencyOperation,
        WeightResidencyState,
        WeightRevisionIdentity,
        WeightRevisionLease,
        WeightRevisionMetadata,
        WeightRevisionView,
        WeightStoragePolicy,
        WeightUpsertClaim,
        WeightUpsertImportSummary,
        WeightUpsertMode,
        WeightUpsertPhase,
    )
    for public_type in public_types:
        assert public_type.__name__ in store.__all__
        assert getattr(store, public_type.__name__) is public_type

    assert WeightResidencyState.HOT.name == "HOT"
    assert WeightMigrationMode.MANUAL.name == "MANUAL"
