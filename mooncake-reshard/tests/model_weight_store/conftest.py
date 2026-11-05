from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from mooncake.reshard.weight._store import backend


@dataclass(frozen=True)
class NativeMappingEntryDouble:
    key: str
    size: int
    affinity_id: str


@pytest.fixture(autouse=True)
def native_mapping_constructor_for_in_memory_tests(monkeypatch, request):
    if request.module.__name__.rsplit(".", 1)[-1] not in {
        "test_managed_revision",
        "test_weight_upsert",
        "test_snapshot_writer",
        "test_management_e2e",
    }:
        return
    original = backend.import_module

    def import_module(name):
        if name == "mooncake.store":
            return SimpleNamespace(WeightAffinityMappingEntry=NativeMappingEntryDouble)
        return original(name)

    monkeypatch.setattr(backend, "import_module", import_module)
