from __future__ import annotations

from mooncake.reshard.weight.store import WeightStore


def test_managed_weight_store_uses_store_aligned_api_names() -> None:
    public_methods = {
        "weight_put",
        "weight_get",
        "weight_is_exist",
        "weight_get_metadata",
        "weight_get_size",
        "weight_list",
        "weight_update",
        "weight_migrate",
        "weight_get_operation",
        "weight_remove",
    }

    assert public_methods <= set(dir(WeightStore))
    assert "begin_managed_weight_snapshot" not in dir(WeightStore)
    assert "get_weight_revision" not in dir(WeightStore)
    assert "list_weight_revisions" not in dir(WeightStore)
    assert "load_weight_revision" not in dir(WeightStore)
