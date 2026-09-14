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
        "weight_update_policy",
        "weight_upsert",
        "weight_migrate",
        "weight_get_operation",
        "weight_remove",
    }

    actual_methods = {
        name
        for name in dir(WeightStore)
        if not name.startswith("_") and callable(getattr(WeightStore, name))
    }

    assert actual_methods == public_methods
