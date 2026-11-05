"""Installed-wheel API checks for model-weight revision entry points."""

from __future__ import annotations

import unittest

import mooncake.store as native_store
from mooncake.reshard.weight.store import WeightStore, WeightStoreWriter


class TestWeightSnapshotApi(unittest.TestCase):
    def test_native_store_has_no_weight_writer_shortcut(self) -> None:
        store = native_store.MooncakeDistributedStore()

        self.assertFalse(hasattr(store, "begin_weight_snapshot"))
        self.assertFalse(hasattr(store, "begin_managed_weight_snapshot"))

    def test_weight_store_writer_replaces_parallelism_api(self) -> None:
        store = native_store.MooncakeDistributedStore()

        self.assertTrue(callable(WeightStore.weight_put))
        self.assertTrue(callable(WeightStore.weight_update_policy))
        self.assertFalse(hasattr(WeightStore, "weight_update"))
        self.assertTrue(issubclass(WeightStoreWriter, object))

        for name in (
            "get_tensor_with_parallelism",
            "batch_get_tensor_with_parallelism",
            "get_tensor_with_parallelism_into",
            "batch_get_tensor_with_parallelism_into",
            "put_tensor_with_parallelism",
            "batch_put_tensor_with_parallelism",
            "put_tensor_with_parallelism_from",
            "batch_put_tensor_with_parallelism_from",
            "upsert_tensor_with_parallelism",
            "upsert_tensor_with_parallelism_from",
            "batch_upsert_tensor_with_parallelism",
            "batch_upsert_tensor_with_parallelism_from",
        ):
            self.assertFalse(hasattr(store, name), name)

        for name in ("ParallelAxis", "TensorParallelism", "ReadTarget"):
            self.assertFalse(hasattr(native_store, name), name)
