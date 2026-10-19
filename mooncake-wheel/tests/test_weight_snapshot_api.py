"""Installed-wheel API checks for model-weight revision entry points."""

from __future__ import annotations

import unittest

import mooncake.store as native_store
from mooncake.reshard.weight import WeightUpsertMode
from mooncake.reshard.weight.store import WeightStore, WeightStoreWriter


class TestWeightSnapshotApi(unittest.TestCase):
    def test_native_config_preserves_session_and_affinity_properties(self) -> None:
        config = native_store.ReplicateConfig()
        self.assertIsNone(config.session_ids)
        self.assertIsNone(config.residency_affinity_ids)
        config.session_ids = ["session"]
        config.residency_affinity_ids = ["affinity"]
        self.assertEqual(config.session_ids, ["session"])
        self.assertEqual(config.residency_affinity_ids, ["affinity"])
        config.residency_affinity_ids = None
        self.assertIsNone(config.residency_affinity_ids)
        self.assertEqual(config.session_ids, ["session"])

    def test_native_weight_metadata_exposes_access_recency(self) -> None:
        self.assertTrue(
            hasattr(native_store.WeightRevisionMetadata, "last_accessed_at_ms")
        )

    def test_native_store_has_no_weight_writer_shortcut(self) -> None:
        store = native_store.MooncakeDistributedStore()

        self.assertFalse(hasattr(store, "begin_weight_snapshot"))
        self.assertFalse(hasattr(store, "begin_managed_weight_snapshot"))

    def test_weight_store_writer_replaces_parallelism_api(self) -> None:
        store = native_store.MooncakeDistributedStore()

        self.assertTrue(callable(WeightStore.weight_put))
        self.assertTrue(callable(WeightStore.weight_upsert))
        self.assertTrue(callable(WeightStore.weight_update_policy))
        self.assertFalse(hasattr(WeightStore, "weight_update"))
        self.assertEqual(int(WeightUpsertMode.PUT_FIRST), 0)
        self.assertEqual(int(WeightUpsertMode.DELETE_FIRST), 1)
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
