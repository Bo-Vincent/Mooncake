from __future__ import annotations

from collections.abc import Sequence

from ..manifest import (
    WeightPlacementManifest,
    WeightRuntimeBindingManifest,
)
from ..storage_manifest import WeightManifest
from .contracts import UploadReceipt, WeightLoadPlan, WeightUploadPlan
from .errors import WeightStoreError
from .backend import (
    StoreBackend,
    StoreConfigFactory,
    default_config_factory,
)
from .load import WeightLoadService
from .payload import PayloadStoreOperations
from .registration import StoreBufferRegistration
from .session import WeightUploadSession
from .upload import WeightUploadService


def _require_upload_plan(plan: WeightUploadPlan) -> None:
    if not isinstance(plan, WeightUploadPlan):
        raise WeightStoreError("plan must be a WeightUploadPlan")


def _require_load_plan(plan: WeightLoadPlan) -> None:
    if not isinstance(plan, WeightLoadPlan):
        raise WeightStoreError("plan must be a WeightLoadPlan")


class WeightStore:
    def __init__(
        self,
        store: object,
        *,
        key_prefix: str = "weights",
        config_factory: StoreConfigFactory | None = None,
        max_range_bytes: int = 64 * 1024 * 1024,
        max_ranges_per_request: int = 1024,
        max_region_segments: int = 1_000_000,
    ) -> None:
        if (
            max_range_bytes <= 0
            or max_ranges_per_request <= 0
            or max_region_segments <= 0
        ):
            raise ValueError("range limits must be positive")
        self.store = StoreBackend(store)
        self.key_prefix = key_prefix.strip("/")
        self.config_factory = config_factory or default_config_factory
        self.max_range_bytes = max_range_bytes
        self.max_ranges_per_request = max_ranges_per_request
        self.max_region_segments = max_region_segments
        self.registration = StoreBufferRegistration(self.store)
        self._payloads = PayloadStoreOperations(self)
        self._session = WeightUploadSession(self, self._payloads)
        self._upload = WeightUploadService(self, self._payloads, self._session)
        self._load = WeightLoadService(self)

    def prepare_upload(
        self,
        source_placement: WeightPlacementManifest,
        source_bindings: Sequence[WeightRuntimeBindingManifest],
        *,
        namespace: str = "default",
    ) -> WeightUploadPlan:
        return self._upload.prepare_upload(
            source_placement,
            source_bindings,
            namespace=namespace,
        )

    def upload(
        self,
        plan: WeightUploadPlan,
        source_placement: WeightPlacementManifest,
        source_binding: WeightRuntimeBindingManifest,
        *,
        pre_registered: bool = False,
        source_worker_id: str | None = None,
    ) -> tuple[UploadReceipt, ...]:
        _require_upload_plan(plan)
        return self._upload.upload(
            plan,
            source_placement,
            source_binding,
            source_worker_id=source_worker_id,
            pre_registered=pre_registered,
        )

    def abort_upload(
        self,
        plan: WeightUploadPlan,
        receipts: Sequence[UploadReceipt],
    ) -> None:
        _require_upload_plan(plan)
        self._session.abort_upload(plan, receipts)

    def finalize_upload_session(self, plan: WeightUploadPlan) -> None:
        _require_upload_plan(plan)
        self._session.finalize_upload_session(plan)

    def commit(
        self,
        plan: WeightUploadPlan,
        receipts: Sequence[UploadReceipt],
    ) -> WeightManifest:
        _require_upload_plan(plan)
        return self._session.commit(plan, receipts)

    def load_manifest(self, manifest_key: str) -> WeightManifest:
        return self._load.load_manifest(manifest_key)

    def plan_load(
        self,
        manifest: WeightManifest,
        target_placement: WeightPlacementManifest,
        target_bindings: Sequence[WeightRuntimeBindingManifest],
    ) -> WeightLoadPlan:
        return self._load.plan_load(manifest, target_placement, target_bindings)

    def load(
        self,
        plan: WeightLoadPlan,
        target_placement: WeightPlacementManifest,
        target_binding: WeightRuntimeBindingManifest,
        *,
        pre_registered: bool = False,
        target_worker_id: str | None = None,
    ) -> None:
        _require_load_plan(plan)
        self._load.load(
            plan,
            target_placement,
            target_binding,
            target_worker_id=target_worker_id,
            pre_registered=pre_registered,
        )


__all__ = ["WeightStore", "WeightStoreError"]
