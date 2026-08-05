from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence, cast

from ..storage_manifest import WeightManifest
from .contracts import UploadReceipt, WeightUploadPlan
from .errors import WeightStoreError
from .payload import PayloadStoreOperations

if TYPE_CHECKING:
    from .client import WeightStore


@dataclass(frozen=True)
class _UploadDecision:
    plan_digest: str
    decision: str


def _plan_digest(manifest: WeightManifest) -> str:
    return hashlib.sha256(manifest.to_json().encode()).hexdigest()


def _decision_payload(plan: WeightUploadPlan, decision: str) -> bytes:
    return json.dumps(
        {
            "decision": decision,
            "plan_digest": _plan_digest(plan.manifest),
            "record_type": "weight-upload-decision",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _decode_decision(value: bytes | bytearray | str) -> _UploadDecision:
    def reject_constant(constant: str) -> None:
        raise ValueError(f"non-finite JSON number is unsupported: {constant}")

    def reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = item
        return result

    parsed: object = json.loads(
        value,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_fields,
    )
    expected = {
        "decision",
        "plan_digest",
        "record_type",
    }
    if not isinstance(parsed, Mapping):
        raise ValueError("upload decision schema fields do not match contract")
    raw = cast(Mapping[object, object], parsed)
    if {key for key in raw if type(key) is str} != expected or any(
        type(key) is not str for key in raw
    ):
        raise ValueError("upload decision schema fields do not match contract")
    raw = cast(Mapping[str, object], raw)
    if raw["record_type"] != "weight-upload-decision":
        raise ValueError("invalid upload decision record_type")
    if raw["decision"] not in ("abort", "commit"):
        raise ValueError("invalid upload decision")
    digest = raw["plan_digest"]
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("invalid upload decision plan_digest")
    return _UploadDecision(plan_digest=digest, decision=raw["decision"])


class WeightUploadSession:
    def __init__(
        self,
        client: WeightStore,
        payloads: PayloadStoreOperations,
    ) -> None:
        self.client = client
        self.payloads = payloads

    def abort_upload(
        self,
        plan: WeightUploadPlan,
        receipts: Sequence[UploadReceipt],
    ) -> None:
        self.payloads.validate_receipts(plan, receipts, require_complete=False)
        persisted = self._load_manifest_if_present(plan.manifest.manifest_key)
        if persisted == plan.manifest:
            raise WeightStoreError("cannot abort a published weight revision")
        self._claim_upload(plan, "abort")
        persisted = self._load_manifest_if_present(plan.manifest.manifest_key)
        if persisted == plan.manifest:
            raise WeightStoreError("cannot abort a published weight revision")
        keys = (
            [operation.target.object_key for operation in plan.operations]
            if persisted is None
            else self.payloads.unreferenced_plan_payload_keys(plan, persisted)
        )
        failures = self.payloads.remove_keys(keys)
        if failures:
            raise WeightStoreError(f"upload cleanup failed: {failures}")

    def finalize_upload_session(self, plan: WeightUploadPlan) -> None:
        decision = self._load_decision_if_present(plan.control_key)
        if decision is None:
            persisted = self._load_manifest_if_present(plan.manifest.manifest_key)
            if persisted == plan.manifest:
                self._claim_upload(plan, "commit")
                decision = self._load_decision_if_present(plan.control_key)
            else:
                raise WeightStoreError("upload session has no terminal decision")
        if decision is None:
            raise WeightStoreError("upload session terminal decision is incomplete")
        self._validate_decision_owner(plan, decision)
        if decision.decision == "commit":
            persisted = self._load_manifest_if_present(plan.manifest.manifest_key)
            if persisted is None:
                raise WeightStoreError("committed upload session has no manifest")
            if persisted != plan.manifest:
                failures = self.payloads.remove_keys(
                    self.payloads.unreferenced_plan_payload_keys(plan, persisted)
                )
                if failures:
                    raise WeightStoreError(
                        f"conflicting upload cleanup failed: {failures}"
                    )
        else:
            persisted = self._load_manifest_if_present(plan.manifest.manifest_key)
            if persisted == plan.manifest:
                raise WeightStoreError("aborted upload has a published manifest")
            keys = (
                [operation.target.object_key for operation in plan.operations]
                if persisted is None
                else self.payloads.unreferenced_plan_payload_keys(plan, persisted)
            )
            failures = self.payloads.remove_keys(keys)
            if failures:
                raise WeightStoreError(f"upload cleanup failed: {failures}")

    def commit(
        self,
        plan: WeightUploadPlan,
        receipts: Sequence[UploadReceipt],
    ) -> WeightManifest:
        self.payloads.validate_receipts(plan, receipts, require_complete=True)
        payload_keys = [receipt.object_key for receipt in receipts]
        if self._current_upload_decision(plan) is None:
            self.payloads.require_complete_payloads(payload_keys)
        self._claim_upload(plan, "commit")
        self.payloads.require_complete_payloads(payload_keys)
        existing = self._load_manifest_if_present(plan.manifest.manifest_key)
        if existing is not None:
            return self._resolve_manifest_commit(plan, existing, payload_keys)

        put_error: Exception | None = None
        result: int | None = None
        try:
            result = self.client.store.put(
                plan.manifest.manifest_key,
                plan.manifest.to_json().encode(),
                self.client.config_factory([plan.manifest.group_id], "metadata"),
            )
        except Exception as error:
            put_error = error

        persisted = self._load_manifest_if_present(plan.manifest.manifest_key)
        if persisted is not None:
            return self._resolve_manifest_commit(plan, persisted, payload_keys)

        detail = f"manifest put failed: {result}"
        if put_error is not None:
            detail = f"manifest put failed: {put_error}"
        raise WeightStoreError(detail) from put_error

    def require_writable(
        self,
        plan: WeightUploadPlan,
        *,
        cleanup_keys: Sequence[str] = (),
    ) -> None:
        decision = self._current_upload_decision(plan)
        if decision is None:
            return
        persisted = self._load_manifest_if_present(plan.manifest.manifest_key)
        if decision == "commit" and (persisted is None or persisted == plan.manifest):
            return
        protected_keys: set[str] = (
            set()
            if persisted is None
            else {fragment.object_key for fragment in persisted.fragments}
        )
        failures = self.payloads.remove_keys(
            [key for key in cleanup_keys if key not in protected_keys]
        )
        detail = (
            "weight upload already chose abort"
            if decision == "abort"
            else "weight upload lost to a conflicting revision"
        )
        if failures:
            detail += f"; cleanup failed: {failures}"
        raise WeightStoreError(detail)

    def _claim_upload(self, plan: WeightUploadPlan, decision: str) -> None:
        put_error: Exception | None = None
        result: int | None = None
        try:
            result = self.client.store.put(
                plan.control_key,
                _decision_payload(plan, decision),
                self.client.config_factory([plan.session_group_id], "metadata"),
            )
        except Exception as error:
            put_error = error

        persisted = self._load_decision_if_present(plan.control_key)
        if persisted is None:
            detail = f"upload decision is not complete; retry: {result}"
            if put_error is not None:
                detail = f"upload decision is not complete; retry: {put_error}"
            raise WeightStoreError(detail) from put_error
        self._validate_decision_owner(plan, persisted)
        if persisted.decision != decision:
            raise WeightStoreError(f"weight upload already chose {persisted.decision}")

    def _current_upload_decision(self, plan: WeightUploadPlan) -> str | None:
        persisted = self._load_decision_if_present(plan.control_key)
        if persisted is None:
            return None
        self._validate_decision_owner(plan, persisted)
        return persisted.decision

    @staticmethod
    def _validate_decision_owner(
        plan: WeightUploadPlan, persisted: _UploadDecision
    ) -> None:
        if persisted.plan_digest != _plan_digest(plan.manifest):
            raise WeightStoreError("upload decision belongs to another plan")

    def _load_decision_if_present(self, control_key: str) -> _UploadDecision | None:
        exists = self.client.store.is_exist(control_key)
        if exists == 0:
            return None
        if exists != 1:
            raise WeightStoreError(
                f"upload decision existence check failed: {control_key}: {exists}"
            )
        try:
            return _decode_decision(self.client.store.get(control_key))
        except Exception as error:
            raise WeightStoreError(f"invalid upload decision: {control_key}") from error

    def _load_manifest_if_present(self, manifest_key: str) -> WeightManifest | None:
        exists = self.client.store.is_exist(manifest_key)
        if exists == 0:
            return None
        if exists != 1:
            raise WeightStoreError(
                f"manifest existence check failed: {manifest_key}: {exists}"
            )
        return self.client.load_manifest(manifest_key)

    def _resolve_manifest_commit(
        self,
        plan: WeightUploadPlan,
        persisted: WeightManifest,
        payload_keys: Sequence[str],
    ) -> WeightManifest:
        if persisted == plan.manifest:
            return persisted
        persisted_keys = {fragment.object_key for fragment in persisted.fragments}
        cleanup_failures = self.payloads.remove_keys(
            [key for key in payload_keys if key not in persisted_keys]
        )
        detail = f"conflicting weight revision: {plan.manifest.manifest_key}"
        if cleanup_failures:
            detail += f"; cleanup failed: {cleanup_failures}"
        raise WeightStoreError(detail)
