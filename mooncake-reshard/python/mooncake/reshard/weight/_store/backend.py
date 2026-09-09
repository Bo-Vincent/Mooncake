"""Fail-closed adapter for the native Mooncake Store object boundary."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import import_module
from typing import Any, Literal, Optional, Protocol, cast

from ..._typing import TypeAlias

from .errors import WeightStoreError
from ..management import (
    WeightManagementError,
    WeightManagementErrorCode,
    WeightManagementTransportError,
    WeightManifestReference,
    WeightResidencyOperation,
    WeightResidencyState,
    WeightRevisionIdentity,
    WeightRevisionLease,
    WeightRevisionMetadata,
    WeightRevisionPage,
    WeightRevisionView,
    WeightStoragePolicy,
    lease_from_native,
    metadata_from_native,
    operation_from_native,
    view_from_native,
)


StoreRecordType: TypeAlias = Literal["payload", "metadata"]
StoreConfigFactory: TypeAlias = Callable[[Sequence[str], StoreRecordType], object]
RangeResults: TypeAlias = tuple[tuple[tuple[int, ...], ...], ...]


class _NativeReplicateConfig(Protocol):
    group_ids: list[str]
    with_hard_pin: bool
    data_type: object


def default_config_factory(
    group_ids: Sequence[str], record_type: StoreRecordType
) -> object:
    """Build a native replication config without importing it in canonical code."""

    try:
        module = import_module("mooncake.store")
        replicate_config_type = getattr(module, "ReplicateConfig")
        object_data_type = getattr(module, "ObjectDataType")
    except (ImportError, AttributeError) as error:
        raise WeightStoreError(
            "native Mooncake Store configuration is unavailable"
        ) from error
    if not callable(replicate_config_type):
        raise WeightStoreError("native ReplicateConfig constructor is invalid")
    try:
        config = cast(_NativeReplicateConfig, replicate_config_type())
        config.group_ids = list(group_ids)
    except AttributeError as error:
        raise WeightStoreError(
            "native Mooncake Store must expose ReplicateConfig.group_ids"
        ) from error
    try:
        config.with_hard_pin = True
        if record_type == "payload":
            config.data_type = object_data_type.WEIGHT
        else:
            config.data_type = object_data_type.METADATA
    except (AttributeError, TypeError, ValueError) as error:
        raise WeightStoreError(
            "native Mooncake Store configuration is invalid"
        ) from error
    return config


class StoreBackend:
    """Normalize the raw Store API to checked operations used by weight flows."""

    def __init__(self, raw: object) -> None:
        self._raw = raw

    def weight_get_object(self, key: str) -> bytes:
        result = self._call("get", key)
        if isinstance(result, bytearray):
            return bytes(result)
        if isinstance(result, bytes):
            return result
        raise WeightStoreError(f"get returned invalid payload for {key}")

    def weight_put_object(self, key: str, value: bytes, config: object) -> int:
        return self._status("put", key, value, config)

    def weight_remove_object(self, key: str, *, force: bool) -> int:
        return self._status("remove", key, force=force)

    def weight_is_exist_object(self, key: str) -> int:
        result = self._call("is_exist", key)
        if type(result) is not int:
            raise WeightStoreError(f"is_exist returned invalid status for {key}")
        return result

    def weight_batch_is_exist(
        self, keys: Sequence[str]
    ) -> Optional[tuple[int, ...]]:
        candidate = self._optional_method("batch_is_exist")
        if candidate is None:
            return None
        result = self._invoke(candidate, list(keys), operation="batch_is_exist")
        if type(result) is int:
            raise WeightStoreError(f"existence check failed: {result}")
        return self._int_sequence(result, "batch_is_exist")

    def weight_batch_put_from(
        self,
        keys: Sequence[str],
        addresses: Sequence[int],
        sizes: Sequence[int],
        config: object,
    ) -> tuple[int, ...]:
        result = self._call(
            "batch_put_from",
            list(keys),
            list(addresses),
            list(sizes),
            config,
        )
        return self._int_sequence(result, "batch_put_from")

    def register_buffer(self, address: int, nbytes: int) -> int:
        return self._status("register_buffer", address, nbytes)

    def unregister_buffer(self, address: int) -> int:
        return self._status("unregister_buffer", address)

    def weight_put_begin_import(
        self,
        identity: WeightRevisionIdentity,
        *,
        payload_group_id: str,
        expected_payload_count: int,
        expected_logical_bytes: int,
        policy: Optional[WeightStoragePolicy],
        affinity_count: int,
        affinity_digest: str,
    ) -> WeightRevisionMetadata:
        value = self._management_call(
            "begin_weight_import",
            *self._identity_args(identity),
            payload_group_id,
            expected_payload_count,
            expected_logical_bytes,
            policy is not None,
            int(policy.preferred_residency) if policy is not None else 0,
            policy.mixed_hot_ratio if policy is not None else 0.5,
            int(policy.migration_mode) if policy is not None else 0,
            affinity_count,
            affinity_digest,
        )
        return metadata_from_native(value)

    def weight_put_commit_import(
        self,
        identity: WeightRevisionIdentity,
        *,
        expected_metadata_generation: int,
        manifest: WeightManifestReference,
    ) -> WeightRevisionMetadata:
        value = self._management_call(
            "commit_weight_import",
            *self._identity_args(identity),
            expected_metadata_generation,
            manifest.manifest_key,
            manifest.manifest_sha256,
            manifest.payload_group_id,
            manifest.payload_keys_sha256,
            manifest.payload_count,
            manifest.logical_bytes,
        )
        return metadata_from_native(value)

    def weight_put_abort_import(
        self, identity: WeightRevisionIdentity, expected_metadata_generation: int
    ) -> WeightRevisionMetadata:
        value = self._management_call(
            "abort_weight_import",
            *self._identity_args(identity),
            expected_metadata_generation,
        )
        return metadata_from_native(value)

    def weight_get_metadata(
        self, identity: WeightRevisionIdentity
    ) -> WeightRevisionView:
        value = self._management_call(
            "get_weight_revision", *self._identity_args(identity)
        )
        return view_from_native(value)

    def weight_list(
        self,
        *,
        tenant_id: str,
        namespace: str,
        resource_id: str,
        page_token: str,
        limit: int,
    ) -> WeightRevisionPage:
        value = self._management_call(
            "list_weight_revisions",
            tenant_id,
            namespace,
            resource_id,
            page_token,
            limit,
        )
        if isinstance(value, WeightRevisionPage):
            return value
        native = cast(Any, value)
        return WeightRevisionPage(
            revisions=tuple(view_from_native(item) for item in native.revisions),
            next_page_token=native.next_page_token,
        )

    def acquire_weight_revision_lease(
        self,
        identity: WeightRevisionIdentity,
        *,
        expected_metadata_generation: int,
        holder: str,
        ttl_ms: int,
    ) -> WeightRevisionLease:
        value = self._management_call(
            "acquire_weight_revision_lease",
            *self._identity_args(identity),
            expected_metadata_generation,
            holder,
            ttl_ms,
        )
        return lease_from_native(value)

    def renew_weight_revision_lease(
        self, *, tenant_id: str, lease_id: int, ttl_ms: int
    ) -> WeightRevisionLease:
        return lease_from_native(
            self._management_call(
                "renew_weight_revision_lease", tenant_id, lease_id, ttl_ms
            )
        )

    def release_weight_revision_lease(self, *, tenant_id: str, lease_id: int) -> None:
        self._management_call("release_weight_revision_lease", tenant_id, lease_id)

    def weight_migrate(
        self,
        identity: WeightRevisionIdentity,
        *,
        expected_metadata_generation: int,
        target_residency: WeightResidencyState,
        mixed_hot_ratio: Optional[float],
    ) -> WeightResidencyOperation:
        value = self._management_call(
            "start_weight_residency_operation",
            *self._identity_args(identity),
            expected_metadata_generation,
            int(target_residency),
            mixed_hot_ratio,
        )
        return operation_from_native(value)

    def weight_update(
        self,
        identity: WeightRevisionIdentity,
        *,
        expected_metadata_generation: int,
        policy: WeightStoragePolicy,
    ) -> WeightRevisionMetadata:
        value = self._management_call(
            "update_weight_policy",
            *self._identity_args(identity),
            expected_metadata_generation,
            int(policy.preferred_residency),
            policy.mixed_hot_ratio,
            int(policy.migration_mode),
        )
        return metadata_from_native(value)

    def weight_get_operation(
        self, *, tenant_id: str, operation_id: int
    ) -> WeightResidencyOperation:
        return operation_from_native(
            self._management_call("query_weight_operation", tenant_id, operation_id)
        )

    def weight_reconcile(
        self, identity: WeightRevisionIdentity
    ) -> WeightRevisionMetadata:
        return metadata_from_native(
            self._management_call(
                "reconcile_weight_revision", *self._identity_args(identity)
            )
        )

    def weight_remove(
        self, identity: WeightRevisionIdentity, expected_metadata_generation: int
    ) -> WeightRevisionMetadata:
        return metadata_from_native(
            self._management_call(
                "delete_weight_revision",
                *self._identity_args(identity),
                expected_metadata_generation,
            )
        )

    def weight_get_into_ranges(
        self,
        addresses: Sequence[int],
        all_keys: Sequence[Sequence[str]],
        all_target_offsets: Sequence[Sequence[Sequence[int]]],
        all_source_offsets: Sequence[Sequence[Sequence[int]]],
        all_sizes: Sequence[Sequence[Sequence[int]]],
    ) -> RangeResults:
        result = self._call(
            "get_into_ranges",
            list(addresses),
            [list(keys) for keys in all_keys],
            [[list(offsets) for offsets in groups] for groups in all_target_offsets],
            [[list(offsets) for offsets in groups] for groups in all_source_offsets],
            [[list(sizes) for sizes in groups] for groups in all_sizes],
        )
        return self._range_results(result)

    def _status(self, method_name: str, *args: object, **kwargs: object) -> int:
        result = self._call(method_name, *args, **kwargs)
        if type(result) is not int:
            raise WeightStoreError(f"{method_name} returned an invalid status")
        return result

    def _management_call(self, method_name: str, *args: object) -> object:
        result = self._call(method_name, *args)
        if not isinstance(result, tuple) or len(result) != 3:
            raise WeightStoreError(
                f"{method_name} returned an invalid management result"
            )
        value, domain_error, transport_error = result
        if type(domain_error) is not int or type(transport_error) is not int:
            raise WeightStoreError(f"{method_name} returned invalid error codes")
        if transport_error:
            raise WeightManagementTransportError(transport_error)
        if domain_error:
            raise WeightManagementError(WeightManagementErrorCode(domain_error))
        if value is None:
            raise WeightStoreError(f"{method_name} returned no value")
        return value

    @staticmethod
    def _identity_args(identity: WeightRevisionIdentity) -> tuple[object, ...]:
        if not isinstance(identity, WeightRevisionIdentity):
            raise WeightStoreError("identity must be a WeightRevisionIdentity")
        return (
            identity.tenant_id,
            identity.namespace,
            identity.resource_id,
            identity.revision,
            identity.weight_generation,
        )

    def _call(self, method_name: str, *args: object, **kwargs: object) -> object:
        method = self._required_method(method_name)
        return self._invoke(method, *args, operation=method_name, **kwargs)

    def _required_method(self, method_name: str) -> Callable[..., object]:
        method = self._optional_method(method_name)
        if method is None:
            raise WeightStoreError(f"Store backend does not implement {method_name}")
        return method

    def _optional_method(
        self,
        method_name: str,
    ) -> Optional[Callable[..., object]]:
        try:
            candidate = getattr(self._raw, method_name, None)
        except Exception as error:
            raise WeightStoreError(
                f"failed to access Store method {method_name}"
            ) from error
        if candidate is None:
            return None
        if not callable(candidate):
            raise WeightStoreError(f"Store attribute {method_name} is not callable")
        return candidate

    @staticmethod
    def _invoke(
        method: Callable[..., object],
        *args: object,
        operation: str,
        **kwargs: object,
    ) -> object:
        try:
            return method(*args, **kwargs)
        except Exception as error:
            raise WeightStoreError(f"{operation} failed: {error}") from error

    @staticmethod
    def _int_sequence(value: object, operation: str) -> tuple[int, ...]:
        if type(value) is int:
            raise WeightStoreError(f"{operation} failed: {value}")
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(
            value, Sequence
        ):
            raise WeightStoreError(f"{operation} returned invalid result sequence")
        values = tuple(cast(Sequence[object], value))
        if any(type(item) is not int for item in values):
            raise WeightStoreError(f"{operation} returned non-integer status")
        return cast(tuple[int, ...], values)

    @classmethod
    def _range_results(cls, value: object) -> RangeResults:
        if type(value) is int:
            raise WeightStoreError(f"get_into_ranges failed: {value}")
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(
            value, Sequence
        ):
            raise WeightStoreError("get_into_ranges returned invalid buffer result")
        buffers: list[tuple[tuple[int, ...], ...]] = []
        for buffer_result in cast(Sequence[object], value):
            buffers.append(cls._range_buffer_result(buffer_result))
        return tuple(buffers)

    @staticmethod
    def _range_buffer_result(value: object) -> tuple[tuple[int, ...], ...]:
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(
            value, Sequence
        ):
            raise WeightStoreError("get_into_ranges returned invalid object result")
        groups: list[tuple[int, ...]] = []
        for group in cast(Sequence[object], value):
            groups.append(StoreBackend._int_sequence(group, "get_into_ranges"))
        return tuple(groups)


__all__ = [
    "RangeResults",
    "StoreBackend",
    "StoreConfigFactory",
    "StoreRecordType",
    "default_config_factory",
]
