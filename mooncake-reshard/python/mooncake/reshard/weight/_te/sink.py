from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ...transfer_engine import (
    MooncakeTransferEngineExecutor,
    TransferBatch,
    TransferDirection,
    TransferEngineError,
)
from ..manifest import (
    RuntimeBindingFragment,
    WeightPlacementManifest,
    WeightRuntimeBindingManifest,
)
from ..planner import TransferOperation, TransferPlan
from .execution import (
    pair_manifests,
    resolve_runtime_executors,
    runtime_binding_fragment,
    validate_lowering_limits,
    validate_manifest_pair,
)
from .registration import (
    MemoryRegistrationLease,
    registered_sources,
    registration_map,
    same_runtime_snapshot,
    validate_registration,
)


@dataclass(frozen=True)
class DirectTransferReceipt:
    source_worker_id: str
    target_endpoint: str
    operation_count: int
    nbytes: int


class MooncakeTransferEngineSink:
    def __init__(
        self,
        engine: Any,
        *,
        max_batch_operations: int = 1024,
        max_region_segments: int = 1_000_000,
        max_completion_drain_attempts: int = 3,
        completion_drain_timeout_ms: int = 1000,
    ) -> None:
        validate_lowering_limits(
            max_batch_operations=max_batch_operations,
            max_region_segments=max_region_segments,
            max_completion_drain_attempts=max_completion_drain_attempts,
            completion_drain_timeout_ms=completion_drain_timeout_ms,
        )
        self.engine = engine
        self.transfer_executor = MooncakeTransferEngineExecutor(
            engine,
            max_completion_drain_attempts=max_completion_drain_attempts,
            completion_drain_timeout_ms=completion_drain_timeout_ms,
        )
        self._pending = self.transfer_executor.pending_manager
        self._engine_identity = self._pending.engine_identity
        self.max_batch_operations = max_batch_operations
        self.max_region_segments = max_region_segments
        self.max_completion_drain_attempts = max_completion_drain_attempts
        self.completion_drain_timeout_ms = completion_drain_timeout_ms

    def execute(
        self,
        plan: TransferPlan,
        source_placement: WeightPlacementManifest,
        source_binding: WeightRuntimeBindingManifest,
        target_placement: WeightPlacementManifest,
        target_bindings: Sequence[WeightRuntimeBindingManifest],
        *,
        target_registrations: Sequence[MemoryRegistrationLease] | None = None,
        source_pre_registered: bool = False,
        source_registrations: Sequence[MemoryRegistrationLease] | None = None,
    ) -> tuple[DirectTransferReceipt, ...]:
        with self.transfer_executor.submission():
            return self._execute_reserved(
                plan,
                source_placement,
                source_binding,
                target_placement,
                target_bindings,
                target_registrations=target_registrations,
                source_pre_registered=source_pre_registered,
                source_registrations=source_registrations,
            )

    def _execute_reserved(
        self,
        plan: TransferPlan,
        source_placement: WeightPlacementManifest,
        source_binding: WeightRuntimeBindingManifest,
        target_placement: WeightPlacementManifest,
        target_bindings: Sequence[WeightRuntimeBindingManifest],
        *,
        target_registrations: Sequence[MemoryRegistrationLease] | None = None,
        source_pre_registered: bool = False,
        source_registrations: Sequence[MemoryRegistrationLease] | None = None,
    ) -> tuple[DirectTransferReceipt, ...]:
        validate_manifest_pair(plan, source_placement, source_binding, "source")
        try:
            source_executors = resolve_runtime_executors(
                plan, source_placement, source_binding, "source"
            )
        except ValueError as error:
            raise TransferEngineError(str(error)) from error
        source_workers = {executor.worker_id for executor in source_executors}
        if len(source_workers) != 1:
            raise TransferEngineError("source manifest spans multiple workers")
        source_worker_id = next(iter(source_workers))

        targets: dict[str, RuntimeBindingFragment] = {}
        target_runtime_lease_ids: dict[str, str] = {}
        target_generations: dict[str, int] = {}
        target_ranks = set()
        expected_target_participants = {
            executor.participant_id for executor in plan.target_executors
        }
        target_pairs = pair_manifests(target_placement, target_bindings, "target")
        for placement, binding in target_pairs:
            validate_manifest_pair(plan, placement, binding, "target")
            if binding.participant_id not in expected_target_participants:
                continue
            try:
                executors = resolve_runtime_executors(
                    plan, placement, binding, "target"
                )
            except ValueError as error:
                raise TransferEngineError(str(error)) from error
            for executor in executors:
                if executor.rank in target_ranks:
                    raise TransferEngineError(
                        f"duplicate target executor rank: {executor.rank}"
                    )
                target_ranks.add(executor.rank)
            for fragment in binding.fragments:
                if fragment.fragment_id in targets:
                    raise TransferEngineError(
                        f"duplicate target fragment: {fragment.fragment_id}"
                    )
                targets[fragment.fragment_id] = fragment
                target_runtime_lease_ids[fragment.fragment_id] = binding.lease_id
                target_generations[fragment.fragment_id] = binding.generation
        expected_target_ranks = {executor.rank for executor in plan.target_executors}
        if target_ranks != expected_target_ranks:
            raise TransferEngineError("target executor set is incomplete")

        local_operations = [
            plan.operations[index]
            for executor in source_executors
            for index in executor.operation_indices
        ]
        if not local_operations:
            return ()

        local = {
            fragment.fragment_id: fragment for fragment in source_binding.fragments
        }
        target_registration_by_id = registration_map(
            target_registrations,
            "target",
        )

        operations_by_endpoint: dict[
            str,
            list[
                tuple[
                    TransferOperation,
                    RuntimeBindingFragment,
                    RuntimeBindingFragment,
                ]
            ],
        ] = {}
        used_sources: dict[str, RuntimeBindingFragment] = {}
        for operation in local_operations:
            planned_source = runtime_binding_fragment(operation.source)
            planned_target = runtime_binding_fragment(operation.target)
            current = local.get(planned_source.fragment_id)
            if current is None or not same_runtime_snapshot(current, planned_source):
                raise TransferEngineError(
                    f"stale source fragment: {planned_source.fragment_id}"
                )
            target = targets.get(planned_target.fragment_id)
            if target is None:
                raise TransferEngineError(
                    f"missing planned target fragment: {planned_target.fragment_id}"
                )
            if not same_runtime_snapshot(target, planned_target):
                raise TransferEngineError(
                    f"stale target fragment: {planned_target.fragment_id}"
                )
            if not target.endpoint:
                raise TransferEngineError(
                    f"target endpoint is empty: {operation.target.fragment_id}"
                )
            try:
                operation.validate_bounds()
            except ValueError as error:
                raise TransferEngineError(
                    f"invalid copy range for {operation.tensor_id}: {error}"
                ) from error
            if operation.repeat > self.max_region_segments:
                raise TransferEngineError(
                    f"transfer region exceeds max_region_segments: "
                    f"{operation.tensor_id}: {operation.repeat} > "
                    f"{self.max_region_segments}"
                )
            validate_registration(
                target,
                target_registration_by_id,
                "target",
                lease_generation=target_generations[target.fragment_id],
                runtime_lease_id=target_runtime_lease_ids[target.fragment_id],
            )
            used_sources[current.fragment_id] = current
            operations_by_endpoint.setdefault(target.endpoint, []).append(
                (operation, current, target)
            )

        with registered_sources(
            self.engine,
            self,
            tuple(used_sources.values()),
            pre_registered=source_pre_registered,
            registrations=source_registrations,
            lease_generation=source_binding.generation,
            runtime_lease_id=source_binding.lease_id,
            resources=(
                source_placement,
                source_binding,
                target_placement,
                tuple(target_bindings),
                source_registrations,
                target_registrations,
            ),
        ):
            receipts = []
            for endpoint in sorted(operations_by_endpoint):
                operations = sorted(
                    operations_by_endpoint[endpoint],
                    key=lambda item: (
                        item[2].address + item[0].target_offset,
                        item[1].address + item[0].source_offset,
                    ),
                )
                source_addresses = []
                target_addresses = []
                sizes = []
                operation_count = 0
                total_bytes = 0
                for operation, source, target in operations:
                    for (
                        source_offset,
                        target_offset,
                        nbytes,
                    ) in operation.iter_segments():
                        source_addresses.append(source.address + source_offset)
                        target_addresses.append(target.address + target_offset)
                        sizes.append(nbytes)
                        operation_count += 1
                        total_bytes += nbytes
                        if len(sizes) == self.max_batch_operations:
                            self._transfer_batch(
                                endpoint,
                                source_addresses,
                                target_addresses,
                                sizes,
                            )
                            source_addresses = []
                            target_addresses = []
                            sizes = []
                if sizes:
                    self._transfer_batch(
                        endpoint,
                        source_addresses,
                        target_addresses,
                        sizes,
                    )
                receipts.append(
                    DirectTransferReceipt(
                        source_worker_id=source_worker_id,
                        target_endpoint=endpoint,
                        operation_count=operation_count,
                        nbytes=total_bytes,
                    )
                )
        return tuple(receipts)

    def _transfer_batch(
        self,
        endpoint: str,
        source_addresses: list[int],
        target_addresses: list[int],
        sizes: list[int],
    ) -> None:
        self.transfer_executor._execute_reserved_batch(
            TransferBatch(
                endpoint=endpoint,
                source_addresses=tuple(source_addresses),
                target_addresses=tuple(target_addresses),
                sizes=tuple(sizes),
            ),
            TransferDirection.WRITE,
        )

    def _retain_pending_ticket(self, ticket: Any) -> str:
        return self._pending._retain_pending_ticket(ticket)

    def _retain_pending_resources(
        self,
        pending_transfer_id: str,
        *,
        registrations: Sequence[int],
        resources: Sequence[Any],
    ) -> None:
        self._pending._retain_pending_resources(
            pending_transfer_id,
            registrations=registrations,
            resources=resources,
        )

    def _reserve_submission(self) -> None:
        self._pending._reserve_submission()

    def _release_submission(self) -> None:
        self._pending._release_submission()

    def pending_transfer_ids(self) -> tuple[str, ...]:
        return self._pending.pending_transfer_ids()

    def pending_transfer_status(self, pending_transfer_id: str) -> str:
        return self._pending.pending_transfer_status(pending_transfer_id)

    def drain_pending_transfer(
        self,
        pending_transfer_id: str,
        *,
        timeout_ms: int = 1000,
    ) -> str:
        return self._pending.drain_pending_transfer(
            pending_transfer_id,
            timeout_ms=timeout_ms,
        )
