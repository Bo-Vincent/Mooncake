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
    registered_targets,
    registration_map,
    same_runtime_snapshot,
    validate_registration,
)


@dataclass(frozen=True)
class DirectReadReceipt:
    source_endpoint: str
    target_worker_id: str
    operation_count: int
    nbytes: int


class MooncakeTransferEngineReader:
    """Execute a local target plan with target-initiated zero-copy reads."""

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
        self.max_batch_operations = max_batch_operations
        self.max_region_segments = max_region_segments
        self.max_completion_drain_attempts = max_completion_drain_attempts
        self.completion_drain_timeout_ms = completion_drain_timeout_ms
        self.transfer_executor = MooncakeTransferEngineExecutor(
            engine,
            max_completion_drain_attempts=max_completion_drain_attempts,
            completion_drain_timeout_ms=completion_drain_timeout_ms,
        )
        self._pending = self.transfer_executor.pending_manager

    def execute(
        self,
        plan: TransferPlan,
        source_placement: WeightPlacementManifest,
        source_bindings: Sequence[WeightRuntimeBindingManifest],
        target_placement: WeightPlacementManifest,
        target_binding: WeightRuntimeBindingManifest,
        *,
        source_pre_registered: bool = True,
        source_registrations: Sequence[MemoryRegistrationLease] | None = None,
        target_pre_registered: bool = False,
        target_registrations: Sequence[MemoryRegistrationLease] | None = None,
    ) -> tuple[DirectReadReceipt, ...]:
        with self.transfer_executor.submission():
            return self._execute_reserved(
                plan,
                source_placement,
                source_bindings,
                target_placement,
                target_binding,
                source_pre_registered=source_pre_registered,
                source_registrations=source_registrations,
                target_pre_registered=target_pre_registered,
                target_registrations=target_registrations,
            )

    def _execute_reserved(
        self,
        plan: TransferPlan,
        source_placement: WeightPlacementManifest,
        source_bindings: Sequence[WeightRuntimeBindingManifest],
        target_placement: WeightPlacementManifest,
        target_binding: WeightRuntimeBindingManifest,
        *,
        source_pre_registered: bool = True,
        source_registrations: Sequence[MemoryRegistrationLease] | None = None,
        target_pre_registered: bool = False,
        target_registrations: Sequence[MemoryRegistrationLease] | None = None,
    ) -> tuple[DirectReadReceipt, ...]:
        if not source_pre_registered:
            raise TransferEngineError("remote source memory must be pre-registered")
        source_registration_by_id = registration_map(source_registrations, "source")
        if target_registrations is not None and not target_pre_registered:
            raise TransferEngineError(
                "target_registrations require target_pre_registered=True"
            )
        validate_manifest_pair(plan, target_placement, target_binding, "target")
        try:
            target_executors = resolve_runtime_executors(
                plan, target_placement, target_binding, "target"
            )
        except ValueError as error:
            raise TransferEngineError(str(error)) from error
        if len(target_executors) != 1:
            raise TransferEngineError(
                "target manifest must describe one local executor"
            )
        target_executor = target_executors[0]

        sources: dict[str, RuntimeBindingFragment] = {}
        source_runtime_lease_ids: dict[str, str] = {}
        source_generations: dict[str, int] = {}
        source_ranks = set()
        expected_source_participants = {
            executor.participant_id for executor in plan.source_executors
        }
        source_pairs = pair_manifests(source_placement, source_bindings, "source")
        for placement, binding in source_pairs:
            validate_manifest_pair(plan, placement, binding, "source")
            if binding.participant_id not in expected_source_participants:
                continue
            try:
                executors = resolve_runtime_executors(
                    plan, placement, binding, "source"
                )
            except ValueError as error:
                raise TransferEngineError(str(error)) from error
            for executor in executors:
                if executor.rank in source_ranks:
                    raise TransferEngineError(
                        f"duplicate source executor rank: {executor.rank}"
                    )
                source_ranks.add(executor.rank)
            for fragment in binding.fragments:
                if fragment.fragment_id in sources:
                    raise TransferEngineError(
                        f"duplicate source fragment: {fragment.fragment_id}"
                    )
                sources[fragment.fragment_id] = fragment
                source_runtime_lease_ids[fragment.fragment_id] = binding.lease_id
                source_generations[fragment.fragment_id] = binding.generation
        expected_source_ranks = {executor.rank for executor in plan.source_executors}
        if source_ranks != expected_source_ranks:
            raise TransferEngineError("source executor set is incomplete")

        targets = {
            fragment.fragment_id: fragment for fragment in target_binding.fragments
        }
        registration_by_id = (
            registration_map(target_registrations, "target")
            if target_pre_registered
            else {}
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
        used_targets: dict[str, RuntimeBindingFragment] = {}
        for index in target_executor.operation_indices:
            operation = plan.operations[index]
            planned_source = runtime_binding_fragment(operation.source)
            planned_target = runtime_binding_fragment(operation.target)
            source = sources.get(planned_source.fragment_id)
            target = targets.get(planned_target.fragment_id)
            if source is None or not same_runtime_snapshot(source, planned_source):
                raise TransferEngineError(
                    f"stale source fragment: {planned_source.fragment_id}"
                )
            runtime_lease_id = source_runtime_lease_ids[source.fragment_id]
            validate_registration(
                source,
                source_registration_by_id,
                "source",
                lease_generation=source_generations[source.fragment_id],
                runtime_lease_id=runtime_lease_id,
            )
            if target is None or not same_runtime_snapshot(target, planned_target):
                raise TransferEngineError(
                    f"stale target fragment: {planned_target.fragment_id}"
                )
            if target_pre_registered:
                validate_registration(
                    target,
                    registration_by_id,
                    "target",
                    lease_generation=target_binding.generation,
                    runtime_lease_id=target_binding.lease_id,
                )
            operation.validate_bounds()
            if operation.repeat > self.max_region_segments:
                raise TransferEngineError(
                    f"transfer region exceeds max_region_segments: "
                    f"{operation.tensor_id}: {operation.repeat} > "
                    f"{self.max_region_segments}"
                )
            used_targets[target.fragment_id] = target
            operations_by_endpoint.setdefault(source.endpoint, []).append(
                (operation, source, target)
            )

        with registered_targets(
            self.engine,
            self._pending,
            tuple(used_targets.values()),
            pre_registered=target_pre_registered,
            resources=(
                source_placement,
                tuple(source_bindings),
                target_placement,
                target_binding,
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
                target_addresses = []
                source_addresses = []
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
                                target_addresses,
                                source_addresses,
                                sizes,
                            )
                            target_addresses = []
                            source_addresses = []
                            sizes = []
                if sizes:
                    self._transfer_batch(
                        endpoint,
                        target_addresses,
                        source_addresses,
                        sizes,
                    )
                receipts.append(
                    DirectReadReceipt(
                        source_endpoint=endpoint,
                        target_worker_id=target_executor.worker_id,
                        operation_count=operation_count,
                        nbytes=total_bytes,
                    )
                )
        return tuple(receipts)

    def _transfer_batch(
        self,
        endpoint: str,
        target_addresses: list[int],
        source_addresses: list[int],
        sizes: list[int],
    ) -> None:
        self.transfer_executor._execute_reserved_batch(
            TransferBatch(
                endpoint=endpoint,
                source_addresses=tuple(source_addresses),
                target_addresses=tuple(target_addresses),
                sizes=tuple(sizes),
            ),
            TransferDirection.READ,
        )

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
