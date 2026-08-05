from __future__ import annotations

from typing import Callable

from mooncake.reshard.weight import (
    DirectReadReceipt,
    MemoryRegistrationLease,
    MooncakeTransferEngineSink,
    TransferPlan,
    bind_logical_transfer_plan,
    plan_placement_transfer,
    plan_placement_transfer_to_local_target,
)
from weight_all_axes_e2e.fixtures import (
    AllAxesFixture,
    CrossDimReaderFixture,
    PackedReaderFixture,
    RuntimeInputs,
)
from weight_all_axes_e2e.models import _CROSS_DIM_SHAPE, _TENSOR_BYTES


ReaderExecutor = Callable[
    [TransferPlan, RuntimeInputs, RuntimeInputs],
    tuple[DirectReadReceipt, ...],
]


def _single_runtime_inputs(placement, binding) -> RuntimeInputs:
    return RuntimeInputs(placement, (binding,))


def _bound_part(placement, binding):
    return next(
        part
        for part in placement.parts
        if part.participant_id == binding.participant_id
    )


def _execute_reader_fixture(
    fixture: PackedReaderFixture,
    execute_reader: ReaderExecutor,
) -> None:
    total_bytes = 0
    for target_placement, target_binding in fixture.targets.active_pairs():
        logical = plan_placement_transfer_to_local_target(
            fixture.source.placement,
            target_placement,
            target_participant_id=target_binding.participant_id,
        )
        assert all(
            not hasattr(operation.target, "address") for operation in logical.operations
        )
        plan = bind_logical_transfer_plan(
            logical,
            (target_binding,),
            source_bindings=fixture.source.bindings,
        )
        axis1 = next(
            operation
            for operation in plan.operations
            if operation.tensor_id == "layers.1.axis1.weight"
        )
        assert (axis1.nbytes, axis1.repeat) == (4, 4)
        assert (axis1.source_stride, axis1.target_stride) == (8, 4)

        target = _single_runtime_inputs(target_placement, target_binding)
        receipts = execute_reader(plan, fixture.source, target)
        assert len(receipts) == 1
        assert (
            receipts[0].source_endpoint
            == fixture.source.bindings[0].fragments[0].endpoint
        )
        assert receipts[0].target_worker_id == target_binding.fragments[0].worker_id
        assert receipts[0].operation_count == 6
        assert receipts[0].nbytes == 52
        total_bytes += receipts[0].nbytes

    assert total_bytes == 104
    fixture.verify()


def _execute_cross_dim_reader_fixture(
    fixture: CrossDimReaderFixture,
    execute_reader: ReaderExecutor,
) -> None:
    expected_segments = {
        "expert-family.axis0-to-axis1": 4,
        "expert-family.axis0-to-axis2": 16,
    }
    for target_placement, target_binding in fixture.targets.active_pairs():
        logical = plan_placement_transfer_to_local_target(
            fixture.sources.placement,
            target_placement,
            target_participant_id=target_binding.participant_id,
        )
        assert all(
            not hasattr(operation.target, "address") for operation in logical.operations
        )
        plan = bind_logical_transfer_plan(
            logical,
            (target_binding,),
            source_bindings=fixture.sources.bindings,
        )
        target_fragment = _bound_part(target_placement, target_binding).fragments[0]
        assert {
            (route.source_pp, route.target_pp) for route in plan.pipeline_routes
        } == {(0, target_fragment.rank.pp)}
        assert len(plan.operations) == _CROSS_DIM_SHAPE[0]
        assert sum(operation.total_bytes for operation in plan.operations) == (
            target_fragment.nbytes
        )

        target = _single_runtime_inputs(target_placement, target_binding)
        receipts = execute_reader(plan, fixture.sources, target)
        assert len(receipts) == 1
        assert (
            receipts[0].source_endpoint
            == fixture.sources.bindings[0].fragments[0].endpoint
        )
        assert receipts[0].target_worker_id == target_binding.fragments[0].worker_id
        assert (
            receipts[0].operation_count == expected_segments[target_fragment.tensor_id]
        )
        assert receipts[0].nbytes == target_fragment.nbytes

    fixture.verify()


def _execute_cross_dim_sink_fixture(
    fixture: CrossDimReaderFixture,
    source_engine,
) -> None:
    logical = plan_placement_transfer(
        fixture.sources.placement,
        fixture.targets.placement,
    )
    assert all(
        not hasattr(operation.target, "address") for operation in logical.operations
    )
    plan = bind_logical_transfer_plan(
        logical,
        fixture.targets.bindings,
        source_bindings=fixture.sources.bindings,
    )
    assert {(route.source_pp, route.target_pp) for route in plan.pipeline_routes} == {
        (0, 1),
        (0, 2),
    }
    source_registrations = tuple(
        MemoryRegistrationLease.from_fragment(
            fragment,
            lease_generation=binding.generation,
            runtime_lease_id=binding.lease_id,
        )
        for binding in fixture.sources.bindings
        for fragment in binding.fragments
    )
    target_registrations = tuple(
        MemoryRegistrationLease.from_fragment(
            fragment,
            lease_generation=binding.generation,
            runtime_lease_id=binding.lease_id,
        )
        for binding in fixture.targets.bindings
        for fragment in binding.fragments
    )
    sink = MooncakeTransferEngineSink(source_engine, max_batch_operations=5)
    receipts = tuple(
        receipt
        for source_placement, source_binding in fixture.sources.active_pairs()
        for receipt in sink.execute(
            plan,
            source_placement,
            source_binding,
            fixture.targets.placement,
            fixture.targets.bindings,
            target_registrations=target_registrations,
            source_pre_registered=True,
            source_registrations=source_registrations,
        )
    )

    assert sum(receipt.nbytes for receipt in receipts) == sum(
        fragment.nbytes for fragment in fixture.targets.placement.fragments
    )
    fixture.verify()


def _execute_te_fixture(
    source_engine,
    fixture: AllAxesFixture,
) -> None:
    logical = plan_placement_transfer(
        fixture.sources.placement,
        fixture.targets.placement,
    )
    assert all(
        not hasattr(operation.target, "address") for operation in logical.operations
    )
    plan = bind_logical_transfer_plan(
        logical,
        fixture.targets.bindings,
        source_bindings=fixture.sources.bindings,
    )
    assert {operation.source.rank.dp for operation in plan.operations} == {0, 1}
    assert {operation.target.rank.dp for operation in plan.operations} == {0, 1, 2}
    assert {operation.source.rank.pp for operation in plan.operations} == {0, 1}
    assert {operation.target.rank.pp for operation in plan.operations} == {0, 1, 2, 3}
    assert {operation.source.rank.ep for operation in plan.operations} == {0, 1}
    assert {operation.target.rank.ep for operation in plan.operations} == {0}

    sink = MooncakeTransferEngineSink(source_engine)
    source_registrations = tuple(
        MemoryRegistrationLease.from_fragment(
            fragment,
            lease_generation=binding.generation,
            runtime_lease_id=binding.lease_id,
        )
        for binding in fixture.sources.bindings
        for fragment in binding.fragments
    )
    target_registrations = tuple(
        MemoryRegistrationLease.from_fragment(
            fragment,
            lease_generation=binding.generation,
            runtime_lease_id=binding.lease_id,
        )
        for binding in fixture.targets.bindings
        for fragment in binding.fragments
    )
    receipts = tuple(
        receipt
        for source_placement, source_binding in fixture.sources.active_pairs()
        for receipt in sink.execute(
            plan,
            source_placement,
            source_binding,
            fixture.targets.placement,
            fixture.targets.bindings,
            target_registrations=target_registrations,
            source_pre_registered=True,
            source_registrations=source_registrations,
        )
    )
    assert sum(receipt.nbytes for receipt in receipts) == (
        len(fixture.weights) * _TENSOR_BYTES * 3
    )
    fixture.verify()
