from __future__ import annotations

import pickle

import pytest

from weight_gpu_e2e.lifetime import allocation_guards_for_bindings

from mooncake.reshard.weight import (
    MemoryRegistrationLease,
    MooncakeTransferEngineReader,
)
from weight_all_axes_e2e.buffers import (
    HostBuffer,
    HostTransferEngine,
    _registered_engine_buffers,
    _wire_safe_reader_payload,
)
from weight_all_axes_e2e.builders import (
    _build_cross_dim_reader_fixture,
    _build_fixture,
    _build_packed_reader_fixture,
)
from weight_all_axes_e2e.execution import (
    _execute_cross_dim_reader_fixture,
    _execute_cross_dim_sink_fixture,
    _execute_reader_fixture,
    _execute_te_fixture,
)


@pytest.mark.parametrize(("source_tp", "target_tp"), ((4, 8), (8, 4)))
def test_all_axes_fixture_executes_one_composite_plan_without_native_runtime(
    source_tp: int,
    target_tp: int,
) -> None:
    engine = HostTransferEngine()
    fixture = _build_fixture(
        revision="unit-test",
        source_tp=source_tp,
        target_tp=target_tp,
        allocate_source=HostBuffer,
        allocate_target=HostBuffer,
        target_endpoint="target:12345",
    )
    with (
        _registered_engine_buffers(engine, fixture.source_buffers.values()),
        _registered_engine_buffers(engine, fixture.target_buffers.values()),
    ):
        _execute_te_fixture(engine, fixture)


def test_reader_fixture_pulls_packed_tp1_weights_into_tp2_targets() -> None:
    engine = HostTransferEngine()
    fixture = _build_packed_reader_fixture(
        revision="unit-test",
        source_endpoint="reader-source:12345",
        target_endpoint="reader-target:12345",
        allocate_source=HostBuffer,
        allocate_target=HostBuffer,
    )

    def execute_reader(plan, sources, target):
        source_owner_bindings = sources.bindings
        target_owner_bindings = target.bindings
        plan, sources, target = _wire_safe_reader_payload(plan, sources, target)
        plan, sources, target = pickle.loads(pickle.dumps((plan, sources, target)))
        target_placement, target_binding = target.single()
        return MooncakeTransferEngineReader(engine).execute(
            plan,
            sources.placement,
            sources.bindings,
            target_placement,
            target_binding,
            source_registrations=tuple(
                MemoryRegistrationLease.from_fragment(
                    fragment,
                    lease_generation=binding.generation,
                    runtime_lease_id=binding.lease_id,
                )
                for binding in sources.bindings
                for fragment in binding.fragments
            ),
            target_pre_registered=True,
            target_registrations=tuple(
                MemoryRegistrationLease.from_fragment(
                    fragment,
                    lease_generation=target_binding.generation,
                    runtime_lease_id=target_binding.lease_id,
                )
                for fragment in target_binding.fragments
            ),
            source_allocation_guards=allocation_guards_for_bindings(
                sources.bindings,
                owner_bindings=source_owner_bindings,
            ),
            target_allocation_guards=allocation_guards_for_bindings(
                (target_binding,),
                owner_bindings=target_owner_bindings,
            ),
        )

    with (
        _registered_engine_buffers(engine, (fixture.source_buffer,)),
        _registered_engine_buffers(engine, fixture.target_buffers),
    ):
        _execute_reader_fixture(fixture, execute_reader)


def test_reader_fixture_reshards_independent_experts_across_dimensions() -> None:
    engine = HostTransferEngine()
    fixture = _build_cross_dim_reader_fixture(
        revision="unit-test",
        source_endpoint="cross-dim-source:12345",
        target_endpoint="cross-dim-target:12345",
        allocate_source=HostBuffer,
        allocate_target=HostBuffer,
    )

    def execute_reader(plan, sources, target):
        target_placement, target_binding = target.single()
        return MooncakeTransferEngineReader(engine).execute(
            plan,
            sources.placement,
            sources.bindings,
            target_placement,
            target_binding,
            source_registrations=tuple(
                MemoryRegistrationLease.from_fragment(
                    fragment,
                    lease_generation=binding.generation,
                    runtime_lease_id=binding.lease_id,
                )
                for binding in sources.bindings
                for fragment in binding.fragments
            ),
            target_pre_registered=True,
            target_registrations=tuple(
                MemoryRegistrationLease.from_fragment(
                    fragment,
                    lease_generation=target_binding.generation,
                    runtime_lease_id=target_binding.lease_id,
                )
                for fragment in target_binding.fragments
            ),
            source_allocation_guards=allocation_guards_for_bindings(sources.bindings),
            target_allocation_guards=allocation_guards_for_bindings((target_binding,)),
        )

    with (
        _registered_engine_buffers(engine, fixture.source_buffers),
        _registered_engine_buffers(engine, fixture.target_buffers),
    ):
        _execute_cross_dim_reader_fixture(fixture, execute_reader)


def test_sink_fixture_reshards_independent_experts_across_dimensions() -> None:
    engine = HostTransferEngine()
    fixture = _build_cross_dim_reader_fixture(
        revision="unit-test",
        source_endpoint="cross-dim-source:12345",
        target_endpoint="cross-dim-target:12345",
        allocate_source=HostBuffer,
        allocate_target=HostBuffer,
    )

    with (
        _registered_engine_buffers(engine, fixture.source_buffers),
        _registered_engine_buffers(engine, fixture.target_buffers),
    ):
        _execute_cross_dim_sink_fixture(fixture, engine)
