from __future__ import annotations

from typing import Callable

from mooncake.reshard._compat import _strict_zip
from mooncake.reshard.weight import (
    ParallelRank,
    PlacementFragment,
    RuntimeBindingFragment,
    TensorDescriptor,
    OwnershipAxis,
    ReplicatedAxis,
    SplitAxis,
    WeightRuntimeBindingManifest,
)
from weight_all_axes_e2e.fixtures import (
    AllAxesFixture,
    CrossDimReaderFixture,
    PackedReaderFixture,
    RuntimeInputs,
)
from weight_all_axes_e2e.models import (
    _CROSS_DIM_SHAPE,
    _READER_BUFFER_BYTES,
    _READER_SOURCE_GUARD,
    _READER_TARGET_SENTINEL,
    _READER_VIEW_OFFSETS,
    _TENSOR_BYTES,
    _weights,
)
from weight_gpu_e2e.buffers import TransferBuffer
from global_placement_helpers import global_placement, runtime_fragment


def _runtime_inputs_from_groups(
    groups,
    *,
    resource_id: str,
    revision: str,
    placement_set_id: str,
    lease_prefix: str,
) -> RuntimeInputs:
    descriptors = {
        descriptor.tensor_id: descriptor
        for entries in groups.values()
        for descriptor, _, _ in entries
    }
    fragments = tuple(
        fragment for entries in groups.values() for _, fragment, _ in entries
    )
    ranks = tuple(ParallelRank(dp=dp, tp=tp, pp=pp, ep=ep) for dp, tp, pp, ep in groups)
    placement = global_placement(
        resource_id=resource_id,
        revision=revision,
        placement_set_id=placement_set_id,
        tensors=tuple(descriptors.values()),
        fragments=fragments,
        ranks=ranks,
    )

    entries_by_rank = {
        ParallelRank(dp=dp, tp=tp, pp=pp, ep=ep): entries
        for (dp, tp, pp, ep), entries in groups.items()
    }
    bindings = []
    for part in placement.parts:
        entries = entries_by_rank.get(part.rank, ())
        worker_id = entries[0][2].worker_id if entries else part.participant_id
        bindings.append(
            WeightRuntimeBindingManifest(
                resource_id=placement.resource_id,
                revision=placement.revision,
                placement_id=placement.placement_id,
                placement_digest=placement.digest,
                participant_id=part.participant_id,
                instance_id=worker_id,
                generation=1,
                lease_id=(
                    f"{lease_prefix}-lease-d{part.rank.dp}-t{part.rank.tp}-"
                    f"p{part.rank.pp}-e{part.rank.ep}"
                ),
                fragments=tuple(fragment for _, _, fragment in entries),
            )
        )
    return RuntimeInputs(placement, tuple(bindings))


def _build_fixture(
    *,
    revision: str,
    source_tp: int,
    target_tp: int,
    allocate_source: Callable[[int], TransferBuffer],
    allocate_target: Callable[[int], TransferBuffer],
    target_endpoint: str,
) -> AllAxesFixture:
    source_dp = 2
    target_dp = 3
    weights = _weights()
    source_fragments: dict[
        tuple[int, int, int, int],
        list[tuple[TensorDescriptor, PlacementFragment, RuntimeBindingFragment]],
    ] = {}
    target_fragments: dict[
        tuple[int, int, int, int],
        list[tuple[TensorDescriptor, PlacementFragment, RuntimeBindingFragment]],
    ] = {}
    source_buffers = {}
    target_buffers = {}

    source_extent = _TENSOR_BYTES // source_tp
    target_extent = _TENSOR_BYTES // target_tp
    for weight in weights:
        descriptor = weight.descriptor()
        for dp_rank in range(source_dp):
            for tp_rank in range(source_tp):
                buffer = allocate_source(source_extent)
                buffer.fill(weight.pattern + tp_rank)
                source_buffers[(weight.tensor_id, dp_rank, tp_rank)] = buffer
                rank = ParallelRank(
                    dp=dp_rank,
                    tp=tp_rank,
                    pp=weight.source_pp,
                    ep=weight.source_ep,
                )
                worker_id = (
                    f"source-d{dp_rank}-t{tp_rank}-"
                    f"p{weight.source_pp}-e{weight.source_ep}"
                )
                placement_fragment = PlacementFragment(
                    tensor_id=weight.tensor_id,
                    global_offset=(tp_rank * source_extent,),
                    local_shape=(source_extent,),
                    nbytes=source_extent,
                    rank=rank,
                )
                source_fragments.setdefault(
                    (dp_rank, tp_rank, weight.source_pp, weight.source_ep), []
                ).append(
                    (
                        descriptor,
                        placement_fragment,
                        runtime_fragment(
                            placement=placement_fragment,
                            tensor=descriptor,
                            fragment_id=f"{worker_id}:{weight.tensor_id}",
                            address=buffer.pointer,
                            worker_id=worker_id,
                            endpoint=f"{worker_id}:12345",
                            owner=buffer,
                        ),
                    )
                )

        for dp_rank in range(target_dp):
            for tp_rank in range(target_tp):
                buffer = allocate_target(target_extent)
                buffer.zero()
                target_buffers[(weight.tensor_id, dp_rank, tp_rank)] = buffer
                rank = ParallelRank(
                    dp=dp_rank,
                    tp=tp_rank,
                    pp=weight.target_pp,
                    ep=weight.target_ep,
                )
                worker_id = (
                    f"target-d{dp_rank}-t{tp_rank}-"
                    f"p{weight.target_pp}-e{weight.target_ep}"
                )
                placement_fragment = PlacementFragment(
                    tensor_id=weight.tensor_id,
                    global_offset=(tp_rank * target_extent,),
                    local_shape=(target_extent,),
                    nbytes=target_extent,
                    rank=rank,
                )
                target_fragments.setdefault(
                    (dp_rank, tp_rank, weight.target_pp, weight.target_ep), []
                ).append(
                    (
                        descriptor,
                        placement_fragment,
                        runtime_fragment(
                            placement=placement_fragment,
                            tensor=descriptor,
                            fragment_id=f"{worker_id}:{weight.tensor_id}",
                            address=buffer.pointer,
                            worker_id=worker_id,
                            endpoint=target_endpoint,
                            owner=buffer,
                        ),
                    )
                )

    return AllAxesFixture(
        sources=_runtime_inputs_from_groups(
            source_fragments,
            resource_id="all-axes-native-e2e",
            revision=revision,
            placement_set_id="all-axes-source",
            lease_prefix="source",
        ),
        targets=_runtime_inputs_from_groups(
            target_fragments,
            resource_id="all-axes-native-e2e",
            revision=revision,
            placement_set_id="all-axes-target",
            lease_prefix="target",
        ),
        source_buffers=source_buffers,
        target_buffers=target_buffers,
        weights=weights,
        source_tp=source_tp,
        target_tp=target_tp,
    )


def _build_packed_reader_fixture(
    *,
    revision: str,
    source_endpoint: str,
    target_endpoint: str,
    allocate_source: Callable[[int], TransferBuffer],
    allocate_target: Callable[[int], TransferBuffer],
) -> PackedReaderFixture:
    descriptors = (
        TensorDescriptor(
            tensor_id="layers.0.axis0.weight",
            global_shape=(8, 5),
            dtype="uint8",
            itemsize=1,
            shard_dims=(0,),
            layer_id=0,
            layout_fingerprint="native-reader:packed:uint8:v1",
            parallel_axes=(SplitAxis("tp", dim=0),),
        ),
        TensorDescriptor(
            tensor_id="layers.1.axis1.weight",
            global_shape=(4, 8),
            dtype="uint8",
            itemsize=1,
            shard_dims=(1,),
            layer_id=1,
            layout_fingerprint="native-reader:packed:uint8:v1",
            parallel_axes=(SplitAxis("tp", dim=1),),
        ),
        TensorDescriptor(
            tensor_id="model.norm.weight",
            global_shape=(16,),
            dtype="uint8",
            itemsize=1,
            shard_dims=(),
            layout_fingerprint="native-reader:packed:uint8:v1",
            parallel_axes=(ReplicatedAxis("tp"),),
        ),
    )
    payloads = (
        bytes(range(1, 41)),
        bytes(range(65, 97)),
        bytes(range(129, 145)),
    )

    source_buffer = allocate_source(_READER_BUFFER_BYTES)
    source_expected = bytearray([_READER_SOURCE_GUARD] * _READER_BUFFER_BYTES)
    source_placement_fragments = []
    source_binding_fragments = []
    source_worker = "reader-source-t0"
    for descriptor, payload, view_offset in _strict_zip(
        descriptors, payloads, _READER_VIEW_OFFSETS
    ):
        source_expected[view_offset : view_offset + len(payload)] = payload
        placement_fragment = PlacementFragment(
            tensor_id=descriptor.tensor_id,
            global_offset=(0,) * len(descriptor.global_shape),
            local_shape=descriptor.global_shape,
            nbytes=len(payload),
            rank=ParallelRank(tp=0),
        )
        source_placement_fragments.append(placement_fragment)
        source_binding_fragments.append(
            runtime_fragment(
                placement=placement_fragment,
                tensor=descriptor,
                fragment_id=f"{source_worker}:{descriptor.tensor_id}",
                address=source_buffer.pointer + view_offset,
                worker_id=source_worker,
                endpoint=source_endpoint,
                storage_address=source_buffer.pointer,
                storage_nbytes=source_buffer.size,
                owner=source_buffer,
            )
        )
    source_buffer.write(bytes(source_expected))
    source = _runtime_inputs_from_groups(
        {
            (0, 0, 0, 0): list(
                _strict_zip(
                    descriptors,
                    source_placement_fragments,
                    source_binding_fragments,
                )
            )
        },
        resource_id="native-reader-e2e",
        revision=revision,
        placement_set_id="native-reader-source",
        lease_prefix="reader-source",
    )

    target_groups = {}
    target_buffers = []
    target_expected = []
    for tp_rank in range(2):
        target_buffer = allocate_target(_READER_BUFFER_BYTES)
        expected = bytearray([_READER_TARGET_SENTINEL] * _READER_BUFFER_BYTES)
        target_buffer.write(bytes(expected))
        target_worker = f"reader-target-t{tp_rank}"
        target_placement_fragments = []
        target_binding_fragments = []
        for descriptor, payload, view_offset in _strict_zip(
            descriptors, payloads, _READER_VIEW_OFFSETS
        ):
            if descriptor.shard_dims == (0,):
                local_shape = (4, 5)
                global_offset = (tp_rank * 4, 0)
                local_payload = payload[tp_rank * 20 : (tp_rank + 1) * 20]
            elif descriptor.shard_dims == (1,):
                local_shape = (4, 4)
                global_offset = (0, tp_rank * 4)
                local_payload = b"".join(
                    payload[row * 8 + tp_rank * 4 : row * 8 + (tp_rank + 1) * 4]
                    for row in range(4)
                )
            else:
                local_shape = descriptor.global_shape
                global_offset = (0,) * len(descriptor.global_shape)
                local_payload = payload
            expected[view_offset : view_offset + len(local_payload)] = local_payload
            placement_fragment = PlacementFragment(
                tensor_id=descriptor.tensor_id,
                global_offset=global_offset,
                local_shape=local_shape,
                nbytes=len(local_payload),
                rank=ParallelRank(tp=tp_rank),
            )
            target_placement_fragments.append(placement_fragment)
            target_binding_fragments.append(
                runtime_fragment(
                    placement=placement_fragment,
                    tensor=descriptor,
                    fragment_id=f"{target_worker}:{descriptor.tensor_id}",
                    address=target_buffer.pointer + view_offset,
                    worker_id=target_worker,
                    endpoint=target_endpoint,
                    storage_address=target_buffer.pointer,
                    storage_nbytes=target_buffer.size,
                    owner=target_buffer,
                )
            )
        target_groups[(0, tp_rank, 0, 0)] = list(
            _strict_zip(
                descriptors,
                target_placement_fragments,
                target_binding_fragments,
            )
        )
        target_buffers.append(target_buffer)
        target_expected.append(bytes(expected))

    return PackedReaderFixture(
        source=source,
        targets=_runtime_inputs_from_groups(
            target_groups,
            resource_id="native-reader-e2e",
            revision=revision,
            placement_set_id="native-reader-target",
            lease_prefix="reader-target",
        ),
        source_buffer=source_buffer,
        target_buffers=tuple(target_buffers),
        source_expected=bytes(source_expected),
        target_expected=tuple(target_expected),
    )


def _build_cross_dim_reader_fixture(
    *,
    revision: str,
    source_endpoint: str,
    target_endpoint: str,
    allocate_source: Callable[[int], TransferBuffer],
    allocate_target: Callable[[int], TransferBuffer],
) -> CrossDimReaderFixture:
    tensor_specs = (
        ("expert-family.axis0-to-axis1", 1, 1, 2),
        ("expert-family.axis0-to-axis2", 129, 2, 2),
    )
    source_descriptors = tuple(
        TensorDescriptor(
            tensor_id=tensor_id,
            global_shape=_CROSS_DIM_SHAPE,
            dtype="uint8",
            itemsize=1,
            shard_dims=(0,),
            layer_id=target_pp,
            layout_fingerprint="native-reader:cross-dim:uint8:v2",
            parallel_axes=(
                OwnershipAxis("pp"),
                SplitAxis("ep", dim=0),
            ),
        )
        for tensor_id, _base, target_pp, _target_shards in tensor_specs
    )
    descriptor_by_id = {
        descriptor.tensor_id: descriptor for descriptor in source_descriptors
    }

    source_buffers = []
    source_groups = {}
    expert_extent = _CROSS_DIM_SHAPE[1] * _CROSS_DIM_SHAPE[2]
    for expert_id in range(_CROSS_DIM_SHAPE[0]):
        worker_id = f"cross-dim-source-e{expert_id}"
        placement_fragments = []
        binding_fragments = []
        for tensor_id, base, _target_pp, _target_shards in tensor_specs:
            payload = bytes(
                base
                + expert_id * expert_extent
                + out_index * _CROSS_DIM_SHAPE[2]
                + in_index
                for out_index in range(_CROSS_DIM_SHAPE[1])
                for in_index in range(_CROSS_DIM_SHAPE[2])
            )
            buffer = allocate_source(len(payload))
            buffer.write(payload)
            source_buffers.append(buffer)
            placement_fragment = PlacementFragment(
                tensor_id=tensor_id,
                global_offset=(expert_id, 0, 0),
                local_shape=(1, *_CROSS_DIM_SHAPE[1:]),
                nbytes=len(payload),
                rank=ParallelRank(pp=0, ep=expert_id),
            )
            placement_fragments.append(placement_fragment)
            binding_fragments.append(
                runtime_fragment(
                    placement=placement_fragment,
                    tensor=descriptor_by_id[tensor_id],
                    fragment_id=f"{worker_id}:{tensor_id}",
                    address=buffer.pointer,
                    worker_id=worker_id,
                    endpoint=source_endpoint,
                    owner=buffer,
                )
            )
        source_groups[(0, 0, 0, expert_id)] = list(
            _strict_zip(
                source_descriptors,
                placement_fragments,
                binding_fragments,
            )
        )

    target_groups = {}
    target_buffers = []
    target_expected = []
    for tensor_id, base, target_pp, target_shards in tensor_specs:
        target_dim = target_pp
        target_extent = _CROSS_DIM_SHAPE[target_dim] // target_shards
        target_descriptor = TensorDescriptor(
            tensor_id=tensor_id,
            global_shape=_CROSS_DIM_SHAPE,
            dtype="uint8",
            itemsize=1,
            shard_dims=(target_dim,),
            layer_id=descriptor_by_id[tensor_id].layer_id,
            layout_fingerprint="native-reader:cross-dim:uint8:v2",
            parallel_axes=(
                OwnershipAxis("pp"),
                SplitAxis("tp", dim=target_dim),
            ),
        )
        for target_rank in range(target_shards):
            global_offset = [0, 0, 0]
            local_shape = list(_CROSS_DIM_SHAPE)
            global_offset[target_dim] = target_rank * target_extent
            local_shape[target_dim] = target_extent
            expected = bytes(
                base
                + expert_id * expert_extent
                + out_index * _CROSS_DIM_SHAPE[2]
                + in_index
                for expert_id in range(_CROSS_DIM_SHAPE[0])
                for out_index in range(
                    global_offset[1], global_offset[1] + local_shape[1]
                )
                for in_index in range(
                    global_offset[2], global_offset[2] + local_shape[2]
                )
            )
            buffer = allocate_target(len(expected))
            buffer.write(bytes([_READER_TARGET_SENTINEL]) * len(expected))
            target_buffers.append(buffer)
            target_expected.append(expected)
            worker_id = f"cross-dim-target-p{target_pp}-t{target_rank}"
            target_fragment = PlacementFragment(
                tensor_id=tensor_id,
                global_offset=tuple(global_offset),
                local_shape=tuple(local_shape),
                nbytes=len(expected),
                rank=ParallelRank(tp=target_rank, pp=target_pp),
            )
            target_binding_fragment = runtime_fragment(
                placement=target_fragment,
                tensor=target_descriptor,
                fragment_id=f"{worker_id}:{tensor_id}",
                address=buffer.pointer,
                worker_id=worker_id,
                endpoint=target_endpoint,
                owner=buffer,
            )
            target_groups.setdefault((0, target_rank, target_pp, 0), []).append(
                (
                    target_descriptor,
                    target_fragment,
                    target_binding_fragment,
                )
            )

    return CrossDimReaderFixture(
        sources=_runtime_inputs_from_groups(
            source_groups,
            resource_id="native-reader-cross-dim-e2e",
            revision=revision,
            placement_set_id="cross-dim-source",
            lease_prefix="cross-dim-source",
        ),
        targets=_runtime_inputs_from_groups(
            target_groups,
            resource_id="native-reader-cross-dim-e2e",
            revision=revision,
            placement_set_id="cross-dim-target",
            lease_prefix="cross-dim-target",
        ),
        source_buffers=tuple(source_buffers),
        target_buffers=tuple(target_buffers),
        target_expected=tuple(target_expected),
    )
