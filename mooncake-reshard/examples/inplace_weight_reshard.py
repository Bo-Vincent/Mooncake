"""CPU byte-copy example of the explicit same-fleet TP2 -> DP2 path.

This exercises planning, fresh binding, registration leases and lifetime guards.
It does not qualify a GPU transport or production scheduler integration.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from uuid import uuid4

from mooncake.reshard.lifetime import AllocationFence, TerminalTransferState
from mooncake.reshard.weight import (
    ParallelRank,
    ParallelTopology,
    PlacementFragment,
    ReplicatedAxis,
    RuntimeBindingFragment,
    SplitAxis,
    TensorDescriptor,
    TopologyParticipant,
    WeightPlacementManifest,
    WeightRuntimeBindingManifest,
)
from mooncake.reshard.weight.inplace import InplaceWeightResharder
from mooncake.reshard.weight.lifetime import (
    AcquiredWeightBinding,
    weight_allocation_fence,
)
from mooncake.reshard.weight.te import (
    MemoryRegistrationLease,
    MooncakeTransferEngineReader,
)


class CpuCopyEngine:
    def register_memory(self, address: int, nbytes: int) -> int:
        return 0

    def unregister_memory(self, address: int) -> int:
        return 0

    def batch_transfer_sync_read(self, endpoint, targets, sources, sizes):
        for target, source, size in zip(targets, sources, sizes, strict=True):
            ctypes.memmove(target, source, size)
        return 0


@dataclass
class CpuPin:
    fence: AllocationFence
    owner: object

    def release_after_terminal(self, terminal_state: TerminalTransferState) -> None:
        self.owner = None


@dataclass
class CpuGuard:
    binding: WeightRuntimeBindingManifest
    owner: object

    def acquire(self, *, transfer_id, expected_binding, required_fragment_ids):
        if expected_binding != self.binding:
            raise ValueError("CPU allocation binding changed")
        return AcquiredWeightBinding(
            binding=self.binding,
            token=CpuPin(
                weight_allocation_fence(
                    self.binding, required_fragment_ids, token_id=uuid4().hex
                ),
                self.owner,
            ),
        )


def placement(*, target: bool) -> WeightPlacementManifest:
    ranks = tuple(
        ParallelRank(dp=i) if target else ParallelRank(tp=i) for i in range(2)
    )
    topology = ParallelTopology(
        tp_size=1 if target else 2,
        pp_size=1,
        ep_size=1,
        dp_size=2 if target else 1,
        participants=tuple(
            TopologyParticipant(f"worker-{i}", rank) for i, rank in enumerate(ranks)
        ),
    )
    tensor = TensorDescriptor(
        tensor_id="weight",
        global_shape=(8,),
        dtype="uint8",
        itemsize=1,
        shard_dims=(0,),
        layout_fingerprint="cpu-example",
        parallel_axes=(ReplicatedAxis("dp"), SplitAxis("tp", dim=0)),
    )
    size = 8 if target else 4
    return WeightPlacementManifest.from_fragments(
        resource_id="example",
        revision="step-1",
        weight_generation=1,
        placement_set_id="target" if target else "source",
        topology=topology,
        tensors=(tensor,),
        fragments=tuple(
            PlacementFragment(
                placement_fragment_id=f"{'target' if target else 'source'}-{i}",
                tensor_id="weight",
                global_offset=(0 if target else i * 4,),
                local_shape=(size,),
                nbytes=size,
                rank=rank,
            )
            for i, rank in enumerate(ranks)
        ),
    )


def bindings(placement, buffers):
    result = []
    for i, part in enumerate(placement.parts):
        fragment = part.fragments[0]
        address = ctypes.addressof(buffers[i])
        result.append(
            WeightRuntimeBindingManifest(
                resource_id=placement.resource_id,
                revision=placement.revision,
                placement_id=placement.placement_id,
                placement_digest=placement.digest,
                participant_id=part.participant_id,
                instance_id=f"cpu-{part.participant_id}",
                generation=1,
                lease_id=f"{placement.placement_set_id}-lease-{i}",
                fragments=(
                    RuntimeBindingFragment(
                        placement_fragment_id=fragment.placement_fragment_id,
                        fragment_id=f"{placement.placement_set_id}-runtime-{i}",
                        address=address,
                        nbytes=fragment.nbytes,
                        worker_id=part.participant_id,
                        endpoint=f"cpu-{i}",
                        device="cpu",
                        itemsize=1,
                        local_shape=fragment.local_shape,
                        strides_bytes=(1,),
                        storage_address=address,
                        storage_nbytes=fragment.nbytes,
                        storage_offset_bytes=0,
                    ),
                ),
            )
        )
    return tuple(result)


def registrations(bindings):
    return tuple(
        MemoryRegistrationLease(
            fragment_id=fragment.fragment_id,
            address=fragment.address,
            nbytes=fragment.nbytes,
            worker_id=fragment.worker_id,
            device=fragment.device,
            itemsize=fragment.itemsize,
            local_shape=fragment.local_shape,
            strides_bytes=fragment.strides_bytes,
            storage_offset_bytes=fragment.storage_offset_bytes,
            storage_address=fragment.storage_address,
            storage_nbytes=fragment.storage_nbytes,
            lease_generation=binding.generation,
            runtime_lease_id=binding.lease_id,
        )
        for binding in bindings
        for fragment in binding.fragments
    )


def main() -> None:
    source_buffers = (
        ctypes.create_string_buffer(b"abcd", 4),
        ctypes.create_string_buffer(b"efgh", 4),
    )
    target_buffers = (ctypes.create_string_buffer(8), ctypes.create_string_buffer(8))
    source, target = placement(target=False), placement(target=True)
    source_bindings, target_bindings = (
        bindings(source, source_buffers),
        bindings(target, target_buffers),
    )
    source_guards = {
        (binding.instance_id, binding.participant_id): CpuGuard(binding, owner)
        for binding, owner in zip(source_bindings, source_buffers, strict=True)
    }
    target_guards = {
        (binding.instance_id, binding.participant_id): CpuGuard(binding, owner)
        for binding, owner in zip(target_bindings, target_buffers, strict=True)
    }
    resharder = InplaceWeightResharder(MooncakeTransferEngineReader(CpuCopyEngine()))
    for binding in target_bindings:
        resharder.execute(
            source,
            source_bindings,
            target,
            binding,
            source_registrations=registrations(source_bindings),
            source_allocation_guards=source_guards,
            target_allocation_guards=target_guards,
        )
    assert tuple(buffer.raw for buffer in target_buffers) == (b"abcdefgh", b"abcdefgh")
    print("CPU same-fleet TP2 -> DP2: both targets equal b'abcdefgh'")


if __name__ == "__main__":
    main()
