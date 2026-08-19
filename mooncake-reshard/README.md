# Mooncake Reshard

`mooncake-reshard` defines framework-neutral contracts and address-free N-D
logical planning for reusable runtime resources. This phase adds the
model-weight manifest and logical planner; runtime binding, storage, and
transfer execution remain separate phases.

Framework-owned adapters outside Mooncake inspect framework runtime objects,
normalize framework-specific values, and construct the typed canonical
manifests. Mooncake core accepts only those typed values; it does not import or
inspect framework objects or accept alternate field names or duck-typed
records.

The public Python API is split by responsibility:

- `mooncake.reshard.contracts` exposes `ResourceManifest`,
  `PlacementManifest`, and `RuntimeBindingManifest` as structural `Protocol`
  contracts for resource-neutral identity and lifecycle;
- `mooncake.reshard.weight` defines model-weight placement and address-free
  N-D planning. `WeightRuntimeBindingManifest` remains an input contract for a
  later runtime-binding phase.

## Weight Placement Model

`WeightPlacementManifest` describes one complete, address-free global logical
placement of a model-weight generation. It contains:

- a `ParallelTopology` with TP, PP, EP, and DP sizes plus the exact selected
  participants;
- per-tensor `SplitAxis(kind, dim)`, `ReplicatedAxis(kind)`, and
  `OwnershipAxis(kind)` entries that distinguish logical sharding, complete
  replicas, and ownership without overloading an optional dimension;
- canonical `TensorDescriptor` values whose only shard representation is
  `shard_dims`;
- one `WeightPlacementPart` for every selected participant;
- canonical global tensor descriptors and N-D logical fragments;
- a placement ID and digest computed after the full part set validates.

`ParallelTopology.world_size` is the selected participant count. It is not
inferred from `tp_size * pp_size * ep_size * dp_size`: parallel axes may share
workers, and a placement may select one complete DP replica while retaining the
runtime's declared `dp_size`. The overall participant map may be non-Cartesian,
but a tensor that declares multiple independent `SplitAxis` values must provide
the Cartesian rank combinations needed to prove each axis-to-dimension split.

For each framework participant, the framework adapter first constructs an
address-free `WeightPlacementPart`. A collection barrier assembles the exact
participant set and validates complete logical tensor coverage. Each part
declares exactly the tensor descriptors referenced by its fragments. For each
live participant that owns fragments, the adapter then constructs one
`WeightRuntimeBindingManifest` with physical fragments, generation, and lease,
attesting the global placement ID and digest. A physical fragment preserves its
item size, view shape, byte strides, storage base, byte offset, and allocation
size so binding validation can prove canonical contiguity and address bounds.

An alias group may span placement parts, so an individual part checks only its
local fragment invariants. `WeightPlacementManifest` performs the global check
after collection: every alias member must be in the complete tensor catalog and
every fragment for every member must declare the identical group. Runtime paths
consume only this globally validated placement.

Empty participants need no runtime binding; any participant referenced by
execution must provide one.

## Logical Planning

The planner consumes complete source and target placement collections:

```python
logical_plan = plan_placement_transfer(source_placements, target_placements)
```

For one target participant, the caller still provides the complete source
placement collection and selects a target placement explicitly:

```python
logical_plan = plan_placement_transfer_to_local_target(
    source_placements,
    target_placement,
)
```

The result is a backend-neutral `LogicalTransferPlan`. It contains N-D
logical-box overlap regions but no GPU address, endpoint, allocation range,
lease, or backend request. TP changes shard boxes, PP uses framework-provided
ownership, EP is represented by logical expert coordinates, and DP selects
complete replicas without changing tensor geometry.

Each `TransferRegion` records overlap offset and shape, source and target base
byte offsets, contiguous `inner_bytes`, outer loop counts, and source/target
byte strides. The planner preserves this compact strided representation rather
than expanding one operation per row or element.

This phase does not consume a committed Store `WeightManifest`, bind a logical
plan to live runtime fragments, submit to the Transfer Engine, or transform
dtype, quantization, packing, swizzle, or checkpoint format. A subsequent
runtime-binding phase may use the existing typed binding contracts to attach
physical locations only after logical planning has completed.

The weight implementation is split by responsibility:

- `types.py` defines tensor and logical-fragment contracts;
- `topology.py` defines parallel sizes and selected participants;
- `part.py` defines one participant's address-free contribution;
- `placement.py` assembles and identifies the complete global placement;
- `runtime.py` defines typed physical bindings;
- `validation.py` checks manifest geometry, coverage, declared storage alias
  groups, and runtime-binding input shape;
- `_planner/` computes address-free N-D overlap regions;
- `manifest.py` preserves the public import surface.

`kv_cache` is reserved as a resource discriminator, but this change does not
define a KVCache manifest. Framework adapters must provide tensor semantics;
Mooncake does not infer them from parameter names.

`weight_placement_to_json` and `weight_placement_from_json` are the explicit
public JSON APIs. Their wire format contains only canonical fields, and
deserialization rejects alternate field names rather than translating
framework-specific input.

Run the contract and static type checks from the repository root:

```bash
PYTHONPATH=mooncake-wheel:mooncake-reshard/python \
python -m pytest -q mooncake-reshard/tests

bash scripts/check_reshard_types.sh
```
