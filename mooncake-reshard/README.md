# Mooncake Reshard

`mooncake-reshard` defines framework-neutral contracts and logical planning for
reusable runtime resources. This change adds the model-weight manifest and N-D
reshard planner; storage and transfer execution are added separately.

Framework-owned adapters outside Mooncake inspect framework runtime objects,
normalize framework-specific values, and construct the typed canonical
manifests. Mooncake core accepts only those typed values; it does not import or
inspect framework objects or accept alternate field names or duck-typed
records.

The public Python API is split by responsibility:

- `mooncake.reshard.contracts` exposes `ResourceManifest`,
  `PlacementManifest`, and `RuntimeBindingManifest` as structural `Protocol`
  contracts for resource-neutral identity and lifecycle;
- `mooncake.reshard.weight` defines model-weight placement, runtime binding,
  and address-free N-D reshard planning.

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

Empty participants need no runtime binding; every participant selected by a
bound plan must provide one.

## Logical Planning

The planner consumes complete source and target placements:

```python
logical_plan = plan_placement_transfer(source_placement, target_placement)
```

The result contains backend-neutral N-D overlap regions. TP and EP use logical
box split/merge, PP routes framework-provided ownership, and DP selects a
complete source replica without changing tensor geometry. Runtime addresses are
introduced only by binding the logical plan to validated participant bindings.
The result is a `TransferPlan`: a validated placement-and-runtime snapshot,
not an executable transfer request and not an allocation lifetime token.
It retains the complete target placement and revalidates full coverage for the
selected target participants. Declared aliases preserve each logical target;
only equal physical ranges within one attested target binding and lease scope
are accepted.

The weight implementation is split by responsibility:

- `types.py` defines tensor and logical-fragment contracts;
- `topology.py` defines parallel sizes and selected participants;
- `part.py` defines one participant's address-free contribution;
- `placement.py` assembles and identifies the complete global placement;
- `runtime.py` defines typed physical bindings;
- `validation.py` checks logical geometry, coverage, declared storage alias
  groups, and addresses;
- `binding.py` validates placement and binding attestation;
- `_planner/` computes N-D overlap regions and late runtime binding;
- `planner.py` exposes the planner public API;
- `manifest.py` preserves the public import surface.

`kv_cache` is reserved as a resource discriminator, but this change does not
define a KVCache manifest. Framework adapters must provide tensor semantics;
Mooncake does not infer them from parameter names.

`weight_placement_to_json` and `weight_placement_from_json` are the explicit
public JSON APIs. Their wire format contains only canonical fields, and
deserialization rejects alternate field names rather than translating
framework-specific input.

`PlanningLimits` bounds both transfer regions and flattened segments. A backend
that needs flat ranges must call `iter_segments(max_segments=...)`; it cannot
expand an N-D region without an explicit cap. A future TE executor remains
responsible for obtaining allocation lifetime guards, revalidating the bound
snapshot, and submitting the lowered operations.

Run the contract and static type checks from the repository root:

```bash
PYTHONPATH=mooncake-wheel:mooncake-reshard/python \
python -m pytest -q mooncake-reshard/tests

bash scripts/check_reshard_types.sh

bash scripts/check_reshard_python39_import.sh
```
