# Mooncake Reshard

`mooncake-reshard` provides framework-neutral contracts and execution adapters
for reusable runtime resources. The model-weight specialization supports global
logical placement, N-D reshard planning, runtime binding, Mooncake Store, and
Transfer Engine execution.

Framework-owned adapters outside Mooncake inspect framework runtime objects,
normalize framework-specific values, and construct the typed canonical
manifests. Mooncake core accepts only those typed values; it does not import or
inspect framework objects or accept alternate field names or duck-typed
records.

The public Python API is split by responsibility:

- `mooncake.reshard.contracts` exposes `ResourceManifest`,
  `PlacementManifest`, and `RuntimeBindingManifest` as structural `Protocol`
  contracts for resource-neutral identity and lifecycle;
- `mooncake.reshard.transfer_engine` owns physical batches, registration leases,
  completion handling, and pending-resource quarantine;
- `mooncake.reshard.weight` defines model-weight placement, planning, binding,
  Store, and TE adapters.

## Weight Placement Model

`WeightPlacementManifest` is one complete, address-free global logical placement
of a model-weight generation. It contains:

- a `ParallelTopology` with TP, PP, EP, and DP sizes plus the exact selected
  participants;
- per-tensor `SplitAxis(kind, dim)`, `ReplicatedAxis(kind)`, and
  `OwnershipAxis(kind)` entries that distinguish logical sharding, complete
  replicas, and ownership without overloading an optional dimension;
- canonical `TensorDescriptor` values whose only shard representation is
  `shard_dims`;
- one `WeightPlacementPart` for every selected participant;
- canonical global tensor descriptors and N-D logical fragments;
- a placement ID and digest computed only after the full part set validates.

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

## Planning And Execution

Logical planning uses singular global manifests:

```python
logical_plan = plan_placement_transfer(source_placement, target_placement)
```

For one target executor, the caller still supplies both complete placements and
selects a participant explicitly:

```python
logical_plan = plan_placement_transfer_to_local_target(
    source_placement,
    target_placement,
    target_participant_id,
)
```

The planner computes backend-neutral N-D overlap regions. PP routes tensor
ownership, EP uses the leading logical expert coordinate for grouped expert
tensors, TP changes logical shard boxes, and DP selects complete replicas
without changing tensor geometry. Runtime binding supplies the physical
addresses only after logical planning. A `LogicalTransferPlan` must cover every
selected target fragment exactly once. Construction and the public bind boundary
both enforce that rule: canonical `TransferRegion` values are checked as N-D
logical boxes without row expansion and fail closed when coverage is incomplete.

Weight Store and TE adapters consume the same bound plan. Store persists weight
payload fragments and their storage manifest; TE lowers regions into bounded
physical transfer batches while preserving allocation, lease, generation, and
completion fences.

Runtime addresses alone do not authorize Store or TE I/O. A framework adapter
must supply an allocation guard for every participating runtime binding. The
guard pins the framework-owned allocation, returns a fresh binding while pinned,
and is released only after terminal completion or explicit cleanup. Process-
local allocation owners are intentionally excluded from serialized bindings and
plans.

The executable plan retains the complete target placement and re-checks exact
coverage for every selected target fragment at its public construction
boundary. A shared target physical range is allowed only for a complete,
declared alias group under the same attested runtime binding and lease scope;
other overlaps fail closed.

## Source Layout

- `types.py` defines tensor and logical-fragment contracts;
- `topology.py` defines parallel sizes and selected participants;
- `part.py` defines one participant's address-free contribution;
- `placement.py` assembles and identifies the complete global placement;
- `runtime.py` defines typed physical bindings;
- `validation.py` checks logical geometry, coverage, declared storage alias
  groups, and addresses;
- `binding.py` validates placement and binding attestation;
- `_planner/` implements N-D overlap planning and late runtime binding;
- `planner.py` preserves the planner public import surface;
- `manifest.py` preserves the public import surface.

- `contracts/` contains resource-neutral contracts and adapter registration;
- `transfer_engine/` contains resource-neutral physical transfer ownership;
- `weight/topology.py`, `weight/part.py`, and `weight/placement.py` define the
  complete logical placement;
- `weight/runtime.py` and `weight/binding.py` define per-participant live
  bindings;
- `weight/_planner/` implements N-D planning and late binding;
- `weight/_store/` and `weight/_te/` implement Store and TE adapters;
- `weight/manifest.py`, `weight/planner.py`, `weight/store.py`, and
  `weight/te.py` preserve the public import surface.

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

The heterogeneous reshard benchmark lives under
`mooncake-reshard/benchmarks/heterogeneous_weight_reshard`.
