# Model Weight Logical Reshard Planner

`mooncake-reshard` plans an address-free conversion between complete model
weight placements. It turns source and target placement collections into
compact N-D logical transfer regions. It does not inspect framework runtime
objects or assign physical GPU addresses.

## Inputs and Output

`plan_placement_transfer(source_placements, target_placements)` takes complete
source and target collections of `WeightPlacementManifest` values. Each
collection supplies the participants, global tensor descriptors, and N-D
logical fragments needed to assemble one placement view. The source and target
must identify the same resource and revision.

`plan_placement_transfer_to_local_target(source_placements,
target_placement)` keeps the full source view but selects one target placement
for a local planner output. It is not a partial-source interface.

Both APIs return a `LogicalTransferPlan`. It contains canonical tensor
descriptors, placement participants, and logical regions only. It contains no
GPU address, endpoint, allocation range, lease, or backend handle.

## N-D Regions

Each `TransferRegion` represents one source/target N-D box overlap. It records
the overlap offset and shape, source and target base byte offsets, contiguous
`inner_bytes`, outer loop counts, and source/target byte strides.

The planner retains a compact strided representation. Cross-dimension overlap
is not expanded into one operation per row or element. A later backend lowering
layer is responsible for applying an explicit operation bound when it
materializes flat transfer segments.

## Parallel Semantics

- **TP** changes source and target logical boxes. The same overlap algorithm
  handles split, merge, and source/target sharding on different dimensions.
- **PP** is explicit framework-provided tensor or layer ownership. The planner
  routes by the placement owner; it does not infer ownership from parameter
  names or a layer-count formula.
- **EP** is represented by a logical expert coordinate. Independent expert
  allocations remain independent logical fragments and are never packed or
  all-gathered by the planner.
- **DP** does not change tensor geometry. It selects a complete source replica
  while preserving the declared topology and participant mapping.

All four axes are resolved by one logical-box plan, rather than by model-wide
per-axis conversion passes.

## Validation

Placement assembly checks tensor descriptors, participant membership, N-D
geometry, ownership, aliases, and logical coverage before planning. The
planner then fails closed when source and target resource, revision, tensor
identity, dtype, global shape, layout fingerprint, ownership, or coverage are
inconsistent.

## Boundary to Later Phases

This phase is limited to logical planning. `WeightRuntimeBindingManifest` is a
typed physical-binding input contract, but no public runtime-binding operation
is introduced here. A later phase can attach physical fragments, generations,
leases, and address bounds to a `LogicalTransferPlan` before producing a
backend-facing plan.

Store snapshots, Transfer Engine lowering and submission, framework activation,
and model-specific format conversion remain outside this logical planner.
Framework adapters own conversion into canonical manifests; Mooncake core does
not infer framework or model semantics from parameter names.
