# Model Weight Reshard Planner

The model-weight planner converts one complete source placement or committed
Store snapshot and one complete target placement into backend-neutral transfer
regions. It does not require runtime addresses while computing logical overlap.

## Inputs

The target is a singular `WeightPlacementManifest`. The source is either a
complete `WeightPlacementManifest` or a committed `WeightManifest` snapshot.
Both forms describe the same model, revision, and weight generation. A
placement carries `ParallelTopology`, participants, and per-participant
`WeightPlacementPart` values; a stored source carries its group/key and
canonical content digest as immutable source provenance.

The public APIs are:

- `plan_placement_transfer(source_placement, target_placement)` for a complete
  source-to-target plan;
- `plan_placement_transfer_to_local_target(source_placement, target_placement,
  target_participant_id)` for one target participant in a global placement.

The local-target placement form still receives complete source and target
placements; the participant ID selects one participant output, not a partial
target manifest.

A stored-source logical plan retains that manifest provenance, but binding must
receive a fresh authoritative manifest from Store or the control plane and
compare its identity and every selected stored fragment before producing a
bound plan. A serialized plan is therefore not trusted as the authority for an
object key or range.

## Planning Flow

1. Validate both global placements and their complete logical coverage.
2. Match logical tensors and intersect their source and target N-D boxes.
3. Produce a reusable `LogicalTransferPlan` without runtime addresses.
4. Validate the required per-participant runtime bindings and bind the logical
   plan into a physical `TransferPlan`.

The bound plan is a validated snapshot for a later executor. It has no submit
or execute API and does not itself retain a GPU allocation lifetime token.

## Transfer Region

Each `TransferRegion` describes one N-D overlap: overlap offset and shape,
source and target base byte offsets, contiguous `inner_bytes`, outer loop
counts, and source and target strides.

Cross-dimension resharding stays at region granularity instead of expanding the
logical plan into one operation per row or element. `iter_segments()` requires
an explicit `max_segments` bound. `PlanningLimits` additionally caps both the
total region count and the plan's total lowered-segment count.

## Parallel Semantics

- TP changes source and target logical boxes. The same overlap algorithm handles
  split, merge, and different source and target shard dimensions.
- PP is framework-provided layer or tensor ownership. Operations are grouped by
  `(source_pp, source_pipeline_stage_id, target_pp, target_pipeline_stage_id)`.
  `pipeline_stage_id` is optional placement metadata and defaults to `rank.pp`
  for a simple pipeline. The planner does not derive PP ownership from names or
  a layer-count formula. Multiple complete PP replicas are supported and
  validated independently; partial coverage cannot span PP owners.
- EP uses the leading logical expert coordinate for grouped expert tensors.
  Independent expert allocations remain independent fragments and are not
  packed or all-gathered by the planner.
- DP does not change tensor geometry. `ReplicatedAxis(kind="dp")` allows a plan
  to choose one complete source replica for a target. `OwnershipAxis(kind="dp")`
  instead routes each tensor only through its declared DP owner; it never
  requires every tensor to exist on every DP rank.

The four axes are resolved in one N-D logical-box plan rather than four
model-wide transformation passes.

## Validation

Global manifest construction rejects missing participants, mixed generations
or topologies, and incomplete or conflicting tensor coverage. Planning then
fails closed when source and target tensor identity, dtype, global shape,
layout fingerprint, ownership, or logical coverage is inconsistent.

Runtime binding rechecks the global placement ID and digest, exact participant
and fragment membership, leases, generations, aliases, and physical address
bounds. The runtime binding remains the only source of truth for live physical
locations; the planner never invents addresses from topology metadata.

`TransferPlan` contains only runtime fragments that carry a
`RuntimeBindingAttestation`. The attestation records the exact placement,
resource, revision, weight generation, lease generation, and physical view at
bind time. It prevents a caller from substituting a same-size view or forging
the plan identity, but it is not an allocation pin. The optional framework
`owner` object is opaque and is not interpreted as a lifetime token.

The bound contract retains the complete target placement and rechecks complete
coverage for every selected target participant. Declared aliases retain one
logical target provenance each; exact physical overlap is accepted only within
the same attested target binding and lease scope. Equal addresses from distinct
runtime binding lifetimes are rejected rather than collapsed.

## Boundaries

The planner performs copy-only logical resharding. It does not infer model
semantics from names or transform dtype, quantization, packing, swizzle, or
kernel-specific layouts.

This phase does not submit to Mooncake TE. A future executor must acquire
source and target allocation lifetime guards from its runtime, revalidate the
selected bound participants immediately before submission, submit the lowered
operations, and release those guards only after completion.
