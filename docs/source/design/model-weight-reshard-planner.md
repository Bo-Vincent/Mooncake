# Model Weight Reshard Planner

The model-weight planner converts one complete source placement and one complete
target placement into backend-neutral transfer regions. It does not require
runtime addresses while computing logical overlap.

## Inputs

Both source and target are singular `WeightPlacementManifest` objects. Each
manifest describes one complete global logical placement of the same model,
revision, and weight generation, including its `ParallelTopology`, participants,
tensors, and per-participant `WeightPlacementPart` values.

The public APIs are:

- `plan_placement_transfer(source_placement, target_placement)` for a complete
  source-to-target plan;
- `plan_placement_transfer_to_local_target(source_placement, target_placement,
  target_participant_id)` for one target participant in a global placement.

The local-target form still receives the complete source and target placement;
the participant ID selects the executor output, not a partial target manifest.

## Planning Flow

1. Validate both global placements and their complete logical coverage.
2. Match logical tensors and intersect their source and target N-D boxes.
3. Produce a reusable `LogicalTransferPlan` without runtime addresses.
4. Validate the required per-participant runtime bindings and bind the logical
   plan into a physical `TransferPlan`.
5. Lower each `TransferRegion` to Store or Transfer Engine operations.

## Transfer Region

Each `TransferRegion` describes one N-D overlap: overlap offset and shape,
source and target base byte offsets, contiguous `inner_bytes`, outer loop
counts, and source and target strides.

Cross-dimension resharding stays at region granularity instead of expanding the
logical plan into one operation per row or element. `iter_segments()` provides
bounded lazy lowering for backends that accept only flat ranges.

## Parallel Semantics

- TP changes source and target logical boxes. The same overlap algorithm handles
  split, merge, and different source and target shard dimensions.
- PP is framework-provided layer or tensor ownership. Operations are grouped by
  `(source_pp, target_pp)`; the planner does not derive PP ownership from names
  or a layer-count formula. Multiple complete PP replicas are supported and
  validated independently; partial coverage cannot span PP owners.
- EP uses the leading logical expert coordinate for grouped expert tensors.
  Independent expert allocations remain independent fragments and are not
  packed or all-gathered by the planner.
- DP does not change tensor geometry. A plan may choose one complete source DP
  replica for a target while both global manifests retain their declared
  `dp_size` and participant mappings.

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

A physical `TransferPlan` accepts live fragments only with a
`RuntimeBindingAttestation`: the exact placement/binding pair has already
passed runtime validation. Its resource ID, revision, generation, executor
routing, fragment leases, and address view are checked again when the plan is
assembled. The attestation serializes only canonical placement/binding inputs;
derived indexes are rebuilt and revalidated after a wire round trip. Store
sources remain distinct: `WeightManifest` plus `WeightLoadPlan` authorizes the
stored object set, while the live target still requires the same attestation.

The executable plan also retains the complete target placement. Its public
construction boundary re-checks that every selected target participant and
fragment is covered exactly once, so truncating an otherwise internally
consistent plan cannot authorize a partial target write. Physical target range
sharing is accepted only for the same complete declared alias group under one
identical runtime binding and lease scope; all other exact or partial overlaps
fail closed. Executor lease snapshots are rebuilt from runtime fragment IDs in
linear time, without quadratic membership scans.

## Boundaries

The planner performs copy-only logical resharding. It does not infer model
semantics from names or transform dtype, quantization, packing, swizzle, or
kernel-specific layouts.
