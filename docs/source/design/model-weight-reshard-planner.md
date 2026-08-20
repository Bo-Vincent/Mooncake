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
placements; the participant ID selects the executor output, not a partial
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

Coverage validation uses an ordered interval scan for 1-D inputs and a
coordinate-compressed sweep for 2-D inputs, both with `O(N log N)` behavior.
For 3-D and higher logical boxes, exact intersection remains supported under an
explicit pairwise-comparison budget; inputs that exceed it fail closed rather
than making validation work unbounded.

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
sources remain distinct: the bound plan retains a typed `StoredManifestIdentity`,
and the Store/control plane supplies the authoritative `WeightManifest` for a
fresh identity-and-fragment check before binding. The live target still requires
the same attestation.

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
