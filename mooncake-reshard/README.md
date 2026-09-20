# Mooncake Reshard

`mooncake-reshard` defines framework-neutral contracts, address-free N-D
logical planning, runtime binding, and Store-backed snapshots for reusable
runtime resources. Model-weight storage and Transfer Engine execution remain
separate runtime phases.

Framework-owned adapters inspect framework runtime objects, normalize
framework-specific values, and construct typed canonical manifests. Mooncake
core accepts only those typed values; it does not import or inspect framework
objects or accept alternate field names or duck-typed records.

The public Python API is split by responsibility:

- `mooncake.reshard.contracts` exposes `ResourceManifest`,
  `PlacementManifest`, and `RuntimeBindingManifest` as structural `Protocol`
  contracts for resource-neutral identity and lifecycle;
- `mooncake.reshard.weight` defines model-weight placement, runtime-binding
  input contracts, and address-free N-D planning.
- `mooncake.reshard.kv_cache` defines KV-cache topology, placement, runtime
  binding, serialization, and logical transfer planning.

## Weight Placement Model

`WeightPlacementManifest` describes one complete, address-free global logical
placement of a model-weight generation. It contains:

- a `ParallelTopology` with TP, PP, EP, and DP sizes plus the selected
  participants;
- per-tensor `SplitAxis(kind, dim)`, `ReplicatedAxis(kind)`, and
  `OwnershipAxis(kind)` entries that distinguish logical sharding, complete
  replicas, and ownership;
- canonical `TensorDescriptor` values whose only shard representation is
  `shard_dims`;
- one `WeightPlacementPart` for every selected participant;
- canonical global tensor descriptors and N-D logical fragments;
- a placement ID and digest computed after the complete part set validates.

`ParallelTopology.world_size` is the selected participant count. It is not
inferred from `tp_size * pp_size * ep_size * dp_size`: parallel axes may share
workers, and a placement may select one complete DP replica while retaining the
runtime's declared `dp_size`. A tensor that declares independent `SplitAxis`
values must provide the rank combinations needed to prove its axis-to-dimension
splits.

Framework adapters first construct address-free `WeightPlacementPart` values.
A collection barrier assembles the complete participant set and validates
logical tensor coverage. The logical planner does not consume GPU addresses or
allocation metadata.

An alias group may span placement parts. `WeightPlacementManifest` performs the
global check after collection: every alias member must be in the complete
tensor catalog and every fragment for every member must declare the identical
group.

## Logical Planning

The planner consumes complete source and target placements:

```python
logical_plan = plan_placement_transfer(source_placement, target_placement)
```

The result is a backend-neutral `LogicalTransferPlan`. TP and EP use N-D
logical-box split and merge, PP routes framework-provided ownership, and DP
selects a complete source replica or follows a declared DP owner. The plan
contains compact N-D overlap regions but no runtime addresses, endpoints,
allocation bounds, leases, or backend request.

`plan_stored_transfer_to_target_placement` accepts a committed address-free
`StoredWeightManifest` as the logical source. The plan retains the manifest's
canonical identity and selected fragments so a later binding layer can verify
that it is still using the intended Store snapshot.

`PlanningLimits` bounds both transfer-region creation and later flattened
segment expansion. A backend must supply an explicit expansion bound; it cannot
turn a compact N-D plan into an unbounded number of operations.

The logical planner is copy-only. It does not infer model semantics from tensor
names or transform dtype, quantization, packing, swizzle, or checkpoint format.

## Runtime Binding

`bind_logical_transfer_plan` attaches selected typed
`WeightRuntimeBindingManifest` values to a `LogicalTransferPlan`. It rechecks
placement identity and digest, runtime fragment geometry, address/allocation
bounds, generation, lease, target coverage, and alias scope before returning a
bound `TransferPlan`.

This plan records runtime evidence but neither submits a copy nor retains the
underlying allocation. A Transfer Engine executor must acquire allocation
guards and revalidate bindings atomically with the submission that consumes
the plan.

## Store Snapshots

`StoredResourceManifest` is the persistent resource base. The concrete
`StoredWeightManifest` adds revision, weight generation, tensor descriptors,
stored fragments, and its canonical digest. It contains no GPU address or
runtime allocation owner.

`WeightStore.weight_put()` begins one immutable revision and returns a
`WeightStoreWriter`. A framework adapter supplies complete placement/binding
inventories, resolves each `weight_put_tensor()` call to canonical fragment
IDs, and provides allocation guards for Store I/O. After every required
fragment succeeds, `commit()` publishes the manifest and revision metadata.

Policy-only changes use
`weight_update_policy(identity, *, policy, expected_metadata_generation)`.
There is no compatibility wrapper for the former public name.

`weight_upsert(snapshot, adapter, *, replacing, expected_metadata_generation,
mode=WeightUpsertMode.PUT_FIRST, tenant_id="default", policy=None,
request_id=None)` creates a writer for a
higher generation in the same
`(tenant_id, namespace, resource_id, revision)` lineage. It replaces one
immutable generation with another rather than overwriting an ordinary Store
key in place. The durable lineage claim provides ordering, CAS fencing, and
recovery; it is not a serving-active pointer.

`PUT_FIRST` is the default. It publishes the target `READY` before fencing the
base, then drains existing base leases and deletes the base. A target failure
leaves the base intact, at a peak storage cost approaching two revisions.
`DELETE_FIRST` requires no active base lease or lifecycle operation, deletes
the base before allowing target import, and therefore trades lower peak storage
for an unavailable and non-rollback window.

When `request_id` is omitted, `WeightStore` deterministically hashes the
canonical base identity, target identity, mode, and
`expected_metadata_generation`. Repeating the same call after a lost begin
response derives the same ID. An explicit `request_id` is used unchanged. A
retry with the same ID and arguments resumes the claim; a competing request
cannot create another successor, and HA recovery continues from the persisted
phase without decreasing the generation watermark.

Reconstructing an interrupted writer with the same request ID, target identity,
and source placement reuses its payload and transaction keys. Manifest creation
time comes from the Store import metadata, so retries also work after manifest
publication. Resubmit every selected tensor fragment before committing the new
writer. The caller must retain the same immutable snapshot contents for that
identity; retry does not hash or compare tensor bytes. Changed placements use
separate payload keys and cannot commit over an existing partial upload or
published manifest. No additional plan record is persisted; transaction decision
retirement retains its existing lifecycle boundary.
Uploads started by older writers with random payload keys must be completed by
their original handles or aborted and cleaned up before using this retry path.

For HA deployments using the etcd batch OpLog,
`weight_management_oplog_capability_confirmed` continues to gate ordinary
Weight metadata OpTypes 8–11. `weight_upsert` additionally requires
`weight_lineage_oplog_capability_confirmed=true` for lineage OpType 12. Upgrade
all standbys to replay OpType 12 before enabling the lineage flag on the active
configuration.

### Public API Migration

This release removes the public `*_with_parallelism` API family and its helper
types `ParallelAxis`, `TensorParallelism`, and `ReadTarget`. This is a breaking
change for applications using the former multi-axis tensor API. Weight snapshot
writers now use the explicit lifecycle:

```python
writer = WeightStore(store).weight_put(descriptor, adapter)
writer.weight_put_tensor(tensor_id, tensor)
identity = writer.identity
manifest = writer.commit()
```

Generation replacement uses the same writer contract:

```python
from mooncake.reshard.weight import WeightUpsertMode

weight_store = WeightStore(store)
base_view = weight_store.weight_get_metadata(identity)
writer = weight_store.weight_upsert(
    next_descriptor,
    adapter,
    replacing=identity,
    expected_metadata_generation=base_view.metadata.metadata_generation,
    mode=WeightUpsertMode.PUT_FIRST,
    request_id="rollout-2026-09-11-001",
)
writer.weight_put_tensor(tensor_id, tensor)
next_identity = writer.identity
manifest = writer.commit()
```

The next descriptor must preserve the base lineage and increase
`weight_generation`. An adapter may materialize the COO sparse updates in
[#3747](https://github.com/kvcache-ai/Mooncake/issues/3747) or
[#3953](https://github.com/kvcache-ai/Mooncake/issues/3953) as a new complete
generation before calling `weight_upsert`. A target that directly depends on
base payloads needs a separate delta-retention and compaction design.

The snapshot writer stores manifest-managed payload fragments and publishes one
`StoredWeightManifest`; it does not produce an ordinary Store tensor object.
Weight snapshot readers use `WeightStore.weight_get()` with the exact revision
identity, target placement, and runtime binding manifests. The existing
single-axis TP APIs named `*_with_tp` remain separate compatibility APIs.

If `commit()` reports a manifest publication failure after the Store records
the commit decision, the writer stays open and preserves its uploaded payloads.
Retry `commit()` on the same writer to complete publication.

`WeightStore.weight_get()` resolves the stored manifest from revision metadata
and holds a revision lease while the target framework supplies the placement,
runtime binding, and allocation guards required by `get_into_ranges`.

## Module Responsibilities

- `types.py` defines tensor and logical-fragment contracts.
- `topology.py` defines parallel sizes and selected participants.
- `part.py` defines one participant's address-free contribution.
- `placement.py` assembles and identifies a complete global placement.
- `runtime.py` defines typed physical-binding input contracts for later phases.
- `validation.py` validates manifest-level geometry, coverage, aliases, and
  runtime-binding shape contracts.
- `_planner/contracts.py` computes N-D logical overlap regions and logical
  coverage.
- `_planner/binding.py`, `bound_contracts.py`, and `bound_validation.py` bind
  a logical plan to typed runtime snapshots and validate physical evidence.
- `planner.py` exposes both logical planning and runtime-binding APIs.
- `manifest.py` preserves the public import surface.

## KV Cache Placement Model

`KVCachePlacementManifest` describes an address-free KV-cache placement over a
selected topology. Each participant contributes a `KVCachePlacementPart`, and
`KVCacheRuntimeBindingManifest` supplies operation-scoped live buffers. Runtime
bindings contain no lease or eviction state; the framework owns pinning and
lifetime. `KVCacheSnapshotDescriptor` independently describes model and token
semantics. Legacy logical planning permits omission for compatibility; the
Runtime-to-Runtime executor requires an explicit snapshot and operation ID.

The KV-cache implementation is split by responsibility:

- `snapshot.py` defines content identity;
- `topology.py`, `part.py`, and `placement.py` define logical placement;
- `runtime.py` and `binding.py` define and validate live buffer bindings;
- `planner.py` defines semantically validated logical and prepared plans;
- `resolved.py` and `transfer.py` validate and lower bounded resolved ranges;
- `executor.py` and `completion.py` execute TE writes and collect all participants;
- `runtime_serde.py` serializes resolved bindings, operations, and receipts;
- `serde.py`, `snapshot_serde.py`, and `plan_serde.py` define strict JSON
  boundaries.

KV planning is source/target-role agnostic. A placement may expose one or more
complete DP replicas, and each local-target plan selects exactly one source DP
replica. Callers may choose that source rank explicitly; otherwise the planner
maps target DP ranks round-robin over the source placement's available DP ranks.

`weight_placement_to_json` and `weight_placement_from_json` are the explicit
public JSON APIs. Their wire format contains only canonical fields, and
deserialization rejects alternate field names rather than translating
framework-specific input.

Run the reshard tests from the repository root:

```bash
PYTHONPATH=mooncake-reshard/python \
python3 -m pytest -q mooncake-reshard/tests

npx --yes pyright --project mooncake-reshard/pyrightconfig.json
```
