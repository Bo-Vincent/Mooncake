# Explicit same-fleet weight reshard

Run from the repository root with the Reshard Python package available:

```sh
PYTHONPATH=mooncake-reshard/python python mooncake-reshard/examples/inplace_weight_reshard.py
MC_RESHARD_PLAN_CACHE_ENABLED=1 MC_RESHARD_PLAN_CACHE_MAX_ENTRIES=8 \
  PYTHONPATH=mooncake-reshard/python python mooncake-reshard/examples/inplace_weight_reshard.py
```

The example executes the new `InplaceWeightResharder.execute` public path using
canonical typed placements, fresh runtime bindings, the existing binder, and the
existing `MooncakeTransferEngineReader`. Two existing participants change from
TP2 to DP2. Both targets are checked byte-for-byte against `b"abcdefgh"`.
The ctypes engine copies CPU bytes synchronously; its registration methods are
CPU demonstration stubs. This is functional CPU evidence, not a GPU, RDMA,
NVLink, SGLang scheduler, or performance qualification.

The framework owner must stop source writes, clear KV state, retain source
allocations until global completion, provision target space, and ensure targets
do not overwrite any source data still needed by a reader. Same-fleet execution
can require scratch/target allocations: it does not mean zero-scratch overwrite.
The example deliberately uses separate source and target buffers. Production
allocation guards must actually pin framework allocator objects; CPU example
pins retain Python buffer owners only.

The entry point requires identical participant ID sets across complete source
and target placements. It does not infer physical GPU identity from opaque
`device` strings. Known overlapping address ranges sharing runtime instance,
worker and device identity through the existing binder are rejected. Other aliases and global
non-overlap remain the framework owner's responsibility. Local-target calls do
not implement a global topology-switch transaction or a global completion fence.

Pass a `WeightTransferPlanCache` explicitly to control caching without reading
environment variables. Otherwise construction reads the cache environment once.
Only address-free logical plans are cached; every call binds current allocations,
leases and generations again. The existing reader retains allocation guards,
registration checks and unknown-completion quarantine. Inspect/drain pending
transfers through the supplied reader. A failure does not trigger checkpoint
fallback or make a quarantined allocation safe to reuse. Existing entry points
retain their behavior and do not read these cache settings.
