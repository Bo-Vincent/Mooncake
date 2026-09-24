# Reshard Bounded In-Flight Transfer Design

## Goal

Allow model-weight reshard execution to keep a bounded number of Transfer
Engine Scatter operations in flight across batches and endpoints while
preserving the existing fail-closed registration and allocation lifetime.

## Design

`TransferEngine::submitScatter()` is already asynchronous. The current Python
binding becomes synchronous because it drains the returned operation before
returning its ticket. Add `submit_scatter_read()` and
`submit_scatter_write()` bindings that return the same ticket without draining
it. Keep the existing synchronous methods unchanged.

The resource-neutral reshard executor accepts `max_inflight_batches`. The
default is `1`, which uses the existing synchronous behavior. Values greater
than `1` require the non-draining Scatter API. The executor submits batches in
deterministic order, drains the oldest ticket when the window is full, and
returns receipts in submission order.

Reader and sink build one batch stream across all selected endpoints, allowing
the window to aggregate work across peers and NICs. Their existing outer
registration and allocation guards remain active until every submitted ticket
reaches a known terminal state.

## Failure Semantics

- A known failed ticket stops further submissions and drains all published
  tickets before returning `FAILED_DRAINED`.
- An unknown or interrupted completion retains every unresolved ticket in one
  composite ticket and hands the active registration frame to the existing
  pending-transfer manager.
- A submit failure without a ticket is completion-unknown and restart-required.
- Explicit drain of a composite pending ticket releases resources only after
  every child ticket reaches a known terminal state.
- The in-flight count never exceeds `max_inflight_batches`.

## Compatibility

Existing synchronous Scatter APIs and the default reshard execution path keep
their current behavior. Engines without the non-draining API continue to work
with `max_inflight_batches=1`; requesting a larger window fails before any
physical transfer is submitted.

## Validation

- Python RED/GREEN tests for window bounds, multi-endpoint overlap, ordered
  receipts, known failure drain, unknown completion quarantine, interruption,
  and missing native capability.
- Native binding tests for submit-without-drain and existing synchronous API
  compatibility.
- Exact-head reshard tests, Pyright, formatting, and documentation build.
- Native TCP and RDMA/GDR smoke where the available host supports them.
- Same-host A/B benchmark for window sizes 1, 2, 4, and 8, reporting throughput
  and end-to-end time.
