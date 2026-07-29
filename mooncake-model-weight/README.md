# Mooncake Model Weight

This module owns Mooncake's framework-neutral model weight metadata,
heterogeneous conversion, and weight-reuse orchestration.

The Python API remains under `mooncake.model_weight`. Source files live in
`python/mooncake/model_weight`, module-specific tests live in `tests`, and
the heterogeneous reshard benchmark lives in `benchmarks`.

The manifest implementation is split by responsibility:

- `types.py` defines tensor and fragment contracts;
- `placement.py` owns address-free serialization and identity;
- `runtime.py` imports framework inventories and defines runtime snapshots;
- `validation.py` checks logical geometry, aliases, and physical ranges;
- `binding.py` projects and binds placement and runtime state;
- `storage_manifest.py` defines persistent Store fragment metadata;
- `manifest.py` preserves the stable public import surface.

The planner, Store adapter, and TE adapter consume these contracts. Their
generic transport and physical storage implementations remain in
`mooncake-transfer-engine` and `mooncake-store`.

Run the model-weight tests from the repository root:

```bash
PYTHONPATH=mooncake-model-weight/python \
python -m pytest -q mooncake-model-weight/tests
```

Run the benchmark tests with the module's benchmark package on `PYTHONPATH`:

```bash
PYTHONPATH=mooncake-model-weight/python:mooncake-model-weight \
python -m pytest -q \
  mooncake-model-weight/benchmarks/heterogeneous_weight_reshard/tests
```

The main Mooncake wheel stages the Python package during wheel construction.
The tracked source of truth remains in this directory.
