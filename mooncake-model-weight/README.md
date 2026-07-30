# Mooncake Model Weight

This module owns Mooncake's framework-neutral model weight metadata,
heterogeneous conversion, and weight-reuse logic.

The Python API remains under `mooncake.model_weight`. Source files live in
`python/mooncake/model_weight`, and module-specific tests live in `tests`.
The main Mooncake wheel stages this package during wheel construction; the
tracked source of truth remains in this directory.

The manifest implementation is split by responsibility:

- `types.py` defines tensor and fragment contracts;
- `placement.py` owns address-free serialization and identity;
- `runtime.py` imports framework inventories and defines runtime snapshots;
- `validation.py` checks logical geometry, aliases, and physical ranges;
- `binding.py` projects and binds placement and runtime state;
- `manifest.py` preserves the stable public import surface.

Run the manifest contract tests from the repository root:

```bash
PYTHONPATH=mooncake-model-weight/python \
python -m pytest -q mooncake-model-weight/tests
```
