#!/usr/bin/env bash
# Verify that the public reshard package imports on the supported Python 3.9 floor.
set -euo pipefail

python_bin="${PYTHON39_BIN:-python3.9}"
if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "Python 3.9 interpreter not found: ${python_bin}" >&2
    exit 127
fi

version="$("${python_bin}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
case "${version}" in
    3.9.*) ;;
    *)
        echo "Expected Python 3.9, got ${version}" >&2
        exit 1
        ;;
esac

if ! grep -Fq "typing_extensions>=4.0; python_version < '3.10'" \
    mooncake-wheel/pyproject.toml; then
    echo "Python 3.9 reshard imports require a declared typing_extensions dependency" >&2
    exit 1
fi

PYTHONPATH="mooncake-reshard/python${PYTHONPATH:+:${PYTHONPATH}}" "${python_bin}" -c '
from mooncake.reshard.weight import (
    ExecutorTransferPlan,
    LogicalTransferPlan,
    PlacementExecutorPlan,
    TransferPlan,
    TransferRegion,
)
print("mooncake-reshard Python 3.9 import: ok")
'
