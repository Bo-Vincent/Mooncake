#!/usr/bin/env bash
# Real TCP transport e2e for generation-aware managed-weight upsert.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
BUILD_DIR=${BUILD_DIR:-"$REPO_ROOT/build"}
LOG_DIR=${LOG_DIR:-/nvme/tmp/mooncake_weight_upsert_tcp_e2e}
MASTER_RPC=${MASTER_RPC:-127.0.0.1:50181}
MASTER_HOST=${MASTER_RPC%:*}
MASTER_PORT=${MASTER_RPC##*:}

MASTER_BIN="$BUILD_DIR/mooncake-store/src/mooncake_master"
STORE_SO=$(ls "$BUILD_DIR"/mooncake-integration/store.cpython-*.so 2>/dev/null | head -1 || true)
MASTER_PID=""

cleanup() {
  if [[ -n "$MASTER_PID" ]]; then
    kill "$MASTER_PID" >/dev/null 2>&1 || true
    wait "$MASTER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

mkdir -p "$LOG_DIR/offload"
if [[ ! -x "$MASTER_BIN" ]]; then
  echo "missing mooncake_master at $MASTER_BIN; build target mooncake_master first" >&2
  exit 2
fi
if [[ -z "$STORE_SO" ]]; then
  echo "missing store python module under $BUILD_DIR/mooncake-integration" >&2
  exit 3
fi

"$MASTER_BIN" \
  --rpc_address="$MASTER_HOST" \
  --rpc_port="$MASTER_PORT" \
  --default_kv_lease_ttl=1000 \
  --enable_offload=true \
  --offload_on_evict=true \
  --promotion_on_hit=false \
  --root_fs_dir="" \
  --enable_http_metadata_server=false \
  --logtostderr=true \
  >"$LOG_DIR/master.log" 2>&1 &
MASTER_PID=$!
sleep 2

if ! kill -0 "$MASTER_PID" >/dev/null 2>&1; then
  echo "mooncake_master failed to start; see $LOG_DIR/master.log" >&2
  exit 4
fi

export PYTHONPATH="$BUILD_DIR/mooncake-integration${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$BUILD_DIR/mooncake-store/src:$BUILD_DIR/mooncake-common:$BUILD_DIR/mooncake-transfer-engine/src${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MOONCAKE_PROTOCOL=tcp
export MOONCAKE_MASTER="$MASTER_RPC"
export MOONCAKE_TE_META_DATA_SERVER=P2PHANDSHAKE
export MOONCAKE_LOCAL_HOSTNAME=localhost:17815
export MC_FORCE_TCP="${MC_FORCE_TCP:-1}"
export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH="$LOG_DIR/offload"
export MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS=1
export MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT=1
export MOONCAKE_OFFLOAD_BUCKET_SIZE_LIMIT_BYTES=1048576

set +e
python3 "$SCRIPT_DIR/weight_upsert_tcp_e2e.py" | tee "$LOG_DIR/client.log"
RC=${PIPESTATUS[0]}
set -e

if [[ "$RC" -eq 0 ]]; then
  echo "PASSED: weight upsert TCP e2e (logs: $LOG_DIR)"
else
  echo "FAILED: weight upsert TCP e2e rc=$RC (logs: $LOG_DIR)" >&2
  tail -n 100 "$LOG_DIR/master.log" >&2 || true
fi
exit "$RC"
