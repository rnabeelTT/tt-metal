#!/usr/bin/env bash
# bfcl_v4 <label> <base_url> <model_id> <concurrency>   (called by text2_run.sh as: bfcl_v4.sh bfcl_v4 <label> ...)
# Official bfcl_eval (gorilla BFCL v4) generation + evaluation in FC mode; harness lives in ~/evals/bfcl.
set -euo pipefail
[ "${1:-}" = bfcl_v4 ] && shift
BFCL=${BFCL_HOME:-$HOME/evals/bfcl}
[ -x "$BFCL/.venv/bin/python" ] || { echo "missing $BFCL/.venv"; exit 1; }
cd "$BFCL"
exec env PYTHONPATH="$BFCL" "$BFCL/.venv/bin/python" -u -m tt_bfcl.run "$@"
