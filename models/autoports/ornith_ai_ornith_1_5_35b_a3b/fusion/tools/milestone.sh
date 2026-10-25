#!/bin/bash
# milestone.sh <label> <container> <image> <profile> [docker env args...]: boot a server (serve.sh; MODELS / DEVS / PORT
# as there), then greedy + garble gate + the bench rows in ROWS (default: 128/128 at 1 / 8 / 16 / 32 users and the 1-user
# TTFT rows), results in runs/<label>/; the server is removed at the end unless KEEP=1. Runs one server at a time: two
# servers benched side by side share the host CPU and skew each other.
set -u
L=$1; C=$2; IMG=$3; P=$4; shift 4
T=$(cd "$(dirname "$0")" && pwd); METAL=$(cd "$T/../../../../.." && pwd)
RUNS=${RUNS:-$HOME/ornith-fusion/runs}; mkdir -p $RUNS/$L
PORT=${PORT:-20000}
"$T/serve.sh" $C $IMG $P "$@" > /dev/null
t0=$(date +%s)
until curl -sf localhost:$PORT/v1/models >/dev/null; do
  if ! docker ps --format '{{.Names}}' | grep -qx $C; then echo "$L: server died"; docker logs $C 2>&1 | grep -v DEBUG | grep -v riscv | tail -n 300 > $RUNS/$L/boot_fail.log; exit 1; fi
  sleep 10
done
echo "$L: up after $(( $(date +%s) - t0 )) s"
docker logs $C 2>&1 | grep -E "decode bucket widths|MTP|program cache entries after warm-up|KV" | grep -v DEBUG | tail -n 8 > $RUNS/$L/boot.log
URL=http://localhost:$PORT RUNS=$RUNS python3 "$T/greedy.py" run $L > /dev/null 2>&1 && echo "$L: greedy done"
if [ "${GARBLE:-1}" = 1 ]; then
  (cd $METAL && QWEN35_QUALITY_PROMPT_DIR=$METAL/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/results/quality_prompts \
    python3 -m models.autoports.ornith_ai_ornith_1_5_35b_a3b.tools.longctx_quality --url http://127.0.0.1:$PORT --model ${MODEL_ID:-ornith-ai/Ornith-1.5-35B-A3B} --tag $L \
    --settings t0.6_k20 --out $RUNS/$L/garble.json 2>&1 | grep -A2 "=== $L" | tail -n 1)
fi
ROWS=${ROWS:-"128:128:1:16 128:128:8:32 128:128:16:64 128:128:32:128 2048:128:1:8 8192:64:1:4"}
RUNS=$RUNS "$T/bench.sh" $C $L $ROWS 2>&1 | grep -E "==|TTFT|TPOT|Output token"
[ "${KEEP:-0}" = 1 ] || docker rm -f $C > /dev/null
