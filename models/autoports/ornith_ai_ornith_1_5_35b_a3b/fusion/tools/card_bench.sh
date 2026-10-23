#!/bin/bash
# card_bench.sh <label> <bundle> <profile> [rows...]: the model card's measurement for one bundle + profile, served
# exactly as shipped (`tt-model serve <bundle> --profile <profile> --detach`, port 20000): the published card's rows
# (batch-1 latency ladder, 128/128 and 1024/256 concurrency ladders; random-token prompts, ignore_eos), then the
# real-text single-user row (realtext.py); GREEDY=1 first runs greedy.py; EVALS="<tasks>" runs eval_coder.sh last. Results in runs/<label>/. One server at a time on an otherwise idle box.
set -u
L=$1; B=$2; P=$3; shift 3
T=$(cd "$(dirname "$0")" && pwd)
RUNS=${RUNS:-$HOME/ornith-fusion/runs}; mkdir -p $RUNS/$L
NAME=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['name'])" "$B" 2>/dev/null || tt-model info "$B" 2>/dev/null | python3 -c "import sys,re; print(re.search(r'\"name\": \"([^\"]+)\"', sys.stdin.read()).group(1))")
C=tt-model-$NAME-$P
docker rm -f $C >/dev/null 2>&1
tt-model serve "$B" --profile $P --detach --port 20000 --no-update-check > $RUNS/$L/serve.log 2>&1
t0=$(date +%s)
until curl -sf localhost:20000/v1/models >/dev/null; do
  docker ps --format '{{.Names}}' | grep -qx $C || { echo "$L: server died"; docker logs $C 2>&1 | grep -v DEBUG | tail -n 30 > $RUNS/$L/boot_fail.log; exit 1; }
  sleep 10
done
echo "$L: $C up after $(( $(date +%s) - t0 )) s"
[ "${GREEDY:-0}" = 1 ] && URL=http://localhost:20000 RUNS=$RUNS python3 "$T/greedy.py" run $L > /dev/null 2>&1 && echo "$L: greedy done"
ROWS=("$@")
if [ ${#ROWS[@]} -eq 0 ]; then
  ROWS=(128:128:1:4 1024:256:1:4 4096:256:1:4 16384:256:1:2 32768:256:1:2 65536:256:1:2 130816:256:1:2
        128:128:2:8 128:128:4:16 128:128:8:32 128:128:16:64 128:128:32:128
        1024:256:2:8 1024:256:4:16 1024:256:8:32 1024:256:16:64 1024:256:32:128)
fi
RUNS=$RUNS "$T/bench.sh" $C $L "${ROWS[@]}" 2>&1 | grep -E "==|TTFT|TPOT"
URL=http://localhost:20000 python3 "$T/realtext.py" 384 | tee $RUNS/$L/realtext.txt
[ -n "${EVALS:-}" ] && PORT=20000 RUNS=$RUNS "$T/eval_coder.sh" $L $EVALS 2>&1 | grep -E "^==|\|.*(pass|exact_match|strict|loose)"
docker rm -f $C > /dev/null
