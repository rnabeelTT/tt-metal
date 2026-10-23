#!/bin/bash
# bench.sh <container> <label> [rows...]   rows: "isl:osl:users:N" (default: the scorecard's batch-1 ladder + the 128/128
# concurrency ladder). Runs `vllm bench serve` inside the serving container (random-token prompts of exactly the input
# length, ignore_eos, one warm-up per row), copies the JSON results to runs/<label>/.
set -e
MODEL_ID=${MODEL_ID:-ornith-ai/Ornith-1.5-35B-A3B}
C=$1; L=$2; shift 2
OUT=${RUNS:-$HOME/ornith-fusion/runs}/$L
mkdir -p $OUT
ROWS=("$@")
if [ ${#ROWS[@]} -eq 0 ]; then
  ROWS=(128:128:1:4 1024:256:1:4 4096:256:1:4 16384:256:1:2 32768:256:1:2 65536:256:1:2
        128:128:2:8 128:128:4:16 128:128:8:32 128:128:16:64 128:128:32:128)
fi
for r in "${ROWS[@]}"; do
  IFS=: read ISL OSL U N <<< "$r"
  F=bench_${ISL}_${OSL}_u${U}.json
  echo "== $L isl=$ISL osl=$OSL users=$U N=$N $(date +%T)"
  docker exec --env HF_HUB_OFFLINE=1 $C vllm bench serve --backend openai --base-url http://localhost:20000 --endpoint /v1/completions \
    --model $MODEL_ID --tokenizer $MODEL_ID \
    --dataset-name random --random-input-len $ISL --random-output-len $OSL --random-range-ratio 0 \
    --num-prompts $N --max-concurrency $U --ignore-eos --num-warmups 1 --seed 42 \
    --percentile-metrics ttft,tpot,itl,e2el --save-result --result-dir /tmp/bench --result-filename $F 2>&1 \
    | grep -E 'Median (TTFT|TPOT)|Output token throughput|Successful requests' || true
  docker cp $C:/tmp/bench/$F $OUT/$F >/dev/null
done
