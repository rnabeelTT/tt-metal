#!/bin/bash
# usage: run_prefill_bench.sh <out dir> [extra docker -e args...]   (bench_prefill.py on the hybrid MiniMax-M2.7)
# Same hybrid settings as serve/serve_m27.sh. env: CTX (131072 / 65536 / 32768 picks FULL 20 / 36 / 44), FULL, LENS,
# REPEAT, LAYERS (QWEN35_LAYER_INDICES: a reduced model for development, e.g. 0,1,2,3).
O=$1; shift
cd "$(dirname "$(readlink -f "$0")")" || exit 1
CTX=${CTX:-131072}
case $CTX in 131072) DF=20 ;; 65536) DF=36 ;; 32768) DF=44 ;; *) DF=20 ;; esac
FULL=${FULL:-$DF}
NS=${NS:-128}  # staging slots, as serve/serve_m27.sh
FL=$(python3 sim_full_layers.py runs/m27/plan_m27.npz runs/m27/ref_train/routing.npz runs/m27/ref_held/routing.npz "$FULL" | tail -1)
python3 -m hybrid.hostk >/dev/null
docker run --rm --device /dev/tenstorrent --entrypoint tt-smi tt-model/qwen3-5-122b-a10b:8a750811ffec -r >/dev/null 2>&1
FORK=1 HF_MODEL=MiniMaxAI/MiniMax-M2.7 timeout 5400 ./run_in_image.sh bench_prefill.py "$O" \
  -e HF_HUB=/hf/hub -e HF_HOME=/hf -e HF_HUB_CACHE=/hf/hub -e HYB_PORT=minimaxai_minimax_m2_7 -e HYB_ENABLE=1 \
  -e HYB_SLOTS=192 -e HYB_DYN_SLOTS=0 -e HYB_SKIP=0 -e HYB_MTP=0 -e HYB_ROWS=4 -e HYB_THREADS=12 \
  -e HYB_LIB=/hyblib/minimax_m27_int4_s152 -e HYB_PLAN=/poc/runs/m27/plan_m27_routing.npz -e HYB_PLAN_WEIGHT=mass \
  -e HYB_FULL_LAYERS="$FL" -e HYB_LRU=32 -e HYB_FETCH_FRAC=0.3 -e HYB_FETCH_MAX=2 \
  -e HYB_DEV_EXPERTS=/hyblib/minimax_m27_dev_s152 -e HYB_STREAM_SLOTS=$NS -e QWEN35_PREFIX_STATE_SLOTS=0 \
  -e POC_MAX_SEQ=$CTX -e QWEN35_MAX_TOKENS_ALL_USERS=$CTX -e BENCH_LENS=${LENS:-512,2048,8192} \
  -e QWEN35_PREFILL_SEGMENT=4096 -e QWEN35_PREFILL_GROUP=${PF_GROUP:-8} -e HYB_STREAM_ROUNDS=3 -e HYB_STREAM_MIN=1 \
  -e HYB_STREAM_CQ1=1 -e BENCH_REPEAT=${REPEAT:-2} ${LAYERS:+-e QWEN35_LAYER_INDICES=$LAYERS} -e PYTHONFAULTHANDLER=1 "$@"
docker rm -f hybrid-moe-poc >/dev/null 2>&1
grep -E "^(PREFILL|BENCH)" "$O/run.log" | cut -c1-900
grep -v "riscv-tt-elf\|warning:" "$O/run.log" | grep -B2 -A14 "Traceback\|TT_FATAL" | tail -30 | cut -c1-300
