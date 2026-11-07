#!/bin/bash
# usage: run_m27_dflash.sh <out dir> [extra docker -e args...]   (traced DFlash on the hybrid MiniMax-M2.7)
# env: DK (drafts per pass, default 3), CTX (default 131072), FULL (HYB_FULL_LAYERS count, default 0), PROMPTS (3), NEW (128)
O=$1; shift
cd "$(dirname "$(readlink -f "$0")")"
FL=""
if [ "${FULL:-0}" != 0 ]; then
  FL=$(python3 sim_full_layers.py runs/m27/plan_m27.npz runs/m27/ref_train/routing.npz runs/m27/ref_held/routing.npz "$FULL" | tail -1)
fi
docker run --rm --device /dev/tenstorrent --entrypoint tt-smi tt-model/qwen3-5-122b-a10b:8a750811ffec -r >/dev/null 2>&1
FORK=1 HF_MODEL=MiniMaxAI/MiniMax-M2.7 timeout 4000 ./run_in_image.sh run_dflash_traced.py "$O" \
  -e HF_HUB=/hf/hub -e HF_HOME=/hf -e HF_HUB_CACHE=/hf/hub -e HYB_PORT=minimaxai_minimax_m2_7 -e HYB_ENABLE=1 \
  -e HYB_SLOTS=192 -e HYB_DYN_SLOTS=0 -e HYB_SKIP=0 -e HYB_MTP=0 -e HYB_ROWS=$((${DK:-3} + 1)) -e DFLASH_K=${DK:-3} -e HYB_THREADS=12 \
  -e HYB_LIB=${LIB:-/hyblib/minimax_m27_int4} -e HYB_PLAN=/poc/runs/m27/plan_m27_routing.npz -e HYB_PLAN_WEIGHT=mass \
  -e POC_MAX_SEQ=${CTX:-131072} -e PYTHONFAULTHANDLER=1 -e POC_PROMPTS=${PROMPTS:-3} -e POC_NEW_TOKENS=${NEW:-128} \
  -e DFLASH_TP=1 -e HYB_FULL_LAYERS=$FL "$@"
docker rm -f hybrid-moe-poc >/dev/null 2>&1
grep -E "^(DFLASHT|MEMONLY|host library)" "$O/run.log" | cut -c1-1100
grep -v "riscv-tt-elf\|warning:\|^ *[0-9]* |\|^ *|" "$O/run.log" | grep -B2 -A14 "Traceback\|TT_FATAL" | grep -v "^ --- " | tail -20 | cut -c1-300
