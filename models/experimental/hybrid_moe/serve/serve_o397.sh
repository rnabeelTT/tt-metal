#!/bin/bash
# Serve Ornith-1.5-397B with the hybrid-MoE overlay and DFlash speculative decoding (single user) on the QB2.
# (from templates/serve_model.sh = serve_m27.sh; no elastic: Qwen3.5's KV is small, one 128K layout)
# usage: serve/serve_m27.sh ; logs -> serve/server_o397.log. env: CTX (max model len, default 131072), FULL (layers with
# every expert on the device: 20 at 128K, 36 at 64K, 44 at 32K; default by CTX), SPEC (1 = DFlash, 0 = plain decode),
# ROWS_SDPA_FROM (shared-KV rows SDPA from this context length; default 0 = off: its program depends on the KV pool
# shape, which elastic resizing changes, and a new variant cannot compile inside the verify capture), PORT (8000), DK (drafts per pass, default 3), PC (prefix caching, default 1), PF_PREFETCH (prompt tokens from which
# prefill prefetches the next layer's experts, default 1024; 0 = off), NS (staging slots, default 212), ELASTIC (default 1: elastic 128K, below). Profiles: CTX=131072 (FULL 20), CTX=65536 (FULL 36), CTX=32768 (FULL 44).
# Tool calls: minimax_m2 tool parser. Reasoning: minimax_m2 splits it into reasoning_content (the chat template opens the
# turn with <think>, so the output has only </think>); minimax_m2_append_think instead leaves <think>...</think> in content.
CTX=${CTX:-131072}
DF=0  # full layers (all 512 experts on the device); tuned from runs/o397/full_layers.txt
FULL=${FULL:-$DF}
NS=${NS:-212}  # prefill streaming slots: two banks of 106 (a hybrid layer has 104 non-resident experts in prefill)
# prefill: 4096-token segments in layer-major groups of 8 (a layer's staged expert copies serve the whole group), up
# to 3 staging rounds per layer step from 1 token on (no CPU experts in prefill), copies on the second command queue;
# from PF_PREFETCH tokens on, the next hybrid layer's experts are copied during the current one (0 = off)
PF_ENV="-e QWEN35_PREFILL_SEGMENT=${PF_SEG:-2048} -e QWEN35_PREFILL_GROUP=${PF_GROUP:-8} -e HYB_STREAM_ROUNDS=3 -e HYB_STREAM_MIN=${PF_MIN:-4}"
# prompts of 1024+ tokens stream every missed expert (runs/m27/pf_min: 16K 8.5 -> 6.5 s, 4K 2.34 -> 2.0 s)
PF_ENV="$PF_ENV -e HYB_STREAM_MIN_LONG=1 -e HYB_STREAM_LONG_FROM=1024"
PF_ENV="$PF_ENV -e HYB_STREAM_CQ1=1 -e HYB_PF_PREFETCH_MIN=${PF_PREFETCH:-1024}"
# vLLM automatic prefix caching (PC=0 turns it off): a resent prefix resumes at its last 4096-token boundary
PF_ENV="$PF_ENV -e QWEN35_PREFIX_CACHING=${PC:-0}"  # the M2.7 scheme refuses GDN models
# Qwen3.5 port gaps (ADD_A_MODEL / HANDOFF): no prefill streaming hooks yet, no zeroed sorted-down output, text only
PF_ENV="$PF_ENV -e QWEN35_PREFILL_MAX_ROWS=${PF_MAX_ROWS:-1024} -e HYB_STREAM=${STREAM:-0} -e HYB_PREFILL_SENTINEL=0 -e QWEN35_VISION=0 -e QWEN35_BATCHED_PREFILL=0"
# rows-mode MoE down op: 4 readers per bank, 4-slot chunks (bench_moe_rows.py: 137 vs 157 us per call)
PF_ENV="$PF_ENV -e QWEN35_MOE_DN_READERS=4 -e QWEN35_MOE_DN_CHUNK=4"
# DFlash drafter context window 256 (8K prompt: 1.79 -> 1.91-2.08 tokens per pass vs 1024; runs/m27/fb_win, fb_win2)
PF_ENV="$PF_ENV -e DFLASH_WINDOW=${DWIN:-256}"
# startup warm-up generation: 384 tokens (64 left the first 3 requests' verify at 50-61 ms against ~40 steady)
PF_ENV="$PF_ENV -e DFLASH_WARMUP=${DWARM:-384}"
# single-trace decode with host <-> device sockets (HYB_SOCKET; SOCK=0: per-layer segments); in socket mode 6 host-kernel
# threads and fetch fraction 0.15 (runs/m27/fb_sock, fb_sock2: short 44.7 -> 48.4 tok/s, 8K 31.9 -> 37.0, fixed text)
if [ "${SOCK:-1}" = 1 ]; then
  PF_ENV="$PF_ENV -e HYB_SOCKET=1 -e HYB_HOSTK_THREADS=${HK_THREADS:-6}"; FFRAC=${FFRAC:-0.15}
  # predictive L3 prefetch of the next layer's CPU experts (runs/m27/fb_pf2: short 48.6 -> 52.5 tok/s, 8K 36.5 -> 37.7)
  PF_ENV="$PF_ENV -e HYB_PREFETCH=${PREFETCH:-1} -e HYB_PF_TOPK=${PF_TOPK:-8}"
else
  FFRAC=${FFRAC:-0.3}
fi
IMG=${HYB_IMAGE:?set HYB_IMAGE to an image built from this branch (tt-model-o397-hybrid.yaml)}
TR_MIB=${TRACE_MIB:-96}
TT_CFG='{"tt": {"trace_mode": "all", "sample_on_device_mode": "all", "trace_region_size": '$((TR_MIB * 1048576))', "l1_small_size": 24576, "fabric_config": "FABRIC_1D"}}'
P=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
REPO=$(cd "$P/../../.." && pwd)  # the checkout this script belongs to (a pinned demo worktree serves its own code)
cd "$P" || exit 1
FL=$(python3 sim_full_layers.py runs/o397/plan_o397.npz runs/o397/ref_train/routing.npz runs/o397/ref_held/routing.npz "$FULL" | tail -1)
# expert fetch: the slack controller (HYB_FETCH_POLICY=auto: budget moved by measured pass time; runs/m27/fb_auto:
# short / 8K same as the fixed 15% / 2; el_long_auto: 40K-token context 3% faster per pass, more fetched as the device
# slows). FETCH_POLICY=fixed FETCH_MAX=2 restores the fixed fraction.
# ELASTIC (default 1 at CTX 131072; ELASTIC=0: the static 128K profile): built as the 32K profile (44 full layers, a 32K-token KV pool); a request that needs more
# context first demotes as many of the 24 layers the 128K profile keeps hybrid as it needs (4K tokens of pool each, least
# valuable first; HYB_ELASTIC_GRADUAL=0: all of them) and grows the pool (hybrid/elastic.py)
if [ "${ELASTIC:-0}" = 1 ] && [ "$CTX" = 131072 ]; then  # off for Qwen3.5 (small KV; not validated)
  FL=$(python3 sim_full_layers.py runs/o397/plan_o397.npz runs/o397/ref_train/routing.npz runs/o397/ref_held/routing.npz 44 | tail -1)
  RANK=$(SIM_ORDER=1 python3 sim_full_layers.py runs/o397/plan_o397.npz runs/o397/ref_train/routing.npz runs/o397/ref_held/routing.npz 44 | tail -1)
  DEM=$(python3 -c "r='$RANK'.split(',');print(','.join(reversed(r[20:])))")  # demotion order: least valuable first
  PF_ENV="$PF_ENV -e QWEN35_KV_TOKENS=${KV_TOKENS:-32768} -e HYB_ELASTIC_DEMOTE=$DEM"
  FULL=44
fi
SPEC_ENV=""
[ "${SPEC:-1}" = 1 ] && SPEC_ENV="-e QWEN35_EXT_SPEC=dflash_serve:build -e QWEN35_EXT_SPEC_K=${DK:-3} -e DFLASH_DRAFTER=ornith-ai/Ornith-1.5-397B-DFlash -e DFLASH_BLOCK=${DBLOCK:-8}"
python3 -m hybrid.hostk >/dev/null && mkdir -p logs cache
docker rm -f hybrid-moe-serve >/dev/null 2>&1
docker run --rm --device /dev/tenstorrent --entrypoint tt-smi "$IMG" -r >/dev/null 2>&1
docker run -d --name hybrid-moe-serve \
  --device /dev/tenstorrent --ipc host --ulimit memlock=-1:-1 \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  -v $HOME/.cache/huggingface:/hf -v $P:/poc -v $P/cache:/cache -v $HOME/.cache/hybrid-moe:/hyblib \
  -v $REPO/models:/opt/tt-metal/models \
  -v $REPO/models/autoports/qwen_qwen3_5_122b_a10b:/extra_models/qwen_qwen3_5_122b_a10b:ro \
  --user "$(id -u):$(id -g)" -p ${PORT:-8000}:8000 \
  -e ARCH_NAME=blackhole -e HF_MODEL=ornith-ai/Ornith-1.5-397B-FP8 -e MESH_DEVICE=P300x2 -e HF_HUB_OFFLINE=1 \
  -e HOME=/tmp -e HF_HOME=/hf -e HF_HUB_CACHE=/hf/hub -e HF_HUB=/hf/hub -e EXTRA_MODELS_DIR=/extra_models \
  -e TT_METAL_LOGGER_LEVEL=Error -e TT_METAL_CACHE=/cache -e TT_METAL_LOGS_PATH=/poc/logs \
  -e QWEN35_TRACE_REGION_MIB=$TR_MIB -e QWEN35_PREFIX_STATE_SLOTS=0 -e QWEN35_MAX_TOKENS_ALL_USERS=$CTX \
  -e HYB_ENABLE=1 -e HYB_PORT=qwen_qwen3_5_122b_a10b -e HYB_SLOTS=${SLOTS:-256} -e HYB_DYN_SLOTS=0 -e HYB_SKIP=0 -e HYB_MTP=0 \
  -e HYB_ROWS=$((${DK:-3} + 1)) -e HYB_THREADS=12 -e HYB_LIB=/hyblib/o397_int4_s192 \
  -e HYB_PLAN=/poc/runs/o397/plan_o397_routing.npz -e HYB_PLAN_WEIGHT=mass -e HYB_FULL_LAYERS=$FL \
  -e HYB_LRU=${LRU:-32} -e HYB_FETCH_FRAC=$FFRAC -e HYB_FETCH_POLICY=${FETCH_POLICY:-auto} -e HYB_FETCH_BUDGET=8 -e HYB_FETCH_MAX=${FETCH_MAX:-8} -e HYB_DEV_EXPERTS=${DEVX-/hyblib/o397_dev_s192} -e HYB_LIB_RESIDENT=${LIB_RESIDENT:-1} -e HYB_PIN_TOP=${PIN_TOP:-0} \
  -e HYB_WCACHE=${WCACHE:-/hyblib/wcache_o397} \
  -e HYB_STREAM_SLOTS=$NS -e DFLASH_TP=1 -e QWEN35_ROWS_SDPA_SHARED_FROM=${ROWS_SDPA_FROM:-0} $SPEC_ENV $PF_ENV \
  -e PYTHONPATH=/poc/serve:/opt/tt-metal:/poc \
  --entrypoint bash "$IMG" -c "cd /tmp && exec vllm serve ornith-ai/Ornith-1.5-397B-FP8 --max-model-len $CTX \
    --max-num-seqs 1 --block-size 64 --additional-config '$TT_CFG' --served-model-name ornith-1.5-397b-hybrid \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 --port 8000" \
  && (docker logs -f hybrid-moe-serve > serve/server_o397.log 2>&1 &)
echo "started (CTX $CTX, $FULL full layers, SPEC ${SPEC:-1}); tail -f $P/serve/server_o397.log"
