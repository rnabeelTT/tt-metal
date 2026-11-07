#!/bin/bash
# Serve MiniMax-M2.7 with the hybrid-MoE overlay and DFlash speculative decoding (single user) on the QB2.
# usage: serve/serve_m27.sh ; logs -> serve/server_m27.log. env: CTX (max model len, default 131072), FULL (layers with
# every expert on the device: 20 at 128K, 36 at 64K, 44 at 32K; default by CTX), SPEC (1 = DFlash, 0 = plain decode),
# PORT (8000), DK (drafts per pass, default 3). Profiles: CTX=131072 (FULL 20), CTX=65536 (FULL 36), CTX=32768 (FULL 44).
CTX=${CTX:-131072}
case $CTX in 131072) DF=20 ;; 65536) DF=36 ;; 32768) DF=44 ;; *) DF=20 ;; esac
FULL=${FULL:-$DF}
NS=$([ "$CTX" -gt 65536 ] && echo 64 || echo 128)  # prefill streaming slots (DRAM at 128K)
IMG=${HYB_IMAGE:-tt-model/qwen3-5-122b-a10b:8a750811ffec}
TR_MIB=${TRACE_MIB:-96}
TT_CFG='{"tt": {"trace_mode": "all", "sample_on_device_mode": "all", "trace_region_size": '$((TR_MIB * 1048576))', "l1_small_size": 24576, "fabric_config": "FABRIC_1D"}}'
P=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
cd "$P" || exit 1
FL=$(python3 sim_full_layers.py runs/m27/plan_m27.npz runs/m27/ref_train/routing.npz runs/m27/ref_held/routing.npz "$FULL" | tail -1)
SPEC_ENV=""
[ "${SPEC:-1}" = 1 ] && SPEC_ENV="-e QWEN35_EXT_SPEC=dflash_serve:build -e QWEN35_EXT_SPEC_K=${DK:-3}"
python3 -m hybrid.hostk >/dev/null && mkdir -p logs cache
docker rm -f hybrid-moe-serve >/dev/null 2>&1
docker run --rm --device /dev/tenstorrent --entrypoint tt-smi "$IMG" -r >/dev/null 2>&1
docker run -d --name hybrid-moe-serve \
  --device /dev/tenstorrent --ipc host --ulimit memlock=-1:-1 \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  -v $HOME/.cache/huggingface:/hf -v $P:/poc -v $P/cache:/cache -v $HOME/.cache/hybrid-moe:/hyblib \
  -v $HOME/FORK-tt-metal/models:/opt/tt-metal/models \
  -v $HOME/FORK-tt-metal/models/autoports/minimaxai_minimax_m2_7:/extra_models/minimaxai_minimax_m2_7:ro \
  --user "$(id -u):$(id -g)" -p ${PORT:-8000}:8000 \
  -e ARCH_NAME=blackhole -e HF_MODEL=MiniMaxAI/MiniMax-M2.7 -e MESH_DEVICE=P300x2 -e HF_HUB_OFFLINE=1 \
  -e HOME=/tmp -e HF_HOME=/hf -e HF_HUB_CACHE=/hf/hub -e HF_HUB=/hf/hub -e EXTRA_MODELS_DIR=/extra_models \
  -e TT_METAL_LOGGER_LEVEL=Error -e TT_METAL_CACHE=/cache -e TT_METAL_LOGS_PATH=/poc/logs \
  -e QWEN35_TRACE_REGION_MIB=$TR_MIB -e QWEN35_PREFIX_STATE_SLOTS=0 -e QWEN35_MAX_TOKENS_ALL_USERS=$CTX \
  -e HYB_ENABLE=1 -e HYB_PORT=minimaxai_minimax_m2_7 -e HYB_SLOTS=192 -e HYB_DYN_SLOTS=0 -e HYB_SKIP=0 -e HYB_MTP=0 \
  -e HYB_ROWS=$((${DK:-3} + 1)) -e HYB_THREADS=12 -e HYB_LIB=/hyblib/minimax_m27_int4_s152 \
  -e HYB_PLAN=/poc/runs/m27/plan_m27_routing.npz -e HYB_PLAN_WEIGHT=mass -e HYB_FULL_LAYERS=$FL \
  -e HYB_LRU=32 -e HYB_FETCH_FRAC=0.3 -e HYB_FETCH_MAX=2 -e HYB_DEV_EXPERTS=/hyblib/minimax_m27_dev_s152 \
  -e HYB_STREAM_SLOTS=$NS -e DFLASH_TP=1 $SPEC_ENV \
  -e PYTHONPATH=/poc/serve:/opt/tt-metal:/poc \
  --entrypoint bash "$IMG" -c "cd /tmp && exec vllm serve MiniMaxAI/MiniMax-M2.7 --max-model-len $CTX \
    --max-num-seqs 1 --block-size 64 --additional-config '$TT_CFG' --served-model-name minimax-m2.7-hybrid --port 8000" \
  && (docker logs -f hybrid-moe-serve > serve/server_m27.log 2>&1 &)
echo "started (CTX $CTX, $FULL full layers, SPEC ${SPEC:-1}); tail -f $P/serve/server_m27.log"
