#!/bin/bash
# Serve Qwen3.5-397B-A17B-FP8 with the hybrid-MoE overlay (eager, batch 1) on the QB2 via the 122B tt-model image.
# usage: serve/serve_397b.sh [plan.npz relative to this dir] ; logs -> serve/server.log
PLAN=${1:-}
IMG=${HYB_IMAGE:-tt-model/qwen3-5-122b-a10b:8a750811ffec}
PORT=${PORT:-8000}
TM=${TRACE_MODE:-all}
TR_MIB=${TRACE_MIB:-512}  # trace region (MiB): MTP's segmented traces (HYB_MTP=0 fits in 256)
TT_CFG='{"tt": {"trace_mode": "'$TM'", "sample_on_device_mode": "all", "trace_region_size": '$((TR_MIB * 1048576))', "l1_small_size": 24576, "fabric_config": "FABRIC_1D"}}'
P=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd); cd "$P" && mkdir -p logs cache && docker run -d --name hybrid-moe-serve \
  --device /dev/tenstorrent --ipc host \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  -v $HOME/.cache/huggingface:/hf -v $P:/poc -v $P/cache:/cache \
  -v $HOME/.cache/hybrid-moe:/hyblib \
  --user "$(id -u):$(id -g)" -p ${PORT}:8000 \
  -e ARCH_NAME=blackhole -e HF_MODEL=Qwen/Qwen3.5-397B-A17B-FP8 -e MESH_DEVICE=P300x2 -e HF_HUB_OFFLINE=1 \
  -e TT_METAL_LOGGER_LEVEL=Error -e TT_METAL_CACHE=/cache -e TT_METAL_LOGS_PATH=/poc/logs \
  -e QWEN35_TRACE_REGION_MIB=$TR_MIB -e QWEN35_PREFIX_STATE_SLOTS=0 -e QWEN35_MAX_TOKENS_ALL_USERS=${MAX_TOKENS:-${MAX_LEN:-8192}} \
  -e HYB_ENABLE=1 -e HYB_SLOTS=${HYB_SLOTS:-256} -e HYB_DYN_SLOTS=${HYB_DYN_SLOTS:-0} -e HYB_SKIP=${HYB_SKIP:-0.1} -e HYB_MTP=${HYB_MTP:-3} -e HYB_LOG_EVERY=${HYB_LOG_EVERY:-0} -e HYB_THREADS=${HYB_THREADS:-12} \
  -e HYB_LIB=/hyblib/qwen35_397b_int4 ${PLAN:+-e HYB_PLAN=/poc/$PLAN} \
  -e PYTHONPATH=/poc/serve:/opt/tt-metal:/poc \
  --entrypoint bash $IMG -c "cd /tmp && exec vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len ${MAX_LEN:-8192} \
    --max-num-seqs 1 --block-size 64 --additional-config '$TT_CFG' --reasoning_parser qwen3 \
    --default-chat-template-kwargs '{\"enable_thinking\": false}' --served-model-name qwen3.5-397b-hybrid --port 8000" \
  && docker logs -f hybrid-moe-serve > serve/server.log 2>&1 &
echo "started; tail -f $P/serve/server.log"
