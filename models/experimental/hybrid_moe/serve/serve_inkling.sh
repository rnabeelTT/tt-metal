#!/bin/bash
# Serve thinkingmachines/Inkling-Small (text) with the hybrid-MoE overlay and native MTP speculative decoding (single
# user) on the QB2, through vLLM + the TT plugin (OpenAI-compatible API on PORT, model name inkling-small-hybrid).
#
# usage: serve/serve_inkling.sh ; logs -> serve/server_inkling.log
# env:   PROFILE  quality (default: all routed experts bfp8, 106 device slots per layer, unfused decode, host routing)
#                 fast    (routed bfp4 with layers 2-17 gate/up bfp4 + down bfp8, 192 slots, fused decode kernels)
#        CTX (max model len, default 32768), K (MTP drafts per verify pass, default 1), PORT (8000),
#        TRACE (1: verify / drafter traces), TRACE_MIB (trace region, default 400), HYB_IMAGE, WCACHE, LIB_S (106),
#        EXTRA (more docker -e flags)
# Code: this checkout's Inkling port, the M2.7 port, qwen36 and hybrid_moe are mounted over the image's copies.
P=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)   # models/experimental/hybrid_moe
REPO=$(cd "$P/../../.." && pwd)
PROFILE=${PROFILE:-quality}
CTX=${CTX:-32768}
K=${K:-1}
case $PROFILE in
  quality) POLICY=all_bfp8; S=106; PENV="-e HYB_FUSED=0 -e HYB_HOST_ROUTE=1" ;;
  # fast: little DRAM beside the experts -> 1024-row prefill launches (half the 2048-row transients)
  fast) POLICY=early_gu4dn8; S=192; PENV="-e HYB_FUSED=1 -e HYB_ROWS=$((K + 1)) -e INKLING_PREFILL_CHUNK=${PCHUNK:-1024}"
        TRACE_MIB=${TRACE_MIB:-320} ;;
  *) echo "PROFILE=$PROFILE: quality | fast"; exit 1 ;;
esac
IMG=${HYB_IMAGE:-tt-model/minimax-m2.7-hybrid-qb2:0c9857a46f16}
HF_HOST=${HF_HOST:-${HF_HOME:-/opt/tenstorrent/models}}
CACHE=${INKLING_CACHE:-$HOME/.cache/inkling}
TR_MIB=${TRACE_MIB:-400}
TT_CFG='{"tt": {"trace_mode": "all", "sample_on_device_mode": "all", "trace_region_size": '$((TR_MIB * 1048576))', "l1_small_size": 24576, "fabric_config": "FABRIC_1D"}}'
MM_LIMIT='{"image": 0, "audio": 0}'  # text only (the towers are not served yet)
PORT_DIR=$REPO/models/autoports/thinkingmachines_inkling_small
(cd "$P" && python3 -m hybrid.hostk >/dev/null)
mkdir -p "$CACHE/tt" "$CACHE/logs" "$HOME/.cache/hybrid-moe"
docker rm -f inkling-serve >/dev/null 2>&1
docker run --rm --device /dev/tenstorrent --entrypoint tt-smi "$IMG" -r >/dev/null 2>&1
docker run -d --name inkling-serve \
  --device /dev/tenstorrent --ipc host --ulimit memlock=-1:-1 \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  -v "$HF_HOST:/hf" -v "$CACHE:/cache" -v "$HOME/.cache/hybrid-moe:/hyblib" \
  -v "$PORT_DIR:/opt/tt-metal/models/autoports/thinkingmachines_inkling_small" \
  -v "$REPO/models/autoports/minimaxai_minimax_m2_7:/opt/tt-metal/models/autoports/minimaxai_minimax_m2_7" \
  -v "$REPO/models/demos/blackhole/qwen36:/opt/tt-metal/models/demos/blackhole/qwen36" \
  -v "$P:/opt/tt-metal/models/experimental/hybrid_moe" \
  -v "$PORT_DIR/vllm_bundle:/extra_models:ro" \
  --user "$(id -u):$(id -g)" -p ${PORT:-8000}:8000 \
  -e ARCH_NAME=blackhole -e HF_MODEL=thinkingmachines/Inkling-Small -e MESH_DEVICE=P300x2 -e HF_HUB_OFFLINE=1 \
  -e HOME=/tmp -e HF_HOME=/hf -e HF_HUB_CACHE=/hf/hub -e EXTRA_MODELS_DIR=/extra_models \
  -e TT_METAL_LOGGER_LEVEL=Error -e TT_METAL_CACHE=/cache/tt -e TT_METAL_LOGS_PATH=/cache/logs \
  -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  -e QWEN35_PRECISION_CONFIG=/opt/tt-metal/models/autoports/thinkingmachines_inkling_small/doc/precision/$POLICY.json \
  -e HYB_ENABLE=1 -e HYB_PORT=thinkingmachines_inkling_small -e HYB_POC_DIR=/opt/tt-metal/models/experimental/hybrid_moe \
  -e HYB_SLOTS=$S -e HYB_LIB=/hyblib/inkling_int4_s${LIB_S:-106} \
  -e HYB_PLAN=/opt/tt-metal/models/experimental/hybrid_moe/runs/inkling/plan_inkling_hyb.npz \
  -e HYB_STREAM_SLOTS=0 -e HYB_SKIP=0 -e HYB_LOWBIT=0 -e HYB_HOSTK_THREADS=${HK_THREADS:-6} -e HYB_DECODE_ROWS=8 \
  -e HYB_WCACHE=${WCACHE:-/cache/wcache} $PENV \
  -e INKLING_SPEC_K=$K -e INKLING_TRACE=${TRACE:-1} ${EXTRA} \
  -e PYTHONPATH=/opt/tt-metal/models/experimental/hybrid_moe/serve:/opt/tt-metal:/opt/tt-metal/models/experimental/hybrid_moe \
  --entrypoint bash "$IMG" -c "cd /tmp && exec vllm serve thinkingmachines/Inkling-Small --max-model-len $CTX \
    --max-num-seqs 1 --block-size 64 --additional-config '$TT_CFG' --served-model-name inkling-small-hybrid \
    --enable-auto-tool-choice --tool-call-parser inkling --reasoning-parser inkling --port 8000 \
    --limit-mm-per-prompt '$MM_LIMIT'" \
  && (docker logs -f inkling-serve > "$P/serve/server_inkling.log" 2>&1 &)
echo "started (PROFILE $PROFILE, CTX $CTX, K $K); tail -f $P/serve/server_inkling.log"
