#!/bin/bash
# Dev vLLM server for Inkling-2-Mini from this checkout (the image's vLLM 0.26 + TT plugin; the port, its bundle and the
# shared code it imports mounted over the image's copies). Detached container i2m-serve (not the stall watcher's i2m-dev / i2m-job prefixes: an idle server is silent).
#
# usage: tools/serve_dev.sh     env: MESH_DEVICE (P150 | P300 | P300x2), SEQS (max-num-seqs, 4), CTX (4096), PORT (8000),
#        I2M_EXPERT_DT (bfp8), I2M_KV_DT (bf16), TRACE_MIB (300), I2M_IMAGE, SAMPLE_DEVICE (1: decode sampling on device)
set -e
P=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
REPO=$(cd "$P/../../.." && pwd)
IMG=${I2M_IMAGE:-tt-model/inkling-small-hybrid-qb2:3d102efec36e}
HF_HOST=${HF_HOST:-${HF_HOME:-/opt/tenstorrent/models}}
CACHE=${I2M_CACHE:-$HOME/.cache/inkling2}
MESH=${MESH_DEVICE:-P150}
case $MESH in P150|P100) FAB=DISABLED ;; *) FAB=FABRIC_1D ;; esac
SD=; [ -n "$SAMPLE_DEVICE" ] && SD=', "sample_on_device_mode": "decode_only"'
TT_CFG='{"tt": {"trace_mode": "all", "trace_region_size": '$((${TRACE_MIB:-300} * 1048576))', "l1_small_size": 24576, "fabric_config": "'$FAB'"'"$SD"'}}'
M=(-v "$P:/opt/tt-metal/models/autoports/thinkingmachines_inkling_2_mini")
for d in models/demos/blackhole/qwen36 models/autoports/minimaxai_minimax_m2_7 models/common/modules/moe; do
  [ -d "$REPO/$d" ] && M+=(-v "$REPO/$d:/opt/tt-metal/$d"); done
M+=(-v "$REPO/tt_metal/impl/dispatch/kernels/cq_prefetch.cpp:/opt/tt-metal/tt_metal/impl/dispatch/kernels/cq_prefetch.cpp:ro")
mkdir -p "$CACHE/tt" "$CACHE/logs" "$CACHE/wcache" "$CACHE/generated"
docker rm -f i2m-serve >/dev/null 2>&1 || true
exec docker run -d --name i2m-serve \
  --device /dev/tenstorrent --ipc host --ulimit memlock=-1:-1 \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  "${M[@]}" -v "$HF_HOST:/hf" -v "$CACHE:/cache" -v "$CACHE/generated:/opt/tt-metal/generated" \
  -v "$P/vllm_bundle:/extra_models:ro" \
  --user "$(id -u):$(id -g)" -p ${PORT:-8000}:8000 \
  -e ARCH_NAME=blackhole -e HF_MODEL=thinkingmachines/Inkling-2-Mini -e MESH_DEVICE=$MESH -e HF_HUB_OFFLINE=1 \
  -e HOME=/tmp -e HF_HOME=/hf -e HF_HUB_CACHE=/hf/hub -e EXTRA_MODELS_DIR=/extra_models \
  -e TT_METAL_LOGGER_LEVEL=Error -e TT_METAL_CACHE=/cache/tt -e TT_METAL_LOGS_PATH=/cache/logs -e I2M_WCACHE=/cache/wcache \
  -e VLLM_ENGINE_READY_TIMEOUT_S=7200 -e I2M_EXPERT_DT=${I2M_EXPERT_DT:-bfp8} -e I2M_KV_DT=${I2M_KV_DT:-bf16} \
  -e PYTHONPATH=/opt/tt-metal ${EXTRA} \
  --entrypoint bash "$IMG" -c "cd /tmp && exec vllm serve thinkingmachines/Inkling-2-Mini \
    --hf-config-path /opt/tt-metal/models/autoports/thinkingmachines_inkling_2_mini/hf_config --trust-remote-code \
    --max-model-len ${CTX:-4096} --max-num-seqs ${SEQS:-4} --block-size 64 --additional-config '$TT_CFG' \
    --served-model-name inkling-2-mini --enable-auto-tool-choice --tool-call-parser inkling --reasoning-parser inkling \
    --port 8000"
