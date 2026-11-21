#!/bin/bash
# Serve tencent/Hy4-flash (dev) on the QB2: the tt-model image with this checkout's models/ mounted over the image's.
# usage: serve/serve_hy4.sh ; logs -> serve/server_hy4.log. env: CTX (default 32768), SPEC (drafts per verify pass, 0 =
# plain decode, default 4; NSEQS > 1: default 0, k > 0 drafts while one request decodes alone), NSEQS (concurrent users, default 1; 2..8 -> the batched engine,
# CTX <= 98304 / NSEQS), PORT (8000), IMG (default the local hy4-flash-qb2 image).
CTX=${CTX:-32768}
NSEQS=${NSEQS:-1}
[ "$NSEQS" -gt 1 ] && SPEC=${SPEC:-0}
K=ttnn/cpp/ttnn/operations/experimental/moe_sorted_dispatch/device/kernels/moe_sorted_dispatch.cpp
IMG=${IMG:-tt-model/hy4-flash-qb2:1e7f2ff5eaab}
P=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
REPO=$(cd "$P/../../.." && pwd)
TT_CFG='{"tt": {"trace_mode": "none", "sample_on_device_mode": "all", "trace_region_size": 419430400, "fabric_config": "FABRIC_1D"}}'
mkdir -p $HOME/.cache/hy4-serve/tt-cache $HOME/.cache/hy4-serve/logs
docker rm -f hy4-serve >/dev/null 2>&1
docker run --rm --device /dev/tenstorrent --entrypoint tt-smi "$IMG" -r >/dev/null 2>&1
docker run -d --name hy4-serve \
  --device /dev/tenstorrent --ipc host --ulimit memlock=-1:-1 \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  -v $HOME/.cache/huggingface:/hf -v $HOME/.cache/hybrid-moe:/weight-cache \
  -v $HOME/.cache/hy4-serve/tt-cache:/cache -v $HOME/.cache/hy4-serve/logs:/logs \
  -v $REPO/models:/opt/tt-metal/models -v $REPO/$K:/opt/tt-metal/$K:ro \
  --user "$(id -u):$(id -g)" -p ${PORT:-8000}:8000 \
  -e ARCH_NAME=blackhole -e HF_MODEL=tencent/Hy4-flash -e MESH_DEVICE=P300x2 -e HF_HUB_OFFLINE=1 \
  -e HOME=/tmp -e HF_HOME=/hf -e HF_HUB_CACHE=/hf/hub -e TT_METAL_CACHE=/cache -e TT_METAL_LOGS_PATH=/logs \
  -e TT_METAL_LOGGER_LEVEL=Error -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  -e EXTRA_MODELS_DIR=/opt/tt-metal/models/autoports/tencent_hy4_flash/vllm_bundle \
  -e HY4_TENSOR_CACHE=/weight-cache/hy4_tt -e HY4_ENGRAM_DIR=/weight-cache/hy4_engram \
  -e HY4_TRACE_REGION=419430400 -e HY4_SPEC_K=${SPEC:-4} ${EXTRA_ENV:-} \
  -e PYTHONPATH=/opt/tt-metal/models/autoports/tencent_hy4_flash/serving:/opt/tt-metal \
  --entrypoint bash "$IMG" -c "cd /tmp && exec vllm serve tencent/Hy4-flash --max-model-len $CTX \
    --max-num-seqs $NSEQS --block-size 64 --additional-config '$TT_CFG' --served-model-name hy4-flash \
    --enable-auto-tool-choice --tool-call-parser hy_v4 --reasoning-parser hy_v4 --port 8000" \
  && (docker logs -f hy4-serve > $P/serve/server_hy4.log 2>&1 &)
echo "started (CTX $CTX, SPEC ${SPEC:-4}, NSEQS $NSEQS); tail -f $P/serve/server_hy4.log"
