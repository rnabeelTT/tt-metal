#!/bin/bash
# serve.sh <name> <image> <profile:p300|p300x2|p150> [extra docker -e args...]
# The published raahemnabeel/ornith-1.5-35b-a3b launch command (tt-model serve --print), with the weights mounted from
# ~/.cache/huggingface and thinking off by default (the scorecard's eval setting).
# PORT (default 20000) / DEVS (TT_VISIBLE_DEVICES, default 0,1 for p300): two p300 servers side by side on the two chip
# pairs, e.g. the published reference on 0,1 :20000 and a development build on DEVS=2,3 PORT=20001.
# MODELS=1 mounts this checkout's models/ over the image's (Python + generic_op kernel changes without a rebuild).
# MODEL_ID (default Ornith) picks the serving recipe (Qwen/Qwen3-Coder-Next too); CACHE_NAME the host cache dir.
set -e
NAME=$1; IMAGE=$2; PROFILE=$3; shift 3
MODEL_ID=${MODEL_ID:-ornith-ai/Ornith-1.5-35B-A3B}
CACHE=$HOME/.cache/tt-model/${CACHE_NAME:-ornith-1.5-35b-a3b-v2}
case $PROFILE in
  p300)   MESH=P300;   VIS="--env TT_VISIBLE_DEVICES=${DEVS:-0,1}"; FAB=FABRIC_1D; QV=1; MM='{"image": 4, "video": 0}';;
  p300x2) MESH=P300x2; VIS="";                             FAB=FABRIC_1D; QV=1; MM='{"image": 4, "video": 0}';;
  p150)   MESH=P150;   VIS="--env TT_VISIBLE_DEVICES=0";   FAB=DISABLED;  QV=0; MM='{"image": 0, "video": 0}';;
esac
case $MODEL_ID in  # each bundle's serving recipe (its tt-model.yaml)
  ornith-ai/*) MENV="--env QWEN35_BATCHED_PREFILL=0 --env QWEN35_VISION=$QV --env QWEN35_VISION_MAX_IMAGES=4 --env QWEN35_VISION_MAX_PIXELS=4194304"
    MARGS=(--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning_parser qwen3
           --default-chat-template-kwargs '{"enable_thinking": false}' --limit-mm-per-prompt "$MM");;
  Qwen/Qwen3-Coder-Next) MENV="--env QWEN35_CHUNKED_PREFILL=1"
    MARGS=(--enable-auto-tool-choice --tool-call-parser qwen3_coder --enable-chunked-prefill --max-num-batched-tokens 2048);;
  *) echo "serve.sh: no recipe for $MODEL_ID"; exit 1;;
esac
KCACHE=${KCACHE:-$CACHE/cache}
METAL=${METAL:-$(cd "$(dirname "$0")/../../../../.." && pwd)}
MNT=""; [ "${MODELS:-0}" = 1 ] && MNT="--volume $METAL/models:/opt/tt-metal/models:ro"
mkdir -p $KCACHE $CACHE/weights $CACHE/tensors
docker rm -f $NAME >/dev/null 2>&1 || true
docker run -d --name $NAME --user $(id -u):$(id -g) --device /dev/tenstorrent --ipc host \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --volume $HOME/.cache/huggingface:/hf --env HF_HOME=/hf --env HF_HUB_OFFLINE=1 \
  --volume $KCACHE:/cache --env TT_METAL_CACHE=/cache \
  --volume $CACHE/weights:/weight-cache --env TT_DIT_CACHE_DIR=/weight-cache \
  --volume $CACHE/tensors:/tensor-cache --env TT_CACHE_PATH=/tensor-cache \
  $MNT --publish ${PORT:-20000}:20000 --env ARCH_NAME=blackhole --env HF_MODEL=$MODEL_ID --env MESH_DEVICE=$MESH \
  --env QWEN35_PROGRAM_GUARD=1 --env QWEN35_TRACE_REGION_MIB=768 $VIS --env VLLM_ENGINE_READY_TIMEOUT_S=3600 $MENV \
  "$@" $IMAGE vllm serve $MODEL_ID --max-model-len 262144 --max-num-seqs 32 --block-size 64 \
  --additional-config "{\"tt\": {\"sample_on_device_mode\": \"all\", \"trace_region_size\": 805306368, \"l1_small_size\": 24576, \"fabric_config\": \"$FAB\"}}" \
  "${MARGS[@]}" --port 20000
