#!/bin/bash
# usage: run_in_image.sh <script.py> <out_dir> [extra docker -e args...]
S=$1; O=$2; shift 2
# one device job at a time (shared with every other user of these scripts): the lock file, and no served model running
if [ -z "${TT_LOCK_HELD:-}" ]; then
  exec 9>"$HOME/.cache/hybrid-moe/ttdev.lock"; flock 9
  while docker ps --format '{{.Names}}' | grep -q '^hybrid-moe-serve$'; do sleep 10; done
fi
P=$(cd "$(dirname "$(readlink -f "$0")")" && pwd); cd "$P" && mkdir -p "$O" logs cache && docker run --rm --name hybrid-moe-poc \
  --device /dev/tenstorrent --ipc host --ulimit memlock=-1:-1 \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  -v $HOME/.cache/huggingface:/hf -v $P:/poc -v $P/cache:/cache -v $HOME/.cache/hybrid-moe:/hyblib \
  ${FORK:+-v $(cd "$P/../../.." && pwd)/models:/opt/tt-metal/models} \
  --user "$(id -u):$(id -g)" -w /opt/tt-metal \
  -e ARCH_NAME=blackhole -e HF_MODEL=${HF_MODEL:-Qwen/Qwen3.5-122B-A10B} -e MESH_DEVICE=P300x2 -e HF_HUB_OFFLINE=1 \
  -e TT_METAL_LOGGER_LEVEL=Error -e HOME=/tmp \
  -e TT_METAL_CACHE=/cache -e TT_METAL_LOGS_PATH=/poc/logs -e PYTHONPATH=/opt/tt-metal:/poc "$@" \
  --entrypoint python ${HYB_IMAGE:-tt-model/qwen3-5-122b-a10b:8a750811ffec} /poc/$S /poc/$O > "$O/run.log" 2>&1
