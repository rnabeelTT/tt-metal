#!/bin/bash
# Run a command inside the hybrid serving image against this checkout's Inkling code (no rebuild: device kernels are
# JIT-compiled, the port is python). Mounts this port and models/experimental/hybrid_moe over the image's copies
# (MOUNT_QWEN36=1 also mounts models/demos/blackhole/qwen36), the HF cache, and the device.
#
# usage: tools/run_dev.sh <cmd...>     e.g.  tools/run_dev.sh pytest -q models/autoports/thinkingmachines_inkling_small/tests/test_sdpa_device.py
# env:   INKLING_IMAGE (default the M2.7 hybrid image), HF_HOST (default $HF_HOME or /opt/tenstorrent/models)
set -e
P=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)    # the port
REPO=$(cd "$P/../../.." && pwd)
IMG=${INKLING_IMAGE:-tt-model/minimax-m2.7-hybrid-qb2:0c9857a46f16}
HF_HOST=${HF_HOST:-${HF_HOME:-/opt/tenstorrent/models}}
CACHE=${INKLING_CACHE:-$HOME/.cache/inkling}
mkdir -p "$CACHE/tt" "$CACHE/logs" "$HOME/.cache/hybrid-moe"
M=(-v "$P:/opt/tt-metal/models/autoports/thinkingmachines_inkling_small"
   -v "$REPO/models/experimental/hybrid_moe:/opt/tt-metal/models/experimental/hybrid_moe")
[ -n "$MOUNT_QWEN36" ] && M+=(-v "$REPO/models/demos/blackhole/qwen36:/opt/tt-metal/models/demos/blackhole/qwen36")
# one device job at a time (shared with the hybrid_moe scripts)
if [ -z "${TT_LOCK_HELD:-}" ]; then exec 9>"$HOME/.cache/hybrid-moe/ttdev.lock"; flock 9; fi
TTY=(); [ -t 0 ] && TTY=(-it)
exec docker run --rm "${TTY[@]}" --name "inkling-dev-$$" \
  --device /dev/tenstorrent --ipc host --ulimit memlock=-1:-1 \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  "${M[@]}" -v "$HF_HOST:/hf" -v "$CACHE:/cache" \
  --user "$(id -u):$(id -g)" -w /opt/tt-metal \
  -e HOME=/tmp -e ARCH_NAME=blackhole -e MESH_DEVICE=P300x2 -e HF_HOME=/hf -e HF_HUB_CACHE=/hf/hub \
  -e HF_HUB_OFFLINE=1 -e TT_METAL_CACHE=/cache/tt -e TT_METAL_LOGS_PATH=/cache/logs \
  -e TT_METAL_LOGGER_LEVEL=${TT_METAL_LOGGER_LEVEL:-Error} -e PYTHONPATH=/opt/tt-metal \
  ${DEV_ENV:+$DEV_ENV} --entrypoint "" "$IMG" "$@"
