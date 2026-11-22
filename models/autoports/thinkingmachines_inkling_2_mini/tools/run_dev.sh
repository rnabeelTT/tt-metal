#!/bin/bash
# Run a command inside the dev image against this checkout's Inkling-2-Mini port (python + JIT kernels: no rebuild).
# Mounts the port (and, MOUNT_SHARED=1 default, the shared model code it imports) over the image's copies, the HF cache
# and the device. One device job at a time (flock shared with the other ports' scripts).
#
# usage: tools/run_dev.sh <cmd...>   env: I2M_IMAGE, HF_HOST, MESH_DEVICE (P150 / P300 / P300x2 ...), NAME (container)
set -e
P=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
REPO=$(cd "$P/../../.." && pwd)
IMG=${I2M_IMAGE:-tt-model/inkling-small-hybrid-qb2:3d102efec36e}
HF_HOST=${HF_HOST:-${HF_HOME:-/opt/tenstorrent/models}}
CACHE=${I2M_CACHE:-$HOME/.cache/inkling2}
mkdir -p "$CACHE/tt" "$CACHE/logs" "$CACHE/wcache" "$HOME/.cache/hybrid-moe"
M=(-v "$P:/opt/tt-metal/models/autoports/thinkingmachines_inkling_2_mini")
if [ "${MOUNT_SHARED:-1}" = 1 ]; then
  for d in models/demos/blackhole/qwen36 models/autoports/minimaxai_minimax_m2_7 models/common/modules/moe; do
    [ -d "$REPO/$d" ] && M+=(-v "$REPO/$d:/opt/tt-metal/$d")
  done
fi
# the prefetcher fix (retire exec_buf read-ahead before host fetches resume): large host -> device writes stall without
# it; not yet in the images (same mount as the inkling_small / hybrid_moe dev scripts)
M+=(-v "$REPO/tt_metal/impl/dispatch/kernels/cq_prefetch.cpp:/opt/tt-metal/tt_metal/impl/dispatch/kernels/cq_prefetch.cpp:ro")
if [ -z "${TT_LOCK_HELD:-}" ]; then exec 9>"$HOME/.cache/hybrid-moe/ttdev.lock"; flock 9; fi
TTY=(); [ -t 0 ] && TTY=(-it)
exec docker run --rm "${TTY[@]}" --name "${NAME:-i2m-dev-$$}" \
  --device /dev/tenstorrent --ipc host --ulimit memlock=-1:-1 \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  "${M[@]}" -v "$HF_HOST:/hf" -v "$CACHE:/cache" \
  --user "$(id -u):$(id -g)" -w /opt/tt-metal \
  -e HOME=/tmp -e ARCH_NAME=blackhole -e MESH_DEVICE=${MESH_DEVICE:-P150} -e HF_HOME=/hf -e HF_HUB_CACHE=/hf/hub \
  -e HF_HUB_OFFLINE=1 -e TT_METAL_CACHE=/cache/tt -e TT_METAL_LOGS_PATH=/cache/logs -e I2M_WCACHE=/cache/wcache \
  -e TT_METAL_LOGGER_LEVEL=${TT_METAL_LOGGER_LEVEL:-Error} -e PYTHONPATH=/opt/tt-metal -e TP=${TP:-1} \
  ${DEV_ENV:+$DEV_ENV} --entrypoint "" "$IMG" "$@"
