#!/bin/bash
# prof_cmd.sh <label> <command...>: Tracy + device profiler of an arbitrary command in the ornith-prof image (MESH, DEVS,
# PROF_ENV as run_layer_prof.sh)
L=$1; shift
METAL=${METAL:-$(cd "$(dirname "$0")/../../../../.." && pwd)}
mkdir -p ${RUNS:-$HOME/ornith-fusion/runs}/prof/$L
docker run --rm --name ornith-prof-$L --user $(id -u):$(id -g) --device /dev/tenstorrent --ipc host \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --volume $HOME/.cache/huggingface:/hf --env HF_HOME=/hf --env HF_HUB_OFFLINE=1 \
  --volume $HOME/.cache/tt-model/${CACHE_NAME:-ornith-1.5-35b-a3b-v2}/cache-gdnpoc:/cache --env TT_METAL_CACHE=/cache \
  --volume $HOME/.cache/tt-model/${CACHE_NAME:-ornith-1.5-35b-a3b-v2}/tensors:/tensor-cache --env TT_CACHE_PATH=/tensor-cache \
  --volume $METAL/models:/opt/tt-metal/models \
  --volume ${RUNS:-$HOME/ornith-fusion/runs}/prof/$L:/opt/tt-metal/generated \
  ${PROF_ENV:-} --env ARCH_NAME=blackhole --env HF_MODEL=${MODEL_ID:-ornith-ai/Ornith-1.5-35B-A3B} --env MESH_DEVICE=${MESH:-P300x2} ${DEVS:+--env TT_VISIBLE_DEVICES=$DEVS} \
  --env TT_METAL_PROFILER_MID_RUN_DUMP=1 --env PYTHONPATH=/opt/tt-metal \
  --workdir /opt/tt-metal --entrypoint bash ornith-prof -c "python -m tracy -r -v --op-support-count=20000 $*" \
  > ${RUNS:-$HOME/ornith-fusion/runs}/prof/$L.log 2>&1
echo "exit $?"
