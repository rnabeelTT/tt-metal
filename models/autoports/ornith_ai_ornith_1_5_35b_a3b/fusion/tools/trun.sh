#!/bin/bash
# trun.sh <cmd...>: run a command in the Ornith serving image ($IMAGE, default the published one; ornith-dev adds pytest)
# on all chips (MESH, default P300x2), with this checkout's models/ + tests/ mounted. TENV: extra docker env args. DEVS: chip list (TT_VISIBLE_DEVICES), e.g. 2,3 with MESH=P300.
METAL=${METAL:-$(cd "$(dirname "$(readlink -f "$0")")/../../../../.." && pwd)}
IMAGE=${IMAGE:-tt-model/ornith-1.5-35b-a3b-v2:0fe927ac6e21}
mkdir -p $HOME/.cache/tt-model/${CACHE_NAME:-ornith-1.5-35b-a3b-v2}/cache-dev
exec docker run --rm ${TNAME:+--name $TNAME} --user $(id -u):$(id -g) --device /dev/tenstorrent --ipc host \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --volume $HOME/.cache/huggingface:/hf --env HF_HOME=/hf --env HF_HUB_OFFLINE=1 \
  --volume $HOME/.cache/tt-model/${CACHE_NAME:-ornith-1.5-35b-a3b-v2}/cache-dev:/cache --env TT_METAL_CACHE=/cache \
  --volume $HOME/ornith-fusion/generated:/opt/tt-metal/generated --volume $METAL/models:/opt/tt-metal/models --volume $METAL/tests:/opt/tt-metal/tests:ro \
  --volume $METAL/conftest.py:/opt/tt-metal/conftest.py:ro --volume $METAL/pytest.ini:/opt/tt-metal/pytest.ini:ro \
  --env ARCH_NAME=blackhole --env HF_MODEL=${MODEL_ID:-ornith-ai/Ornith-1.5-35B-A3B} --env MESH_DEVICE=${MESH:-P300x2} \
  --env PYTHONPATH=/opt/tt-metal ${DEVS:+--env TT_VISIBLE_DEVICES=$DEVS} ${TENV:-} --workdir /opt/tt-metal --entrypoint bash $IMAGE -c "$*"
