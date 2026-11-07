#!/bin/bash
# Run pytest from the fork's models/ tree inside the serving image (the image's ttnn; the fork's python models code).
# usage: run_fork_pytest.sh <log file> <pytest args...>
LOG=$1; shift
P=$(cd "$(dirname "$(readlink -f "$0")")" && pwd); F=$HOME/FORK-tt-metal
mkdir -p "$P/cache" "$P/logs" "$P/pyext"
docker run --rm --name hybrid-moe-poc --device /dev/tenstorrent --ipc host \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  -v $HOME/.cache/huggingface:/hf -v $P:/poc -v $P/cache:/cache -v $HOME/.cache/hybrid-moe:/hyblib \
  -v $F/models:/opt/tt-metal/models -v $F/conftest.py:/opt/tt-metal/conftest.py -v $F/pytest.ini:/opt/tt-metal/pytest.ini -v $F/tests:/opt/tt-metal/tests \
  --user "$(id -u):$(id -g)" -w /opt/tt-metal \
  -e ARCH_NAME=blackhole -e HF_MODEL=${HF_MODEL:-MiniMaxAI/MiniMax-M2.7} -e MESH_DEVICE=P300x2 -e HF_HUB_OFFLINE=1 \
  -e HF_HOME=/hf -e HF_HUB_CACHE=/hf/hub -e HOME=/tmp -e TT_METAL_CACHE=/cache -e TT_METAL_LOGS_PATH=/poc/logs \
  -e PYTHONPATH=/opt/tt-metal:/poc/pyext ${EXTRA_ENV} \
  --entrypoint bash ${HYB_IMAGE:-tt-model/qwen3-5-122b-a10b:8a750811ffec} -c \
  "[ -d /poc/pyext/pytest ] || pip install -q --target /poc/pyext pytest==8.3.5 pytest-timeout >/dev/null 2>&1; python -m pytest $*" > "$LOG" 2>&1
echo "exit $?"
