#!/bin/bash
# run_layer_prof.sh <pytest -k expr> <label>: Tracy + device profiler of the Ornith per-layer perf tests on P300x2 (all 4 chips)
K=$1; L=$2
mkdir -p ${RUNS:-$HOME/ornith-fusion/runs}/prof/$L
docker run --rm --name ornith-prof-$L --user $(id -u):$(id -g) --device /dev/tenstorrent --ipc host \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --volume $HOME/.cache/huggingface:/hf --env HF_HOME=/hf --env HF_HUB_OFFLINE=1 \
  --volume $HOME/.cache/tt-model/ornith-1.5-35b-a3b-v2/cache-gdnpoc:/cache --env TT_METAL_CACHE=/cache \
  --volume ${METAL:-$HOME/tt-metal}/models:/opt/tt-metal/models \
  --volume ${METAL:-$HOME/tt-metal}/conftest.py:/opt/tt-metal/conftest.py:ro --volume ${METAL:-$HOME/tt-metal}/pytest.ini:/opt/tt-metal/pytest.ini:ro \
  --volume ${METAL:-$HOME/tt-metal}/tests:/opt/tt-metal/tests:ro \
  --volume ${RUNS:-$HOME/ornith-fusion/runs}/prof/$L:/opt/tt-metal/generated \
  --env PYTEST_ADDOPTS="-k '$K'" --env ARCH_NAME=blackhole --env HF_MODEL=ornith-ai/Ornith-1.5-35B-A3B --env MESH_DEVICE=P300x2 \
  --env QWEN35_LAYER_TESTS_POLICY=shipped --env TT_METAL_PROFILER_MID_RUN_DUMP=1 \
  --workdir /opt/tt-metal --entrypoint bash ornith-prof -c \
  "python -m tracy -r -v --op-support-count=20000 -m pytest -x -q -s -m perf models/autoports/ornith_ai_ornith_1_5_35b_a3b/tests/test_multichip_decoder.py" \
  > ${RUNS:-$HOME/ornith-fusion/runs}/prof/$L.log 2>&1
echo "exit $?"
