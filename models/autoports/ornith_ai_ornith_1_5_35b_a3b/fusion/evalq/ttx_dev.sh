#!/bin/bash
# ttx_dev.sh <MESH_DEVICE> <pytest args...>: tt-transformers device tests in the ttnn==0.79.0 wheel container (+ SFPI 7.78.0)
M=$1; shift
mkdir -p $HOME/ornith-fusion/ttx-cache
exec docker run --rm --name ttx-dev -u $(id -u):$(id -g) -e HOME=/tmp -e PYTHONPATH=/w/src -e MESH_DEVICE=$M -e TT_METAL_CACHE=/cache \
  --device /dev/tenstorrent --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  -v $HOME/ornith-fusion/ttx-cache:/cache -v $HOME/tt-transformers:/w -w /w ttx-ttnn079 \
  python -m pytest -p no:cacheprovider -q "$@"
