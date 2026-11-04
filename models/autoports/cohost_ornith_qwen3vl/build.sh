#!/bin/bash
# Build the co-host image for one Ornith size:  ./build.sh 122b | 35b
#
# Inputs (all local images / checkouts, pinned below):
#   BASE       the Ornith package image (its tt-metal build, vLLM plugin and model code, unchanged)
#   QWEN3VL    the qwen3-vl-embedding-8b package image: source of the Qwen3-VL model code (embedder + reranker share
#              it) and the helper packages it imports (models/ttt_compat, models/tt_dit, models/common/tt_ccl.py)
#   COHOST     a tt-model-manager checkout with mixed co-host groups (branch raahem/cohost-mixed-groups)
# Output: image tt-model/cohost-ornith-<size>-qwen3vl:<tag>, printed on the last line.
set -euo pipefail
SIZE=${1:?usage: build.sh 122b|35b}
HERE=$(cd "$(dirname "$0")" && pwd)
case $SIZE in
  122b) BASE=${BASE:-tt-model/ornith-1.5-122b:b98336b9ce62} ;;
  35b)  BASE=${BASE:-tt-model/ornith-1.5-35b-a3b:4ce68d7f2568} ;;
  *) echo "size must be 122b or 35b" >&2; exit 2 ;;
esac
QWEN3VL=${QWEN3VL:-tt-model/qwen3-vl-embedding-8b:03461ae75cdb}
COHOST=${COHOST:-$HOME/fork-tt-model-manager}
CTX=$(mktemp -d "${TMPDIR:-/tmp}/cohost-ctx.XXXXXX")
trap 'rm -rf "$CTX"' EXIT
mkdir -p "$CTX/context/models/autoports" "$CTX/context/models/common" "$CTX/context/tt-cohost/tt_kernel" \
         "$CTX/context/tt-cohost/templates"

# 1. Qwen3-VL model code + its helpers: only packages the base image does not have (nothing it ships is replaced)
c=$(docker create "$QWEN3VL")
for p in models/autoports/qwen_qwen3_vl_reranker_8b models/autoports/qwen_qwen3_vl_embedding_8b models/ttt_compat \
         models/tt_dit; do
  docker run --rm --entrypoint test "$BASE" ! -e "/opt/tt-metal/$p" || { echo "base already has $p" >&2; exit 1; }
  docker cp -q "$c:/opt/tt-metal/$p" "$CTX/context/$(dirname $p)/"
done
docker cp -q "$c:/opt/tt-metal/models/common/tt_ccl.py" "$CTX/context/models/common/"   # vision towers only
docker rm "$c" >/dev/null
find "$CTX/context" -name __pycache__ -prune -exec rm -rf {} +
rm -rf "$CTX/context/models/autoports/qwen_qwen3_vl_reranker_8b/demo"   # gradio demo: not served here

# 2. the co-host runtime (mixed groups) and the adapter
cp "$COHOST/src/tt_kernel/__init__.py" "$CTX/context/tt-cohost/tt_kernel/"
cp -r "$COHOST/src/tt_kernel/cohost" "$CTX/context/tt-cohost/tt_kernel/"
cp -r "$HERE/cohost_qwen3vl" "$CTX/context/tt-cohost/"
find "$CTX/context/tt-cohost" -name __pycache__ -prune -exec rm -rf {} +

# 3. Ornith chat templates: the checkpoint's own, with the one-line fix for conversations whose last message is not
#    a user turn (agent loops end on a tool result: the stock template raises "No user query found in messages")
for s in 35b 122b; do
  repo=$([ $s = 122b ] && echo Ornith-1.5-122B || echo Ornith-1.5-35B-A3B)
  rev=$([ $s = 122b ] && echo 7681ef608624e7532d8164128bcffdbdbaf1b1af || echo 10fbf86fed7ecee4a061f8b499a618f46001cac1)
  src=${HF_HUB:-/opt/tenstorrent/models/hub}/models--ornith-ai--$repo/snapshots/$rev/chat_template.jinja
  sed "s/    {{- raise_exception('No user query found in messages.') }}/    {%- set ns.last_query_index = messages|length - 1 %}/" \
    "$src" > "$CTX/context/tt-cohost/templates/ornith-1.5-$s.jinja"
  grep -q "No user query" "$CTX/context/tt-cohost/templates/ornith-1.5-$s.jinja" && { echo "template patch failed" >&2; exit 1; }
done

cp "$HERE/Dockerfile" "$CTX/"
cp "$(command -v uv)" "$CTX/context/uv"   # installer for the two wheels (the base image has no pip)
SRC=$(cd "$HERE" && git rev-parse --short HEAD 2>/dev/null || echo dev)
RT=$(cd "$COHOST" && git rev-parse --short HEAD)
TAG=${TAG:-tt-model/cohost-ornith-$SIZE-qwen3vl:$SRC-rt$RT}
docker build --build-arg BASE="$BASE" \
  --label org.tenstorrent.tt-model=cohost-ornith-$SIZE-qwen3vl \
  --label org.tenstorrent.tt-model.kind=cohost \
  --label org.tenstorrent.tt-model.repo=raahemnabeel/cohost-ornith-$SIZE-qwen3vl \
  --label org.tenstorrent.tt-model.profiles=see-manifest \
  --label org.tenstorrent.tt-model.base="$BASE" \
  --label org.tenstorrent.tt-model.qwen3vl-source="$QWEN3VL" \
  --label org.tenstorrent.tt-model.cohost-runtime="$RT" \
  -t "$TAG" "$CTX"
echo "$TAG"
