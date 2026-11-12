#!/bin/bash
# Build a co-host image:  ./build.sh 122b | 35b | qwen38
#   122b / 35b: Ornith-1.5 (FROM its package image) + Qwen3-VL-Embedding-8B + Qwen3-VL-Reranker-8B
#   qwen38:     Qwen3.8-27B DFlash2 + vision (the v6 thin bundle mando2222/qwen3.8-27b-dflash2-vision-p300x2-q4kv-v6-thin-cp4,
#               vLLM 0.30, PyPI ttnn 0.79.0) + the same two 8B models
#
# Inputs (all local images / checkouts, pinned below):
#   BASE       the Ornith package image (its tt-metal build, vLLM plugin and model code, unchanged)
#   QWEN3VL    the qwen3-vl-embedding-8b package image: source of the Qwen3-VL model code (embedder + reranker share
#              it) and the helper packages it imports (models/ttt_compat, models/tt_dit, models/common/tt_ccl.py)
#   COHOST     a tt-model-manager checkout with mixed co-host groups (branch raahem/cohost-mixed-groups)
# Output: image tt-model/cohost-ornith-<size>-qwen3vl:<tag>, printed on the last line.
set -euo pipefail
SIZE=${1:?usage: build.sh 122b|35b|qwen38}
HERE=$(cd "$(dirname "$0")" && pwd)
case $SIZE in
  122b) BASE=${BASE:-tt-model/ornith-1.5-122b:b98336b9ce62} ;;
  35b)  BASE=${BASE:-tt-model/ornith-1.5-35b-a3b:5b4af07ab13a} ;;   # 2026-11-10: batched short-prompt prefill, wider MoE dispatch
  qwen38) BASE=${BASE:-tt-model/ornith-1.5-122b:b98336b9ce62}   # system libraries and the tt user only
          Q38_REPO=mando2222/qwen3.8-27b-dflash2-vision-p300x2-q4kv-v6-thin-cp4
          Q38_REV=${Q38_REV:-67af7db5a391635a798c493907ec1515291b4016}
          SFPI_URL=https://github.com/tenstorrent/sfpi/releases/download/7.78.0/sfpi_7.78.0_x86_64_debian.txz
          SFPI_SHA=464c64759e441b841d6bc06216dbd470d7a0f188e0ffa9ac8ebb2e404cf244db ;;
  *) echo "size must be 122b, 35b or qwen38" >&2; exit 2 ;;
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
# v0.82's link count is a ttnn call this v0.79 build lacks; the base's own helper answers it from its device table
sed -i 's/^    return ttnn.get_num_links(mesh_device, cluster_axis)$/    from models.common.modules.tt_ccl import get_num_links as v079_num_links\n\n    return v079_num_links(mesh_device, cluster_axis)/' \
  "$CTX/context/models/common/tt_ccl.py"
grep -q "ttnn.get_num_links" "$CTX/context/models/common/tt_ccl.py" && { echo "tt_ccl patch failed" >&2; exit 1; }
cp "$HERE/compat/tensor_utils_v082.py" "$CTX/context/"   # appended to the base's tensor_utils (Dockerfile)
# the vision tower wrapper was written for one chip; here its blocks are tensor-parallel over the mesh
for p in "$HERE"/patches/*.patch; do patch -p1 --forward --silent -d "$CTX/context" < "$p"; done
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

DOCKERFILE=Dockerfile
if [ "$SIZE" = qwen38 ]; then
  DOCKERFILE=Dockerfile.qwen38
  # the thin bundle at its pinned revision
  hf download "$Q38_REPO" --revision "$Q38_REV" --local-dir "$CTX/context/q38" >/dev/null
  rm -rf "$CTX/context/q38/.cache"
  # SFPI 7.78.0, checked
  SFPI_CACHE=${XDG_CACHE_HOME:-$HOME/.cache}/cohost-build/$(basename "$SFPI_URL")
  if ! echo "$SFPI_SHA  $SFPI_CACHE" | sha256sum -c - >/dev/null 2>&1; then
    mkdir -p "$(dirname "$SFPI_CACHE")"; curl -fsSL -o "$SFPI_CACHE" "$SFPI_URL"
  fi
  echo "$SFPI_SHA  $SFPI_CACHE" | sha256sum -c - >/dev/null
  tar -xJf "$SFPI_CACHE" -C "$CTX/context"     # -> context/sfpi/ (the base image has no xz)
  # the uv release the bundle was built with (PROVENANCE.json): it knows the pinned interpreter
  UV_TGZ=${XDG_CACHE_HOME:-$HOME/.cache}/cohost-build/uv-0.12.11-x86_64-unknown-linux-gnu.tar.gz
  [ -s "$UV_TGZ" ] || curl -fsSL -o "$UV_TGZ" https://github.com/astral-sh/uv/releases/download/0.12.11/uv-x86_64-unknown-linux-gnu.tar.gz
  tar -xzf "$UV_TGZ" -C "$CTX" && mv "$CTX/uv-x86_64-unknown-linux-gnu/uv" "$CTX/context/uv.q38"
  # a standalone `models` package for the Qwen3-VL code: models/common from the v0.79 Ornith tree, plus the packages
  # staged above (the PyPI ttnn ships no models/)
  M="$CTX/context/cohost-models/models"; mkdir -p "$M/autoports"
  c=$(docker create "$BASE"); docker cp -q "$c:/opt/tt-metal/models/common" "$M/"; docker cp -q "$c:/opt/tt-metal/models/__init__.py" "$M/" 2>/dev/null || touch "$M/__init__.py"; docker rm "$c" >/dev/null
  cp "$CTX/context/models/common/tt_ccl.py" "$M/common/"
  cat "$CTX/context/tensor_utils_v082.py" >> "$M/common/tensor_utils.py"   # as the Dockerfile does for 122b / 35b
  cp -r "$CTX/context/models/ttt_compat" "$CTX/context/models/tt_dit" "$M/"
  cp -r "$CTX/context/models/autoports/qwen_qwen3_vl_reranker_8b" "$CTX/context/models/autoports/qwen_qwen3_vl_embedding_8b" "$M/autoports/"
  touch "$M/autoports/__init__.py"
  find "$M" -name __pycache__ -prune -exec rm -rf {} +
  rm -rf "$CTX/context/models"
  # Qwen3.8's chat template with the same agent-loop fix
  src=${HF_HUB:-/opt/tenstorrent/models/hub}/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/chat_template.jinja
  sed -E "s/^( *)\{\{- raise_exception\('No user query found in messages.'\) \}\}/\1{%- set ns.last_query_index = messages|length - 1 %}/" \
    "$src" > "$CTX/context/tt-cohost/templates/qwen3.8-27b.jinja"
  grep -q "No user query" "$CTX/context/tt-cohost/templates/qwen3.8-27b.jinja" && { echo "qwen3.8 template patch failed" >&2; exit 1; }
fi
cp "$HERE/$DOCKERFILE" "$CTX/Dockerfile"
cp "$(command -v uv)" "$CTX/context/uv"   # installer for the two wheels (the base image has no pip)
SRC=$(cd "$HERE" && git rev-parse --short HEAD 2>/dev/null || echo dev)
RT=$(cd "$COHOST" && git rev-parse --short HEAD)
NAME=cohost-ornith-$SIZE-qwen3vl; [ "$SIZE" = qwen38 ] && NAME=cohost-qwen38-27b-qwen3vl
TAG=${TAG:-tt-model/$NAME:$SRC-rt$RT}
docker build --build-arg BASE="$BASE" \
  --label org.tenstorrent.tt-model=$NAME \
  --label org.tenstorrent.tt-model.kind=cohost \
  --label org.tenstorrent.tt-model.repo=raahemnabeel/$NAME \
  --label org.tenstorrent.tt-model.profiles=see-manifest \
  --label org.tenstorrent.tt-model.base="$BASE" \
  --label org.tenstorrent.tt-model.qwen3vl-source="$QWEN3VL" \
  --label org.tenstorrent.tt-model.cohost-runtime="$RT" \
  -t "$TAG" "$CTX"
echo "$TAG"
