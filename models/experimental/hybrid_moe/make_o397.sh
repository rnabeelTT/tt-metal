#!/bin/bash
# Ornith-1.5-397B (from templates/make_artifacts.sh). Offline artifacts for a hybrid-MoE model: token sets -> CPU reference routing -> expert plan -> full-layer sets ->
# int4 host library -> device-format (bfp4) expert copies. This is exactly what was run for MiniMax-M2.7 (2026-11-14),
# with the model-specific values pulled up here. Copy it next to your model's scripts and edit the CHANGE ME block.
#
# usage: templates/make_artifacts.sh [texts|ref|plan|int4|dev|all]   (default all; every step skips finished output)
# needs: a venv with torch (cpu), transformers, safetensors, numpy, pandas, pyarrow (PY below); the HF checkpoint and
# the four datasets quality_texts.py reads in the HF cache; for `dev`: the serving image and the device (nothing
# else using it).
set -e

# ---------------------------------------------------------------- CHANGE ME
MODEL=ornith-ai/Ornith-1.5-397B-FP8           # HF repo of the checkpoint (must be in the HF cache)
TAG=o397                                # short name: runs/<TAG>/, ~/.cache/hybrid-moe/<TAG>_*
N_LAYERS=60                            # num_hidden_layers (MoE layers only, if some are dense: see ADD_A_MODEL.md)
REF=ref_o397.py                         # your model's CPU reference (copy ref_m27.py and adapt)
SIM=sim_o397.py                         # hit-rate sim / plan writer (copy sim_m27.py: set H, I, E at its top)
KEYS=qwen35                            # build_int4_library.py checkpoint key style: qwen35 | minimax_m2 | add yours
S=192                                  # device slots kept per hybrid layer in the plan; the library holds the rest
FULL_MIN=0                             # full layers of your longest-context profile: every other layer can be hybrid
IMG=${HYB_IMAGE:-tt-model/minimax-m2.7-hybrid-qb2:0c9857a46f16}   # an image with this branch's tt-metal (for `dev`)
PY=${PY:-$HOME/m27-work/venv/bin/python}
# ---------------------------------------------------------------- end

D=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)   # works from templates/ or from a copy next to the scripts
if [ -f "$D/build_int4_library.py" ]; then P=$D; else P=$(cd "$D/.." && pwd); fi   # models/experimental/hybrid_moe
R=$P/runs/$TAG; LIB=$HOME/.cache/hybrid-moe; mkdir -p "$R" "$LIB"
export HF_HUB=${HF_HUB:-$HOME/.cache/huggingface/hub} TOK_REPO=$MODEL REF_THREADS=${REF_THREADS:-16}
SNAP=$(ls -d "$HF_HUB/models--${MODEL/\//--}/snapshots/"* | head -1)
cd "$P"
step=${1:-all}
run() { [ "$step" = all ] || [ "$step" = "$1" ]; }

if run texts; then  # 1000-token texts: train (plan) and held-out (evaluation); GSM8K, MBPP, multilingual chat, code
  [ -f $R/texts_train.json ] || $PY quality_texts.py $R/texts_train.json train
  [ -f $R/texts_held.json ] || $PY quality_texts.py $R/texts_held.json
fi
if run ref; then  # BF16 CPU reference over both sets: per-token routing (hours on CPU; REF_THREADS)
  [ -f $R/ref_held/routing.npz ] || $PY $REF $R/texts_held.json $R/ref_held
  [ -f $R/ref_train/routing.npz ] || $PY $REF $R/texts_train.json $R/ref_train
fi
if run plan; then  # expert order per layer by routed mass + the HYB_PLAN file (the train routing, plan_from_routing.py)
  [ -f $R/plan_$TAG.npz ] || $PY $SIM $R/ref_train/routing.npz $R/ref_held/routing.npz $R/plan_$TAG.npz | tee $R/sim_$TAG.txt
  [ -f $R/plan_${TAG}_routing.npz ] || $PY plan_from_routing.py $R/ref_train/routing.npz $R/plan_${TAG}_routing.npz
  for n in 0 4 8 12 16; do  # full-layer sets per profile (edit the counts to your profiles)
    echo "full@$n: $($PY sim_full_layers.py $R/plan_$TAG.npz $R/ref_train/routing.npz $R/ref_held/routing.npz $n | tail -1)"
  done | tee $R/full_layers.txt
fi
FL=$($PY sim_full_layers.py $R/plan_$TAG.npz $R/ref_train/routing.npz $R/ref_held/routing.npz $FULL_MIN 2>/dev/null | tail -1)
HYBL=$($PY -c "f={int(x) for x in '$FL'.split(',') if x};print(','.join(str(i) for i in range($N_LAYERS) if i not in f))")
if run int4; then  # int4 (group 32: the host kernel needs gs dividing 64) of every expert outside each layer's top S
  echo "hybrid layers: $HYBL"
  HYB_INT4_GS=32 HYB_LIB_KEYS=$KEYS HYB_LIB_COLD=$R/plan_$TAG.npz:$S THREADS=16 \
    $PY build_int4_library.py "$SNAP" $LIB/${TAG}_int4_s$S "$HYBL"
fi
if run dev; then  # the same experts in the device's bfp4 TP layout (ttnn packing needs the device)
  docker run --rm --device /dev/tenstorrent --ipc host --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
    -v $HOME/.cache/huggingface:/hf:ro -v $P:/poc -v $LIB:/hyblib --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e ARCH_NAME=blackhole -e THREADS=8 -e TT_METAL_LOGGER_LEVEL=Error -e PYTHONPATH=/opt/tt-metal:/poc \
    --entrypoint python $IMG /poc/build_dev_experts.py /hf/hub/models--${MODEL/\//--}/snapshots/$(basename $SNAP) \
    /hyblib/${TAG}_dev_s$S /poc/runs/$TAG/plan_$TAG.npz:$S "$HYBL"
fi
echo "done: $step (runs/$TAG, $LIB/${TAG}_*)"
