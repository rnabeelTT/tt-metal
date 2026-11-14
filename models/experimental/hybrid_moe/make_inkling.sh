#!/bin/bash
# Offline artifacts for Inkling-Small on the hybrid overlay (from templates/make_artifacts.sh): token sets -> CPU
# reference routing (ref_inkling.py) -> expert plan (sim_inkling.py: 8 experts of 258 per token, the 2 shared ones
# pinned resident) -> full-layer sets -> int4 host library -> device-format (bfp4) expert copies. Experts are read
# through the port's StreamingStateDict (HYB_LIB_KEYS=port:...), which folds route_scale x global_scale into them.
#
# usage: templates/make_artifacts.sh [texts|ref|plan|int4|dev|all]   (default all; every step skips finished output)
# needs: a venv with torch (cpu), transformers, safetensors, numpy, pandas, pyarrow (PY below); the HF checkpoint and
# the four datasets quality_texts.py reads in the HF cache; for `dev`: the serving image and the device (nothing
# else using it).
set -e

# ---------------------------------------------------------------- CHANGE ME
MODEL=thinkingmachines/Inkling-Small    # HF repo of the checkpoint (must be in the HF cache)
TAG=inkling                            # short name: runs/<TAG>/, ~/.cache/hybrid-moe/<TAG>_*
N_LAYERS=42                            # num_hidden_layers (layers 0-1 are dense: never hybrid, see HYBL below)
REF=ref_inkling.py                     # the CPU reference
SIM=sim_inkling.py                     # hit-rate sim / plan writer (+ the model-layer-indexed routing)
KEYS=port:thinkingmachines_inkling_small  # experts via the port's StreamingStateDict.one_expert
S=${S:-194}                            # device slots per hybrid layer (2 shared + 192 routed); the library holds the rest
FULL_MIN=${FULL_MIN:-8}                # full layers of the longest-context profile (set from runs/inkling/full_layers.txt)
IMG=${HYB_IMAGE:-tt-model/minimax-m2.7-hybrid-qb2:0c9857a46f16}   # an image with this branch's tt-metal (for `dev`)
PY=${PY:-$HOME/inkling-work/venv/bin/python}
# ---------------------------------------------------------------- end

P=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)   # models/experimental/hybrid_moe
R=$P/runs/$TAG; LIB=$HOME/.cache/hybrid-moe; mkdir -p "$R" "$LIB"
export HF_HUB=${HF_HUB:-${HF_HUB_CACHE:-/opt/tenstorrent/models/hub}} TOK_REPO=$MODEL REF_THREADS=${REF_THREADS:-16}
export TT_SRC=${TT_SRC:-$P/../../autoports} PYTHONPATH=${PYTHONPATH:-$(cd "$P/../../.." && pwd)}
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
  # the HYB_PLAN file from the model-layer-indexed routing sim_inkling.py wrote, with the sim's exact order embedded
  [ -f $R/plan_${TAG}_hyb.npz ] || $PY plan_from_routing.py $R/plan_${TAG}_routing.npz $R/plan_${TAG}_hyb.npz $R/plan_$TAG.npz
  for n in $FULL_MIN 36 44; do  # full-layer sets per profile (edit the counts to your profiles)
    echo "full@$n: $($PY sim_full_layers.py $R/plan_$TAG.npz $R/plan_${TAG}_routing.npz $R/plan_${TAG}_held_routing.npz $n $S | tail -1)"
  done | tee $R/full_layers.txt
fi
FL=$($PY sim_full_layers.py $R/plan_$TAG.npz $R/plan_${TAG}_routing.npz $R/plan_${TAG}_held_routing.npz $FULL_MIN $S 2>/dev/null | tail -1)
HYBL=$($PY -c "f={int(x) for x in '$FL'.split(',') if x};print(','.join(str(i) for i in range(2, $N_LAYERS) if i not in f))")
if run int4; then  # int4 (group 32: the host kernel needs gs dividing 64) of every expert outside each layer's top S
  echo "hybrid layers: $HYBL"
  HYB_INT4_GS=32 HYB_LIB_KEYS=$KEYS HYB_LIB_COLD=$R/plan_$TAG.npz:$S THREADS=16 \
    $PY build_int4_library.py "$SNAP" $LIB/${TAG}_int4_s$S "$HYBL"
fi
if run dev; then  # the same experts in the device's bfp4 TP layout (ttnn packing needs the device)
  docker run --rm --device /dev/tenstorrent --ipc host --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
    -v $(dirname $HF_HUB):/hf:ro -v $P:/poc -v $LIB:/hyblib --user "$(id -u):$(id -g)" \
    -v $P/../../autoports/thinkingmachines_inkling_small:/opt/tt-metal/models/autoports/thinkingmachines_inkling_small \
    -e HOME=/tmp -e ARCH_NAME=blackhole -e THREADS=8 -e TT_METAL_LOGGER_LEVEL=Error -e PYTHONPATH=/opt/tt-metal:/poc \
    -e HYB_LIB_KEYS=$KEYS -e HF_HUB_CACHE=/hf/hub -e HF_HOME=/hf \
    --entrypoint python $IMG /poc/build_dev_experts.py /hf/hub/models--${MODEL/\//--}/snapshots/$(basename $SNAP) \
    /hyblib/${TAG}_dev_s$S /poc/runs/$TAG/plan_$TAG.npz:$S "$HYBL"
fi
echo "done: $step (runs/$TAG, $LIB/${TAG}_*)"
