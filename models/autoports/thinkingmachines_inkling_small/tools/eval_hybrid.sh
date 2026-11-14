#!/bin/bash
# Teacher-forced quality of the full Inkling model through the hybrid MoE overlay (the quality gate), in the image.
#
# usage: tools/eval_hybrid.sh [out.json]      env: S (device slots per layer, default 194), POLICY (doc/precision stem,
#        default routed_bfp4), EVAL_TEXTS / EVAL_T (subset), FULL (HYB_FULL_LAYERS list), EXTRA (more docker -e flags)
# Needs: the int4 library ~/.cache/hybrid-moe/inkling_int4_s$S (make_inkling.sh int4), the plan
# models/experimental/hybrid_moe/runs/inkling/plan_inkling_hyb.npz (make_inkling.sh plan), the reference
# ~/.cache/inkling/ref/held (ref_inkling.py).
set -e
P=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
S=${S:-194}
POLICY=${POLICY:-routed_bfp4}
OUT=${1:-/cache/eval_s${S}_${POLICY}.json}
SNAP=$(basename $(ls -d ${HF_HOST:-/opt/tenstorrent/models}/hub/models--thinkingmachines--Inkling-Small/snapshots/*))
export DEV_ENV="-v $HOME/.cache/hybrid-moe:/hyblib \
 -e HF_MODEL=/hf/hub/models--thinkingmachines--Inkling-Small/snapshots/$SNAP \
 -e QWEN35_PRECISION_CONFIG=/opt/tt-metal/models/autoports/thinkingmachines_inkling_small/doc/precision/$POLICY.json \
 -e HYB_ENABLE=1 -e HYB_PORT=thinkingmachines_inkling_small -e HYB_POC_DIR=/opt/tt-metal/models/experimental/hybrid_moe \
 -e HYB_FUSED=0 -e HYB_SLOTS=$S -e HYB_LIB=/hyblib/inkling_int4_s$S \
 -e HYB_PLAN=/opt/tt-metal/models/experimental/hybrid_moe/runs/inkling/plan_inkling_hyb.npz \
 -e HYB_STREAM_SLOTS=0 -e HYB_SKIP=0 -e HYB_LOWBIT=0 -e HYB_HOSTK_THREADS=${HOSTK_THREADS:-6} \
 ${FULL:+-e HYB_FULL_LAYERS=$FULL} ${EVAL_TEXTS:+-e EVAL_TEXTS=$EVAL_TEXTS} ${EVAL_T:+-e EVAL_T=$EVAL_T} ${EXTRA}"
exec "$P/tools/run_dev.sh" python models/autoports/thinkingmachines_inkling_small/tools/teacher_force.py \
  /opt/tt-metal/models/experimental/hybrid_moe/runs/inkling/texts_held.json /cache/ref/held "$OUT"
