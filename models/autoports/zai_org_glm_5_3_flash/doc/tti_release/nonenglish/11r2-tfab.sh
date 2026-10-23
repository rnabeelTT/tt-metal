#!/bin/bash
# Stage 11 review #2 (P2 non-English): device teacher-forcing A/B per precision policy (one process each) from the
# pinned snapshot s11r2_snap (git archive HEAD); weights from the bundle cache (read only by the guard: no conversion).
# usage (inside run-locked): 11r2-tfab.sh TEXTS_JSON OUTDIR policy...   (policy = default | none | <config json>)
source ~/glm53-flash-bringup/env.sh >/dev/null
TEXTS=$1; OUT=$2; shift 2
SNAP=$HOME/glm53-flash-bringup/s11r2_snap
export GLM53_TP=4 GLM53_CACHE_DIR=$HOME/.cache/tt-model/glm-5.3-flash-e192-blackhole/weights/glm53_flash
export GLM53_CKPT=$HOME/glm53-flash-bringup/tfab_overlay
echo "snapshot $(cat $SNAP/BASE_COMMIT); md5 tt/*.py:"; (cd $SNAP && md5sum models/autoports/zai_org_glm_5_3_flash/tt/*.py)
mkdir -p $OUT
for pol in "$@"; do
  case $pol in default) P=""; id=default;; none) P=none; id=none;; *) P=$pol; id=$(basename $pol .json);; esac
  echo "=== policy $id ($P) $(date -u)"; df -h / | tail -1
  env -C $SNAP PYTHONPATH=$SNAP:$TT_METAL_HOME:$TT_MODEL_BRINGUP_ROOT/runtime GLM53_PRECISION=$P \
    $TT_METAL_HOME/python_env/bin/python $HOME/glm53-flash-bringup/scripts/11r2-tf-device.py $TEXTS $OUT/tt_$id.json
  echo "=== policy $id exit=$? $(date -u)"; df -h / | tail -1
done
