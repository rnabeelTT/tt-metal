#!/bin/bash
# mmfix_release.sh [keys...]: re-publish with the plugin fix for the image-burst crash (vllm-tt-plugin-pub 8b1f8b1).
# Gate adds the burst test (128 HLE requests incl. images, 32 at once) that killed the published server on 2026-10-30.
# Card-only change: package; if the image tag equals the published one, push without a serve. Otherwise gate the local
# build (greedy 8/8 + 8/8 vs the last hub smoke, one 1-user row), push, fresh pull, hub smoke (greedy vs local).
# Always: the pulled README carries the new card, and the pulled manifest names the packaged image.
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH
set -o pipefail
R=$HOME/ornith-fusion/runs
T=$HOME/tt-metal-bringups/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
img() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['container']['built']['image'])" "$1"; }

serve() {  # <label> <bundle>; sets C
  local L=$1 B=$2
  C=tt-model-$NAME-p300x2
  docker rm -f $C >/dev/null 2>&1
  tt-model serve "$B" --profile p300x2 --detach --port 20000 --no-update-check > $R/$L/serve.log 2>&1
  local t0=$(date +%s)
  until curl -sf localhost:20000/v1/models >/dev/null; do
    docker ps --format '{{.Names}}' | grep -qx $C || { echo "$L: server died"; docker logs $C 2>&1 | grep -v DEBUG | tail -n 300 > $R/$L/boot_fail.log; return 1; }
    sleep 10
  done
  echo "$L: up after $(( $(date +%s) - t0 )) s; image $(docker inspect -f '{{.Config.Image}}' $C)"
}

gate() {  # <label> <greedy reference>
  local L=$1 REF=$2
  URL=http://localhost:20000 RUNS=$R python3 $T/greedy.py run $L > /dev/null 2>&1
  python3 $T/greedy.py compare $REF $L 2>&1 | grep -E "^(sequential|concurrent):" | tee $R/$L/greedy_cmp.txt
  RUNS=$R $T/bench.sh $C $L 128:128:1:4 2>&1 | grep -E "TTFT|TPOT|ROW FAILED"
  ~/evals/text2/.venv/bin/python $R/mm_burst_test.py $MODEL_ID 128 32 > $R/$L/mm_burst.txt 2>&1; tail -n 1 $R/$L/mm_burst.txt
  curl -sf localhost:20000/v1/models >/dev/null || { echo "server down after the burst"; return 1; }
  grep -q "^MM_BURST ok=128/128" $R/$L/mm_burst.txt && [ "$(grep -c '8/8 identical' $R/$L/greedy_cmp.txt)" = 2 ]
}

release() {  # <key> <tree> <manifest> <MODEL_ID> <CACHE_NAME|-> <repo> <greedy reference>
  local K=$1 Q=$2 MAN=$3 REPO=$6 REF=$7
  export MODEL_ID=$4; if [ "$5" = - ]; then unset CACHE_NAME; else export CACHE_NAME=$5; fi
  NAME=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['name'])" $Q/$MAN)
  local B=$Q/build/$NAME L=mmfix_rel_$K H=mmfix_hub_$K
  rm -rf $R/$L $R/$H; mkdir -p $R/$L $R/$H
  echo "== $K ($NAME -> $REPO) $(date +%T)"
  local OLD=$(img $HOME/.cache/tt-model/pulled/${REPO/\//__}/tt_kernel_manifest.json)  # vs the PUBLISHED image: gate whenever it differs
  (cd $Q && tt-model package --container $MAN > $R/$L/package.log 2>&1); echo "$K: package exit $?"
  grep -E "^\s*!" $R/$L/package.log
  grep -q "## Intended use" $B/README.md && grep -q "^license:" $B/README.md || { echo "$K: bundle lacks the new card"; return 1; }
  local NEW=$(img $B/tt_kernel_manifest.json)
  echo "$K: image $OLD -> $NEW"
  if [ "$NEW" != "$OLD" ]; then
    serve $L $B/tt_kernel_manifest.json || return 1
    gate $L $REF; local g=$?
    docker rm -f $C >/dev/null
    [ $g = 0 ] || { echo "$K: GATE fail, not pushed"; return 1; }
    echo "$K: GATE ok"
  fi
  tt-model push $B --publish > $R/$L/push.log 2>&1; echo "$K: push exit $?"; tail -n 3 $R/$L/push.log
  tt-model pull $REPO > $R/$H/pull.log 2>&1; echo "$K: pull exit $?"
  local P=$HOME/.cache/tt-model/pulled/${REPO/\//__}
  echo "$K: pulled image $(img $P/tt_kernel_manifest.json)"
  curl -sL "https://huggingface.co/$REPO/raw/main/README.md" > $R/$H/README.md
  cmp -s <(sed '/^tags:/,/^[a-z_]*:/{/^- /d}' $R/$H/README.md) <(sed '/^tags:/,/^[a-z_]*:/{/^- /d}' $B/README.md) \
    && echo "$K: hub README = packaged README (tags aside)" || { echo "$K: hub README differs:"; diff <(cat $B/README.md) $R/$H/README.md | head -20; }
  if [ "$NEW" != "$OLD" ]; then
    serve $H $REPO || return 1
    URL=http://localhost:20000 RUNS=$R python3 $T/greedy.py run $H > /dev/null 2>&1
    python3 $T/greedy.py compare $L $H 2>&1 | grep -E "^(sequential|concurrent):"
    ~/evals/text2/.venv/bin/python $R/mm_burst_test.py $MODEL_ID 128 32 > $R/$H/mm_burst.txt 2>&1; tail -n 1 $R/$H/mm_burst.txt
    docker rm -f $C >/dev/null
  fi
  echo "== $K done $(date +%T)"
}

while chips_busy; do sleep 60; done
echo "mmfix release start $(date +%T)"
for k in ${@:-o122 q122}; do
  case $k in
    orn) release orn $HOME/tt-metal-bringups models/autoports/ornith_ai_ornith_1_5_35b_a3b/tt-model-fast.yaml \
           ornith-ai/Ornith-1.5-35B-A3B - raahemnabeel/ornith-1.5-35b-a3b hub_orn ;;
    cn) release cn $HOME/tt-metal-codernext models/autoports/qwen_qwen3_coder_next/tt-model-fast.yaml \
           Qwen/Qwen3-Coder-Next - raahemnabeel/qwen3-coder-next hub_cn ;;
    q122) release q122 $HOME/tt-metal-q122 models/autoports/qwen_qwen3_5_122b_a10b/tt-model-fast.yaml \
           Qwen/Qwen3.5-122B-A10B qwen3-5-122b-a10b raahemnabeel/qwen3-5-122b-a10b card_hub_q122 ;;
    o122) release o122 $HOME/tt-metal-q122 models/autoports/qwen_qwen3_5_122b_a10b/tt-model-ornith122.yaml \
           ornith-ai/Ornith-1.5-122B ornith-1.5-122b raahemnabeel/ornith-1.5-122b card_hub_o122 ;;
  esac
done
echo "mmfix release done $(date +%T)"  # the caller lifts evalq.pause
