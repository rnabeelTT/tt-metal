#!/bin/bash
# pc_release.sh [keys...]: release the prefix-caching builds into the four fast repos (user's go-ahead 2026-10-28).
# Per model: package -> serve the local build p300x2 as shipped -> gate (prefix caching active, greedy 8/8 + 8/8 vs the
# last hub smoke of the published build, a resent 24k prompt >= 3x faster) -> push --publish -> fresh pull -> hub smoke
# (greedy vs the local run, resent prompt, 1-user / 32-user rows). A failed gate stops that model before the push.
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH
set -o pipefail
R=$HOME/ornith-fusion/runs
T=$HOME/tt-metal-bringups/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools

serve() {  # <label> <bundle>: tt-model serve --detach p300x2, wait for the API; sets C
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

checks() {  # <label> <greedy reference>: greedy, resent prompt, bench rows, prefix log; prints GATE ok|fail
  local L=$1 REF=$2 ok=1
  URL=http://localhost:20000 RUNS=$R python3 $T/greedy.py run $L > /dev/null 2>&1
  python3 $T/greedy.py compare $REF $L 2>&1 | grep -E "^(sequential|concurrent):" | tee $R/$L/greedy_cmp.txt
  [ "$(grep -c '8/8 identical' $R/$L/greedy_cmp.txt)" = 2 ] || ok=0
  (cd $T && TAG=$L python3 prefix_bench.py repeat 8192 32768 > $R/$L/repeat.log 2>&1)
  grep '^{' $R/$L/repeat.log
  python3 - $R/$L/repeat.log <<'EOF' || ok=0
import json, sys
r = [json.loads(l[7:]) for l in open(sys.argv[1]) if l.startswith("RESULT ")][0]["rows"]
sys.exit(0 if r[-1]["speedup"] >= 3 and all(x["identical"] for x in r) else 1)
EOF
  docker logs $C 2>&1 | grep -E "Recurrent-state prefix caching|prefix cache:|Automatic prefix caching" | tee $R/$L/prefix_log.txt
  grep -q "Recurrent-state prefix caching: 32 snapshot slots" $R/$L/prefix_log.txt || ok=0
  RUNS=$R $T/bench.sh $C $L 128:128:1:4 128:128:32:128 2>&1 | grep -E "TTFT|TPOT|ROW FAILED"
  [ $ok = 1 ] && echo "$L: GATE ok" || echo "$L: GATE fail"
  return $((1 - ok))
}

release() {  # <key> <tree> <manifest> <MODEL_ID> <CACHE_NAME|-> <repo> <greedy reference>
  local K=$1 Q=$2 MAN=$3 REPO=$6 REF=$7
  export MODEL_ID=$4; if [ "$5" = - ]; then unset CACHE_NAME; else export CACHE_NAME=$5; fi
  NAME=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['name'])" $Q/$MAN)
  local B=$Q/build/$NAME L=rel_$K H=hub_$K
  rm -rf $R/$L $R/$H; mkdir -p $R/$L $R/$H
  echo "== $K ($NAME -> $REPO) $(date +%T)"
  (cd $Q && tt-model package --container $MAN > $R/$L/package.log 2>&1); echo "$K: package exit $?"
  grep -q "### Prefix caching" $B/README.md && grep -q -- "--enable-prefix-caching" $B/tt_kernel_manifest.json \
    || { echo "$K: bundle lacks the prefix caching card or flag"; return 1; }
  local IMG=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['container']['built']['image'])" $B/tt_kernel_manifest.json)
  echo "$K: built $IMG"
  serve $L $B/tt_kernel_manifest.json || return 1
  checks $L $REF; local g=$?
  docker rm -f $C >/dev/null
  [ $g = 0 ] || { echo "$K: not pushed"; return 1; }
  tt-model push $B --publish > $R/$L/push.log 2>&1; echo "$K: push exit $?"; tail -n 3 $R/$L/push.log
  tt-model pull $REPO > $R/$H/pull.log 2>&1; echo "$K: pull exit $? $(tail -n 2 $R/$H/pull.log | tr -s ' \n' ' ')"
  serve $H $REPO || return 1
  checks $H $L
  docker rm -f $C >/dev/null
  echo "== $K done $(date +%T)"
}

while chips_busy; do sleep 60; done
echo "pc release start $(date +%T)"
for k in ${@:-orn cn q122 o122}; do
  case $k in
    orn) release orn $HOME/tt-metal-bringups models/autoports/ornith_ai_ornith_1_5_35b_a3b/tt-model-fast.yaml \
           ornith-ai/Ornith-1.5-35B-A3B - raahemnabeel/ornith-1.5-35b-a3b fin_smoke_orn2 ;;
    cn) release cn $HOME/tt-metal-codernext models/autoports/qwen_qwen3_coder_next/tt-model-fast.yaml \
           Qwen/Qwen3-Coder-Next - raahemnabeel/qwen3-coder-next fin_smoke_cn2 ;;
    q122) release q122 $HOME/tt-metal-q122 models/autoports/qwen_qwen3_5_122b_a10b/tt-model-fast.yaml \
           Qwen/Qwen3.5-122B-A10B qwen3-5-122b-a10b raahemnabeel/qwen3-5-122b-a10b fin_smoke_q122b ;;
    o122) release o122 $HOME/tt-metal-q122 models/autoports/qwen_qwen3_5_122b_a10b/tt-model-ornith122.yaml \
           ornith-ai/Ornith-1.5-122B ornith-1.5-122b raahemnabeel/ornith-1.5-122b fin_smoke_o122 ;;
  esac
done
echo "pc release done $(date +%T)"
