#!/bin/bash
# ref_evals2.sh: original-card evals with their own harnesses (user request 2026-10-28), after ref_evals.sh.
#   o122: SciCode test split, with background, subproblem pass@1 (Artificial Analysis protocol)   card 53.8
#   cn  : Aider polyglot, 225 exercises, 2 tries, diff and whole edit formats                     card 66.2
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH
R=$HOME/ornith-fusion/runs
serve() {  # <label> <repo> <name>; sets C
  local L=$1 B=$2; C=tt-model-$3-p300x2; mkdir -p $R/$L
  docker rm -f $C >/dev/null 2>&1
  tt-model serve "$B" --profile p300x2 --detach --port 20000 --no-update-check > $R/$L/serve.log 2>&1
  local t0=$(date +%s)
  until curl -sf localhost:20000/v1/models >/dev/null; do
    docker ps --format '{{.Names}}' | grep -qx $C || { echo "$L: server died"; docker logs $C 2>&1 | grep -v DEBUG | tail -n 300 > $R/$L/boot_fail.log; return 1; }
    sleep 10
  done
  echo "$L: up after $(( $(date +%s) - t0 )) s; image $(docker inspect -f '{{.Config.Image}}' $C)"
}
while systemctl --user is-active -q cards-release || systemctl --user is-active -q ref-evals || chips_busy; do sleep 60; done
echo "ref evals2 start $(date +%T)"
if serve ref_o122 raahemnabeel/ornith-1.5-122b ornith-1.5-122b; then
  echo "== o122 scicode $(date +%T)"
  RUNS_ROOT=$R ~/evals/scicode_run.sh ref_o122_scicode http://localhost:20000/v1 ornith-ai/Ornith-1.5-122B 32 32768 > $R/ref_o122/scicode.log 2>&1
  echo "o122 scicode exit $? $(date +%T)"; grep "^SCICODE" $R/ref_o122/scicode.log
  docker rm -f $C >/dev/null
fi
if serve ref_cn raahemnabeel/qwen3-coder-next qwen3-coder-next; then
  for fmt in diff whole; do
    echo "== cn aider $fmt $(date +%T)"
    ~/evals/aider_run.sh ref_cn_aider_$fmt http://localhost:20000/v1 Qwen/Qwen3-Coder-Next 32 $fmt > $R/ref_cn/aider_$fmt.log 2>&1
    echo "cn aider $fmt exit $? $(date +%T)"; grep "^AIDER" $R/ref_cn/aider_$fmt.log
  done
  docker rm -f $C >/dev/null
fi
echo "ref evals2 done $(date +%T)"
