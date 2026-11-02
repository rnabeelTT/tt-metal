#!/bin/bash
# 122B swap (user, 2026-10-27): the original is already copied to raahemnabeel/qwen3-5-122b-a10b-v1 (hf repos duplicate,
# byte-identical); now the fast build goes into the official raahemnabeel/qwen3-5-122b-a10b (keeps its likes), files only
# the original had are removed from it (they stay in -v1 and in the repo history), then a fresh-pull hub smoke on p300x2.
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH MODEL_ID=Qwen/Qwen3.5-122B-A10B CACHE_NAME=qwen3-5-122b-a10b
C=$HOME/tt-metal-q122; R=$HOME/ornith-fusion/runs; T=$C/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
REPO=raahemnabeel/qwen3-5-122b-a10b
while systemctl --user is-active -q cn-rename || chips_busy; do sleep 60; done
echo "q122 swap start $(date +%T)"
(cd $C && tt-model package --container models/autoports/qwen_qwen3_5_122b_a10b/tt-model-fast.yaml > $R/package_q122swap.log 2>&1; echo "package exit $?")
B=$C/build/qwen3-5-122b-a10b
grep -q "v1 -> current" $B/README.md && grep -q "tt-model serve $REPO --profile" $B/README.md || { echo "README lacks the renamed card"; exit 1; }
python3 -c "import json; print('image', json.load(open('$B/tt_kernel_manifest.json'))['container']['built']['image'])"
tt-model push $B > $R/push_q122swap.log 2>&1; echo "push exit $?"; tail -n 4 $R/push_q122swap.log
python3 - "$B" "$REPO" > $R/q122_stale.txt <<'PY'
import json, os, sys, urllib.request, hashlib
b, repo = sys.argv[1], sys.argv[2]
local = {os.path.relpath(os.path.join(d, f), b) for d, _, fs in os.walk(b) for f in fs}
m = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}"))
remote = {s["rfilename"] for s in m["siblings"]}
man = urllib.request.urlopen(f"https://huggingface.co/{repo}/raw/main/tt_kernel_manifest.json").read()
assert man == open(os.path.join(b, "tt_kernel_manifest.json"), "rb").read(), "remote manifest is not the new bundle's"
missing = local - remote
assert not missing, f"bundle files missing remotely: {sorted(missing)[:5]}"
for f in sorted(remote - local - {".gitattributes"}):
    print(f)
PY
echo "stale check exit $?: $(wc -l < $R/q122_stale.txt) files only the original had"
if [ -s $R/q122_stale.txt ]; then
  hf repos delete-files $REPO $(cat $R/q122_stale.txt) --commit-message "Remove files of the original bundle (kept in raahemnabeel/qwen3-5-122b-a10b-v1)" > $R/q122_delete.log 2>&1
  echo "delete exit $?"; tail -n 2 $R/q122_delete.log
fi
L=fin_smoke_q122b; rm -rf $R/$L; mkdir -p $R/$L
tt-model pull $REPO > $R/$L/pull.log 2>&1; echo "pull exit $? $(tail -n 3 $R/$L/pull.log | tr -s ' \n' ' ')"
GREEDY=1 $T/card_bench.sh $L $REPO p300x2 128:128:1:4 128:128:32:128
grep -h -m2 "image" $R/$L/serve.log
python3 $T/greedy.py compare card_q122fast_p300x2 $L 2>&1 | tail -n 3
echo "q122 swap done $(date +%T)"
