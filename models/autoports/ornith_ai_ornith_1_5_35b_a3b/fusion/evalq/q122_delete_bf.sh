#!/bin/bash
# After the 122B swap's hub smoke passes (user, 2026-10-27): delete the now-redundant -blackhole-fast repo. The original
# bundle is kept in raahemnabeel/qwen3-5-122b-a10b-v1; the fast one is in raahemnabeel/qwen3-5-122b-a10b.
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH
R=$HOME/ornith-fusion/runs; L=$R/q122_swap.log
while systemctl --user is-active -q q122-swap; do sleep 60; done
echo "q122 delete check $(date +%T)"
grep -q "q122 swap done" $L && grep -q "concurrent: 8/8 identical" $L && grep -q "^delete exit 0\|stale check exit 0: 0 files" $L || { echo "smoke / swap did not pass: not deleting"; exit 1; }
rt=$(grep -o "decode [0-9.]* tok/s" $L | tail -n 1 | awk '{print $2}')
python3 -c "import sys; sys.exit(0 if float('${rt:-0}') >= 120 else 1)" || { echo "real text $rt tok/s below 120: not deleting"; exit 1; }
python3 - <<'PY' || { echo "official / v1 check failed: not deleting"; exit 1; }
import json, urllib.request
off = json.load(urllib.request.urlopen("https://huggingface.co/api/models/raahemnabeel/qwen3-5-122b-a10b"))
v1 = json.load(urllib.request.urlopen("https://huggingface.co/api/models/raahemnabeel/qwen3-5-122b-a10b-v1"))
m = json.loads(urllib.request.urlopen("https://huggingface.co/raahemnabeel/qwen3-5-122b-a10b-v1/raw/main/tt_kernel_manifest.json").read())
print("official likes", off.get("likes"), "v1 files", len(v1["siblings"]), "v1 image", m.get("container", {}).get("built", {}).get("image"))
assert len(v1["siblings"]) == 721
PY
hf repos delete raahemnabeel/qwen3-5-122b-a10b-blackhole-fast -y 2>&1 | tail -n 2; echo "delete repo exit $?"
