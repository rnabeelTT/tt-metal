#!/usr/bin/env bash
# Offline check of one bench through the real dispatcher against the local stub:
#   verify.sh <bench> [port]
# prints one rendered prompt, then RESULT lines for stub modes gold (expect ~100),
# random (expect ~chance) and junk (expect ~0). Never touches a model server.
set -uo pipefail
B=$1 PORT=${2:-28123}
H=$(cd "$(dirname "$0")" && pwd)
ROOT=${VERIFY_ROOT:-$H/verify_runs}
cd "$H"
echo "===== $B: example item"
.venv/bin/python - "$B" <<'EOF'
import json, sys
b = sys.argv[1]
it = json.loads(open(f"data/{b}/items.jsonl").readline())
m = json.load(open(f"data/{b}/meta.json"))
print(f"items={m['n_items']} full={m.get('full_size')} sample={m.get('sample')} groups={len(m['groups'])}")
print("--- id:", it["id"], "| group:", it["group"], "| gold:", str(it.get("gold"))[:200])
for msg in it["messages"]:
    print(f"--- [{msg['role']}]\n{msg['content']}")
EOF
for mode in gold random junk; do
  .venv/bin/python -m qt.stub_server $PORT "$B" $mode & SP=$!
  for i in $(seq 50); do curl -sf localhost:$PORT/v1/models >/dev/null && break; sleep 0.2; done
  rm -rf "$ROOT/$mode/$B"
  RUNS_ROOT=$ROOT WMT_SCORE=${WMT_SCORE:-1} "$H/../qwen_text_run.sh" "$B" "$mode" "http://127.0.0.1:$PORT/v1" Qwen/Qwen3.5-122B-A10B 32 2>"$ROOT/$B.$mode.stderr" | sed "s/^/[$mode] /"
  kill $SP; wait $SP 2>/dev/null
done
