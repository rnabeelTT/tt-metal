#!/bin/bash
# smoke.sh <label> [env args]: boot the Ornith fast server (gdn_wip models/), one 64-token request with a 180 s timeout,
# report ok / HANG, remove the server and reset the chips on a hang.
export PATH=$HOME/ornith-fusion/bin:$PATH
L=$1; shift
T=$HOME/tt-metal-gdnwip/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
IMG=tt-model/ornith-1.5-35b-a3b-blackhole-fast:e7d45a33dd80
MODELS=1 $T/serve.sh orn-smoke $IMG p300x2 --env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_MTP=0 "$@" > /dev/null
until curl -sf localhost:20000/v1/models >/dev/null; do
  docker ps --format '{{.Names}}' | grep -qx orn-smoke || { echo "$L: server died"; docker logs orn-smoke 2>&1 | grep -E "TT_THROW|Error" | head -3; exit 1; }
  sleep 10
done
[ -n "${HOLD_ON_HANG:-}" ] && HOLD=1
r=$(curl -s --max-time ${REQ_TIMEOUT:-180} localhost:20000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"ornith-ai/Ornith-1.5-35B-A3B","prompt":"The capital of France is","max_tokens":64,"temperature":0}')
if echo "$r" | grep -q '"text"'; then
  echo "$L: ok: $(echo "$r" | python3 -c 'import json,sys; print(repr(json.load(sys.stdin)["choices"][0]["text"][:120]))')"
  docker rm -f orn-smoke >/dev/null
else
  echo "$L: HANG ($r)"
  if [ -n "${HOLD:-}" ]; then echo "$L: server kept for inspection"; exit 2; fi
  docker rm -f orn-smoke >/dev/null; ~/.tenstorrent-venv/bin/tt-smi -r >/dev/null 2>&1
fi
