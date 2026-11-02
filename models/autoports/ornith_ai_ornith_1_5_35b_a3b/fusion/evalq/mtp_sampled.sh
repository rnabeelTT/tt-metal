#!/bin/bash
# MTP decision, measurement 1: draft acceptance under Ornith's sampling settings (generation_config: T 1.0, top-k 20,
# top-p 0.95; and T 0.6) vs greedy, real text 1 user, on the fused build + QWEN35_MTP=3; then the same three on the
# plain fused server (MTP off) for the baseline tok/s. Acceptance per phase from MTP_STATS deltas (logged every 32 passes).
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH MODEL_ID=ornith-ai/Ornith-1.5-35B-A3B
R=$HOME/ornith-fusion/runs; T=$HOME/tt-metal-bringups/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
IMG=tt-model/ornith-1.5-35b-a3b-blackhole-fast:a7e5b6904100
while systemctl --user is-active -q final-hub-smoke2 || chips_busy; do sleep 60; done
echo "mtp sampled start $(date +%T)"
E="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=1 --env QWEN35_ATTN_DECODE_OPS=1 --env QWEN35_ROWS_TOPK=1"
FB="--env QWEN35_FABRIC_BOUNDARY=1 --env QWEN35_FB_MAX_ROWS=4 --env QWEN35_FB_CONSUMER=1 --env QWEN35_FB_ROUTER=1 --env QWEN35_MOE_SPLIT_PAIR=1"
stats() { docker logs $1 2>&1 | grep "MTP_STATS" | tail -n 1 | python3 -c "
import re,sys; s=sys.stdin.read(); m=re.search(r'(\d+) passes, (\d+) tokens.*hist \[([^]]*)\]', s)
print(m.group(1), m.group(2), m.group(3).replace(' ','')) if m else print('0 0 0')"; }
phases() {  # <container> <label> <mtp 0/1>
  for S in "" "1.0,20,0.95" "0.6,20,0.95"; do
    [ $3 = 1 ] && b=($(stats $1))
    URL=http://localhost:20000 SAMPLE=$S python3 $T/realtext.py 384 | tail -n 1 | tee -a $R/$2/realtext.txt
    if [ $3 = 1 ]; then
      a=($(stats $1))
      python3 -c "
p=$((${a[0]}-${b[0]})); t=$((${a[1]}-${b[1]})); h=[x-y for x,y in zip([${a[2]}],[${b[2]}])]
print(f'MTPACC ${S:-greedy}: {p} passes, {t} tokens = {t/max(p,1):.2f} per pass, hist {h}')" | tee -a $R/$2/realtext.txt
      docker logs $1 2>&1 | grep MTP_STAGES | tail -n 1 | grep -o "MTP_STAGES.*" | tee -a $R/$2/realtext.txt
    fi
  done
}
L=mtpsamp_k3; rm -rf $R/$L
MODELS=1 KEEP=1 ROWS="128:128:1:4" $T/milestone.sh $L orn-mtp $IMG p300x2 $E $FB --env QWEN35_MTP=3 && phases orn-mtp $L 1
docker rm -f orn-mtp >/dev/null 2>&1
L=mtpsamp_off; rm -rf $R/$L
MODELS=1 KEEP=1 ROWS="128:128:1:4" $T/milestone.sh $L orn-off $IMG p300x2 $E $FB && phases orn-off $L 0
docker rm -f orn-off >/dev/null 2>&1
echo "mtp sampled done $(date +%T)"
