#!/bin/bash
# 122B step 7 again with the whole server log kept (q122s7 died at boot; the saved tail held only the API traceback):
# full step-7 env; if it dies, step 7 without the fabric boundary (rows top-k + the GDN head value split).
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH MODEL_ID=Qwen/Qwen3.5-122B-A10B CACHE_NAME=qwen3-5-122b-a10b
R=$HOME/ornith-fusion/runs
T=$HOME/tt-metal-q122/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
IMG=tt-model/qwen3-5-122b-a10b-blackhole-fast-v2:3a9fa14cc3dd
while systemctl --user is-active -q p300-fix || chips_busy; do sleep 60; done
echo "q122 s7b start $(date +%T)"
OPS="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=1 --env QWEN35_ATTN_DECODE_OPS=1 --env QWEN35_MTP=0 --env VLLM_ENGINE_READY_TIMEOUT_S=14400"
FB="--env QWEN35_FABRIC_BOUNDARY=1 --env QWEN35_FB_MAX_ROWS=4 --env QWEN35_FB_CONSUMER=1 --env QWEN35_FB_ROUTER=1 --env QWEN35_MOE_SPLIT_PAIR=1"
ROWS="128:128:1:8 128:128:2:8 128:128:4:16 128:128:8:32 128:128:32:128 2048:128:1:4 8192:64:1:2"
for V in full nofb; do
  L=q122s7$V; rm -rf $R/$L; mkdir -p $R/$L
  X="$OPS --env QWEN35_ROWS_TOPK=1"; [ $V = full ] && X="$X $FB"
  ( for i in $(seq 1 120); do docker ps --format '{{.Names}}' | grep -qx q122-$V && break; sleep 5; done
    docker logs -f q122-$V > $R/$L/server_full.log 2>&1 ) &
  MODELS=1 GARBLE=1 KEEP=1 ROWS="$ROWS" $T/milestone.sh $L q122-$V $IMG p300x2 $X && {
    docker logs q122-$V 2>&1 | grep -E "rows top-k|fabric boundary" | sort -u | head -4
    URL=http://localhost:20000 python3 $T/realtext.py 384 | tee $R/$L/realtext.txt | tail -n 1
    docker rm -f q122-$V >/dev/null
    echo "== q122 greedy card_q122v2_p300x2 vs $L"; python3 $T/greedy.py compare card_q122v2_p300x2 $L 2>&1 | tail -n 3
    break; }
  grep -vE "DEBUG|riscv" $R/$L/server_full.log | grep -E "TT_FATAL|TT_THROW|Error|error|clash|Traceback" | head -8
  docker rm -f q122-$V >/dev/null 2>&1
done
echo "q122 s7b done $(date +%T)"
