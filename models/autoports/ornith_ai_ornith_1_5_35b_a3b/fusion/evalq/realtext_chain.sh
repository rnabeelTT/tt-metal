#!/bin/bash
# real-text single-user decode: reference MTP on, new MTP on, new MTP off (one measured at a time)
T=$HOME/tt-metal-bringups/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
IMG=tt-model/ornith-1.5-35b-a3b-v2:0fe927ac6e21
wait_up() { until curl -sf localhost:$1/v1/models >/dev/null; do docker ps --format '{{.Names}}' | grep -qx $2 || { echo "$2 died"; return 1; }; sleep 10; done; }
DEVS=0,1 PORT=20000 $T/serve.sh ornith-ref $IMG p300 >/dev/null
MODELS=1 DEVS=2,3 PORT=20001 $T/serve.sh ornith-dev $IMG p300 --env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 >/dev/null
wait_up 20000 ornith-ref && wait_up 20001 ornith-dev
URL=http://localhost:20000 python3 $T/realtext.py 384 | sed 's/^/ref MTP on: /'
URL=http://localhost:20001 python3 $T/realtext.py 384 | sed 's/^/new MTP on: /'
MODELS=1 DEVS=2,3 PORT=20001 $T/serve.sh ornith-dev $IMG p300 --env QWEN35_MTP=0 --env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 >/dev/null
wait_up 20001 ornith-dev
URL=http://localhost:20001 python3 $T/realtext.py 384 | sed 's/^/new MTP off: /'
docker exec ornith-dev true
