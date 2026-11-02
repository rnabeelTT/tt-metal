#!/bin/bash
# the blackhole-fast card: every bundle x profile, one at a time, after the box is idle
export PATH=$HOME/ornith-fusion/bin:$PATH
T=$HOME/tt-metal-bringups/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
NEW=$HOME/tt-metal-bringups/build/ornith-1.5-35b-a3b-blackhole-fast/tt_kernel_manifest.json
PUB=$HOME/.cache/tt-model/pulled/raahemnabeel__ornith-1.5-35b-a3b/tt_kernel_manifest.json
while [ -n "$(docker ps -q)" ]; do sleep 30; done
echo "box idle $(date +%T)"
for P in p300x2 p300 p150; do
  $T/card_bench.sh card_new_$P $NEW $P
  $T/card_bench.sh card_pub_$P $PUB $P
done
echo "card chain done $(date +%T)"
