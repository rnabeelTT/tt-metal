#!/bin/bash
# 32-bit rows top-k (ornith_fusion c1f25770879 / coder_next_fusion bbc206b412e) into the fast bundles:
# (1) Coder-Next 2-die op test (75968 columns) + p300 served check from the tree (garble, greedy vs cnfast_p300g, rows,
# real text); (2) Ornith repackaged (name unchanged until a later repackage, user), p300 card rows as shipped (vs card_fast2_p300,
# the 16-bit guard) + a p300x2 spot check; (3) Coder-Next packaged, p300 card rows (vs card_cnfast_p300) + p300x2 spot.
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH
R=$HOME/ornith-fusion/runs; O=$HOME/tt-metal-bringups; C=$HOME/tt-metal-codernext
TO=$O/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools; TC=$C/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
CNIMG=tt-model/qwen3-coder-next:5190768f7a5a
while chips_busy; do sleep 60; done
echo "tk32 ports start $(date +%T)"
# (the Coder-Next 2-die op test passed in runs/tk32_ports_1.log: exact at B 1/2/8, 96 us vs 667 us)
E="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=1 --env QWEN35_ATTN_DECODE_OPS=1 --env QWEN35_ROWS_TOPK=1"
L=cntk_p300; rm -rf $R/$L
MODEL_ID=Qwen/Qwen3-Coder-Next CACHE_NAME=qwen3-coder-next-blackhole MODELS=1 GARBLE=1 KEEP=1 ROWS="128:128:1:8 128:128:8:32" $TC/milestone.sh $L cn-tk $CNIMG p300 $E && {
  docker logs cn-tk 2>&1 | grep -E "rows top-k" | sort -u | head -2
  URL=http://localhost:20000 python3 $TC/realtext.py 384 | tee $R/$L/realtext.txt | tail -n 1
  docker rm -f cn-tk >/dev/null; }
python3 -c "
import json; d=json.load(open('$R/$L/garble.json')); print('garble', d['summary']); print([(r['n_words'], r['text'][:50]) for r in d['rows']][:3])"
python3 $TC/greedy.py compare cnfast_p300g $L 2>&1 | grep -E "^(sequential|concurrent):"
(cd $O && tt-model package --container models/autoports/ornith_ai_ornith_1_5_35b_a3b/tt-model-fast.yaml > $R/package_ofast4.log 2>&1; echo "ornith package exit $?")
NEW=$O/build/ornith-1.5-35b-a3b-blackhole-fast/tt_kernel_manifest.json
python3 -c "import json; print('image', json.load(open('$NEW'))['container']['built']['image'])"
rm -rf $R/card_fast3_p300 $R/card_fast3_p300x2
GREEDY=1 $TO/card_bench.sh card_fast3_p300 $NEW p300
python3 $TO/greedy.py compare card_fast2_p300 card_fast3_p300 2>&1 | grep -E "^(sequential|concurrent):"
GREEDY=1 $TO/card_bench.sh card_fast3_p300x2 $NEW p300x2 128:128:1:4 128:128:8:32 1024:256:32:128
python3 $TO/greedy.py compare card_ov3_p300x2 card_fast3_p300x2 2>&1 | grep -E "^(sequential|concurrent):"
(cd $C && tt-model package --container models/autoports/qwen_qwen3_coder_next/tt-model-fast.yaml > $R/package_cnfast3.log 2>&1; echo "cn package exit $?")
NEW=$C/build/qwen3-coder-next/tt_kernel_manifest.json
python3 -c "import json; print('image', json.load(open('$NEW'))['container']['built']['image'])"
rm -rf $R/card_cnfast3_p300 $R/card_cnfast3_p300x2
GREEDY=1 $TC/card_bench.sh card_cnfast3_p300 $NEW p300
python3 $TC/greedy.py compare card_cnfast_p300 card_cnfast3_p300 2>&1 | grep -E "^(sequential|concurrent):"
GREEDY=1 $TC/card_bench.sh card_cnfast3_p300x2 $NEW p300x2 128:128:1:4 128:128:8:32 1024:256:32:128
python3 $TC/greedy.py compare card_cnfast_p300x2 card_cnfast3_p300x2 2>&1 | grep -E "^(sequential|concurrent):"
echo "tk32 ports done $(date +%T)"
