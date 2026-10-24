#!/bin/bash
# stage 12 correctness: the release accuracy gate (teacher forcing + prefill check, AIME24 refs p0/p1/p2, 1 slot x
# 243,712, shipped defaults: no GLM53_* overrides) on the shipped code (snapshot s12_snap3 = 8a51274702a; runtime code
# equal to the published image a6ceb44bd5b). Pass: min top-1 >= 0.91, top-5 and top-100 = 1.00 (stage-8 gate).
B=$HOME/glm53-flash-bringup; source $B/env.sh >/dev/null
S=$B/s12_snap3; A=$B/artifacts/s12/tf_release; mkdir -p $A
MD=$S/models/autoports/zai_org_glm_5_3_flash; R=$TT_METAL_HOME/models/autoports/zai_org_glm_5_3_flash/doc/datatype_sweep/refs
W=$HOME/.cache/tt-model/glm-5.3-flash-e192-blackhole/weights
PY=$TT_METAL_HOME/python_env/bin/python
CLEAN="-u GLM53_PRECISION -u GLM53_MC -u GLM53_DSA_BUCKETS -u GLM53_DSA_ROW_BUCKETS -u GLM53_SENT_ONCE -u GLM53_MOE_BUCKETS -u GLM53_DSA_VIEWS -u GLM53_LAYERS"
GLM53_CKPT=$B/s12_ckpt PYTHONPATH=$S $PY $B/scripts/s12/cache_guard.py $W/glm53_flash || exit 9
echo "=== tf release gate $(cat $S/BASE_COMMIT) $(date -u)"
env -C $S $CLEAN PYTHONPATH=$S:$TT_METAL_HOME:$TT_MODEL_BRINGUP_ROOT/runtime GLM53_TP=4 GLM53_CKPT=$B/s12_ckpt GLM53_CACHE_DIR=$W/glm53_flash \
  timeout -k 60 9000 $PY -m models.autoports.zai_org_glm_5_3_flash.tools.dtype_sweep --id s12_release --seq-len 243712 \
  --do ready,extra --extra-refs $R/readiness_aime24_chat_p1.refpt,$R/readiness_aime24_chat_p2.refpt \
  --reference $MD/readiness_aime24_chat.refpt --report-dir $A --out $A/record.json 2>&1 | grep -E "top1|Error|Traceback|ptcheck" | cut -c1-300
echo "rc ${PIPESTATUS[0]}"
python3 - $A/record.json <<'P'
import json, sys
d = json.load(open(sys.argv[1])); rows = {"p0": d}
rows.update({k.replace("readiness_aime24_chat_", ""): v for k, v in d.get("extra_refs", {}).items()})
ok = True
for name, r in rows.items():
    for path in ("teacher_forcing", "prefill_check"):
        x = r[path][0]; good = x["top1"] >= 0.91 and x["top5"] == 1.0 and x["top100"] == 1.0; ok &= good
        print(f"TFGATE {name} {path} top1 {x['top1']:.2f} top5 {x['top5']:.2f} top100 {x['top100']:.2f} {'ok' if good else 'FAIL'}")
print("TFGATE RESULT", "PASS" if ok else "FAIL", "| ptcheck", d.get("ptcheck", {}).get("equal", d.get("ptcheck")) if isinstance(d.get("ptcheck"), dict) else "")
P
echo "=== done $(date -u)"
