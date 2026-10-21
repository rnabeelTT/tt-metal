#!/usr/bin/env bash
# Stage 9 / 10: vLLM server of the GLM-5.3-Flash (E192) port straight from the tree (the bundle's flags), then the
# readiness stages and the served smokes against it. Profile numbers come from serving_profiles.json (never constants).
#
# Env: PROFILE_NAME (p300x2 | long | wide | reduced | any profile name of serving_profiles.json), SEQS, MLEN (override
#      the profile), STAGES (readiness stages, default sampling,qualitative,benchmark), SPROFILE (smoke | full sampling
#      profile), TAG, HOLD=1 (keep the server up; the device lock stays held until the server exits), NOSTAGES=1,
#      SMOKE=1 (run 09-smoke.py), SMOKE_ARGS, BURST, LAYERS=0,3,4 (reduced serving target: GLM53_LAYERS), PORT,
#      ASYNC=0 (turn supports_async_decode off; default on, the adapter default), TRACE_BYTES,
#      EXTRA_ARGS, TMO (readiness timeout s), READY_TMO.
# Device access: through run-locked.sh (FIFO ticket queue, tt-smi -r before and after). The script re-executes itself
# under the lock.
source "$HOME/glm53-flash-bringup/env.sh" >/dev/null; cd "$TT_METAL_HOME" || exit 1
MD=models/autoports/zai_org_glm_5_3_flash
SPGET="python -m models.autoports.zai_org_glm_5_3_flash.tools.serving_profiles get"
case "${PROFILE_NAME:-p300x2}" in
  long) PN=p300x2-long;; wide) PN=p300x2-wide;; 1m) PN=p300x2-1m;; reduced) PN=reduced;; *) PN=${PROFILE_NAME:-p300x2};;
esac
TAG=${TAG:-$PN}; L=$BRINGUP_ROOT/logs/09; mkdir -p $L
if [ "${GLM53_LOCKED:-0}" != 1 ]; then
  exec env GLM53_LOCKED=1 PROFILE_NAME="$PN" TAG="$TAG" $BRINGUP_ROOT/scripts/run-locked.sh $L/serve-$TAG.chain.log "$0" "$@"
fi
# SNAP=<dir>: run from a pinned snapshot (git archive + uncommitted autoport files); it goes first on PYTHONPATH
if [ -n "${SNAP:-}" ]; then cd "$SNAP" || exit 1; export PYTHONPATH="$SNAP:$PYTHONPATH"; fi
echo "=== cwd $PWD base $(cat BASE_COMMIT 2>/dev/null || git rev-parse HEAD)"; md5sum $MD/tt/*.py | sed 's/^/md5 /'
export TT_METAL_LOGGER_LEVEL=${TT_METAL_LOGGER_LEVEL:-Error} MESH_DEVICE=P300x2 HF_MODEL=zai-org/GLM-5.3-Flash
# glm5_next needs transformers >= 5.16; python_env has 5.12.1 -> the serving overlay goes first on PYTHONPATH
# (built once: pip install --target $BRINGUP_ROOT/serve_overlay --no-deps transformers==5.19.0 tokenizers==0.23.3 huggingface_hub==2.2.0)
export PYTHONPATH="$BRINGUP_ROOT/serve_overlay:$PYTHONPATH"
export EXTRA_MODELS_DIR=$PWD/$MD/vllm_bundle
export GLM53_CKPT GLM53_PROGRAM_GUARD=${GUARD:-1} VLLM_ENGINE_READY_TIMEOUT_S=${READY_TMO:-3600}
export GLM53_CACHE_DIR=${GLM53_CACHE_DIR:-$HOME/.cache/glm53_flash}   # warm load ~110 s (cold ~25 min)
export GLM53_VLLM_ASYNC_DECODE=${ASYNC:-1}   # the adapter default (stage 9 review #1 P1-1)
# GLM53_PRECISION unset: the adapter installs the selected policy named in serving_profiles.json
# one-shot (status/wide_probe_pending): the final wide run first re-probes 32 slots with the wide profile's trace region
# (160 MiB, stage-10 MoE row buckets), ingests it into the live data file and copies it into the snapshot; a failed
# probe skips the wide server (no retry)
if [ "$TAG" = final-wide ] && [ -e $BRINGUP_ROOT/status/wide_probe_pending ]; then
  rm -f $BRINGUP_ROOT/status/wide_probe_pending
  echo "=== WIDE: 32-slot G9 probe first $(date -Is)"
  SNAP=$SNAP NOLOCK=1 INGEST=1 TMO=7200 $BRINGUP_ROOT/scripts/09-g9-probe.sh 32; prc=$?
  grep -E "decode buckets" $L/g9_s32.log | tail -2 | sed 's/^.*decode buckets/WIDE decode buckets/' | cut -c1-400
  env -u PYTHONPATH -u VIRTUAL_ENV /home/raahem/.tenstorrent-venv/bin/tt-smi -r > /dev/null 2>&1
  if [ $prc != 0 ]; then echo "WIDE: 32-slot probe failed (exit $prc): wide server skipped"; exit 3; fi
  cp $TT_METAL_HOME/$MD/serving_profiles.json $SNAP/$MD/serving_profiles.json
  cp $TT_METAL_HOME/$MD/tt-model.yaml $SNAP/$MD/tt-model.yaml
  echo "WIDE: probe ingested; profile now $($SPGET max_model_len p300x2-wide 2>/dev/null | tail -1) tokens"
fi
# the profile's trace region (stage 10: per profile, serving_profiles.trace_region), else the file-level value
TRB=${TRACE_BYTES:-$(python -c "
from models.autoports.zai_org_glm_5_3_flash.tt import serving_profiles as s
d = s.load()
f = getattr(s, 'trace_region', None)
print(f(d, '$PN') if f and '$PN' != 'reduced' else d['trace_region_bytes'])" 2>/dev/null | tail -1)}; export GLM53_TRACE_REGION=$TRB
if [ "$PN" = reduced ]; then   # inner loop only, never evidence
  SEQS=${SEQS:-4}; MLEN=${MLEN:-8192}; export GLM53_LAYERS=${LAYERS:-0,3,4}
else
  SEQS=${SEQS:-$($SPGET max_num_seqs $PN)}; MLEN=${MLEN:-$($SPGET max_model_len $PN)}
  # the profile's process settings (serving_profiles.json layouts.<layout>.env, e.g. GLM53_DSA_SHARD=1 for p300x2-1m)
  while IFS= read -r kv; do [ -n "$kv" ] && export "$kv" && echo "profile env: $kv"; done < <($SPGET env $PN 2>/dev/null | grep "=")
  [ -n "$SEQS" ] && [ -n "$MLEN" ] || { echo "unknown profile $PN (serving_profiles.json)"; exit 2; }
fi
[ -n "${LAYERS:-}" ] && export GLM53_LAYERS=$LAYERS
PORT=${PORT:-8131}
# stop only this script's own earlier server (process group recorded in the pid file); never pkill -f
stop_srv() { kill -TERM -- -$1 2>/dev/null; for i in $(seq 1 24); do kill -0 -- -$1 2>/dev/null || return 0; sleep 5; done; kill -KILL -- -$1 2>/dev/null; sleep 2; }
if [ -f $L/server-$TAG.pid ]; then OLD=$(cat $L/server-$TAG.pid); grep -qa api_server /proc/$OLD/cmdline 2>/dev/null && stop_srv $OLD; fi
ADDCFG='{"tt": {"sample_on_device_mode": "all", "trace_region_size": '$TRB', "l1_small_size": 24576, "fabric_config": "FABRIC_1D"}}'
echo "=== serve $TAG seqs=$SEQS mlen=$MLEN layers=${GLM53_LAYERS:-all} async=$GLM53_VLLM_ASYNC_DECODE trace=$TRB cache=$GLM53_CACHE_DIR ckpt=$GLM53_CKPT $(date -Is)"
# served under two names: the checkpoint path (the readiness runner sends --hf-model as the model name) and the HF id
SRVCMD=(python -m vllm.entrypoints.openai.api_server --model "$GLM53_CKPT" --served-model-name "$GLM53_CKPT" $HF_MODEL
  --tokenizer "$GLM53_CKPT" --hf-overrides '{"architectures": ["TTGlm5NextForConditionalGeneration"]}'
  --block-size 128 --max-num-seqs $SEQS --max-model-len $MLEN --port $PORT
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 ${EXTRA_ARGS:-} --additional-config "$ADDCFG")
{ echo "SERVER_CMD: $(printf '%q ' "${SRVCMD[@]}")"; env | grep -E "^(GLM53_|VLLM_|EXTRA_MODELS_DIR|MESH_DEVICE|HF_MODEL|TT_METAL_LOGGER)" | sort | sed 's/^/SERVER_ENV: /'; } | tee $L/servercmd-$TAG.txt
setsid "${SRVCMD[@]}" > $L/server-$TAG.log 2>&1 &
SRV=$!; echo $SRV > $L/server-$TAG.pid
trap 'stop_srv $SRV' EXIT; trap 'exit 143' TERM INT   # the setsid server is outside timeout's process group
for i in $(seq 1 ${BOOT_POLLS:-240}); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; kill -0 $SRV 2>/dev/null || break; sleep 15; done
if ! curl -sf http://127.0.0.1:$PORT/health >/dev/null; then
  echo "server did not come up"; grep -E "Error|Traceback|assert|TT_FATAL|TT_THROW|PROGRAM_GUARD" -A3 $L/server-$TAG.log | grep -v LOGGER | head -40 | cut -c1-300
  stop_srv $SRV; exit 1
fi
echo "server healthy $(date -Is)"
grep -E "GLM53 KV pool|DSA pools|warm-up phase|PROGRAM_GUARD|sampler lockstep|weight cache" $L/server-$TAG.log | cut -c1-260 | tail -10
RC=0
if [ "${NOSTAGES:-0}" != 1 ]; then
  # one runner invocation per stage: a failed sampling stage no longer skips qualitative / benchmark (the runner stops
  # at the first failing stage); RC = the first non-zero stage exit
  : > $L/readiness-$TAG.log
  for st in $(echo ${STAGES:-sampling,qualitative,benchmark} | tr , ' '); do
    timeout ${TMO:-14400} python -m readiness_check.run_vllm_server --model-dir $MD --hf-model "$GLM53_CKPT" \
      --stages $st --server-url http://127.0.0.1:$PORT --sampling-profile ${SPROFILE:-smoke} \
      --max-num-seqs $SEQS --max-model-len $MLEN --block-size 128 >> $L/readiness-$TAG.log 2>&1; r=$?
    echo "readiness $TAG stage $st exit=$r"; [ $RC = 0 ] && RC=$r
  done
  echo "readiness $TAG exit=$RC"
  grep -E "PASS|FAIL|passed|failed|Error|Traceback" $L/readiness-$TAG.log | grep -v LOGGER | cut -c1-220 | tail -15
  # run_vllm_server rewrites readiness_vllm/ on every run: snapshot it now
  rm -rf $L/readiness_vllm-$TAG; cp -r $MD/readiness_vllm $L/readiness_vllm-$TAG 2>/dev/null
fi
# QCAPTURE=<dir>: the served token ids of the 6 qualitative prompts (scripts/09-qcapture.py, P2-1)
if [ -n "${QCAPTURE:-}" ]; then python3 $BRINGUP_ROOT/scripts/09-qcapture.py http://127.0.0.1:$PORT $QCAPTURE 2>&1 | tee $L/qcapture-$TAG.log; fi
if [ -n "${SMOKE:-}" ]; then SMOKE_JSON=$L/smoke-$TAG.maxeffort.json python3 $BRINGUP_ROOT/scripts/09-smoke.py http://127.0.0.1:$PORT ${SMOKE_ARGS:-} 2>&1 | tee $L/smoke-$TAG.log; fi
# NEEDLE: served passkey retrieval (scripts/09-needle.py), default on for the p300x2-1m profile; NEEDLE_CASES = TOKENS:DEPTH
NEEDLE=${NEEDLE:-$([ "$PN" = p300x2-1m ] && echo 1 || echo 0)}
# stage-6 review addition: the final p300x2 run (replicated layout, DSA + MoE buckets and pool-key views active) also
# runs one needle at ~200K tokens, passkey at ~30% depth
if [ "$TAG" = final-p300x2 ] && [ -z "${NEEDLE_CASES:-}" ]; then NEEDLE=1; NEEDLE_CASES="200000:0.3"; fi
if [ "$NEEDLE" = 1 ]; then
  timeout ${NEEDLE_TMO:-14400} python3 $BRINGUP_ROOT/scripts/09-needle.py http://127.0.0.1:$PORT $L/needle-$TAG.json ${NEEDLE_CASES:-1040000:0.1 300000:0.5} 2>&1 \
    | tee $L/needle-$TAG.log | grep -E "NEEDLE|Error|Traceback" | cut -c1-300
fi
# DIAG=1: rerun the plugin tests named in DIAG_TESTS (node ids) and scripts/09-diag.py against this server
if [ "${DIAG:-0}" = 1 ]; then
  [ -n "${DIAG_TESTS:-}" ] && (cd $HOME/vllm-tt-plugin && timeout 3600 python -m pytest $DIAG_TESTS -v --tt-server-url=http://127.0.0.1:$PORT \
    --tt-model-name="$GLM53_CKPT" --tt-max-num-seqs $SEQS -p no:cacheprovider > $L/diag-tests-$TAG.log 2>&1; \
    echo "diag tests exit=$? (log $L/diag-tests-$TAG.log)"; grep -E "PASSED|FAILED|SKIPPED" $L/diag-tests-$TAG.log | tail -8)
  python3 $BRINGUP_ROOT/scripts/09-diag.py http://127.0.0.1:$PORT $L/diag-$TAG.json 2>&1 | tee $L/diag-$TAG.log | cut -c1-300
fi
# stage 10: serving perf sweep (the GLM-4.5-Air card methodology: random-token prompts of exact length, greedy,
# ignore_eos, closed-loop users); SWEEP_ARGS e.g. "--users 1,2,4,8,16 --inputs 128,1024,4096,16384,65536,131072,262144"
if [ -n "${SWEEP:-}" ]; then
  mkdir -p $BRINGUP_ROOT/artifacts/s10
  python3 $BRINGUP_ROOT/scripts/10-perf-sweep.py $BRINGUP_ROOT/artifacts/s10/sweep_$TAG.json --url http://127.0.0.1:$PORT \
    --model zai-org/GLM-5.3-Flash --tokenizer "$GLM53_CKPT" --profile $TAG --max-users $SEQS --max-model-len $MLEN \
    ${SWEEP_ARGS:-} 2>&1 | tee $L/sweep-$TAG.log | grep -vE "^\s*$" | tail -40
fi
[ "${HOLD:-0}" = 1 ] && { echo "HOLD: server pid $SRV running; stop it with kill -TERM -- -$SRV (process group; run-locked resets after)"; wait $SRV; exit 0; }
stop_srv $SRV
echo "=== done $(date -Is)"; exit $RC
