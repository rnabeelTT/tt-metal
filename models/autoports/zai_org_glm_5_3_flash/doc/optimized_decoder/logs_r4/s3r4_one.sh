#!/bin/bash
# stage-3 review #4: ONE decisive case per run-locked.sh ticket (QPRIO unset), on a pinned snapshot = git archive HEAD
# + the stage-3 edits of the live tree (md5s logged). usage: s3r4_one.sh NAME [args]
#   ab | conv | suite | watcher_suite | watcher_decode1 | tracy POLICY CASE [POS]   (POLICY "-" = SELECTED)
source ~/glm53-flash-bringup/env.sh >/dev/null; cd $TT_METAL_HOME
NAME=$1; shift
case "$NAME $*" in "tracy - test_perf_prefill["*"-128]"*) echo "$(date -Is) skipped (128-row prefill dropped to save device time): $*" >> ~/glm53-flash-bringup/logs/s3r4.status; exit 0 ;; esac
L=~/glm53-flash-bringup/logs; ST=$L/s3r4.status; LIVE=$TT_METAL_HOME/models/autoports/zai_org_glm_5_3_flash
M=models/autoports/zai_org_glm_5_3_flash
FILES="tt/optimized_decoder.py tests/test_optimized_decoder.py tests/test_kda_decode_conv_one_die.py tools/s3r4_ab.py"
TAG=$NAME; [ "$NAME" = tracy ] && TAG=tracy_${1}_$(echo "$2" | tr -c 'a-zA-Z0-9_-' '_')${3:+_pos$3}
SN=$HOME/glm53-flash-bringup/s3r4_snapshot_$TAG; rm -rf $SN; mkdir -p $SN
git -C $TT_METAL_HOME archive HEAD models conftest.py pytest.ini | tar -x -C $SN
for f in $FILES; do cp $LIVE/$f $SN/$M/$f; done
git -C $TT_METAL_HOME rev-parse HEAD > $SN/BASE_COMMIT
(cd $SN/$M && md5sum $FILES) > $SN/MD5SUMS
echo "$(date -Is) $TAG snapshot $SN base $(cat $SN/BASE_COMMIT): $(tr '\n' ' ' < $SN/MD5SUMS)" >> $ST
export PYTHONPATH=$SN:$PYTHONPATH GLM53_PERF_SEED=0
T=$SN/$M/tests; P="$TT_METAL_HOME/python_env/bin/pytest -sv -p no:cacheprovider"; PY=$TT_METAL_HOME/python_env/bin/python
D=$LIVE/doc/optimized_decoder; R=~/glm53-flash-bringup/scripts/run-locked.sh; WD=$HOME/glm53-flash-bringup/s3r4_watch
WENV="GLM53_WORKER_L1_SIZE=1444864 TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_LOGGER_LEVEL=Warning"
st() { echo "$(date -Is) $1: $(grep -E '^=+ .*(passed|failed|error)|tripped assert|=== exit code' $2 | tail -2 | tr '\n' ' ')" >> $ST; }
wsave() {  # watcher evidence into the doc (small files only)
  W=$1; out=$2; rm -rf $out; mkdir -p $out
  cp $W/generated/watcher/kernel_names.txt $out/ 2>/dev/null
  gzip -c $W/generated/watcher/watcher.log > $out/watcher.log.gz
  echo "$(date -Is) $TAG watcher suspicious lines: $(cat $W/generated/watcher/watcher.log 2>/dev/null | grep -ciE 'error|assert|sanitiz|overflow|exception|race|tripped|pending')" >> $ST
}
case $NAME in
ab) $R $L/s3r4_ab.log timeout -k 60 3600 bash -c "cd $SN && $PY -m models.autoports.zai_org_glm_5_3_flash.tools.s3r4_ab"
  st ab $L/s3r4_ab.log ;;
conv) $R $L/s3r4_conv_one_die.log timeout -k 60 1800 bash -c "cd $SN && $P $T/test_kda_decode_conv_one_die.py"
  st conv $L/s3r4_conv_one_die.log ;;
suite) $R $L/s3r4_suite.log timeout -k 60 14400 bash -c "cd $SN && $P $T/test_optimized_decoder.py"
  st suite $L/s3r4_suite.log ;;
watcher_suite) W=$WD/suite; rm -rf $W; mkdir -p $W
  $R $L/s3r4_watcher_suite.log timeout -k 60 10800 bash -c "cd $SN && env $WENV TT_METAL_LOGS_PATH=$W $P $T/test_optimized_decoder.py -k 'test_real_weights or test_batched_users or test_batch32_all_active or test_inactive_slot_state or test_decode_only or test_single_slot_decode or test_single_slot_sparse_decode or (test_lengths and (kda_dense-2049 or dsa_moe-2051 or kda_moe-2049 or dsa_moe-1))'"
  st watcher_suite $L/s3r4_watcher_suite.log; wsave $W $D/watcher_correctness_r4/suite ;;
watcher_decode1) W=$WD/decode1; rm -rf $W; mkdir -p $W
  $R $L/s3r4_watcher_decode1.log timeout -k 60 5400 bash -c "cd $SN && env $WENV GLM53_DECODER=optimized TT_METAL_LOGS_PATH=$W $P '$T/test_perf_decoder.py::test_perf_traced_decode[kda_dense-1]' '$T/test_perf_decoder.py::test_perf_traced_decode[kda_moe-1]' '$T/test_perf_decoder.py::test_perf_traced_decode[dsa_moe-1]'"
  st watcher_decode1 $L/s3r4_watcher_decode1.log; wsave $W $D/watcher_correctness_r4/decode1 ;;
tracy) POL=$1; CASE=$2; POS=$3; sp=PERF_DECODE; case $CASE in *prefill*) sp=PERF_PREFILL ;; esac
  out=$D/candidates_r4/$([ "$POL" = - ] && echo final || echo $POL)/$(echo "$CASE" | sed 's/.*:://' | tr '[]' '__')${POS:+_pos$POS}; mkdir -p $out
  PENV="GLM53_DECODER=optimized"; [ "$POL" != - ] && PENV="$PENV GLM53_OPT=$POL"; [ -n "$POS" ] && PENV="$PENV GLM53_PERF_POS=$POS"
  $R $L/s3r4_tracy.log timeout -k 60 2400 bash -c "rm -rf $TT_METAL_HOME/generated/profiler/reports; cd $SN && env $PENV $PY -m tracy -r -p -v -m pytest '$T/test_perf_decoder.py::$CASE' -s -p no:cacheprovider > $out/console.log 2>&1"
  csv=$(ls -t $TT_METAL_HOME/generated/profiler/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1)
  if [ -n "$csv" ]; then cp "$csv" $out/ops.csv
    tt-perf-report $out/ops.csv --start-signpost $sp --end-signpost ${sp}_END --csv $out/perf_report.csv > $out/perf_report.console.log 2>&1
    tt-perf-report $out/ops.csv --start-signpost $sp --end-signpost ${sp}_END --no-summary > $out/perf_report.txt 2>&1
    rm -f $out/*.png; gzip -f $out/ops.csv; fi
  echo "$(date -Is) $TAG done: $(grep -h '^PERF' $out/console.log | tr '\n' ' ')" >> $ST ;;
esac
