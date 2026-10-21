#!/bin/bash
# stage-3 review #5: ONE ticket on a pure git archive of COMMIT: full stage-3 suite, watcher subset, B=1 watcher decode
COMMIT=$1; source ~/glm53-flash-bringup/env.sh >/dev/null; cd $TT_METAL_HOME
L=~/glm53-flash-bringup/logs; ST=$L/s3r5.status; M=models/autoports/zai_org_glm_5_3_flash
SN=$HOME/glm53-flash-bringup/s3r5_snapshot; rm -rf $SN; mkdir -p $SN
git -C $TT_METAL_HOME archive $COMMIT models conftest.py pytest.ini | tar -x -C $SN
git -C $TT_METAL_HOME rev-parse $COMMIT > $SN/BASE_COMMIT
(cd $SN/$M && md5sum tt/optimized_decoder.py tt/dsa.py tt/multichip_decoder.py tt/moe.py tt/moe_opt.py tests/test_optimized_decoder.py) > $SN/MD5SUMS
echo "$(date -Is) snapshot $SN pure archive of $(cat $SN/BASE_COMMIT): $(tr '\n' ' ' < $SN/MD5SUMS)" >> $ST
W=$HOME/glm53-flash-bringup/s3r5_watch; rm -rf $W; mkdir -p $W/suite $W/decode1
cat > $SN/run.sh <<EOS
set -x
export PYTHONPATH=$SN:\$PYTHONPATH GLM53_PERF_SEED=0
P="$TT_METAL_HOME/python_env/bin/pytest -sv -p no:cacheprovider"; T=$SN/$M/tests
WENV="GLM53_WORKER_L1_SIZE=1444864 TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_LOGGER_LEVEL=Warning"
cd $SN
echo "##### SUITE"; \$P \$T/test_optimized_decoder.py; echo "##### SUITE rc=\$?"
env -u PYTHONPATH -u VIRTUAL_ENV /home/raahem/.tenstorrent-venv/bin/tt-smi -r >/dev/null 2>&1
echo "##### WATCHER_SUITE"; env \$WENV TT_METAL_LOGS_PATH=$W/suite \$P \$T/test_optimized_decoder.py -k 'test_real_weights or test_batched_users or test_batch32_all_active or test_inactive_slot_state or test_decode_only or test_single_slot_decode or test_single_slot_sparse_decode or (test_lengths and (kda_dense-2049 or dsa_moe-2051 or kda_moe-2049 or dsa_moe-1))'; echo "##### WATCHER_SUITE rc=\$?"
env -u PYTHONPATH -u VIRTUAL_ENV /home/raahem/.tenstorrent-venv/bin/tt-smi -r >/dev/null 2>&1
echo "##### WATCHER_DECODE1"; env \$WENV GLM53_DECODER=optimized TT_METAL_LOGS_PATH=$W/decode1 \$P "\$T/test_perf_decoder.py::test_perf_traced_decode[kda_dense-1]" "\$T/test_perf_decoder.py::test_perf_traced_decode[kda_moe-1]" "\$T/test_perf_decoder.py::test_perf_traced_decode[dsa_moe-1]"; echo "##### WATCHER_DECODE1 rc=\$?"
EOS
~/glm53-flash-bringup/scripts/run-locked.sh $L/s3r5_gate.log timeout -k 60 9000 bash $SN/run.sh
for k in suite decode1; do echo "$(date -Is) watcher $k suspicious: $(grep -ciE 'error|assert|sanitiz|overflow|exception|race|tripped|pending' $W/$k/generated/watcher/watcher.log 2>/dev/null)" >> $ST; done
echo "$(date -Is) results: $(grep -E '^=+ .*(passed|failed)|##### .* rc=' $L/s3r5_gate.log | tr '\n' ' ')" >> $ST
echo "$(date -Is) s3r5 done" >> $ST
