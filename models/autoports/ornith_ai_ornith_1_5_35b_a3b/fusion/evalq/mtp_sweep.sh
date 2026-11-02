#!/bin/bash
# mtp_sweep.sh <prefix> <image> <MODELS 0|1> [docker env args...]: MTP parameters for one model on p300x2 (serve.sh;
# MODEL_ID / CACHE_NAME / T = the fusion tools dir from the caller). Per config one server:
#   off      QWEN35_MTP=0                          real text 1 user (+ 128/128 row)
#   a1..a4   k = 1..4, adaptive (MTP while ONE request is live; plain batched decode otherwise), 32 slots:
#            real text 1 user, 128/128 1 user, acceptance from the server log; a3 also real text at 2/4/8/16 users
#            (= plain fused decode, the comparison for multi-user MTP)
#   m1 m2 m3 k = 1 / 2 / 3, MTP over every live slot (QWEN35_MTP_ADAPTIVE=0; slots x (k + 1) <= 32: 16 / 10 / 8 slots):
#            real text at 2 / 4 / 8 (/ 16) users and 1 user
# Results: runs/<prefix>_<config>/ (realtext*.txt, acc.log), summary lines on stdout.
P=$1; IMG=$2; M=$3; shift 3
R=$HOME/ornith-fusion/runs
cfgs=("off:0:1:32" "a1:1:1:32" "a2:2:1:32" "a3:3:1:32" "a4:4:1:32" "m1:1:0:16" "m2:2:0:10" "m3:3:0:8")
for c in "${cfgs[@]}"; do
  IFS=: read N K A S <<< "$c"
  L=${P}_$N; mkdir -p $R/$L
  E="--env QWEN35_MTP=$K --env QWEN35_MTP_ADAPTIVE=$A --env VLLM_ENGINE_READY_TIMEOUT_S=14400"
  MAXSEQS=$S MODELS=$M GARBLE=0 KEEP=1 PORT=20000 ROWS="128:128:1:8" $T/milestone.sh $L mtp-srv $IMG p300x2 "$@" $E || { echo "$L: no server"; docker rm -f mtp-srv >/dev/null 2>&1; continue; }
  URL=http://localhost:20000 python3 $T/realtext.py 384 | tee $R/$L/realtext.txt
  case $N in
    a3) U="2 4 8 16";; m1) U="2 4 8 16";; m2|m3) U="2 4 8";; *) U="";;
  esac
  for u in $U; do CONC=$u URL=http://localhost:20000 python3 $T/realtext.py 384 | tee -a $R/$L/realtext_conc.txt; done
  docker logs mtp-srv 2>&1 | grep -E "acceptance hist" | tail -n 3 > $R/$L/acc.log; tail -n 1 $R/$L/acc.log | cut -c1-220
  docker rm -f mtp-srv >/dev/null
done
