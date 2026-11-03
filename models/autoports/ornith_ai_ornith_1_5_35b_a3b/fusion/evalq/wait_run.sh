#!/bin/bash
# wait_run.sh <run label>: block until that run's own log has its RESULT / JUDGE_PENDING line or a TIMEOUT, or the
# queue reports a server death after this script started. (Matching evalq.log alone catches old lines; TIMEOUT lands mid-line after a progress bar.)
R=$HOME/ornith-fusion/runs; L=$R/$1.log; n0=$(wc -l < $R/evalq.log)
until grep -qE "^(RESULT|JUDGE_PENDING)|TIMEOUT after" $L 2>/dev/null || tail -n +$((n0+1)) $R/evalq.log | grep -q 'server died'; do sleep 60; done
date +%T; grep -E "^(RESULT|JUDGE_PENDING)|TIMEOUT after" $L 2>/dev/null; tail -n +$((n0+1)) $R/evalq.log | grep 'server died'; tail -n 2 $L | cut -c1-200
