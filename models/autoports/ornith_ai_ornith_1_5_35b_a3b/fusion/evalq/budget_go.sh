#!/bin/bash
# after ornfix: rewrite the queue for the 81,920 budget, resume; release the held q122 lines once the orn reruns start
R=$HOME/ornith-fusion/runs; cd $R
while systemctl --user is-active -q ornfix; do sleep 30; done
cp evalq.txt evalq.txt.bak5
ok=0; grep -q 'orn hle requeued' orn_fix_chain.log && ok=1
python3 budget_requeue.py $ok && echo "queue rewritten (orn ok=$ok) $(date +%T)"
rm -f evalq.pause; echo "queue resumed $(date +%T)"
n0=$(wc -l < evalq.log)
until tail -n +$((n0+1)) evalq.log | grep -qE '^== (orn|cn) '; do sleep 60; done
python3 - <<'PY'
import fcntl
with open("evalq.txt", "r+") as f:
    fcntl.flock(f, fcntl.LOCK_EX); s = f.read().replace("#HOLD-Q122 ", ""); f.seek(0); f.truncate(); f.write(s)
PY
echo "q122 lines released $(date +%T)"
