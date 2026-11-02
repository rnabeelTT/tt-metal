#!/bin/bash
# After the Qwen3.5-122B HLE generation: pause evalq, release Ornith-1.5-35B with the image-split scheduler fix (gated, incl. MM burst),
# then requeue the Ornith-1.5-35B HLE run (it died on 2026-11-02 16:49: published image 9f0546abcdb8 lacks plugin 8b1f8b1).
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH
R=$HOME/ornith-fusion/runs; cd $R
touch evalq.pause; echo "pause requested $(date +%T)"
until grep -q '^paused, stopping' <(tail -n 3 evalq.log); do sleep 30; done
echo "queue paused $(date +%T)"
./mmfix_release_pub.sh orn
P=$HOME/.cache/tt-model/pulled/raahemnabeel__ornith-1.5-35b-a3b/tt_kernel_manifest.json
python3 -c "import json,sys; print('pulled image', json.load(open(sys.argv[1]))['container']['built']['image'])" $P
if grep -q '^MM_BURST ok=128/128' mmfix_hub_orn/mm_burst.txt 2>/dev/null; then
  mv ref_orn_hle ref_orn_hle_crashed_1649
  python3 - <<'PY'
import fcntl
with open('evalq.txt','r+') as f:
    fcntl.flock(f, fcntl.LOCK_EX); L=f.read().split('\n')
    i=[k for k,l in enumerate(L) if l.startswith('q122 ')][0]
    L.insert(i,'orn text2 hle SAMPLE_N=128 MAX_TOKENS=32768 CONC=32')
    f.seek(0); f.truncate(); f.write('\n'.join(L))
PY
  echo "orn hle requeued; queue stays paused for the token-budget decision (user, 2026-11-02) $(date +%T)"
else echo "release not verified: queue stays paused"; fi
