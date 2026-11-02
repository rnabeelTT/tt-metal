# Box scripts (snapshot of ~/ornith-fusion/runs on qb2-120-p06t08)

One-off chains, release / smoke scripts and the eval queue as they ran on the 4-die Blackhole box. Paths are the box's
(`$HOME/ornith-fusion/runs`, `$HOME/evals`, worktrees `$HOME/tt-metal-*`); adapt them on another machine.

- Queue: `../tools/evalq.sh` reads `evalq.txt` (one task per line: `<model key> <runner> <bench> [KEY=VAL ...]`;
  `#`-prefixed lines are skipped; `evalq.pause` pauses at a run boundary and frees the chips).
- Releases: `mmfix_release_pub.sh` (gate vs the published image: greedy 8/8 seq + conc, a 1-user row, a 128-request
  image burst; push; fresh pull; hub smoke), `cards_release.sh`, `pc_release.sh`.
- `o122p8/`, `o122s7/`: Ornith-1.5-122B precision variants (8-bit attention / GDN / lm_head) served from the released
  image with a policy, env and a mounted `tp_common.py` (see REF_EVALS.md, numerics section).
