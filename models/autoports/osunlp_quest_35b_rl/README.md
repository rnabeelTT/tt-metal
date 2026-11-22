# osunlp/QUEST-35B-RL on Blackhole

QUEST-35B-RL is a Qwen3.5-35B-A3B fine-tune with the same `qwen3_5_moe` architecture, tensor names, vision tower and
image processor as Ornith-1.5-35B-A3B. It is served by the Ornith autoport unchanged
(`models/autoports/ornith_ai_ornith_1_5_35b_a3b`, fused decode of its `fusion/HANDOFF.md`), with `HF_MODEL` set to this
checkpoint. Only the package manifest is new.

- `tt-model.yaml`: package manifest (`tt-model package --container models/autoports/osunlp_quest_35b_rl/tt-model.yaml`,
  published as `raahemnabeel/quest-35b-rl`). Ornith's profiles and precision: bfp8 routed experts on two and four
  dies, bfp4 on one die; vision on two and four dies.
  Differences from Ornith's manifest:
  - MTP is off on every profile, because the checkpoint has no `mtp.*` tensors.
  - Prefix caching is off on one die. Ornith never ran it there, because MTP kept it off.
  - The sampling defaults are T 1.0 / top_p 0.95 / top_k 20, because the checkpoint has no `generation_config.json`.
- `smoke.py`: bring-up smoke test against a running server. It checks a thinking answer, a thinking-off answer, a
  parsed tool call, and an image question on a generated picture.
- `card_perf.py`: builds the card's performance tables from the bring-up runs.
- Bring-up chain: `~/ornith-fusion/runs/quest/quest_chain.sh` on qb2-120-p06t08. It serves each profile as shipped,
  then runs smoke, greedy, bench rows, real-text decode and the 128-request image burst, and finally thinking-off
  evals on p300.
