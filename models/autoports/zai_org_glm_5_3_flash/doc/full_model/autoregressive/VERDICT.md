# Stage 6 readiness autoregressive check (full 45-layer model, TP4 p300x2)

Runner: `readiness_check.run_autoregressive` (prompt `autoregressive_prompt.txt`, 59 tokens,
`tokenizer.encode(text, add_special_tokens=True)` so there are no `[gMASK]<sop>` tokens: this is a raw continuation of a
chat model, which the qualitative-check skill counts as labeled stress coverage only), 256 new tokens.
Driven by `tools/full_model_checks.py run` (one generator build, shared with the qualitative suite). Env:
`GLM53_TP=4 GLM53_MAX_BATCH=4 GLM53_MAX_SEQ_LEN=4096 GLM53_TRACE_REGION=268435456`. Log:
`~/glm53-flash-bringup/logs/s6q_tt.log`.

**HF side precomputed.** The runner's `_hf_generate_greedy` loads the 328 GB parent checkpoint with `from_pretrained`,
and python_env's transformers has no `glm5_next`, so it cannot run here. `tools/hf_full_ref.py --mode autoregressive` (CPU,
unmodified HF `Glm5NextTextModel` on the same E192 FP8 checkpoint, same tokenization, greedy, generation_config eos ids)
produced `hf_tokens.json`. `tools/readiness.py --hf-tokens` monkeypatches `_hf_generate_greedy` to return those ids after
checking the prompt ids and budget. `autoregressive_meta.json["hf_reference"]` records this.

## Result
- TT path: `Glm53Generator.generate(next_input=None)` with traced decode and on-device greedy split sampling. 256 tokens,
  23 s. A second identical call gave **bit-identical tokens** (`tt_repeat_identical: true`).
- `check_degenerate_output.py --scope autoregressive --missing-artifacts critical`: **exit 0, "No degenerate output
  detected"**. adjacent_duplication 0.0, trigram_loop_fraction 0.23 (advisory threshold 0.5).
- HF vs TT: they agree on token 0 (" a") and diverge at token 1. HF goes " a strange old man climbing the western
  slope…", a coherent story that turns into Q&A after "The end.". TT goes " a butterfly fluttering past her window.", and
  from token ~25 it loops one sentence: "Elena, being a keen observer, watched as the butterfly fluttered past her window."
- **Is the loop a bug or a model property?** `hf_full_ref.py --mode score-trajectory` teacher-forces HF over the TT
  trajectory (`hf_scores_tt_trajectory.json`):
  - HF top-1 matches TT's token 97.7 % of the time and top-5 matches 99.6 %.
  - Every TT token from position 40 on (the loop) is HF's own top-1, with a top-1 margin of at least 1.75 logits.
    Given the same context, the HF model loops the same way.
  - Position 1 is a near-tie, even inside HF: batch-prefill HF ranks " small" 15.31 > " strange" 15.06 > " stranger"
    14.19 = " butterfly" 14.19 (rank 2). HF's own incremental decode picked " strange".
  - TT's off-top-1 picks (rank 2 at positions 1 and 8, rank 3 at 9, rank 5 at 12, 1–2 logits below HF's top) are
    precision drift of the size teacher forcing shows (top-1 93 %, top-5 100 %). They are not token-feedback or position
    errors.
- Verdict: **pass**. There is no decode-loop bug: no doubled tokens, deterministic, and HF endorses every looping token.
  The phrase loop is greedy raw-continuation behaviour after an early near-tie divergence. The chat-format qualitative
  suite (`../qualitative/VERDICT.md`) is the quality gate.
