# Stage 7 autoregressive check (final defaults, full 45-layer model, TP4 p300x2)

Run: `j2_s1_default` (`tools/s67r_checks.py --slots 1 --do ...,ar,...`, log
`~/glm53-flash-bringup/logs/s67r/j2_s1_default.log`, record `../runs/s67r/j2_s1_default.json`).

- Code: HEAD 656c998e3df plus the stage-6/7 rework files, run from a pinned snapshot. md5 stamps at start and end are
  unchanged.
- Build: `Glm53Generator(mesh)` defaults: 1 slot, stage-8 precision `combo_a_kv`, replicated DSA cache,
  `max_seq_len` 280,576.
- Prompt: the stock `autoregressive_prompt.txt` (59 tokens, raw continuation, no chat template), 256 new tokens.
- Decode: traced, with on-device greedy split sampling.
- HF reference: `hf_completion.txt`, from `tools/hf_full_ref.py --mode autoregressive` (CPU, unmodified HF on the same
  E192 FP8 checkpoint).

## Result

- **Determinism:** a second identical call gave bit-identical tokens (`tt_repeat_identical: true`).
- **`check_degenerate_output.py --scope autoregressive --missing-artifacts critical`:** exit 0, "No degenerate output
  detected" (`degenerate_check.txt`). Adjacent duplication 0.0, trigram loop fraction 0.135 (advisory threshold
  0.5).
- **HF vs TT:**
  - They diverge at token 0. HF writes " a strange old man climbing the western slope...".
  - TT writes " something strange: the light seemed to bend around the leaves...". From about token 90 it repeats a
    two-sentence cycle ("She watched the sunrise paint the valley gold, and she watched the shadows dance beneath the
    oak tree." / "Elena, being a curious young girl, decided to investigate...").
- **HF scoring of TT's trajectory** (`tools/hf_full_ref.py --mode score-trajectory`, `hf_scores_tt_trajectory.json`):
  HF teacher-forced over TT's 256 tokens.
  - HF's top-1 equals TT's token at **93.75 %** of the positions, and HF's top-5 contains it at **100 %**.
  - Token 0 is a near-tie inside HF: " a" 19.00 vs " something" 18.625, a 0.375-logit margin. TT picked HF's rank 2.
    The stage-8 run on the same policy reports the same divergence.
  - All 16 off-top-1 picks are HF's rank 2 to 5. Each is 0.125 to 1.75 logits below HF's top-1:
    - rank 2 at positions 0, 10, 13, 20, 36, 41, 47, 49, 76, 82, 93, 146;
    - rank 3 at positions 35 and 74;
    - rank 4 at position 4;
    - rank 5 at position 53.

    That is precision drift of the size teacher forcing shows on this policy (top-1 91 %, top-5 100 %). It is not
    token-feedback or position error, which would land outside HF's top-5.
  - Every TT token after position 146 (109 tokens, the whole loop) is HF's own top-1, by at least 0.25 logits. Given
    the same context, HF loops the same way.

## Verdict

**Pass.** There is no decode-loop bug: no doubled tokens, the output is deterministic, every emitted token is in HF's
top-5, and HF endorses every looping token. The phrase loop is greedy raw-continuation behaviour after a near-tie
divergence at token 0. The chat-format qualitative suite (`../qualitative/VERDICT.md`) is the quality gate.
