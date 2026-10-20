# Stage 6 qualitative check (full 45-layer model, TP4 p300x2)

**Prompt format** (`qualitative_prompt_format.json`):
- Checkpoint `~/glm53-flash-bringup/ckpt/glm53_flash_e192_fp8` (zai-org/GLM-5.3-Flash, E192 pruned), tokenizer class
  `TokenizersBackend`, `chat_template.jinja` present, so the mode is **chat**.
- Each prompt from the shared suite `readiness_check/vllm_prompts.txt` (6 prompts, split on blank lines like
  `run_vllm_server`) is one user message, rendered with `apply_chat_template(..., add_generation_prompt=True)` and default
  kwargs. The template inserts `<|system|>Reasoning Effort: Max` and the generation prompt ends in `<|assistant|><think>`,
  so the model thinks first. That is the checkpoint's declared default.
- Rendered text and token ids are in `rendered_prompts.json`. The TT-side tokenizer (python_env transformers 5.12)
  reproduced them exactly for all 6 prompts.

**Generation**:
- TT: `Glm53Generator.generate` (traced decode, on-device greedy split sampling), observed through `next_input` (never
  overriding), stopping after the first generation_config eos id. Budget 2048 tokens. Results in `tt_outputs.json` and
  `qualitative_outputs.json`.
- HF control: `tools/hf_full_ref.py --mode qualitative`, CPU, same checkpoint and rendering, greedy, 128 tokens per prompt
  (about 25 min per prompt on CPU). Results in `hf_outputs.json`; token-level comparison in `comparison.json`.

**Degenerate check**: `check_degenerate_output.py --missing-artifacts critical` returns **exit 0** on both the TT outputs
(`degenerate_check_tt.txt`; max adjacent_duplication 0.020, max trigram loop 0.18) and the HF controls
(`degenerate_check_hf.txt`).

| id | TT tokens (eos) | TT == HF prefix | TT final answer |
|---|---|---|---|
| q1 haiku | 1170 (yes) | 38 tokens | "Data flows like rain— / patterns hidden, slowly learned— / the model awakes." Correct 5-7-5. |
| q2 supervised vs unsupervised | 1022 (yes) | 75 | Clear answer-key vs no-labels explanation, with examples, analogy and a one-line summary. |
| q3 story | 2048 (no) | 10 | Still thinking at 2048 tokens: coherent story planning, then starts the draft ("…discovered a workshop hidden beneath the old clock tower."). |
| q4 thermodynamics | 1035 (yes) | 27 | Correct First/Second/Third laws (ΔU = Q − W, entropy, absolute zero) plus a zeroth-law note. |
| q5 French | 257 (yes) | 31 | **"Bonjour, how are you today?"**: the translation is left in English. The HF control does the same ("Bonjour, how are you today?" … "How are you today?" = "How are you today?"), so this is model behaviour of the E192-pruned checkpoint, not a TT defect. |
| q6 Fibonacci | 2048 (no) | 29 | Still thinking at 2048 tokens. Correct iterative, sequence, recursive and `@cache` code with an O(n²) self-correction. |

After each divergence, TT and HF stay on the same topic and structure. Examples:
- q1: both draft "Data flows like streams" as Option 1.
- q2: both use the same answer-key analogy and spam/cat-dog/house-price examples.
- q4: both open with the zeroth law, then the first law.

Divergence points are single-word choices, e.g. "data, patterns, training" vs "training data, neural networks". This
matches the teacher-forcing top-1 of 93 % and top-5 of 100 %.

No wrong language, prompt echo, doubled subwords, control-token leakage or gibberish appears in any TT output. The final
answers end with `<|user|>` (154827), which is one of the generation_config eos ids.

## Anomalies past the 128-token HF window (stage-6 review #1 P2-2, rework 2026-10-18)

The HF control stops at 128 tokens, so two TT oddities had no control. `tools/hf_full_ref.py --mode score-trajectory`
(CPU, unmodified HF `Glm5NextTextModel` on the same E192 checkpoint) teacher-forces HF over the TT token ids and ranks
each emitted token (`hf_scores_tt_q1.json.gz`, `hf_scores_tt_q5.json`; logs `~/glm53-flash-bringup/logs/s67r/hfscore_s6_q{1,5}.log`):

| Prompt | Tokens scored | HF top-1 agreement | HF top-5 agreement | Window |
|---|---|---|---|---|
| q1 haiku | 1170 | 94.5 % | **100 %** | the "Wisdom" spiral, tokens 456-705: top-1 94 %, top-5 100 %; the 15 non-top-1 picks are rank 1-2 near-ties (HF top-1 margin 0.125-0.625 logits, two at 1.5 and 2.1) |
| q5 French | 257 | 98.0 % | **100 %** | "I shouldn over-explain" (token 187): `' shouldn'` is a single tokenizer token and HF's own top-1 there (28.9 vs 22.3 for `' don'`); the next token `' over'` is HF top-1 too (28.3 vs 25.5) |

- q1 spiral ("\"Wisdom\" isn't a word ... W-I-S-D-O-M is \"wisdom\" which is not a word"): every token of the window
  is in HF's top-5 and 94 % are HF's top-1. Given the same context, the HF model writes the same spiral. It is a
  model behaviour of the E192 checkpoint under "Reasoning Effort: Max" (a self-doubt loop over a word it has just
  written), not a TT logit or vocab-slice defect. The stage-7 rerun of q1 takes another path at a near-tie and ends on
  EOS with a correct haiku.
- q5 "shouldn over-explain": the missing `'t` is HF's choice too (`' shouldn'` then `' over'`, both HF top-1 with
  margins of 6.6 and 2.8 logits). Model behaviour, not a dropped token.

Both are classified **controlled (model behaviour)**.

**Verdict: pass.** TT chat-format output is coherent and correct, and matches the HF control in content. The one quality
defect (q5 untranslated) reproduces in HF. q3 and q6 hit the 2048-token budget while still in "Reasoning Effort: Max"
thinking. That is a property of the effort setting, not repetition. The vLLM stage's 256-token qualitative budget will
see thinking text only, unless the request disables or lowers reasoning effort.
