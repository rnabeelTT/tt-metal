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

**Verdict: pass.** TT chat-format output is coherent and correct, and matches the HF control in content. The one quality
defect (q5 untranslated) reproduces in HF. q3 and q6 hit the 2048-token budget while still in "Reasoning Effort: Max"
thinking. That is a property of the effort setting, not repetition. The vLLM stage's 256-token qualitative budget will
see thinking text only, unless the request disables or lowers reasoning effort.
