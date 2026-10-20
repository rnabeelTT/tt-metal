# Stage 7 qualitative check (optimized full model, 45 layers, TP4 p300x2)

**Run**: `tools/s7_probe.py --do checks` on the stage-7 final build (`~/glm53-flash-bringup/logs/s7_final_s1.log`;
1 slot, max_seq_len 16,384, stage-7 defaults with `ag_persistent=0`, which is the `DEFAULT_MC` at HEAD). Prompt format
as in stage 6 (`doc/full_model/qualitative/qualitative_prompt_format.json`): chat template, one user message,
`add_generation_prompt=True`, "Reasoning Effort: Max", generation prompt ends in `<|assistant|><think>`. The rendered
prompt ids are the stage-6 ones (`rendered_prompts.json`), re-checked against the TT tokenizer. Greedy, traced decode,
on-device split sampling, stop at the first generation_config eos id, budget 2048 tokens. Outputs:
`tt_outputs.json`, `qualitative_outputs.json`.

**Controls**: the HF control of stage 6 (`doc/full_model/qualitative/hf_outputs.json`, CPU, same checkpoint and rendering,
128 tokens) and the stage-6 TT outputs (previous-stage control, `doc/full_model/qualitative/tt_outputs.json`).

**Degenerate check** (`check_degenerate_output.py --scope all qualitative_outputs.json`, rework 2026-10-18):
**exit 0, "No degenerate output detected"** (`degenerate_check.txt` / `.json`). Max adjacent duplication 0.024 (q1, q6),
max trigram-loop fraction 0.203 (q5, the repeated quoted sentence below; advisory threshold 0.5).

| id | tokens (eos) | answer (snippet) | vs stage 6 |
|---|---|---|---|
| q1 haiku | 1069 (yes) | "Neurons hum in code— / gradients descend the dark, / loss fades into light." 5-7-5 | stage 6 picked another draft ("Data flows like rain— ..."); both correct. The stage-6 "Wisdom" spiral does not occur here (the path differs at a near-tie earlier) |
| q2 supervised vs unsupervised | 1096 (yes) | labeled vs unlabeled data, answer-key analogy, spam / clustering examples, "In a nutshell" summary | same content and structure |
| q3 story | 2048 (no) | planning, then a complete draft in progress ("...none of it was magic. It was pumps. It was springs...") | stage 6 was still planning at 2048; both coherent |
| q4 thermodynamics | 1122 (yes) | First / Second / Third laws with the zeroth-law bonus | same |
| q5 French | 365 (yes) | **"Bonjour, how are you today?"** (translation left in English) | same defect in stage 6 and in the HF control: E192 checkpoint behaviour |
| q6 Fibonacci | 2048 (no) | iterative / sequence / nth-number code with docstrings and `ValueError` checks | same |

No wrong language (apart from the q5 model behaviour above), prompt echo, doubled subwords, control-token leakage, cross-
request leakage or gibberish. Every finished answer ends with `<|user|>` (154827), a generation_config eos id. q3 and q6
reach the 2048-token budget in "Reasoning Effort: Max", as in stage 6.

**Verdict: pass.** Stage-7 outputs are coherent and correct and match the stage-6 and HF controls in content; the one
quality defect (q5) reproduces in the HF control.
