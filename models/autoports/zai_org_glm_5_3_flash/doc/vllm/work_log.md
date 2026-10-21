# Stage 9 (vLLM integration): work log

Device jobs run from pinned snapshots under `~/glm53-flash-bringup/` (`git archive` of a commit; `BASE_COMMIT` and
`MD5SUMS` in each snapshot). Every job takes one ticket in the FIFO device queue (`scripts/run-locked.sh`).
Logs: `~/glm53-flash-bringup/logs/09/`. Scripts: `~/glm53-flash-bringup/scripts/`.

## A0: host checks (2026-10-20)

- `tests/test_generator_vllm_host.py` passes on every snapshot. The count is now 22.
- `python -m models.autoports.zai_org_glm_5_3_flash.tools.serving_profiles check` reports OK.

## A1 + A3: reduced model, layers 0, 3, 4 (2026-10-20, snapshot s9a = 656c998e3df + uncommitted files)

Command: `scripts/s9_a1a3.sh <snapshot>` in one ticket. The reduced target is an inner-loop tool, not evidence.

- A1 `scripts/09_validate_adapter.py` (no vLLM): `VALIDATE_RESULT PASS`.
  - kv_identity; warm-up with 848 programs.
  - Prefill greedy equals the host argmax at lengths 37, 300 and 2051.
  - 24 traced decode steps, sync and async read; slot remap.
  - Die lockstep: 0 of 64 sampled steps disagree.
  - Program guard: 848 = 848.
- A3 reduced vLLM server (4 sequences, max_model_len 8192):
  - The server is healthy after 90 s.
  - Greedy repeat is identical.
  - Non-aligned prompts of 1, 3, 127, 129, 2049, 2051, 2381 and 4223 tokens answer.
  - Page growth: 400 tokens.
  - A burst of 4 equals the solo greedy outputs.
  - Chat and tool gates do not apply to a 3-layer model (`09-smoke.py` gates them on the full model only).

## A2: G9 capacity probes, combo_a_kv_dense4_g4, replicated fp8 DSA cache, 1,024 MiB workspace

Command: `SNAP=<snapshot> INGEST=1 scripts/09-g9-probe.sh S` (`tools/memory_probe.py capacity`). The probe runs
load, pool, warm-up and capture, then one prompt near the pool top and a traced greedy decode. The decode must copy
the passage.

| slots | snapshot | pool blocks | max_model_len | prompt | copy | free at prefill peak | trace |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | s9c (19a4dda4f47) | 2,546 | 325,632 | 325,571 | 1.0 | 234 MB | 39.4 MB |
| 16 | s9c | 1,933 | 243,712 | 245,699 | 1.0 | 248 MB | 43.6 MB |
| 32 | s9c | 1,229 | 151,552 | 153,539 | 1.0 | 413 MB | 48.7 MB |

Evidence: `doc/vllm/capacity/s1`, `s16`, `s32` (capacity.json, capacity_probe.log, provenance.json with the tt md5).
The tt/ changes after s9c touch only the block-sharded path and the standalone generator budget, not the replicated
serving pool. The combo_a_kv 1-slot probe (superseded) is in `doc/vllm/capacity/s1_combo_a_kv`.

## A4: full model, profile p300x2 (16 x 243,712), smoke (2026-10-20, snapshot s9d = 1c874c88f58)

Command: `scripts/s9_step.sh A4 <snapshot>` (`09-serve.sh`, `PROFILE_NAME=p300x2 SPROFILE=smoke SMOKE=1`).

- The server is healthy after 2.5 min (warm weight cache).
- Pool: 1,933 blocks (measured).
- Precision: combo_a_kv_dense4_g4.
- PROGRAM_GUARD is active and does not trip.
- Plugin sampling smoke: 3 passed, 1 skipped.
- Served smokes: `SMOKE_RESULT PASS`.
  - The chat answer and the `read_file` tool call are correct.
  - logprobs, repetition penalty, min_p and seeded requests answer.
  - Greedy repeat is identical.
  - Non-aligned prompts of 1 to 4223 tokens continue coherently.
  - Page growth: 400 tokens.
  - A burst of 8 equals the solo greedy outputs.
- The readiness qualitative stage failed with "Chat completion returned no text content". The template is
  always-thinking, so a 256-token request ends inside the reasoning.
  - Fix: readiness runs serve with `--default-chat-template-kwargs {"reasoning_effort":"low"}`. This is the
    template's own knob.
  - The bundle keeps the default (max) effort.
  - The served smoke adds a max-effort control: two prompts with `max_tokens` 4096. The content must arrive.

## Notes

- Indexer cost: the sparse DSA indexer selects over all pools of the model context, not over the filled prefix. The
  model context comes from max_model_len (`pad_page_table` pads every block table to T/128 columns). So the decode
  time grows with max_model_len. The A6 and A8 numbers are the baseline before the stage-10 DSA width buckets.
