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
  - Evidence: the `CHECK` lines in `logs/09/s9_a1a3.log`. A later stage-10 run overwrote
    `logs/09/validate_adapter.json` (the script had a fixed output path; it now takes the path as an argument).
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

## A5: full model, profile p300x2, full plugin sampling suite (2026-10-21, snapshot s9f = 94708e55cf3)

Command: `scripts/s9_step.sh A5 <snapshot>` (`SPROFILE=full SMOKE=1`; readiness served with low reasoning effort).

- Plugin suite: 69 passed, 4 failed, 4 skipped. The skips are the chunked-prefill tests (not supported) and
  all-vocab chat logprobs. The runner stops after a sampling failure, so qualitative and benchmark ran in A6
  (same profile and server flags).
- Served smokes: `SMOKE_RESULT PASS`, the same as A4.
- Max-effort control (the shipped mode, `max_tokens` 4096):
  - Haiku: stop after 756 tokens, coherent reasoning. Content: "Data flows like streams, / Patterns hidden in the
    noise— / The machine awakes."
  - Train time: stop after 142 tokens. Content: "2 hours and 35 minutes", correct.
  - Evidence: `logs/09/smoke-p300x2-a5.maxeffort.json`.

### Classification of the 4 failures (A6 server, `scripts/09-diag.py`, `logs/09/diag-p300x2.json`)

The 4 node ids fail again on the A6 server. The diagnosis:

1. `TestPresencePenalty::test_different_presence_penalties` and 2. `::test_presence_penalty_mixed_batch`:
   **model property of the test prompt, not a sampler fault.**
   - On "a b c a b c a b c", the smallest top-1/top-2 logprob gap on an already-generated token is 2.25. The largest
     penalty in the test is 2.0, so no greedy token can change.
   - Positive control ("Once upon a time, in a small village,"; gap 0.375 on " Lily"): presence 2.0 changes the
     greedy output.
   - Mixed batch of 16 (8 at 0.0, 8 at 2.0): 8/8 rows equal the solo 0.0 output and 8/8 rows equal the solo 2.0
     output. Each request gets its own penalty.
3. `TestHostOnlyParameters::test_bad_words` and 4. `test_dp1_full_capacity_mixes_structured_and_plain_requests`:
   **prompt-format artifact of the always-thinking template, not a sampling fault.**
   - The 100-token (bad_words) and 8-token (choice) budgets end inside the reasoning (finish `length`), so `content`
     is None.
   - With room to finish: bad_words stops after 167 tokens, the content has no banned word. The choice stops after 46
     tokens with content "blue".
   - The template has no thinking switch: generation always opens `<think>`, and `reasoning_effort` accepts only
     low, high or max.
   - An empty think block prefilled per request (`continue_final_message`) does not help. The reasoning parser then
     files the answer under `reasoning` (choice: "yellow"), and `content` is still None.

## A6: qualitative and benchmark, profile p300x2 (snapshot s9f)

- Qualitative (6 prompts, greedy and sampled, low reasoning effort): coherent and on topic.
  - `check_degenerate_output.py --scope vllm`: no degenerate output.
  - The French prompt gives "Bonjour, how are you today?" (the rest stays English). The stage-6 HF control
    (`doc/full_model/qualitative/VERDICT.md`, q5) does the same, so this is model behaviour of the E192 checkpoint.
- Primary single user, 128/128/1, greedy (baseline before the stage-10 DSA width buckets):
  - TTFT median 263 ms.
  - TPOT mean 119.3 ms (8.4 t/s/u); ITL median 119.0 ms.
  - Output throughput 8.3 tok/s.
- CI serving burst, 100/100/32, greedy:
  - TTFT median 14.5 s; TPOT mean 179.6 ms.
  - Output throughput 74.6 tok/s.
- The standalone 1-slot decode at a 16K context is 49.9 ms (stage 8). Stage 10 attributes most of the 119 ms to the
  16-slot build and profiles it. These numbers are the baseline before the stage-10 changes.

### A6 sweep (same server; `artifacts/s10/sweep_p300x2.json`, `scripts/10-perf-sweep.py`)

Random-token prompts of the exact length, greedy, `ignore_eos`, closed-loop users. 15 of 15 rows complete, with no
failed or short requests and no prompt-length mismatch.

| input | output | users | TTFT (ms) | TPOT (ms) | t/s/user | total out t/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 128 | 1 | 245 | 109.8 | 9.1 | 9.0 |
| 1,024 | 256 | 1 | 766 | 111.5 | 9.0 | 8.8 |
| 4,096 | 256 | 1 | 2,986 | 113.3 | 8.8 | 8.0 |
| 16,384 | 256 | 1 | 13,935 | 113.7 | 8.8 | 6.0 |
| 65,536 | 256 | 1 | 57,759 | 114.5 | 8.7 | 2.9 |
| 131,072 | 256 | 1 | 116,170 | 114.1 | 8.8 | 1.8 |
| 243,456 | 256 | 1 | 216,555 | 114.0 | 8.8 | 1.0 |
| 128 | 128 | 2 | 365 | 122.4 | 8.2 | 16.1 |
| 128 | 128 | 4 | 964 | 125.8 | 7.9 | 30.3 |
| 128 | 128 | 8 | 1,916 | 143.6 | 7.0 | 50.7 |
| 128 | 128 | 16 | 3,822 | 178.7 | 5.6 | 77.2 |
| 1,024 | 256 | 2 | 1,146 | 124.7 | 8.0 | 15.5 |
| 1,024 | 256 | 4 | 3,046 | 128.4 | 7.8 | 28.7 |
| 1,024 | 256 | 8 | 6,089 | 144.8 | 6.9 | 47.6 |
| 1,024 | 256 | 16 | 12,170 | 180.5 | 5.5 | 70.4 |

- TPOT at 1 user does not grow with position (109.8 ms at 128 tokens, 114.0 ms at 243,456): the decode cost is set
  by the build (16 slots, max_model_len 243,712), not by the filled context.
- The 243,456-token prompt (the profile limit minus 256) answers.

## A7: profile p300x2-wide (32 x 151,552), snapshot s9f (2026-10-21)

Command: `scripts/s9_step.sh A7 <snapshot>` (`SPROFILE=smoke SMOKE=1 BURST=32 SWEEP=1`, low reasoning effort for the
readiness stages).

- Readiness: exit 0.
  - Plugin sampling smoke: 3 passed, 1 skipped.
  - Qualitative: no degenerate output.
  - Primary 128/128/1: TTFT median 294 ms, TPOT mean 120.9 ms, ITL median 120.1 ms, 8.2 tok/s.
  - CI burst 100/100/32: 32 completed, TTFT median 7.8 s, TPOT mean 167.7 ms, 132.3 tok/s.
- Served smokes: `SMOKE_RESULT PASS`.
  - Burst of 32 concurrent requests: 32/32 equal to their solo greedy outputs (105.7 tok/s aggregate).
  - Max-effort control: both prompts stop with content.
- Sweep (`artifacts/s10/sweep_p300x2-wide.json`): 10 of 10 rows, with no failed or short requests.

| input | output | users | TTFT (ms) | TPOT (ms) | t/s/user | total out t/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 128 | 1 | 256 | 120.4 | 8.3 | 8.2 |
| 1,024 | 256 | 1 | 774 | 122.3 | 8.2 | 8.0 |
| 16,384 | 256 | 1 | 12,307 | 124.6 | 8.0 | 5.8 |
| 131,072 | 256 | 1 | 101,427 | 125.4 | 8.0 | 1.9 |
| 128 | 128 | 8 | 1,995 | 133.9 | 7.5 | 54.2 |
| 128 | 128 | 16 | 3,986 | 150.7 | 6.6 | 89.2 |
| 128 | 128 | 32 | 7,954 | 168.7 | 5.9 | 139.9 |
| 1,024 | 256 | 8 | 6,158 | 133.5 | 7.5 | 50.8 |
| 1,024 | 256 | 16 | 12,305 | 150.9 | 6.6 | 80.9 |
| 1,024 | 256 | 32 | 24,604 | 166.3 | 6.0 | 122.6 |

## A8: profile p300x2-long (1 x 325,632), snapshot s9f (2026-10-21)

Command: `scripts/s9_step.sh A8 <snapshot>` (`NOSTAGES=1 SMOKE=1 SMOKE_ARGS="--long 325568" SWEEP=1`).

- Pool: 2,546 blocks (measured); page-table width 2,544.
- Served smokes: `SMOKE_RESULT PASS`.
  - Long prompt: a 325,568-token prompt answers in 326 s with a coherent continuation.
  - Burst of 8 (one slot, queued): 8/8 equal to solo greedy.
  - Max-effort control: both prompts stop with content.
- Sweep (`artifacts/s10/sweep_p300x2-long.json`): 5 of 5 rows, with no failed or short requests.

| input | output | users | TTFT (ms) | TPOT (ms) | t/s/user |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 128 | 1 | 236 | 56.5 | 17.7 |
| 4,096 | 256 | 1 | 3,201 | 60.2 | 16.6 |
| 65,536 | 256 | 1 | 64,592 | 60.0 | 16.7 |
| 262,144 | 256 | 1 | 260,888 | 60.1 | 16.6 |
| 325,376 | 256 | 1 | 324,233 | 60.0 | 16.7 |

- The 1-slot build decodes about 2x faster than the 16-slot p300x2 build (56.5 vs 109.8 ms at 128 tokens).
- TPOT is flat from 4K to 325K tokens. The 128-token row is lower because its decode positions stay in the dense
  regime (2,050 tokens or fewer, no indexer selection).

## A8-1M: profile p300x2-1m (1 x 1,048,576, block-sharded fp8), the stage-9 gate run (snapshot s9g = 73c0029ee79)

Command: `scripts/s9_step.sh A8-1M <snapshot>` (`SPROFILE=full SMOKE=1 BURST=2 SMOKE_ARGS="--long 1048000"`, needle on
by default for this profile, low reasoning effort for the readiness stages). Server healthy after 2.5 min. Pool:
8,193 vLLM blocks + 1 dummy (sharded layout). The gate artifact `readiness_vllm/` comes from this server.
`09-vllm.check.sh` exits 0: no degenerate output; the context contract is OK at 1,048,576.

- Plugin suite, full profile, max_num_seqs 1: 64 passed, 9 failed, 4 skipped.
  - 4 failures are the known ones (A5/A6 classification).
  - The other 5 need at least 2 requests: `test_uniform_noseed_varied`, the repetition- and frequency-penalty
    variety and mixed-batch tests. They build the request list from `max_batch_size` and get one request at 1
    sequence ("Only 1/1 were varied"). They pass at 16 sequences (A5) and 32 sequences (A7).
- Qualitative and benchmark ran on the same server after the needles (`status/needle_post.sh`).
  - Qualitative: coherent; no degenerate output. French q5 is the known E192 behaviour.
  - Primary 128/128/1: TTFT median 234.5 ms; TPOT mean 73.5 ms (13.6 t/s/u); ITL median 72.7 ms (P99 76.4 ms);
    13.4 tok/s.
  - CI burst 100/100/32 (queued on one slot): 32/32 completed; TPOT mean 73.8 ms; 13.2 tok/s.
- Needle retrieval through vLLM (`scripts/09-needle.py`, `logs/09/needle-p300x2-1m.json`). The filler is shuffled
  tt-metal source and doc files; the passkey is a random 7-digit code; the question is at the end; reasoning effort
  low, greedy.

| case | prompt tokens | passkey depth | TTFT | answer | result |
| --- | ---: | ---: | ---: | --- | --- |
| 1M | 1,038,965 | 0.082 | 2,307 s | 8197047 | PASS |
| 300K | 297,730 | 0.363 | 661 s | 1983584 | PASS |

- Served smokes:
  - A 1,048,000-token prompt answers coherently in 2,306 s.
  - Chat, tool, logprobs, penalties, min_p, seeded, greedy repeat, non-aligned 1 to 4223 and page growth are OK.
  - Max-effort control: both prompts stop with content.
  - `SMOKE_BURST 2`: 1/2 equal to solo greedy (see the burst diagnosis below).
- Burst diagnosis on the same server (`scripts/09-burst-diag.py`, `logs/09/burstdiag-p300x2-1m.json`):
  - Plain repeats and a concurrent pair are identical to solo for both prompts.
  - After a 5,000-token request (different vLLM block ids), prompt 0 diverges at output token 47. The top-1/top-2
    gap there is 0.5625.
  - On the block-sharded layout a block's die is its id modulo 4, so the block ids set the reduction order.
  - Follow-up: `scripts/09_shard_place.py` compares logits bitwise for a repeat, a 4-block-aligned shift and a
    1-block shift (reduced model, explicit page tables).

## Notes

- Indexer cost: the sparse DSA indexer selects over all pools of the model context, not over the filled prefix. The
  model context comes from max_model_len (`pad_page_table` pads every block table to T/128 columns). So the decode
  time grows with max_model_len. The A6 and A8 numbers are the baseline before the stage-10 DSA width buckets.
