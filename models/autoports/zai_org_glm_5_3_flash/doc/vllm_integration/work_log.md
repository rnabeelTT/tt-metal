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
    sequence ("Only 1/1 were varied"). They pass at 16 sequences (A5). (A7 ran the smoke profile only; the full suite at 32 runs in the final serving run.)
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

## Block-placement control (2026-10-21, snapshot s9g, `logs/09/shard_place.json`)

Command: `scripts/09_shard_place.py OUT` in one ticket (reduced layers 0, 3, 4, `GLM53_DSA_SHARD=1` + the layout
env, max_seq_len 16,384, a 6,001-token prompt in 47 blocks, host-logits mode, prefill + 8 traced greedy decode steps).

| case | blocks | bit-identical to base [1..47] | max abs logit diff | tokens equal |
| --- | --- | --- | ---: | --- |
| repeat | [1..47] | yes | 0 | yes |
| 4-block-aligned shift | [5..51] | yes | 0 | yes |
| 1-block shift | [2..48] | no | 0.527 | yes |

Verdict: placement-only numerics of the block-sharded layout. The same die placement gives bit-identical logits. The
A8-1M burst divergence (gap 0.5625 after a block-id change) is this effect, not stale state.

## Review #1 P1-2 / P1-1: stale token after a host-sampled step; async decode (2026-10-21)

- P1-2 (commit 3f5b9df1474): a host-sampled decode step advances the device position but skips the device sampler, so
  `tok_buf` keeps the step's input token. The next device-sampled step merged with that stale token. Fix:
  `Glm53Generator._dev_sampled` per slot (cleared by a host-sampled step and by a prefill). The merge runs only under
  async decode, never on `reset_sampling_state`, and only for slots whose latest token was device-sampled. Host test
  `test_host_sampled_step_then_device_step_reads_host_token` (fails on the old code in both modes, passes now).
- P1-1 device test: `scripts/09_async_validate.py` on snapshot s9i (HEAD 0bd45f315e1 + the P1-2 hunks), reduced model
  (layers 0, 3, 4), replicated layout, `GLM53_VLLM_ASYNC_DECODE=1`, the plugin's contract-1 call sequence through the
  adapter. Evidence: `doc/vllm/evidence/async_validate.json`.

| request | prompt | tokens | blocks | events | equal to solo control |
| --- | ---: | ---: | ---: | --- | --- |
| A | 37 | 121 | 2 | finishes at step 120 (drain, slot_remap condense) | yes |
| B | 300 | 421 | 6 | page growth, remap, host steps | yes |
| C | 1,000 | 421 | 12 | page growth, remap, host steps | yes |
| D | 2,051 | 421 | 20 | page growth, remap, host steps | yes |
| E | 129 | 300 | 4 | prefilled into A's freed slot between decodes | yes |

  - 413 overlapped steady steps (reload_inputs False; step N+1 submitted before step N is read), 13 of them with
    reload_page_table only (block growth), 7 host-authoritative reloads, 3 host-sampled steps (device -> host -> device).
  - The solo controls: each request alone, synchronous, in the same physical slot on the same block ids.
  - Program guard: no program compiled after the capture.
  - Result: PASS. `supports_async_decode` is now True by default (`GLM53_VLLM_ASYNC_DECODE=0` turns it off).

## Review #1 P3: decode result vs the build's slot count (2026-10-21, snapshot s9i)

`~/glm53-flash-bringup/scripts/s9_slot_width.sh` (one ticket; `09_slot_width_check.py` per build): reduced model (layers
0, 3, 4), replicated layout, one 600-token request, host-logits prefill + 16 traced decode steps, logits saved per build.
Evidence: `doc/vllm/evidence/slot_width_compare.txt`.

| build (slots) | request slot | logits vs the 1-slot build | max abs diff | tokens |
| ---: | ---: | --- | ---: | --- |
| 8 | 0 | bit-identical | 0 | equal |
| 16 | 0 | bit-identical | 0 | equal |
| 16 | 15 | bit-identical | 0 | equal |
| 32 | 0 | differ from decode step 1 | 0.0625 | equal |
| 32 | 31 | differ (bit-identical to slot 0 of the same build) | 0.0625 | equal |

- The 32-slot build changes about 76% of the vocab logits by 1 to 2 bf16 ULPs (logits of magnitude 4 to 8). Prefill is
  identical, so the difference is in the decode path above 16 rows (the unfused KDA conv path, multichip_decoder.py).
- All rows of one build give the same result for the same request (slot 0 = slot 31). The size is bf16 rounding of a
  different reduction order, not a defect; no fix. A near-tie greedy token can differ between a 32-slot and a smaller
  build.

## Final serving run (review #1 P1-3, 2026-10-21, code freeze 0a2663d0b6b)

One device ticket: `~/glm53-flash-bringup/scripts/s9_final.sh <snapshot>` (snapshot s9k = pure `git archive` of
0a2663d0b6b; `tt-smi -r` between parts; a leftover-process check after every server and probe: none). Evidence:
`doc/vllm/evidence/final/` (`s9_final_summary.txt` is the condensed log).

1. G9 re-probes (workspace 1,024 MiB): long 2,548 blocks -> 325,632 (64 MiB region); p300x2 / p150x4 1,891 blocks ->
   239,616 (96 MiB region, capture 97.7 MB); wide at 96 MiB: trace capture out of memory, then at 160 MiB (decision of
   the coordinator): MoE row buckets [1, 2, 4, 8, 16, 32], capture 107.3 MiB, 1,105 blocks -> 137,216. Copy 1.0 on all.
2. p300x2-long, async off then on: readiness benchmark (table in README); the async-off server also captured the
   served token ids of the 6 qualitative prompts (`scripts/09-qcapture.py`).
3. The standalone generator (`scripts/09_standalone_q.py`, the long build) on those prompt ids: equal tokens on all 6
   prompts (36 / 256 / 256 / 215 / 256 / 256 tokens).
4. p300x2: full plugin suite at 16 (69 / 4 / 4), qualitative, benchmark (copied to `logs/09/final/readiness-p300x2/`
   for the stage-10 AFTER cells), served smokes PASS (burst 8/8 equal to solo), the 197,077-token needle PASS
   ("8197047", TTFT 159.5 s), sweep 15/15 rows.
5. p300x2-wide (on 0a2663d0b6b, superseded by the re-run on 0bcf89f7fd6 below): full plugin suite at 32 (70 / 3 / 4), qualitative, benchmark, sweep 10/10 rows; served
   smokes FAIL on the burst gate only: 4/32 streams equal to solo (stage-10 MoE row buckets, open item).
6. p300x2-1m: sampling smoke (3 / 0 / 1), qualitative, benchmark -> `readiness_vllm/`; the 1,048,000-token prompt
   answers in 2,308 s; the burst gate 1/2 (the documented block-placement numerics); the 1M needle PASS (1,038,965
   tokens, "8197047", TTFT 2,310 s); `09-vllm.check.sh` exit 0 on the live tree.

Sweep, p300x2 (1 user, input tokens: TTFT / TPOT): 128: 252 ms / 69.9 ms; 1,024: 770 ms / 71.8 ms; 4,096: 2.4 s /
73.5 ms; 16,384: 10.0 s / 73.9 ms; 65,536: 42.9 s / 75.8 ms; 131,072: 98.4 s / 77.9 ms; 239,360: 194.8 s / 79.3 ms.
Users at 128 input tokens (TPOT / total tok/s): 2: 74.8 / 25.6; 4: 90.0 / 41.3; 8: 119.9 / 59.6; 16: 172.6 / 79.4.
At 1,024 input tokens: 2: 76.5 / 24.3; 4: 91.8 / 38.7; 8: 121.7 / 55.2; 16: 174.4 / 72.3.

Sweep, p300x2-wide (0a2663d0b6b, superseded by the re-run below; 1 user: 128: 268 ms / 76.0 ms; 1,024: 811 ms / 77.9 ms; 16,384: 9.9 s / 80.2 ms;
131,072: 92.8 s / 83.9 ms). Users at 128 input tokens: 8: 125.9 / 56.8; 16: 182.0 / 75.5; 32: 163.3 / 142.7. At 1,024:
8: 127.7 / 52.8; 16: 183.8 / 69.2; 32: 160.9 / 124.7.

### Exact server commands and environment (P2-4)

Every server runs from a pinned snapshot (`SNAP=<dir>` in `scripts/09-serve.sh`: cwd and PYTHONPATH first), with the
serving overlay (`serve_overlay`: transformers 5.19) on PYTHONPATH and `--additional-config` from the profile:
`{"tt": {"sample_on_device_mode": "all", "trace_region_size": <profile region>, "l1_small_size": 24576,
"fabric_config": "FABRIC_1D"}}`. The readiness servers add `--default-chat-template-kwargs {"reasoning_effort":"low"}`
(the 256-token qualitative requests); the bundle does not.

**p300x2-long (async on)** (`../vllm/evidence/final/long-async1_servercmd.txt`)

```bash
GLM53_CACHE_DIR=/home/raahem/.cache/glm53_flash
GLM53_PROGRAM_GUARD=1
GLM53_TRACE_REGION=67108864
GLM53_VLLM_ASYNC_DECODE=1
HF_MODEL=zai-org/GLM-5.3-Flash
MESH_DEVICE=P300x2
VLLM_ENGINE_READY_TIMEOUT_S=3600
EXTRA_MODELS_DIR=<snapshot>/models/autoports/zai_org_glm_5_3_flash/vllm_bundle
python -m vllm.entrypoints.openai.api_server --model $GLM53_CKPT --served-model-name $GLM53_CKPT zai-org/GLM-5.3-Flash --tokenizer $GLM53_CKPT --hf-overrides \{\"architectures\":\ \[\"TTGlm5NextForConditionalGeneration\"\]\} --block-size 128 --max-num-seqs 1 --max-model-len 325632 --port 8131 --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 --default-chat-template-kwargs \{\"reasoning_effort\":\"low\"\} --additional-config \{\"tt\":\ \{\"sample_on_device_mode\":\ \"all\"\,\ \"trace_region_size\":\ 67108864\,\ \"l1_small_size\":\ 24576\,\ \"fabric_config\":\ \"FABRIC_1D\"\}\}
```

**p300x2-long (async off, A/B)** (`../vllm/evidence/final/long-async0_servercmd.txt`)

```bash
GLM53_CACHE_DIR=/home/raahem/.cache/glm53_flash
GLM53_PROGRAM_GUARD=1
GLM53_TRACE_REGION=67108864
GLM53_VLLM_ASYNC_DECODE=0
HF_MODEL=zai-org/GLM-5.3-Flash
MESH_DEVICE=P300x2
VLLM_ENGINE_READY_TIMEOUT_S=3600
EXTRA_MODELS_DIR=<snapshot>/models/autoports/zai_org_glm_5_3_flash/vllm_bundle
python -m vllm.entrypoints.openai.api_server --model $GLM53_CKPT --served-model-name $GLM53_CKPT zai-org/GLM-5.3-Flash --tokenizer $GLM53_CKPT --hf-overrides \{\"architectures\":\ \[\"TTGlm5NextForConditionalGeneration\"\]\} --block-size 128 --max-num-seqs 1 --max-model-len 325632 --port 8131 --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 --default-chat-template-kwargs \{\"reasoning_effort\":\"low\"\} --additional-config \{\"tt\":\ \{\"sample_on_device_mode\":\ \"all\"\,\ \"trace_region_size\":\ 67108864\,\ \"l1_small_size\":\ 24576\,\ \"fabric_config\":\ \"FABRIC_1D\"\}\}
```

**p300x2** (`../vllm/evidence/final/p300x2_servercmd.txt`)

```bash
GLM53_CACHE_DIR=/home/raahem/.cache/glm53_flash
GLM53_PROGRAM_GUARD=1
GLM53_TRACE_REGION=100663296
GLM53_VLLM_ASYNC_DECODE=1
HF_MODEL=zai-org/GLM-5.3-Flash
MESH_DEVICE=P300x2
VLLM_ENGINE_READY_TIMEOUT_S=3600
EXTRA_MODELS_DIR=<snapshot>/models/autoports/zai_org_glm_5_3_flash/vllm_bundle
python -m vllm.entrypoints.openai.api_server --model $GLM53_CKPT --served-model-name $GLM53_CKPT zai-org/GLM-5.3-Flash --tokenizer $GLM53_CKPT --hf-overrides \{\"architectures\":\ \[\"TTGlm5NextForConditionalGeneration\"\]\} --block-size 128 --max-num-seqs 16 --max-model-len 239616 --port 8131 --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 --default-chat-template-kwargs \{\"reasoning_effort\":\"low\"\} --enable-log-requests --additional-config \{\"tt\":\ \{\"sample_on_device_mode\":\ \"all\"\,\ \"trace_region_size\":\ 100663296\,\ \"l1_small_size\":\ 24576\,\ \"fabric_config\":\ \"FABRIC_1D\"\}\}
```

**p300x2-wide** (`../vllm/evidence/final/wide_servercmd.txt`)

```bash
GLM53_CACHE_DIR=/home/raahem/.cache/glm53_flash
GLM53_PROGRAM_GUARD=1
GLM53_TRACE_REGION=167772160
GLM53_VLLM_ASYNC_DECODE=1
HF_MODEL=zai-org/GLM-5.3-Flash
MESH_DEVICE=P300x2
VLLM_ENGINE_READY_TIMEOUT_S=3600
EXTRA_MODELS_DIR=<snapshot>/models/autoports/zai_org_glm_5_3_flash/vllm_bundle
python -m vllm.entrypoints.openai.api_server --model $GLM53_CKPT --served-model-name $GLM53_CKPT zai-org/GLM-5.3-Flash --tokenizer $GLM53_CKPT --hf-overrides \{\"architectures\":\ \[\"TTGlm5NextForConditionalGeneration\"\]\} --block-size 128 --max-num-seqs 32 --max-model-len 137216 --port 8131 --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 --default-chat-template-kwargs \{\"reasoning_effort\":\"low\"\} --additional-config \{\"tt\":\ \{\"sample_on_device_mode\":\ \"all\"\,\ \"trace_region_size\":\ 167772160\,\ \"l1_small_size\":\ 24576\,\ \"fabric_config\":\ \"FABRIC_1D\"\}\}
```

**p300x2-1m** (`../vllm/evidence/final/1m_servercmd.txt`)

```bash
GLM53_CACHE_DIR=/home/raahem/.cache/glm53_flash
GLM53_DSA_INDEX_BLOCK_MB=32
GLM53_DSA_SHARD=1
GLM53_MOE_SORTED_ROWS=512
GLM53_PROGRAM_GUARD=1
GLM53_TRACE_REGION=67108864
GLM53_VLLM_ASYNC_DECODE=1
HF_MODEL=zai-org/GLM-5.3-Flash
MESH_DEVICE=P300x2
VLLM_ENGINE_READY_TIMEOUT_S=3600
EXTRA_MODELS_DIR=<snapshot>/models/autoports/zai_org_glm_5_3_flash/vllm_bundle
python -m vllm.entrypoints.openai.api_server --model $GLM53_CKPT --served-model-name $GLM53_CKPT zai-org/GLM-5.3-Flash --tokenizer $GLM53_CKPT --hf-overrides \{\"architectures\":\ \[\"TTGlm5NextForConditionalGeneration\"\]\} --block-size 128 --max-num-seqs 1 --max-model-len 1048576 --port 8131 --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 --default-chat-template-kwargs \{\"reasoning_effort\":\"low\"\} --additional-config \{\"tt\":\ \{\"sample_on_device_mode\":\ \"all\"\,\ \"trace_region_size\":\ 67108864\,\ \"l1_small_size\":\ 24576\,\ \"fabric_config\":\ \"FABRIC_1D\"\}\}
```

## p300x2-wide re-run after the stage-10 MoE-bucket fix (2026-10-21, 0bcf89f7fd6)

Command: `~/glm53-flash-bringup/scripts/s9_wide_rerun.sh <snapshot>` with REPROBE=1, one ticket (snapshot s9l = pure
`git archive` of 3872b583092 = 0bcf89f7fd6 + stage-4 docs; tt/ code identical to 0bcf89f7fd6). Evidence:
`doc/vllm/evidence/final/wide-r2_*`.

- Incidents before the run (infrastructure, not model results): the first attempt (ticket wide2) hung at device open.
  An out-of-memory kill of the preceding stage-3 gate (host memory: a CPU HF scoring started next to it) left UMD's
  inter-process chip mutexes (`/dev/shm/TT_UMD_LOCK.CHIP_IN_USE_*`, `MEM_BARRIER_*`) held by a dead process. The
  coordinator removed the stale lock files with the user's approval and stopped the hung probe; the ticket released
  with a board reset. The HF scoring now runs only on an idle host.
- 32-slot G9 probe (160 MiB region): MoE row buckets [1, 2, 4, 8, 16, 32], all on the sorted path; capture 107.3 MiB;
  1,105 blocks -> 137,216 (unchanged); copy 1.0.
- Server (`SERVER_CMD` in `wide-r2_servercmd.txt`; GLM53_TRACE_REGION 167,772,160, async on):
  - burst of 32 concurrent greedy requests: 32/32 equal to solo greedy (0a2663d0b6b: 4/32). `SMOKE_RESULT PASS`.
  - primary 128/128/1: TTFT 278 ms, TPOT 120.89 ms, ITL 120.9 / P99 121.9 ms, 8.2 tok/s.
  - CI 100/100/32: TTFT 7.8 s, TPOT 161.1 / P99 161.5 ms, 134.6 tok/s.
  - sweep (8/8 rows, no failed or short requests), TPOT / total tok/s: 128 input: 1 user 121.0 / 8.2, 8 users
    130.0 / 55.2, 16 users 145.8 / 91.2, 32 users 160.1 / 144.8; 1,024 input: 1 user 122.8 / 8.0, 8 users 132.4 / 51.4,
    16 users 146.3 / 82.2, 32 users 163.5 / 123.6.
  - leftover processes after the probe and the server: none.

## Notes

- Indexer cost: the sparse DSA indexer selects over all pools of the model context, not over the filled prefix. The
  model context comes from max_model_len (`pad_page_table` pads every block table to T/128 columns). So the decode
  time grows with max_model_len. The A6 and A8 numbers are the baseline before the stage-10 DSA width buckets.
