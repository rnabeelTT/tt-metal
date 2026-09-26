# Qwen3.5-122B-A10B — stage 10 (optimized vLLM) work log

Times UTC, 2026-09-26. Baseline = stage 9 commit 0e57369c7a6 (`doc/vllm_integration/`): served 48 layers at
`--max-num-seqs 32 --max-model-len 262144`, primary 128/128/1 TTFT 474 ms, TPOT 124.2 ms (8.05 t/s/u), CI burst 100/100/32
120.7 tok/s; single-slot server TPOT 23.9 ms = the full-model step. Gaps named by the stage-9 review: (1) one decode trace at
max_batch_size -> a lone user pays the 32-wide MoE step; (2) ~120 ms of TTFT per admission at 32 slots = the host round trip
of the GDN state in `prefill_serving`; (3) legacy `reset_batch` decode-input contract, overlap off.

### 02:15 — plan (`~/qwen35-122b-bringup/artifacts/stage10_plan.md`): decode bucketing first (qwen36 pattern: one decode trace per
power-of-two width, selected from the plugin's front-packed active prefix), then the device-side GDN slot write, then the
explicit decode-input contract / overlap proof, long-context requests, before/after with the same runner.

### 04:01 — width probe (`scripts/10-probe-widths.py`, 2 layers, B=4 build, eager decode at widths 1/2/4 through the base
Generator): widths 1, 2, 4 ran and compiled (956 -> 1149 programs), then the first width-2 step of the third run HUNG on
device. tt-triage (`doc/optimized_vllm/triage/width2_reshape_hang_triage.txt`): the running op was
`ReshapeViewDeviceOperation` on a `[2, 1, 768]` bf16 TILE tensor (the decode embedding output, `[B, 1, dim/TP]` ->
`[1, 1, B, dim/TP]`), every worker's `reader_reshape_tiled` waiting in `cb_reserve_back`, writer PC outside the kernel. For
1 < B < 32 that reshape is a real tiled re-pack (one padded tile row per token -> B rows in one tile); at B = 1 and B = 32 it
is a view / a plain tile-row move, which is why neither the B=1 build nor the B=32 server ever ran it. Killed, `tt-smi -r`
(two bus errors, third reset OK).

### 04:20 — fix + implementation: `Qwen35MoeModel._decode_tokens_view` hands the embedding a `[1, B]` token row (a view of
the persistent `[1,1,1,32]` buffer) so the embedding emits `[1, B, dim/TP]` and the step's reshape to `[1, 1, B, dim/TP]` is a
free view at EVERY width (the B=1 and B=32 streams of the adapter tests are byte-identical to stage 9's, so the layout change
is numerically transparent). Decode bucketing: `tt/decode_buckets.py` (widths, width selection from the active prefix, per-width
trace stores bound into the shared Generator, sampler trace namespace per width, device token / position carry-over on a width
switch, host-input slicing); generator `_warm_decode_eager(B)` / `_capture_decode_trace(B)` / `capture_traces(widths)`; adapter
`_select_decode_bucket` (full width for host-logits mode and for `slot_remap` steps; an unflagged switch forces `reset_batch`),
`DECODE_BUCKETS` histogram line; KV budget accounts for the trace region (768 MiB: the 512 MiB region was 440 MiB full in
stage 9). New test `test_vllm_adapter_decode_buckets` (B=4: widths 1/2/4).

### 04:30 — test iterations (2 layers): (i) a strict trace-region check failed every test on the test mesh's small default
region -> the check binds only when the KV budget ran in the process; (ii) the guard fired at the first width switch
(1147 -> 1148): the carry-over's device-to-device `ttnn.copy` of the token buffer is a program -> host round trip of the 32
integers instead; (iii) the width-1 stream reproduces the B=1 BUILD's tokens and the width-4 stream the B=4 build's: on the
degenerate 2-layer model they fork at token 1 at a 0.03-logit tie (7.656 vs 7.625; probed with a host-logits step on the
rebuilt state) -> the test asserts determinism per width sequence, agreement to the first fork, and the fork being a near
tie (< 1 logit) — the same class of effect as stage 9's B=1 vs B=32 servers (900 tokens identical, 1000-token fork at
0.125). Result: 5 passed (`logs/10/adapter-test2.log` 4/4 existing, `adapter-test6.log` the bucket test: widths used
{4: 29, 1: 22, 2: 7}, 12 switches, no program compiled after the captures).

### 04:52 — 48-layer "after" run started (`scripts/10-serve.sh` TAG 48l-buckets: same runner / stages / profile / lengths /
mesh / sampling mode as stage 9; TT config trace_region_size 768 MiB): widths [1, 2, 4, 8, 16, 32]; KV pool 11.066 GiB =
1,820,160 tokens (−41,088 tokens for the +256 MiB trace region). (A first launch of the script waited on its own pgrep
pattern — the flock already serializes device jobs; the wait line was removed.)

### 05:05 — device-side GDN slot write implemented while the run warms up (`Qwen35MoeModel._snapshot_gdn_scratch_device`:
`ttnn.clone` of the bound B=1 scratch's recurrent state + K conv taps per GDN layer; `_write_gdn_slot_device`: `write_slot`
per layer; `prefill_serving` binds the scratch per request, prefills, snapshots, rebinds the batched buffers and writes the
slot BEFORE any further trace replay — a clone held across a replay would alias the trace's baked intermediates, the qwen36
note). Warm-up `_warm_slot_writes` runs the same device path for every slot. Not in the running server (imported before the
edit); exercised next by the adapter tests (queued) and a second 48-layer run.

### 05:12 — first bucketed 48-layer run (`doc/optimized_vllm/readiness_vllm_48l-buckets/`, sampling suite only: the runner
stops after a failed suite): 70 passed / 4 skipped / 3 FAILED — `test_batch1_seed_reproducible[0,1]` (one seeded user,
temperature 2, top-k 50, 10 sequential runs -> 10 unique streams) and `test_request_isolation::test_mixed_params_batch`.
Server facts: widths [1, 2, 4, 8, 16, 32] captured (1682 programs, none compiled after the captures through the whole
suite), trace region 603 of 768 MiB, KV pool 1,820,160 tokens, DRAM 30.0 / 31.08 GiB; `DECODE_BUCKETS` histogram shows
single-user tests on the width-1 trace. The queued final run and hold chain were cancelled until the failure is understood.

### 05:20 — localisation. (1) 2-layer adapter test `test_vllm_adapter_seeded_buckets` (B=4 build): seeded T=2 streams at
widths 1 / 2 / 4 reproduce across two runs -> the seed manager / sampler mechanics are width-agnostic. (2) Held bucketed
48-layer server, `scripts/10-seedprobe.py` (`artifacts/seedprobe-base.json`): N identical seeded requests concurrently,
R rounds: width 1 -> 6 unique of 6; width 2 -> 1 unique of 12; width 4 -> 1 of 24; width 8 -> 1 of 24; width 16 -> 1 of 48;
GREEDY 64-token re-runs at width 1 -> 6 unique of 6 (identical first ~160 chars, forks later). So the served width-1 step
alone is intermittently non-deterministic, run to run, in the model output (not the sampler); widths >= 2 are
bit-reproducible. Same SDPA-decode geometry at widths 1 and 2 (default max_cores_per_head_batch 16 -> 16 cores per head at
both; tree reduction fixed-order) -> SDPA is not the discriminator. Candidates: the MoE gather path's T == 1 special cases,
an async-readback race on the fast 24 ms step, or a B == 1 special case in a shared kernel. In-process probe queued
(`scripts/10-w1probe.py`: traced sync / traced async-overlapped / eager / eager with the masked MoE path, widths 1 vs 2 / 4).

### 05:50 — root cause (in-process probe `scripts/10-w1probe.py`, 48-layer B=32 build through the adapter, 24 greedy tokens
per stream, `artifacts/w1probe.json`, `logs/10/w1probe.log`):

| experiment | width | path | two runs |
|---|---|---|---|
| E1 | 1 | traced, synchronous reads | DIFFER (first fork at token 4) |
| E3 / E6 | 2 / 4 | traced, synchronous reads | identical |
| E2 | 1 | traced, async reads overlapped like the plugin | DIFFER (token 2) |
| E4 | 1 | eager (no trace) | DIFFER (token 4) |
| E5 | 1 | eager, MoE masked path forced (GATHER_MAX_TOKENS 0) | identical |

So the width-1 non-determinism is in the model's device ops (eager too, no trace / readback involvement) and it is the
gather-mode MoE at T = 1: widths 2 and 4 run the same gather kernels per token but through per-token SLICES of the
activation / router ids / scores, while T = 1 took a shortcut (the block's L1-interleaved activation and the router
outputs handed straight to `sparse_matmul`, no concat). This is the stage-7 "intermittent LSB non-bit-identity of the
attention layer's traced decode at B=1" (MoE-knob dependent, `moe_router_l1` / L1 logits path) seen at full scale: on 48
layers a flipped expert at a router near-tie changes whole tokens. Fix: `Qwen35MoE._routed_gather` runs T = 1 through the
same slice / concat code as T > 1 (`QWEN35_MOE_GATHER_T1_SHORTCUT=1` restores the shortcut for probing); the masked path
is the alternative (deterministic in E5). Verification queued: traced width-1 streams x2 + traced step time at widths
1 / 2 / 4 / 8 with the fix (`w1probe-fix`), then the full 48-layer readiness run.

### 06:25 — the T = 1 gather rewrite (explicit clones) did NOT remove the width-1 non-determinism (`w1probe-fix2`: traced
width-1 streams differ in 3 of 3 pairs, fork at token 2; width 2 identical). The two outcomes are always the same two
streams, one of them == the width-2 stream (bimodal = a near tie at token 2 flipped by LSB noise). E5's "masked MoE is
deterministic" rested on two runs = weak evidence under a coin-flip fault. Traced step times (deferred readback):
width 1 25.1 ms, width 2 37.7 ms, width 4 50.8 ms (gather path: ~12.6 ms per extra token = per-token op count),
width 8 79.8 ms (masked path), width 32 124 ms (stage 9).

### 06:45 — layer-level bisection (`scripts/10-bisect.py`, eager decode, every layer / mixer / MoE output copied to host at
the forking step, 4 repetitions; `artifacts/bisect-2.json`, `logs/10/bisect-2.log`): layers 0..3 (three GDN layers and the
first attention layer) are BIT-IDENTICAL across runs at that step; the first differences appear in layer 4 (a GDN layer):
the layer-4 output (`L04_gdn_out`) in all three comparisons; within layer 4 the GDN mixer output differs in one run (1 bf16
LSB, ~900 of 3072 elements) and only the MoE output in the other two (~1e-4), then every later layer at LSB level; the greedy token flips where a near tie is hit. With identical layer-4 inputs and no data
dependence, this is a device-side race/timing effect on the width-1 step that starts right after the first attention
layer's ops (paged_update_cache on a 1-core shard grid, SDPA-decode with 16 cores per head of which most exit early at
B = 1, head concat / slice at B = 1) - the stage-7 "intermittent attention-layer replay non-bit-identity at B=1". Not a
property of one arithmetic op; widths >= 2 never show it (E3 / E6 / seedprobe: 0 differences in > 100 streams).
Remedies under measurement: (a) serve single users on the width-2 trace (known deterministic; +12.6 ms per token with
the gather MoE), (b) masked MoE path at width 1 (`w1probe-masked`: determinism x4 + step times).

### 07:10 — masked MoE at width 1 (`w1probe-masked`, GATHER_MAX_TOKENS 0): 6 / 6 identical width-1 streams (traced sync,
traced sync again, async-overlapped), but 69.5 ms per step (width 2: 75.7, width 4: 76.6, width 8: 79.8) - the masked path
costs ~70-80 ms at any T <= 8, unusable for single users. Gather-variant probe in EAGER mode (`scripts/10-gatherprobe.py`,
6 x 16-token streams per variant, `artifacts/gatherprobe-1.json`): clones (current) 4 unique / 6; stage-9 shortcut 6 / 6;
sparse-matmul outputs in DRAM 5 / 6; plain reshape + sum tail 6 / 6; both 5 / 6; the single token duplicated into a
2-row T = 2 batch 5 / 6. Every gather variant is non-deterministic at width 1 - including the exact T = 2 code that is
deterministic with two REAL users - so the MoE is not the trigger: the masked path "fixed" it by slowing the step
(70 ms), i.e. a timing race left by a B = 1-specific op elsewhere in the layer (attention or GDN at B = 1), which is what
the bisection (first differences after the first attention layer) suggested. Probe queued: attention block padded to two
identical rows at B = 1 (`attn_pad2`).

### 07:40 — attention / GDN padding probes (`gatherprobe-attn`, `gatherprobe-gdn`, eager, 6 x 16-token streams each): the
attention block padded to two identical rows at B = 1 -> 5 / 6 unique (x2); the GDN block padded -> 6 / 6 and 5 / 6; both
padded -> 5 / 6. Together with the gather variants: no block-level padding or gather rewrite removes the width-1
non-determinism; the only reproducible width-1 configuration is the slow masked MoE (70 ms), and every configuration
with two or more live rows is bit-reproducible. The remaining B = 1-only ops are outside the decoder blocks (token view /
embedding, norms, LM head and sampler pad at one row) or a timing-dependent race between blocks; isolating it needs
tt-metal-level tooling (watcher / kernel timestamps) beyond this stage's budget -> recorded as an open tt-metal item.

### 07:50 — decision: `QWEN35_DECODE_MIN_WIDTH=2` (default): the width-1 trace is not captured; a single live user runs on the
width-2 trace with one idle row (37.7 ms per step measured in-process; the idle row carries token 0 / position -1 like the
31 idle rows of the stage-9 B=32 trace). The T = 1 gather shortcut is restored (the copies changed nothing; the standalone
B=1 generator keeps its stage-8/9 numerics); the probe knobs stay (`QWEN35_MOE_GATHER_T1_SHORTCUT`,
`QWEN35_MOE_GATHER_MAX_TOKENS`, `GATHER_KNOBS`, `QWEN35_SDPA_DECODE_GRID`). Adapter tests (`logs/10/adapter-test9.log`):
6 passed (prefill/decode B=1/4, stale inputs B=1/4 incl. the contract-v1 arms, decode buckets [1,2,4] mechanics with
MIN_WIDTH=1, seeded widths with the default [2,4]). Final sequence launched: full 48-layer run (`48l-final`) then the
held-server chain.

### 08:40 — second bucketed 48-layer run (widths 2..32, slot write on device, contract v1; `readiness_vllm_48l-minwidth2/`):
72 passed / 4 skipped / 1 FAILED: `test_mixed_params_batch` — request "temp 0.5, seed 42, repetition penalty 3" reproduced
in run 1 and differed at token 4 in run 2 ('10. 45 ACP vs .' vs '10. 45 ACP, .'). The single-user seeded tests passed (width 2
is deterministic). Mechanism: the plugin admits the 10-request burst as 1 + 9 (server log "Prefilling 1 request(s)" then
"Prefilling 9"), so the first-admitted request decodes at width 2 before the batch reaches width 16, and short requests
finishing shrink the width; the shuffled second run gives the same request a different width history. Per-row numerics
differ between widths (gather-mode MoE at 2-4 vs masked at 8+; SDPA-decode cores per head 16 / 16 / 13 / 6 / 3), so a
near tie at temperature 0.5 flips. Composition-independent numerics would need one MoE path for every width (masked at
width 2 = 75 ms; gather at 32 = 400 ms) and a fixed SDPA core count: not in this stage. Rule adopted: a batch with any live
SEEDED row runs at the full width (stage-9 numerics, `QWEN35_SEEDED_FULL_WIDTH=1` default; `DECODE_BUCKETS` line counts the
forced steps); unseeded batches (greedy and unseeded sampling, the common serving case) use the buckets. Consequence:
seeded single users decode at 124 ms per token; greedy / unseeded single users at ~38 ms. Adapter test updated (seeded
rows -> width B; a greedy single user -> width 2). Third full run queued behind the held-server chain.

### 08:50 — held-server chain (`scripts/10-hold-chain.sh`, widths 2..32, `readiness_vllm_48l-hold/server.log`): readiness
benchmark against the held server at concurrency 8 (16 requests: TPOT 93.8 ms, 10.7 t/s/u, 67.4 tok/s, TTFT p50 3.24 s) and
32 (64 requests: TPOT 131.1 ms, 142.8 tok/s, TTFT p50 12.0 s) -> `concurrency/`. Direct checks (`direct_checks_stage10.json`):
131,072-token request TTFT 132.2 s and 262,080-token request TTFT 283.4 s, both 64 coherent tokens (the 262144 contract
served end to end; ~1 ms per prompt token at long context, vs 0.17 ms per token for a 2048-token prompt: attention over
the earlier blocks grows with the position); per-width served latency N = 1 / 2 / 4 / 8 / 16 / 32: TPOT 37.7 / 39.4 / 51.8 /
93.1 / 107.6 / 125.1 ms, aggregate 24.6 / 43.1 / 60.8 / 66.8 / 100.6 / 140.6 tok/s, TTFT 0.41 s for one user (+~0.4 s per
concurrently admitted request); chat page growth on the width-4 trace vs solo on width 2: all 400 tokens, coherent, forks at
tokens 11 / 53 / 55 (cross-width numerics -> a solo run is not a same-width control; same-composition repeat queued as
`10-direct2.py`). `DECODE_BUCKETS` over the chain: {2: 1577, 4: 525, 8: 379, 16: 127, 32: 269} steps, 16 switches;
`PROGRAM_CACHE 1591 ... new: 0` after every admission.

### 09:07 — same-composition controls (`scripts/10-direct2.py`, held server; its server.log was NOT saved: the chain's copy step
had a wrong `mkdir` path (`logs/10-hold2-chain.log` lines 8-9) and its runner log name collided with the first hold chain's —
both fixed in `10-hold2-chain.sh` after review round 1; surviving evidence: the JSON, `logs/10/direct-checks-stage10-part2.log`,
the chain stdout, and the script's fixed configuration): chat page
growth x2 on the width-4 trace [True, True, True]; raw 3 x 300-token page growth x2 [True, True, True]; the stage-9
1000-token prompt x3 on the width-2 trace: 1 unique, == the stage-9 served text. Third full run + adapter test rerun queued.

### 10:05 — FINAL evidence run (`scripts/10-serve.sh` TAG 48l-final, third attempt; `readiness_vllm_48l-final/`, `logs/10/vllm-48l-final.log`):
run_vllm_server PASS — sampling 73 passed / 4 skipped (11:05), qualitative 6 / 6 coherent, primary 128/128/1: TTFT 426.4 ms,
TPOT 37.70 ms = 26.52 t/s/u, ITL p99 41.2 ms, 24.5 tok/s (stage 9: 474.1 / 124.19 / 8.05 / 7.88); CI burst 100/100/32: TTFT p50
11.50 s, TPOT 133.8 ms, 129.4 tok/s (stage 9: 13.57 s / 130.85 / 120.7). `DECODE_BUCKETS` {2: 3582, 4: 24, 8: 77, 16: 21, 32: 1183},
46 switches, 649 seeded full-width steps; `DECODE_CONTRACT v1 steps 4875, reloads 498`; no legacy-contract warning; 1591
programs, `new: 0` throughout; trace 578 / 768 MiB, DRAM 29.99 / 31.08 GiB. Adapter tests 6 / 6 (`logs/10/adapter-test11.log`).
`10-optimized-vllm.check.sh` exit 0 (`logs/10/check.log`). perf_summary.json written (`scripts/10-perf-summary.py`).
Greedy qualitative completions match stage 9's for the first 134-360 characters, then fork (width 2 vs width 32 numerics).
Review round 1 launched (`artifacts/stage_review_prompt_10.md`).

### 10:35 — review round 1: more-work-needed on one item: the second held server's log was cited as evidence but never saved
(`10-hold2-chain.sh` copied into a directory it had created under the wrong name; its runner log also overwrote the first
hold chain's). Fixed: citations corrected to the surviving evidence (JSON, direct-check log, chain stdout), script paths
fixed. Advisories applied: CI-burst TPOT attributed (mean +2.3%, p99 +8.3% = admission-time slot_remap full-width steps and
forced reloads during the burst's fill; the width-32 step itself unchanged per the held-server c32 run); README section 7
gained the untested "T=1 gather inside the width-2 step" cell, the ~75 ms TTFT remainder, the KV-pool delta of the 768 MiB
region and the per-admission full-width blip; trace figure 603 -> 578 MiB (final run); bisection wording aligned with
bisect-2.json; DECODE_BUCKETS now also counts slot_remap full-width steps and forced resets; adapter bucket test arm (d):
stale host tokens across flagged / unflagged width switches reproduce the correctly fed streams (`logs/10/adapter-test12.log`).

### 11:05 — review round 2: clean-pass. Advisories: `bringup/` (untracked stage-6 references) noted as non-stage input; plugin
registration hunk committed locally in ~/vllm-tt-plugin (not pushed); the width-1 race is recorded as an untracked open
item with its repro scripts. Stage 10 committed locally.
