# vLLM integration — work log (stage 9, Qwen/Qwen3.5-122B-A10B on (1,4) P300x2)

Times UTC, 2026-09-25/26. Driver workspace `~/qwen35-122b-bringup/` (`scripts/09-*.sh`, `logs/09/`); stage-8 baseline
commit e8de3cdd6ba; vLLM TT plugin `~/vllm-tt-plugin` @ 5961e48 (editable install), vLLM 0.26.0.

### 21:20 — contract study (`~/qwen35-122b-bringup/artifacts/stage9_notes.md`)
Plugin: `initialize_vllm_model` -> `get_max_tokens_all_users` (KV block budget) -> legacy `allocate_kv_cache(shape, dtype,
num_layers)` -> `warmup_model` in two phases (prefill / decode with enable_trace=False, then True) -> `prefill_forward(tokens,
page_table, kv_cache, prompt_lens, start_pos, empty_slots, sampling_params)` returning `(tokens | logits, rope_deltas)` (the
HF config has mrope, so the runner unpacks two values) -> `decode_forward(...)` with the plugin's kwargs, `read_from_device`,
then `read_decode_output(async_read=True)` / `process_decode_output_host`. `sample_on_device_mode: all` (enforced by the
readiness runner) means the plugin expects SAMPLED TOKENS from prefill too. Chunked prefill is disabled for models
without `supports_chunked_prefill` (full prompts up to max_model_len).

### 21:40 — implementation
- `tt/prefill_len.py` / `tt/model.py::_run_bucket_dev`: the bucket program selects a 32-row block (row 0 = the last real
  token) and emits BOTH the replicated next-token logits (host / readiness path, unchanged shape [1,1,vocab]) and the
  pre-gather vocab-sharded logits [1,1,32,vocab/TP] — the decode sampler's exact input spec, so the serving adapter samples
  the prefill token on device with the programs the decode warm-up already compiled (no compile after the captures).
  Bucket traces store both outputs; `prefill_masked_bucket(return_sharded=True)`.
- `tt/generator.py`: `external_kv=True` construction (no pool, no warm-up), `tt_generator=` (ONE tt_transformers Generator
  object: the adapter), `prebuilt=(args, model)`, `bind_kv_cache(kv, num_blocks, width)` (scratch block = vLLM's num_blocks;
  warm-up page table at vLLM's width), `warm_programs` / `capture_traces` (the two plugin phases; `_warm_decode_trace` split
  into `_warm_decode_eager` + `_capture_decode_trace`), `prefill_serving(tokens, page_table, prompt_lens, slots, sampler)`
  (per request through the warmed buckets on the B=1 scratch, GDN state written into the request's slot, on-device sampling
  of row 0 while the bucket output is live), `_prefill_single(want_sharded=True)`.
- `tt/generator_vllm.py`: `Qwen35MoeForConditionalGeneration(Qwen36ForCausalLM)`; multimodal processor re-registered with
  image / video limits 0 (text-only scope); `initialize_vllm_model` (context from `doc/context_contract.json`
  datatype_sweep_stage8, `QWEN35_LAYER_INDICES` for the reduced target), `get_max_tokens_all_users` = 0.85 x the measured
  pool (2.25 M tokens) unless `QWEN35_MAX_TOKENS_ALL_USERS`, `allocate_kv_cache` (+1 scratch block, 32-aligned, policy KV
  dtype), warm-up hooks mapped onto the two phases, `prefill_forward` (slots = `empty_slots`, start_pos must be 0, sampler
  = the request's params broadcast to all sampler rows -> `apply_prefill_state` -> `sample(enable_trace=False)` -> row 0),
  `decode_forward` = GDN `slot_remap` then `Generator.decode_forward` (plugin kwargs untouched, program guard).
  `model_capabilities`: async decode True (split contract implemented; to be proven by the plugin's stale-token tests),
  overlap False, prefix caching False, chunked prefill False.
- Registration: `vllm_metadata.json` (EXTRA_MODELS_DIR bundle) and `~/vllm-tt-plugin/src/vllm_tt_plugin/platform.py::
  register_tt_models` ("TTQwen3_5MoeForConditionalGeneration"; plugin repo change, not pushed).
- `tests/test_generator_vllm.py`: adapter contract on the 2-layer model without vLLM (construction, KV hand-off, two-phase
  warm-up, prefill logits vs on-device greedy tokens, split decode contract with deferred reads, slot remap, program guard).
Smoke chain `scripts/09-smoke.sh` launched: bucket-program equivalence (stock / dev / traced, 1..4097 tokens), adapter test,
generator suite.

### 22:35 — first smoke: bucket-program equivalence bit-identical (stock / dev / traced, 1..4097 tokens,
`prefill_bucket_bench_2layers_stage9.json`); adapter test FAILED on `Input Tensor is not allocated` in the prefill sampler:
`tt_all_gather` (all_gather_async) frees its INPUT, so the vocab-sharded logits handed to the sampler were gone
(`logs/09/probe-sh*.log`: allocated after the linear, freed after the gather). Fix: gather a DRAM clone and keep the sharded
tensor; probe: eager and traced on-device prefill sampling both return the host argmax (1146). The generator suite's
`test_generator_cross_request_token_feedback` failed its strict re-run equality at a near tie (the documented LSB
non-determinism; the same assertion class made near-tie aware in stage 7) — made near-tie aware with the eager truth set.
Smoke chain relaunched, 2-layer server run queued behind it.

### 22:55 — second smoke: adapter test still failed, both in the TEST: `process_decode_output_host` returns
`(tokens, log_probs)` (the plugin unpacks it the same way) and the tokens come back as a float tensor — the test now
unpacks the tuple and casts to int64. Generator suite: `test_generator_greedy_sampler_paths[topk_candidates]` failed its
strict re-run equality at the same near tie as the cross-request test (step ~4 of an 8-token stream on the repetitive
24-token probe prompt; the token-out bench's blocking re-run forks at a near tie in every policy incl. stage 7, steps
8-59) — made near-tie aware with the eager truth set like the other two tests. Re-run queued (`09-smoke2.sh`), 2-layer
server run queued behind it.

### 23:08 — adapter contract test PASSED for B=1 and B=4 (`logs/09/adapter-test2.log`): on-device prefill sampling == host
argmax for every request, deferred-read decode contract, identical greedy streams for the B=1 and B=4 builds
(slot 0: 1146 467 4625 43506 99211 10303 227805), program cache 903 -> 903 / 949 -> 949 (no compile after the captures).
Generator suite 9 / 10: `test_generator_long_prompt_segments[4097]` strict `dec == dec2` at a 0.125-logit near tie
(prefill(T+1) top-2 [12116, 1810]; the two decodes are the two members) — the last strict equality of the suite made
near-tie aware (both decodes must be in the near-tie set); re-run queued behind the server run. First 2-layer
`run_vllm_server` run started (serve, sampling smoke, qualitative; `logs/09/vllm-2l.log`).

### 23:10 — first `run_vllm_server` launch (2 layers): the engine registered the adapter, disabled chunked prefill for it,
opened the (1,4) mesh with FABRIC_1D, took `get_max_tokens_all_users` = 1,911,360 (0.85 x the contract pool), built the
model with the shipped precision policy, then failed in `allocate_kv_cache`: vLLM counts the HF config's 12 attention
layers while the reduced target has 1 — the adapter now allocates for the layers the built model has (warning logged;
vLLM's per-layer budget is host bookkeeping). Server run re-queued behind the generator-suite re-run.

### 23:45 — server up on the 2-layer target (warm-up phases run, 949 programs, prefill of 1 + 3 requests clean), then the
program guard stopped the FIRST served decode: 949 -> 955 programs. Bisecting with an in-process probe of the sampling
parameter classes (greedy, k/p/temperature, seeds, penalties, mixed: all +0 programs, `logs/09/probe-programs*.log`) and
then logging the plugin's exact decode kwargs (`QWEN35_VLLM_DEBUG_KWARGS=1`, `server.log`): the plugin hands the first decode
a NON-identity `slot_remap` [1, 2, 3, 0] (batch condense after the 7-request mixed batch of `test_mixed_params_batch` went
through 4 slots) -> `TPGatedDeltaNet.remap_slots` (per-slot slice + concat + in-place copy on the recurrent and conv states)
had never run: 6 programs compiled after the captures. Fix: `_warm_slot_remap` (a rotation and its inverse) in the batched
prefill warm-up, both construction paths; the adapter test now drives a non-identity remap and asserts the program count.
The tt-metal Debug/Op logging route (`Program Cache Hit`) is compiled out of this Release build — not usable for this.

### 23:58 — with the slot-remap warm-up the mixed batch decoded for five rounds; then (a) the program guard fired again at
955 -> 956 on the `test_min_p` request (host sampling: `sampling_params=None` -> the host-logits decode variant whose
full-vocab LM-head all-gather the on-device-sampling warm-up never ran) - fixed by running the eager host-logits decode in
warm-up phase 1 and capturing its trace in phase 2 (`capture_traces(warm_host_trace=True)`; the standalone constructor
gets the eager pass too); (b) `test_mixed_params_batch`: "Request 0 should be deterministic/reproducible" (temp 2.0, seed
42) - runs 1 and 2 differed from the FIRST token. Probe (`logs/09/probe-seed.log`): the prefill token of a seeded request is
reproducible in a slot (3 / 3) but the very first use of another slot and one decode stream differed -> the seed manager
derives the device seed from (request seed, salt) where the salt is the smallest one not used by ANOTHER slot holding the
same seed (`_next_free_salt`, liveness-blind); finished requests keep their seed until the next DECODE deactivates them
(`deactivate_slots_except`), so a prefill that admits new same-seed requests before that decode (run 2's three seed-42
requests) gets different salts than on a fresh server. Fix in the adapter: at prefill, deactivate the seed state of every
slot that is neither running (positions >= 0 at the last decode, tracked as `_live_slots`) nor being admitted, then apply
the requests' prefill state in admission order. Adapter test + 2-layer server run re-queued.

### 00:05 — smoke sampling tests now run on a live server (no engine error; 2 passed, 1 skipped, 1 failed): the remaining
failure is again seeded reproducibility (temp 0.5, seed 42), because the two runs admitted the three seed-42 requests in a
different order ("Prefilling 3 request(s) ... lengths [3, 6, 3]" vs "[6, 3, 3]") and the common sampler salts a request whose
seed another live slot already holds (`SeedManager._next_free_salt`, smallest free salt among same-seed slots): the stream of
a seeded request depended on its admission rank among same-seed siblings. Adapter: `seed_manager.salt_duplicate_seeds =
False` (the `args.salt_duplicate_seeds` knob) - (seed, position) alone defines the stream; identical seeded requests in one
batch produce identical text, which is what a seed promises and what `test_same_seeds_reproduce_across_batches` checks.
Ghost-slot deactivation at admission stays (it also protects the unsalted path's `_seed_active` bookkeeping).

### 00:18 — reduced target (layers 0 and 3, max_num_seqs 32, max_model_len 4096) PASSES `run_vllm_server` with the FULL sampling
profile, the qualitative stage and both benchmark profiles (`logs/09/vllm-2lfull.log`; artifacts kept as
`doc/vllm_integration/reduced_target_2l/` — inner-loop evidence only, NOT the model's accuracy or performance). The earlier
smoke run (max_num_seqs 4) passed too (`logs/09/vllm-2l.log`). 48-layer run launched: `run_vllm_server --max-num-seqs 32
--max-model-len 262144 --block-size 64 --sampling-profile full`, all four stages (`logs/09/vllm-48l.log`).

### 00:30 — first 48-layer server build (262144 / 32): weights loaded (7:41), KV pool bound (29,897 vLLM blocks + scratch,
1.914 M tokens = 11.64 GiB from the 0.85 x pool rule), phase 1 warm-up done, then OOM in phase 2 while capturing the
2048-token bucket trace: the expert-sorted MoE dispatch asked for a 512 MiB DRAM buffer with 0.29 GiB free
(`MEMORY_REPORT` after phase 1: 30.434 of 31.331 GiB allocated). The 0.85 rule used the B=1 footprint of the context
contract (17.66 GiB); at 32 slots the non-KV footprint is 18.79 GiB (+36.5 MB per slot: GDN state 19.8 MB + sampler /
penalty rows + B-sized buffers) and the prefill transients need ~1 GiB more. `get_max_tokens_all_users` now budgets
`total - (nonkv_B1 + (B-1) x 36.5 MB) - 1.25 GiB headroom` -> 1.86 M tokens (11.32 GiB) at 32 slots, 2.04 M at 1 slot.

### 01:05 — 48-layer `run_vllm_server` (262144 / 32, full profile, qualitative, both benchmarks): PASS (`logs/09/vllm-48l.log`;
artifacts snapshotted to `doc/vllm_integration/readiness_vllm_48l/`). Build 7:43, KV pool 29,114 blocks = 1.86 M tokens
(11.32 GiB), DRAM after both warm-up phases 30.21 of 31.33 GiB, 1186 programs, program guard on throughout. Sampling suite
73 passed / 4 skipped (13:03). Qualitative: 6 / 6 greedy and sampled completions coherent and on topic (thinking-process
openings like the stage-7 suite). Primary single-user benchmark (128 / 128 / 1): TTFT 472.3 ms, TPOT 124.1 ms = 8.06 t/s/u,
ITL p99 126.5 ms. CI serving burst (100 / 100 / 32): TTFT p50 13.45 s (32 prefills serialised before any decode), TPOT
130.9 ms, output throughput 121.2 tok/s. The served decode step at max_num_seqs 32 is the B=32 decode trace (one trace at
max_batch_size, no decode bucketing): ~124 ms per step whether one or 32 users are active, vs 24.1 ms for the B=1 token-out
step of the full-model generator -> the single-user headline pays the 32-wide MoE decode. B=1 server benchmark queued to
measure the adapter's own overhead at equal batch; decode bucketing (per-width traces + slot condense) is the stage-10 item.

### 01:18 — direct checks against a held 48-layer server (`scripts/09-direct.sh` / `09-direct.py`, `direct_checks/`):
`/v1/models` max_model_len 262144; chat request coherent; non-aligned prompts 1000 / 2381 / 4097 tokens accepted with exact
`prompt_tokens`, 32 coherent tokens each (5.0 / 6.6 / 8.0 s wall at the B=32 decode step); page growth: 3 concurrent
300-token generations, request 0 == its solo greedy control; burst of 32: all complete, 117.7 tok/s aggregate.
NOTE: the hold server overwrote `readiness_vllm/server.log` of the passing 48-layer run before it was snapshotted
(`readiness_vllm_48l/server.log.HOLD_RUN_NOT_THIS_RUN`); the other artifacts of that run are intact; a final
self-snapshotting run (`09-serve-final.sh`, TAG 48l-final) is queued behind the B=1 benchmark to carry a matching server.log.

### 01:33 — single-slot 48-layer server benchmark (`--max-num-seqs 1`, 128 / 128 / 1; `readiness_vllm_48l-b1/`): TTFT 352.2 ms,
TPOT 23.91 ms = 41.83 t/s/u, ITL p99 27.4 ms — equal to the full-model deferred token-out step (23.94 ms): the adapter adds
no measurable per-token overhead at equal batch; +8 ms TTFT. The 32-slot server's 124 ms/token is the B=32 decode trace
(MoE gathers up to 256 experts for 32 tokens), i.e. decode bucketing is the stage-10 lever, not vLLM orchestration.
Serialised CI burst at 1 slot: 37.0 tok/s aggregate (TTFT p99 83 s = queueing).

### 02:04 — final self-snapshotting 48-layer run (`09-serve-final.sh`, TAG 48l-final; `readiness_vllm_48l-final/` incl. its
own server.log): PASS again — sampling 73 passed / 4 skipped (13:02), qualitative 6 / 6, primary 128/128/1 TTFT 474.1 ms,
TPOT 124.2 ms = 8.05 t/s/u, CI burst TTFT p50 13.57 s, TPOT 130.8 ms, 120.7 tok/s (first run: 472.3 / 124.1 / 8.06;
13.45 s / 130.9 / 121.2). Build 7:37, KV pool 1,861,248 tokens, DRAM 30.21 / 31.33 GiB after warm-up, 1186 programs.
README head now cites the final run and the repeat.

### 02:10 — `09-vllm.check.sh` exit 0 (`logs/09/check.log`): no degenerate output in the 48-layer qualitative files, context
contract OK (262144 target = supported). The reduced target's qualitative outputs (meaningless 2-layer text, flagged by the
checker as expected) and its server.log (mentions its 4096 cap) were removed from the tree (`reduced_target_2l/NOTE.md`).
Review round 1 launched.

### 02:14 — review round 1: more-work-needed. P1: the page-growth check's concurrent stream 1 (150-token raw book prompt, 300
greedy tokens) ends in a repetition loop and only stream 0 had a solo control, so the evidence could not separate greedy
degeneration from cross-request / page-table corruption. P2: six README claims not matching the artifacts (Prefilling N
coverage "1..32" — the final log's max is 31; "every figure within 2 ms" — the CI-burst TTFT differed by 119 ms; the
page-table refresh attributed to `reset_batch` — it is the base Generator's per-step host compare; section 5 cited the run
whose server.log was overwritten; "29,114 blocks = 1,861,248 tokens" — 29,114 x 64 = 1,863,296; section order 1 2 3 5 4 6 7).
Hard-check gaps: no positive guard-state line inside the engine, no adapter-level stale-input test, the 1000-token continuation
drifting without a control, memory counters / shutdown force-kill not stated. All README items corrected (documentation).

### 02:20 — rework code: the adapter logs `PROGRAM_GUARD active|OFF` at construction and `PROGRAM_CACHE <n> entries ...
(after the warm-up captures: <base>; new: <n-base>)` after every prefill admission and every 2000 decode steps
(`generator_vllm.py::_log_program_cache`). New device test `test_vllm_adapter_stale_inputs` (B = 1 / 4, 2 layers): a
10-step greedy stream fed a WRONG host token (11) and position (+777) on every non-reset step must equal the correctly fed
control (device-resident inputs authoritative), and a page table changed at step 1, before every row crosses token 64 (first version: block 1
-> a fresh block) must be picked up (the only trace input refreshed on that step) with the stream unchanged; every other
step hands an unchanged table (nothing copied). `09-direct2.py`: a solo control for EVERY concurrent stream (raw and
chat-formatted page growth), first-divergence locator + top-2 margin probe, 900 tokens added as a second 1024-bucket length,
served text compared with the standalone generator (`09-control1000.py`). First test iteration: the shared 19-token snippets
gave 56..63-token prompts only when joined and a constant-token 2-layer control stream (cannot show the state advancing) ->
book-text prompts + a non-constant-control assertion; the synchronous read returns `(tokens, log_probs)` for B > 1 -> unpack.

### 02:48 — `09-fix-chain.sh` (device): standalone 48-layer control (`direct_checks/standalone_control.json`): 900-token
continuation " this. Mrs. Southcott had recently attained her five-and-twentieth blessed birthday, ..."; 1000-token
"apped out theirs. Mere messages in the eastern or the western world, had not the one-half of the fame...". Held 48-layer
server (`readiness_vllm_48l-hold2/server.log`: `PROGRAM_GUARD active`, 18 `PROGRAM_CACHE 1186 ... new: 0` lines through the
checks) + `09-direct2.py` (`logs/09/direct-checks2.log`, `direct_checks/direct_checks.json`): max_model_len 262144; chat
"The capital of France is Paris." (stop); 900: served == standalone control; 1000: served == the first round's served text
(deterministic across two servers) but forks from the standalone generator at continuation token 10 ("earthly order" served
vs "eastern or the western world" standalone; the book has "earthly order of events") -> margin probe queued
(`09-margin1000.py`); 2381 / 4097 verbatim; page growth raw: all three concurrent streams == their solo controls, stream 1's
loop ("a very great thing, and ...") reproduces ALONE (loop score 4 in both) = greedy raw-completion degeneration of the
instruct model, controlled; page growth chat (3 x 400 tokens, 1 -> 7 blocks each): all three == solo controls, no
repetition, coherent prose; burst 32/32 at 118.0 tok/s. `09-vllm.check.sh` exit 0 on the updated artifacts.

### 03:30 — adapter tests (`logs/09/adapter-test6.log`, 2 layers): 4 passed — `test_vllm_adapter_prefill_decode[b1,b4]`
(904 / 956 programs, none after the captures) and the new `test_vllm_adapter_stale_inputs[b1,b4]`: 10-step control streams
(non-constant: e.g. slot 0 at B=4 `[82981, 863, 96165, 16663, ...]`), the wrong-host-token/position stream and the
changed-page-table stream byte-identical to the control, program count unchanged. (The `TT_THROW: SubDeviceManagerTracker is
not initialized` lines in the log are the common sampler's `reset_trace` releasing its trace at test teardown after the mesh
closed — the same warning appears in the stage-8 generator test logs; not a served-path event.) First margin probe
(`logs/09/margin1000.log`) measured at a CHARACTER-level common prefix ("...the\nea", not a token boundary) -> discarded;
token-level fork located from the standalone token ids + the re-encoded served text: continuation token 9, 'earth' (served)
vs 'east' (standalone); second probe queued (`09-margin2-chain.sh`).

### 03:45 — token-level fork margin (`logs/09/margin1000b.log`, `direct_checks/fork1000_margin.json`): at prompt(1000) + 9
common continuation tokens the standalone generator's eager prefill logits are 'east' 19.625 / 'earth' 19.5 / 'ear' 18.375:
the served path took 'earth' (-0.125 vs the eager argmax), one bf16 LSB at magnitude 16..32 — the documented LSB near-tie
effect, not a serving-path defect (900-token served == standalone; 2381 / 4097 verbatim). Anomaly ledger: controlled.
Evidence set complete; `09-vllm.check.sh` rerun; review round 2 launched (`artifacts/stage_review_prompt_09_r2.md`).

### 03:50 — review round 2: more-work-needed on ONE documentation row (README section 2 still described the superseded
0.85 x pool rule for `get_max_tokens_all_users`; rewritten to the footprint rule with the log line). Advisory items taken:
the stale-input test's page-table arm was non-discriminating (a fresh private block behaves the same whether or not the
device table was refreshed) -> the arm now starts on a table whose block 1 is SHARED by every row and switches to the
control's private blocks at step 1, two to five steps before the rows (58..61-token prompts) cross token 64, so a stale device table would corrupt the rows' KV
across each other (B = 4) and change the stream; rerun queued. The two misattributed server-log copies
(`readiness_vllm_48l/server.log.HOLD_RUN_NOT_THIS_RUN`, `readiness_vllm_48l-b1/server.log.FINAL_RUN_NOT_THIS_RUN`) were
deleted (the NOTE.md files record the loss). README section 4 now says why the parent adapter's `supports_async_decode:
False` caveat (GDN continuity) does not apply here. Note on the 2-layer adapter test logs: the stale-input CONTROL stream of
the same 58-token prompt differs between the B=1 and the B=4 build from token 3 (`adapter-test6.log` lines 171 / 370) — the
2-layer model is degenerate (near ties everywhere) and the batch width changes the MoE accumulation order; the test does
not assert cross-B equality, and on 48 layers the B=32 server and the B=1 standalone agree byte-for-byte after 900 tokens.

### 04:00 — adapter tests with the discriminating page-table arm (`logs/09/adapter-test7.log`): 4 passed (B=1: 904 programs,
B=4: 956; stale-input streams == controls; the shared-block -> private-block table switch at step 1, 2..5 steps before
the boundary crossing, leaves the B=4 streams unchanged). Review round 3 launched (`artifacts/stage_review_prompt_09_r3.md`).

### 04:20 — review round 3: clean-pass. Advisories applied (documentation): the "one step before the crossing" wording
(the switch is at step 1, 2..5 steps before the rows cross token 64), the served trace-region usage (440 of 512 MiB after
both warm-up phases, README section 3; section 7 no longer claims the extra width traces fit in 512 MiB), work_log order.
Stage 9 committed locally.
