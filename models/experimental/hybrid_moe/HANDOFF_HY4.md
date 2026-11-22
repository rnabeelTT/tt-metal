# Hy4-flash on one QB2: running log

Model: [tencent/Hy4-flash](https://huggingface.co/tencent/Hy4-flash), official BF16 checkpoint (440 GB, 148 shards).
220B parameters (155.5B backbone + 63.5B Engram + 0.43B ViT + 0.85B DFly2 drafter), ~11.3B active per text token.
Reference implementation: transformers PR 50903 at `ManaEstras/transformers@c2ec850510a37978ea83df226589623d5f2be776`
(`src/transformers/models/hy_v4_flash/modeling_hy_v4_flash.py`; a copy lives in `~/m27-work/hy4/`). The DFly2 drafter,
the `hy_v4` reasoning / tool parsers and the serving recipe exist only in vLLM PR 68296.

Branch `raahem/hy4-flash-hybrid` (fork rnabeelTT/tt-metal), off `hybrid-moe-starter` (a76f430eeea). Port:
`models/autoports/tencent_hy4_flash/` (layout follows the Inkling port: own engine + vLLM adapter).

## 2026-11-22: 8-user serving profile, throughput, startup (paused 23:15; resume from ~/m27-work/hy4/orch/RESUME.md)

Resume point with open issues, lane handoffs (`~/m27-work/hy4/{hs,tput,fuse}/HANDOFF.md`), gates and the tradeoff
list: `~/m27-work/hy4/orch/RESUME.md`. Quality floor (never regress; non-exact changes are opt-in): b8 0.8650 /
0.2229, long3k prefill 0.8269 / 0.2933, long3k decode 0.8273 / 0.2930.

Profiles (`serve/serve_hy4.sh`): `PROFILE=default` 8 users x 20480 ctx, SPEC=4; `PROFILE=long` 1 user x 223232, SPEC=4.

| commit | change | before -> after |
|---|---|---|
| e7f05a41532 | long-ctx decode: indexer VLEN + PADSENT default | |
| a8293160602 | PROFILE default / long | |
| 4a68892999e | FAST_DRAW sampling, multi-core MoE prefill dispatch, step stats | 8u temp 0.9 148.9 -> 194.0 tok/s; 32K TTFT 13.25 -> 12.35 s |
| 2851a7a1cee | Engram O_DIRECT load, tt/fastload.py | server ready 1265 -> 379 s |
| 83c1274b31e | decode run-ahead, Engram mlock, input precopy | ready 379 -> 200 s; 8u 195.1 -> 209.4 tok/s |
| 087d1418749 | fast top-p + batched sampling | 8u top-p 0.9 87 -> 187 tok/s |
| 2adc9c99500 | exact fusions Q_SHARD / ROPE_TAIL / IDX_SELX | 8u @2.9K step 44.5 -> 39.4 ms; 1u @18.7K 53.6 -> 54.8 tok/s |

Served (default, greedy): 58.7 / 72.0 / 126.5 / 211.0 tok/s at 1/2/4/8 users, TTFT 0.21-0.8 s.

**Known issue:** run-ahead (83c1274b31e) is not bit-exact when a request finishes (the remaining row is taken from the
batched pass instead of the solo path). Until `~/m27-work/hy4/hs/patches/06-runahead-strict.patch` passes ra_t, serve
with `HY4_RUNAHEAD=0` for exact greedy output.

## Prefix reuse (2026-11-20): multi-turn / repeated prefixes skip their prefill (tt/engine_batch.py)

Engine-internal, vLLM's prefix caching stays off (the adapter sends the full prompt; `alloc(ids=...)` picks the slot).
- A finished request's slot keeps its caches + fed tokens (`kept`); its lane is unloaded at finish (`_park`: indexer
  group keys lane region -> slot; KV rows are already in the slot's cache). New prompt -> the free slot with the longest
  reusable prefix (ties lowest), else LRU. `Hy4Model.rewind(r, ids)`: pos = r, indexer `rewind(r)` (device group keys
  < r/4 kept, the rest zeroed, host mirror re-read), Engram history = ids[r-2:r]; then `prefill(ids[r:])`.
- Exact mode (HY4_PC_EXACT=1, default): r is a fresh prefill's chunk boundary (2048, 2048 + chunk, ...) inside the kept
  prefill's whole chunks (`_full`): every chunk then runs exactly as in a fresh prefill of the new prompt -> bit-identical
  output, no new programs. Prefixes < 2048, decoded tokens and a prompt's last partial chunk are recomputed.
- SWA ring (2592 rows/slot = chunk + window + 32): resume needs [r - 512, r) not overwritten -> r >= highest written
  position (`Hy4Model._hiw`: prefill padding; decode + 96) - 2080. Divergence > ~2K tokens before the old end: no reuse.
- Default OFF (HY4_PREFIX_CACHE=1 enables; with it off finish / alloc / prefill behave as before). HY4_PC_EXACT=0: 512-aligned resume (dense regime / decoded tokens too), device-noise-level
  differences only (4 layers: greedy flips on near-ties, as a fresh prefill at another chunking does).
- Gate `~/m27-work/hy4/pc_t.py` (4 layers, 4 slots): multi-turn via a batch lane, 3-turn chain, divergence, ring
  overwritten -> no reuse, 9.6K prompt (8192 reused), packed interplay: reuse vs fresh max|dlogit| 0.0 over 32
  teacher-forced steps, greedy identical, every case. Served A/B: `~/m27-work/hy4/queue_pc.sh` (serve/hy4_multiturn.py).
- Two pre-existing bugs it exposed (HEAD: two fresh prefills of one prompt in different slots differed, first row up to
  3.1 / decode 0.5-2 in the logits, 4 layers + warm + traces; full model up to 15), fixed regardless of the flag:
  1. Prefill constants cached across requests (`_rows_const` tails / RoPE, `_IDX_DEV`, `_BAND`, `_IOTA`, `_BASE_T`,
     `_ZEROS`, indexer `_gtab` / `_off_cache`) and the indexers' lazy device clones (`_open_pending`, `_tk_pending`,
     `_gk_pending`) were allocated after a decode trace's capture and lived into its replays, which overwrite them
     (`TT_METAL_TRACE_ALLOC_TRACKING=1` names them). `Hy4Model.prefill_caches_release()` after every prefill (both
     engines): host views materialized, entries created since `prefill_caches_mark()` (warm-up, before its first
     capture) freed.
  2. Single-path (solo) decode traces are keyed (rows, regime, bucket), not by slot, but captured the active slot's
     per-slot indexer tensors (group keys, token / group windows): a sequence decoding alone in another slot read and
     wrote the capture slot's group keys. `Hy4BatchEngine._solo_in / _solo_out`: the solo slot's indexer state lives in
     fixed buffers while solo (copied in / back; all single-path traces capture those).
  After both: 4 layers, 8 slots, warm + traces, tracker on: every case 0.0 / identical (`pc_logs/pctrk6.log`); full
  model (Engram, warm, traces, 8 x 12K, tracker on): 9/9 cases bit-identical, +0 programs per reuse prefill
  (`pc_logs/pcfull5.log`). Engine prefill time fresh -> reused: 4861 tok 2.22 -> 0.54 s, 7042 3.01 -> 0.54 s, 9615
  4.17 -> 0.80 s, 3440 1.67 -> 0.80 s.

## Round 3 (2026-11-19, evening): multi-user serving (up to 8 users), prefill

Target (decided): up to 8 concurrent users (buckets 1/2/4/8), maximize throughput and TTFT; 16+ rows per pass hit the
MoE cliff (HY4_MOE_FUSED_ROWS_MAX=8: 1/4/8/16/32 rows = 32/38/42/139/252 ms per pass).

Commits: 4c5eb46aeb3 + 87593ec516a prefill (fused HC over row tiles, row-parallel sparse attention, repeat_interleave
index expansion, dispatch counters in RISC local memory: 32K TTFT 17.7 -> 12.03 s); 80c7a9e45c4 stage 1 (cache slots,
model.set_slot, tt/slab_write); 9666fb76725 stage 2 (tt/decode_batch.py: batched decode); 16fdce8448f stage 3
(tt/engine_batch.py + adapter); 0299fe60a24 dense-only passes skip the indexer score; 60e7e3c5187 vectorized host
inputs; e3000600fab warm-up covers every lane / long chunks; d4a497afb88 packed prefill of simultaneous prompts.

Design (details ~/m27-work/hy4/BATCH_DESIGN.md): every request owns a model slot (its cache regions: base = slot x rows;
per-slot indexer / Engram state); while it decodes it holds a lane of Hy4BatchDecoder: lane l's 64-row KV / token /
group windows are rows 64 l .. of L-lane staging tensors, updated in one pass by block-diagonal one-hot maps, written to
each slot's cache with one slab_write per cache (idle lanes -> trash rows). Sparse layers: one indexer score over the
lanes' group keys masked per row to its lane's visible groups (skipped when every row is < 2048 tokens: dense lists).
Traces per (rows, score width). One live request decodes on the single path (lane written back: no lane overhead);
it rejoins the batch on the next multi-request step or prefill. Spec decoding (DFly2, SPEC=4 default at any NSEQS): the
solo request speculates (verify passes of k + 1 rows); 2+ live requests decode plain in the batch.

| decode, no Engram, 12K slots (tools/serve_batch_smoke.py) | 1 user | 2 | 4 | 8 |
|---|---|---|---|---|
| ms per step | 32.0 | 40.1 | 44.4 | 51.0 |
| tok/s total | 31.2 | 49.9 | 90.1 | 156.9 |

8-user gate (tools/batch_gate.py, Engram on, 8 texts teacher-forced as 8 users): top-1 0.8650 / KL 0.2229 vs single
path 0.8624 / 0.2306; pass 69.9 -> 58.5 ms over the round (dense-only skip, vectorized host). Batched vs single top-1
differ at 7.9% of positions, split evenly vs HF (122 / 112): device numerics, not a bug; outputs are identical across
arrival order / remap / host-sampled steps; no state leak from a slot's previous request.

Served (vLLM, NSEQS=8 CTX=12288, Engram on, greedy, 256 tokens, first run: Engram still loading into RAM):
1 / 2 / 4 / 8 users -> 25.7 / 39.5 / 68.7 / 110.0 tok/s aggregate (13.7 per user at 8); TTFT median 0.43 / 0.92 /
1.57 / 2.72 s (simultaneous arrivals are prefilled one after another, ~0.35 s each).
Second run (every fix above, benchmark after "engram tables in RAM", 933 s after start): 1 / 2 / 4 / 8 users ->
28.1 / 44.8 / 82.1 / 142.6 tok/s aggregate (17.8 per user at 8); TTFT median 0.27 / 0.47 / 0.94 / 1.85 s; chat checks
decode 28.5 tok/s, TTFT 0.24 s; 8.8K-token prompt TTFT 12.84 -> 3.95 s; zero trace releases over the session.

Serving: `NSEQS=8 CTX=12288 serve/serve_hy4.sh` (SPEC=4 by default: drafts while one request decodes alone; the script mounts
moe_sorted_dispatch.cpp over the image's, which predates it). Budget: users x max_model_len <= 98304
(HY4_SLOT_TOKENS_MAX): 8 x 16K runs out of DRAM on a long prefill after warm-up (0.32 GiB free), 8 x 12K leaves
0.91 GiB (a 12K prompt with 7 live users: TTFT 4.6 s, then 61 ms steps). serve/hy4_conc.py: N concurrent streams.

Correctness bug fixed after the served runs: prefill chunks past 2048 tokens build their index lists on the device
(indexer._expand: group ids -> token rows) without the slot's cache base, so a long prompt in any slot > 0 attended to
slot 0's KV (served needle check: one wrong token in every multi-user run; single-user / slot 0 and every gate were
unaffected). Fixed: Hy4Indexer.cache_base (set by Hy4Attention.set_slot) added in _expand; long3k prefill gate on
slot 3 (teacher_force --slot 3) = slot-0 baseline exactly (top-1 0.8269 / KL 0.2933); long3k decode on slot 3 = baseline
(0.8273 / 0.2930). Second cause of the needle failure: single-path decode score buckets were powers of two that fit
the slot (12K slot: up to 8192 tokens), so decode at 8846 tokens raised "past the largest indexer bucket" and vLLM ended
the request after its first token (a 32K single-user slot fails the same way past 16K). Fixed: a last bucket at the
slot's full group capacity (tt/decode.py), and the indexer iota padded 32 columns past it (a full-width ttnn.slice
aliased the iota, which _iota_rows then deallocated). Engine-level needle (8845-token prompt, Engram on): slot 0
alone, slot 2 alone, slot 2 with 7 live users give the same full reply (the model declines to repeat the code: model
behaviour, not yet compared with the CPU reference). The new bucket is not in the warm-up yet (one serve-time compile).

Known gaps / next:
- Engram loads into RAM in the background (~930 s); until then long prompts read the NVMe (8.8K-token TTFT 12.6 s).
- Burst TTFT: simultaneous short prompts now share one packed prefill pass (Hy4Model.forward_packed: prompts on the
  flash attention's batch dimension, B x Tb an existing bucket; HY4_PF_PACK=1): 2 / 4 / 8 prompts 0.49 / 1.03 / 2.05 s ->
  0.35 / 0.67 / 1.10 s; 8-user gate with packed prefill top1 0.8625 / KL 0.2317. The rest is device time per row.
- Serial prefill is itself sensitive to the bucket a prompt is padded to: one layer (3) gives last-row logits differing
  by 1.44 for the same 77-token prompt at 128 vs 256 rows (others ~0.09). Not investigated; packed prefill inherits it.
- Served step ~10 ms above the engine step (vLLM per-step host work); the plugin's async mode needs the sampled token
  and Engram rows on device to pipeline (host sampling + host Engram block it).
- Device: the batched pass costs ~+2.1 ms / 6 layers over a single-path 8-row pass: indexer score (fixed for dense
  passes), lane one-hot window matmuls (+810 us / 6 layers: a per-lane ring window would drop the 512x512 keep maps).

## Optimization round 2 (2026-11-19, afternoon): prefill, serving fixes, long-context correctness

Served (vLLM, QB2, single user, greedy, RAM Engram, k=4): decode 42.5 -> 48.5 tok/s mean (35-74); short TTFT 0.30 s;
first 6.6K-token prompt TTFT 11.6 -> 4.4 s (3.7 s warm); long-context decode 12.7 -> 36-40 tok/s; the same long
prompt now answers identically request to request.

| prefill TTFT (warm, no Engram) | morning | now |
|---|---|---|
| 128 / 512 / 1K tokens | 0.29 / 0.50 / 0.81 s | 0.23 / 0.36 / 0.52 s |
| 2K / 4K / 8K | 1.43 / 3.05 / 6.32 s | 0.86 / 1.96 / 4.18 s |
| 16K / 32K | 12.9 / 26.1 s | 8.65 / 17.7 s |

Quality vs the CPU reference: decode-path gate top-1 0.862, KL 0.231; long3k prefill top-1 0.797 -> 0.827, KL 0.490 ->
0.330; long3k decode path top-1 0.804 -> 0.827, KL 0.442 -> 0.296.

Long-context correctness bug (fixed, aad2085f288): tt/indexer._topk masked the row-major indexer_score_dsa output with
ttnn.where against a tiled condition; that mixed-layout where wrote wrong values (1.0 everywhere in isolation; in the
model, rows of the masked scores left unwritten, holding an earlier allocation's data). Every selecting chunk / decode
step past 2048 tokens picked part of its top-512 groups from garbage, differently per request. Scores are tilized
first now. Also: the group cache is zeroed per request (keys a previous request left past the context leaked into
the visible scores: indexer_score_dsa's visible outputs depend on the keys it reads past the causal range).

Changes: router w = gather(s, ids) (no [T,16,512] fp32 one-hot); explicit 2D-multicast matmul configs (common.mm);
HiFi2 for bfp8 / LoFi for bfp4 expert prefill matmuls; flash SDPA (sinks, sliding window, banded mask + block map)
for the first chunk and later SWA chunks; nlp head split / concat; cached dummy group mask for the grouped sparse
matmul; moe_sorted_dispatch kernel without volatile; streamed Engram projections; fused indexer pool; DFly2 k=4;
warm-up covers later chunks and long-context decode; drafter row slices precompiled.

Tried and dropped: 5-row fabric boundary for verify (-1.4 ms per verify pass, but its L1 reservation made prefill's
CBs clash while serving); fused HC on 32 cores (slower); lower fidelity for the HC mix / sparse SDPA (no time gain).

Open: fused HC ~105 us per layer (POST math-thread bound); attention glue ~80 small ops per layer; sparse-layer
sparse_sdpa in later prefill chunks (~12 ms per layer per 2K chunk: per-token key gathers); package rebuild + publish
(the ttnn moe_sorted_dispatch kernel change needs the image rebuilt).

## Optimization round 2026-11-19 (decode 8.5 -> 42.5 tok/s served, quality at parity)

Measured on the box (QB2, 4 dies), single user, chat prompts, served through vLLM 0.26 (serve/serve_hy4.sh, dev image
with this tree's models/ mounted), greedy, RAM-resident Engram:

| metric | before (d27ad1b) | after |
|---|---|---|
| model output | garbage (every layer PCC ~0.3 vs the reference) | top-1 0.863 / KL 0.231 (decode-path gate, 8 seqs) |
| plain decode step (full model, traced) | 117.9 ms (8.5 tok/s) | 45.2 ms (22.1 tok/s) |
| device time per step | 76.4 ms | 40.7 ms |
| served decode, DFly2 k=5 | n/a (drafter broken) | 42.5 tok/s mean (30-62), 2.4 tokens / verify pass |
| TTFT, short chat | ~0.4 s | 0.25 s |

Correctness bug (fixed, 75652f2a73b): commit 62d4122 switched the HC stream sums to fast_reduce_nc; on the [1, hc, T, 1]
row sums of squares it returned wrong sums, so every layer's hyper-connection weights were wrong and the pushed branch
generated garbage. The DFly2 "traced drafts accept 1.00" was this bug (the target was garbage): with it fixed the traced
drafter accepts 2.4-3.2 tokens per pass and matches eager drafts.

What changed (decode; prefill paths unchanged unless noted):
- Host: Engram tables in RAM (anon copy loaded in the background, page-cache pages dropped per chunk, throttled on
  memory pressure; this host's page cache was dropped repeatedly) 27.6 -> 0.5 ms per token; only the real logits rows
  leave the device 4.2 -> 0.9 ms.
- Streamed decode linears (qwen36 decode_linear, per-bank column streams, 32-row whole-tile mode + fp32 out added):
  fused [q_a | kv_a | indexer key | head weights], gate, q_b, o_proj, indexer query, [latent down | shared g | u],
  [latent up ; shared down], dense MLP, LM head (1255 -> 371 us), router as bf16 hi + lo. bfp8 streams at HiFi2.
- Width-sharded RMSNorm for decode rows (60 -> 10 us at 3072).
- Fused hyper-connection program (tt/hc_fused): previous HC post + next HC pre in one program, 16 cores exchanging
  partials, FP32 weight math, exact hi / lo selections: 176 -> 58 us per sublayer boundary.
- MoE: the qwen36 fused expert kernels at top-16 (ROUTE_WIDE 128-byte route records, single and multi-row), the
  "keys" router (sigmoid + bias on the Tensix, integer top-k on the data-movement cores).
- Fabric boundary (all-reduce + norm in one fabric op) for one-row passes (HY4_BOUNDARY_ROWS=1: its sharded receive
  buffers reserve L1 on every core; 8 rows crowded the streamed linears' CBs off L1).
- DFly2: drafter on streamed linears (13.7 -> 8.0 ms per draft pass), verify rows on the multi-row fused MoE (6 rows
  1.16x a 1-row pass), speculative decoding in the engine (greedy + exact speculative sampling), vLLM adapter returns
  1..k+1 tokens per step (M2.7 shim).
- Trace guard (tt/decode.py _guard_programs): programs compiled after a capture can sit in a trace's freed
  intermediates and get overwritten by replays (hang); traces are released and re-captured whenever the program cache
  grows. Prefill chunks are padded to buckets (128..2048) compiled at warm-up so short prompts compile nothing at serve.
- Serving: vLLM 0.26 needed the plain architecture registered (serving/hy4_vllm_stub.py, a model-interface stub) and a
  flat rope_parameters view without mrope keys (else the plugin takes its M-RoPE prefill path).
- tt-metal main API drift fixed in the qwen36 kernels (kernel_profiler.hpp path, custom_mm layout enum, rsqrt template,
  setup_fabric_connection(mesh_device=...)).

Tools: tools/decode_bench.py (traced A/B with saved logits, host cProfile), tools/teacher_force.py --decode-from (the
decode-path quality gate), tools/gold_probe.py (per-layer streams vs the reference), tools/bench_{dsh,hc,norm,sdpa}.py,
serve/hy4_client.py (TTFT / tok/s / tool call / needle). Quality gate recipe: ~/m27-work/hy4/run_gate.sh TAG [ENV=..]
then compare_hy4.py over ~/m27-work/hy4/dev_dec.

Open (next levers): fused HC 58 us x 2 per layer (SFPU slot reductions), attention glue (~150 us per layer: reshapes,
wuk / wuv batched matmuls, sparse_sdpa over 32 padded rows), Engram layers (~250 us each), boundary for verify rows,
prompts > 2048 tokens compile per-chunk programs (indexer score width grows with the chunk position) -> re-captures,
verify-row numerics differ from 1-row numerics (spec output != plain greedy; both valid), package rebuild + publish.

## Status for takeover (2026-11-18, end of day 1)

Branch `raahem/hy4-flash-hybrid` (rnabeelTT/tt-metal), worktree `~/hy4-flash`, host build (`build_metal_hy4.sh`, CPM
deps from tarballs in ~/m27-work/cpm_src), env `~/m27-work/hy4/run.sh <cmd>` (python_env + the PR-50903 transformers
overlay in ~/m27-work/hy4/pyoverlay; takes the device lock ~/.cache/hybrid-moe/ttdev.lock). Ornith-397B on :20000 is
stopped (it held the board). Weights: tencent/Hy4-flash rev d88c9df in the HF cache; tensor cache
~/.cache/hybrid-moe/hy4_tt (bfp4 experts), Engram FP8 store ~/.cache/hybrid-moe/hy4_engram.

Working (measured on the box, single user, greedy):
- Full 42-layer model on the 4 dies, every expert resident (bfp4); DRAM 24.3 GiB used / 7.6 free per die at 32K.
- Quality vs the CPU BF16 reference (teacher forced, 12 texts): top-1 0.845, confident 0.917, KL 0.266 (Inkling
  0.873 / 0.951 / 0.200; M2.7 0.848 / 0.933 / 0.196); long3k (indexer active) 0.784 / 0.884 / KL 0.57.
- Decode: traced (tt/decode.py Hy4Decoder), exact vs the eager path; ~84 ms device + ~6 ms host per token, ~10-11
  tok/s. Profile: op-latency bound (~200 small ops per layer), not bandwidth (floor ~8 ms/token).
- TTFT (prefill branch, merged): 200 tok 0.38 s, 1K 0.94 s, 2K 1.68 s (was ~1 s / ~6 s / 82 s at 3K before).
  See "## Prefill" below.

In progress / broken:
- DFly2 drafter on device (tt/dfly.py, Hy4SpecDecoder, tools/spec_bench.py): output always equals plain greedy.
  With eager drafts acceptance is 2.5-3.2 tok/verify pass (12.5 tok/s on one prompt); once draft passes replay as
  traces acceptance drops to 1.00 even though a replayed draft equals the eager draft for the same inputs
  (HY4_DFLY_CHECK=1). Last change (persistent trace outputs, commit 62d4122) made it 1.00 everywhere: suspect the
  draft reads stale target state (model.last_hn / indexer.sel / scratch rows) relative to the verify trace ordering.
  Next step: HY4_SPEC_DEBUG=1 spec_bench prints drafts vs target picks per pass.
- 6-row verify passes: rows 4-5 differ slightly from 1-row passes (PCC 0.99999 per layer, compounds to ~0.99).
- Fused MoE decode kernels (qwen36 moe_decode) support Hy4 shapes but cap top-k at 15 (Hy4 = 16): off; SWIGLU_LIMIT
  clamp define added to the kernels (gated by MOE_SWIGLU_LIMIT).
- Engram device part in L1 drifts (PCC 0.993): kept in DRAM.
- The prefill merge (d39004b) was not re-tested on device after merging.
- Not done: vLLM serving test (adapter tt/generator_vllm.py + serving/ parsers untested end to end), package not
  published (a local image was built, nothing pushed), B > 1, vision, 1M context (KV budget: bf16 latent ~15 KB/token/die).

Biggest perf levers left: (1) fix traced DFly2 (2.5-3x tokens/pass); (2) fuse the HC math / shard the residual like
qwen38_flash_next (its gr_* kernels); (3) DRAM-sharded decode matmuls; (4) a fused MoE decode for top-16.

## Decisions (2026-11-18, approved)

- Day 0: text only, one user, no drafter (SPEC=0); DFly2 is the fast follow-up. Every device op is written for B rows
  so B = 2..4 users can be turned on once they pass the same quality gate. Vision later.
- Context: the maximum (1,048,576) as long as short sessions lose nothing (speed or quality).
- Quality is judged against the CPU BF16 reference (teacher forced: top-1, KL), never against an earlier TT run.

## Why this is not a hybrid-expert model

41 MoE layers x 512 routed experts, each in a 1536-wide latent space: gate/up `[3072, 1536]` (2 x 1536 out, 1536 in)
and down `[1536, 1536]` = 7.08M weights. All 20,992 experts in bfp4 (0.5625 B/w) = 83.6 GB = 77.8 GiB: they ALL fit on
the 4 dies with the rest of the model (~8 GiB at 8-bit) and the KV cache (MLA latent 576 + indexer key 128 per token on
the 12 sparse layers: ~9 KB/token, 1.2 GB at 128K, ~9 GB at 1M; the 30 SWA layers keep 512 positions). ~25 GiB is left
for higher-precision experts (e.g. bfp8 downs) and buffers. So: no CPU experts, no per-layer host round trip; decode
can be one trace per step.

The host holds only the Engram tables (layers 1, 11, 21, 31; 16 hash heads x 96 dims each; 165M rows per layer;
63.4B params, 118 GB in BF16). The hash ids depend only on token ids (`HYV4FlashEngramLayer.hash_ngrams`), so the
host computes them and the per-head gathered rows ahead of each pass and uploads `[T, 16 x 96]` per Engram layer
(~3 KB per token per layer). The model reads the rows through a per-vector FP8 e4m3 round trip
(`_fp8_round_trip`, scale = amax / 448 per 96-wide head vector), so storing each row as FP8 + its bf16 scale is
lossless: ~62 GB of host RAM. The rest of the Engram layer (per-head RMS norms, value / key projections, the
per-stream gates) runs on the device.

## Architecture notes (from the reference code)

- Residual: 4 hyper-connection streams of 3072 (`hc_mult` 4). Each sublayer: `HyperConnection` (fp32): `mixes =
  (flat @ fn.T) * rsqrt(mean(flat^2) + eps)` over the flattened 12288; `pre = sigmoid(mix_pre * s0 + b_pre) + hc_eps`,
  `post = hc_magnitude(2.0) * sigmoid(mix_post * s1 + b_post) + hc_eps`; sublayer input = sum_i pre_i * stream_i;
  output streams = post_i * sublayer_out + stream_i. Final: `HyperHead` (4 mixes, sigmoid + eps, weighted sum), then
  `model.norm`, LM head (config asks for an fp32 LM head).
- Layer 0 dense MLP (13824) with `post_fc2_norm`; layers 1-41 MoE: sigmoid router + `e_score_correction_bias`, top-16
  of 512 (n_group 1), weights renormalised then x 3.998; experts get `routed_expert_down_proj(x)` (3072 -> 1536), SwiGLU
  with clamps (gate <= 10, |up| <= 10), outputs summed then `routed_expert_up_proj` (1536 -> 3072); plus one shared
  expert (3072 -> 1536 -> 3072, no clamp) on the full-width input; `combine_output_norm` closes the block.
- Attention: gated MLA, q_lora 1024, kv_lora 512, qk 192 nope + 64 rope, v 256; `gate_proj` (sigmoid, per head x 256)
  multiplies the attention output; learned sinks per head; `post_proj_norm` after `o_proj`. Sparse layers (0, 1, 5, 9,
  ..., 41: 12 of them): 32 heads, rope theta 1e7, compressed indexer. SWA layers (30): 64 heads, theta 1e4, window 512.
  RoPE is GLM-style interleaved on the last 64 dims; for text the 3 MRoPE sections carry the same position, so it is
  plain 1-D RoPE.
- Compressed indexer (sparse layers): token key = RMSNorm(W_tk x) (128); groups of 4 tokens pooled with an fp32 gated
  softmax (+ `intra_group_ape`), RMSNorm; group keys carry the RoPE phase of their first token. Query = W_q q_resid
  (16 heads x 128, RoPE on the last 64). score(g) = sum_h w_h * relu(q_h . k_g), w = W_hw x * 16^-0.5 * 128^-0.5.
  A query keeps the top 511 complete earlier groups + its own group (2048 tokens). Dense below 2048 total tokens
  (`kv_length <= candidate_topk`): short sessions never run the indexer.
- Engram: added to the 4 streams before the attention HC of layers 1, 11, 21, 31 (`engram_layer_ids` 2, 12, 22, 32
  are 1-based).

## Base and device port (2026-11-18)

- The branch merges upstream main (0fe57024bca) into the starter: it brings the upstream Qwen3.8-Flash-Next port
  (`models/demos/blackhole/qwen38_flash_next`: QSA = the same 4-token-block / 2048-token sparse attention, a 4-stream
  gated residual, an n-gram table read by the host, 512-expert BF4 MoE via `moe_compute`, fused decode kernels) and the
  DeepSeek-V4 pieces (`sparse_sdpa`, `indexer_score_dsa`, `topk_large_indices`, `update_padded_kv_cache`). Conflicts:
  kda / tracy took upstream, qwen36 kept the starter's copy (the M2.7 port uses it), sparse matmul keeps both the
  grouped MoE factory and upstream's prefetcher factory. Host build: `build_metal_hy4.sh` (untracked copy of
  build_metal.sh that points CPM at tarballs in ~/m27-work/cpm_src: GitHub git clones ran at ~70 KB/s here).
- Port `models/autoports/tencent_hy4_flash/` (day 0 = eager, one user, no drafter):
  `tt/config.py`, `tt/weights.py` (lazy safetensors), `tt/common.py` (uploads, all-reduce, RMSNorm),
  `tt/hc.py` (hyper-connections, fp32), `tt/rope.py` (bf16 tables exactly as HF, fp32 rotation by a +-1 matmul),
  `tt/attention.py` (absorbed MLA; row-major 608-wide latent cache [c 512 | k_rope 64 | sink 32] written with
  `slice_write`; every attention goes through `sparse_sdpa` with host-built index lists; the learned sink is a
  reserved cache row scored against q's two sink columns s_hi + s_lo), `tt/indexer.py` (token keys read back, groups
  pooled on the host in fp32 exactly as the reference, mirrored to a device cache; device scoring + top-512),
  `tt/moe.py` (latent MoE via sparse_matmul gather / chunk-masked paths, clamped SwiGLU, TP over the expert
  intermediate; dense MLP), `tt/engram.py` (device half), `tt/model.py`, `tt/engine.py`, `tt/generator_vllm.py`,
  `tools/generate.py`, `tests/test_primitives_device.py`, `tests/test_layer_device.py`.
- Layout: residual streams replicated on all 4 dies (the HC math is then local); one all-reduce per sublayer.
- Next after correctness: traces (position-dependent inputs from the host as one-hot / index tensors), fused HC
  (qwen38's gr_* kernels), device-side index lists, the DFly2 drafter, B = 2..4.

## Prefill (2026-11-18, branch raahem/hy4-prefill)

Borrowed from M2.7 (expert-sorted MoE) and tt-metal PR 61548 (rows split over chips, larger chunks):
- `tt/prefill_moe.py` SortedExperts: moe_sorted_dispatch + grouped sparse matmuls (padding groups skipped) + gather /
  gather-add rows, Hy4's clamped SwiGLU in the 1536 latent. MoE 327 -> 17 ms per layer at 2048 rows; sub-chunks of
  2048 rows (HY4_MOE_SUB: the dispatch admits ~3K rows at top-16 of 512).
- Row split (HY4_ROW_SPLIT_MIN=128): `Hy4Layer.forward_split` keeps each die's quarter of the rows for the HC / norms /
  Engram; sublayer inputs all-gathered, partials reduce-scattered (`Hy4Attention.forward_part`, `Hy4MoE.forward_part`,
  `Hy4DenseMLP.forward_part`). Prefill row counts are multiples of 128.
- Index lists: position-only lists (SWA, dense) uploaded once per chunk and shared by the layers
  (`Hy4Attention._idx_device`); selecting chunks (p0 >= 2048) build them on the device from the top-512
  (`Hy4Indexer._expand` + the per-chunk tail `_idx_tail`): no readback.
- Group pooling on the device for prefill (`Hy4Indexer._pool_prefill_device`, HY4_PF_DEVICE_POOL=1; decode's fp32
  math): no token-key readback. The host views decode's `init_stage` reads (`mirror`, `tk_hist`, `open_keys`) are
  properties now, read back lazily from the device.
- `Hy4Model.prefill(ids)`: the dense-regime chunk (to 2048) first, then HY4_PREFILL_CHUNK chunks (<= the model's
  `chunk`, which sizes the SWA rings): the engine should call it instead of looping forward() in fixed chunks, and build
  the model with chunk=8192.
- Head only on the rows asked for (`Hy4Model._head_split`, 32-row blocks): the full gather before the head OOMed
  (3 GiB) at 8K / 16K chunks.
- Traced first chunk (`Hy4Model._prefill_traced`, HY4_PF_TRACE=1, buckets 128..2048 at p0 = 0; needs
  HY4_TRACE_REGION ~1e9): last-row logits bit-equal to eager (PCC 1.0, same top-1 at 200 / 1000 / 2000) but no
  faster (0.389 vs 0.38 s, 0.94 vs 0.94, 1.68 vs 1.68): prefill is device-bound, so it is off by default. The time is
  data movement (repeat / slice / typecast / sum, the fp32 HC), not dispatch.
- Not touched (decode owner): `tt/hc.py` (in prefill `ttnn.sum(dim=1)` over the streams costs transpose + reduce +
  transpose on [1, 4, T/4, 3072] fp32; `ttnn.experimental.fast_reduce_nc(dims=[1])` is one op).

## Log

- 2026-11-18: plan approved; branch + worktree `~/hy4-flash`; BF16 download started (`~/m27-work/logs/hy4_download.log`).
- 2026-11-18 (serving worker): vLLM 0.26 front-end + DFly2 reference.
  - vLLM 0.26.0 / transformers 5.12.1 (image 2f9b96d1f881) know neither `hy_v4_flash` nor the `hy_v4` parsers.
    `models/autoports/tencent_hy4_flash/serving/` adds them: `sitecustomize.py` (post-import hooks in every process:
    config class into `AutoConfig`, `hy_v4` reasoning / tool parsers as lazy vLLM parsers; nothing imported
    eagerly; keeps the HYB_ENABLE overlay hook), `hy4_config.py`, `hy_v4_reasoning_parser.py` and
    `hy_v4_tool_parser.py` (vLLM PR 68296 @3f21671, imports adapted; the tool parser has no xgrammar structural tag
    on 0.26, so `tool_choice` required / named take vLLM's generic JSON-schema path; auto parses the native
    `<tool_calls:opensource>` tokens). Bundle: `vllm_bundle/hy4_flash/vllm_metadata.json`, arch
    `HYV4FlashForConditionalGeneration` (the checkpoint's own: no `--hf-overrides`), main class
    `models.autoports.tencent_hy4_flash.tt.generator_vllm:HYV4FlashForConditionalGeneration` (to be written: text
    only, `supports_multimodal = False`). vLLM derives max_model_len 1,048,576, vocab 120,832, eos 120025,
    sliding_window 512 (from text_config), head_size 64 (`head_dim` field; the engine owns the caches anyway),
    sampling defaults T 0.9 / top_p 1 from generation_config.json. Serve with `PYTHONPATH=<port>/serving:...`,
    `--reasoning-parser hy_v4 --tool-call-parser hy_v4 --enable-auto-tool-choice`.
  - Tests (CPU, in the image; the image has no pytest, mount a pure-python copy): `serving/tests/` 20 passed
    (registration, config, reasoning thinking / no_think, tool calls typed by schema, malformed -> content, streaming
    at 1 / 2-3 / 6 tokens per step equals non-streaming, chat-template structure).
  - Chat template: system turn always present, ends `<｜reasoning_mode:opensource｜>reasoning_effort:{high|max|no_think}`
    (anything else raises: OpenAI `reasoning_effort` low / medium requests would fail); generation prompt ends with
    `<think:opensource>` (thinking) or `<think:opensource></think:opensource>` (no_think); earlier assistant reasoning
    is dropped unless tools are present.
  - DFly2 (`models/experimental/hybrid_moe/dfly_hy4_ref.py`, tests `test_dfly_hy4_ref.py`: 10 passed incl. the real
    weights): one parallel pass of 6 rows [bonus, MASK x 5] at anchor..anchor+5 through 5 dense MLA layers (target
    sparse-layer attention: sinks, sigmoid gate, post-proj norm, theta 1e7) that read AND write the target's latent
    caches of layers 25/29/33/37/41 and attend to that layer's indexer recall for the last accepted row (+ the causal
    block); row 0 gets `mtp_fusion` with the target's final hidden; then 5 sequential `hidden_correction` + LM-head
    steps (greedy or temperature sampling). No prefill role. `confidence_head` unused (vLLM ignores it).
    Engine interface: per verify pass keep, for the last accepted row, the final normalized hidden and the recalled
    positions of layers 25/29/33/37/41 (<= 2048 each; all positions < anchor below 2048 tokens); the draft's block
    latents may go to a 6-row scratch instead of the target cache (equivalent: the next verify overwrites them).

### 2026-11-18: CPU BF16 reference + Engram host store (reference worker)

- venv `~/m27-work/venv-hy4` (torch 2.15 cpu, transformers 5.24.0.dev0 = PR 50903 @c2ec850).
- `engram_store.py`: `convert` writes the 4 tables as FP8 e4m3 rows + a bf16 scale per row to
  `~/.cache/hybrid-moe/hy4_engram/` (`L<li>_fp8.npy` 15.8 GB + `L<li>_scale.npy` 0.33 GB each, 64.7 GB total; ~1-4 min per
  table). `EngramRows(store_dir, ckpt_dir).rows(li, ids, p0)` returns the Engram layer's `embeddings` after the per-head
  norms and the flatten (bf16 [T, 1536], the value_proj / key_projs input); keeps the last 2 tokens per sequence itself.
  `python engram_store.py test`: hash ids, n-gram tokens (incl. incremental), rows after the fp8 round trip AND the
  normed rows bit-identical to the HF code on all 4 layers (4096 random tokens, 40 pads). Two traps found:
  (1) the reference returns `x + (q - x)`, which is +0 where q is -0: dequant adds +0.0; (2) `nn.RMSNorm(96, eps=None)`
  on bf16 input computes in fp32 with eps = finfo(**fp32**).eps = 1.19e-7 (not bf16's 7.8e-3) and applies the weight
  in fp32 before one rounding. Lookup: 0.4 ms (1 token, 4 layers, warm), EngramRows 0.9 ms (1 token) / 1.3 ms (3);
  8K tokens 0.29 s warm. Engram mask: only pad (120002) / image / video tokens zero the WHOLE Engram output.
- `ref_hy4.py`: runs HF's own `HYV4FlashTextModel.forward` (masks, indexer, caches) with the non-expert weights in RAM
  (14.8 GB anon) and the experts / Engram tables memory-mapped from the shards (page cache). `tf` (teacher forced,
  chunked through the HF cache, REF_GOLD streams) and `gen` (greedy). Attention: REF_ATTN=fp32 (default: HF's eager
  math with fp32 q/k/v; HF's own softmax runs in bf16) | hf. Router fp32 as HF; LM head fp32 as HF.
- `compare_hy4.py`: device `lp_<tag>_<name>.npy` [T, V] or `dev_<tag>_<name>.npz` (top1, lp_at_ref) vs ref_<name>.npz.
- Reference outputs `~/m27-work/hy4/ref/`: texts.json (8 held-out 1000-token texts, quality_texts.py with the Hy4
  tokenizer), prompts.json (4 chat prompts, reasoning_effort no_think), gen.json, tf_texts.json (the 8 texts + the 4
  prompts with their greedy continuations), ref_<name>.npz, routing_T<T>.npz, gold_<name>.pt (streams entering every
  layer [42, T, 4, 3072] + final hidden, for math_gsm8k and the 4 chat sequences).
- Greedy (no_think): "Hello! I'm Hunyuan, a large model developed by Tencent. ...", "72 km/h" (correct), a Fibonacci
  function, fluent Chinese on the Great Wall. Teacher forced on its own continuations: top-1 0.96-1.0, NLL 0.07-0.09.
  Held-out texts (nll / top-1): gsm8k 3.52/0.60, mbpp 5.04/0.47, en 5.55/0.37, zh 5.16/0.43, fr 4.40/0.49,
  py file 2.02/0.61, markdown 3.42/0.46, livecodebench 4.29/0.39: a very peaked model (top-1 p 0.84 on gsm8k), so the
  misses cost 10-19 nats each; top-1 / KL are the metrics to watch, not NLL.
- Timing (8 cores, warm page cache): 8 x 1000 tokens 407 s (9.7 s per layer); greedy 1.1-1.5 s/token warm (7.7 cold).
- Checks: layer 0 (indexed attention + compressed indexer) in float64, one 3000-token pass vs 1024-token chunks through
  the HF cache: bit-identical (0.0) incl. past 2048 where the indexer selects. Noise floor of the BF16 reference:
  fp32-vs-bf16 attention softmax + batch 1 vs 8: top-1 agreement 0.928 (gsm8k) / 0.96-1.0 (chat); full model single
  pass vs chunked over 3000 tokens: 0.97 / 0.91 / 0.86 (positions 0-1K / 1-2K / 2-3K). A TT port within ~0.9 top-1
  of this reference is at the reference's own noise level.
- Note: config.json spells the attention types `deepseek_sparse_attention` / `sliding_attention`; transformers
  remaps the first to `indexed_attention` (its rope_parameters key too). `mrope_interleaved` is ignored with a warning
  (the interleave is always applied); for text the 3 M-RoPE planes are equal, so it is plain 1-D RoPE.
