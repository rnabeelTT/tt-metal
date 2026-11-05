# DFlash 2 speculative decoding for Ornith-1.5-122B (branch `ornith122_dflash`)

Goal: higher single-user decode throughput on p300x2 with unchanged output quality, using the public DFlash 2 drafter
`incoai/Ornith-1.5-122B-DFlash2` (6 layers, hidden 3072, block 16, 8 target context layers [1, 7, 14, 20, 26, 32, 39, 45],
non-causal sliding-window attention 2048, two-tap dynamic grouped convs, top-16 candidate selector; reference math:
github.com/z-lab/dflash `dflash/model.py` `DFlash2DraftModel`).

## Design (as built)

One round, one user (`tt/dflash_traced.py`, block B = 8 by default):

1. **Draft** (device, ~5.4 ms): the drafter over [last token, mask x 15] (32 tile rows, positions start..), tensor
   parallel over the 4 dies (q / kv heads and MLP columns split, o / down projections reduce-scattered + gathered), its
   context K / V in per-layer rings of 2272 slots (sliding window 2048), flash SDPA with a host-written visibility row,
   the target's LM head; per-shard top-32 candidates + the selector projection read back; the selector path (7
   sequential first-order choices over top-16 candidates, codebooks on the host) gives the B - 1 drafts.
2. **Verify** (device, ~21-27 ms at B = 8): the model's own decode layers over the B rows of the one user (GDN layers in
   the decode ops' rows mode, attention with per-row paged K/V writes, the fused multi-row MoE), the 8 context layers'
   rows captured for the drafter, argmax (greedy) or per-shard top-k candidates (sampled) read back.
3. **Accept** (host): greedy = longest prefix of drafts equal to the argmax; sampled = exact speculative sampling
   (`SpecSampler.accept`, the selector's distribution as q): the emitted stream has the target's distribution.
4. **Commit**: none - the next verify starts from the accepted row's GDN snapshot / conv history (a 4-byte host write);
   the bound state is materialized only when the request leaves the DFlash phase.
5. **Drafter context** (~1.2 ms): the accepted rows' context features -> the drafter's K / V rings.

Serving (`tt/dflash_serving.py`, `QWEN35_DFLASH=1`): the adaptive MTP plumbing - a lone request decodes in DFlash
rounds, any multi-user step runs the plain batched decode (the rings are not advanced there). Prefill builds the
drafter's context inside the traced bucket programs (layer hook + ctx program). The guard: after 6 rounds a request that
produced fewer tokens than the plain decode would have in the same time goes to the plain decode for good.

## Pieces

| file | what |
|---|---|
| `tt/dflash_ref.py` | torch reference of the drafter (host): context K/V, block hidden, selector path |
| `tt/dflash_device.py` | the drafter on the device (replicated bfp8 weights, K/V rings of 2112 slots, fixed 32-row programs) |
| `tt/dflash_traced.py` | one-user traced round: MTP verify pass (rows mode, + context-layer capture) / commit, drafter ctx / draft |
| `tt/dflash_serving.py` | serving: the adaptive MTP plumbing (lone request -> DFlash phase), prefill context in the bucket traces, acceptance guard |
| `tools/dflash_acceptance.py` | eager acceptance on the device target with the host drafter (+ greedy agreement) |
| `tools/dflash_traced_run.py` | the traced device loop: tokens per round, ms per token, per-program walls |

Env: `QWEN35_DFLASH=1` (needs `QWEN35_MTP=0`; off with chunked prefill), `QWEN35_DFLASH_BLOCK` (8),
`QWEN35_DFLASH_GUARD_ROUNDS` (6), `QWEN35_DFLASH_PLAIN_MS` (7.6: the plain decode's ms / token the guard compares
with), `QWEN35_DFLASH_MARGIN` (1.0), `QWEN35_DFLASH_MIN_TAU` (1.5), `QWEN35_DFLASH_FUSED` (1: the fused verify),
`QWEN35_DFLASH_DRAFTER` (a directory or repo id; default: the bundle's auxiliary weights entry).
Bundle: `tt-model-ornith122.yaml` profile `p300x2-dflash` (opt-in), drafter as `auxiliary_weights` (pinned).

How to run: `tools/dflash_traced_run.py --bs 8` (device loop vs plain greedy, per-trace timings),
`tools/dflash_acceptance.py` (host drafter, eager target), `tests/test_gdn_rows_ops.py` (rows mode + base / history,
bit-exact vs row-by-row), `tools/dflash_drafter_prof.py` / `tools/dflash_pass_prof.py` (profiles); served A/B:
`~/ornith-fusion/runs/dfl_ab.sh` (TAG=..., ARMS="on off"), evals `~/ornith-fusion/runs/dfl_evals.sh`.

## Measurements

### Acceptance (eager, greedy, thinking on, 96 new tokens; `runs/dflash/acc1.log`)

Device target (default env), host drafter (`tools/dflash_acceptance.py`), block 16:

| prompt | tokens / round | equal to plain greedy |
|---|---|---|
| code (LRU cache) | 6.25 | first 42 (then a near-tie rewording) |
| math (quadratic) | 8.08 | first 34 |
| prose (product description) | 4.13 | first 67 |
| mean | 6.15 | |

The divergences are bf16 near-ties between the 16-row verify pass and the one-row decode (different kernels: GDN prefill
form, multi-row MoE); both continuations read correctly (e.g. "To achieve O(1) for both..." vs "The standard approach is...").
The drafter is therefore correct (a broken drafter accepts ~0 drafts) and its published acceptance (MT-Bench 5.4, code
6.4-8.6 on SGLang) holds on this target.

### First traced device loop (MTP rows verify pass, replicated drafter; `runs/dflash/tr1..tr3`)

Greedy, thinking on, 128 new tokens, p300x2. Plain decode (served card) is 7.55 ms / token.

| block | mean tokens / round | round | ms / token |
|---|---|---|---|
| 16 | 4.53 | ~80 ms | 18.0 |
| 12 | 4.02 | ~75 ms | 20.6 |
| 8 | 4.26 | ~68 ms | 17.7 |

Round at block 16 (each trace replayed alone): verify pass 34.8 ms (47 ms in the loop: the 16 different drafts route to
a union of ~60-70 experts per layer, the warm-up rows to 8), GDN commit 14.1, drafter 12.0, drafter context 2.8,
candidate top-k 0.9. Per layer (traced, rows mode): GDN layer 640-650 us at 1 or 16 rows (the prefill-form chunk scan
is row independent), attention layer 239 us (1 row) / 526 us (16 rows: per-row K/V writes), MoE 115-130 us (identical
rows). A fused decode GDN layer is ~211 us. Conclusions: the verify must run on the fused decode ops; the commit's scan
replay must go; the drafter is op-overhead bound (1.9 ms per layer for ~0.35 ms of weight traffic); block 8 keeps most
of block 16's acceptance at a lower verify cost.

### GDN decode ops, rows mode (`gdn_decode/ops.py` + kernels, `tests/test_gdn_rows_ops.py`)

`GdnDecode(..., snap=)`: the B rows are consecutive positions of ONE user. Conv: each row's three previous inputs come
from the carried rows / earlier rows of the tile (`ROWS` conv reader), register untouched. Head: one core per lane
(value head x part) runs the lane's rows in order; row r reads the state the previous row wrote (snapshot r - 1, local
semaphore from the writer) and writes its own state to snapshot r; the bound state is untouched. Op test vs the same ops
run row by row (state / register in place): **bit-exact** (max error 0) for R = 1 / 4 / 16, every output row and every
snapshot. The verify pass therefore reproduces the plain decode's GDN numerics exactly, and the commit is a copy of the
accepted row's snapshot (+ the conv register from the pass's in-proj rows).
Lesson: `ttnn.generic_op` returns its LAST io tensor - the output must stay last (a snapshot appended after it came back
as "the output").

### Drafter layer (`tools/dflash_drafter_prof.py`, no target model)

1.92 -> 1.43 ms per layer: `nlp_create_qkv_heads` + `rotary_embedding` (heads / RoPE 209 us -> ~60), flash SDPA with a
mask over the ring (ring 2272 slots so ring + block = 2304 = 9 x 256; attention core 579 us -> ~150), dynamic-conv
expansion by matmul (191 us per conv; `repeat_interleave` measured slower, 278 us). Remaining: ~50 small replicated ops
per layer at ~10-15 us each.

### Fused verify + tensor-parallel drafter (`runs/dflash/tr5..tr7`)

Verify on the decode layers (GDN rows mode, snapshot commit), traced, greedy, 128 new tokens, shipped decode flags.
Each trace replayed alone (block 16 / 8): pass 29.5 / 19.0 ms, commit 2.4, drafter ctx 1.2, draft 5.4 (TP over the 4
dies; was 12.0 replicated), candidate top-k 0.9. The fused verify also raised acceptance (code 5.6 -> 7.3 tokens per
round at block 16): its GDN numerics equal the plain decode's, which is what the drafter was trained against.

ms per emitted token (plain decode 7.55):

| prompt | block 16 | block 8 |
|---|---|---|
| code | 7.10 (tau 7.26) | **6.80** (5.38) |
| math | **6.96** (7.78) | 7.02 (5.86) |
| prose | 11.63 (4.57) | 9.38 (4.10) |
| chat | 16.42 (3.17) | 12.70 (3.02) |

Block 8 is the default (`QWEN35_DFLASH_BLOCK`). Per layer, fused rows path, traced (identical rows, so no MoE union):
GDN 242 / 355 / 487 us at 1 / 8 / 16 rows (~16 us per extra row: the head's rows run in order on each lane's core),
attention 226 / 407 / 515 us (~26 us per extra row: one paged K/V update per row - `paged_update_cache(share_cache=)`
would write them in one call but is rejected for paged caches by its C++ validation). With real drafts the 8 rows route
to a union of experts that adds ~7.5 ms per pass (the 256-expert MoE's DRAM reads: an inherent per-row cost here).
