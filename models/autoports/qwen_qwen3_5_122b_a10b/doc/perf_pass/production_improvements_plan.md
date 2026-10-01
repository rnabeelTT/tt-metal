# Production improvements after the 2026-10-01 deployment (plan + measurements)

Baseline = the pushed bundle (Hub revision 668d1f0b; MTP k=3 adaptive, vision on, thinking off, 32 slots, 262144 ctx).
Numbers from the readiness suite, the deployed-model sweep (`artifacts/stage17/perf_sweep_hub.json`, pass 1) and the
measurements below. Re-measure after each change in the production configuration (readiness single-user benchmark,
the sweep rows that the change targets); the deferred sweep rows (single-user TPOT column, sampled rows) are re-run
once at the end on the improved model.

| metric (baseline) | value |
|---|---|
| 1 user, 128/128: TTFT / TPOT | 396 ms / 16.6 ms (60.3 t/s/u, MTP) |
| 2 / 4 / 8 / 16 / 32 users TPOT (short context) | 37 / 44 / 64 / 103 / 116-131 ms |
| 32 users, 128-token prompts: TTFT / total out | 7.7 s / 184 tok/s |
| 32 users, 4096-token prompts: TTFT | 39.7 s |
| 1 user, 130816-token prompt: TTFT | 48.5 s (2 700 tok/s) |
| 32 users x 32768 / 130816 tokens: TPOT | 425 / 512 ms |

## Measurements (2026-10-01, reduced model unless stated)

**Decode step by width** (traced batched step per layer kind; `artifacts/stage15/tracy_decode_*-oct1`): GDN layer
1.35 / 2.16 / 2.75 ms, attention layer 1.22 / 1.94 / 2.63 ms at 8 / 16 / 32 users; 36 x GDN + 12 x attn = 63 / 101 /
130 ms = the served TPOT, so the step is entirely layer device time. Per-op at 16 users (attention layer, 1676 us
device): the two MoE sparse matmuls 795 us (47 %), tilize-with-padding 272 us (16 %), 10 binary ops 185 us; the pairs
path reads one expert block per (user, expert) pair (128 reads at 16 users, ~50 % bandwidth efficiency) and spends
~20 % in relayout glue. At 8 users (GDN layer, 934 us): sparse matmuls 338 us (36 %), 15 binary ops 154 us.

**Served 128-token prefill, full model, production adapter** (`scripts/18-prefill-fixed-cost.py`): 238 ms per request =
193 ms logits readback that waits for the traced bucket (~17 ms embedding / head rows / norm / LM head + all-gather,
~3.7 ms x 48 layers: op-launch bound) + 28.8 ms GDN slot write (36 layers x (1 slice_write + 4 conv-tap rebuilds of
4 small ops)) + 2.6 ms snapshot + 4 ms resets + 0.8 ms staging. The plugin hands every waiting request to one
prefill_forward call and we run them one by one; a step is prefill-only or decode-only and the plugin scheduler
prefers prefill while any is pending (`vllm_tt_plugin/scheduler.py`), so 31 short prompts block decode for 7.4 s.
The plugin also disables chunked prefill for any model with `output_tokens_per_step > 1` (MTP).

## Plan (ordered)

1. **Decode MoE at 8-32 rows -> expert-sorted grouped path** (`QWEN35_DECODE_SORTED_MIN_ROWS`, knob added
   2026-10-01 19:02). A/B: `scripts/18-decode-sorted-ab.sh` (per-width step time + PCC). Target: 16-user TPOT
   103 -> ~80 ms, 32-user 130 -> ~110 ms; unblocks multi-user MTP re-measurement. Result (19:11): REJECTED as is - slower at every width (GDN 2.26 / 2.51 / 2.91 ms vs 1.32 / 2.13 / 2.72 at 8 / 16 / 32 users); the sorted dispatch's fixed cost dominates at <= 32 rows. Next: Tracy of the sorted path at 16 rows, then either a small-T variant (one-kernel dispatch + grouped matmul without the gather / un-dispatch round trips) or the pairs path's glue (tilize 16 %, slices) trimmed instead.
2. **MTP prefill tax**: the head's prompt rows run inside every bucket trace (they need the main model's hidden at
   every prompt position, which is not kept, so they cannot be deferred to MTP entry). Measure the tax exactly
   (`18-prefill-cost-mtpoff.sh`: the same 128-token split with MTP off) and, if it is more than the head layer's
   ~4 ms, trim the glue (norm clone, select matmul, layout changes, concat) or move the K/V write off the bucket.
3. **GDN slot write**: conv-tap writes through fewer ops (indexed row fill / stacked taps). Target: 28.8 -> ~10 ms per
   prefill (-8 % of the 128-token request).
4. **Batched short-prompt prefill**: several short prompts per bucket pass through the batched GDN chunk form (the
   MTP verify pass already runs B users x 32 rows) + batched prefill attention. Target: ~2.5x on short-prompt bursts.
   Largest change; after 1-3.
5. **Plugin asks**: prefill / decode alternation in `TTScheduler` (gpt-oss-style "prefill step, then D decode steps"),
   chunked prefill with block output (MTP), automatic prefix caching (our GDN carried-state primitive can snapshot
   the state at the prefix boundary). Not shippable from this tree (the bundle pins the plugin revision).
6. **Width-1 router chain fusion** (~5 ms of the 32 ms plain step); fused GDN decode kernel later.
7. **Long-context decode at high occupancy** (32768 x 32 users 425 ms): Tracy at 32 users x 32K first.
