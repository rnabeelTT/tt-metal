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

## Plan (ordered, revised 19:25)

1. **Multi-core TILE row-gather kernel** - DONE 2026-10-01 (`ttnn.moe_gather_rows`, commit 1dc31fd3d68): prefill
   MoE gather 1.80 -> 1.27 ms per layer at 2048 tokens (MoE ~8.3 -> ~7.7 ms, -29 ms per segment); the expert-sorted
   decode path -0.28 ms per layer step, now ahead of the pairs / masked path at 32 rows (-3 / -8 %), still behind at
   8 / 16. **Next half**: the down projection's output untilize (0.74 ms at 2048 tokens, 0.25 ms at 16 rows) ->
   `untilize_out` in the grouped sparse matmul factory (row-major output straight into the gather-add kernel).
   Targets after both: 16-user TPOT ~103 -> ~90 ms, 32-user 130 -> ~115, prefill MoE -1.3 ms per layer; then
   re-A/B `QWEN35_DECODE_SORTED_MIN_ROWS` (16 or 32) and re-measure multi-user MTP.
2. ~~MTP prefill tax~~ measured at 7 ms of 238 (closed).
3. **GDN slot write**: conv-tap writes through fewer ops. Target 28.8 -> ~10 ms per prefill.
4. **Per-layer op count of the 128-token bucket** (~3.5 ms x 48 = 170 of 231 ms): the prefill form of the GDN layer
   and the attention layer at bucket 128 have the decode problem (tens of tiny ops); fused norm+matmul and the
   router chain apply to prefill too. Measure with Tracy of the traced bucket 128 (the eager Tracy today was
   12 ms per layer and not representative).
5. **Batched short-prompt prefill** (several prompts per bucket pass; batched GDN chunk form + batched prefill
   attention). ~2.5x on short bursts; largest change.
6. **Plugin asks** (not shippable from this tree): scheduler prefill / decode alternation, chunked prefill with block
   output, automatic prefix caching (+ our GDN state snapshot at the prefix boundary).
7. Width-1 router chain fusion; long-context decode at high occupancy (Tracy at 32 users x 32K first).
