# Study: `sjettTT/qwen3.8-flash-next_p150x4` bundle (what carries over)

Read-only study of the bundle's code (image `tt-model/qwen3.8-flash-next:6e8c55ef7f2f`, model code under
`/opt/tt-metal/models/demos/blackhole/qwen38_flash_next`, ~54K lines; tt-metal fork `02e1453c2bcd`, not an ancestor of our
base). Qwen3.8-Flash-Next: 48 layers hybrid GDN + DSA-style sparse attention ("QSA"), 512 experts top-10, expert width 640,
one MTP layer, per-layer n-gram embeddings (PLE). Served by its own ASGI server (not vLLM), one interactive user, 4x p150.
Its numbers: decode 38.6 tok/s (1 user, greedy), 68 tok/s with `--mtp 4`; prefill 3,003 tok/s with 2048-row "slabs".

## MoE (not portable as code; the design confirms our direction)
- Expert parallel (128 experts per chip, no TP inside an expert), one custom fused op `ttnn.experimental.moe_compute`
  (a **forked** signature: `local_combine`, `zero_fill_non_owned_rows`, `prefill_rings` do not exist in our tt-metal) with
  bfp4 weights pre-packed in the op's ring layout. The counting sort lives inside the kernel. Combine through
  `deepseek_moe_fast_reduce_nc_fused` or a model-local `generic_op` (0.14 ms/layer at 2048 rows).
- Their expert stream per layer per 2048 rows: 1.84 ms (3 core rings), i.e. every expert's weights read **once**
  (354 MB per chip per layer at ~190 GB/s). Ours (grouped sparse matmul, 32-row groups, ~761 groups at 2048 tokens) reads
  the expert block once **per group**: ~1.0 GB per chip per layer -> 4.3 ms. Lever: process all groups of one expert
  against one pass over its weights (ragged M per expert); expected 4.3 -> ~1.5 ms/layer.
- Their combine is 0.14 ms vs our un-dispatch 2.1 ms (+0.73 ms untilize). Lever: a multi-core gather-add rows kernel.
- Decode: same DRAM-bound regime as ours (cost ~ distinct experts per chip, ~7-11 us each); nothing to borrow.

## MTP (design reusable, code must be rewritten; ~60% blueprint, ~20% near-copyable)
- k = 3/4/5 drafts; verify = one padded 32-row pass through all layers + the MTP layer on the R = k+1 rows
  ("alignment rows" fed with the target's own argmaxes, so d_1 of the next pass comes free); then a draft trace runs the MTP
  layer k-1 more single rows. Traces: `verify_first`, `commit`, `draft` (3 captures greedy; up to 8 with sampling forms).
- Greedy accept fully on device (`accept_rows`): eq(argmax, drafts) -> upper-triangular matmul running count -> prefix
  length `a` -> gather next token -> `P += a + 1`; one 268-byte readback per pass, next token row never crosses the host.
- No snapshot rollback: KV rows written by absolute position and overwritten; GDN recurrent/conv state committed lazily
  from `a` (per-row prefix states, pick slot `a`, or re-run the chunk kernel with beta/g zeroed past `a`).
- Sampled form: exact point-mass speculative sampling on the host from top-32 candidate lists (75-line pure-Python file,
  `ttnn/speculative_sampling.py`, directly portable); optional one-core device accept (k <= 4).
- MTP weights (theirs: hyper-connection mixers) differ from Qwen3.5's `mtp.pre_fc_norm_*`, single `mtp.fc [2H->H]`,
  `mtp.norm`; embedding and LM head shared in both.
- Never ran under vLLM (their server only). For us: vLLM tt plugin speculative-decoding contract + on-device sampling is
  new work. Reported gain 1.76x at 1 user (2.9 tokens/pass chat, 4.5-4.75 code/json).

## Prefill / GDN / CCL
- "Slab" = 2048 rows in one traced pass per layer, 2D-multicast matmuls, GDN as ONE `chunk_gated_delta_rule` call over
  the slab (chunk_size 32, fp32 state inside the kernel) plus a fused pre/post rows program (3.27 -> 0.45 ms/layer);
  no cross-chip segment parallelism, all on CQ0, host prepares slab k+1 during replay k. We already run 2048-token
  segments through a traced segment program; the GDN glue is the part to compare (see work_log turn 5).
- CCL: Linear topology, FABRIC_1D, plain `ttnn.all_gather` over `all_gather_async` at slab sizes (0.59 -> 0.23 ms);
  nothing named "Warp400" in code (README term for the p300 cross-card links). `qb2_p300_1x4_line` descriptor is for
  a 2x p300 host whose cluster type tt-metal would otherwise classify as CUSTOM; our p300x2 opens with the stock
  descriptor and needs nothing from it.
- Decode: on-device argmax into a persistent token row, non-blocking readback overlapped with the next trace, second
  CQ only for early reads; the HEAD/TAIL split exists only for their PLE host lookup.

## Actions taken from this study
1. Kernel-side dispatch (already in flight before the study): `ttnn.experimental.moe_sorted_dispatch`.
2. Next prefill levers, in order of value/effort: gather-add un-dispatch kernel; one weight pass per expert in the
   grouped sparse matmul; GDN prefill glue audit.
3. MTP: planned as its own turn (verify/commit/draft traces, device accept, lazy GDN commit), after the prefill levers.
