# MTP (multi-token prediction) plan for Qwen3.5-122B-A10B on p300x2

Status: planned (2026-09-30), to start after perf-pass turn 5 ships. Sources: the checkpoint's `mtp.*` keys (785 tensors in
shards 37-39), vLLM 0.26's `qwen3_5_mtp.py` (reference math), the Qwen3.8 bundle study (`qwen38_bundle_study.md`,
trace/accept design), and a read of `vllm-tt-plugin` (no speculative-decoding support today).

## The MTP head (from the checkpoint and vLLM's reference)
- `mtp.fc [H, 2H]`, `mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden` (zero-centred RMSNorm like the main model),
  ONE decoder layer `mtp.layers.0.*` = gated full attention (q/k/v/o + q_norm/k_norm, partial RoPE) + the same MoE
  (router `mlp.gate`, 256 routed experts stored UNFUSED per expert `experts.N.{gate,up,down}_proj`, shared expert +
  `shared_expert_gate`), `mtp.norm`; embeddings and LM head shared with the main model (`mtp_use_dedicated_embeddings=false`).
- Draft step for one token: `h = fc(cat(norm_e(embed(tok)), norm_h(hidden)))`, `h = layer(h, pos)` (own KV cache, positions =
  the token's position), `logits = lm_head(norm(h))`, `draft = argmax`. `hidden` = the main model's final hidden state of the
  token being extended (the state that produced `tok`'s logits) AFTER the model's final norm (vLLM's Qwen3NextModel
  returns `self.norm(hidden, residual)` and the proposer feeds exactly that; the head's `pre_fc_norm_hidden` is applied on
  top). k drafts = k chained steps, each feeding the MTP layer's own normed output as the next hidden.

## Serving contract (route b: model returns up to k+1 accepted tokens per step)
- `vllm-tt-plugin` refuses `speculative_config` (`platform.py:1480`) and never reads `scheduled_spec_decode_tokens`; its
  fixed-width block-output path (`model_capabilities["output_tokens_per_step"]=K`, `TTScheduler` reserves K placeholders,
  `_coerce_output_block` needs a rectangular `[B, K]` block) is the hook. Plugin changes (small, all in the plugin):
  1. accept rows of variable width `1..K` (`_get_output_tokens`, `_coerce_output_block`, strip trailing -1),
  2. advance `num_tokens` by the per-row length (`_apply_sampled_tokens_to_state`),
  3. `TTScheduler._update_request_with_output`: allow `1 <= len <= K` (keep subtracting the K reservation),
  4. `_is_still_prefilling`: up to K outstanding tokens is steady state,
  5. decode input stays `[B,1]` last token + position; run without host/device overlap first.
  Inherited restrictions: no prefix caching / chunked prefill / host sampling / logprobs / structured output on the MTP
  profile -> ship MTP as an opt-in serve profile (`QWEN35_MTP=k`), default profile unchanged.
- Route (a), proper vLLM spec-decode (`take_draft_token_ids`, `scheduled_spec_decode_tokens` as `[B,1+k]` inputs,
  rejection sampler), is the long-term shape; do (b) first, measure, then decide.

## Device design (from the Qwen3.8 blueprint, adapted)
- Per step per user: VERIFY = the main model on the R = k+1 rows `[t, d1..dk]` at positions P..P+k (a width-R "prefill-like"
  decode: batch dimension = candidates for attention with the same page table; GDN needs the R rows as a short sequence),
  then greedy accept on device (eq -> prefix count via triangular matmul -> `a`), next token = argmax row `a`; the MTP layer
  runs on the same R rows fed with the target's own argmaxes (alignment rows) so d1' is free; then k-1 chained draft rows.
- State: KV rows written by absolute position (rejected rows overwritten next step); GDN recurrent + conv state committed
  lazily from `a` (per-row prefix states, select slot `a`); the current width-1 trace stays for non-MTP users.
- Traces: `verify_first` (verify + accept + alignment), `commit`, `draft` (k-1 rows). One readback per step: `[a, tokens]`.
- Start greedy-only (temperature 0 / sampled requests fall back to the plain decode trace); exact point-mass speculative
  sampling for sampled requests later (Qwen3.8's 75-line host routine is portable).
- Weights: 1 extra layer (~1/48 of the expert bytes: ~0.4 GiB bfp4 per chip + attention), MTP experts need the per-expert
  -> fused `[E, 2I, H]` repack in `weights.py` (today `mtp.*` are dropped).

## Findings so far (2026-09-30)
- Head weights load in the main layers' fused layout (`StreamingStateDict.mtp_state`); torch reference in
  `tests/reference_mtp.py`; device head in `tt/mtp.py` (the optimized decoder layer as virtual layer 48).
- Precision: with the shipped bfp4 mixers the MTP layer's output reaches only PCC 0.95 vs the reference (main attention
  layers: 0.995); bfp8 attention weights (HiFi2) bring it to 0.99 (routed experts' dtype irrelevant) -> the head's layer
  is built with bfp8 attention projections (~20 MB per die).

## Prefill requirement (found 2026-09-30 08:40)
- The MTP layer keeps its own KV cache and must see every prompt position: during prefill it runs over the bucket with
  rows (token t+1, final-normed hidden t) for t < P (the row at P takes the first sampled token), i.e. the head's layer
  in prefill mode right after the main model's bucket program (+1/48 layer of prefill work, +1 attention layer of KV).
  Drafting from an empty MTP cache gives ~0 % acceptance.

## Measured acceptance (2026-09-30, `tools/mtp_acceptance.py`, k = 4, 64 greedy steps, head with bfp8 attention)
| prompt | tokens per pass | first-draft agreement | histogram (0..4 accepted) |
|---|---|---|---|
| prose | 2.89 | 0.78 | 14 / 15 / 13 / 8 / 14 |
| code | 4.86 | 1.00 | 0 / 1 / 1 / 4 / 58 |
| json | 3.98 | 0.84 | 10 / 4 / 4 / 5 / 41 |
| chat | 3.03 | 0.84 | 10 / 14 / 17 / 10 / 13 |
Expected single-user speed-up = tokens per pass / (verify-pass cost in decode steps); at ~1.6x decode per pass:
prose 1.8x, chat 1.9x, json 2.5x, code 3x (k = 5-6 pays on code / json).

## Verify-pass design (decided 2026-09-30 after reading the GDN and KDA kernels)
- The bucket prefill path is out (the 128-token bucket costs 191 ms per 48 layers: more than R decode steps). The verify
  pass is a "rows" mode per layer: GDN in prefill form on ONE 32-row chunk (R valid rows, recurrent + conv state carried
  from the user's decode state: `TPGatedDeltaNet.forward_prefill` with the chunk-outer carry, or the KDA kernels
  `qkv_causal_conv1d_silu` + `prepare_chunk_recurrence` + `recurrent_chunk_scan`, which take a 32-token chunk and an
  initial state), attention as a batched decode of the R rows (same page-table row, positions P..P+k; the KV of all rows
  is written before the SDPA reads, so row j sees rows < j), MoE through the pairs path on the R real rows only
  (R*k pairs). Estimated pass cost ~1.3-1.5x a decode step at R = 5; the GDN rows part is the term to fuse later.
- Accept on device (Qwen3.8 arithmetic), commit the GDN state lazily from the accepted count (per-row prefix states
  from the chunk scan, select slot a), KV rows overwritten by position.

## GDN state carry for the verify rows (plumbing, from gdn/tp.py)
- Batched decode buffers: `rec_state [B, Nv, Dk, Dv]` (slot = row), `conv_states[m] [1, B, qkv_dim_tp]` for the K = 4
  shift-register taps. The B = 1 prefill form carries `rec_state`(scratch) + `conv_carry [1, K-1, qkv_dim_tp]`;
  `assemble_batched_state` maps `conv_states[m][:, u] = conv_new[u][:, m-1]` (m = 1..K-1, tap 0 zeroed), so the reverse
  for a verify pass of slot s is `conv_carry[:, m-1] = conv_states[m][:, s]` and `rec = rec_state[s]`; afterwards
  `write_slot(s, rec_new, taps)` (existing) commits the R accepted rows' state (lazy commit: run the chunk on the accepted
  prefix only, or keep per-row prefix states and select).
- Attention rows: `TPAttention.rows_kv_write` (added 2026-09-30) writes the R rows' K/V one row at a time (the per-user
  paged update races within one 32-position tile), batched SDPA reads unchanged.

## Status (2026-09-30 11:45)
- The v1 loop WORKS on the full model: prose k = 4, 2.09 tokens per pass over 48 tokens (acceptance 0:10 1:7 2:1 3:2
  4:3), greedy-equal until a bf16 near-tie. The verify pass is exact up to bf16 numerics (`tools/mtp_pass_check.py`:
  rows vs eager decode PCC 0.9966-0.9991, every argmax disagreement one 0.125-margin near-tie). Two pitfalls found on the
  way: (1) the head must be allocated BEFORE any trace capture (a replayed prefill trace zeroed a head built after it ->
  `build_generator(mtp_head=True)`, tools use `warm_prefill_trace=False`); (2) the B = 1 loop runs the pass in place on
  the bound GDN state with a device snapshot for the commit (the scratch-bind + slot-copy path failed an allocation
  assert).
- Next: GDN-only commit check, other prompts, then the traced pass (fixed BW = 8 rows, persistent inputs, device accept),
  the head's prefill inside the bucket prefill, plugin route (b).

## Status (2026-09-30 10:00) and the commit design
- Done: head (bfp8 attention) PCC 0.99; acceptance measured; rows mode per layer validated (attention: per-row K/V
  write, PCC 0.995-0.996; GDN: prefill form with the carried state, PCC 0.9986-0.9999); whole-model verify pass over 5
  rows = sequential decode at PCC 0.998-0.9995 (`tt/mtp_rows.py`, `tt/mtp_verify.py`); v1 eager loop
  (`mtp_generate_eager`) under test vs plain greedy.
- Exact GDN commit without a second full pass: the accepted rows 0..a of a verify pass have exactly the activations the
  pass computed (causality), so the commit only needs the GDN layers replayed on the pass's saved normed inputs with
  `valid_len = a + 1` (36 x [1,1,128,H] bf16 = 27 MB kept per pass; the valid_len mask is a device tensor -> trace-safe).
  Cost ~ the GDN share of a pass (~15 ms traced estimate) instead of a whole second pass; skipped when a == k.
- Estimated traced pass: ~32 ms (48 layers in rows mode, decode-like) + ~15 ms (GDN prefill form over decode-form)
  + ~15 ms commit = ~1.9 decode steps -> prose 1.5x, chat 1.6x, json 2.1x, code 2.6x single-user; the fused GDN rows
  kernel is the lever after that.

## Milestones and gates
1. Host reference: torch MTP head forward on CPU vs the checkpoint (draft acceptance rate on the book/AIME prompts, k=1..4)
   -> expected tokens per step. 2. TT MTP layer PCC vs the reference (stage-1 style test). 3. Width-R verify + device accept
   + lazy GDN commit, single user, determinism vs plain decode (greedy streams must match up to bf16 near-ties, measured).
4. Plugin variable-width output + serve profile; served tok/s at 1 / 4 / 8 users vs today (30.6 t/s/u at 1 user).
