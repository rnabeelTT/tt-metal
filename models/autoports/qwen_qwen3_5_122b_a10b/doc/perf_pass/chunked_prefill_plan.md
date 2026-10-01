# Chunked prefill interleaved with decode (turn 7 plan, 2026-10-01)

## What the plugin does (vllm-tt-plugin, read 2026-10-01)
- `model_capabilities["supports_chunked_prefill"] = True` + vLLM's default `enable_chunked_prefill` keep chunking on
  (`platform.py:175-219`); `output_tokens_per_step > 1` disables it (our MTP key `max_output_tokens_per_step` is not read).
  Budget = `--max-num-batched-tokens` (2048 for `vllm serve`); chunk = min(remaining, leftover budget, long-prefill
  threshold). Boundaries are RAGGED: several prefills share one budget, a new request gets what is left.
- One `prefill_forward(tokens, page_table, kv_cache, prompt_lens, start_pos, empty_slots[, sampling_params])` per
  prefill step over all prefilling rows (new + continuing): `tokens` = the full history [N, max end], `start_pos` =
  num_computed_tokens, `prompt_lens` = this chunk's END (not the prompt length). Rows with more tokens to come are
  "intermediate" (`intermediate_prefill_mask`, model_runner.py:1222): if any row is intermediate, device sampling is
  OFF for the whole step and the model must return logits [N,1,vocab] (intermediate rows' logits are discarded).
- A scheduler step is either one prefill call or one decode call, never both; prefill is preferred while any chunk is
  pending, so running decodes stall for the whole prompt anyway (decode between chunks only under KV pressure). Chunking
  therefore bounds the step length and lets prompts share steps; it does not by itself cut the running users' gap.
- State: KV is paged at absolute positions (page_table + start_pos). The GDN state of a half-prefilled request is the
  model's problem: `empty_slots[u]` is re-picked every prefill step (plugin hazard: a continuation can get a different
  slot and another prefill can take the old one), so key the carried state on `page_table[u, 0]` (a request's first
  block is unique and stable) and release it at the final chunk.
- Reference models (tt_transformers Generator) floor `start_pos` to an alignment and recompute the tail; harmless for
  KV, WRONG for GDN (double integration) - do not copy that.

## Model-side design (smallest change)
- `Qwen35MoeGenerator._prefill_single(tokens[s0:E], pt, start=s0)`: the segment loop takes an absolute start
  (`prefill_masked_bucket(seg, pt, actual_len, chunk_start=cs)` already resets the GDN state only at cs == 0 and
  carries it across segments in the bound B = 1 scratch; attention runs with `chunk_start_idx = cs` over [0, E)).
- Alignment A = PREFILL_SEGMENT (2048): a chunk [s, e) is processed as [s0, E) with s0 = the request's own processed
  end (== floor(s / A) * A when the plugin's budget is a multiple of A) and E = floor(e / A) * A for an intermediate
  row, E = e for a final row; the unaligned tail of an intermediate chunk is deferred (recomputed next chunk from its
  tokens, which the plugin always resends), so the GDN state never integrates a token twice and the KV of the tail is
  simply rewritten. Intermediate rows return zero logits (skip the LM head).
- Carried state per pending request (keyed by first block id): device clones of the scratch rec_state + K conv taps
  (`_snapshot_gdn_scratch_device`) after an intermediate chunk; copied back into the scratch (`copy_slot_into_scratch`
  style) before the next chunk; dropped at the final chunk, whose state goes to the decode slot as today.
- Which rows are final: the plugin does not tell the model. Smallest fix = a shim (like mtp_plugin_shim) that passes
  `intermediate_prefill_mask` (or the full prompt lengths) into the `prefill_forward` kwargs; fallback heuristic:
  `sampling_params is not None` => every row final.
- MTP: the head's prompt-rows hook already runs per segment with `chunk_start`; `begin_prefill` per chunk with the
  request keyed by first block, `end_prefill` only at the final chunk.
- Warm-up: the masked-bucket programs are warm for every bucket at `chunk_start = 0`; a continuation uses the same
  programs with a different chunk_start (host-int) - verify no new program compiles (program guard) in the first test.
- Validation: plugin `tests/tt/test_chunked_prefill.py` (needs `--tt-chunked-prefill-budget`); then a served long
  prompt (8K / 32K) with `--max-num-batched-tokens 2048`: equal completion to the unchunked server, TTFT and the
  running users' gap measured with a concurrent short request.
