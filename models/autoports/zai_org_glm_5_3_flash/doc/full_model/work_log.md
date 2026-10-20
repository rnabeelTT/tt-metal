# Stage 6 work log (full model)

## 2026-10-16

- `tt/model.py` (Glm53Model), `tt/generator.py` (Glm53Generator, `build_generator`) and `tt/decode_inputs.py`
  (per-step DSA / KDA decode inputs computed on the device from the position tensor) written. `tools/readiness.py`
  opens the (1, 4) mesh with fabric, l1_small and a trace region (the stock runners open a bare mesh).
- HF full-model reference on CPU (`tools/hf_full_ref.py`, lazy FP8 experts): every layer 0..44 is bitwise equal to the
  per-layer reference; `readiness_aime24_chat.refpt` generated (161 prompt tokens, 100 generated, top-k 100).
- Reduced probe (`tests/test_model_probe.py`, layers 0, 3, 4): the traced-greedy test hung. Bisect with
  `tools/model_step_probe.py` (one sync per phase) located the hang in a prefill AFTER decode trace replays. Cause:
  prefill builds persistent constants lazily; built after the capture, they landed in freed trace-intermediate memory and
  the next replay overwrote them. Fix: `_warm` prefills every constructed length before any capture, and captures the
  sampler trace right after the model trace. Probe 3 / 3 (`runs/s6_probe4.log`).
- Also fixed: positions now advance on the device at the end of `decode_step`; `reset_state` used `x * 0` (kept NaN,
  leaked one state copy per tensor) and now copies zeros and frees the temporary.

## 2026-10-17

- First full-model readiness: OOM in the 2048-row warm-up prefill (21 MB free per bank; 8 slots, 512 MiB trace
  region). The sampler then rejected 4 slots (`format_sampling_params` needs a multiple of 32): the model now pads the
  logits rows to `SR = ceil32(S)` and the token buffer is SR wide.
- Teacher forcing 93 / 100 / 100 and prefill check 92 / 100 / 100 (4 slots, 256 MiB trace region; `runs/`).
- Autoregressive (256 tokens, HF side precomputed on CPU with the same model) and the 6-prompt qualitative suite pass
  (`autoregressive/VERDICT.md`, `qualitative/VERDICT.md`).
- Memory (`memory_budget.json`, `memory/`): measured weights, KDA state per slot, DSA bytes per token, decode trace
  46 MiB (default region now 64 MiB). The MoE prefill runs its routed experts in row blocks (worst-case dispatch buffer
  183 MB, live twice) and the DSA indexer uses 64 MB blocks: both bit-identical. 1-slot capacity probe: 104,515-token
  prompt, 24 / 24 decode steps correct. `doc/context_contract.json`: supported_context 104,448.
- Weight cache (`weight_cache.md`): the load hang was the driver pin of a file mmap; cache files are staged through a
  memfd. Full model warm load 110 s vs 24 min, byte-identical weights.
- Stage 7 later changed the defaults that this stage measured (chunk plan, MoE split rule, ag_persistent off); the
  final-default readiness is in `doc/optimized_full_model/`.

## 2026-10-18: stage-6 review #1 / stage-7 review #1 rework

Logs: `~/glm53-flash-bringup/logs/s67r/` (every device job stamped with git HEAD + the md5 of the snapshot's
`tt/*.py`; `status` lists every job's one-line result). Device jobs ran from pinned snapshots (`~/glm53-flash-bringup/
s67r_snap*`, `s67r_final` = HEAD 656c998e3df + this rework's files) because other agents edited the shared tree.

Code (tt/generator.py, tt/model.py, tt/decode_inputs.py, tt/dsa.py, tt/dsa_shard.py, tt/weight_cache.py,
tt/functional_decoder.py; guard lines in tt/moe.py / tt/moe_opt.py):
- `build_generator` / `Glm53Generator`: defaults from the DRAM budget (`plan_context`): 1 slot, the largest context of
  the fastest DSA layout that fits; the block-sharded layout is chosen automatically when the request exceeds the
  replicated capacity (bf16 latent only); a request that fits nowhere raises `ValueError` before any weight loads, and
  the standalone generator re-checks the pool against the free DRAM measured after load. Budget source:
  `serving_profiles.json` (stage 9) when its precision id is the build's, else the built-in stage-7 numbers; DSA bytes
  per block from the latent dtype; reserve max(data, 1200 MiB). Trace region default from the data file.
- Decode-overrun guard: positions >= max_seq_len raise on the host (`generate`, `decode_forward`, serving decode) and are
  inactive on the device (`DeviceDecodeInputs.refresh`: no cache row written, open-pool state kept).
- Sampler: every variant (greedy, penalties) compiled before any capture, including the penalty token-count programs
  (the cause of "Cannot load new binaries during trace capture"), and captured back to back after the model trace;
  lockstep sampling (identical per-draw seeds + die-0 token broadcast) on in the standalone generator too.
- Weight cache: per-layer MoE precision exceptions key their own moe directory; checkpoint shard fingerprint (size +
  mtime) in the manifest; `_build_lm_head_ds` in the top digest.
- Prefill/decode transition made allocation-free after the capture: the DSA open-pool hand-off uses request-independent
  programs (one-hot row select + slot mask) instead of slices at (prompt length, slot); `_warm_extra` compiles the
  remaining request-dependent programs (KDA commit per slot, last-row tile per offset, the all-logits head) before the
  capture. Checked by tt-metal's trace allocation tracker (`TT_METAL_TRACE_ALLOC_TRACKING=1`): 0 live post-capture
  allocations at 1 / 8 / 32 slots, replicated and block-sharded, and on the stage-9 adapter smoke.
- Address-stability guard: `Glm53Generator.check_trace_addresses` (after every prefill and before every reset) and an
  `is_allocated` check of every buffer a decode replay reads.

Defects found and fixed (in order):
1. Penalty-sampler capture compiled programs inside the capture (count_tokens path): warm-up now runs it eagerly.
2. Deferred DSA pool (standalone `allocate_dsa=False`) with the sampler built first: traced decode != eager. The sampler
   is now built after the pool (`Glm53Model._init_sampler`); the deferred path passes the reduced probe and the adapter
   smoke's traced == eager check. The standalone default keeps the HEAD allocation order (`GLM53_DEFER_POOL=0`).
3. Penalty update deallocating a view of the token buffer: our call site counts a clone (`Glm53Model._init_sampler`).
4. Request-dependent slice programs compiled after the capture (open-pool hand-off and friends): see above.
5. **The full-model decode corruption (teacher forcing 14 / 32 / 90)**: `Glm53Model.decode_step` padded the logits rows
   to the sampler width with `ttnn.pad` and then `ttnn.deallocate(logits)`. At S < 32 the pad stays inside the tile
   padding and returns a VIEW of `logits`; `ttnn.deallocate` defaults to force=True and freed the storage of the
   returned tensor - the decode trace's persistent logits output. After the capture the allocator handed that block to
   later tensors; every replay then wrote logits into them, and the sampler read logits that other ops overwrote. It
   was latent since stage 6: harmless while nothing was allocated in that block, deterministic once something was (DRAM
   pressure: 65,536 + 480 MB ballast, 102,400 tokens at 1 slot; or any change in allocation order, e.g. the stage-5
   dsa_idx_rep option). Found by elimination: not post-capture allocations (tracker clean), not uninitialized reads
   (NaN-poisoned free DRAM, full and reduced model), not the sampler, not the MoE split, not trace-only (eager decode
   failed too); `ttnn.deallocate(force=False)` made the failing run pass (92 / 100 / 100 vs 14 / 32 / 91 in the same
   session), then a wrapper that records every deallocate leaving a still-referenced buffer named the call site.
   False trails: per-layer host reads / clones and pre-step state snapshots made the failure disappear only because
   they changed the allocation sequence. Fix: no deallocate after the pad. Same pattern (benign) fixed in
   `DeviceDecodeInputs._into`; five benign stage-3 sites in moe.py / moe_opt.py now free with force=False;
   `models/common/sampling/tt_sampling.py:1170` (`sampling_values`, freed after its last use) noted for upstream.
   Permanent check: `GLM53_DEALLOC_CHECK=1` (tests/test_model_probe.py::test_no_shared_buffer_frees).
