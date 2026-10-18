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
