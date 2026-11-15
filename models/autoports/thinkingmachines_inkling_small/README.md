# Inkling-Small on one QB2 (hybrid MoE)

[thinkingmachines/Inkling-Small](https://huggingface.co/thinkingmachines/Inkling-Small) (276B total / 12B active, BF16)
on a QB2 (2 x p300c = 4 Blackhole dies, TP4 mesh `P300x2`), served through vLLM + the TT plugin. The experts do not
fit in the dies' DRAM, so the port runs under the hybrid-MoE overlay (`models/experimental/hybrid_moe`): each MoE
layer keeps its 191 most-used experts on the device, the other 67 are computed on the host CPU from int4 copies
(decode) or streamed to the device over PCIe (prefill). Text only for now.

Running log, measurements and open items: `models/experimental/hybrid_moe/HANDOFF_INKLING.md`.

## Model, as ported

- 42 layers, hidden 4096, 32 query / 8 KV heads of 128. Layers 0-1 dense (I 16384), 2-41 MoE: 256 routed experts
  (I 2048) top-6 on sigmoid + bias, plus 2 shared experts weighted in the same normalisation (`tt/optimized_decoder.py`:
  the shared experts are slots 256, 257 of the expert tensors, always picked).
- Attention (`tt/attention.py`): 35 sliding-window layers (512) with ring caches, 7 global layers (5, 11, ..., 41)
  with paged caches; no RoPE: a learned relative bias over the last 1024 (global) / 512 (sliding) positions; q / k RMS
  norms; log-scaling of q and the bias past 128K positions (prefill and verify pass `tau`).
- Four short causal convolutions per layer (`tt/sconv.py`, K = 4, fp32): on K, V, the attention output and the MLP
  output. Rows of up to 29 run as one fused program (`tt/sconv_kernels`); the carried 3-row state is rolled back on
  rejected speculative rows by committing only the accepted rows.
- Native MTP drafter (`tt/inkling_mtp.py`): 8 chained depths in the checkpoint, 2 used (k = 2 drafts per verify pass).

## Serving

```
models/experimental/hybrid_moe/serve/serve_inkling.sh            # CTX=32768, PORT=8000, STREAM=1
python models/experimental/hybrid_moe/serve/inkling_features.py  # reasoning / tools / streaming / sampling checks
python models/experimental/hybrid_moe/serve/inkling_long.py 8000 1000,32000,120000   # needle at N tokens
```

- `tt/generator_vllm.py`: the vLLM adapter (`InklingForConditionalGeneration`, registered from `vllm_bundle/`).
  Single user (`--max-num-seqs 1`); the engine owns every cache (vLLM's KV allocation is a token budget only).
- `tt/inkling_serve.py`: the engine. Bucketed, layer-major chunked prefill (+ the drafter's prompt pass), then
  speculative decoding: verify rows, commit, draft, all replayed as traces; the hybrid MoE exchanges routing and the
  CPU experts' partials with the host through device sockets inside the verify trace. Greedy output equals plain
  decoding; sampling is exact speculative sampling.
- Reasoning effort is a chat-template kwarg (`"chat_template_kwargs": {"reasoning_effort": "none" .. "max"}`,
  default high); vLLM's `inkling` reasoning and tool parsers.
- Sampling: temperature / top-k / top-p / seed and the presence / frequency / repetition penalties run inside the
  speculative step (`tt/sampling.py`: exact speculative sampling, the penalties applied to every verify row with the
  drafts before it counted). Requests the plugin samples on the host (min_p, logit_bias, bad words, allowed token ids,
  min_tokens, logprobs, structured outputs) take a one-token step that returns the logits (`_host_sampled_decode`):
  every parameter works, at ~23-26 tok/s instead of ~35-45.
- Structured outputs (`response_format` json_schema / json_object) work. `tool_choice: "required"` produces a valid
  JSON list of calls, but vLLM's `inkling` tool parser does not convert it, so it arrives as message content
  (`tool_choice: "auto"`, the usual path, returns proper `tool_calls`).
- Every prefill program is compiled in the warm-up, before the decode traces are captured (a program first compiled
  after them can have its kernels overwritten by the replays and hang the device); the engine re-captures if the
  program cache grows anyway (`_new_programs`, logged).
- Package: `tt-model-hybrid.yaml` (`tt-model package --container ...`); the first boot builds the host expert copies
  from the official checkpoint.

## Configuration (one: quality at parity, then speed)

- Experts on the device: 191 per layer, gate/up bfp4, down bfp4 (bfp8 in layers 2-17); the other 67 on the host CPU
  (int4).
- Decode: fused kernels, host <-> device sockets, MTP k=2.
- Prefill: missed experts streamed over PCIe (gate/up bfp4, down bfp8 copies), the fewest-token ones on the CPU (bf16
  int4 kernel); `STREAM=0` puts them all on the CPU.

## Quality (teacher-forced vs the BF16 CPU reference, 8 held-out 1000-token texts)

| configuration | top-1 | top-1 (confident) | KL (top-32) |
|---|---|---|---|
| prefill path (CPU experts) | 0.866 | 0.947 | 0.213 |
| prefill path, bf16 CPU kernel (`hostk_moe_csr`) | 0.868 | 0.950 | 0.209 |
| decode path (verify rows of 3) | 0.870 | 0.951 | 0.203 |
| the first parity configuration (106 bfp8 experts on device, ~3x slower; retired) | 0.862 | 0.945 | 0.218 |
| MiniMax-M2.7 package, same measurement | 0.848 | 0.933 | 0.196 |
| two valid CPU references against each other (noise floor) | 0.925 | 0.978 | 0.081 |

Task evals through the vLLM server (`serve/evalq.py`, greedy, reasoning effort none): GSM8K 97.0% (200 problems),
IFEval 88.0% strict / 90.0% loose prompt-level, 91.5% strict instruction-level (200 prompts).

## Tests and tools

- `tests/`: device / host unit tests (attention, sconv incl. `test_sconv_kernel.py`, router, MoE, layers) against
  the reference fixtures (`models/experimental/hybrid_moe/ref_inkling.py`).
- `tools/teacher_force.py` (the quality gate), `tools/engine_bench.py` (decode, dev signal), `tools/prefill_bench.py`,
  `tools/layer_prof.py` (device time per op), `tools/spec_bench.py` (the original dev harness).
