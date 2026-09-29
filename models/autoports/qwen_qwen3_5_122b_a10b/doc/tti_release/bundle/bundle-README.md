---
tags:
- blackhole
- p150x4
- p300x2
- tt-model-cache
- tt-model-container
- vllm-plugin
---

# qwen3-5-122b-a10b-blackholex4

Qwen3.5-122B-A10B (256-expert MoE, 48 layers of Gated DeltaNet + gated attention) served text-only on four Blackhole dies through vLLM and the Tenstorrent plugin: tensor-parallel mixers, experts sharded on the intermediate dimension, on-device sampling, per-width decode traces, 262144-token context, up to 32 concurrent sequences.

Runs on **p300x2** or **p150x4** — see the serve profiles below.

Packaged and published with [tt-model-manager](https://github.com/tenstorrent/tt-model-manager) 0.1.0 (manifest schema 5.1).

## Quickstart

```bash
tt-model pull  raahemnabeel/qwen3-5-122b-a10b-blackholex4 --with-weights
tt-model serve raahemnabeel/qwen3-5-122b-a10b-blackholex4
```

`pull --with-weights` downloads the Docker image and the [`Qwen/Qwen3.5-122B-A10B`](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) weights at `dc4d348443bc740c68e2d77492492c11606384d5` (into your HF cache; they are not in the image). `serve` starts an OpenAI-compatible server on port 20000 (or the next free port, if that one is busy); the first start compiles kernels for your device, which takes several minutes, and the server is ready when it logs `Application startup complete`.

### Use with your client
Point an OpenAI-compatible client at `http://127.0.0.1:20000` (or the port `tt-model serve` printed) with model
`Qwen/Qwen3.5-122B-A10B`. Thinking is OFF by default in this bundle (the validated release configuration); turn it on
per request with `"chat_template_kwargs": {"enable_thinking": true}` — the reasoning then arrives in the message's
`reasoning` field (vLLM 0.26; older clients know it as `reasoning_content`) and needs a large `max_tokens`
(thousands): when the budget ends inside the think block, `content` is empty. In thinking mode the model tends to
deliberate at length and can repeat itself on constraint-heavy prompts (also with the bf16 reference weights); use
the model card's general thinking preset (temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5) rather than
greedy decoding there.
Tool calls use the `qwen3_xml` parser.

## Serve profiles

One image serves every profile below; pick one with `--profile`.

| profile | hardware | mesh | max_num_seqs | max_model_len |
| --- | --- | --- | --- | --- |
| `p300x2` *(default)* | p300x2 | P300x2 | 32 | 262144 |
| `p150x4` | p150x4 | P150x4 | 32 | 262144 |

## Provenance

The exact sources the image was built from — `code/` in this repo is byte-identical to the model code inside the image:

| component | built from |
| --- | --- |
| tt-metal | a local checkout — commit not published *(dirty tree — the image includes uncommitted changes)* |
| vLLM | [`v0.26.0`](https://github.com/vllm-project/vllm/releases/tag/v0.26.0) |
| vllm-tt-plugin | a local checkout — commit not published |
| `code/` digest | `39e6b36b56f864a9` (sha256, first 16 hex digits) |
| built | 2026-09-27T10:15:01+00:00 by tt-model 0.1.0 |
