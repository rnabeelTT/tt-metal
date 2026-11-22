# Qwen3-VL-Reranker-8B on Blackhole

A TTNN port of [`Qwen/Qwen3-VL-Reranker-8B`](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B), served as a
`/v1/rerank` HTTP server from one Blackhole chip (p150) or several chips (one model copy per chip).

The model scores a (query, document) pair. Either side can be text, images, video frames or a mix. The score is
`logit("yes") - logit("no")` at the last position of the official reranker prompt, and the relevance is
`sigmoid(score)`.

## Layout

| Path | Role |
| --- | --- |
| `tt/inputs.py` | Host preprocessing shared by TT and HF: the official prompt (system prompt, `<Instruct>/<Query>/<Document>`), pixel budgets, special-token-preserving truncation, MRoPE positions from HF's `get_rope_index`. |
| `tt/text_model.py` | The 36-layer text tower as a prefill-only scorer, plus the precision policies and the matmul blocking table. |
| `tt/vision_tower.py`, `tt/vision/` | The vision tower: the Blackhole qwen36 ViT vendored from the Ornith-1.5 bundle, plus the DeepStack mergers. |
| `tt/runner.py`, `tt/traced.py` | Host batching, bucket inputs, and captured-trace replay of the text tower. |
| `tt/engine.py`, `tt/proc_engine.py` | The per-chip scoring engine and the one-process-per-chip multi-chip mode. |
| `server/` | FastAPI app: `/v1/rerank`, `/v1/score`, `/health`, `/v1/models`. |
| `reference/hf_reference.py` | HF CPU reference scorer. |
| `tests/` | The fixed evaluation pair set and the server tests (fake engines, no device). |
| `config/precision.json` | Precision policies; `selected` is what serves. |
| `config/matmul_blocking.json` | Autotuned `minimal_matmul` blocking for this chip's 11x10 grid. |
| `tt-model.yaml` | tt-model-manager container manifest (`kind: tt-dit-server`). |

## How it works

**Text tower.** One pass scores `B` pairs padded to a bucket length `T`, using right padding; causal attention keeps
padding out of every real row. The decoder is Qwen3's: fused QKV, per-head QK RMSNorm, RoPE in the Meta channel
pairing, causal GQA SDPA, and a SwiGLU MLP. Projections run through `ttnn.experimental.minimal_matmul`, with the
gate and up weights tile-interleaved for its fused SwiGLU, and with blocking autotuned per shape. A one-hot matmul
picks each pair's last real row. The final RMSNorm and a 2-row head (`lm_head[yes] - lm_head[no]`) give the score.
The other 151,934 vocabulary rows are never loaded.

**MRoPE.** Qwen3-VL's interleaved 3D rotary embedding has no special kernel. Host code builds cos/sin rows from the
`[3, T]` positions, and `rotary_embedding_llama` applies them.

**Vision.** The ViT is the Blackhole qwen36 implementation from the Ornith-1.5 bundle. It has 27 blocks over the
padded patch sequence, with block-diagonal SDPA windows (the real patches, then the padding). A video runs one
temporal frame group at a time, which is exact because HF's attention is block-diagonal per frame group. Changes for
this checkpoint and one chip:

- The three DeepStack mergers are tapped after blocks 8, 16 and 24.
- Patch embedding runs on device. The Conv3d's stride equals its kernel, so it is a linear layer.
- The last ViT blocks carry activation outliers around 1e4, so LayerNorm statistics use fp32 and the SDPA inputs and
  attention output projection stay bf16. This raised merger PCC from 0.90 to 0.99.
- Projections use `minimal_matmul`, and SDPA runs at HiFi2 with 256-row chunks.

**Merging vision features.** Visual features enter the text tower through `ttnn.embedding` gathers against
`[zero row; visual rows]` tables, both at the input embedding and after decoder layers 0 to 2 (DeepStack). The rows
are never copied to host.

**Serving.** Text-only pairs of a request are sorted by length and packed into captured-trace passes (34 `(B, T)`
slots from 32 pairs at 128 tokens to 1 pair at 32K; a small cost model picks which pairs share a padded pass). Pairs
with images or video run one at a time; an image shared by several pairs (an image query) is encoded once per call.
Every chip -- also the single-chip profile -- runs in its own process that sees only its device
(`TT_VISIBLE_DEVICES`), so the HTTP server, the host preprocessing and each chip never share a GIL; a request's
documents fan out across chips. A die process that dies or stops answering is reported by `/health` (503
`degraded`) and skipped by dispatch.

**Inputs.** Preprocessing follows the official reranker (`QwenLM/Qwen3-VL-Embedding`, `Qwen3VLReranker`): system
prompt, `<Instruct>/<Query>/<Document>`, images 4..1800 tokens, video files at `fps` (default 1) with at most 64
frames through `qwen_vl_utils` (a PyAV reader replaces torchvision's, which 0.26 removed; only the sampled frames are
decoded), frame lists with the official total-pixel budget, and text truncated to 10240 tokens keeping every special
token. The default instruction is the official Python one, "Given a search query, retrieve relevant candidates that
answer the query." (the sentence-transformers / vLLM paths on the model card default to "Retrieve text relevant to
the user's query."). A pair may be up to 32768 tokens long.

## Precision

The selected policy is `p1_mlp_lofi`:

| Part | Precision |
| --- | --- |
| Text weights | bfp8 |
| Activations | bf16 |
| Attention projections | HiFi2 |
| MLP | LoFi |
| SDPA | HiFi4 |
| Score head | fp32 with HiFi4 |

The vision tower uses bfp8 weights and bf16 activations. Its precision settings are in `tt/vision/precision.py`.
Policies were selected on the fixed pair set against HF fp32; the sweep, accuracy and performance evidence is in `doc/`.

## Run

```bash
export QWEN3VL_RERANKER_MESH_SHAPE=1x1   # 1x2 / 1x4: one model copy per chip
python -m uvicorn models.autoports.qwen_qwen3_vl_reranker_8b.server.app:app --port 20000
curl -s localhost:20000/v1/rerank -H 'content-type: application/json' -d '{
  "query": "A woman playing with her dog on a beach at sunset.",
  "documents": ["A woman shares a joyful moment with her golden retriever on a beach at sunset.",
                {"image": "<base64 jpeg>"}, "The stock market fell sharply on Tuesday."],
  "instruction": "Retrieve images or text relevant to the user'\''s query.", "raw_scores": true}'
```

## Demo

`QWEN3VL_RERANKER_DEMO=1` mounts a Gradio page at `/demo` (and redirects `/` there): rank a bundled, openly licensed
library (61 images, 42 multilingual texts, 29 code snippets, 10 videos; `demo/library/LICENSES.md`) against a text or
image query, or rerank your own candidates. It calls the in-process service, the same path as `/v1/rerank`. Without the
flag gradio is never imported. The package's `demo` and `p300x2-demo` profiles set it.

## Results

Measured on a QuietBox 2 (four p300c chips); full tables and the raw rows are in `doc/`.

- **Accuracy vs HF** (`doc/accuracy.md`): on the 340-pair fixed set (text, multilingual, code, images, video, long
  documents) the score has Pearson 0.9991 with HF fp32, a mean difference of 0.107 logits and the same top-1
  document in all 41 groups. MTEB and MIRACL subsets land within 0.15 NDCG@10 points of HF; on ViDoRe v3 pages TT
  scores 97.9 vs HF's 97.0 NDCG@5 (score Pearson 0.9994; one group's top-1 flips in TT's favour).
- **Speed** (`doc/performance.md`): one chip scores ~9.3k tokens/s of ~520-token text pairs (a 512-token pair in
  61 ms); four chips ~32.6k tokens/s (62 pairs/s). Images run at ~8 pairs/s per chip, 1.8K-token document pages at ~2.
