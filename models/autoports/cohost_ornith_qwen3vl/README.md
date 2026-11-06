# Co-host: Ornith-1.5 + Qwen3-VL-Embedding-8B + Qwen3-VL-Reranker-8B on one QuietBox 2

Three models from one process on all four Blackhole chips, each on its own OpenAI-style endpoint:

| member | kind | port (base 8000) | API |
| --- | --- | --- | --- |
| Ornith-1.5-122B (or -35B-A3B) | vLLM | 8000 | chat / completions, tool calls, reasoning, images |
| Qwen3-VL-Reranker-8B | app | 8004 | `/v1/rerank`, `/v1/score` |
| Qwen3-VL-Embedding-8B | app | 8005 | `/v1/embeddings` |
| co-host status | | 8006 | readiness, DRAM and L1 per member and phase |

## Why it is built this way

- **Ornith-1.5-122B needs all four chips** (about 18 GiB of each chip's 31 GiB before its KV pool), so the 8B models
  must share its chips. One chip can be opened by one process, so they share its process: a tt-model `kind: cohost`
  group. Ornith is a vLLM member; the embedder and reranker are FastAPI app members. Groups that mix the two kinds need
  tt-model branch `raahem/cohost-mixed-groups` (rnabeelTT/tt-kernel-package-manager), which also carries
  `tt-model deploy`.
- **One 8B copy per chip, not two.** Both 8B checkpoints are Qwen3-VL-8B (every tensor has the same shape), so
  `cohost_qwen3vl` builds ONE text tower over the mesh with every weight sharded by chip: chips 0-1 hold the embedder,
  chips 2-3 the reranker (`QWEN3VL_COHOST_LANES`). Same program, own weights, own inputs on each chip: one pass serves
  both models' requests. Two replicated copies would need about 16 GiB per chip, which does not fit next to the 122B.
- **Not 1x1 submeshes:** a submesh gets its own allocator, blind to the parent mesh's, so a submesh model would be
  given DRAM the LLM already uses. Everything here allocates on the one parent mesh.
- **One tt-metal build per process.** Each image is `FROM` the Ornith package image it serves (its exact tt-metal
  build, vLLM plugin and model code). The two Ornith images share the C++ build but ship different versions of the
  qwen36 decode kernels, so each size gets its own image. The Qwen3-VL code (written on tt-metal v0.82) is added as new
  packages and runs on the Ornith images' v0.79 tree (checked on a mock mesh: every program shape builds).

## Vision (profile `ornith<size>-vision`, the 35B default)

- Both 8B models get their Qwen3-VL vision tower, tensor-parallel over the four chips. The tower code (`vision_tower.py`
  from the single-model packages) was written for one chip, so `patches/vision_tower_tp.patch` shards its input along
  the hidden dim and gathers the merger and DeepStack outputs back to whole tables on every chip;
  `compat/tensor_utils_v082.py` and the `tt_ccl.py` edit in `build.sh` cover the v0.82 helpers the tower imports.
- An image pair runs untraced, so its programs must all exist before the traces are captured (Ornith's program guard
  stops the engine otherwise): the warm-up runs every patch bucket's tables through the text tower at every text bucket.
  Inputs are one image each; image sets and videos get HTTP 400 (`engine.check`, via `patches/service_check.patch`).
- Validated on the 35B (2026-11-06): page-image embeddings vs the single-model embedder, cosine 0.995; reranker picks the
  source page 8/8; Ornith's greedy output unchanged with images flowing. Not yet run on the 122B.

## Files

| file | what |
| --- | --- |
| `cohost_qwen3vl/split.py` | the chip-sharded text tower (`SplitScorer`) |
| `cohost_qwen3vl/engine.py` | traced passes, the combiner that packs both models' requests into one pass, `Lane` engines |
| `cohost_qwen3vl/member.py` | the co-host app members `embed` / `rerank` and their HTTP apps (the stock server module, one copy per model) |
| `Dockerfile`, `build.sh` | the image: `./build.sh 122b` or `./build.sh 35b` |
| `make_package.py` | stages manifest + card + OCI image for `tt-model push`; holds the DRAM budget |
| `deploy/ornith35b-qwen3vl-separate.yaml` | the no-co-host layout for the 35B (it fits on one board): three separate servers |

## Build, stage, publish

```bash
./build.sh 122b                                   # prints the image tag
~/fork-tt-model-manager/.venv/bin/python make_package.py 122b <tag> /tmp/stage-122b
tt-model push /tmp/stage-122b --public            # raahemnabeel/cohost-ornith-122b-qwen3vl
```

## Run

```bash
uv tool install --force git+https://github.com/rnabeelTT/tt-kernel-package-manager@raahem/cohost-mixed-groups
tt-model pull raahemnabeel/cohost-ornith-122b-qwen3vl --with-weights
tt-model serve raahemnabeel/cohost-ornith-122b-qwen3vl --port 8000      # profile ornith122b
curl -s localhost:8006/                                                 # status
```

## Validate (on hardware, in order)

1. Solo baselines with the group's settings: `--profile ornith122b-solo` and `--profile qwen3vl-solo`; greedy
   outputs saved (16 prompts per model; embeddings and rerank scores for a fixed set).
2. The group (`ornith122b`): same outputs, each member asked alone, then all three at once.
3. Each 8B model against its single-model package (`raahemnabeel/qwen3-vl-embedding-8b`, `-reranker-8b`): embedding
   cosine and rerank score deltas (the reranker runs `p0_bfp8_hifi2` here, `p1_mlp_lofi` alone).
4. Status endpoint: DRAM per chip after every phase, L1 held per core, `compiles_after_capture` after a benchmark.
5. If the boot runs out of DRAM: `--profile ornith122b-lean`. If a program fails with an L1 clash, rerun with
   `TT_COHOST_L1_REPORT=1` to see who holds what.
