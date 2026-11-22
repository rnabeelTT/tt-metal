# Recovered from the published bundles

No git branch held this autoport's final code. It was extracted (2026-11-22) from the `code/` folder of the published
tt-model bundles on Hugging Face:

| Published package (Hub, revision) | What is here |
|---|---|
| `raahemnabeel/qwen3-vl-reranker-8b` (`f24735c4ab`) | the whole autoport, one level up (`../`), as bundled; the demo videos (`demo/library/videos/*.mp4`) are left in the bundle |
| `raahemnabeel/qwen3-vl-embedding-8b` (`51c46c7268`) | `qwen3-vl-embedding-8b/`: that bundle's copy of the autoport (its `config/precision.json`, `server/`, `tt/engine_api.py` differ from the reranker's) |
| `raahemnabeel/qwen3-reranker-4b` (`e5c8e95839`) | `qwen3-reranker-4b/`: that bundle's copy (its `config/precision.json`, `server/`, `tt/engine.py`, `tt/inputs.py` differ) |

The two smaller bundles ship only `__init__.py`, `config/`, `server/` and `tt/` (no `tt-model.yaml`, tests or docs).
The rest of each bundle's `code/` (`models/common`, `models/ttt_compat`, `models/tt_dit/utils`, ...) is upstream code
from around 2026-11-02; it differs from `origin/main` of 2026-11-08 in 75-196 files and was not restored here: take it
from the bundle when rebuilding exactly. The packages were built from `/home/raahem/tt-metal` (see `../tt-model.yaml`
`source:`), whose branch was not kept.
