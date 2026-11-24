# Published models (raahemnabeel on Hugging Face)

This branch, `raahem/published-models`, holds every published fast and hybrid model in one tree, so a new bring-up
can start here and read the closest existing port. Two other branches stay separate:

- `hybrid-moe-starter`: the clean starting point for a new hybrid (CPU + TT expert) model
  (`models/experimental/hybrid_moe/ADD_A_MODEL.md`).
- `raahem/cohost-ornith-qwen3vl`: the three co-hosted LLM + Qwen3-VL packages.

**Exact published code lives in the tags, not necessarily at this branch's tip.** The models share code (the hybrid
overlay in `models/experimental/hybrid_moe/`, the qwen36 MoE / attention modules, the `moe_sorted_dispatch` kernel)
that diverged per model. Merging them kept the newest version of each conflicting file (each merge commit lists its
choices), so an older model may not serve unchanged from the tip. To rebuild a package exactly, check out its tag.
Each tag is the newest commit whose files match every file in that bundle's `code/` folder (Python and the bundled
kernel sources; the bundles do not carry the tt_metal / ttnn C++, which was not compared).

Agent skills (picked up from `.claude/skills/`, symlinks into `models/experimental/hybrid_moe/skills/`):
`blackhole-decode-perf` (decode-speed review once a new model runs with reference quality) and `hybrid-moe-bringup`.

Since 2026-11-24 (the Hy4-flash merge) this branch is on upstream tt-metal 0fe57024bca (2026-11-18); the models published before then were built on upstream 05875866f20 (2026-09-09), so their tags are the way to rebuild them.

The per-model branches these came from were deleted on 2026-11-22 and 2026-11-24; each tip is kept as the tag `archive/<branch>`
(e.g. `archive/ornith_opt`, `archive/raahem/inkling-small-hybrid`), so old branch names in cards and handoffs still
resolve with `git show archive/<branch>`.

| Package (Hub) | Published | Tag (exact code) | Package yaml | Source branch |
|---|---|---|---|---|
| [hy4-flash](https://huggingface.co/raahemnabeel/hy4-flash) | 2026-11-24 | `published/hy4-flash` | `models/autoports/tencent_hy4_flash/tt-model.yaml` | `raahem/hy4-flash-hybrid` |
| [inkling-2-mini](https://huggingface.co/raahemnabeel/inkling-2-mini) | 2026-11-24 | `published/inkling-2-mini` | `models/autoports/thinkingmachines_inkling_2_mini/tt-model.yaml` | `raahem/inkling-2-mini` |
| [inkling-small-hybrid-qb2](https://huggingface.co/raahemnabeel/inkling-small-hybrid-qb2) | 2026-11-21 | `published/inkling-small-hybrid-qb2` | `models/autoports/thinkingmachines_inkling_small/tt-model-hybrid.yaml` | `raahem/inkling-small-hybrid` |
| [minimax-m2.7-hybrid-qb2](https://huggingface.co/raahemnabeel/minimax-m2.7-hybrid-qb2) | 2026-11-18 | `published/minimax-m2.7-hybrid-qb2` | `models/autoports/minimaxai_minimax_m2_7/tt-model-hybrid.yaml` | `raahem/qwen35-397b-hybrid-moe` |
| [ornith-1.5-397b-hybrid-qb2](https://huggingface.co/raahemnabeel/ornith-1.5-397b-hybrid-qb2) | 2026-11-16 | `published/ornith-1.5-397b-hybrid-qb2` | `models/autoports/qwen_qwen3_5_122b_a10b/tt-model-o397-hybrid.yaml` | `ornith397-hybrid` |
| [ornith-1.5-35b-a3b](https://huggingface.co/raahemnabeel/ornith-1.5-35b-a3b) | 2026-11-10 | `published/ornith-1.5-35b-a3b` | `models/autoports/ornith_ai_ornith_1_5_35b_a3b/tt-model-fast.yaml` | `ornith_opt` |
| [qwen3-5-122b-a10b](https://huggingface.co/raahemnabeel/qwen3-5-122b-a10b) | 2026-11-07 | `published/qwen3-5-122b-a10b` | `models/autoports/qwen_qwen3_5_122b_a10b/tt-model-fast.yaml` | `q122_opt` (in `ornith397-hybrid`) |
| [ornith-1.5-122b](https://huggingface.co/raahemnabeel/ornith-1.5-122b) | 2026-11-07 | `published/ornith-1.5-122b` | `models/autoports/qwen_qwen3_5_122b_a10b/tt-model-ornith122.yaml` | `q122_opt` (in `ornith397-hybrid`) |
| [qwen3-coder-next](https://huggingface.co/raahemnabeel/qwen3-coder-next) | 2026-11-07 | `published/qwen3-coder-next` | `models/autoports/qwen_qwen3_coder_next/tt-model-fast.yaml` | `coder_next_opt` |
| [mira-sft-35b-a3b](https://huggingface.co/raahemnabeel/mira-sft-35b-a3b) | 2026-11-04 | `published/mira-sft-35b-a3b` | `models/autoports/mira_mmdeepresearch_mira_sft_35b_a3b/tt-model.yaml` | `ornith_fusion` (in `ornith_opt`, `fusion_all`) |
| [quest-35b-rl](https://huggingface.co/raahemnabeel/quest-35b-rl) | 2026-11-03 | `published/quest-35b-rl` | `models/autoports/osunlp_quest_35b_rl/tt-model.yaml` | `ornith_fusion` |
| [tongyi-deepresearch-30b-a3b](https://huggingface.co/raahemnabeel/tongyi-deepresearch-30b-a3b) | 2026-11-04 | `published/tongyi-deepresearch-30b-a3b` | `models/autoports/alibaba_nlp_tongyi_deepresearch_30b_a3b/tt-model.yaml` | `tongyi_deepresearch` (in `fusion_all`) |
| [mirothinker-1.7-mini](https://huggingface.co/raahemnabeel/mirothinker-1.7-mini) | 2026-11-03 | `published/mirothinker-1.7-mini` | `models/autoports/miromind_ai_mirothinker_1_7_mini/tt-model.yaml` | `mirothinker_mini` (in `fusion_all`) |
| [qwen3-coder-30b-a3b](https://huggingface.co/raahemnabeel/qwen3-coder-30b-a3b) | 2026-10-27 | `published/qwen3-coder-30b-a3b` | `models/demos/blackhole/qwen3_coder_30b_a3b/tt-model-fast.yaml` | `c30_fusion` (in `fusion_all`) |
| [qwen3-vl-reranker-8b](https://huggingface.co/raahemnabeel/qwen3-vl-reranker-8b), [qwen3-vl-embedding-8b](https://huggingface.co/raahemnabeel/qwen3-vl-embedding-8b), [qwen3-reranker-4b](https://huggingface.co/raahemnabeel/qwen3-reranker-4b) | 2026-11-02 | `published/qwen3-vl-reranker-8b` (recovered, autoport only) | `models/autoports/qwen_qwen3_vl_reranker_8b/tt-model.yaml` | none: recovered from the bundles, see `models/autoports/qwen_qwen3_vl_reranker_8b/recovered/README.md` |

Not here on purpose: the GLM packages (glm-4.5-air, glm-5.3-flash), the `-v1` original bundles (private), unpublished
work (`raahem/hy4-flash-hybrid`, `raahem/m27-rows-sdpa`, `qwen4_prep`).

## Adding a model

Branch off this branch (or `hybrid-moe-starter` for a hybrid model), and when the model is published: merge it back
here, tag the commit the bundle was built from as `published/<package name>`, and add a row above.
