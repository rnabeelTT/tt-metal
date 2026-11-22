# Accuracy

| eval | metric | this package | HF reference | size |
| --- | --- | ---: | ---: | --- |
| Fixed pair set: 340 pairs in 41 query groups (English, multilingual, code, images, text + image, video, 2K-10K-token documents) | Pearson r of the score | 0.9991 | 1 (HF fp32) | 340 pairs |
| | mean / max abs. score difference (logits) | 0.107 / 0.39 | 0 | |
| | same top-1 document per group | 41/41 | | |
| | same order for pairs HF separates by >= 0.5 | 100.0% | | 949 pairs |
| MTEB reranking, English (SciDocsRR + AskUbuntuDupQuestions subset) | NDCG@10 | 94.89 | 95.01 | 60 groups, 600 pairs |
| MIRACL reranking, de / ja / ar subset | NDCG@10 | 97.84 | 97.85 | 45 groups, 450 pairs |
| ViDoRe v3 (HR + computer science), document page images | NDCG@5 | 97.89 | 96.97 | 40 groups, 320 pairs |
| MMEB-v2 MSCOCO text-to-image | Hit@1 | 100.00 | 100.00 | 40 groups, 320 pairs |
| MMEB-v2 MSCOCO image-to-text | Hit@1 | 100.00 | 100.00 | 40 groups, 320 pairs |
| 64 s 720p video (7,399 tokens), query "cars and buses driving along a city road" | score | -0.637 | -0.672 | 1 pair |
| 64 s 720p video (7,396 tokens), query "a recipe for chocolate cake" | score | -7.125 | -7.023 | 1 pair |

Accuracy: the HF reference is the transformers implementation on CPU with the official reranker's prompt and preprocessing, on exactly the same inputs (fp32 for the fixed pair set, bf16 elsewhere). The benchmark rows are fixed subsets with short candidate lists, scored with NDCG on exponential gain: they show parity with HF, not leaderboard numbers. The MMEB subsets (1 positive + 7 random candidates) are saturated and only a sanity check.

Precision policy: `p1_mlp_lofi` (selected) against `p0_bfp8_hifi2` on the fixed set gives Pearson 0.99915 vs 0.99954, mean |d| 0.1067 vs 0.0997, top-1 41/41 for both, and order agreement at margin 0.5 of 949/949 vs 949/949. p1 is 10-20% faster on text (see performance.md). Raw rows are in raw/.
