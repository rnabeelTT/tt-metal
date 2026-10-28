# Original-card benchmarks on our packages

User request 2026-10-28: every benchmark the official Hugging Face card reports goes on our card, measured the way the
card measures it, next to the card's figure. Decisions (user, 2026-10-28):
- LLM judges: Claude agents in this session judge (no external APIs). Judge inputs are written as JSONL by the harness.
- Sets over ~3,000 items: fixed random sample of 1,000 (seed 1234; multilingual sets spread evenly over languages);
  the sample ids are saved so reruns use the same items. The card states the sample size and the error bar.
- Video and VM / emulator / 3D benchmarks: listed on the card as not supported by this package, not run.
- Earlier evals (scorecard, thinking off) are not reused: no card measured that way.

Common settings unless a row says otherwise: served from the hub repo as shipped, p300x2, thinking on (the checkpoint's
default), the checkpoint's generation_config sampling, up to 32 requests in flight, max 32,768 generated tokens, only
the answer (`content`) scored. Runs: `~/ornith-fusion/runs/ref_*`; chains `runs/ref_evals.sh`, `runs/ref_evals2.sh`.

Status: **queued** (harness ready, waiting for chips) · **setup** (harness being prepared) · **run** (in progress) ·
**done** · **n/s** (not supported by the package) · **open** (needs a decision or a service we do not have yet).

## Ornith-1.5-35B-A3B (card: ornith-ai/Ornith-1.5-35B-A3B; results averaged over 5 runs)

| Benchmark | Card | Ours | Status | Harness / note |
| --- | ---: | ---: | --- | --- |
| GPQA Diamond | 89.2 | | queued | lm-eval r1_gpqa_diamond, 198 x 5 runs, T1.0/0.95/20 |
| HLE (no tools) | 25.6 | | setup | Claude-judged (card: Claude 4.6 Opus judge) |
| HLE (with tools) | 33.4 | | open | needs search / code tools |
| Terminal-Bench 2.1 (Terminus-2) | 67.8 | | setup | Harbor/Terminus-2; card: 32 CPU / 48 GB per task |
| Terminal-Bench 2.1 (Claude Code) | 68.5 | | setup | |
| SWE-bench Verified | 79 | | setup | OpenHands per card |
| SWE-bench Pro | 59.6 | | setup | |
| SWE-bench Multilingual | 71.4 | | setup | |
| DeepSWE | 22 | | setup | feasibility check |
| Frontier-Bench v0.1 | 5.1 | | setup | feasibility check |
| NL2Repo | 46.2 | | setup | feasibility check |
| SWE Atlas - QnA | 39.8 | | setup | feasibility check |
| MCP-Atlas | 70.2 | | setup | feasibility check (Claude judge) |
| Toolathlon-Verified | 48.7 | | setup | feasibility check (official service) |
| WideSearch | 67.8 | | open | needs a search backend |
| BrowseComp | 67.6 | | open | needs a search backend |
| ClawEval | 72.5 | | setup | feasibility check |

## Ornith-1.5-122B (card: ornith-ai/Ornith-1.5-122B)

| Benchmark | Card | Ours | Status | Harness / note |
| --- | ---: | ---: | --- | --- |
| SciCode | 53.8 | | queued | official harness (inspect), test split, with background, subproblem pass@1 (AA protocol) |
| HLE (w/o tools) | 37.68 | | setup | Claude-judged |
| HLE (with tools) | 48.0 | | open | needs search / code tools |
| TerminalBench 2.1 | 81.2 | | setup | |
| CyberGym | 80.0 | | setup | feasibility check |
| NL2Repo | 51.2 | | setup | feasibility check |
| SWE-bench Pro (no-hack) | 76.9 | | setup | OpenHands per card |
| SWE-bench Verified (non-hack) | 84.5 | | setup | |
| SWE-bench Multilingual (non-hack) | 76.4 | | setup | |
| WideSearch | 75.2 | | open | needs a search backend |
| BrowseComp | 78.5 | | open | needs a search backend |
| MCP-Atlas | 74.6 | | setup | feasibility check |
| Toolathlon-Verified | 70.8 | | setup | feasibility check |
| ClawEval | 79.4 | | setup | feasibility check |

## Qwen3-Coder-Next (card: Qwen/Qwen3-Coder-Next; no thinking)

| Benchmark | Card | Ours | Status | Harness / note |
| --- | ---: | ---: | --- | --- |
| Aider (polyglot) | 66.2 | | queued | aider benchmark, 225 exercises, 2 tries, diff and whole (Qwen's format unpublished), T1.0/0.95/40 |
| SWE-Bench Verified (SWE-Agent) | 70.6 | | setup | |
| SWE-Bench Multilingual (SWE-Agent) | 62.8 | | setup | |
| SWE-Bench Pro (SWE-Agent) | 44.3 | | setup | |
| Terminal-Bench 2.0 (Terminus-2 json) | 36.2 | | setup | |

## Qwen3.5-122B-A10B (card: Qwen/Qwen3.5-122B-A10B)

Language:

| Benchmark | Card | Ours | Status | Harness / note |
| --- | ---: | ---: | --- | --- |
| MMLU-Pro | 86.7 | | setup | 1,000 sample |
| MMLU-Redux | 94.0 | | setup | |
| C-Eval | 91.9 | | setup | test labels are not public |
| SuperGPQA | 67.1 | | setup | 1,000 sample |
| IFEval | 93.4 | | queued | lm-eval ifeval, all 541 |
| IFBench | 76.1 | | setup | |
| MultiChallenge | 61.5 | | setup | Claude-judged |
| AA-LCR | 66.9 | | setup | Claude-judged |
| LongBench v2 | 60.2 | | setup | |
| HLE w/ CoT | 25.3 | | setup | Claude-judged |
| GPQA Diamond | 86.6 | | queued | lm-eval r1_gpqa_diamond, 198 |
| HMMT Feb 25 | 91.4 | | setup | |
| HMMT Nov 25 | 90.3 | | setup | |
| SWE-bench Verified | 72.0 | | setup | |
| Terminal Bench 2 | 49.4 | | setup | |
| LiveCodeBench v6 | 78.9 | | queued | lm-eval, code_generation_lite test6 (175, 2025-01..04) |
| CodeForces | 2100 | | open | Qwen's own query set (not public) |
| OJBench | 39.5 | | setup | |
| FullStackBench en / zh | 62.6 / 58.7 | | setup | |
| BFCL-V4 | 72.2 | | setup | |
| TAU2-Bench | 79.5 | | setup | simulated user |
| VITA-Bench | 33.6 | | setup | simulated user |
| DeepPlanning | 24.1 | | setup | |
| HLE w/ tool | 47.5 | | open | needs search / code tools |
| Browsecomp | 63.8 | | open | needs a search backend |
| Browsecomp-zh | 69.9 | | open | needs a search backend |
| WideSearch | 60.5 | | open | needs a search backend |
| Seal-0 | 44.1 | | open | needs a search backend |
| MMMLU | 86.7 | | setup | 1,000 sample over 14 languages |
| MMLU-ProX | 82.2 | | setup | 1,000 sample over 29 languages |
| NOVA-63 | 58.6 | | setup | |
| INCLUDE | 82.8 | | setup | 1,000 sample |
| Global PIQA | 88.4 | | setup | |
| PolyMATH | 68.9 | | setup | |
| WMT24++ | 78.3 | | setup | XCOMET-XXL scoring, no GPU here |
| MAXIFE | 87.9 | | setup | |

Vision language (images; up to 4 per request, resized to at most 4 MP by the server):

| Benchmark | Card | Ours | Status | Harness / note |
| --- | ---: | ---: | --- | --- |
| MMMU | 83.9 | | setup | |
| MMMU-Pro | 76.9 | | setup | |
| MathVision | 86.2 | | setup | fixed \boxed{} prompt |
| Mathvista(mini) | 87.4 | | setup | |
| DynaMath | 85.9 | | setup | |
| ZEROBench / ZEROBench_sub | 9 / 36.2 | | setup | |
| VlmsAreBlind | 96.7 | | setup | |
| BabyVision (with / without CI) | 40.2 / 34.5 | | setup | without CI first |
| RealWorldQA | 85.1 | | setup | |
| MMStar | 82.9 | | setup | |
| MMBench EN-DEV-v1.1 | 92.8 | | setup | |
| SimpleVQA | 61.7 | | setup | judged |
| HallusionBench | 67.6 | | setup | |
| OmniDocBench1.5 | 89.8 | | setup | |
| CharXiv(RQ) | 77.2 | | setup | |
| MMLongBench-Doc | 59.0 | | setup | many pages per document vs 4 images per request |
| CC-OCR | 81.8 | | setup | |
| AI2D_TEST | 93.3 | | setup | |
| OCRBench | 92.1 | | setup | |
| ERQA | 62.0 | | setup | |
| CountBench | 97.0 | | setup | |
| RefCOCO(avg) | 91.3 | | setup | grounding, scored from text boxes |
| ODInW13 | 44.5 | | setup | |
| EmbSpatialBench | 83.9 | | setup | |
| RefSpatialBench | 69.3 | | setup | |
| LingoQA | 80.8 | | setup | |
| ScreenSpot Pro | 70.4 | | setup | high-resolution screenshots vs the 4 MP resize |
| TIR-Bench (with / without CI) | 53.2 / 42.5 | | setup | |
| V* (with / without CI) | 93.2 / 90.1 | | setup | |
| SLAKE | 81.6 | | setup | |
| PMC-VQA | 63.3 | | setup | |
| MedXpertQA-MM | 67.3 | | setup | |
| Hypersim | 12.7 | | n/s | 3D |
| SUNRGBD | 36.2 | | n/s | 3D |
| Nuscene | 15.4 | | n/s | 3D |
| VideoMME (w sub. / w/o sub.) | 87.3 / 83.9 | | n/s | video input not supported by this package |
| VideoMMMU | 82.0 | | n/s | video |
| MLVU | 87.3 | | n/s | video |
| MVBench | 76.6 | | n/s | video |
| LVBench | 74.4 | | n/s | video |
| MMVU | 74.7 | | n/s | video |
| OSWorld-Verified | 58.0 | | n/s | needs a VM harness |
| AndroidWorld | 66.4 | | n/s | needs an emulator harness |
