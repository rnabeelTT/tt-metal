# Original-card benchmarks on our packages

User request 2026-10-28: every benchmark the official Hugging Face card reports goes on our card, measured the way the
card measures it, next to the card's figure. Decisions (user, 2026-10-28):
- LLM judges: Claude agents in this session judge (no external APIs). Judge inputs are written as JSONL by the harness.
- Sets over ~3,000 items: fixed random sample of 1,000 (seed 1234; multilingual sets spread evenly over languages);
  the sample ids are saved so reruns use the same items. The card states the sample size and the error bar.
- Video and VM / emulator / 3D benchmarks: listed on the card as not supported by this package, not run.
- Earlier evals (scorecard, thinking off) are not reused: no card measured that way.
- LLM-judged benchmarks (user, 2026-10-29): kept where the benchmark is a headline one with no rule-scored
  alternative (HLE, BrowseComp, BrowseComp-zh, WideSearch, Seal-0, Claw-Eval, the judged image sets), with the judge
  named on the card; dropped where Claude would act inside the episode (TAU2 and VITA simulated users) and the
  low-visibility judged extras (MultiChallenge, AA-LCR, DeepPlanning, MAXIFE). Judging: Claude agents with the
  official judge prompt, schema-validated verdicts, then one audit pass over borderline "correct" verdicts.
- Skip rule (user, 2026-10-28): at most one or two benchmarks may be skipped, and only ones that are both very hard
  to run here and of little relevance to readers. The main, doable ones that users look at (SWE-bench, Terminal-Bench,
  GPQA, HLE, LiveCodeBench, MMLU-Pro, IFEval, HMMT, BrowseComp, TAU2, MMMU and the like) are always run.

Token budget (2026-10-29): Qwen's guidance for these checkpoints is 32,768 output tokens for most queries and 81,920
for hard reasoning. At 32,768, 56% of Ornith-1.5-122B's HLE answers and 20% of its SciCode steps were cut mid-reasoning
(scored as wrong), so HLE, SciCode, GPQA, LiveCodeBench, OJBench, HMMT and PolyMATH run at 81,920 (16 in flight).
Runs that already finished at 32,768 are extended with `tools/extend_truncated.py`: a truncated answer is continued
from its own reasoning up to 81,920 tokens (an exact equivalent of one generation at that budget), image items are
regenerated at 81,920, finished answers are kept.

Common settings unless a row says otherwise: served from the hub repo as shipped, p300x2, thinking on (the checkpoint's
default), the card's thinking-mode sampling (Ornith-1.5-122B and Qwen3.5-122B: T1.0/0.95/20 + presence_penalty 1.5 for
general tasks, T0.6/0.95/20 without penalty for precise coding = SciCode, LiveCodeBench, OJBench, FullStackBench,
Codeforces; Ornith-1.5-35B and Coder-Next: their generation_config), up to 32 requests in flight, max 32,768 generated tokens, only
the answer (`content`) scored. Runs: `~/ornith-fusion/runs/ref_<model>_<bench>`; queue runner `tools/evalq.sh` reads `runs/evalq.txt` (one task per line,
finished ones in `runs/evalq.done`). Harnesses live in `~/evals` (text2, SciCode, aider, ...; not in this repo).

Status: **queued** (harness ready, waiting for chips) · **setup** (harness being prepared) · **run** (in progress) ·
**staged** (ready; joins the queue after the quick evals) · **done** · **dropped** (user decision) · **n/s** (not supported by the package) · **open** (needs a decision or a service we do not have yet).

## Ornith-1.5-35B-A3B (card: ornith-ai/Ornith-1.5-35B-A3B; results averaged over 5 runs)

| Benchmark | Card | Ours | Status | Harness / note |
| --- | ---: | ---: | --- | --- |
| GPQA Diamond | 89.2 | | queued | lm-eval r1_gpqa_diamond, 198 x 5 runs, T1.0/0.95/20 |
| HLE (no tools) | 25.6 | | queued | text2 hle: 2,500 Qs (images included), official prompts; judged by Claude agents (card: Claude 4.6 Opus / o3-mini) |
| HLE (with tools) | 33.4 | | staged | 2,500 incl. images; CAIS tools protocol (search <=20 results, fetch 60k chars, full HLE blocklist, no-network Python 3.11 sandbox) |
| Terminal-Bench 2.1 (Terminus-2) | 67.8 | | staged | Harbor 0.25 + Terminus-2 json, T1.0/top_p 1.0, 128K, 4 h; 8 CPU per task (card 32), 1 run (card 5) |
| Terminal-Bench 2.1 (Claude Code) | 68.5 | | staged | Harbor + Claude Code 2.1.126 via /v1/messages; Ornith's shipped chat template (the cards' modification) is what we serve |
| SWE-bench Verified | 79 | | staged | OpenHands SDK, T1.0/top_p 0.95, 256K, no network, git history squashed; official swebench 5.0.2 evaluator |
| SWE-bench Pro | 59.6 | | staged | Pro v1 (731, as the cards); OpenHands; Scale evaluator; fix-commit refs removed |
| SWE-bench Multilingual | 71.4 | | staged | OpenHands, as Verified |
| DeepSWE | 22 | | staged | Pier 0.3.1 + Claude Code 2.1.126, deep-swe@435ee89e (113), T1.0/0.95, 256K, agent without network |
| Frontier-Bench v0.1 | 5.1 | | staged | terminal-bench@3.0.0: 70 of 74 tasks (4 need an H100); Terminus-2, TB 2.1 settings |
| NL2Repo | 46.2 | | n/s | skipped (user): per-task test images unpublished; card used 400K context (we serve 262K) |
| SWE Atlas - QnA | 39.8 | | staged | Harbor + mini-swe-agent 2.4.6, SWE Atlas @49e4af3 (incl. rubric fixes), T1.0/0.95, 128K, 5 runs; rubric judged by Claude |
| MCP-Atlas | 70.2 | | n/s | skipped (user, 2026-10-29): 16 of 36 MCP servers need SaaS accounts |
| Toolathlon-Verified | 48.7 | | open | official service offline since 2026-10-28 (GitHub issue #99); client ready (outbound only, no public endpoint) |
| WideSearch | 67.8 | | staged | 200 tasks, 50 turns, no context management (as the Qwen card); official scorer, Item-F1; Claude-judged cells |
| BrowseComp | 67.6 | | staged | full 1,266; SearXNG (DuckDuckGo/Yahoo/Mojeek) + fetch, benchmark-leak blocklist, 100 turns, context folding at 240k; Claude-judged |
| ClawEval | 72.5 | | staged | claw-eval@5680b8b (300 tasks), T0.6/0.95/20, 256K; grader judge by Claude; multi_turn user via the Claude relay; web via SearXNG |

## Ornith-1.5-122B (card: ornith-ai/Ornith-1.5-122B)

| Benchmark | Card | Ours | Status | Harness / note |
| --- | ---: | ---: | --- | --- |
| SciCode | 53.8 | 39.9 at 32k (truncated: 58/288) | rerun | official harness (inspect), test split, with background, subproblem pass@1 (AA protocol); rerun at 81,920; main-problem 13.8 at 32k |
| HLE (w/o tools) | 37.68 | | queued | rerun with the general-task preset (the first run used T0.6 without penalty: of the 339 sampled answers cut at 32,768 tokens, 317 were still cut at 81,920 and 16 finished); fixed random 1,000 of 2,500 (seed 1234, images included; user 2026-10-29: the full set at 81,920 tokens is ~40 h of decode), official prompts, 81,920 tokens (32k answers continued exactly); judged by Claude agents (card: Claude 4.6 Opus / o3-mini) |
| HLE (with tools) | 48.0 | | queued | the same fixed 1,000 of 2,500 (incl. images); CAIS tools protocol (search <=20 results, fetch 60k chars, full HLE blocklist, no-network Python 3.11 sandbox) |
| TerminalBench 2.1 | 81.2 | | queued | Harbor + Claude Code (the card's footnote); 8 CPU per task, 1 run |
| CyberGym | 80.0 | | n/s | skipped (user): 1,507 tasks with ~10 TB of images; level and harness unpublished |
| NL2Repo | 51.2 | | n/s | skipped (user): per-task test images unpublished; card used 400K context (we serve 262K) |
| SWE-bench Pro (no-hack) | 76.9 | | queued | Pro v1; OpenHands + don't-reward-hack notice; hack review of trajectories by Claude |
| SWE-bench Verified (non-hack) | 84.5 | | queued | OpenHands, no network, git squashed, don't-reward-hack notice |
| SWE-bench Multilingual (non-hack) | 76.4 | | queued | OpenHands, as Verified |
| WideSearch | 75.2 | | queued | 200 tasks, 50 turns, no context management (as the Qwen card); official scorer, Item-F1; Claude-judged cells |
| BrowseComp | 78.5 | | queued | full 1,266; SearXNG (DuckDuckGo/Yahoo/Mojeek) + fetch, benchmark-leak blocklist, 100 turns, context folding at 240k; Claude-judged |
| MCP-Atlas | 74.6 | | n/s | skipped (user, 2026-10-29): 16 of 36 MCP servers need SaaS accounts |
| Toolathlon-Verified | 70.8 | | open | official service offline since 2026-10-28 (GitHub issue #99); client ready (outbound only, no public endpoint) |
| ClawEval | 79.4 | | queued | claw-eval@5680b8b (300 tasks), T0.6/0.95/20, 256K; grader judge by Claude; multi_turn user via the Claude relay; web via SearXNG |

## Qwen3-Coder-Next (card: Qwen/Qwen3-Coder-Next; no thinking)

| Benchmark | Card | Ours | Status | Harness / note |
| --- | ---: | ---: | --- | --- |
| Aider (polyglot) | 66.2 | | queued | aider benchmark, 225 exercises, 2 tries, diff and whole (Qwen's format unpublished), T1.0/0.95/40 |
| SWE-Bench Verified (SWE-Agent) | 70.6 | | staged | SWE-agent 300 steps, T1.0/0.95/40 (the chart); plus OpenHands (tech-report column 71.3) |
| SWE-Bench Multilingual (SWE-Agent) | 62.8 | | staged | SWE-agent; plus OpenHands (tech report 64.3) |
| SWE-Bench Pro (SWE-Agent) | 44.3 | | staged | Pro v1, Scale's SWE-agent fork + Scale evaluator |
| Terminal-Bench 2.0 (Terminus-2 json) | 36.2 | | staged | Harbor + Terminus-2 json, 3 h (assumed), 8 CPU per task |

## Qwen3.5-122B-A10B (card: Qwen/Qwen3.5-122B-A10B)

Language:

| Benchmark | Card | Ours | Status | Harness / note |
| --- | ---: | ---: | --- | --- |
| MMLU-Pro | 86.7 | | queued | qtext: card MCQ prompt, 0-shot, 1,000 of 12,032 |
| MMLU-Redux | 94.0 | | queued | qtext: redux-2.0, 1,000 of 5,700 over 57 subjects |
| C-Eval | 91.9 | | queued | qtext: val split 1,346 (test labels hidden) |
| SuperGPQA | 67.1 | | queued | qtext: 1,000 of 26,529 |
| IFEval | 93.4 | | queued | lm-eval ifeval, all 541 |
| IFBench | 76.1 | | queued | text2 ifbench, 300, official run_eval.py |
| MultiChallenge | 61.5 | | dropped | dropped (user, 2026-10-29): low-visibility; LLM-judged |
| AA-LCR | 66.9 | | dropped | dropped (user, 2026-10-29): low-visibility; LLM-judged |
| LongBench v2 | 60.2 | | queued | official pred.py prompt, 503 |
| HLE w/ CoT | 25.3 | | queued | text2 hle: 2,500 Qs (images included), official prompts; judged by Claude agents (card: Claude 4.6 Opus / o3-mini) |
| GPQA Diamond | 86.6 | | queued | lm-eval r1_gpqa_diamond, 198 |
| HMMT Feb 25 | 91.4 | | queued | MathArena data, 4 samples/problem, 81,920 max tokens |
| HMMT Nov 25 | 90.3 | | queued | MathArena data, 4 samples/problem, 81,920 max tokens |
| SWE-bench Verified | 72.0 | | staged | OpenHands, 200K (Qwen's scaffold is internal: labelled 'OpenHands, not Qwen's scaffold') |
| Terminal Bench 2 | 49.4 | | staged | 2.0, Terminus-2, 3 h, T1.0/0.95/20, 80K max tokens, 256K |
| LiveCodeBench v6 | 78.9 | | queued | lm-eval, code_generation_lite test6 (175, 2025-01..04) |
| CodeForces | 2100 | | n/s | Qwen's own query set, not public: not reproducible |
| OJBench | 39.5 | | queued | 232 problems, C++, DMOJ judge, 65,536 max tokens |
| FullStackBench en / zh | 62.6 / 58.7 | | queued | SandboxFusion, 1,687 each |
| BFCL-V4 | 72.2 | | queued | official bfcl_eval; web_search part via SearXNG |
| TAU2-Bench | 79.5 | | dropped | dropped (user, 2026-10-29): Claude would play the simulated customer (~13k turns), shaping the episodes, not just grading them |
| VITA-Bench | 33.6 | | dropped | dropped (user, 2026-10-29): Claude would play the simulated user and the rubric judge |
| DeepPlanning | 24.1 | | dropped | dropped (user, 2026-10-29): low-visibility; travel plans need an LLM conversion step |
| HLE w/ tool | 47.5 | | staged | 2,500 incl. images; CAIS tools protocol (search, fetch, full HLE blocklist, no-network Python sandbox) |
| Browsecomp | 63.8 | | staged | full 1,266; SearXNG + fetch, benchmark-leak blocklist, 100 turns, folding at 240k; Claude-judged |
| Browsecomp-zh | 69.9 | | staged | 289; Chinese engines (Sogou/360), robots.txt blocks baike/zhihu; Claude-judged |
| WideSearch | 60.5 | | staged | 200 tasks, 50 turns, no context management (as the Qwen card); official scorer, Item-F1; Claude-judged cells |
| Seal-0 | 44.1 | | staged | 111; official SealQA grader template; answers are time-sensitive vs today's web; Claude-judged |
| MMMLU | 86.7 | | queued | qtext: 14 languages (Yoruba dropped as in the Qwen3 report), 1,000 |
| MMLU-ProX | 82.2 | | queued | qtext: 29 languages, 1,000 |
| NOVA-63 | 58.6 | | n/s | dataset not public (HF repo holds only a README) |
| INCLUDE | 82.8 | | queued | qtext: include-base-44, 1,000 |
| Global PIQA | 88.4 | | queued | qtext: official generation prompt, 136 languages, 1,000 of 13,600 |
| PolyMATH | 68.9 | | queued | qtext: official scorer, DW-ACC over 18 languages, 1,000 of 9,000, 81,920 max tokens |
| WMT24++ | 78.3 | | queued | qtext: 1,000 segments over 55 pairs; Qwen's harder subset is not public; XCOMET-XXL licence needed to score |
| MAXIFE | 87.9 | | dropped | dropped (user, 2026-10-29): low-visibility; the card's metric includes LLM-judged instructions |

Vision language (images; up to 4 per request, resized to at most 4 MP by the server):

| Benchmark | Card | Ours | Status | Harness / note |
| --- | ---: | ---: | --- | --- |
| MMMU | 83.9 | | queued | 900 val, Qwen's script settings (T1.0, 40,960 tokens, pixel range) |
| MMMU-Pro | 76.9 | | queued | vision harness (VLMEvalKit / official scorers), Qwen3-VL report prompts; judged parts by Claude |
| MathVision | 86.2 | | queued | 1,000 of 3,040, fixed boxed prompt, Qwen's script settings (T1.0, 40,960 tokens) |
| Mathvista(mini) | 87.4 | | queued | vision harness (VLMEvalKit / official scorers), Qwen3-VL report prompts; judged parts by Claude |
| DynaMath | 85.9 | | queued | vision harness (VLMEvalKit / official scorers), Qwen3-VL report prompts; judged parts by Claude |
| ZEROBench / ZEROBench_sub | 9 / 36.2 | | queued | vision harness (VLMEvalKit / official scorers), Qwen3-VL report prompts; judged parts by Claude |
| VlmsAreBlind | 96.7 | | queued | vision harness (VLMEvalKit / official scorers), Qwen3-VL report prompts; judged parts by Claude |
| BabyVision (with / without CI) | 40.2 / 34.5 | | queued | without CI: 388 x 3 samples, judged every item; with CI skipped (user, 2026-10-29): no published code-interpreter harness |
| RealWorldQA | 85.1 | | queued | vision harness (VLMEvalKit / official scorers), Qwen3-VL report prompts; judged parts by Claude |
| MMStar | 82.9 | | queued | vision harness (VLMEvalKit / official scorers), Qwen3-VL report prompts; judged parts by Claude |
| MMBench EN-DEV-v1.1 | 92.8 | | queued | vision harness (VLMEvalKit / official scorers), Qwen3-VL report prompts; judged parts by Claude |
| SimpleVQA | 61.7 | | queued | vision harness (VLMEvalKit / official scorers), Qwen3-VL report prompts; judged parts by Claude |
| HallusionBench | 67.6 | | queued | vision harness (VLMEvalKit / official scorers), Qwen3-VL report prompts; judged parts by Claude |
| OmniDocBench1.5 | 89.8 | | queued | 1,355 pages, official v1_5 evaluator incl. CDM (docker) |
| CharXiv(RQ) | 77.2 | | queued | vision harness (VLMEvalKit / official scorers), Qwen3-VL report prompts; judged parts by Claude |
| MMLongBench-Doc | 59.0 | | queued | up to 120 pages tiled into <=4 images of <=4 MP: package limit, not comparable to the card |
| CC-OCR | 81.8 | | queued | vision harness (VLMEvalKit / official scorers), Qwen3-VL report prompts; judged parts by Claude |
| AI2D_TEST | 93.3 | | queued | vision harness (VLMEvalKit / official scorers), Qwen3-VL report prompts; judged parts by Claude |
| OCRBench | 92.1 | | queued | vision harness (VLMEvalKit / official scorers), Qwen3-VL report prompts; judged parts by Claude |
| ERQA | 62.0 | | queued | vision harness (VLMEvalKit / official scorers), Qwen3-VL report prompts; judged parts by Claude |
| CountBench | 97.0 | | queued | vision harness (VLMEvalKit / official scorers), Qwen3-VL report prompts; judged parts by Claude |
| RefCOCO(avg) | 91.3 | | queued | vision harness (VLMEvalKit / official scorers), Qwen3-VL report prompts; judged parts by Claude |
| ODInW13 | 44.5 | | queued | vision harness (VLMEvalKit / official scorers), Qwen3-VL report prompts; judged parts by Claude |
| EmbSpatialBench | 83.9 | | queued | vision harness (VLMEvalKit / official scorers), Qwen3-VL report prompts; judged parts by Claude |
| RefSpatialBench | 69.3 | | queued | vision harness (VLMEvalKit / official scorers), Qwen3-VL report prompts; judged parts by Claude |
| LingoQA | 80.8 | | queued | vision harness (VLMEvalKit / official scorers), Qwen3-VL report prompts; judged parts by Claude |
| ScreenSpot Pro | 70.4 | | queued | 1,581; 48% of screenshots exceed 4 MP and are downscaled (package limit) |
| TIR-Bench (with / without CI) | 53.2 / 42.5 | | queued | without CI: 1,215, official extraction judged; with CI skipped (user, 2026-10-29): no published code-interpreter harness |
| V* (with / without CI) | 93.2 / 90.1 | | queued | without CI: 191; with CI skipped (user, 2026-10-29): no published code-interpreter harness |
| SLAKE | 81.6 | | queued | vision harness (VLMEvalKit / official scorers), Qwen3-VL report prompts; judged parts by Claude |
| PMC-VQA | 63.3 | | queued | vision harness (VLMEvalKit / official scorers), Qwen3-VL report prompts; judged parts by Claude |
| MedXpertQA-MM | 67.3 | | queued | vision harness (VLMEvalKit / official scorers), Qwen3-VL report prompts; judged parts by Claude |
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
