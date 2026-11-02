# Image benchmarks of the Qwen3.5-122B-A10B card — local harness

Scores an OpenAI-compatible server (vLLM, thinking on, answer in `content`) on the image
benchmarks of the Qwen3.5-122B-A10B model card "Vision Language" table, following Qwen's
published protocol as closely as it is published. Entry point: `~/evals/vision_run.sh`.

```
~/evals/vision_run.sh list                                    # benches, card figure, requests per run
~/evals/vision_run.sh <bench> <label> <base_url> <model_id> <concurrency>
  e.g. ~/evals/vision_run.sh mmmu ornith122b http://localhost:20000/v1 Qwen/Qwen3.5-122B-A10B 32
~/evals/vision_judge_ingest.sh <bench> <label> [judgments.jsonl] # after judging: official metric, status=final
~/evals/vision/selftest.sh <bench> [n]                        # offline: stub server gold (~100%) / garbage (~chance)
```
Env: `MAX_TOKENS` (32768) `TEMPERATURE` (0.6) `TOP_P` (0.95) `TOP_K` (20) `PRESENCE_PENALTY` (unset)
`LIMIT` (first N selected items per sub-set; smoke tests) `REQUEST_TIMEOUT` (14400 s) `PHASE` (all|infer|score)
`KEEP_REASONING=1` (store reasoning text too) `OMNIDOC_CDM=docker` (formula CDM via the official image).

Output `~/ornith-fusion/runs/<label>/<bench>/`: `responses.jsonl` (resumable cache — rerun the same
command to retry failures), `sample_ids.json`, `run_config.json`, `score/` (official scorer artefacts),
`judge/` (judge_inputs.jsonl / judgments.jsonl), `result.json`, `run.log`.
Last line: `RESULT <bench> <metric>=<value> n= errors= truncated= empty= judge_pending= status= card=`;
when an LLM judge is part of the official protocol the line before it is `JUDGE_PENDING <path>`.

## How it works
* `vtt/run.py` loads data (VLMEvalKit dataset classes, or `vtt/extra_benches.py` loaders), fixes the
  subset (`samples/<bench>.json`; > ~3000 items → 1000, seed 1234, stratified/grouped where the metric
  needs it), renders one user turn per item (images first, then text — Qwen3-VL convention) with the
  prompt from the Qwen3-VL tech report Appendix B, sends it with up to `concurrency` requests in flight,
  and scores only `message.content` (reasoning is ignored, as Qwen's own scripts do with
  `split("</think>")[-1]`).
* Images: the protocol's `[min_pixels, max_pixels]` is applied client-side (qwen_vl_utils semantics,
  factor 32) — e.g. MMMU upscales small images to ~1 MP as Qwen's script does; otherwise originals are
  sent and the server's own ≤ 4 MP cap applies (counted in `stats.server_downscaled_images`).
  Requests with > 4 images are packed into ≤ 4 labelled composites (`stats.items_images_tiled`).
* Scoring calls the official scorer code (VLMEvalKit `dataset.evaluate`, Qwen's ODinW code,
  lmms-eval SLAKE, Lingo-Judge, TIR-Bench `calculate_score.py`, OmniDocBench v1_5 evaluator).
* LLM judges: no external API is ever called. Every `build_judge`/OpenAI client inside VLMEvalKit is
  replaced by a deferred judge (`vtt/judge.py`): rule-based extraction runs first exactly as upstream;
  prompts that still need the judge are written verbatim to `judge/judge_inputs.jsonl`
  (`id, bench, judge_model_in_official_protocol, judge_prompt, judge_images, expected_verdict_format,
  item_id, question, reference, model_answer` — item fields only when unambiguously matched).
  Write `{"id": ..., "verdict": "<raw judge reply>"}` lines, run `vision_judge_ingest.sh`; the official
  code then consumes the verdicts and computes the metric. Multi-stage judges just produce new pending
  prompts on the next pass. API env vars are pointed at a dead local port as a backstop.

## Data (`LMUData/` VLMEvalKit TSVs, `data/` others) — ~27 GB total after clean-up
VLMEvalKit TSVs were fetched from the bucket origin `opencompass.oss-cn-shanghai.aliyuncs.com`
(the `opencompass.openxlab.space` CDN currently serves an expired certificate) and every file was
verified against VLMEvalKit's pinned `DATASET_MD5` (`vtt/fetch_vlmeval.py`). Pinned sources:
VLMEvalKit e6fcedaf (2026-10-28), lmms-eval 6f00355, Qwen3-VL repo (evaluation/), GLIPModel/GLIP@66ee3ae,
OmniDocBench HF v1_5@91fe284 + GitHub branch v1_5@59b103c, TIR-Bench GitHub HEAD + HF Agents-X/TIR-Bench,
wayveai/LingoQA + wayveai/Lingo-Judge, BoKelvin/SLAKE, RadGenome/PMC-VQA (test_clean images only, ranged reads).

## Known VLMEvalKit issues patched at runtime (kept out of the VLMEvalKit tree)
* `VLMBlind.evaluate` compares `str` to the int64 `answers` its TSV loader yields → all numeric items 0.
  Answers are cast to str before scoring.
* `Point2DParser.log_hint` references an undefined `cls.logger` → every RefSpatial parse fails → 0%.
  A logger is attached.
* `build_dataset('ERQA')` / `('RefSpatial')` resolve to other classes than the TSVs fetched; the EASI
  classes (`ERQABench`, `RefSpatialBench`) are instantiated explicitly.

See the per-benchmark notes in `vtt/benches.py` / `vtt/extra_benches.py` (`notes`), echoed into
`run_config.json` of every run.

## "with CI" (code interpreter) variants — not implemented
The card reports BabyVision, TIR-Bench and V* as "with CI / without CI"; only the without-CI variants are
wired here. Qwen publishes no CI harness for Qwen3.5 (the Qwen3-VL report's tool setting, Appendix B.9,
is a single `image_zoom_in_tool(bbox_2d, label, img_idx)` crop tool, not a code interpreter; the TIR-Bench
paper's "with tool" baseline is OpenAI's hosted code interpreter). A with-CI run would need:
1. the server started with tool calling enabled (vLLM `--enable-auto-tool-choice` + the Qwen3.5 tool
   parser) and a `python`/`code_interpreter` function in `tools`;
2. a sandbox (e.g. a throw-away docker container with python + pillow/numpy/opencv, no network) that
   receives the input images as files, executes each tool call, and returns stdout plus any images the
   code saves;
3. a multi-turn loop (cap ~10-20 turns) that appends tool results — returned images as new `image_url`
   parts — and drops `reasoning` from history (card best practice 4). The 4-images-per-request limit
   means older tool images must be dropped or tiled;
4. the same scorers on the final `content`.
