# Qwen3.5-122B-A10B — TTI release (stage 11)

`tt-inference-server` v0.22.0 release workflow run as a client against the autoport vLLM server (external-server topology,
no Docker) on this p300x2 host, ci-nightly configuration. `RUN_NOTES.md` is the handoff record (server mode, host, versions,
commands, exit codes, autoport implementation check, recovery); `tti_local_integration.patch` is the local TTI change that
lets v0.22.0 select the autoport (impl + dev catalog template + eval config + spec-test matrix).

**Stage-11 status: `release-readiness-ci-subset-pass`** — the TTI release workflow (ci-nightly) passes against the autoport
with the catalog status FUNCTIONAL and nothing waived (run 11, 2026-09-29): Acceptance PASS, Evals 2 / 2 graded against a
same-setting canonical reference (the bf16 HF checkpoint answering the identical requests on CPU: device 92.66 vs HF 89.91 on
IFEval; 77.5 vs 80.0 on GPQA-Diamond, a pass by one item under TTI's sample-count rule, with 18 device and 13 HF answers of the
40 cut at the 4096-token budget on both sides), Spec Tests 22 / 22, 23 benchmark rows (no perf targets exist, so ungraded). The
model card's published numbers are thinking-mode scores; the served non-thinking configuration is 0.992 / 0.895 of them,
and the full-set non-thinking scores (87.99 / 68.69) have no same-setting reference — hence the CI-subset qualifier.
Details, every run and the row classification: `RUN_NOTES.md` ("Stage-11 status line", runs 5-11).

| item | result | artifact |
|---|---|---|
| external-server smoke | `run.py --workflow benchmarks --limit-samples-mode smoke-test --disable-trace-capture`: exit 0, completed 8 / failed 0 (16 / 4 tokens x 8) | `smoke/`, `run_specs/runtime_model_spec_smoke.json` |
| release run 2 (ci-nightly, status EXPERIMENTAL) | exit 0, "Acceptance PASS (0 blockers)" under EXPERIMENTAL (accuracy not enforced at that status: the gpqa FAIL is informational there); evals 1 passed / 1 failed; 23 benchmark rows (ungraded, no perf targets); spec tests no-op (suite matrix, patched afterwards) | `reports_output/release/report_*.md`, `reports_output/release/data/`, `run_specs/runtime_model_spec_release.json`, `logs/tti-release.log` |
| release run 10 (ci-nightly, FUNCTIONAL, same-setting references) | exit 0, Acceptance PASS — SUPERSEDED: its gpqa reference (60.0) was scored with the wrong lm-eval filter (RUN_NOTES run 8b) | `reports_output/release10/` |
| **release run 11 (ci-nightly, status FUNCTIONAL, corrected references) — the report of record** | **exit 0, Acceptance PASS, Evals 2 / 2 PASS (ifeval 92.66 vs HF 89.91 = 1.031; gpqa 77.5 vs HF 80.0 = 0.969, one item above TTI's sample-count threshold), Spec Tests 22 / 22, nothing waived** | `reports_output/release11/`, `run_specs/runtime_model_spec_release11.json`, `logs/tti-release11.log`, `reports_output/hf_same_setting_control.json` |
| ifeval | CI subset 109: 92.66 (PASS vs HF 89.91 same setting); FULL SET 541 non-thinking greedy: 87.99, ratio 0.942 -> FAIL (run 5); thinking mode full set: 94.35 on the 354 answered rows, 61.74 overall (34.6 % empty: think block not closed in 16K tokens, run 6 / 7) | `reports_output/evals_full_nothink/`, `reports_output/evals_full_thinking/`, RUN_NOTES |
| gpqa_diamond_cot_zeroshot | CI subset 40: 77.5 (31 / 40; PASS vs HF 80.0 = 32 / 40 in the same setting, one-item margin, run 11; in run 2 it showed as "waived" only because the catalog status was EXPERIMENTAL); FULL SET 198 non-thinking greedy: 68.69, 0.793 of the thinking-mode published number (run 5); thinking-mode full set not completable in budget (runs 6 / 9) | `reports_output/evals_full_nothink/`, RUN_NOTES |
| thinking-mode evals (run 3, same subsets) | INVALID as accuracy evidence: ifeval 62.39 with ~40 / 109 responses = lm-eval client `TimeoutError` (30-min thinking generations); stopped at the gpqa boundary (exit 143 = the stop). Open follow-up, not a model finding | `logs/tti-evals-thinking.log`, `reports_output/evals_thinking/`, RUN_NOTES run 3 |
| vLLM parameter conformance spec tests, thinking-default server (run 3) | FAIL in the harness: 11 / 22 parametrizations read `content` as a string, null for reasoning-only responses; 1 read timeout (30 s) on a thinking generation | `logs/tti-spec-tests-thinking.log`, `reports_output/spec_tests_thinking/` |
| vLLM parameter conformance spec tests, release configuration (run 4) | exit 0, **Acceptance PASS**, 22 / 22 parametrizations (coherence, determinism, logprobs, max_tokens, n, seeding, penalties, seed reproducibility, stop) | `logs/tti-spec-tests-nothink.log`, `reports_output/spec_tests/`, `run_specs/runtime_model_spec_spec_tests.json` |
| autoport implementation check | PASS: run specs and report metadata name `models/autoports/qwen_qwen3_5_122b_a10b` / `qwen35-122b-autoport`; server log shows the autoport's guard / bucket lines | `RUN_NOTES.md`, `logs/vllm-server-release-run.log` |
| tt-model container bundle (v5.1) | built (attempts 7-9; final image `36b6aa8bb6e4`, thinking OFF by default), served from the staged manifest on a reset device: `/v1/models` 262144, chat, `qwen3_xml` tool call, 2381-token non-aligned prompt, 32 concurrent requests 126 tok/s, 131072-token prompt; pushed private to `raahemnabeel/qwen3-5-122b-a10b-blackholex4`; served again from the Hub on a reset device with the same checks passing (round 2: cold pull + `docker load` of attempt 8; round 3: final image `36b6aa8bb6e4` by digest identity with the pushed revision, 32 concurrent 126.1 tok/s, 131072-token prompt, `tt-model curl` with defaults); current Hub revision `ed5f26b9` = README-only card fix on top, same code and image (RUN_NOTES bundle section) | `bundle/`, `../../tt-model.yaml` |
| `11-tti-release.check.sh` | exit 0 | `~/qwen35-122b-bringup/logs/11/check.log` |

Served benchmark rows worth noting (128 output tokens, greedy, from the release report): 1 user TPOT 37.7-40.0 ms at every
prompt length up to 131072 tokens (TTFT 0.39 s at 128 tokens, 7.8 s at 8192, 31 s at 32768, 64 s at 65536, 133 s at 131072);
32 users at 128 / 128: TPOT 127 ms, 288 total tok/s; the long-prompt concurrent points are dominated by serialised prefill
(TTFT 3-4 min at 8192-16384 x 15-31 users). All rows are the stage-10 serving configuration (widths 2..32, 768 MiB trace region).

Limits of this evidence: the release report's eval rows are ci-nightly CI-subset numbers (the full-set runs 5 / 6 / 9 are in
`reports_output/evals_full_*` and RUN_NOTES); the published scores are thinking-mode numbers; the ci-nightly subsets have
same-setting HF references (run 8b) that TTI grades against in run 11; the full sets have none, so their ratios are to the
published thinking-mode scores.
