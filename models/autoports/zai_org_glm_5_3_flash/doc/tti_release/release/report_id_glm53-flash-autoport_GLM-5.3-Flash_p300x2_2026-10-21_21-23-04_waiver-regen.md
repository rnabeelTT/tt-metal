## Tenstorrent Model Release Summary: zai-org/GLM-5.3-Flash on P300X2

### Metadata: zai-org/GLM-5.3-Flash on P300X2

```json
{
    "model_name": "GLM-5.3-Flash",
    "model_repo": "zai-org/GLM-5.3-Flash",
    "device": "P300X2",
    "generated_at": "2026-10-21T20:17:51+00:00",
    "report_id": "id_glm53-flash-autoport_GLM-5.3-Flash_p300x2_2026-10-21_21-23-04_waiver-regen",
    "workflow": "release",
    "server_mode": "API",
    "run_command": "python run.py --dev-mode --model zai-org/GLM-5.3-Flash --tt-device p300x2 --workflow release --service-port 8141 --no-auth --skip-system-sw-validation --limit-samples-mode ci-nightly",
    "runtime_model_spec_json": "/home/raahem/glm53-flash-bringup/tti/workflow_logs/runtime_model_specs/runtime_model_spec_2026-10-21_16-42-31_id_glm53-flash-autoport_GLM-5.3-Flash_p300x2_ZSiaE1Ol.json",
    "model_id": "id_glm53-flash-autoport_GLM-5.3-Flash_p300x2",
    "inference_engine": "vLLM",
    "tt_metal_commit": null,
    "vllm_commit": null,
    "model_impl": "glm53-flash-autoport"
}
```

### Acceptance Criteria

- Acceptance status: ✅ `PASS`
- Model status: `FUNCTIONAL`
- Benchmarks: 🟨 `NA` (0/12 passed, 12 NA)
- Evals: ✅ `PASS` (0/2 passed, 1 waived, 1 NA)
- Spec Tests: ✅ `PASS` (1/1 passed)
- Agentic Targets: 🟨 `NA` (no blocks present)
- All acceptance criteria passed.

Waived rows (TTI `known_issues` of the model spec):

- `evals:LLM Eval — gpqa_diamond_cot_zeroshot`: Accuracy check failed. (waived: Issue-waived (stage 11, lead/user decision 2026-10-23; models/autoports/zai_org_glm_5_3_flash/doc/tti_release/RUN_NOTES.md). lm-eval full set 66.67 %, CI subset 70.0 %; card protocol 74.24 % against the card's 79.8 % (same effort, budget and sampling). The stage-8 non-expert precision changes are ruled out (H1 chemistry: stage-7 policy 58.1 % vs shipped 54.8 %, not significant); answer extraction, effort rendering and budget handling checked. The residual -5.6 points (about 1.8 sigma) exceeds the card's stated +/-2-3 point run noise and is unexplained; bfp4 vs NVFP4 routed experts were not tested.)

---

### Accuracy Evaluations for zai-org/GLM-5.3-Flash on P300X2

| Task                      | Tolerance | Published Score | Published Score Ref                                                                                    | gpu_reference_score_ref                           | Score | Ratio to Published | Ratio to Reference | Accuracy Check | priority | mean_seconds_per_task |
|:--------------------------|:----------|:----------------|:-------------------------------------------------------------------------------------------------------|:--------------------------------------------------|:------|:-------------------|:-------------------|:---------------|:---------|:----------------------|
| ifeval                    | 0.05      | N/A             | none: no IFEval number for GLM-5.3-Flash or its E192 subset                                            | none: no same-setting reference of the E192 model | 74.31 | N/A                | N/A                | 🟨 NA          | must     | 100.2                 |
| gpqa_diamond_cot_zeroshot | 0.05      | 79.8            | https://huggingface.co/autotrust/GLM5.3-Flash-E192-DGX-Spark (GPQA-Diamond, effort=low, 65,536 tokens) | N/A                                               | 70    | 0.8772             | N/A                | WAIVED (known issue)        | must     | 49.88                 |

Note: The ratio to published scores defines if eval ran roughly correctly, as the exact methodology of the model publisher cannot always be reproduced. For this reason the accuracy check is based first on being equivalent to the GPU reference within a +/- tolerance. If a value GPU reference is not available, the accuracy check is based on the direct ratio to the published score.

---

### vLLM Benchmark for zai-org/GLM-5.3-Flash on P300X2

| Concurrency | Num Requests | ISL    | OSL  | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Input (TPS) | Tput Output (TPS) | Tput Total (TPS) | Req Tput (RPS) |
|:------------|:-------------|:-------|:-----|:----------|:--------------|:--------------|:----------|:----------|:-----------------|:------------------|:-----------------|:---------------|
| 1           | 8            | 128    | 128  | 251.2     | 249.8         | 260.5         | 71.5      | 9333.4    | 13.7             | 13.7              | 27.4             | 0.107          |
| 1           | 4            | 128    | 1024 | 248.4     | 248.4         | 253.5         | 73.0      | 74901.3   | 1.7              | 13.7              | 15.4             | 0.013          |
| 1           | 4            | 1024   | 128  | 824.6     | 814.3         | 860.4         | 74.4      | 10271.1   | 99.7             | 12.5              | 112.2            | 0.097          |
| 1           | 4            | 2048   | 128  | 1401.2    | 1388.5        | 1441.9        | 77.5      | 11242.0   | 182.2            | 11.4              | 193.6            | 0.089          |
| 1           | 4            | 4096   | 128  | 5989.3    | 5987.9        | 5999.9        | 77.6      | 15839.1   | 258.6            | 8.1               | 266.7            | 0.063          |
| 1           | 2            | 8192   | 128  | 15079.1   | 15079.1       | 15087.5       | 77.7      | 24943.5   | 328.4            | 5.1               | 333.5            | 0.040          |
| 1           | 2            | 8192   | 1024 | 15097.1   | 15097.1       | 15104.8       | 77.8      | 94636.3   | 86.6             | 10.8              | 97.4             | 0.011          |
| 1           | 2            | 10000  | 1024 | 19626.3   | 19626.3       | 19651.7       | 77.8      | 99184.5   | 100.8            | 10.3              | 111.1            | 0.010          |
| 1           | 2            | 16384  | 128  | 33351.0   | 33351.0       | 33408.8       | 77.7      | 43217.0   | 379.1            | 3.0               | 382.1            | 0.023          |
| 1           | 1            | 32768  | 128  | 69716.7   | 69716.7       | 69716.7       | 77.6      | 79569.3   | 411.8            | 1.6               | 413.4            | 0.013          |
| 1           | 1            | 65536  | 128  | 142672.5  | 142672.5      | 142672.5      | 77.5      | 152517.1  | 429.7            | 0.8               | 430.5            | 0.007          |
| 1           | 1            | 131072 | 128  | 288528.3  | 288528.3      | 288528.3      | 77.3      | 298349.0  | 439.3            | 0.4               | 439.8            | 0.003          |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: No perf targets are configured for these sweep points, so these rows are reported for information only and are not graded.

---

## 📋 Summary

| Metric         | Value                     |
|:---------------|:--------------------------|
| Total Tests    | 2                         |
| Passed         | 2                         |
| Failed         | 0                         |
| Skipped        | 0                         |
| NA             | 0                         |
| Attempted      | 2                         |
| Success Rate   | 100.0%                    |
| Total Duration | 1324.35s                  |
| Total Attempts | 2                         |
| Generated      | 2026-10-21T20:17:51+00:00 |

## 🧪 Test Results

| Status  | Test Name                | Duration | Attempts | Description                                       |
|:--------|:-------------------------|:---------|:---------|:--------------------------------------------------|
| ✅ PASS | LoggerForkSafetyTest     | 0.00s    | 1        | Test for logging fork safety to prevent deadlocks |
| ✅ PASS | VLLMParamConformanceTest | 1324.35s | 1        | vLLM chat/completions parameter conformance       |

---

### Logger Fork Safety for zai-org/GLM-5.3-Flash on P300X2

| Child Result |
|:-------------|
| OK           |

---

### Vllm Chat Completions for zai-org/GLM-5.3-Flash on P300X2

| Endpoint URL                              | model_name            | Task                  |
|:------------------------------------------|:----------------------|:----------------------|
| http://127.0.0.1:8141/v1/chat/completions | zai-org/GLM-5.3-Flash | vllm_chat_completions |

#### Parameter Conformance Summary

| Test Case                    | Status  | Summary    |
|:-----------------------------|:--------|:-----------|
| test_coherence_verbatim_echo | ✅ PASS | 1/1 passed |
| test_determinism_parameters  | ✅ PASS | 3/3 passed |
| test_logprobs                | ✅ PASS | 1/1 passed |
| test_max_tokens              | ✅ PASS | 2/2 passed |
| test_n                       | ✅ PASS | 2/2 passed |
| test_non_uniform_seeding     | ✅ PASS | 1/1 passed |
| test_penalties               | ✅ PASS | 9/9 passed |
| test_seed_reproducibility    | ✅ PASS | 1/1 passed |
| test_stop                    | ✅ PASS | 2/2 passed |

#### Detailed Test Results

| Test Case                    | Parametrization                                                      | Status    |
|:-----------------------------|:---------------------------------------------------------------------|:----------|
| test_coherence_verbatim_echo | test_coherence_verbatim_echo                                         | ✅ PASSED |
| test_determinism_parameters  | test_determinism_parameters[temperature-0.0]                         | ✅ PASSED |
| test_determinism_parameters  | test_determinism_parameters[top_k-1]                                 | ✅ PASSED |
| test_determinism_parameters  | test_determinism_parameters[top_p-0.01]                              | ✅ PASSED |
| test_logprobs                | test_logprobs                                                        | ✅ PASSED |
| test_max_tokens              | test_max_tokens[10]                                                  | ✅ PASSED |
| test_max_tokens              | test_max_tokens[5]                                                   | ✅ PASSED |
| test_n                       | test_n[2]                                                            | ✅ PASSED |
| test_n                       | test_n[3]                                                            | ✅ PASSED |
| test_non_uniform_seeding     | test_non_uniform_seeding                                             | ✅ PASSED |
| test_penalties               | test_penalties[frequency_penalty-1.2-natural_repetition-messages1]   | ✅ PASSED |
| test_penalties               | test_penalties[frequency_penalty-1.2-repeat_trap-messages0]          | ✅ PASSED |
| test_penalties               | test_penalties[frequency_penalty-1.2-semantic_repetition-messages2]  | ✅ PASSED |
| test_penalties               | test_penalties[presence_penalty-1.2-natural_repetition-messages1]    | ✅ PASSED |
| test_penalties               | test_penalties[presence_penalty-1.2-repeat_trap-messages0]           | ✅ PASSED |
| test_penalties               | test_penalties[presence_penalty-1.2-semantic_repetition-messages2]   | ✅ PASSED |
| test_penalties               | test_penalties[repetition_penalty-1.5-natural_repetition-messages1]  | ✅ PASSED |
| test_penalties               | test_penalties[repetition_penalty-1.5-repeat_trap-messages0]         | ✅ PASSED |
| test_penalties               | test_penalties[repetition_penalty-1.5-semantic_repetition-messages2] | ✅ PASSED |
| test_seed_reproducibility    | test_seed_reproducibility                                            | ✅ PASSED |
| test_stop                    | test_stop[stop_seq0]                                                 | ✅ PASSED |
| test_stop                    | test_stop[stop_seq1]                                                 | ✅ PASSED |
