## Tenstorrent Model Release Summary: Qwen/Qwen3.5-122B-A10B on P300X2

### Metadata: Qwen/Qwen3.5-122B-A10B on P300X2

```json
{
    "model_name": "Qwen3.5-122B-A10B",
    "model_repo": "Qwen/Qwen3.5-122B-A10B",
    "device": "P300X2",
    "generated_at": "2026-09-29T06:32:58+00:00",
    "report_id": "id_qwen35-122b-autoport_Qwen3.5-122B-A10B_p300x2_2026-09-29_08-46-31",
    "workflow": "release",
    "server_mode": "API",
    "run_command": "python run.py --dev-mode --model Qwen/Qwen3.5-122B-A10B --device p300x2 --workflow release --limit-samples-mode ci-nightly --service-port 8131 --no-auth --skip-system-sw-validation",
    "runtime_model_spec_json": "/home/raahem/tt-inference-server/workflow_logs/runtime_model_specs/runtime_model_spec_2026-09-29_06-12-45_id_qwen35-122b-autoport_Qwen3.5-122B-A10B_p300x2_3ckS78LH.json",
    "model_id": "id_qwen35-122b-autoport_Qwen3.5-122B-A10B_p300x2",
    "inference_engine": "vLLM",
    "tt_metal_commit": null,
    "vllm_commit": null,
    "model_impl": "qwen35-122b-autoport"
}
```

### Acceptance Criteria

- Acceptance status: ✅ `PASS`
- Model status: `FUNCTIONAL`
- Benchmarks: 🟨 `NA` (0/23 passed, 23 NA)
- Evals: ✅ `PASS` (2/2 passed)
- Spec Tests: ✅ `PASS` (1/1 passed)
- Agentic Targets: 🟨 `NA` (no blocks present)
- All acceptance criteria passed.

---

### Accuracy Evaluations for Qwen/Qwen3.5-122B-A10B on P300X2

| Task                      | Tolerance | Published Score | Published Score Ref                                                         | GPU Reference Score | gpu_reference_score_ref                                                                                                                                                                                                                                 | Score | Ratio to Published | Ratio to Reference | Accuracy Check | priority | mean_seconds_per_task |
|:--------------------------|:----------|:----------------|:----------------------------------------------------------------------------|:--------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------|:-------------------|:-------------------|:---------------|:---------|:----------------------|
| ifeval                    | 0.05      | 93.4            | https://huggingface.co/Qwen/Qwen3.5-122B-A10B (IFEval, thinking mode)       | 89.91               | ci-nightly ifeval (doc_ids 0-108), bf16 HF Qwen3_5MoeForCausalLM on CPU, same chat requests, enable_thinking=False, greedy, max_new_tokens 2048; prompt_level_strict_acc; 2026-09-29 (autoport doc/tti_release/RUN_NOTES.md run 8b) [CI_NIGHTLY subset] | 92.66 | 0.9921             | 1.031              | ✅ PASS        | must     | 3.946                 |
| gpqa_diamond_cot_zeroshot | 0.05      | 86.6            | https://huggingface.co/Qwen/Qwen3.5-122B-A10B (GPQA Diamond, thinking mode) | 60                  | ci-nightly gpqa_diamond_cot_zeroshot (doc_ids 0-39), bf16 HF on CPU, same requests, enable_thinking=False, greedy, max_new_tokens 4096; exact_match flexible-extract; 2026-09-29 (RUN_NOTES run 8b) [CI_NIGHTLY subset]                                 | 77.5  | 0.8949             | 1.292              | ✅ PASS        | must     | 19.54                 |

Note: The ratio to published scores defines if eval ran roughly correctly, as the exact methodology of the model publisher cannot always be reproduced. For this reason the accuracy check is based first on being equivalent to the GPU reference within a +/- tolerance. If a value GPU reference is not available, the accuracy check is based on the direct ratio to the published score.

---

### vLLM Benchmark for Qwen/Qwen3.5-122B-A10B on P300X2

| Concurrency | Num Requests | ISL    | OSL  | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Input (TPS) | Tput Output (TPS) | Tput Total (TPS) | Req Tput (RPS) |
|:------------|:-------------|:-------|:-----|:----------|:--------------|:--------------|:----------|:----------|:-----------------|:------------------|:-----------------|:---------------|
| 1           | 8            | 128    | 128  | 397.8     | 399.6         | 418.2         | 37.7      | 5189.5    | 24.7             | 24.7              | 49.3             | 0.193          |
| 32          | 256          | 128    | 128  | 12358.3   | 12372.2       | 12417.2       | 127.0     | 28490.9   | 143.8            | 143.8             | 287.5            | 1.123          |
| 1           | 4            | 128    | 1024 | 379.0     | 377.2         | 392.2         | 37.8      | 39005.4   | 3.3              | 26.3              | 29.5             | 0.026          |
| 32          | 128          | 128    | 1024 | 12317.8   | 12328.6       | 12351.8       | 140.4     | 155939.7  | 26.3             | 210.1             | 236.4            | 0.205          |
| 1           | 4            | 1024   | 128  | 1185.2    | 1185.3        | 1190.0        | 37.7      | 5977.8    | 171.3            | 21.4              | 192.7            | 0.167          |
| 32          | 128          | 1024   | 128  | 37956.6   | 37927.7       | 38058.3       | 126.6     | 54035.7   | 606.4            | 75.8              | 682.2            | 0.592          |
| 1           | 4            | 2048   | 128  | 1971.2    | 1970.3        | 1977.9        | 37.8      | 6767.8    | 302.6            | 18.9              | 321.5            | 0.148          |
| 32          | 128          | 2048   | 128  | 63063.2   | 63062.8       | 63151.8       | 126.6     | 79137.3   | 828.1            | 51.8              | 879.9            | 0.404          |
| 1           | 4            | 4096   | 128  | 3894.6    | 3893.5        | 3898.3        | 37.9      | 8709.8    | 470.3            | 14.7              | 485.0            | 0.115          |
| 32          | 128          | 4096   | 128  | 124753.7  | 124775.0      | 124854.7      | 127.2     | 140912.6  | 930.2            | 29.1              | 959.2            | 0.227          |
| 1           | 2            | 8192   | 128  | 7768.7    | 7768.7        | 7773.1        | 38.0      | 12592.3   | 650.5            | 10.2              | 660.7            | 0.079          |
| 31          | 62           | 8192   | 128  | 240798.4  | 240699.8      | 240926.0      | 129.4     | 257233.9  | 987.2            | 15.4              | 1002.7           | 0.120          |
| 1           | 2            | 8192   | 1024 | 7789.9    | 7789.9        | 7792.5        | 38.0      | 46694.0   | 175.4            | 21.9              | 197.4            | 0.021          |
| 28          | 56           | 8192   | 1024 | 217436.9  | 217382.4      | 217522.1      | 141.2     | 361872.7  | 633.9            | 79.2              | 713.1            | 0.077          |
| 1           | 2            | 10000  | 1024 | 9722.0    | 9722.0        | 9730.4        | 38.0      | 48627.0   | 205.6            | 21.1              | 226.7            | 0.021          |
| 23          | 46           | 10000  | 1024 | 223321.6  | 223329.3      | 223351.1      | 141.7     | 368310.0  | 624.5            | 63.9              | 688.4            | 0.062          |
| 1           | 2            | 16384  | 128  | 15572.0   | 15572.0       | 15576.3       | 38.1      | 20415.5   | 802.5            | 6.3               | 808.8            | 0.049          |
| 15          | 30           | 16384  | 128  | 233417.2  | 233382.9      | 233505.8      | 112.5     | 247700.9  | 992.2            | 7.8               | 999.9            | 0.061          |
| 1           | 1            | 32768  | 128  | 31423.1   | 31423.1       | 31423.1       | 38.4      | 36299.5   | 902.7            | 3.5               | 906.2            | 0.028          |
| 7           | 7            | 32768  | 128  | 219921.7  | 219973.6      | 219975.8      | 98.6      | 232445.1  | 986.8            | 3.9               | 990.6            | 0.030          |
| 1           | 1            | 65536  | 128  | 63910.2   | 63910.2       | 63910.2       | 39.0      | 68869.1   | 951.6            | 1.9               | 953.5            | 0.015          |
| 3           | 3            | 65536  | 128  | 191735.0  | 191846.4      | 191847.9      | 55.3      | 198763.0  | 989.2            | 1.9               | 991.1            | 0.015          |
| 1           | 1            | 131072 | 128  | 132586.9  | 132586.9      | 132586.9      | 40.0      | 137666.7  | 952.1            | 0.9               | 953.0            | 0.007          |

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
| Total Duration | 1471.71s                  |
| Total Attempts | 2                         |
| Generated      | 2026-09-29T06:32:58+00:00 |

## 🧪 Test Results

| Status  | Test Name                | Duration | Attempts | Description                                       |
|:--------|:-------------------------|:---------|:---------|:--------------------------------------------------|
| ✅ PASS | LoggerForkSafetyTest     | 0.00s    | 1        | Test for logging fork safety to prevent deadlocks |
| ✅ PASS | VLLMParamConformanceTest | 1471.71s | 1        | vLLM chat/completions parameter conformance       |

---

### Logger Fork Safety for Qwen/Qwen3.5-122B-A10B on P300X2

| Child Result |
|:-------------|
| OK           |

---

### Vllm Chat Completions for Qwen/Qwen3.5-122B-A10B on P300X2

| Endpoint URL                              | model_name             | Task                  |
|:------------------------------------------|:-----------------------|:----------------------|
| http://127.0.0.1:8131/v1/chat/completions | Qwen/Qwen3.5-122B-A10B | vllm_chat_completions |

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
