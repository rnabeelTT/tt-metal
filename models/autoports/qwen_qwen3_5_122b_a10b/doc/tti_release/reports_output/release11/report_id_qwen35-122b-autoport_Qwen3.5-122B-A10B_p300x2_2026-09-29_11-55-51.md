## Tenstorrent Model Release Summary: Qwen/Qwen3.5-122B-A10B on P300X2

### Metadata: Qwen/Qwen3.5-122B-A10B on P300X2

```json
{
    "model_name": "Qwen3.5-122B-A10B",
    "model_repo": "Qwen/Qwen3.5-122B-A10B",
    "device": "P300X2",
    "generated_at": "2026-09-29T09:42:19+00:00",
    "report_id": "id_qwen35-122b-autoport_Qwen3.5-122B-A10B_p300x2_2026-09-29_11-55-51",
    "workflow": "release",
    "server_mode": "API",
    "run_command": "python run.py --dev-mode --model Qwen/Qwen3.5-122B-A10B --device p300x2 --workflow release --limit-samples-mode ci-nightly --service-port 8131 --no-auth --skip-system-sw-validation",
    "runtime_model_spec_json": "/home/raahem/tt-inference-server/workflow_logs/runtime_model_specs/runtime_model_spec_2026-09-29_09-22-07_id_qwen35-122b-autoport_Qwen3.5-122B-A10B_p300x2_Lw9dsPFy.json",
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

| Task                      | Tolerance | Published Score | Published Score Ref                                                         | GPU Reference Score | gpu_reference_score_ref                                                                                                                                                                                                                                                                                                                                            | Score | Ratio to Published | Ratio to Reference | Accuracy Check | priority | mean_seconds_per_task |
|:--------------------------|:----------|:----------------|:----------------------------------------------------------------------------|:--------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------|:-------------------|:-------------------|:---------------|:---------|:----------------------|
| ifeval                    | 0.05      | 93.4            | https://huggingface.co/Qwen/Qwen3.5-122B-A10B (IFEval, thinking mode)       | 89.91               | ci-nightly ifeval (doc_ids 0-108), bf16 HF Qwen3_5MoeForCausalLM on CPU, same chat requests, enable_thinking=False, greedy, max_new_tokens 2048; prompt_level_strict_acc; 2026-09-29 (autoport doc/tti_release/RUN_NOTES.md run 8b) [CI_NIGHTLY subset]                                                                                                            | 92.66 | 0.9921             | 1.031              | ✅ PASS        | must     | 3.94                  |
| gpqa_diamond_cot_zeroshot | 0.05      | 86.6            | https://huggingface.co/Qwen/Qwen3.5-122B-A10B (GPQA Diamond, thinking mode) | 80                  | ci-nightly gpqa_diamond_cot_zeroshot (doc_ids 0-39), bf16 HF on CPU, same requests, enable_thinking=False, greedy, max_new_tokens 4096; exact_match flexible-extract scored with this venv's boxed_choice filter (lm_eval 0.4.10.dev0); 2026-09-29 (RUN_NOTES run 8b; an earlier value of 60.00 came from the older multi_choice_regex filter) [CI_NIGHTLY subset] | 77.5  | 0.8949             | 0.9688             | ✅ PASS        | must     | 19.54                 |

Note: The ratio to published scores defines if eval ran roughly correctly, as the exact methodology of the model publisher cannot always be reproduced. For this reason the accuracy check is based first on being equivalent to the GPU reference within a +/- tolerance. If a value GPU reference is not available, the accuracy check is based on the direct ratio to the published score.

---

### vLLM Benchmark for Qwen/Qwen3.5-122B-A10B on P300X2

| Concurrency | Num Requests | ISL    | OSL  | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Input (TPS) | Tput Output (TPS) | Tput Total (TPS) | Req Tput (RPS) |
|:------------|:-------------|:-------|:-----|:----------|:--------------|:--------------|:----------|:----------|:-----------------|:------------------|:-----------------|:---------------|
| 1           | 8            | 128    | 128  | 398.4     | 401.2         | 413.6         | 37.7      | 5190.7    | 24.7             | 24.7              | 49.3             | 0.193          |
| 32          | 256          | 128    | 128  | 12349.9   | 12336.3       | 12464.2       | 127.0     | 28481.9   | 143.8            | 143.8             | 287.6            | 1.123          |
| 1           | 4            | 128    | 1024 | 384.2     | 380.1         | 405.0         | 37.8      | 39008.9   | 3.3              | 26.3              | 29.5             | 0.026          |
| 32          | 128          | 128    | 1024 | 12383.2   | 12373.9       | 12450.2       | 140.4     | 155999.9  | 26.3             | 210.0             | 236.3            | 0.205          |
| 1           | 4            | 1024   | 128  | 1183.7    | 1181.7        | 1190.8        | 37.7      | 5974.7    | 171.4            | 21.4              | 192.8            | 0.167          |
| 32          | 128          | 1024   | 128  | 38011.1   | 37989.7       | 38067.7       | 126.6     | 54088.1   | 605.8            | 75.7              | 681.5            | 0.592          |
| 1           | 4            | 2048   | 128  | 1967.8    | 1968.3        | 1972.3        | 37.8      | 6763.8    | 302.8            | 18.9              | 321.7            | 0.148          |
| 32          | 128          | 2048   | 128  | 63029.5   | 63056.9       | 63090.9       | 126.6     | 79103.2   | 828.5            | 51.8              | 880.3            | 0.405          |
| 1           | 4            | 4096   | 128  | 3896.3    | 3896.7        | 3898.9        | 37.9      | 8711.0    | 470.2            | 14.7              | 484.9            | 0.115          |
| 32          | 128          | 4096   | 128  | 124685.4  | 124681.7      | 124758.4      | 127.2     | 140843.6  | 930.6            | 29.1              | 959.7            | 0.227          |
| 1           | 2            | 8192   | 128  | 7776.2    | 7776.2        | 7783.6        | 38.0      | 12600.4   | 650.1            | 10.2              | 660.3            | 0.079          |
| 31          | 62           | 8192   | 128  | 240870.8  | 240758.9      | 241014.7      | 129.2     | 257279.6  | 987.1            | 15.4              | 1002.5           | 0.120          |
| 1           | 2            | 8192   | 1024 | 7782.1    | 7782.1        | 7782.2        | 38.0      | 46691.6   | 175.4            | 21.9              | 197.4            | 0.021          |
| 28          | 56           | 8192   | 1024 | 217477.3  | 217437.3      | 217549.7      | 141.2     | 361915.4  | 633.8            | 79.2              | 713.0            | 0.077          |
| 1           | 2            | 10000  | 1024 | 9721.4    | 9721.4        | 9728.2        | 38.0      | 48628.7   | 205.6            | 21.1              | 226.7            | 0.021          |
| 23          | 46           | 10000  | 1024 | 223313.3  | 223248.9      | 223413.7      | 141.6     | 368159.4  | 624.7            | 64.0              | 688.7            | 0.062          |
| 1           | 2            | 16384  | 128  | 15571.7   | 15571.7       | 15575.0       | 38.1      | 20414.7   | 802.5            | 6.3               | 808.8            | 0.049          |
| 15          | 30           | 16384  | 128  | 233562.1  | 233587.0      | 233590.8      | 112.5     | 247847.4  | 991.6            | 7.7               | 999.3            | 0.060          |
| 1           | 1            | 32768  | 128  | 31461.5   | 31461.5       | 31461.5       | 38.4      | 36339.2   | 901.7            | 3.5               | 905.2            | 0.028          |
| 7           | 7            | 32768  | 128  | 219749.6  | 219800.8      | 219802.3      | 98.6      | 232278.0  | 987.5            | 3.9               | 991.4            | 0.030          |
| 1           | 1            | 65536  | 128  | 63938.6   | 63938.6       | 63938.6       | 39.0      | 68887.4   | 951.3            | 1.9               | 953.2            | 0.015          |
| 3           | 3            | 65536  | 128  | 191948.2  | 192059.3      | 192060.8      | 55.3      | 198976.3  | 988.1            | 1.9               | 990.0            | 0.015          |
| 1           | 1            | 131072 | 128  | 132665.8  | 132665.8      | 132665.8      | 40.2      | 137769.6  | 951.4            | 0.9               | 952.3            | 0.007          |

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
| Total Duration | 1471.96s                  |
| Total Attempts | 2                         |
| Generated      | 2026-09-29T09:42:19+00:00 |

## 🧪 Test Results

| Status  | Test Name                | Duration | Attempts | Description                                       |
|:--------|:-------------------------|:---------|:---------|:--------------------------------------------------|
| ✅ PASS | LoggerForkSafetyTest     | 0.00s    | 1        | Test for logging fork safety to prevent deadlocks |
| ✅ PASS | VLLMParamConformanceTest | 1471.95s | 1        | vLLM chat/completions parameter conformance       |

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
