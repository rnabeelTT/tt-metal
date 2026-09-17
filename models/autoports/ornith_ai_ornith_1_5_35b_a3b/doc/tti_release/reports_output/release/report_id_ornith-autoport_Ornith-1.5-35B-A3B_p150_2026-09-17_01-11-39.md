## Tenstorrent Model Release Summary: ornith-ai/Ornith-1.5-35B-A3B on P150

### Metadata: ornith-ai/Ornith-1.5-35B-A3B on P150

```json
{
    "model_name": "Ornith-1.5-35B-A3B",
    "model_repo": "ornith-ai/Ornith-1.5-35B-A3B",
    "device": "P150",
    "generated_at": "2026-09-17T00:37:29+00:00",
    "report_id": "id_ornith-autoport_Ornith-1.5-35B-A3B_p150_2026-09-17_01-11-39",
    "workflow": "release",
    "server_mode": "API",
    "run_command": "python run.py --model ornith-ai/Ornith-1.5-35B-A3B --runtime-model-spec-json /tmp/claude-1003/-home-raahem-tt-metal/ce4a8061-1dc4-4464-a943-ff7d6bb636e4/scratchpad/tti_work/ornith_autoport_p150_spec.json --tt-device p150 --workflow release --no-auth --skip-system-sw-validation --server-url http://127.0.0.1:8000 --service-port 8000 --limit-samples-mode ci-nightly",
    "runtime_model_spec_json": "/home/raahem/tt-inference-server/workflow_logs/runtime_model_specs/runtime_model_spec_2026-09-17_00-19-44_id_ornith-autoport_Ornith-1.5-35B-A3B_p150_Mobt_r4A.json",
    "model_id": "id_ornith-autoport_Ornith-1.5-35B-A3B_p150",
    "inference_engine": "vLLM",
    "tt_metal_commit": "72f2033326e",
    "vllm_commit": "0.26.0",
    "model_impl": "ornith-autoport"
}
```

### Acceptance Criteria

- Acceptance status: ❌ `FAIL`
- Model status: `EXPERIMENTAL`
- Benchmarks: 🟨 `NA` (0/12 passed, 12 NA)
- Evals: 🟨 `NA` (0/1 passed, 1 NA)
- Spec Tests: ❌ `FAIL` (0/1 passed, 1 failed)

#### Blockers

- `spec.spec_tests:Vllm Chat Completions`: Vllm Chat Completions reported status=fail (attempts=1)
- `task:spec_tests`: Task 'spec_tests' failed (exit=1) after producing a report block.

---

### LLM Eval — ifeval for ornith-ai/Ornith-1.5-35B-A3B on P150

| Task   | Tolerance | Score | Ratio to Published | Ratio to Reference | Accuracy Check | priority | mean_seconds_per_task |
|:-------|:----------|:------|:-------------------|:-------------------|:---------------|:---------|:----------------------|
| ifeval | 0.05      | 60.71 | N/A                | N/A                | 🟨 NA          | must     | 37.99                 |

Note: The ratio to published scores defines if eval ran roughly correctly, as the exact methodology of the model publisher cannot always be reproduced. For this reason the accuracy check is based first on being equivalent to the GPU reference within a +/- tolerance. If a value GPU reference is not available, the accuracy check is based on the direct ratio to the published score.

---

### vLLM Benchmark for ornith-ai/Ornith-1.5-35B-A3B on P150

| Concurrency | Num Requests | ISL    | OSL  | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Input (TPS) | Tput Output (TPS) | Tput Total (TPS) | Req Tput (RPS) |
|:------------|:-------------|:-------|:-----|:----------|:--------------|:--------------|:----------|:----------|:-----------------|:------------------|:-----------------|:---------------|
| 1           | 8            | 128    | 128  | 672.0     | 672.2         | 681.9         | 30.3      | 4521.4    | 28.3             | 28.3              | 56.6             | 0.221          |
| 1           | 4            | 128    | 1024 | 673.7     | 671.9         | 684.8         | 30.7      | 32126.9   | 4.0              | 31.9              | 35.9             | 0.031          |
| 1           | 4            | 1024   | 128  | 4787.8    | 4773.6        | 4874.0        | 30.6      | 8668.3    | 118.1            | 14.8              | 132.9            | 0.115          |
| 1           | 4            | 2048   | 128  | 5422.0    | 5425.8        | 5564.7        | 30.6      | 9303.7    | 220.1            | 13.8              | 233.9            | 0.107          |
| 1           | 4            | 4096   | 128  | 10556.0   | 10530.6       | 10850.7       | 30.7      | 14458.4   | 283.3            | 8.9               | 292.1            | 0.069          |
| 1           | 2            | 8192   | 128  | 20688.0   | 20688.0       | 20859.1       | 31.0      | 24623.1   | 332.7            | 5.2               | 337.9            | 0.041          |
| 1           | 2            | 8192   | 1024 | 20690.0   | 20690.0       | 20857.8       | 30.9      | 52333.0   | 156.5            | 19.6              | 176.1            | 0.019          |
| 1           | 2            | 10000  | 1024 | 25490.9   | 25490.9       | 25689.4       | 31.1      | 57319.5   | 174.5            | 17.9              | 192.3            | 0.017          |
| 1           | 2            | 16384  | 128  | 41404.6   | 41404.6       | 41841.5       | 31.5      | 45406.8   | 360.8            | 2.8               | 363.6            | 0.022          |
| 1           | 1            | 32768  | 128  | 84731.2   | 84731.2       | 84731.2       | 32.6      | 88871.4   | 368.7            | 1.4               | 370.2            | 0.011          |
| 1           | 1            | 65536  | 128  | 174854.8  | 174854.8      | 174854.8      | 34.3      | 179206.2  | 365.7            | 0.7               | 366.4            | 0.006          |
| 1           | 1            | 131072 | 128  | 369740.0  | 369740.0      | 369740.0      | 38.4      | 374614.3  | 349.9            | 0.3               | 350.2            | 0.003          |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: No perf targets are configured for these sweep points, so these rows are reported for information only and are not graded.

---

## 📋 Summary

| Metric         | Value                     |
|:---------------|:--------------------------|
| Total Tests    | 2                         |
| Passed         | 1                         |
| Failed         | 1                         |
| Skipped        | 0                         |
| NA             | 0                         |
| Attempted      | 2                         |
| Success Rate   | 50.0%                     |
| Total Duration | 672.34s                   |
| Total Attempts | 2                         |
| Generated      | 2026-09-17T00:37:29+00:00 |

## 🧪 Test Results

| Status  | Test Name                | Duration | Attempts | Description                                       |
|:--------|:-------------------------|:---------|:---------|:--------------------------------------------------|
| ✅ PASS | LoggerForkSafetyTest     | 0.00s    | 1        | Test for logging fork safety to prevent deadlocks |
| ❌ FAIL | VLLMParamConformanceTest | 672.34s  | 1        | vLLM chat/completions parameter conformance       |

---

### Logger Fork Safety for ornith-ai/Ornith-1.5-35B-A3B on P150

| Child Result |
|:-------------|
| OK           |

---

### Vllm Chat Completions for ornith-ai/Ornith-1.5-35B-A3B on P150

| Endpoint URL                              | model_name                   | Task                  |
|:------------------------------------------|:-----------------------------|:----------------------|
| http://127.0.0.1:8000/v1/chat/completions | ornith-ai/Ornith-1.5-35B-A3B | vllm_chat_completions |

#### Parameter Conformance Summary

| Test Case                    | Status  | Summary    |
|:-----------------------------|:--------|:-----------|
| test_coherence_verbatim_echo | ❌ FAIL | 0/1 passed |
| test_determinism_parameters  | ✅ PASS | 3/3 passed |
| test_logprobs                | ✅ PASS | 1/1 passed |
| test_max_tokens              | ✅ PASS | 2/2 passed |
| test_n                       | ✅ PASS | 2/2 passed |
| test_non_uniform_seeding     | ❌ FAIL | 0/1 passed |
| test_penalties               | ❌ FAIL | 4/9 passed |
| test_seed_reproducibility    | ✅ PASS | 1/1 passed |
| test_stop                    | ❌ FAIL | 1/2 passed |

#### Detailed Test Results

| Test Case                    | Parametrization                                                      | Status    | Message                                                                                  |
|:-----------------------------|:---------------------------------------------------------------------|:----------|:-----------------------------------------------------------------------------------------|
| test_coherence_verbatim_echo | test_coherence_verbatim_echo                                         | ❌ FAILED | Traceback: TypeError: argument of type 'NoneType' is not iterable                        |
| test_determinism_parameters  | test_determinism_parameters[temperature-0.0]                         | ✅ PASSED |                                                                                          |
| test_determinism_parameters  | test_determinism_parameters[top_k-1]                                 | ✅ PASSED |                                                                                          |
| test_determinism_parameters  | test_determinism_parameters[top_p-0.01]                              | ✅ PASSED |                                                                                          |
| test_logprobs                | test_logprobs                                                        | ✅ PASSED |                                                                                          |
| test_max_tokens              | test_max_tokens[10]                                                  | ✅ PASSED |                                                                                          |
| test_max_tokens              | test_max_tokens[5]                                                   | ✅ PASSED |                                                                                          |
| test_n                       | test_n[2]                                                            | ✅ PASSED |                                                                                          |
| test_n                       | test_n[3]                                                            | ✅ PASSED |                                                                                          |
| test_non_uniform_seeding     | test_non_uniform_seeding                                             | ❌ FAILED | Traceback: Failed: Request failed for seed 1: 'NoneType' object has no attribute 'strip' |
| test_penalties               | test_penalties[frequency_penalty-1.2-repeat_trap-messages0]          | ❌ FAILED | Traceback: AttributeError: 'NoneType' object has no attribute 'lower'                    |
| test_penalties               | test_penalties[frequency_penalty-1.2-semantic_repetition-messages2]  | ❌ FAILED | Traceback: AttributeError: 'NoneType' object has no attribute 'lower'                    |
| test_penalties               | test_penalties[presence_penalty-1.2-semantic_repetition-messages2]   | ❌ FAILED | Traceback: AttributeError: 'NoneType' object has no attribute 'lower'                    |
| test_penalties               | test_penalties[repetition_penalty-1.5-natural_repetition-messages1]  | ❌ FAILED | Traceback: AttributeError: 'NoneType' object has no attribute 'lower'                    |
| test_penalties               | test_penalties[repetition_penalty-1.5-repeat_trap-messages0]         | ❌ FAILED | Traceback: AttributeError: 'NoneType' object has no attribute 'lower'                    |
| test_penalties               | test_penalties[frequency_penalty-1.2-natural_repetition-messages1]   | ✅ PASSED |                                                                                          |
| test_penalties               | test_penalties[presence_penalty-1.2-natural_repetition-messages1]    | ✅ PASSED |                                                                                          |
| test_penalties               | test_penalties[presence_penalty-1.2-repeat_trap-messages0]           | ✅ PASSED |                                                                                          |
| test_penalties               | test_penalties[repetition_penalty-1.5-semantic_repetition-messages2] | ✅ PASSED |                                                                                          |
| test_seed_reproducibility    | test_seed_reproducibility                                            | ✅ PASSED |                                                                                          |
| test_stop                    | test_stop[stop_seq0]                                                 | ❌ FAILED | Traceback: TypeError: argument of type 'NoneType' is not iterable                        |
| test_stop                    | test_stop[stop_seq1]                                                 | ✅ PASSED |                                                                                          |
