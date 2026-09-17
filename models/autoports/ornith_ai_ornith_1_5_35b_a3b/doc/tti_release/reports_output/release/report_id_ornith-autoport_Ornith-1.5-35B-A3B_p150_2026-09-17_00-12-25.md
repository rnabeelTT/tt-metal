## Tenstorrent Model Release Summary: ornith-ai/Ornith-1.5-35B-A3B on P150

### Metadata: ornith-ai/Ornith-1.5-35B-A3B on P150

```json
{
    "model_name": "Ornith-1.5-35B-A3B",
    "model_repo": "ornith-ai/Ornith-1.5-35B-A3B",
    "device": "P150",
    "generated_at": "2026-09-16T23:37:33+00:00",
    "report_id": "id_ornith-autoport_Ornith-1.5-35B-A3B_p150_2026-09-17_00-12-25",
    "workflow": "release",
    "server_mode": "API",
    "run_command": "python run.py --model ornith-ai/Ornith-1.5-35B-A3B --runtime-model-spec-json /tmp/claude-1003/-home-raahem-tt-metal/ce4a8061-1dc4-4464-a943-ff7d6bb636e4/scratchpad/tti_work/ornith_autoport_p150_spec.json --tt-device p150 --workflow release --no-auth --skip-system-sw-validation --server-url http://127.0.0.1:8000 --service-port 8000 --limit-samples-mode ci-nightly",
    "runtime_model_spec_json": "/home/raahem/tt-inference-server/workflow_logs/runtime_model_specs/runtime_model_spec_2026-09-16_23-21-10_id_ornith-autoport_Ornith-1.5-35B-A3B_p150_1MpJSgJy.json",
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
| ifeval | 0.05      | 32.14 | N/A                | N/A                | 🟨 NA          | must     | 35.05                 |

Note: The ratio to published scores defines if eval ran roughly correctly, as the exact methodology of the model publisher cannot always be reproduced. For this reason the accuracy check is based first on being equivalent to the GPU reference within a +/- tolerance. If a value GPU reference is not available, the accuracy check is based on the direct ratio to the published score.

---

### vLLM Benchmark for ornith-ai/Ornith-1.5-35B-A3B on P150

| Concurrency | Num Requests | ISL    | OSL  | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Input (TPS) | Tput Output (TPS) | Tput Total (TPS) | Req Tput (RPS) |
|:------------|:-------------|:-------|:-----|:----------|:--------------|:--------------|:----------|:----------|:-----------------|:------------------|:-----------------|:---------------|
| 1           | 8            | 128    | 128  | 671.4     | 671.5         | 681.2         | 30.3      | 4522.3    | 28.3             | 28.3              | 56.6             | 0.221          |
| 1           | 4            | 128    | 1024 | 673.8     | 670.8         | 683.4         | 30.4      | 31821.4   | 4.0              | 32.2              | 36.2             | 0.031          |
| 1           | 4            | 1024   | 128  | 4802.2    | 4799.2        | 4868.9        | 30.5      | 8678.6    | 118.0            | 14.7              | 132.7            | 0.115          |
| 1           | 4            | 2048   | 128  | 5420.8    | 5422.8        | 5565.2        | 30.6      | 9311.2    | 219.9            | 13.7              | 233.7            | 0.107          |
| 1           | 4            | 4096   | 128  | 10556.7   | 10533.3       | 10846.7       | 30.7      | 14461.5   | 283.2            | 8.9               | 292.1            | 0.069          |
| 1           | 2            | 8192   | 128  | 20691.9   | 20691.9       | 20865.7       | 31.0      | 24632.9   | 332.6            | 5.2               | 337.8            | 0.041          |
| 1           | 2            | 8192   | 1024 | 20687.0   | 20687.0       | 20852.2       | 31.0      | 52433.0   | 156.2            | 19.5              | 175.8            | 0.019          |
| 1           | 2            | 10000  | 1024 | 25488.4   | 25488.4       | 25688.2       | 31.1      | 57331.2   | 174.4            | 17.9              | 192.3            | 0.017          |
| 1           | 2            | 16384  | 128  | 41405.1   | 41405.1       | 41840.8       | 31.5      | 45404.5   | 360.8            | 2.8               | 363.7            | 0.022          |
| 1           | 1            | 32768  | 128  | 84731.5   | 84731.5       | 84731.5       | 32.4      | 88847.1   | 368.8            | 1.4               | 370.3            | 0.011          |
| 1           | 1            | 65536  | 128  | 174861.2  | 174861.2      | 174861.2      | 34.3      | 179213.5  | 365.7            | 0.7               | 366.4            | 0.006          |
| 1           | 1            | 131072 | 128  | 369750.9  | 369750.9      | 369750.9      | 38.2      | 374600.5  | 349.9            | 0.3               | 350.2            | 0.003          |

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
| Total Duration | 715.18s                   |
| Total Attempts | 2                         |
| Generated      | 2026-09-16T23:37:33+00:00 |

## 🧪 Test Results

| Status  | Test Name                | Duration | Attempts | Description                                       |
|:--------|:-------------------------|:---------|:---------|:--------------------------------------------------|
| ✅ PASS | LoggerForkSafetyTest     | 0.00s    | 1        | Test for logging fork safety to prevent deadlocks |
| ❌ FAIL | VLLMParamConformanceTest | 715.17s  | 1        | vLLM chat/completions parameter conformance       |

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
| test_determinism_parameters  | ❌ FAIL | 1/3 passed |
| test_logprobs                | ✅ PASS | 1/1 passed |
| test_max_tokens              | ✅ PASS | 2/2 passed |
| test_n                       | ✅ PASS | 2/2 passed |
| test_non_uniform_seeding     | ❌ FAIL | 0/1 passed |
| test_penalties               | ✅ PASS | 9/9 passed |
| test_seed_reproducibility    | ✅ PASS | 1/1 passed |
| test_stop                    | ✅ PASS | 2/2 passed |

#### Detailed Test Results

| Test Case                    | Parametrization                                                      | Status    | Message                                                                                                                                                                                                                                                       |
|:-----------------------------|:---------------------------------------------------------------------|:----------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| test_coherence_verbatim_echo | test_coherence_verbatim_echo                                         | ❌ FAILED | Traceback: AssertionError: Coherence guard failed: model did not echo the sentence verbatim (likely gibberish from a corrupted forward pass). Expected to find 'The quick brown fox jumps over the lazy dog.' in output, got: 'The user wants me to repeat... |
| test_determinism_parameters  | test_determinism_parameters[temperature-0.0]                         | ❌ FAILED | Traceback: AssertionError: temperature=0.0 was not deterministic. Output 1: '<think>   The user is asking a simple factual question about the capital of France. This is straightforward and well-known information.   </think>      Paris.', Output 2: '<... |
| test_determinism_parameters  | test_determinism_parameters[top_k-1]                                 | ❌ FAILED | Traceback: AssertionError: top_k=1 was not deterministic. Output 1: '<think>   The user is asking a simple factual question about the capital of France. This is basic general knowledge. The answer is Paris. They want a concise answer, so I'll keep it... |
| test_determinism_parameters  | test_determinism_parameters[top_p-0.01]                              | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_logprobs                | test_logprobs                                                        | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_max_tokens              | test_max_tokens[10]                                                  | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_max_tokens              | test_max_tokens[5]                                                   | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_n                       | test_n[2]                                                            | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_n                       | test_n[3]                                                            | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_non_uniform_seeding     | test_non_uniform_seeding                                             | ❌ FAILED | Traceback: Failed: Request failed for seed 8: HTTPConnectionPool(host='127.0.0.1', port=8000): Read timed out. (read timeout=30)                                                                                                                              |
| test_penalties               | test_penalties[frequency_penalty-1.2-natural_repetition-messages1]   | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_penalties               | test_penalties[frequency_penalty-1.2-repeat_trap-messages0]          | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_penalties               | test_penalties[frequency_penalty-1.2-semantic_repetition-messages2]  | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_penalties               | test_penalties[presence_penalty-1.2-natural_repetition-messages1]    | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_penalties               | test_penalties[presence_penalty-1.2-repeat_trap-messages0]           | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_penalties               | test_penalties[presence_penalty-1.2-semantic_repetition-messages2]   | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_penalties               | test_penalties[repetition_penalty-1.5-natural_repetition-messages1]  | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_penalties               | test_penalties[repetition_penalty-1.5-repeat_trap-messages0]         | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_penalties               | test_penalties[repetition_penalty-1.5-semantic_repetition-messages2] | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_seed_reproducibility    | test_seed_reproducibility                                            | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_stop                    | test_stop[stop_seq0]                                                 | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_stop                    | test_stop[stop_seq1]                                                 | ✅ PASSED |                                                                                                                                                                                                                                                               |
