## Tenstorrent Model Release Summary: Qwen/Qwen3.5-122B-A10B on P300X2

### Metadata: Qwen/Qwen3.5-122B-A10B on P300X2

```json
{
    "model_name": "Qwen3.5-122B-A10B",
    "model_repo": "Qwen/Qwen3.5-122B-A10B",
    "device": "P300X2",
    "generated_at": "2026-09-26 16:48:52",
    "report_id": "id_qwen35-122b-autoport_Qwen3.5-122B-A10B_p300x2_2026-09-26_16-48-52",
    "workflow": "spec_tests",
    "server_mode": "API",
    "run_command": "python run.py --dev-mode --model Qwen/Qwen3.5-122B-A10B --device p300x2 --workflow spec_tests --service-port 8131 --no-auth --skip-system-sw-validation",
    "runtime_model_spec_json": "/home/raahem/tt-inference-server/workflow_logs/runtime_model_specs/runtime_model_spec_2026-09-26_16-08-35_id_qwen35-122b-autoport_Qwen3.5-122B-A10B_p300x2__5DGKb79.json",
    "model_id": "id_qwen35-122b-autoport_Qwen3.5-122B-A10B_p300x2",
    "inference_engine": "vLLM",
    "tt_metal_commit": null,
    "vllm_commit": null,
    "model_impl": "qwen35-122b-autoport"
}
```

### Acceptance Criteria

- Acceptance status: ❌ `FAIL`
- Model status: `EXPERIMENTAL`
- Benchmarks: 🟨 `NA` (no blocks present)
- Evals: 🟨 `NA` (no blocks present)
- Spec Tests: ❌ `FAIL` (0/1 passed, 1 failed)
- Agentic Targets: 🟨 `NA` (no blocks present)

#### Blockers

- `spec.spec_tests:Vllm Chat Completions`: Vllm Chat Completions reported status=fail (attempts=1)
- `task:spec_tests`: Task 'spec_tests' failed (exit=1) after producing a report block.

---

## 📋 Summary

| Metric         | Value               |
|:---------------|:--------------------|
| Total Tests    | 2                   |
| Passed         | 1                   |
| Failed         | 1                   |
| Skipped        | 0                   |
| NA             | 0                   |
| Attempted      | 2                   |
| Success Rate   | 50.0%               |
| Total Duration | 2416.07s            |
| Total Attempts | 2                   |
| Generated      | 2026-09-26 16:48:52 |

## 🧪 Test Results

| Status  | Test Name                | Duration | Attempts | Description                                       |
|:--------|:-------------------------|:---------|:---------|:--------------------------------------------------|
| ✅ PASS | LoggerForkSafetyTest     | 0.00s    | 1        | Test for logging fork safety to prevent deadlocks |
| ❌ FAIL | VLLMParamConformanceTest | 2416.06s | 1        | vLLM chat/completions parameter conformance       |

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
| test_coherence_verbatim_echo | ❌ FAIL | 0/1 passed |
| test_determinism_parameters  | ✅ PASS | 3/3 passed |
| test_logprobs                | ✅ PASS | 1/1 passed |
| test_max_tokens              | ✅ PASS | 2/2 passed |
| test_n                       | ✅ PASS | 2/2 passed |
| test_non_uniform_seeding     | ❌ FAIL | 0/1 passed |
| test_penalties               | ❌ FAIL | 1/9 passed |
| test_seed_reproducibility    | ❌ FAIL | 0/1 passed |
| test_stop                    | ❌ FAIL | 1/2 passed |

#### Detailed Test Results

| Test Case                    | Parametrization                                                      | Status    | Message                                                                                                                        |
|:-----------------------------|:---------------------------------------------------------------------|:----------|:-------------------------------------------------------------------------------------------------------------------------------|
| test_coherence_verbatim_echo | test_coherence_verbatim_echo                                         | ❌ FAILED | Traceback: TypeError: argument of type 'NoneType' is not iterable                                                              |
| test_determinism_parameters  | test_determinism_parameters[temperature-0.0]                         | ✅ PASSED |                                                                                                                                |
| test_determinism_parameters  | test_determinism_parameters[top_k-1]                                 | ✅ PASSED |                                                                                                                                |
| test_determinism_parameters  | test_determinism_parameters[top_p-0.01]                              | ✅ PASSED |                                                                                                                                |
| test_logprobs                | test_logprobs                                                        | ✅ PASSED |                                                                                                                                |
| test_max_tokens              | test_max_tokens[10]                                                  | ✅ PASSED |                                                                                                                                |
| test_max_tokens              | test_max_tokens[5]                                                   | ✅ PASSED |                                                                                                                                |
| test_n                       | test_n[2]                                                            | ✅ PASSED |                                                                                                                                |
| test_n                       | test_n[3]                                                            | ✅ PASSED |                                                                                                                                |
| test_non_uniform_seeding     | test_non_uniform_seeding                                             | ❌ FAILED | Traceback: Failed: Request failed for seed 0: 'NoneType' object has no attribute 'strip'                                       |
| test_penalties               | test_penalties[frequency_penalty-1.2-natural_repetition-messages1]   | ❌ FAILED | Traceback: AttributeError: 'NoneType' object has no attribute 'lower'                                                          |
| test_penalties               | test_penalties[frequency_penalty-1.2-repeat_trap-messages0]          | ❌ FAILED | Traceback: AttributeError: 'NoneType' object has no attribute 'lower'                                                          |
| test_penalties               | test_penalties[frequency_penalty-1.2-semantic_repetition-messages2]  | ❌ FAILED | Traceback: AttributeError: 'NoneType' object has no attribute 'lower'                                                          |
| test_penalties               | test_penalties[presence_penalty-1.2-repeat_trap-messages0]           | ❌ FAILED | Traceback: AttributeError: 'NoneType' object has no attribute 'lower'                                                          |
| test_penalties               | test_penalties[presence_penalty-1.2-semantic_repetition-messages2]   | ❌ FAILED | Traceback: AttributeError: 'NoneType' object has no attribute 'lower'                                                          |
| test_penalties               | test_penalties[repetition_penalty-1.5-natural_repetition-messages1]  | ❌ FAILED | Traceback: AttributeError: 'NoneType' object has no attribute 'lower'                                                          |
| test_penalties               | test_penalties[repetition_penalty-1.5-repeat_trap-messages0]         | ❌ FAILED | Traceback: AttributeError: 'NoneType' object has no attribute 'lower'                                                          |
| test_penalties               | test_penalties[repetition_penalty-1.5-semantic_repetition-messages2] | ❌ FAILED | Traceback: AttributeError: 'NoneType' object has no attribute 'lower'                                                          |
| test_penalties               | test_penalties[presence_penalty-1.2-natural_repetition-messages1]    | ✅ PASSED |                                                                                                                                |
| test_seed_reproducibility    | test_seed_reproducibility                                            | ❌ FAILED | Traceback: requests.exceptions.ReadTimeout: HTTPConnectionPool(host='127.0.0.1', port=8131): Read timed out. (read timeout=30) |
| test_stop                    | test_stop[stop_seq0]                                                 | ❌ FAILED | Traceback: TypeError: argument of type 'NoneType' is not iterable                                                              |
| test_stop                    | test_stop[stop_seq1]                                                 | ✅ PASSED |                                                                                                                                |
