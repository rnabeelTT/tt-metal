## Tenstorrent Model Release Summary: Qwen/Qwen3.5-122B-A10B on P300X2

### Metadata: Qwen/Qwen3.5-122B-A10B on P300X2

```json
{
    "model_name": "Qwen3.5-122B-A10B",
    "model_repo": "Qwen/Qwen3.5-122B-A10B",
    "device": "P300X2",
    "generated_at": "2026-09-26T20:17:31+00:00",
    "report_id": "Qwen__Qwen3.5-122B-A10B_2026-09-26T201731+0000",
    "workflow": "evals",
    "server_mode": "API",
    "run_command": "python run.py --dev-mode --model Qwen/Qwen3.5-122B-A10B --device p300x2 --workflow evals --service-port 8131 --no-auth --skip-system-sw-validation",
    "runtime_model_spec_json": "/home/raahem/tt-inference-server/workflow_logs/runtime_model_specs/runtime_model_spec_2026-09-26_19-04-19_id_qwen35-122b-autoport_Qwen3.5-122B-A10B_p300x2_S_LDRDdr.json",
    "model_id": "id_qwen35-122b-autoport_Qwen3.5-122B-A10B_p300x2",
    "inference_engine": "vLLM",
    "tt_metal_commit": null,
    "vllm_commit": null,
    "model_impl": "qwen35-122b-autoport"
}
```

### Acceptance Criteria

- Acceptance status: ❌ `FAIL`
- Model status: `FUNCTIONAL`
- Benchmarks: 🟨 `NA` (no blocks present)
- Evals: ❌ `FAIL` (0/2 passed, 2 failed)
- Spec Tests: 🟨 `NA` (no blocks present)
- Agentic Targets: 🟨 `NA` (no blocks present)

#### Blockers

- Accuracy check failed. (2 blocks)
  - `evals:LLM Eval — ifeval`
  - `evals:LLM Eval — gpqa_diamond_cot_zeroshot`

---

### Accuracy Evaluations for Qwen/Qwen3.5-122B-A10B on P300X2

| Task                      | Tolerance | Published Score | Published Score Ref                                                         | gpu_reference_score_ref | Score | Ratio to Published | Ratio to Reference | Accuracy Check | priority | mean_seconds_per_task |
|:--------------------------|:----------|:----------------|:----------------------------------------------------------------------------|:------------------------|:------|:-------------------|:-------------------|:---------------|:---------|:----------------------|
| ifeval                    | 0.05      | 93.4            | https://huggingface.co/Qwen/Qwen3.5-122B-A10B (IFEval, thinking mode)       | TBD                     | 87.99 | 0.942              | N/A                | ❌ FAIL        | must     | 13.08                 |
| gpqa_diamond_cot_zeroshot | 0.05      | 86.6            | https://huggingface.co/Qwen/Qwen3.5-122B-A10B (GPQA Diamond, thinking mode) | TBD                     | 68.69 | 0.7932             | N/A                | ❌ FAIL        | must     | 14.95                 |

Note: The ratio to published scores defines if eval ran roughly correctly, as the exact methodology of the model publisher cannot always be reproduced. For this reason the accuracy check is based first on being equivalent to the GPU reference within a +/- tolerance. If a value GPU reference is not available, the accuracy check is based on the direct ratio to the published score.
