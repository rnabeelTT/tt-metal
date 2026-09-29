## Tenstorrent Model Release Summary: Qwen/Qwen3.5-122B-A10B on P300X2

### Metadata: Qwen/Qwen3.5-122B-A10B on P300X2

```json
{
    "model_name": "Qwen3.5-122B-A10B",
    "model_repo": "Qwen/Qwen3.5-122B-A10B",
    "device": "P300X2",
    "generated_at": "2026-09-26T11:06:09+00:00",
    "report_id": "Qwen__Qwen3.5-122B-A10B_2026-09-26T110609+0000",
    "workflow": "benchmarks",
    "server_mode": "API",
    "run_command": "python run.py --dev-mode --model Qwen/Qwen3.5-122B-A10B --device p300x2 --workflow benchmarks --service-port 8131 --no-auth --skip-system-sw-validation --disable-trace-capture --limit-samples-mode smoke-test",
    "runtime_model_spec_json": "/home/raahem/tt-inference-server/workflow_logs/runtime_model_specs/runtime_model_spec_2026-09-26_11-05-58_id_qwen35-122b-autoport_Qwen3.5-122B-A10B_p300x2___H2ko11.json",
    "model_id": "id_qwen35-122b-autoport_Qwen3.5-122B-A10B_p300x2",
    "inference_engine": "vLLM",
    "tt_metal_commit": null,
    "vllm_commit": null,
    "model_impl": "qwen35-122b-autoport"
}
```

### Acceptance Criteria

- Acceptance status: ✅ `PASS`
- Model status: `EXPERIMENTAL`
- Benchmarks: 🟨 `NA` (0/1 passed, 1 NA)
- Evals: 🟨 `NA` (no blocks present)
- Spec Tests: 🟨 `NA` (no blocks present)
- Agentic Targets: 🟨 `NA` (no blocks present)
- All acceptance criteria passed.

---

### vLLM Benchmark for Qwen/Qwen3.5-122B-A10B on P300X2

| Concurrency | Num Requests | ISL | OSL | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Input (TPS) | Tput Output (TPS) | Tput Total (TPS) | Req Tput (RPS) |
|:------------|:-------------|:----|:----|:----------|:--------------|:--------------|:----------|:----------|:-----------------|:------------------|:-----------------|:---------------|
| 1           | 8            | 16  | 4   | 314.4     | 313.1         | 328.5         | 36.5      | 423.9     | 37.7             | 9.4               | 47.2             | 2.358          |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: No perf targets are configured for these sweep points, so these rows are reported for information only and are not graded.
