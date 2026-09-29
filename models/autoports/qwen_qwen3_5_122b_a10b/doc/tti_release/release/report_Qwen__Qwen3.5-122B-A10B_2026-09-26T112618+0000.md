## Tenstorrent Model Release Summary: Qwen/Qwen3.5-122B-A10B on P300X2

### Metadata: Qwen/Qwen3.5-122B-A10B on P300X2

```json
{
    "model_name": "Qwen3.5-122B-A10B",
    "model_repo": "Qwen/Qwen3.5-122B-A10B",
    "device": "P300X2",
    "generated_at": "2026-09-26T11:26:18+00:00",
    "report_id": "Qwen__Qwen3.5-122B-A10B_2026-09-26T112618+0000",
    "workflow": "release",
    "server_mode": "API",
    "run_command": "python run.py --dev-mode --model Qwen/Qwen3.5-122B-A10B --device p300x2 --workflow release --service-port 8131 --no-auth --skip-system-sw-validation --limit-samples-mode ci-nightly",
    "runtime_model_spec_json": "/home/raahem/tt-inference-server/workflow_logs/runtime_model_specs/runtime_model_spec_2026-09-26_11-06-09_id_qwen35-122b-autoport_Qwen3.5-122B-A10B_p300x2_1iQaWYk2.json",
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
- Benchmarks: 🟨 `NA` (0/23 passed, 23 NA)
- Evals: ✅ `PASS` (1/2 passed, 1 waived)
- Spec Tests: 🟨 `NA` (no blocks present)
- Agentic Targets: 🟨 `NA` (no blocks present)
- All acceptance criteria passed.

---

### Accuracy Evaluations for Qwen/Qwen3.5-122B-A10B on P300X2

| Task                      | Tolerance | Published Score | Published Score Ref                                                         | gpu_reference_score_ref | Score | Ratio to Published | Ratio to Reference | Accuracy Check | priority | mean_seconds_per_task |
|:--------------------------|:----------|:----------------|:----------------------------------------------------------------------------|:------------------------|:------|:-------------------|:-------------------|:---------------|:---------|:----------------------|
| ifeval                    | 0.05      | 93.4            | https://huggingface.co/Qwen/Qwen3.5-122B-A10B (IFEval, thinking mode)       | TBD                     | 92.66 | 0.9921             | N/A                | ✅ PASS        | must     | 3.898                 |
| gpqa_diamond_cot_zeroshot | 0.05      | 86.6            | https://huggingface.co/Qwen/Qwen3.5-122B-A10B (GPQA Diamond, thinking mode) | TBD                     | 77.5  | 0.8949             | N/A                | ❌ FAIL        | must     | 19.57                 |

Note: The ratio to published scores defines if eval ran roughly correctly, as the exact methodology of the model publisher cannot always be reproduced. For this reason the accuracy check is based first on being equivalent to the GPU reference within a +/- tolerance. If a value GPU reference is not available, the accuracy check is based on the direct ratio to the published score.

---

### vLLM Benchmark for Qwen/Qwen3.5-122B-A10B on P300X2

| Concurrency | Num Requests | ISL    | OSL  | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Input (TPS) | Tput Output (TPS) | Tput Total (TPS) | Req Tput (RPS) |
|:------------|:-------------|:-------|:-----|:----------|:--------------|:--------------|:----------|:----------|:-----------------|:------------------|:-----------------|:---------------|
| 1           | 8            | 128    | 128  | 393.6     | 397.1         | 402.7         | 37.7      | 5185.3    | 24.7             | 24.7              | 49.4             | 0.193          |
| 32          | 256          | 128    | 128  | 12286.7   | 12266.0       | 12414.9       | 127.0     | 28418.9   | 144.1            | 144.1             | 288.2            | 1.126          |
| 1           | 4            | 128    | 1024 | 378.0     | 378.2         | 389.8         | 37.8      | 39002.6   | 3.3              | 26.3              | 29.5             | 0.026          |
| 32          | 128          | 128    | 1024 | 12268.5   | 12274.4       | 12365.7       | 140.4     | 155889.0  | 26.3             | 210.2             | 236.5            | 0.205          |
| 1           | 4            | 1024   | 128  | 1181.1    | 1180.8        | 1185.4        | 37.7      | 5971.9    | 171.5            | 21.4              | 192.9            | 0.167          |
| 32          | 128          | 1024   | 128  | 37889.0   | 37897.2       | 37940.3       | 126.6     | 53965.3   | 607.2            | 75.9              | 683.1            | 0.593          |
| 1           | 4            | 2048   | 128  | 1966.8    | 1965.3        | 1976.1        | 37.8      | 6763.6    | 302.8            | 18.9              | 321.7            | 0.148          |
| 32          | 128          | 2048   | 128  | 62968.6   | 62934.1       | 63100.8       | 126.6     | 79041.8   | 829.1            | 51.8              | 880.9            | 0.405          |
| 1           | 4            | 4096   | 128  | 3906.2    | 3907.5        | 3913.3        | 37.9      | 8721.3    | 469.6            | 14.7              | 484.3            | 0.115          |
| 32          | 128          | 4096   | 128  | 124618.7  | 124571.5      | 124699.1      | 127.2     | 140775.5  | 931.1            | 29.1              | 960.2            | 0.227          |
| 1           | 2            | 8192   | 128  | 7766.8    | 7766.8        | 7769.5        | 38.0      | 12590.3   | 650.6            | 10.2              | 660.8            | 0.079          |
| 31          | 62           | 8192   | 128  | 240594.3  | 240589.2      | 240627.7      | 129.2     | 256996.8  | 988.1            | 15.4              | 1003.6           | 0.121          |
| 1           | 2            | 8192   | 1024 | 7768.5    | 7768.5        | 7770.7        | 38.0      | 46674.7   | 175.5            | 21.9              | 197.5            | 0.021          |
| 28          | 56           | 8192   | 1024 | 217310.8  | 217295.8      | 217354.4      | 141.2     | 361761.6  | 634.0            | 79.3              | 713.3            | 0.077          |
| 1           | 2            | 10000  | 1024 | 9738.4    | 9738.4        | 9764.6        | 38.0      | 48643.1   | 205.6            | 21.1              | 226.6            | 0.021          |
| 23          | 46           | 10000  | 1024 | 223333.6  | 223286.5      | 223416.6      | 141.6     | 368208.5  | 624.6            | 64.0              | 688.6            | 0.062          |
| 1           | 2            | 16384  | 128  | 15580.0   | 15580.0       | 15600.8       | 38.1      | 20422.7   | 802.2            | 6.3               | 808.5            | 0.049          |
| 15          | 30           | 16384  | 128  | 233489.8  | 233514.0      | 233516.2      | 112.4     | 247763.4  | 991.9            | 7.7               | 999.7            | 0.060          |
| 1           | 1            | 32768  | 128  | 31406.9   | 31406.9       | 31406.9       | 38.4      | 36287.3   | 903.0            | 3.5               | 906.5            | 0.028          |
| 7           | 7            | 32768  | 128  | 219613.7  | 219666.4      | 219667.9      | 98.6      | 232139.4  | 988.1            | 3.9               | 991.9            | 0.030          |
| 1           | 1            | 65536  | 128  | 63988.3   | 63988.3       | 63988.3       | 39.0      | 68944.1   | 950.6            | 1.9               | 952.4            | 0.015          |
| 3           | 3            | 65536  | 128  | 191795.6  | 191895.0      | 191896.5      | 55.3      | 198816.9  | 988.9            | 1.9               | 990.8            | 0.015          |
| 1           | 1            | 131072 | 128  | 132608.7  | 132608.7      | 132608.7      | 40.0      | 137689.8  | 951.9            | 0.9               | 952.9            | 0.007          |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: No perf targets are configured for these sweep points, so these rows are reported for information only and are not graded.
