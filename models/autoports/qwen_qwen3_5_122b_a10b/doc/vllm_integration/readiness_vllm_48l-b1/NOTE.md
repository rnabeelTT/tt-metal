Single-slot benchmark run (`--max-num-seqs 1 --max-model-len 262144`, stages serve + benchmark only; `logs/09/vllm-48l-b1.log`).
Only the benchmark files belong to this run. `server.log` was overwritten by the next run before the snapshot (the misattributed copy was deleted after review round 2);
the sampling / qualitative files of the 32-slot run were removed from this folder (they live in `readiness_vllm_48l/`).
