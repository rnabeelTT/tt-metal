- p150x4 and p150x2 run the same meshes as p300x2 / p300 but were not run on p150 boards; the p150 profile ran on one
  p300c die (single-user rows only).
- Every newly admitted request re-captures the decode traces (as in the published bundle), which shows up as TTFT under
  concurrent load.
- The evals ran on the p300x2 profile only. Agentic benchmarks (SWE-Bench Verified, LiveCodeBench) were not run.
