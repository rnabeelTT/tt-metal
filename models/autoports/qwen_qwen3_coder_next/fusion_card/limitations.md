- Two dies (`p300`, `p150x2`) run without the fabric layer boundary (its persistent receive buffers do not fit next to
  the two-die prefill) and keep the stock sampler top-k (the live-rows top-k carries 16-bit vocabulary indices; a die's
  vocabulary slice is 75968 columns on two dies).
- The device sampler supports `top_k` up to 32. This checkpoint's recommended default is `top_k 40`, which is clamped to
  32 rather than moved to host sampling.
- Agentic benchmarks (SWE-Bench Verified, LiveCodeBench), which Qwen publishes for this checkpoint, were not run; the
  evals ran on the p300x2 profile only.
- The p150x4 and p150x2 profiles run the same meshes as p300x2 / p300 but were not run on p150 boards.
