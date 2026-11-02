- MTP speculative decoding is off: the MTP verify pass still runs the unfused rows path, which is slower than the fused
  plain step.
- The card rows were measured on the default `p300x2` profile and on `p300x2-longctx` (single-user rows). The `-burst`
  and `-text` profiles run the same decode code; the p150x4 profiles run the same mesh but were not run on p150 boards.
- The device sampler supports `top_k` up to 32.
- Benchmark evals ran with thinking off; IFEval and LiveCodeBench on subsets (100 / 50 questions). The model card's own
  benchmarks (agentic coding and tool use) were not reproduced here.
- The first start converts the 250 GB bf16 checkpoint and takes about 20 minutes on p300x2; later starts about 14.
