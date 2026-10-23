- The single-user MTP pass still runs the published (unfused) rows path for its verify / commit; only its MoE uses the
  fused kernels. Moving the whole MTP pass onto the fused path is the next step.
- The card rows were measured on the default `p300x2` profile. The `-burst`, `-longctx` and `-text` profiles run the
  same decode code (they change prefill batching, chunking and the vision tower); the p150x4 profiles run the same mesh
  but were not run on p150 boards.
- The device sampler supports `top_k` up to 32.
- Benchmark evals ran on subsets (40 / 100 / 50 questions) with thinking off.
