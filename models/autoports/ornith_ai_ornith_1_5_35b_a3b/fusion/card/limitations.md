- The single-user MTP pass still runs the v2 (unfused) rows path for its verify / commit; on random-token prompts the
  MTP-on single-user TPOT is above the plain fused decode step (p300: 19.4 vs 15.1 ms), on real text MTP still wins
  (74 vs 66 tok/s). Moving the MTP pass onto the fused kernels is the next step.
- One die (`p150`, `p100`) keeps the v2 MoE: the fused kernels read 1 or 2 up|gate columns per DRAM bank and expert,
  one die holds 4 (and p100 has 7 banks).
- The p150x4 and p150x2 profiles run the same meshes as p300x2 / p300 but were not run on p150 boards.
- The device sampler supports `top_k` up to 32.
- Benchmark evals ran on subsets (40 / 100 / 50 questions) on the p300 profile only.
