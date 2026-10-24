- MTP speculative decoding is off on the multi-die profiles: its verify / commit pass still runs the unfused path. A
  fused multi-token verify would bring it back; until then plain fused decode is faster.
- One die (`p150`, `p100`) keeps the blackhole-fast decode path (v2 MoE, MTP on): the fused MoE kernels read 1 or 2
  up|gate columns per DRAM bank and expert, one die holds 4 (and p100 has 7 banks); the GDN / attention ops were not run
  on one die.
- The p150x4 and p150x2 profiles run the same meshes as p300x2 / p300 but were not run on p150 boards.
- The device sampler supports `top_k` up to 32.
- Benchmark evals ran on subsets (40 / 100 / 50 questions) on the p300 profile only.
