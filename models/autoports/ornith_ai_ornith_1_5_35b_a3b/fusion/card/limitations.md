- MTP speculative decoding is off on the multi-die profiles: the MTP verify pass still runs the unfused rows path, and
  with it on, single-user real-text decode is slower than the fused plain step (four dies: 86 vs 218 tok/s). A fused
  multi-token verify is the way back to MTP.
- Two dies (`p300`, `p150x2`) run without the fabric layer boundary: its persistent receive buffers do not fit next to
  the two-die prefill's circular buffers.
- Two dies also keep the stock sampler top-k: the live-rows top-k carries vocabulary indices in 16 bits, and a die's
  slice of the 248320-token vocabulary is 124160 columns on two dies (62080 on four).
- One die (`p150`, `p100`) keeps the original MoE: the fused kernels read 1 or 2 up|gate columns per DRAM bank and
  expert, one die holds 4 (and p100 has 7 banks).
- The p150x4 and p150x2 profiles run the same meshes as p300x2 / p300 but were not run on p150 boards.
- The device sampler supports `top_k` up to 32.
- Benchmark evals ran on subsets (40 / 100 / 50 questions, plus the full 198-question GPQA-Diamond for this build) on the
  p300 profile only.
