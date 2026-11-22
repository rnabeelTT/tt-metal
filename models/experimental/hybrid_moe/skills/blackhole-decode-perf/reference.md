# Blackhole decode playbook: measured numbers and kernel notes

From the "Blackhole Decode Playbook" (https://claude.ai/artifact/QyM6VmQFFZLdaoWzRYGYxQ). All on a QuietBox 2 (four
Blackhole dies, p300x2), one user, real chat text, October 2026, unless noted. Sources: the Ornith fusion handoff
(branch `ornith_fusion`, `fusion/HANDOFF.md`) and each model's Hugging Face card.

## Results

| Model | What carried over | 1 user, tok/s | 32 users, total tok/s | Quality |
|---|---|---|---|---|
| Ornith-1.5-35B-A3B | all of it (origin) | 58.5 -> 217.9 | 354 -> 614 | within 1 SE |
| Qwen3-Coder-Next | all; 512-expert mask | 42.8 -> 177.7 | 211 -> 417 | within 1 SE |
| Qwen3.5-122B-A10B | all; 3-tile hidden size | 45.9 -> 131.8 | 197 -> 344 | GPQA identical |
| Qwen3-Coder-30B-A3B | fused MoE, expert-parallel | 49.5 -> 95.5 | 617 -> 853 | within 1 SE |
| gpt-oss-20b (reference) | its own fused path | 88 -> 305 | 1,176 -> 1,753 (P150x4, 128-token prompts) | BFP8 experts kept for accuracy |

32-user totals: 1024-token prompts, 256 output tokens. gpt-oss-120b: 15.28 -> 4.06 ms/token.

## Ornith budget (original bundle, 19.15 ms/token)

MoE 7.0 ms, DeltaNet mixer 6.2, layer boundary 4.1, attention 1.6, other 0.25. Floor ~1.4 ms (2.9 GB active weights
across four dies). After fusion: 4.56 ms (~31% of floor). gpt-oss-20b: 3.27 ms at 63% of its floor.

One DeltaNet layer: 85 ops in 488 us -> 14 ops in 145.5 us.

| Group | Before | After |
|---|---|---|
| MoE | 28 ops, 177 us | 3 ops, 38 us |
| DeltaNet mixer | 46 ops, 206 us (45 us of it gaps) | 4 ops, 66 us |
| Attention mixer | 41 ops, 156 us | 7 ops, 67 us |
| Boundary | 4 collectives + add + norm + copies, ~103 us | one fabric op, 7.7 us |
| Routed expert read (gate/up, per die) | 41.4 us, 108 GB/s (21%) | 16.9 us, 264 GB/s (52%) |
| DeltaNet head | 68 us | 29 us row-only, 21.6 us split over 4 cores |
| Sampler | 355 us | 94 us |

Fabric boundary + two fused-in matmuls saved ~1.3 ms/token; row work + sampler ~0.4 ms/token. Ornith at 8 users:
36.7 -> 18.9 ms per step (several users read the union of their experts once).

## Kernel notes that cost days

- Fused ops are `generic_op` kernels in Python-side model code: they ship without rebuilding ttnn.
- Fusing one sub-block buys single digits (Ornith's first fused DeltaNet step: -8% at 8 users, +5% at 1). The large
  wins came from fusing whole groups.
- A fused op still pays ~3.5 us ramp and tail, capping a 6.8 MB projection near 380 GB/s. That is why the boundary is
  folded into a neighbouring matmul's program.
- Interleaved layout: page p sits in bank p mod 8, so an expert's column is contiguous in one bank and can be read in
  place. gpt-oss balanced banks with a rotated duplicate copy instead (costs DRAM; its decode-only copies are ~5.9 GB
  per chip).
- Two banks sharing one NoC row ran their cores 40% slower; moving the second bank's readers one row inward, reading
  on the other NoC, fixed it.
- Restore the NoC read state before a kernel exits (a leaked virtual channel slowed prefill 3%). Never write into
  another core's buffer without a handshake from it in the same program (a served hang).
- A 1-row all-reduce padded to 64 bfp8 tiles took 9.0 us; packed into 2 bf16 tiles, 4.0 us. Check what the collective
  actually moves.
- The fabric boundary covers 1-4 rows; at 8 rows it matched the old path, so wider batches keep the replicated
  all-reduce. gpt-oss's fused path likewise serves at most 4 users.
- Build model-wide objects (persistent L1 buffers) when the layers are built. Building lazily at the first decode
  shifted L1 addresses between prefill warmup and trace capture, and capture failed.
- `VectorMode::R` covers row 0 in two SFPU iterations, not one. Scalar RISC code is slow (~23-50 cycles per value):
  keep data in registers and split work across both data-movement RISCs.
- Sampler: 100 cores each keep a sorted top-32 of their slice of the live rows, one core heap-merges; ties ordered by
  index (dropped ~15 tie-break ops).
- Each port found a bug the next model would have hit: a 256-bit expert mask that dropped experts past 255, a boundary
  kernel that rejected a 3-tile hidden size, a sampler index that wrapped at 65,536 vocab columns per die. Test the
  edges of your model's sizes.

## Speculative decoding on Ornith (why it stayed off there)

MTP as built: four CPU round trips per pass (5.2 ms) and an unfused verify; 63-81 tok/s vs 218 plain. At 2.70-2.71
accepted tokens per pass, a pass must take under 12.4 ms to match plain decode and under 10.8 ms to be 15% faster;
the best realistic design landed at 11.5-13.5 ms. gpt-oss-20b's EAGLE-3 keeps the whole loop on device, a verify costs
~1.5 plain steps, and at 1.72 tokens kept per step it gains 1.17x.

## Still on the table (Ornith)

Programs resident across the layer that prefetch the next layer's weights during the boundary; LM head per-bank
stream + top-k (written, unfinished); the boundary beyond 4 rows and per-expert row selection at 32 users; two-die
and one-die paths; new mixers (Kimi delta attention, MLA, hyper-connections) need their own fused kernels.
