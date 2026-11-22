---
name: blackhole-decode-perf
description: Decode-speed review for a model that already runs correctly on Blackhole (QB2, p150/p300). Use once a new base model serves with quality matching its CPU reference, before packaging, and whenever planning decode optimization; measure against the DRAM-bandwidth floor, keep a per-layer op budget, and apply the fixes that took Ornith / Qwen3 / gpt-oss decode 2-4x faster with unchanged evals.
---

# Blackhole decode playbook

Source: the "Blackhole Decode Playbook" (https://claude.ai/artifact/QyM6VmQFFZLdaoWzRYGYxQ, read it with the Artifact
tool if you have one). It covers Ornith-1.5-35B-A3B (58.5 -> 217.9 tok/s, one user, four dies), Qwen3-Coder-Next
(42.8 -> 177.7), Qwen3.5-122B-A10B (45.9 -> 131.8), Qwen3-Coder-30B-A3B (49.5 -> 95.5) and gpt-oss-20b (88 -> 305).
Every eval stayed within one standard error. The measured numbers behind each rule are in `reference.md`, next to
this file.

**When:** after the base model is running and its quality has been judged against the CPU reference
(`hybrid-moe-bringup` rule 2). Do not optimize a model whose numerics are not settled yet: every fix below is a
numerics change that must be re-gated.

## The model to keep in your head

Decode is a streaming problem. One user's token reads every active weight out of DRAM once, so the speed limit is
DRAM bandwidth (8 banks per die, ~512 GB/s nominal), not math. Our bringups ran at ~7% of that limit. The weights were
already fast (dense matmuls at ~63% of bandwidth); the time went to the gaps around them: ~85-125 small ops per layer,
four chip-to-chip collectives per layer, 1-row work in 32-row tiles, and host round trips.

## Procedure

1. **Compute the floor first.** Floor (ms/token) = active weight bytes read per token / (dies x 512 GB/s). Report
   every decode result as tok/s **and** % of floor, per model and per op. An op far below the floor is a target; one
   near it is done.
2. **Build a per-layer budget from a traced device profile** (signposted, decode trace replay, not eager). List us
   per layer for boundary / mixer (attention, DeltaNet, ...) / MoE / outside (embedding, LM head, sampler), plus the op
   count and the idle-gap time of each group. Check the sum matches the measured step (within a few %). Re-plan from
   this table after every change, never from projections.
3. **Price the op count.** Each op costs ~3-5 us including the gap after it, however little it does. Ops per layer x
   that is a cost you can compute before profiling. Aim for single-digit programs per layer on the decode path.
4. **Attack in this order** (largest wins first on four dies), keeping each change only if it measures faster:
   1. **Fuse the MoE**: one program with the router's top-k inside, streaming each chosen expert in place from the
      bank that holds it, readers placed beside each bank (Ornith: 28 ops -> 3, 177 -> 38 us per layer).
   2. **Fuse the layer boundary**: every die keeps the full residual; one fabric op does the exchange + residual add +
      RMSNorm, and the next projection / router starts streaming inside it (4 collectives + adds + norms, ~103 us ->
      one 7.7 us op).
   3. **Fuse the mixer** into a few programs (DeltaNet 46 ops -> 4, attention 41 -> 7).
   4. **Do row work on rows**: cut tile-wide SFPU/pack work down to the live row(s), spread small heads over more
      cores, and do the sampler as a per-core top-k over the live rows plus a merge (355 -> 94 us).
   5. **LM head**: per-bank streaming and top-k inside the LM head (gpt-oss reads at ~504 GB/s vs our 274).
5. **Build every fused op for 1 to 32 rows** and measure 1, 8 and 32 users at each change (tok/s per user and total).
   Single- and multi-user decode share one path.
6. **Keep the loop on device.** Sampling, accept logic and draft loops belong inside the trace. Count host round trips
   per token and measure host time apart from device time.
7. **Decide on speculative decoding with numbers.** Measure accepted tokens per pass on real text at the model's
   default sampling, and the verify cost on the *fused* path. Break-even pass time = plain step x accepted tokens.
   Build or keep it only if the projection clears a set bar (e.g. +15%). See the hybrid note below: the answer differs.
8. **Stop at diminishing returns**, not at a checklist: stop when changes buy under ~3% of served speed and less than
   ~10% is left above the floor you can actually reach.

## Gates on every change

- Fixed-prompt greedy output check after each change. **Identical output after a numerics change means the change is
  not running**; find out why before measuring speed.
- Op-level PCC for each new kernel, a coherence probe on sampled long text, and the teacher-forced comparison against
  the CPU reference (KL, top-1) at milestones. Full evals at milestones only.
- Prefill time (TTFT at several lengths) must not get worse; check at each milestone.
- Benchmark the **server as shipped** (vLLM, real chat text; random-token prompts mislead with speculation), including
  decode at long context.

## In the hybrid overlay

The device part of a hybrid model (attention, layer boundary, static and fetch expert slots, LM head) follows the
rules above unchanged. Three things differ:

- **The step is not DRAM-bound.** Each MoE layer makes host turns (CPU experts, fetches), so the chip waits on the host
  and the host waits on the chip. Before fusing anything, split a decode pass into *host blocked on chips*, *host
  work* and *device time* (Inkling-Small: 29.4 ms verify pass = ~15 ms host blocked on chips + ~12-14 ms host work,
  plus 3.9 ms draft). Device fusion only shortens the first part.
- **CPU experts are host-DDR-bound.** Measure host read bandwidth (Inkling's QB2: DIMMs at 3600 MT/s, ~46 GB/s, kernel
  reaches ~36 GB/s) and the device hit rate in decode (98% on Inkling-Small, ~88% in a long prefill). The CPU-expert
  wait is hit-rate x bandwidth; check PCIe link width too (`lspci -vv`: our QB2 cards run Gen4 x4, not x8).
- **Speculation pays here** (DFlash, MTP): a pass is dominated by fixed host turns, so verifying 3-4 tokens costs about
  one pass. The playbook's "MTP stays off" result is for fully on-device fused decode at ~4.6 ms/token; redo the
  break-even arithmetic from step 7 with your own numbers rather than copying either answer.

## Evidence to leave

In the model's HANDOFF: the floor arithmetic, the per-layer budget table before and after each step (us, op count,
gaps), tok/s at 1 / 8 / 32 users with % of floor, the greedy and teacher-forced quality results per step, and the
changes that measured slower and were dropped (with numbers), so the next person does not retry them.
