# Plan: Qwen3.5-397B-A17B on QB2 via host-assisted MoE

## Context

The QB2 (4 × Blackhole p300c, 127 GiB device memory) currently tops out at Qwen3.5-122B-A10B. In serving, the 122B uses 17.1 GiB of weights, 11.2 GiB of KV pool and ~3.4 GiB of buffers per die. The host has 249 GB DDR5 and an 8-core Ryzen 7 9700X, and neither does much during decode.

A POC (`models/experimental/hybrid_moe/`) showed that a device expert cache that follows routing, with the host CPU computing missed experts, works on the 122B:
- **Hit rate:** 91.6% with half the experts treated as on-device, measured in the real decode loop.
- **Output:** coherent with misses computed on the host.
- **Measured costs:** ~0.10 ms per-layer round trip in a split-trace benchmark; CPU int4 kernel at 32 GB/s; PCIe from DDR at 14 GB/s, with the two running together at ~46 GB/s.

**Outcome wanted:** a single-user (batch 1) vLLM demo of Qwen3.5-397B-A17B on the QB2, at ≥20 tok/s. The projection is 36–41 tok/s if the round trip is engineered to ~0.1 ms per layer. Go as fast as possible. There are no schedule estimates; gates decide progress.

## Locked decisions

- **Host kernel:** int4, group-wise (torch `_weight_int4pack_mm_for_cpu`) first. Build an exact bfp4 kernel only if an accuracy gate fails.
- **Demo surface:** vLLM OpenAI API through the existing 122B adapter with `--max-num-seqs 1`. No plugin change is needed for the demo.
- **Speed floor:** ~20 tok/s is acceptable. Round-trip optimisation does not block the demo.
- **Cache policy:** dynamic global LRU, warm-started from routing frequency. Missed bytes are split between CPU and PCIe, calibrated at startup. Slices copied over PCIe land in cache slots, so promotion needs no extra traffic.

## Ground rules

- **Code location:** tt-metal branch `raahem/qwen35-hybrid-moe`, created from `qwen3_5_122b_a10b_mtp_fix` in `~/tt-metal-rel-122b`. Local commits only, never push.
- **Device use:** one device job at a time. Run in the 122B image via `run_in_image.sh` (this dir) (mounts `/hf`, `/cache`, hugepages) with the tt-metal tree bind-mounted over `/opt/tt-metal`, or with the in-tree python env once built. `QWEN35_PROGRAM_GUARD=0` for eager and debug runs.
- **Fail loudly:** raise on broken invariants, with values printed. No silent fallbacks in the hybrid path.
- **Evidence and notes:** JSON plus logs under `models/experimental/hybrid_moe/<step>/`. Nothing goes in the plugin repo's `docs/`.
- **Human approval** is needed only for deleting files (Phase 0) and before a step that would overwrite the shipped 122B bundle or caches.

## Phase 0: prerequisites

1. **Disk audit (read-only), then approved cleanup to ≥700 GB free.** Candidates: `models--raahemnabeel--qwen3-5-122b-a10b-blackholex4-backup-*`, duplicate 122B HF snapshots, GLM snapshots under `~/glm53-flash-bringup/`, old `tt-model/*` Docker images, `~/.cache/tt-model/*` for retired bundles.
2. **Get the checkpoint.** Find the 397B FP8 repo (`Qwen/Qwen3.5-397B-A17B-FP8` or the official equivalent) and download it into the HF cache. Check its config: 60 layers, 512 experts, top-10, hidden 4096, expert width 1024, shared width 1024, `full_attention_interval` 4. Record the expert tensor layout (fused `experts.gate_up_proj` or one tensor per expert) and the FP8 scale layout (`weight_scale_inv` block size).
3. **Free RAM after the 122B loads** (in-image, record `MemAvailable`). This decides whether the host library is fully resident (needs ≥235 GB free) or memory-mapped from NVMe with the page cache.

## Workstream A: hybrid machinery on the 122B

Built on the 122B because it fits fully, so every result has an all-on-device reference. Files are under `models/autoports/qwen_qwen3_5_122b_a10b/tt/`.

**A1. Expert slot cache (device side)**
- `moe.py` `Qwen35MoE.__init__` (`:198-232`): allocate `up_gate` / `down` with C slots in dim 1 instead of E. Add a per-layer device `slot_of[E]` uint16 table and a `resident[E]` mask.
- In `_routed_pairs` (`optimized_decoder.py:421-503`, the single-user decode path) and `_routed_gather` (`moe.py:472`, width 1): map router `ids` through `slot_of` (device gather/embedding) before `indices=`, and multiply `scores` by `resident[ids]`.
- Size everything that was sized E by C instead: `ones_mask` (`moe.py:256`), `_pairs_mask(G, E)` (`optimized_decoder.py:505`). The router keeps the true E.
- Sorted dispatch (`moe_dispatch.py`) is not used at width 1–2; raise if it is reached in hybrid mode.
- New module `tt/expert_cache.py`:
  - slot allocator, global LRU and frequency warm start
  - CQ1 writes of expert slices into slots, plus `slot_of` / `resident` table updates
  - events ordering CQ1 writes before the CQ0 segment that reads them
- **Gate A1:** with C = E and every expert resident, greedy decode tokens are bit-identical to the current model on the 16 POC prompts.

**A2. Per-layer decode trace split**
- In `model.py:294-298` (layer loop of `_forward_decode`), capture one trace segment per layer. Activations hand over through persistent buffers. Prologue (embedding/RoPE `:289`, `:350`) and epilogue (`:299-307`, `plus_one`) are their own segments.
- Model it on `mtp_traced.py:277-330` (named multi-trace capture and replay) and `decode_buckets.py:67-94` (per-store swap).
- Inside each MoE layer, use two segments:
  - **a:** pre-MoE plus router plus hits, ending in one packed DRAM buffer `[ids | scores | x]`.
  - **b:** add the host contribution, then reduce and the rest of the layer.
- Host step between a and b: one `ttnn.to_torch` of the packed buffer from die 0, then the host worker, then one `copy_host_to_device_tensor` of the result into a persistent buffer.
- Replay replaces the single `execute_trace` at `tt_transformers/tt/generator.py:2255-2256` when `QWEN35_HYBRID=1`. Sampling stays a separate trace.
- **Gate A2:** with C = E (no misses, host step returns zeros), tokens match A1, and the round-trip overhead per layer is reported. It must be ≤0.25 ms (floor). Then optimise toward 0.1 ms, without blocking.

**A3. Host worker**
- New `tt/host_experts.py`:
  - int4 pack of all experts, from HF bf16 for the 122B and FP8 dequantised for the 397B
  - persistent 8-thread torch pool (`torch.set_num_threads(8)`)
  - `compute(layer, ids, scores, x) -> [H]` in fp32 accumulate
  - weights in pinned RAM, or memory-mapped per Phase 0.3
- Reuse the kernel and packing from `models/experimental/hybrid_moe/bench2_cpu.py`. Reuse the expert math (gate/up split, SiLU, down, score weighting) from `models/experimental/hybrid_moe/hybrid_poc.py` `wrap_route`.
- PCIe/CPU split: at startup, measure CPU GB/s and CQ1 GB/s together (as in `bench3_contention.py`) and set `q* = B_pcie / (B_pcie + B_cpu)`. The PCIe share of each miss is written into slots by A1's cache manager and computed by the device in segment b.
- **Gate A3:** kernel ≥30 GB/s on a library larger than the CPU cache. Per-expert output PCC ≥0.99 against bf16 torch.

**A4. 122B end to end at C = 50% of experts**
- Traced decode with `QWEN35_HYBRID=1` and `QWEN35_HYBRID_SLOTS_FRAC=0.5`.
- **Gate A:**
  - hit rate ≥88%
  - teacher-forced top-1 agreement with all-on-device ≥95% over the POC prompts
  - ≥35 tok/s single user
  - per-token breakdown (device / round trips / misses) logged
- If tok/s is below 35 but the round trip is ≤0.25 ms per layer, continue. The 397B floor is 20.

## Workstream B: port the 397B

Runs alongside A, mostly on the host until B3 needs the device.

**B1. Scaffold**
- Copy `models/autoports/qwen_qwen3_5_122b_a10b` to `models/autoports/qwen_qwen3_5_397b_a17b`.
- Set `HF_MODEL_ID` (`model_config.py:21`).
- Port the coder_next large-k router top-k (`qwen_qwen3_coder_next/tt/moe.py:42-48`, `:393-406`, `ROUTER_TOPK_ROUTE_K`) into `moe.py:385`.
- Port the coder_next decode thresholds (`optimized_decoder.py:76-81`, `:418-433`: `pairs_max_tokens=32`, `pairs_min_tokens=2`).

**B2. Weight loading**
- Add FP8 block dequant (`weight_scale_inv`) to `tt/weights.py` `StreamingStateDict`. Dequantise per tensor on read, to bf16, then the existing precision policy.
- If experts are stored per tensor, reuse `qwen_qwen3_coder_next/tt/weights.py` `fuse_experts` (`:81-84`, `:180-198`).
- Same dequant path feeds the A3 int4 host library build.

**B3. Shapes and configs**
- Verify `multichip_decoder.py:99-114` fixed core counts (88/33/16/48) divide the new N-tile counts at hidden 4096 and per-die expert width 256; adjust where they don't.
- Check the divisibility asserts at `:336-340` and the `model_config.py:97-99` norm threshold at dim 4096.
- `generator_vllm.py:383` `_KV_LAYERS` → 15. Re-derive `_PER_SLOT_GIB` and `doc/context_contract.json` for 45 GDN and 15 attention layers.

**B4. Layer accuracy**
- Copy `tests/test_functional_decoder.py` `_Harness` (`:78-166`) and `tests/reference.py` (per-layer HF loading with lazy expert reads) to the new port.
- PCC ≥0.995 prefill/decode for each layer kind (GDN+MoE, attention+MoE), real weights, on the (1,4) mesh via `_MeshHarness` (`tests/test_multichip_decoder.py:138-168`).

**B5. Memory plan**
- Per-die budget from real allocations: shared weights, buffers, one 128K user's KV, and C slots per layer.
- **Gate B:** layer PCC passes, and the memory plan gives C ≥45% of experts per layer.

## Phase C: integrate

1. **Host library:** build the int4 library for all 397B experts (B2 dequant feeding A3 packing), saved as one memory-mappable file per layer under `~/.cache/hybrid-moe/qwen35_397b/`.
2. **Full run:** 60 layers with `QWEN35_HYBRID=1` and C from B5. The 16 POC prompts must produce coherent text.
3. **Routing locality:** record routing for 512 experts / top-10 (reuse `capture_routing.py`) and replay it with `sim_cache.py` / `sim_throttle.py`. Hit rate must be ≥85%.
4. **Accuracy:** teacher-forced top-1 agreement against HF reference layers on short texts ≥90%. If it fails, switch to the exact bfp4 host kernel.
5. **Performance:** breakdown and tok/s compared with `models/experimental/hybrid_moe/project.py`.
- **Gate C:** coherent output, hit rate ≥85%, ≥20 tok/s.

## Phase D: demo

1. **vLLM serving:** use the 122B adapter path, copied to the new port's `generator_vllm.py`, with `--max-num-seqs 1`. Register the architecture as the 122B is (`TTQwen3_5MoeForConditionalGeneration` registration in the plugin is reused or mirrored for the new class). If a plugin registration change is needed, make it a paired plugin PR with pre-commit evidence, per AGENTS.md.
2. **Demo stats:** a live line in the server log or a side endpoint with tok/s, hit rate and host share. Optionally a side-by-side with the 122B bundle.
3. **Fresh-start check:** from a fresh device reset, server up and a 10-turn chat with no errors.
4. **Run notes** in `models/experimental/hybrid_moe/demo/RUN_NOTES.md`: startup time, memory, measured numbers, limits.

## Verification, end to end

- **A1/A2 exactness:** bit-identical tokens against the shipped path at C = E.
- **Accuracy:** teacher-forced top-1 agreement (122B against all-on-device; 397B against HF reference layers); per-layer PCC tests in the new port's `tests/`.
- **Cache:** in-situ hit rate logged per run; routing replays through the existing simulators.
- **Performance:** per-token breakdown from the hybrid runner, compared with `project.py` (inputs: `bench1.json`, `bench2.json`, bench3 log, `routing/sim*.json`).
- **Demo:** vLLM OpenAI API chat over the 16 POC prompts plus a 10-turn conversation from a fresh device reset.

## Fallbacks

| Trigger | Fallback |
|---|---|
| A2 round trip >0.25 ms per layer | Fold the host step into fewer sync points (read back during segment a's tail); accept ~20 tok/s |
| C3 hit rate <85% | Increase C by trimming KV and buffers (smaller max context for the demo) |
| Phase 0.3 RAM headroom too low | Memory-mapped library on NVMe; cold experts load more slowly |
| Accuracy gate fails | Exact bfp4 host kernel |
| Converted weights reload slowly | `QWEN35_TENSOR_CACHE=1` for the 397B weights |
