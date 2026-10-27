# Ornith-1.5-35B-A3B decode fusion: handoff

Goal: bring single-user decode of Ornith 1.5 35B (published as `raahemnabeel/ornith-1.5-35b-a3b`, tt-model package
`ornith-1.5-35b-a3b-v2`) into gpt-oss-blackhole-fast territory by fusing the decode layer, the way
`jashansinghTT/gpt-oss-20b-blackhole-fast` / `-120b-blackhole-fast` did (about 50 ops/layer down to 8, 18% to 55% of DRAM
bandwidth, 88 to 305 tok/s on 4 chips). Everything must be reusable for the coming Ornith 1.5 122B bringup (same
architecture, larger; it will need in-place expert streaming, not duplicate weights) and, where possible, for the other
MoE ports (Qwen3-Coder-30B, Qwen3-Coder-Next, Qwen3.5-122B-A10B, GLM-4.5-Air).

## Ground rules

- Never push to existing branches or existing HF repos. Work on `ornith_fusion` (this branch) or new branches off it;
  any new package or HF repo gets a new name.
- Base the work on the latest published code, never the v1 repos/trees. This branch descends from
  `c33d9745fa2` (tip of `ornith_v2_mtp_fix`), which is what the published image `tt-model/ornith-1.5-35b-a3b-v2:0fe927ac6e21` was built from.
- Move fast. Measure, change, measure again. Keep evals to what a change needs: a greedy-output check for every change,
  a perf row or two, and the scorecard's evals only before calling a milestone done.
- Don't read token/credential files. The HF login on the box is enough; override only `HF_HUB_CACHE` / `HF_DATASETS_CACHE`.

## Branch contents

| Commit / path | What |
|---|---|
| `242796b435c` | `ttnn.experimental.kda.recurrent_decode` + `deepseek_prefill.kda_decode_conv` ported from `origin/glm_5_3_flash`, extended for grouped q/k heads (`num_qk_heads`), per-head decay (`per_head_decay`) and a partial active batch (`batch`); `TPGatedDeltaNet._forward_decode_fused` in `models/demos/blackhole/qwen36/tt/gdn/tp.py` behind `QWEN35_GDN_FUSED_DECODE=1` (off by default). Op test: `tests/test_gdn_fused_ops.py` (PCC 1.000000 at TP2 and TP4 head shapes, B = 1/2/4/32). |
| `b8aff3d95dd` | The flag requests the fp32 GDN state through the precision policy (the port checks that the built model matches its policy). |
| `fusion/tools/` | See below. Paths default to `$HOME/tt-metal` (this repo, `METAL`), `$HOME/ornith-fusion/runs` (`RUNS`), `$HOME/lm-eval-poc` (`LMEVAL`). |
| `fusion/results/` | All measurements quoted here. |

### Tools

- `Dockerfile.poc`: two stages. The builder runs `build_metal.sh` on this tree (dev image `ubuntu-22.04-dev-amd64`, a git stub
  tag so the version string matches). The runtime stage overlays the rebuilt `build_Release/lib` + `_ttnn.so` + the new ops'
  kernel sources + the changed Python files on the published image.
  - Build it: `docker build --build-context metalsrc=$METAL -t tt-model/ornith-1.5-35b-a3b-v2:<tag> -f Dockerfile.poc .`
  - **Every changed Python file must get its own COPY line in the runtime stage**, because the builder excludes `models/`. Forgetting this once
    made a "POC" run that was really the baseline. Better: rebuild with `models/` copied wholesale.
  - Tag the builder too (`--target builder -t ornith-gdnpoc-builder`); `Dockerfile.prof` takes the Tracy binaries from it.
- `Dockerfile.prof`: the POC image + Tracy capture/csvexport + pytest + tt-perf-report.
- `serve.sh <name> <image> <p300|p300x2|p150> [--env ...]`: the published `tt-model serve` command (thinking off, port 20000).
  - Weights come from `~/.cache/huggingface` (`HF_HUB_OFFLINE=1`). Pass `KCACHE=<dir>` to keep a changed build's kernel cache separate.
  - Useful envs: `QWEN35_MTP=0` (MTP off), `QWEN35_GDN_FUSED_DECODE=1`.
- `bench.sh <container> <label> [isl:osl:users:N ...]`: `vllm bench serve` inside the container; JSON goes to `$RUNS/<label>/`.
  `summarize.py A B` prints them side by side.
- `greedy.py run <label>` / `compare A B`: 8 prompts at temperature 0, sent sequentially (MTP path) and 8-wide concurrently.
  - **Bit-identical output after a numerics change means the change isn't running.**
- `eval.sh <label> [r1_gpqa_diamond|ifeval|livecodebench_v6]`: the scorecard's abbreviated evals against :20000.
  - Runs from the evals-common lm-eval (`tstescoTT/lm-evaluation-harness` @ `be23028b`); add `livecodebench_v6.yaml` under `lm_eval/tasks/livecodebench/`.
  - Needs `sympy math_verify latex2sympy2_extended` in its venv, and a private `NLTK_DATA` (punkt, punkt_tab). NLTK rejects group-writable dirs.
- `run_layer_prof.sh "<pytest -k expr>" <label>`: Tracy + device profiler of the port's per-layer perf tests on P300x2 (all 4 chips).
  - Mounts this repo's `models/`, `tests/`, `conftest.py` and `pytest.ini` into the profiling image. The filter goes through `PYTEST_ADDOPTS`, because `python -m tracy` re-splits arguments.
  - Example: `"traced_decode_batched and b1-"` profiles one GDN layer (L0) and one attention layer (L3), traced single-user decode, shipped precision policy.
- `analyze.py <ops_perf csv> [iters]`: per-op table of the signposted decode windows.
  - The first all-gather's "gap" is the idle time between trace replays, not real.
- `segments.py <csv> [iters] [--ops]`: the same windows split into boundary / mixer / MoE (found from the
  collectives, so it survives op-count changes); `--ops` lists the MoE ops of one step.
- `trun.sh <cmd>`: run a command in an image (`IMAGE`, default the published one; `ornith-dev` = + pytest,
  `ornith-prof` = + profiler tools) on all 4 chips with this checkout's `models/` + `tests/` mounted; `TENV` = extra
  docker args. `Dockerfile.dev` builds `ornith-dev` (copy a static `uv` next to it).
- `zones.py <profile_log_device.csv> [chip]`: per-op, per-RISC kernel and `DeviceZoneScopedN` zone durations
  (min / median / max over cores). Use with `TT_METAL_DEVICE_PROFILER=1` in `ornith-prof` and an explicit
  `ttnn.ReadDeviceProfiler(dev)`; `tests/test_moe_decode_ops.py prof` does this for the fused MoE ops.
- `mbench_moe.py`: traced op microbenchmark of MoE variants (one chip); compare variants, not absolute times.
- Box qb2-120-p06t08 (where steps 1-2 ran): the user was added to the `docker` group mid-session; until a fresh
  login put `docker-shim` first on `PATH` as `docker` (it runs through `sg docker`). Weights were downloaded
  anonymously; `refs/main` was written by hand for `HF_HUB_OFFLINE`.

### Hardware notes (QuietBox 2, 4x p300c)

- Chips come in pairs. `TT_VISIBLE_DEVICES=0,1` (p300) works; a single chip fails with a fabric mesh descriptor error.
- p300x2 = all 4 chips.
- One server or profiling run at a time.
- A cold boot of the server takes about 6 min of kernel compiles; a warm kernel cache takes about 40 s.

## Measurements (all on this box)

### Baseline, published image, p300 (2 chips, MTP on): `results/base_p300`

| Users, 128/128 | TPOT ms | tok/s |
|---|---|---|
| 1 | 18.58 | 47.4 |
| 8 | 47.75 | 140.6 |
| 32 | 84.53 | 271.5 |

- Long context at 1 user: 1k/256 23.0 ms, 4k 24.0, 16k 26.6, 32k 22.1, 64k 19.8.
- Evals: GPQA-Diamond 0.775 (40 questions), IFEval prompt-strict 0.76 (100), LiveCodeBench v6 0.36 (50).

### Fused GDN decode POC, p300: `results/poc_p300`

| Users | TPOT change |
|---|---|
| 1 | 19.55 ms (+5%, worse: the fp32 state on the MTP path) |
| 8 | 43.80 ms (−8.3%) |
| 32 | 78.48 ms (−7.2%) |

- Greedy outputs differ slightly but stay coherent. Evals were not run.
- Conclusion: fusing one sub-block buys single digits. The gains come from fusing the whole layer.

### Baseline, published image, p300x2 (4 chips, MTP off, 1 user): `results/base4_mtp0`

19.15 ms per token, 50 tok/s, at both 128/128 and 1k/256.

### Step 0: per-layer device profile (p300x2, single-user traced decode, shipped policy)

`results/b1_layers_*`. Wall time per layer: GDN 0.490 ms, attention 0.437 ms. Device breakdown per layer, scaled to 40 layers (30 GDN + 10 attention):

| Group | Ops/layer | µs/layer | Per token | Share |
|---|---|---|---|---|
| MoE (router, top-k, routed + shared experts) | 28 | 176 | 7.0 ms | 37% |
| GDN mixer | 46 | 206 | 6.2 ms | 32% |
| Boundary (2× all-gather + RMSNorm, 2× reduce-scatter, residual adds) | 11 | 103 | 4.1 ms | 22% |
| Attention mixer | 41 | 156 | 1.6 ms | 8% |
| Outside the layers (embedding, LM head, sampling) | | | <1 ms | |

The sum matches the measured 19.15 ms step to within about 1%.

Per-op bandwidth, device 0. All weights are bfp8 in the shipped 4-chip policy `ornith_routed_bfp8_mixers_bfp8`, including the routed experts.

| Op | Time | Bandwidth | Note |
|---|---|---|---|
| Routed gate/up SparseMatmul (8 experts × 2048×256) | 41.4 µs | 108 GB/s, 21% | On only 8 cores |
| Routed down SparseMatmul (8 × 128×2048) | 12.3 µs | 181 GB/s, 35% | |
| Router matmul (2048×256 bf16, 1 MB) | 24.6 µs | 43 GB/s | |
| Top-k chain (TopkRoutePrep, TopkLargeIndices, TopkRouteFinish, 2× Slice, Softmax, Untilize) | ~25 µs | | 7 ops |
| Shared-expert gate/up (2048×288) | 11.3 µs | | On 9 cores |
| GDN in-projection (2048×3136) | 21 µs | 63% | |
| GDN out-projection | 8 µs | | |
| Attention QKV (2048×2560) | 17.5 µs | | |
| SdpaDecode | 13.6 µs | | |
| All-gather / reduce-scatter | about 11 µs each | | 4 per layer |

The dense mixer matmuls are already efficient. The waste is the ~40 small ops around them, the collectives, and the MoE.

### Steps 1-2 (QB2 box qb2-120-p06t08; per-layer = device 0, b1 traced decode, shipped policy)

Baseline re-measured here: layers 0.487 / 0.442 ms (GDN / attention), MoE 177.4 us per layer, served TPOT 19.20 ms
(4 chips, MTP off, 128/128, 1 user).

| Change | MoE us/layer (ops) | Layer GDN / attn us | Served TPOT |
|---|---|---|---|
| Baseline | 177.4 (29) | 488 / 434 | 19.20 ms |
| Step 1a: fused router top-k + softmax (`RouterTopK`, one generic_op) | 158.1 (23) | 468 / 413 | |
| Step 1b: decode router 1D on 8 cores, interleaved weight (24.7 us + 2 reshards -> 7.5 us) | 136.8 (21) | 446 / 392 | 17.90 ms |
| Step 2 v1: fused MoE decode, 3 ops (`QWEN35_MOE_FUSED_DECODE=1`) | 38.4 (3) | 346 / 293 | 13.97 ms (14.00 rerun) |

Step 2 v1 per-op (layer profile): router matmul 8.2-9.5 us, `ExpertGateUp` 19.8 us, `ExpertDown` 8.5 us, 1.1 us of
gaps. Served: 19.20 -> 13.97 ms is -27% (71.6 tok/s single user, 4 chips, MTP off). Greedy (step2v1 vs base4_here):
all 16 outputs differ and read coherent. Scorecard evals at concurrency 1 (so they exercise the single-user fused
path; the scorecard runs 8 in flight, which takes the batched path): `runs/step2v1_c1/evals`, see below.

- Step 1 greedy: outputs differ from the baseline (the change runs) and stay coherent; early divergences are
  near-tie rewordings (the baseline's own sequential and concurrent outputs differ the same way).
- Step 1 knobs that did not help: gate|up sparse matmul at 16 cores with 16-wide tiles, in0_block_w 16/32/64
  (42.9-53 us vs 41.6 us). The sparse matmul is not core-bound; step 2 replaces it.
- Step 2 v1 design (models/demos/blackhole/qwen36/tt/moe_decode, shared by the Qwen3.5-MoE ports):
  - Router: ttnn 1D matmul (HiFi4, BF16 weight) over [H, E + 32]; column E = the shared-expert gate.
  - `ExpertGateUp` (16 cores = 4 per column pair): top-k + softmax on every core (NCRISC and BRISC each take half
    the logits, then merge; register-resident selection), the shared expert streams during it, then the routed
    columns stream straight from the interleaved ttnn expert tensor (page p in bank p % 8, so an expert's column is
    contiguous in one bank: in-place mode (b), no duplicate weights), custom_mm, SwiGLU x routing weight.
  - `ExpertDown` (16 cores = 2 per bank): every slot's down columns stacked along K, one custom_mm, written into
    row 0 of the partial-sum tiles; the layer's reduce-scatter reads them directly.
  - Numerics: custom_mm is LoFi only; measured operand widths: weights (srcA) 5 significant bits, activations (srcB)
    7. bfp8 weights as-is lose ~1.8% per matmul (truncation bias; op test: output scale 0.948). The expert weights are
    now stored rounded to 5 significant bits (unbiased; cache files `*.r5`, prefill reads the same weights) and the
    gate|up outputs rounded to 7 bits: output scale 0.989, PCC 0.99998 vs torch on the stored weights. The remaining
    ~1% is the normed hidden's last bit (srcB truncation); rounding it in-kernel is the next numerics item if needed.
  - Layer decode PCC vs HF (real weights, positions 37 / 333): unchanged to 6 decimals (0.99998 / 0.99994).
  - Gate|up timing: 15 us median core, 19.8 us max: the column pair on DRAM banks 0 + 4 (both on logical row 9) is
    ~40% slower per byte (moving its cores to rows 8 / 7: no help). The in-place layout fixes which banks a column
    lives in; gpt-oss balanced the same banks with a rotated duplicate copy.
  - Scalar code on these RISCs is slow (~23-45 cycles per logit in the first top-k): keep data in registers (the
    kernel stack is in L1), avoid 64-bit divisions, split across RISCs. Top-k went 9.3 -> ~4 us (NCRISC tail).
  - Tried, no gain: 3 / 6 / 8 cores per pair (vs 4), 1 / 4 down cores per bank (vs 2); pair-0 cores on rows 8 / 7
    (20.8 / 24.9 us vs 19.8), around bank 4's reader (28.4), its readers on NOC1 (20.2); `ExpertMoE` (gate|up and
    down in ONE program on disjoint cores, down weights prefetched during gate|up, semaphore handoff; correct, kept in
    experts.py): 30 us vs 20 + 8.5 + gap, because the down reads of banks 0 / 4 slow the pair-0 cores to 28 us.
  - Why pair 0 is slow: its cores read BOTH bank 0 and bank 4 across logical row 9. The down op reads the same banks
    at ~60 GB/s per bank when each core reads only its own bank. Fix (next): split-pair gate|up, each core streams only
    its bank's column (up cores near banks 0-3, gate cores near banks 4-7), the gate cores write silu(g) * w and the up
    cores u into two activation buffers, and the down cores form act = u * silu(g) w on the SFPU while their weights
    stream. Then the one-program merge (`ExpertMoE`) should pay off, and the router could move in as well
    (router weights split into two 5-significant-bit halves so LoFi custom_mm reads them exactly).
  - 2-chip (p300) and the 122B: the kernels are written for any `2 I / 32` and `H / 32` that are multiples of 8
    (the 122B's packed up|gate has 2 columns per bank and expert, read with the strided path; not yet run).

### Where the 14 ms goes now (per layer, device 0): GDN 346 us = boundary 102 + mixer 207 + MoE 38; attention 293 us
= boundary 100 + mixer 156 + MoE 38. Re-plan from this: the boundary (step 3, ~4 ms per token) and the GDN mixer
(step 4, ~6.2 ms) are now the big items; the remaining MoE work (split-pair, merge, router) is worth ~0.4-0.6 ms.

## Principle (from the user, 2026-10-22): scale single-user and concurrent decode together

Every fused op is built for M rows (decode widths 1..32, and the MTP verify's 2 rows), not only for one token. A
single-user-only path is allowed only where the multi-row version is intractable or clearly not worth it, and that
is written down next to it. Measure each step at 1 user and at concurrency: per-layer b1 / b8 / b32
(`traced_decode_batched and (b1- or b8- or b32-)`), served TPOT at 1 / 8 / 32 users.

Status against that: step 1's router matmul config covers every decode width (<= 32 rows); the fused top-k and the
step 2 v1 fused MoE are one-row only (`T == 1`); widths 2-16 take the pairs path, 32 the expert-sorted path (they see
only the rounded weights). Multi-user 4-chip baseline rows (u8 / u32, MTP off) still to be measured.

Revised order (re-planned from the step 0-2 profiles):
- 2b. Multi-row fused MoE: per-row top-k + expert union (UINT16 list) + per-(row, expert) weights; each core streams
  its column of every union expert once and multiplies all rows (custom_mm in0 tiles of 1/2/4/8 rows; >8 rows in
  8-row passes over the same weights in L1); down writes one output row per user. Reference: gpt-oss DFlash union ops
  (`models/demos/gpt_oss/tt/fused_decode/union.py`, `stream_*_union_*.cpp` on main). Do the split-pair gate|up
  (each core streams only its own bank) at the same time. Keep, per width, whichever of fused / pairs / sorted wins.
- 3. Boundary (all-reduce + residual + RMSNorm) for M rows from the start.
- 4-8 as below, each multi-row.

### Step 2b: multi-row fused MoE (decode widths 2..8) + the 2-chip layout

Working setup from here (user's call, 2026-10-22): the published image runs as a p300 reference server on chips 0,1
(port 20000, `DEVS=0,1`), development runs on chips 2,3 (`MESH=P300 DEVS=2,3` for trun / run_layer_prof, a p300 dev
server on `PORT=20001`); milestones are measured on both p300x2 (all 4 chips, the reference numbers) and p300.
Quality reference per numerics change: `tools/longctx_quality.py` (garble gate, prompts frozen in
`results/quality_prompts`) + greedy; the 3-task scorecard only at packaging.

- `moe_decode/union.py`: `RouteRows` (one core per row: top-k + softmax into a route table) -> `ExpertGateUpRows`
  (union of the rows' experts; every union expert's columns read once for all rows on [Mp, 32] tiles, Mp = 2 / 4 / 8;
  per-row weights through an identity matmul, as the gpt-oss DFlash union ops) -> `ExpertDownRows` (union slots
  stacked along K, accumulated in chunks). Op test rows 2 / 3 / 4 / 8, TP4 and TP2 shapes: ids exact, worst-row PCC
  >= 0.99996.
- 2-chip / 122B layout (256 expert columns per device): an expert's up and gate columns share a bank, interleaved per
  K row; now read as one contiguous [K, 2] block in 16 KB packets, one custom_mm with ct = 2 (was one 1 KB request
  per tile: gate|up 225 -> 57 us max at 2 rows, 770 -> 149 at 8).
- 2-chip per-layer (device 0), baseline -> fused, MoE us: b1 155.9 -> 56.2, b2 304.8 -> 93.9, b4 450.9 -> 157.8,
  b8 721.9 -> 309.3; layer step GDN / attention: b1 511 / 419 -> 420 / 329, b2 681 / 594 -> 468 / 378,
  b4 859 / 759 -> 564 / 457, b8 1335 / 1034 -> 773 / 596 us.
- What bounds 8 rows: per union slot ~2.6 us of core time (unpack 1.7 us for the 2 x 64 bfp8 weight tiles, SFPU
  0.7 us for silu / weights / rounding on 8 rows); 4 cores per pair. More cores per pair measured slower (the extra
  cores sit off the banks' rows and their reads collapse). Next levers: a cheaper SFPU tail (fold the per-row weight
  into the down op's input), K-split across a pair's cores with a partial-sum exchange.
- Widths 9..32 still take the pairs (<= 16) / expert-sorted (32) paths.

Milestone (2026-10-22, MTP off, 128/128, `results/qb2_steps12/`): TPOT ms / tok/s/u / total tok/s.

| Mesh | Users | Published | Steps 1 + 2 + 2b (`QWEN35_MOE_FUSED_DECODE=1`) |
|---|---|---|---|
| p300x2 (4 chips) | 1 | 19.19 / 52.1 / 50.0 | 13.99 / 71.5 / 67.4 |
| | 8 | 36.67 / 27.3 / 176.8 | 19.26 / 51.9 / 286.6 (+62%) |
| | 32 | 61.98 / 16.1 / 346.6 | 60.87 / 16.4 / 352.1 |
| p300 (2 chips) | 1 | 21.64 / 46.2 / 44.4 | 16.46 / 60.8 / 57.5 |
| | 8 | 47.45 / 21.1 / 141.3 | 28.89 / 34.6 / 208.2 (+47%) |
| | 32 | 84.86 / 11.8 / 273.8 | 86.91 / 11.5 / 265.7 (-3%, not yet explained) |

Quality: garble gate (lower is better; the flagged "non-words" were all real technical terms, check `examples`, not only
the score): p300x2 published 11.13 vs new 9.45; p300 reference 10.71 vs new 15.56 (one prose sample at 29.97 with
only real words flagged; the reference's own two prose samples score 21.5 / 5.8). Greedy outputs coherent at 1 and 8
users on both meshes. Step 2 v1 evals (p300x2, concurrency 1): GPQA-D 0.80, IFEval prompt-strict 0.72.

Milestone 2 (fused MoE at every decode width, 2..32 rows in groups of 8, `QWEN35_MOE_FUSED_ROWS_MAX` = 32):

| Mesh | Users | Published | New |
|---|---|---|---|
| p300x2 | 1 | 19.19 / 52.1 / 50.0 | 13.95 / 71.7 / 67.2 |
| | 8 | 36.67 / 27.3 / 176.8 | 18.89 / 52.9 / 290.6 (+64%) |
| | 16 | 54.73 / 18.3 / 226.3 | 24.54 / 40.7 / 393.5 (+74%) |
| | 32 | 61.98 / 16.1 / 346.6 | 38.25 / 26.1 / 467.9 (+35%) |
| p300 | 1 | 21.64 / 46.2 / 44.4 | 16.47 / 60.7 / 57.5 |
| | 8 | 47.45 / 21.1 / 141.3 | 28.91 / 34.6 / 210.9 (+49%) |
| | 16 | 75.02 / 13.3 / 174.0 | 40.23 / 24.9 / 278.3 (+60%) |
| | 32 | 84.86 / 11.8 / 273.8 | 59.69 / 16.8 / 344.0 (+26%) |

Garble gate p300 new 8.05 (reference 10.71; flagged words all real); greedy coherent at 8 concurrent. 2-chip per-layer
MoE (GDN layer): b16 1291 -> 541 us (gate|up 376 + down 149), b32 ~1570 -> 948 us (561 + 370); DRAM floors ~390 /
~620 us. At 32 rows the union approach computes all 32 rows for every union expert although ~1.6 rows routed to each:
a per-expert row gather (custom_mm with the routed rows only) is the next MoE lever there. The activation buffer is
allocated per step (a persistent 32-row buffer, 2.7 MB per device on 2 chips, made the model's other programs' circular
buffers clash with L1 buffers at boot).

### Step 3: replicated-residual decode boundary (`QWEN35_DECODE_REPLICATED=1`)

`models/demos/blackhole/qwen36/tt/decode_boundary.py`: on TP meshes the decode residual stays replicated on every device
(16-core width-sharded L1); the GDN / attention mixers and the MoE return per-device partial sums and each boundary is
`residual + all_reduce_async(partial)` (BFP8 in, one persistent scratch + global semaphore per site) followed by the local
sharded RMSNorm. Model entry gathers the embedding rows once, the final norm runs locally. The layer dispatches on the
input width, so the MTP head / verify paths (fractured rows) keep the reduce-scatter contract. The per-layer harness
gathers in / partitions out (`segments.py` reports those ops apart).

- Layer PCC (real weights, 2 chips) unchanged: decode GDN 0.99998 / attention 0.99993, b32 per-user min 0.99997 / 0.99994.
- 2-chip per-layer, device 0, b1 (`segments.py`): boundary 88.8 -> 51.6 us; GDN layer 416.0 -> 373.2, attention
  323.4 -> 279.4 us; b16 GDN 1110 -> 1079 us. Device 1's fused MoE is 60-66 us vs device 0's 48-52 (the all-reduce
  absorbs the skew): a per-device imbalance worth ~12 us per layer.
- Served p300 (chips 2,3, MTP off, 128/128), TPOT ms / tok/s/u / total tok/s:

| Users | Reference (published) | Milestone 2 | Step 3 |
|---|---|---|---|
| 1 | 21.64 / 46.2 / 44.4 | 16.47 / 60.7 / 57.5 | 15.06 / 66.4 / 61.0 |
| 8 | 47.45 / 21.1 / 141.3 | 28.91 / 34.6 / 210.9 | 27.93 / 35.8 / 214.6 |
| 16 | 75.02 / 13.3 / 174.0 | 40.23 / 24.9 / 278.3 | 38.99 / 25.6 / 284.4 |
| 32 | 84.86 / 11.8 / 273.8 | 59.69 / 16.8 / 344.0 | 59.28 / 16.9 / 343.6 |

- Quality: garble gate 7.41 (milestone 2: 8.05, reference 10.71; flagged words all real); greedy coherent.

### TTFT (user requirement: no prefill regression)

1-user TTFT, p300, median ms (chips 2,3 vs the reference on 0,1; the two servers' benches run one after the other: run
side by side they share the host CPU and the reference read TPOT 24.98 instead of 21.64):

| Input | Reference | Fused MoE + step 3 | Fused MoE (no step 3) | Fused MoE off |
|---|---|---|---|---|
| 128 | 132.7 | 137.7 | 137.8 | 132.5 |
| 2048 | 417.0 | 430.6 | 429.9 | 418.8 |
| 8192 | 1602 | 1659 | 1663 | - |

The ~3% comes with `QWEN35_MOE_FUSED_DECODE=1` (step 3 adds nothing), not from the 5-bit weight rounding (136.4 / 428.4
without it), not from decode heating the chips (same gap with 2-token outputs), and not per layer (2048-token GDN layer
10465 vs 10518 us). Reproduced outside vLLM with `tools/prefill_time.py` (full model, traced buckets): 2048 tokens 428.5
-> 443.9 ms, 8192 1611 -> 1652 ms; with `QWEN35_MOE_FUSED_ROWS_MAX=1` (no multi-row op sets) 8192 is back to 1614 ms.

Cause and fix: the fused readers (`gu`, `gur`, `dn`, `dnr`) set a static VC (`(bank + reader) & 3`) and transaction ids
on the read command buffer (`noc_async_read_one_packet_set_state<true>`, `noc_async_read_set_trid`). The firmware
re-runs `noc_init` (static VC 1, packet tag 0) only when a kernel's NoC mode differs from the previous kernel's
(`brisc.cc`), so the state outlived the kernel and every later kernel's reads on those cores ran on that VC. More cores
touched, more slowdown: the 16 / 32-row down op on 32 cores cost ~3 %, a 64-core variant 6.6 %. Bisect: not the weight
rounding, not decode heating, not DRAM placement (3 / 17 MB padding: no change), not the L1 activation buffer (DRAM
activations: same), not building the op sets (built but unused: fine), only running them. Now every reader restores
the command buffer before it exits (`kernels/noc_state.hpp`). Full-model prefill, p300, fused on vs off: 2048 tokens
429.1 vs 429.6 ms, 4096 817.0 vs 817.8, 8192 1609.7 vs 1611 (was 1652). Any kernel that uses the stateful NoC read API
with a custom VC or a transaction id needs the same restore (the gpt-oss fused ops on main, which these readers
follow, are worth checking for it; they are not in this tree).

### MTP (the published bundle's default: k = 3 drafts while one request is live)

Served p300, 128/128 random-token prompts, 1 user: new build MTP on 19.39 ms (MTP off 15.06 ms); the published build
MTP on 26.30 ms in this run (18.58 in the step-0 baseline with N = 4: random-token acceptance varies widely with the
prompt set). The MTP pass / commit / draft still run the unfused GDN prefill-form rows path; only the pass's MoE uses
the fused rows ops (4 rows), and the plain decode step the MTP replaces is now much cheaper, so speculation buys less.
Real-text single-user comparison (`tools/realtext.py`: 12 chat prompts x 384 tokens, temperature 0, p300):

| Build | decode tok/s | ms / token |
|---|---|---|
| published, MTP on (its default) | 51.1 | 19.56 |
| new (fused MoE + step 3), MTP on | 73.4 (+44 %) | 13.62 |
| new, MTP off | 66.2 (+30 %) | 15.10 |

On real text MTP still pays (+11 % over MTP off), so the new bundle keeps the published default (MTP on). Plan step 7
(MTP verify on the fused kernels) is the lever on top: the pass / commit still run the unfused GDN rows path.

### Release: raahemnabeel/ornith-1.5-35b-a3b-blackhole-fast (user's go-ahead, 2026-10-22)

`tt-model-fast.yaml` = the v2 manifest + `QWEN35_MOE_FUSED_DECODE=1`, `QWEN35_DECODE_REPLICATED=1`, the plugin build
the v2 image shipped (`~/vllm-tt-plugin-pub`: upstream cacf1e7 + the v2 box's local platform.py / model_runner.py
changes, recovered from the image), every v2 profile. Image built from `ornith_fusion` 034604dac
(`tt-model/ornith-1.5-35b-a3b-blackhole-fast:1e153d8053d6`, staged in `build/`).

- One die (p150 / p100) keeps the v2 MoE: 4 up|gate columns per bank (and 7 banks on p100) are outside the fused
  kernels; `fused.supported()` now checks it (the p150 profile crashed at the kernels' assertion before). p150 with
  the new code: greedy at 8 concurrent identical to the published build, 32-user TPOT 119.8 vs 119.2 ms.
- Evals, p300, MTP on, thinking off, the step-0 settings (`tools/eval.sh`): GPQA-D (40) 0.85 (published 0.775),
  IFEval prompt-strict (100) 0.77 / loose 0.82 (0.76), LiveCodeBench v6 (50) 0.40 (0.36).
- Card: `tools/card_bench.sh` serves each bundle exactly as shipped (`tt-model serve <bundle> --profile P`) and runs the
  published card's rows + the real-text row; `tools/card_tables.py` renders published -> new tables;
  `card/assemble.py p300x2 p300 p150` writes the yaml `card:` block (validated by tt-model's `load_container_manifest`:
  no publish / performance gaps). Raw runs: `~/ornith-fusion/runs/card_{new,pub}_<profile>`.
- Card results (2026-10-23, served as shipped, MTP on; real text = median per prompt, 12 prompts x 384 tokens):

  | profile | real text tok/s (v2 -> fast) | 128/128 total out tok/s, 2 / 8 / 32 users | 1024/256, 32 users | TTFT |
  | --- | --- | --- | --- | --- |
  | p300x2 | 58.5 -> 84.0 (+44 %) | +61..+83 % at 2-16 users, 32 users 345.8 -> 464.6 (+34 %) | 353.8 -> 481.5 (+36 %) | unchanged |
  | p300 | 52.6 -> 76.6 (+46 %) | 64.4 -> 95.4, 140.0 -> 214.5, 270.8 -> 352.0 | 271.1 -> 347.1 (+28 %) | unchanged |
  | p150 | 42.6 -> 44.1 (same code) | +-0 % | 193.0 -> 194.7 | unchanged |

  p150 single-user random-token rows are not reproducible with adaptive MTP on one die: the same v2 build measured
  15.9 (v2 card), 52.4 (milestone) and 27.0 ms (card run) TPOT at 128/128, so the card replaces that table with a
  parity note (`SAME_DECODE` in card_tables.py). The p300 / p300x2 single-user rows improve in every row
  (p300x2 +5..+44 %, p300 +25..+121 %), but carry the same MTP noise; trust real text and the MTP-off numbers for single-user claims.
- Package re-built from bf9c8cbd7 (runtime code = 034604dac; only tooling / the card changed), then
  `tt-model push build/ornith-1.5-35b-a3b-blackhole-fast --publish` (2026-10-23: public + catalog,
  https://huggingface.co/raahemnabeel/ornith-1.5-35b-a3b-blackhole-fast). Hub smoke test 2026-10-23: `tt-model serve
  raahemnabeel/ornith-1.5-35b-a3b-blackhole-fast --profile p300x2` pulled image 26053285ad0b, up after 440 s, all 16
  greedy requests answered (thinking on by default: the 256 tokens were reasoning; greedy.py now keeps that text).

### Port: Qwen3-Coder-Next (branch coder_next_fusion, worktree ~/tt-metal-codernext)

Its `moe.py` was identical to Ornith's published one; the four Ornith diffs (moe / model / model_config /
multichip_decoder) applied cleanly. Layer PCC (real weights, 2 chips) found a shared bug: the multi-row union mask
(`router_topk.hpp route_union`) was 256 bits, so Coder-Next's experts 256..511 fell out of the union (attention b32
per-user min 0.99197 vs 0.99957 unfused). Fixed on coder_next_fusion (512-bit mask): 0.99959; merge into
ornith_fusion after the Ornith release (it cannot change Ornith's 256-expert results). Merged (631b45db5f2).
Package: `qwen_qwen3_coder_next/tt-model-fast.yaml` -> raahemnabeel/qwen3-coder-next-blackhole-fast, built from the
worktree (submodules initialised with `--reference` to the main checkout). Its p300 profile runs bfp4 experts at TP2
(2 columns per bank): supported, but only the bfp8 TP2 shapes were op-tested; check p300 greedy / PCC before the card.

Milestone (2026-10-23, p300x2, published image + the worktree's models/ mounted; `runs/cn_new` vs `runs/cn_pub`, no MTP):

| row | published | fused | |
| --- | --- | --- | --- |
| 128/128, 1 user | 23.36 ms (42.8 tok/s) | 14.30 ms (69.9 tok/s) | -39 % |
| 2048/128, 1 user | 23.44 ms | 14.42 ms (69.3 tok/s) | -38 % |
| 8 users | 47.43 ms (21.1 tok/s/u, 169 agg) | 23.08 ms (43.3 tok/s/u, 347 agg) | -51 % |
| 16 users | 72.05 ms (13.9 tok/s/u, 222 agg) | 30.49 ms (32.8 tok/s/u, 525 agg) | -58 % |
| 32 users | 111.58 ms (9.0 tok/s/u, 287 agg) | 55.09 ms (18.2 tok/s/u, 581 agg) | -51 % |
| 1024/256, 32 users | 128.54 ms | 71.57 ms | -44 % |
| real text, 1 user (12 x 384) | 42.8 tok/s | 69.8 tok/s | +63 % |

(agg = users / TPOT, decode only.) Evals through the chat API, temperature 0, full sets (`tools/eval_coder.sh`):
IFEval prompt-strict 0.782 vs published 0.799 (+-0.018), loose 0.826 vs 0.841; MBPP (rescored, `rescore_mbpp.py`)
0.820 vs 0.826 (500); HumanEval (`humaneval_chat`) published 0.939, fused in the card run. The stock tasks broke:
humaneval_instruct / mbpp_instruct score 0 through the chat API (gen_prefix), gsm8k_cot_llama no longer loads (dataset
id `gsm8k`; `lm_tasks/gsm8k_cot_llama_hf`). Card runs: `runs/cn_card_chain.sh` (p300x2 + p300, both bundles, greedy
compare), then `runs/cn_gsm8k.sh` (GSM8K on both, p300x2). Then: `fusion_card/performance_head.md`, assemble, publish.

Card runs (as shipped, 2026-10-23): p300x2 real text 42.8 -> 69.8 tok/s, 1 user 23.35 -> 14.28 ms at 128 in and
25.98 -> 16.87 ms at 130816 in, 32 users 111.23 -> 54.80 ms (288 -> 584 tok/s total), 1024/256 32 users 127.66 ->
71.20 ms; p300 real text 39.4 -> 58.2, 1 user 25.36 -> 17.16 ms, 32 users 134.43 -> 67.80 ms; TTFT unchanged on both.
HumanEval (humaneval_chat) 0.939 -> 0.945, MBPP 0.826 -> 0.820, GSM8K strict 0.954 -> 0.955, IFEval strict 0.799 -> 0.782.
**Published 2026-10-23**: https://huggingface.co/raahemnabeel/qwen3-coder-next-blackhole-fast (public + catalog),
image tt-model/qwen3-coder-next-blackhole-fast:25e9f20fe4f2 (its models/ tree byte-identical to the benchmarked
349324dd4450). Hub smoke test queued (`runs/cn_hub_smoke.sh`).
Greedy text differs from the published build's (rounded weights), coherent on both meshes (p300 = bfp4 experts).

### Port: Qwen3.5-122B-A10B (branch q122_fusion, worktree ~/tt-metal-q122)

- Latest published = raahemnabeel/qwen3-5-122b-a10b, built from tt-metal 8fec52c4cb (origin/qwen3_5_122b_a10b_mtp_fix;
  the Hub `code/` matches it) with plugin 92403f3 = `~/vllm-tt-plugin-pub` (identical files). This tree's 122B autoport
  was older (no MTP context-end guard / fresh-slot phase entry): q122_fusion takes the autoport from 8fec52c4cb, then the
  Coder-Next port diff (moe / model / model_config / multichip_decoder) applied with one hand-merged hunk.
- Shapes: hidden 3072 (12 tiles per bank, new), 256 experts top-8, moe_intermediate 1024 -> 256 columns per die at TP4
  (2 per bank), routed experts bfp4. Op test mode `q122`; layer PCC via its test_multichip_decoder.py (pinned to 1x4).
- Dev serving: the Ornith fast image (same C++ as q122_fusion) with q122's models/ mounted (`serve.sh` recipe sets
  EXTRA_MODELS_DIR); reference = the published image tt-model/qwen3-5-122b-a10b-blackholex4:cb33207c2577.
- `~/ornith-fusion/runs/q122_chain.sh`: Ornith Hub smoke test, then 122B op test, PCC fused vs unfused, served new vs
  published (greedy, garble, rows, real text). Manifest draft `qwen_qwen3_5_122b_a10b/tt-model-fast.yaml`
  -> raahemnabeel/qwen3-5-122b-a10b-blackhole-fast.
- 2026-10-23: op test `q122` passed (single row out PCC 0.99998, rows 1..32 worst row >= 0.99996, ids exact) after the
  ExpertDown readers fix. Layer PCC vs HF (real weights, 4 dies, shipped policy), fused / unfused: GDN decode 0.99965 /
  0.99967, attention decode 0.99471 / 0.99477, B=32 per-user min GDN 0.99719 / 0.99717, attention 0.99532 / 0.99548;
  prefill identical.
- Served milestone 2026-10-23 (p300x2, Ornith fast image + q122 models/ vs the published image; MTP on at 1 user):

  | row | published | fused |
  | --- | --- | --- |
  | real text, 1 user (median, 12 x 384) | 46.8 tok/s | 64.9 tok/s (+39 %) |
  | 1 user 2048/128 | 29.57 ms | 22.06 ms |
  | 8 users | 63.75 ms (15.7 tok/s/u) | 36.18 ms (27.6 tok/s/u) |
  | 16 users | 96.03 ms (10.4 tok/s/u) | 51.32 ms (19.5 tok/s/u) |
  | 32 users | 114.00 ms (8.8 tok/s/u, 198 total) | 66.51 ms (15.0 tok/s/u, 278 total) |
  | 1024/256, 32 users | 115.09 ms | 67.99 ms |
  | TTFT 128 / 2048 / 32 users | 295 / 687 / 6284 ms | 259 / 640 / 6301 ms |

  1 user 128/128 random tokens: 21.49 -> 21.22 ms (adaptive MTP, noisy). Garble gate: mean 5.86 -> 4.21 per 1k,
  the fused max (14.4) is technical vocabulary the word list lacks (outlier, ablation, rescale). Greedy text differs
  (rounded weights), coherent. Card chain `runs/q122_card_chain.sh`: p300x2 card rows done for both bundles (pub
  realtext median 45.9). Evals new / pub: GPQA-40 0.65 / 0.775, IFEval strict 0.87 / 0.85 (loose 0.90 / 0.87),
  LCB-50 0.54 / 0.54. GPQA: 3 vs 8 discordant (McNemar p 0.23), same truncation count (18/40 at 4096 tokens) on both;
  `runs/q122_gpqa.sh` ran all 198: 0.677 / 0.697 (12 vs 16 discordant, McNemar p 0.57, within one stderr +-0.033).
  longctx rows (p300x2-longctx, MTP off, 1 user): TPOT 31.3 -> 17.6 ms (31.9 -> 56.8 tok/s, +78 %); 16K TTFT 4388 ->
  4408 ms. **Published 2026-10-23: raahemnabeel/qwen3-5-122b-a10b-blackhole-fast** (q122_fusion 883cc57c369, image
  302caba622c3; models/ byte-identical to the benchmarked af3d5f52a5eb, only codegen include order + _ttnn.so differ).
  Hub smoke test queued (`runs/q122_hub_smoke.sh`).
  The shipped `p300x2-longctx` / `p150x4-longctx` profiles do not boot (both bundles): the plugin disables chunked MM
  input and one image item (4096 tokens) exceeds the 2048-token budget. q122_fusion b7e15dd9876 sets the budget to
  4096 (intermediate chunks stay whole 2048-token segments); `runs/q122_longctx.sh` re-packages and reruns the longctx
  rows (published bundle with `SERVE_ARGS="--max-num-batched-tokens 4096"`, card_bench now passes vLLM args through).

### Scorecards (user request 2026-10-23)

One page for every finished model (before / after throughput, what changed, quality):
https://claude.ai/artifact/KX1BqBmjfm5GnYnYonZv63 (private to the user until shared; source
`/tmp/claude-1004/.../scratchpad/scorecards.html` in the session that made it). Update it as each model's card lands
(122B next, then Coder-30B / GLM, then the MTP decisions).

### Shared box (2026-10-23)

Another user (`hous`) runs TT Studio v2.12 on qb2-120-p06t08 (containers `tt_studio_*`, the backend maps
/dev/tenstorrent). The run chains treat only non-`tt_studio_*` containers as "chips busy" (`~/ornith-fusion/bin/
chips_busy`), so a model TT Studio deploys still blocks them; one started mid-run would collide.

### Kernel generalization for the remaining ports (2026-10-23)

- **Any m up|gate columns per bank** (`moe_decode/experts.py gu_layout`, `kernels/gu_layout.hpp`): m even = one
  [kt, m] bank block per slot holding m / 2 complete pairs; m odd = banks q and q + 4 together (the old m = 1 path).
  DST = m (even) / 2 m (odd) tiles (+1 at multi-row widths), full-sync above 4; ring sized by `RING_BYTES`. m = 1 / 2
  keep the same reads, DST tiles and ring sizes. Unblocks one die (Ornith p150, m = 4), Coder-30B (EP, m = 6), GLM TP4
  (m = 3 only with I padded 352 -> 384). DST indices must be compile-time (`static_for`): runtime ones crashed the SFPI
  compiler (internal compiler error in gimple-rvtt-synth, m = 2 rows). Op test `mcols` (2026-10-23, synthetic weights
  rounded as stored, bfp4): m = 3 / 4 / 6 single-row, one-program and 2 / 8 / 32 rows all out PCC >= 0.99997, ids
  exact; the m = 1 / 2 regression checks unchanged (`runs/mcols_op.log`).
- **ExpertDown readers default** = enough for <= 4 output columns per core: the 122B (hidden 3072, 12 columns per bank)
  ran 6 per core in a 4-tile half-sync FP32 DST with the old default of 2 (fixed on q122_fusion too, 1af7c3de943).
- **sigb router** (GLM-4.5-Air: sigmoid + selection bias, normalized; `ROUTER_SIGB` in the routing kernels): FP32
  logits, sigmoid in Q23 (Q24 input, 2^f by Taylor-8 in Q30, Newton reciprocal), bias as Q23 int32 (one DRAM page per
  layer). Host-checked: sigmoid 1.3e-7 abs vs double; top-k equal to the fp64 reference on 2000 random trials outside
  2e-5 near-ties; weights 4.9e-4 (BF16). `RouteScores` writes [M, k] score / id tiles for a port's own expert paths.
- **shared=False** (routed experts only; `NO_SHARED` in the down kernels, slot lists from 1): Coder-30B has no shared
  expert; GLM's is bfp8 while its routed gate|up is bfp4 (one CB format per op).

### Port: GLM-4.5-Air (branch glm_fusion, worktree ~/tt-metal-glm; base = the published commit 13ab966ee923)

- 13ab966 changed shared code (tp_common out_dtype, attention/tp.py) that ornith_fusion does not have, so the branch
  is the published commit + the moe_decode package + the fusion tools, not a branch off ornith_fusion.
- Published card (p300x2): 1 user 20.5 ms (MTP), 32 users 106.8 ms (9.4 tok/s/u, 218 total); "decode is op-overhead
  bound" (31.4 ms token-out for every expert dtype in its sweep).
- Step G1 (implemented, queued `runs/glm_chain.sh`): `QWEN35_ROUTER_FUSED_SIGB=1` replaces the 12-op router chain of
  every decode step (<= 32 rows) with one RouteScores op; experts / shared expert unchanged; no layout or prefill change.
- Step G2 (open): fused experts. Its I = 1408 gives 352 columns per die at TP4 (2.75 per bank): the in-place reads
  need padding to 384 (+9 % expert work in prefill and ~1.5 GiB per die of KV pool: a TTFT regression the user ruled
  out) or a split-K design (each bank's cores reduce partial sums across banks). Shared expert stays a ttnn matmul
  (bfp8 vs bfp4 routed).

### Port: Qwen3-Coder-30B-A3B (published bf7b56f0324f, models/demos/blackhole/qwen3_coder_30b_a3b)

Expert-parallel (32 whole experts per die at TP4, 64 at TP2, bfp4), no shared expert, softmax top-8 of 128; decode
< 4 rows: sparse_matmul over the local slots, >= 4 rows: sorted. Card: 1 user 20.2 ms, 32 users 37.2 ms (585 total):
the multi-user path is already strong, single user is the gap. Its decode MoE per layer: the replicated router chain
(~12 ops, 88.9 us by its own docstring) + two dynamic sparse matmuls over the 32 local slots (~83 us) + SwiGLU /
weighting / sum.
- Implemented 2026-10-23 (branch c30_fusion off ornith_fusion, worktree ~/tt-metal-c30): `EP_EXPERTS` in the moe_decode
  kernels (per-die window of the global top-k; the offset from a per-device [1,1,1,32] UINT32 tensor in bank 0 at the
  same address on every die; live slot count in route[15] / the mailboxes; zero partials when the window is empty),
  `GATE_FIRST` ([gate | up] packing). bfp4 experts need no re-rounding (LoFi reads 4 significant bits exactly).
  `QWEN3_MOE_FUSED_DECODE=1` in tt/multichip_decoder.py (decode widths <= QWEN3_MOE_FUSED_ROWS_MAX, before the sorted
  path); `tt-model-fast.yaml` -> raahemnabeel/qwen3-coder-30b-a3b-blackhole-fast. Op test mode `ep`; `runs/c30_chain.sh`
  (op tests, decode vs HF fused off / on, package, card rows both bundles on p300x2 + evals HumanEval / MBPP / GSM8K /
  IFEval, p300 (the default profile), p150 single-user + greedy (one die: all 128 experts in one window)). Card sources
  in `qwen3_coder_30b_a3b/fusion_card` (c30_fusion 13ff64c527e): fill the eval placeholders, `python assemble.py`.
- Measured 2026-10-24 (both bundles as shipped, `runs/card_c30{new,pub}_*`): decode PCC vs HF 6/6 fused off and on;
  op test ep rows 1..32 PCC >= 0.99996. Single user, 128/128 TPOT published -> fast: p300x2 20.2 -> 10.5 ms (49.5 ->
  95.5 tok/s real text, +93 %), p300 26.3 -> 11.6 ms (38.0 -> 86.3 tok/s, +127 %), p150 36.6 -> 12.9 ms (27.2 -> 76.9
  tok/s, +183 %). 32 users 1024/256 p300x2: 37.8 -> 23.8 ms, 26.5 -> 42.0 tok/s/u, 617 -> 853 total (+38 %); p300 49.8 ->
  32.9 ms, 479 -> 645 total. TTFT flat or lower on every row (p300x2 128k 12.70 -> 12.64 s). Evals p300x2 (full sets,
  published -> fast): HumanEval 0.939 / 0.933, MBPP 0.802 / 0.786, GSM8K strict 0.919 / 0.914, IFEval prompt strict
  0.795 / 0.813; greedy coherent, parts from the published after 17-1074 chars. Card filled (c30_fusion e3baa73624a).
- Published 2026-10-24: raahemnabeel/qwen3-coder-30b-a3b-blackhole-fast (public, catalog), c30_fusion e3baa73624a,
  image 8b91e57afdd3 (models/ + ttnn python identical to the measured 15bb5792c390; only the card changed).

### Step 4 design: GDN decode mixer (op survey 2026-10-24, before the gdn_prof numbers)
Shipped decode mixer (`qwen36/tt/gdn/tp.py:1584-1737` forward_decode, recurrence in
`experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py:397-524`), TP4 b1, device 0
(`runs/prof/fused3`): 46 ops, 161 us kernel + 45 us gaps = 207 us. In-proj 20.4 (qkvzab [2048, 3136] bfp8, 63 % of
DRAM BW), out-proj 7.9 ([1024, 2048] bfp8); everything between is overhead around ~1 us of real state traffic (8 heads x
128 x 128 bf16 read + write): 4 slices 7.8, conv shift copies 8.7, conv mul/mac/silu 25.5 (12 of it gaps), q/k/v slices
12.3, one-hot GQA expand matmuls 23.1, reshapes ~40, gating 9.1, q/k L2 norms on 1 core 10.5, recurrence 9 ops 36.1,
gated norm + silu(z) 11.3. TP2 b1: 272 us; TP2 b16: ~456 us kernel, the recurrence ~208 of it (state moved by 4 ops).

Design (generic_op kernels in models/, like moe_decode, so every tree and image picks them up without a ttnn rebuild):
- **Op A, conv:** split over the qkv width tiles (64 at TP4) x all B rows (conv_states rows of different users share
  tiles, so the split must be by tile column, as the POC's kda_decode_conv does). Reads the in-proj output (L1), the 3
  carried rows, taps; writes silu(conv) for q/k/v and shifts conv_states[1..3] in place (layout unchanged: the MTP
  pass's rebuild_conv_carry / commit_scan / snapshot read it).
- **Op B, head core:** one unit per (user, value head) (B x Nv: 8 at b1 TP4, 256 at b32): GQA by index (k head =
  v head / 2, no expand matmul), q/k L2 norm (fold scale), beta = sigmoid(b), g = -exp(A) softplus(a + dt_bias),
  recurrence on the head's own [128, 128] state tiles (bf16 in DRAM, FP32 in DST; option fp32 state), o, then the
  per-head RMSNorm x silu(z) (the norm is over the 128-wide head, local to the unit), written in the out-proj input
  layout. rec_state layout unchanged ([Bmax, Nv, 128, 128] TILE).
- In-proj and out-proj stay ttnn matmuls (later: tune toward full BW, ~13 + ~5 us; or fold the in-proj into op A's
  reader).
- Projected TP4 b1: 20 + 8 (projections) + A ~4 + B ~8 + 4 gaps ~= 42 us vs 207 -> ~5 ms off the 14 ms token (30 GDN
  layers). Multi-row: B's work scales with the state bytes (b32 TP4: 8 MB read + write, ~40 us), vs the 4-op state
  shuffle today. The same kernels serve the 122B (Nv 16 per die at TP4, hidden 3072) and Qwen3-Coder-Next (also GDN).
- Starting point: the POC kernels (ttnn experimental kda recurrent_decode / kda_decode_conv: fp32-only, DRAM-only,
  gating and norm-gate outside, hi/lo-split fp32 matmuls); the gdn_prof chain (POC on/off, b1/b8) gives their per-op
  times. MTP verify keeps the unfused rows path (step 7) - the layouts above keep it working.

Implemented (2026-10-24): `models/demos/blackhole/qwen36/tt/gdn_decode/` (ops.py: GdnDecode
conv + head programs; kernels conv_* / head_*), wired into `TPGatedDeltaNet.forward_decode` behind
`QWEN35_GDN_DECODE_OPS=1` (gdn/tp.py `_gdn_ops_setup` / `_forward_decode_ops`; bf16 state, needs the tile-padded fused
qkvzab weight and the 1D decode matmuls; per-width output buffers 1..32 built in the constructor). Op test
`tests/test_gdn_decode_ops.py` (tp4 / tp2 / q122 shapes, B 1 / 2 / 8 / 32, vs torch FP32: conv shift exact, users >= B
untouched, conv / y / state PCC). `runs/gdn_op.sh` (queued after the Coder-30B chain): op test -> Ornith GDN layer
tests ops 0 / 1 -> per-layer profile b1 / b8 ops 0 / 1.
Op test on device (2026-10-24, first run, all pass): tp4 / tp2 / q122, B 1 / 2 / 8 / 32, two trials each: conv PCC
1.000000 and shift exact, y PCC >= 0.999998 (max err <= 2.9e-3 of max), state PCC 0.999999 (max err <= 3.1e-3 of max).
Ornith GDN layer tests, 4 dies, shipped policy, ops on (after fix 1f7dd87e04c; the first run's second decode hit a
freed buffer: `tpc.matmul_1d_decode` deallocates its input's L1 "copy", which for the L1-interleaved persistent y IS
y; an exception inside trace capture then hangs the test teardown, so trun timeouts must also `docker rm` the
container): all pass at the ops-off PCC. Traced decode steps 0-3 0.999985 / 0.999988 / 0.999988 / 0.999986 (off
0.999984 / 0.999988 / 0.999988 / 0.999985), B=32 per-user min 0.99997, B=16 0.99998, 128 decode steps 0.99997,
real weights vs HF decode pos 37 / 333 0.99998 / 0.99999, prefill after traced decode, non-aligned continuation,
fallback audit. (test_mc_shipped_policy_pcc fails with ops off and on alike: the HF reference module's 3072 vs 2048
shape mismatch, a test issue.)
Per-layer profile (`runs/prof/gdnops_f{0,1}`, GDN layer, traced batched decode, device 0), ops off -> on: b1 mixer
46 ops 203.8 us -> 4 ops 108.1 us (in-proj 19.3, conv 7.8 on 64 cores, head 68.3 on 8 cores, out-proj 8.0), layer
285.3 -> 189.9 us; b8 mixer 261.0 -> 115.0 us (head 72.0 on 64 cores), layer 460.7 -> 314.3 us. The head kernel is the
gap to the 42 us projection: device zones per unit (math thread, `runs/prof/gdnzones`): gates 15.8, q/k L2 norms 13.3,
gated norm 11.1, decay pass 10.3, delta 6.4, read_state 4.7, rank-1 4.5, q.k + transposes 1.2 us. The matmuls are
cheap; the time is full-tile SFPU / pack work on tiles that carry one row. Head v2 (in progress): decay folded out (k.(dS)
= d (k.S), the bf16 state goes straight into the matmuls with no split, rank-1 = d S + k (x) u in DST), row-vector SFPU
on rows 0-1 only (VectorMode::R, 1 iteration), per-unit scalars read back by value (read_tile_value) instead of
column broadcasts, sdh / sdl / sdx CBs gone (192 KB of L1).
Head v2 done (c125c29a836): per unit 68 -> 32 us (zones: gates 8.6 = mostly the reader's first delivery, q/k norms 6.9,
rank-1 5.2, gated norm 3.4, read_state 3.2, delta 2.7, q.k 1.1, transposes 0.5). Op test 24 / 24 at the v1 accuracy
(y PCC >= 0.999998, state >= 0.999999), deterministic over 3 runs; ops-on layer tests all pass (traced step 0
0.999985, B=32 min 0.99997, 128 steps 0.99997, real weights 0.99998 / 0.99999). Layer profile (`runs/prof/gdnops_v2`):
b1 mixer 108.1 -> 71.9 us (head op 33.2), layer 189.9 -> 154.2 us; b8 mixer 115.0 -> 79.4 us (head 36.3), layer 314.3
-> 278.8 us. Bugs found on the way (worth knowing for any generic_op kernel here):
- a CB push must not wrap: cb_nrm had 2 pages and took a 2-page then a 1-page push per unit, so every other unit's
  2-page push ran past the CB end into the next CB (the LLK assert is compiled out); the symptom was a wrong 3rd / 5th
  unit per core that moved with timing and vanished under DPRINT;
- Blackhole L1 cache: a RISC load of an address the NOC (or the packer) rewrote needs `invalidate_l1_cache()` first;
- VectorMode::R needs two SFPU iterations to cover row 0 (one covers part of it);
- FPU work after SFPU work on the same DST tiles needs `TTI_STALLWAIT(STALL_MATH, WAIT_SFPU)`.
Single-user decode with MTP on runs the verify pass, which keeps the unfused GDN path: the ops help multi-user decode and
the MTP-off profiles (-longctx); a fused multi-token verify is the follow-up if the served numbers justify it.
Served A/B 2026-10-24 (`runs/gdn_ms.sh`: Ornith p300x2, the published fast image + this tree, the fast env, only
QWEN35_GDN_DECODE_OPS differs; `runs/gdnms_*`), ops off -> on:
- MTP off, 1 user 128/128: TPOT 11.84 -> 7.22 ms (84.5 -> 138.5 tok/s, +64 %); 16K input 12.16 -> 7.65 ms; real text
  84.2 -> 149.1 tok/s (median prompt 84.3 -> 138.1; one prompt stopped early with ops on: the translation prompt
  says it once instead of repeating it).
- MTP on (the shipped default; verify pass = unfused GDN): 1 user 17.29 -> 17.35 ms, real text 86.5 -> 86.1 tok/s.
- 8 users 18.31 -> 12.24 ms (54.6 -> 81.7 tok/s/u, 437 -> 654 total, +50 %); 32 users 38.95 -> 28.99 ms (25.7 -> 34.5
  tok/s/u, 822 -> 1104 total, +34 %).
- TTFT unchanged: 128 166 / 168 ms, 2K 391 / 395, 8K 1365 / 1366, 16K 2622 / 2638.
- Garble gate (t0.6, per 1k): MTP off 9.08 / 9.87 (max 14.65 / 23.37), MTP on 10.8 / 13.32 - inside the range of
  earlier milestones; greedy parts early (chars 20-273) with equally good wording, outputs coherent.
Consequence: with the fused mixer, MTP off beats the shipped MTP-on default by +73 % on real text (149 vs 86 tok/s),
and MTP off also frees chunked prefill. The MTP sweeps (`ornith_mtp.sh`, `q122_mtp.sh`) now run on this build
(models/ mounted, ops on), so the k / MTP-off decision is made against it; a fused multi-token verify is only worth it
if the sweep shows MTP gaining on top of 149 tok/s.
Coder-Next (coder_next_fusion: step-4 commits cherry-picked; same GDN shapes, 36 GDN layers, no MTP head), `runs/cn_gdn.sh`
2026-10-24: layer tests 17 / 17 ops off and on (B=32 min 0.99997, B=16 0.99996); GDN layer b1 293.7 -> 162.7 us (mixer
203.9 -> 72.1); served p300x2 (published image + this tree, fused env), ops off -> on: 1 user 14.29 -> 8.82 ms, real text
69.8 -> 113.0 tok/s (+62 %); 2K / 8K input 14.40 / 14.55 -> 8.91 / 9.08 ms; 8 users 22.72 -> 15.76 ms; 32 users 53.95 ->
42.22 ms (128/128), 70.15 -> 59.47 ms (1024/256); TTFT unchanged (182 / 484 / 1850 -> 175 / 485 / 1843 ms).
Reader: gate inputs pushed first (afbafb1224c): head 32.0 -> 30.9 us per unit.
122B (q122_fusion, step-4 commits cherry-picked; Nv 16 per die at TP4), `runs/q122_gdn.sh` 2026-10-24: layer tests
20 / 20 ops off and on (B=32 min 0.99719 / 0.99717); GDN layer b1 360.3 -> 211.3 us (mixer 249.3 -> 96.1); served p300x2
(fast image + this tree), ops off -> on: MTP off 1 user 17.60 -> 11.50 ms (56.8 -> 87.0 tok/s), real text 56.7 -> 86.7
tok/s (+53 %); MTP on (shipped) 1 user 19.53 -> 20.62 ms, real text 58.0 -> 59.5 (unchanged: verify pass unfused);
8 users 35.88 -> 26.81 ms (27.9 -> 37.3 tok/s/u, 223 -> 298 total); 32 users 66.57 -> 48.46 ms (15.0 -> 20.6 tok/s/u,
481 -> 660 total); TTFT unchanged (128: 254 / 257, 2K 640 / 643, 8K 2299 / 2299 ms); garble MTP off 5.92 -> 3.51 (max
11.38 -> 8.81), MTP on 4.81 -> 5.58; greedy coherent. MTP off beats the shipped MTP-on by +46 % on real text.
Reader gin-first cherry-picked to q122_fusion (c03bb78b2a5).

Attention layer next (Ornith, b1, `runs/prof/attn_b1`): layer 237 us = boundary 44 + mixer 156 (41 ops: 124 kernel + 31.5
gaps) + MoE 37. In the mixer only the QKV matmul (16.8), SDPA decode (13.6) and o-proj (8.1) are substantive; ~85 us is
glue (3 slices / copies after the QKV matmul, create-heads, 3 sharded->interleaved, q / k RMSNorm on 1 core each + gate
mul, RoPE twice with transposes / slices / concat, 2 FillPad (4.7 each), 2 paged cache updates, sigmoid gate (unary +
mul), concat heads, reshard). A fused attention-decode op (QKV out -> norm + RoPE + cache write + gate, then SDPA, then
o-proj) targets ~50 us: ~1 ms off Ornith's 7.2 ms MTP-off token (10 attention layers; the 122B has 12, Coder-Next 12).

### Step 5: attention decode ops (`QWEN35_ATTN_DECODE_OPS=1`, models/demos/blackhole/qwen36/tt/attn_decode)

Two generic ops around the existing QKV matmul / paged cache update / paged SDPA decode / o-proj:
- prep: from the fused [q | k | v | gate] projection row of each user, the heads as tile rows (the layout SDPA decode
  and paged_update_cache read): q / k RMSNorm x (1 + w) + rotate-half RoPE on the first 64 dims, v copied, sigmoid gate.
  Work items (user, q / k+v / gate) over 3 B cores. q goes to DRAM (SDPA decode takes an interleaved Q from DRAM only),
  k / v straight into the KV-update height shard (no reshard), gate L1. Outputs allocated per call (see the 122B note).
- post: SDPA output x gate, each head row written into the user's row of the o-proj input (one item per user).
Replaces 41 ops (b1) with 7. Bugs / lessons:
- Blackhole DRAM reads keep the address modulo 64: a 32-byte row read into a destination at another offset modulo 64
  lands wrong (odd heads came back zero). The reader reads the 576-byte 64-aligned span holding a row's two face rows
  into scratch and copies the rows on the RISC (non-volatile unrolled copy: 4.7 -> ~1 us per q item vs volatile).
- QKV output in L1 instead of DRAM: no gain at b1 (7 us reads either way), 2x slower at b8 (L1 bank contention).
- Compute on the FPU (row-broadcast (1 + w), r from a matmul of the squares with an all-ones tile so r fills every
  column, DST-reuse multiply; RoPE products as row broadcasts): 10.2 -> 4.4 us per q item; gate sigmoid on rows 0..15
  only (VectorMode::R): 4.1 -> 2.25 us. Rows other than the heads are never zeroed (never read downstream).
Op test (`tests/test_attn_decode_ops.py`, tp4 / tp2 / q122 shapes, B = 1 / 2 / 8 / 32): q / k PCC 0.999997 (TF32 FPU
operands; max error 5-9e-3 of max, bf16 rounding is 3.9e-3), v exact, gate 0.999996, post 0.999999.
Ornith attention layer tests 17 / 17 (decode vs HF 0.99994 on vs 0.99994 off). Per-layer profile (p300x2, b1 traced,
GDN ops on): attention mixer 155.5 -> 67.1 us (QKV matmul 17.0, prep 10.6, 2 x cache update 3.1, SDPA 12.4, post 4.7,
o-proj 8.1), layer 237 -> 148 us; b8 mixer 196 -> 80 us, layer 387 -> 270 us. MTP verify (rows mode) keeps the op
chain. Cherry-picked to coder_next_fusion and q122_fusion. Served A/B: `runs/attn_ms.sh` (Ornith, MTP off),
`runs/ports_attn.sh` (Coder-Next / 122B layer tests, profiles, served A/B).
Served A/B 2026-10-24 (`runs/attn_ms.sh`: Ornith p300x2, published fast image + gdn_wip models/, fast env + GDN ops,
MTP off; `runs/attnms_f*`), attention ops off -> on: 1 user 7.22 -> 6.48 ms (138.5 -> 154.3 tok/s); real text 149.3 ->
166.5 tok/s (+11.5 %, median prompt 138.3 -> 153.5); 2K / 8K / 16K input 7.29 / 7.40 / 7.58 -> 6.62 / 6.72 / 6.89 ms;
8 users 11.83 -> 10.89 ms (84.5 -> 91.8 tok/s/u, 676 -> 735 total); 32 users 29.03 -> 26.93 ms (34.4 -> 37.1 tok/s/u,
1102 -> 1188 total); TTFT unchanged (128 121 / 119, 2K 338 / 338, 8K 1300 / 1283, 16K 2624 / 2622 ms); garble 9.43 /
8.04 (max 15.3 / 16.2); greedy coherent (parts at chars 22 / 122, equally good wording).
Ornith MTP off vs the published blackhole-fast (MTP on, 86.5 tok/s real text): 166.5 tok/s, 1.92x.
Ports (`runs/ports_attn.sh`, 2026-10-24, p300x2, GDN ops on in both arms):
- Coder-Next: attention layer tests 17 / 17 on and off. Attention layer b1 mixer 155.3 -> 67.2 us, layer 246.0 -> 156.6;
  b8 196.4 -> 80.1 / 444.0 -> 322.1. Served (`cnattn_f*`, MTP n/a): 1 user 8.78 -> 7.88 ms (113.9 -> 126.9 tok/s); real
  text 113.5 -> 126.1 tok/s (+11 %); 2K / 8K input 8.87 / 9.04 -> 8.03 / 8.18 ms; 8 users 15.66 -> 14.22 ms (63.9 ->
  70.3 tok/s/u); 32 users 42.49 -> 42.35 ms (128/128) and 60.27 -> 56.47 ms (1024/256; 16.6 -> 17.7 tok/s/u); TTFT
  unchanged (176 / 173, 480 / 489, 1845 / 1860 ms); greedy 1 / 8 identical, coherent (same answers, reworded).
- 122B: attention layer tests 17 / 17 on and off. Attention layer b1 mixer 180.1 -> 92.3 us, layer 296.6 -> 207.8; b8
  219.4 -> 105.4 / 644.5 -> 515.9. Served ops off (`q122attn_f0`, MTP off): 1 user 11.49 ms (87.0 tok/s), real text
  86.8 tok/s. Ops on: the server died at prefill warmup (statically allocated CBs of a GDN prefill matmul clashed with
  L1 buffers at 1208320 vs CB end 1226240): the persistent per-width outputs pinned L1, mostly the k / v height shards
  (16 KB per core per width). Fix (ac88dc55d3d / 2332a6e2bd3): outputs allocated per call and freed after use; op test
  16 / 16 again. Rerun queued (`runs/q122attn_f1.sh`).

### Step 6: decode add + RMSNorm boundary op (`QWEN35_ADDNORM_DECODE=1`, models/demos/blackhole/qwen36/tt/addnorm)

One generic op on the 16 replicated-residual cores (decode_boundary.py layout: [1, 1, B, H] width-sharded, T = H / 512
tiles per core) replaces LayerNorm (sharded) + sharded->interleaved (+ the residual add): h = residual (+ all-reduce
result), per-core sum of squares as a matmul with an all-ones tile (every column = the row sum), that tile sent to all
16 cores (face 0 only for <= 16 rows) + one semaphore increment each, sum of the 16 tiles, rsqrt, out = h x r (column
broadcast) x gamma (DST reuse) written L1 interleaved for the mixer / MoE matmuls. residual / reduced / h are CBs on
their own shards (`ttnn.cb_descriptor_from_sharded_tensor`; generic_op's cache re-binds the addresses). Layer
(`layer.py _forward_decode_addnorm`): norm(x) -> mixer -> all-reduce (no add) -> (h, norm(h)) -> MoE -> h + all-reduce.
Op test (`tests/test_addnorm_ops.py`, H 2048 / 3072, B 1 / 8 / 32, add / no add): out PCC >= 0.999995, h within one
bf16 ulp. v1 9.4-10 us per call (slower than LayerNorm 6.2 + S2I 1.0): zones showed the exchange 3.4 us (32 x 1 KB
writes + 16 increments) and the local copies; v2 (CBs on the shards, face-0 exchange, 4 output tiles per acquire)
7.3 us (exchange 2.0, add + squares + partial sum 1.5, r 0.7, output 0.5); HiFi2 / LoFi measured no faster and lose
accuracy. Lesson: a runtime-arg layout change must move every index (v2 first hung: the NOC-coordinate loop still
started at the old index, so the semaphore increments went to the wrong cores).
Served hang (fixed, 6a9aa912d91 / 499e9a539e5): the first Ornith server with the op hung on its first real decode
(layer tests, the 2-layer generator tests and the op test all passed). Bisect with `runs/smoke.sh` (boot + one
request, 180 s timeout): addnorm off ok, on hang. Cause: a core can start the next program while other cores still run
the previous one; the writer's partial-sum tile went into the other cores' slot CB as soon as its own compute was done,
i.e. possibly into L1 a slower core's previous program (the all-reduce) was still using. Semaphores are set up before
any core of a program starts, CB memory is not; tt-metal's own sharded LayerNorm has its receivers signal ready before
data lands in their CBs. Fix: each writer increments a ready semaphore on all 16 cores at kernel start and writes its
tile only once its own ready count reaches 16 (the signal overlaps the 1.5 us partial-sum compute). Rule for every
generic op here: never write into another core's CB without a handshake from that core in the same program. (Watcher
can't be used on the full model: the bigger kernels overflow the kernel-config buffer; tt-triage needs ttexalens, not
in the image.) test_generator's long-prompt check fails with addnorm on at T=2381 on a near-tie (prefill top-4 logits
10.562 / 10.5 / 10.438 / 10.438; the decode picks the third, the test allows the top 2); T=4097 (margin 0.44) matches.
GDN head (same step): the compute's unpack thread waited ~5 us for the gated-norm weight (the reader sent it after
zeroing + three params-tile DRAM reads) before the first gates; now the weight is waited for before the first gated norm
only and sent after the first unit's state, k is zeroed before the q / k gather and v after: head b1 31.1 -> 27.8 us, b8
34.3 -> 31.5 us (op test unchanged). The gate SFPU now runs on the row-0 iterations only (no measurable change: the
gates were waiting, not computing; at B=32 a warm unit's gates take 0.9 us).
Ornith layer profile (p300x2, GDN + attention ops on), step 5 -> step 6: GDN layer b1 153.4 -> 145.5 us (mixer 71.1 ->
66.5, boundary + MoE 82.3 -> 79.0), b8 278.5 -> 268.8; attention layer b1 148.1 -> 144.6, b8 269.9 -> 270.7. Layer tests
34 / 34. Applied to ornith_fusion / q122_fusion / coder_next_fusion (fbf834d99f9 / 1d436cf9402 / 1f4c6bf6f73; handshake fix
499e9a539e5 / e4fdfe0c6ab / 20bc5889aa4). Served (`runs/step6_ms.sh`, vs attnms_f1) and ports (`runs/ports_step6.sh`)
running. Tried and dropped: q and k split in one acquire loop (q / k norms 6.88 -> 6.82 us, noise).
Remaining b1 GDN layer (145.5 us): boundary 50 (2 x all-reduce 9-11, 2 x addnorm 7.8, router matmul 8.3 on 9 cores
(1 MB bf16, ~15 GB/s per core), 2 x I2S 1.3, add 1.6), mixer 66.5 (in-proj 19, conv 7.8, head ~28, out-proj 8), MoE 28.
Next candidates: the head's q / k norms (6.9 us of hi / lo splits per unit) and splitting a unit's V columns across
cores at low batch (b1 runs 8 of 110 cores); the router matmul with K split across more cores; cross-layer add + norm
(layer L's h + all-reduce and layer L+1's first norm in one call: -2 us per layer).

**Step 6 served result (2026-10-24, Ornith p300x2, same setup as attnms_f1; MTP off):** addnorm ON (`step6ms`, with
the handshake): 1 user 6.49 ms (vs 6.48), 8 users 10.77 ms, then a hang at 32 users (a second, not yet understood hang:
the 1 / 8-user rows passed). The per-layer profile gain (-4 us / layer, measured before the handshake) did not reach the
served step. **Dropped: addnorm stays in the tree behind its flag, off, not shipped.** The GDN head fix alone
(`step6b`, addnorm off): 1 user 6.48 -> **6.37 ms (154.3 -> 157.0 tok/s)**, real text 166.5 -> **169.0 tok/s**, 2K / 8K /
16K 6.62 / 6.72 / 6.89 -> 6.54 / 6.63 / 6.80 ms, 8 users 10.89 -> 10.73 ms (91.8 -> 93.2 tok/s/u, 735 -> 746 total), 32
users 26.93 -> 27.18 ms (noise level), TTFT unchanged (119 / 121, 338 / 337, 1283 / 1284, 2622 / 2653 ms), greedy 8 / 8
identical to step 5 (the change only reorders reads), garble 6.06 (8.04 at step 5; sampling noise). -0.11 ms matches the
profile (-3 us x 30 GDN layers).

### v2 releases (2026-10-24, user: publish new repos + scorecards when free)

Queue `runs/run_queue2.sh`: `cn_v2_chain` (Coder-Next blackhole-fast-v2: GDN + attention ops; card rows p300x2 / p300 +
HumanEval / MBPP / GSM8K / IFEval vs the release-time blackhole-fast runs card_cnnew_* / cn_new), `ornith_v2_chain`
(tt-model-fast-v2.yaml: GDN + attention ops, MTP off on the multi-die profiles, one die unchanged; card rows vs
card_new_*, GPQA 40 / IFEval 100 / LCB 50 on p300 vs fast_p300), `q122_v2_chain` (same for the 122B, MTP off; vs
card_q122new_* / q122ev_new), then glm_chain, ornith_p150_chain, q122_hub_smoke, the MTP sweeps. New repos:
raahemnabeel/{qwen3-coder-next,ornith-1.5-35b-a3b,qwen3-5-122b-a10b}-blackhole-fast-v2. MTP decision:
off wherever the fused GDN step runs (Ornith real text 169 vs 86 tok/s, 122B 94 vs 60): the verify pass keeps the
unfused path; a fused multi-token verify is the way back to MTP. Kernel work continues in ~/tt-metal-gdnwip only (the
release trees are packaged from ornith_fusion / q122_fusion / coder_next_fusion as they stand).

v2 queue outcome (2026-10-24): Ornith p300x2 card rows done (1 user real text 169.1 tok/s, 32 users 27.4 ms); the p300
(2-chip) servers of Ornith v2 and Coder-Next v2 died at boot: prefill warmup TT_THROW "statically allocated circular
buffers clash with L1 buffers" (the GDN decode ops' persistent conv_out / per-width y L1 buffers lowered every core's
L1 ceiling; the 2-chip in-proj's CBs need almost all of it). Fixed (gdn_decode: both allocated per call; on all four
branches): Ornith v2 p300 boots, 1 user 14.83 -> 8.70 ms (67.4 -> 114.9 tok/s), real text 75.0 -> 124.3 tok/s, 8 users
28.22 -> 17.91 ms, 32 users 58.29 -> 40.34 ms, TTFT 188 -> 137 ms (vs blackhole-fast, MTP on). The 122B MTP sweep was
stopped after a4 (multi-user MTP on the unfused verify loses: Ornith m3 27.5 tok/s vs 169 MTP off). User, 2026-10-25:
no hurry to ship; keep going on the scoped levers until returns are clearly diminishing; releases after.

### Step 7: where the remaining time goes, and the gpt-oss reference (2026-10-25)

gpt-oss-20b on this box type (QuietBox 2, 1x4, TP 4, ~3.6 B active) decodes one user at 2.47 ms / token on main
(#60161, models/demos/gpt_oss/tt/fused_decode): streamed matmuls per DRAM bank, the boundary (fabric all-reduce + add +
RMSNorm) inside the next matmul's program, top-32 sampling in the LM head. Ornith (40 layers) was at 6.37 ms: far from
that, so not at diminishing returns overall; the per-op fusions in the old layout were.
Measurements behind the re-plan (Ornith p300x2):
- Decode matmuls (`tools/mbench_mm.py`, traced, 1 row): ttnn 1D mcast tops out at ~350 GB/s (in-proj 6.8 MB 19.6 us,
  qkv 16.3, out / wo 7.9-8.4, router (bf16 1.2 MB) 9.9); ttnn DRAM-sharded is 1.4-2.4x slower on every shape (1-3
  readers / bank, 16 / 32 activation cores): the DRAM-sharded lever is only reachable as a custom per-bank stream.
- All-reduce (`tools/mbench_ccl.py`): all_reduce_async 9.0 us for 1 row padded to 64 BFP8 tiles; packed to 2 dense BF16
  tiles 4.0 us (8 rows 6.4, 32 rows worse: 15-17 vs 9).
- Token breakdown (2-layer generator test, `test_generator -k greedy_sampler_paths`): sampler trace 0.551 ms per token
  (31 ops; device 0 kernel time 483 us: ttnn.topk over all 32 rows of the [32, 62080] logits 279 us, vocab mask 28,
  pad 26, two all-gathers ~29, ~15 tie-break ops ~3 us each, ttnn.sampling 28); LM head ~0.3 ms (~410 GB/s, near
  bandwidth); readback + host ~0.1 ms.

Fabric boundary (`QWEN35_FABRIC_BOUNDARY=1`, models/demos/blackhole/qwen36/tt/fabric_boundary, gdn_wip 0a945cb0e82):
one-row decode boundaries as ONE op per site after gpt-oss's DecodeBoundary: the 16 to-rows cores pack row 0 of the
producer's L1 BF16 partial (2 x 32 B per tile) into this device's receive slot, the boundary core multicasts it over
the fabric (ring hops; the 4 chips are cabled as a ring, FABRIC_1D included), sums the slots in slot order + residual
+ RMSNorm (gpt-oss compute kernel), the to-rows cores write the normed row as [1, 1, 1, H] in the consumer's layout (L1
interleaved; DRAM before the LM head). The residual stays flat on one core between boundaries; out-projections write
L1 BF16 partials and the fused MoE returns its own buffer in this mode. Op test (`tests/test_fabric_boundary.py`):
residual bit-identical on every device, eager + traced; 7.7 us per all-reduce boundary traced (single-core pack: 10.2)
vs ~18 us for reshard + all_reduce_async + add + norm + the consumers' copies. Generator tests 9 / 10 (the 10th: T=2381,
a 4-way BF16 near-tie, top-4 logits within 0.125; the fabric path picks rank 2, the old path rank 1). Served (`runs/fbms`,
p300x2, MTP off, vs step6b): 1 user 6.37 -> 5.55 ms (157.0 -> 180.2 tok/s, +15 %), real text 169.0 -> 178.3 tok/s,
2K / 8K / 16K 6.54 / 6.63 / 6.80 -> 5.69 / 5.82 / 5.96 ms; 8 / 32 users unchanged (10.67 / 27.13 ms, multi-row path);
TTFT unchanged (121 / 337 / 1284 -> 118 / 338 / 1285 ms); greedy sequential 0 / 8 identical (diverge at chars 40-532,
coherent), concurrent 8 / 8 identical (B > 1 path); garble 8.78 (step6b 6.06, earlier milestones 7.4-10.7).

Sampler rows top-k (`QWEN35_ROWS_TOPK=1`, models/demos/blackhole/qwen36/tt/decode_sampling): the per-device
mask + ttnn.topk replaced by a scan over the live rows only (100 cores keep a sorted top-32 per row as packed u32 keys,
one core heap-merges them) for decode buckets of <= 8 rows; the rest of the sampler unchanged (its forward runs with
ttnn.topk pointed at it). Op test (`tests/test_rows_topk.py`): exact top-32 incl. lower-index tie order, B 1 / 2 / 8;
traced 94 us vs 355 for pad + ttnn.topk. The RISC scan runs ~50 cycles per logit (24 us for 640 logits per core) and
the one-core merge 31 us: both RISCs per core + a merge tree would get it to ~25 us. The logits always carry 32 slot
rows; the live count is the served path's sampler trace bucket (set_trace_bucket per decode width).

Served rows top-k (`runs/fbtk`, vs fbms): 1 user 5.55 -> 5.30 ms (188.7 tok/s), real text 178.3 -> 187.0, 8 users
10.67 -> 10.42 ms, TTFT unchanged; the log shows the 1 / 2 / 4 / 8-row buckets on it; greedy identical (same top-32
set). Then the sampler's tie-break pass (~15 ops, the #33492 ttnn.topk workaround) is skipped on this path: RowsTopK
orders ties by index and ttnn.sampling takes the first position (test_tiebreak_input_adjust documents that): 2-layer
sampler trace 0.549 -> 0.228 ms.

Consumer fusion (`QWEN35_FB_CONSUMER=1`, models/demos/blackhole/qwen36/tt/decode_linear, gdn_wip 7a0b85951da):
- DecodeLinear: decode-only per-bank column-major weight copy (+0.33 GB per device for Ornith's in-proj + QKV; built
  from the loaded weights on the first decode, ~15 s of boot), HiFi2 matmul_tiles, x rows only, output a normal tile
  tensor. Standalone it barely beats ttnn (in-proj 18.1 vs 19.8 us, QKV 14.7 vs 16.7, out-proj worse): ~3.5 us of
  ramp + tail per op caps 6.8 MB near 380 GB/s; the matmul itself is 4.9 us (not compute-bound; LoFi no faster).
- Fused: the layer's MoE->mixer boundary runs the next layer's in-proj / QKV in its program (gpt-oss consumer_parts):
  the weights stream while the boundary computes; the projection cores read the normed row in ONE 4 KB read each once
  the boundary notifies (40 cores x 128 x 32 B reads of the boundary core serialized on its NOC endpoint: +25-40 us),
  then build their tiles locally. Boundary packers moved to rows 2 / 5 and the boundary core to (9, 2), off the
  banks' reader rows (0, 1, 3, 4, 6, 7, 9). Op test (`tests/test_fused_consumer.py`, 4 chips): y PCC 0.99995; traced
  in-proj 28.3 -> 21.2 us, QKV 27.8 -> 22.2 us vs boundary + ttnn (the first trace of a session reads 47 us: warmup).
- Served (`runs/fbcons`, vs fbtk; includes the tie-break skip): 1 user 5.30 -> 4.83 ms (188.7 -> 207.0 tok/s), real
  text 187.0 -> 205.1 tok/s, 2K / 8K / 16K 4.97 / 5.11 / 5.23 ms; 8 users 10.72 (10.42, to re-check), 32 users 26.84;
  TTFT unchanged (119 / 338 / 1285 ms); greedy sequential 0 / 8 identical, coherent; garble 8.99.
Day total (1 user, p300x2, MTP off): 6.37 -> 4.83 ms (157 -> 207 tok/s, +32 %), real text 169 -> 205 tok/s.

Router inside the mixer->MoE boundary (`QWEN35_FB_ROUTER=1`; the boundary also keeps its normed-row tiles for the
gate|up op; DecodeLinear on the replicated BF16 router at HiFi4, bank-padding columns not streamed): device boundary +
router 18.0 -> 12.8 us (GDN layer) / 16.6 -> 14.3 (attention); served (`runs/fbrt`): 1 user 4.83 -> 4.76 ms, real text
205.1 -> 209.3 tok/s; garble 12.11 (re-measured 9.51 in the next run: noise of a few points).

MoE split-pair + off-row readers (`QWEN35_MOE_SPLIT_PAIR=1`, multi-row ops, m = 1): one bank per core (4 per bank), the
gate cores send w * silu(g) to their up partners (credit-controlled ring in the up core's cb_sc), up cores load it via
the identity matmul (down op unchanged). Alone it was slower (8 rows 74.6 -> 91 us at E = 64): the per-core profile
showed the median core at the bank-bandwidth floor either way and the tail on banks 0 / 4, whose optimal readers share
NOC row 9. Fix: the second bank on a shared row gets its readers on the next row inward reading on NOC1 (Y first:
its responses stay off row 9's links) (`experts.place_banks_offrow`). E = 256 op test gate|up rows 1 / 2 / 4 / 8:
18.4 / 31.5 / 56.1 / 94.3 -> 16.9 / 23.4 / 40.1 / 65.9 us. (The op test's default TEST_E = 64 shrinks the 8-row union
to ~42 slots vs ~58 in the model: that was the 74.6 vs 97.5 "gap".) Served (`runs/fbsp`, vs fbrt / the v2 card):
2 / 4 / 8 / 16 / 32 users 7.69 / 7.96 / 9.61 / 13.36 / 24.37 ms (card 8.19 / 8.63 / 10.60 / 14.59 / 27.44; 8 users
104.1 tok/s/u, 833 total; 32 users 41.0 tok/s/u, 1313 total), 1 user 4.72 ms (unchanged path), real text 210.7 tok/s;
greedy concurrent 0 / 8 identical (runs), garble 9.51. The same off-row placement on DecodeLinear: neutral (its cap
is per-op ramp, not row 9).

Fabric boundary for 2..4 rows (`QWEN35_FB_MAX_ROWS`, default 4): row r's flat vector at pages r * tiles; packers and
to-rows cores move every row (to-rows: one read per row of the boundary core's output then local copies: 16 x 64
small reads of one core serialized on its endpoint), the compute sums all slots 4 tiles per acquire and runs the
RMSNorm per row batched (2 rows per acquire, one hw startup per phase); the multi-row MoE writes its partial into a
per-call L1 buffer in this mode. Traced 1 / 2 / 8 rows 7.6 / 9.4 / 28.3 us per boundary (8 rows ~19 us of device
time vs ~20-24 for the replicated path: no gain, hence the limit of 4). Served (`runs/fbmr`, vs fbsp): 2 users 7.69
-> 6.93 ms, 4 users 7.96 -> 7.37 ms; 1 / 8 / 16 / 32 users 4.73 / 9.60 / 13.45 / 23.90 ms (unchanged paths); garble
10.61. Lesson: a model-wide object with persistent L1 buffers must be built at layer construction (before any
warmup): the boundary built at the first decode landed between the prefill warmup and the prefill trace capture,
shifted L1 addresses, and the prefill MoE dispatch missed the program cache in the capture (TT_FATAL "cannot load new
binaries during trace capture", then a hung teardown; found with a script that prints the traceback and os._exit()s).

Day 2026-10-25 (p300x2, MTP off, 128/128 TPOT; vs the v2 release card rows): 1 user 6.37 -> 4.73 ms (157 -> 211
tok/s; real text 169 -> 210); 2 users 8.19 -> 6.93; 4 users 8.63 -> 7.37; 8 users 10.60 -> 9.60 (94 -> 104 tok/s/u);
16 users 14.59 -> 13.45; 32 users 27.44 -> 23.90 (36.4 -> 41.8 tok/s/u, 1339 total).
Dev env for these numbers: QWEN35_FABRIC_BOUNDARY=1 QWEN35_ROWS_TOPK=1 QWEN35_FB_CONSUMER=1 QWEN35_FB_ROUTER=1
QWEN35_MOE_SPLIT_PAIR=1 QWEN35_FB_MAX_ROWS=4 on top of the v2 env (runs/fb_ms.sh EXTRA).

GDN head value split (`QWEN35_GDN_VSPLIT`, default up to 4 parts per head while units x parts fit one item per core:
1 / 2 users on 4 chips, 2 parts at 4 users): each part reads q / k whole and its slices of v / z / state / norm weight,
runs the recurrence on its slice; the gated RMSNorm's sum of o^2 is exchanged between the parts (head_writer.cpp: ready
handshake, slot writes + semaphore, sum in part order: the same total on every part). Head zones at B=1 before: gates
2.8, q/k L2 norms 8.25, read state 3.2, q.k 1.1, delta 2.7, transposes 0.5, rank-1 5.2, gated norm 3.4 us (only the
state-width phases split). Op test tp4 / tp2 B 1 / 2 / 8 / 32 unchanged accuracy; head 28.9 -> 21.6 us (B=1), 29.3 ->
23.9 (B=2). Served (`runs/fbvs`, vs fbmr): 1 user 4.73 -> 4.56 ms (219 tok/s), real text 209.9 -> 218.0 tok/s, 2 users
6.93 -> 6.69 ms, 4 users 7.33; 8 / 32 users 9.51 / 24.30; garble 12.04 (the gate has read 10-12 for the last four
runs vs 6-9 before: a quality checkpoint, the scorecard subsets one request at a time, runs next: `runs/devq_chain.sh`).
Day (1 user): 6.37 -> 4.56 ms (157 -> 219 tok/s, +40 %), real text 169 -> 218 tok/s.
Next small levers: the LM head on DecodeLinear (`QWEN35_LM_STREAM=1`, written, to measure: 261 us at ~274 GB/s
today), the conv folded into the fused in-proj (~0.2 ms), the sampler's all-gathers into the top-k op (~0.1 ms).

Quality checkpoint (`runs/devq_p300x2`, the day's dev config, p300x2, scorecard subsets one request at a time: the
1-row path): GPQA-D (40) 0.75 +-0.07 (blackhole-fast measured the same way 0.85, published 0.775), IFEval prompt
strict / loose (100) 0.77 / 0.86 (0.77 / 0.82), LiveCodeBench v6 (50) 0.46 (0.40); garble 10.79. All within one
stderr: no quality loss shown. The p300 (2-chip) server of the dev config died at boot (`runs/devq_p300`, the log was
cut at 30 lines: tools fixed; root cause next).

### Release: one fast repo per model (user, 2026-10-26)

No -v2 / -v3 repos: each model has its original repo plus ONE fast repo, `raahemnabeel/<model>-blackhole-fast`, and
the new build is pushed into that existing repo as its next revision (user's explicit override of the no-push rule for
the -blackhole-fast repos only; the originals are never touched). Ornith: `tt-model-fast.yaml` now carries steps 1-7
(tt-model-fast-v2 / -v3 manifests removed); the card compares original -> blackhole-fast (`fusion/card/assemble.py`,
NEW=card_fast_<profile> = runs/card_ov3_p300x2 / card_ov3_p300 (the step-7 build served as shipped) and card_new_p150
(one die unchanged since the first fast build), PUB=card_pub_*).
Card rows (vs original): p300x2 1 user 22.4 -> 4.55 ms (44.5 -> 220 tok/s), real text 58.5 -> 217.9 tok/s, 32 users
62.0 -> 24.0 ms (346 -> 591 tok/s total); p300 1 user 23.3 -> 7.9 ms (42.9 -> 126 tok/s), 32 users 84.8 -> 41.5 ms;
TTFT equal or lower on every row (130K: 30.1 -> 28.8 s on p300x2).
Evals (p300, scorecard subsets, `runs/ov3_p300`): IFEval strict / loose 0.82 / 0.87, GPQA-D (40) 0.675 +-0.075 (vs
0.85 for the first fast build, 0.775 original): GPQA trend 0.85 / 0.75 / 0.675 over the last three builds is inside
noise at n = 40 but checked on the full 198 (`runs/ov3_p300_full`) before publishing; LCB pending.
Ports: the 122B and Coder-Next trees take ornith_fusion's `models/demos/blackhole/qwen36` wholesale (they lacked the
GLM-era moe_decode generalization and step 7) plus the model-level hooks (`_install_rows_topk`, `_decode_layers_fabric`,
`fabric_boundary*` args, MoE `router_consumer` / `partial="fabric"`); served p300x2 milestones with the step-7 env
(`runs/ports_step7.sh`, vs card_cnv2_p300x2 / card_q122v2_p300x2: Coder-Next 1 user 7.75 ms, 122B 10.45 ms).
Then their tt-model-fast.yaml the same way, card rows + evals, publish all three.
**Bug found at packaging (2026-10-26): rows top-k on 2 dies.** The v3 p300 garble rows were pure punctuation (token ids
0-19) and scored 0.0 (no words to count); devq_p300c had the same. RowsTopK packs the column into 16 bits (sort key
`0xFFFF - idx`, uint16 output); Ornith / 122B on 2 dies have 124160 columns per die. Op check
(`~/ornith-fusion/diag/rtk_wide.py`, guard lifted): W=62080 exact, W=124160 indices [0, 1, 2, 3] with NaN values.
Greedy runs never use it (force-argmax path), and the 2-die evals sampled clean (host-side sampling for seeded
requests is the likely reason; not confirmed). Fix (9afa272327b): `_install_rows_topk` skips widths > 65536, RowsTopK
asserts it. p300 after the fix (`runs/ofix_p300`, `runs/card_fast2_p300`): garble 8.77 (clean text), 1 user 8.55 ms
(117 tok/s; 7.9 with the broken op), 8 users 18.5 ms (13.0 with it): a 32-bit-index rows top-k (per-core local
index + core offset in the merge, uint32 output) would win ~5 ms at 8 users on 2 dies: not done, for the user to decide.
Coder-Next on 2 dies (75968 columns) is guarded the same way.
Evals final (p300, `runs/ov3_p300`, `runs/ov3_p300_full`): GPQA-D 40 0.675 / 198 0.707 +-0.032 (first 40 of the full
run: 0.75), IFEval 0.82 / 0.87, LCB 0.40.
Coder-Next step 7 (`runs/cns7` vs card_cnv2_p300x2, committed 21c2aa864e2): 1 user 7.75 -> 5.60 ms (129 -> 179 tok/s),
real text 128 -> 178, 2 / 4 / 8 users 10.18 / 11.04 / 14.40 -> 8.38 / 9.13 / 12.73, 32 users 41.73 -> 35.79 ms; TTFT
unchanged; greedy coherent (diverges by synonyms). 122B step 7 (`runs/q122s7`): server died at boot, the kept log was
only the API traceback; rerun with the full log + a no-fabric-boundary fallback (`runs/q122_s7b.sh`).
**Ornith published** (2026-10-26 18:13) into raahemnabeel/ornith-1.5-35b-a3b-blackhole-fast (image
tt-model/ornith-1.5-35b-a3b-blackhole-fast:801c00a07a4c; card original -> fast with the re-measured p300 rows).
122B: the q122s7 boot failure was fb_compute's `4 % tiles == 0` static assert (hidden 3072 = 3 flat tiles; the loops
already take RG = 4 / tiles): relaxed to `tiles <= 4`. Served (`runs/q122s7cfull` vs card_q122v2): 1 user 10.45 -> 7.55
ms (96 -> 132 tok/s), real text 95.3 -> 131.8, 2 / 4 / 8 users 12.53 / 16.44 / 24.82 ms, 32 users 44.78, TTFT unchanged,
garble 7.49 (flagged words all real: hardcoding, piecewise, ...). Without the boundary (`q122s7nofb`): 1 user 9.96 ms.
Image naming (user, 2026-10-26): package `name` = the plain model name (image tt-model/qwen3-coder-next,
tt-model/qwen3-5-122b-a10b; repos keep -blackhole-fast); tt-model always tags the image with the digest prefix (its
`retag_to_digest`), so the `:<hash>` tag stays unless tt-model changes. Release chains: `runs/cn_fast_chain.sh`,
`runs/q122_fast_chain.sh` (systemd units cn-fast, q122-fast).
**Coder-Next published** (2026-10-26 21:05) into raahemnabeel/qwen3-coder-next-blackhole-fast (image
tt-model/qwen3-coder-next:5190768f7a5a). **122B release** (`runs/q122_fast_chain.sh`, image
tt-model/qwen3-5-122b-a10b:629859374c71): p300x2 1 user 30.2 (MTP) -> 7.6 ms, real text 45.9 (MTP) -> 131.8 tok/s,
32 users 113.4 -> 45.1 ms (199 -> 347 tok/s total); p300x2-longctx real text 31.9 -> 131.9; TTFT equal or lower;
evals (p300x2, `runs/q122ev_fast` vs q122ev_pub / q122gp_pub): GPQA-D 198 0.697 = 0.697, IFEval strict 0.85 -> 0.88,
LCB 50 0.54 = 0.54; garble 3.3. Card committed 6a49acb93ca (q122_fusion), pushed by `runs/q122_publish.sh`.
**32-bit rows top-k** (gdn_wip 32687bad143, cherry-picked to ornith_fusion c1f25770879 and coder_next_fusion
bbc206b412e): scan keys carry the core-local column (16 bits suffice inside a core), the merge orders by value then
device column (core first column + local) and writes UINT32 indices above 65536 columns. On device (`runs/topk32.sh`):
op test exact at 2 dies x 124160 and 4 dies x 62080 (cross-core tie included), traced 108 us vs 674 us for pad +
ttnn.topk on 2 dies (98 vs 355 on 4). Ornith p300 served (`runs/tk32_p300` vs ofix_p300, the 16-bit guard): 1 user
8.55 -> 7.92 ms (117 -> 126 tok/s), 8 users 18.5 -> 17.6 ms, real text 116.4 -> 125.6 tok/s, garble 8.77 -> 4.48;
p300x2 1 user 4.57 ms (4.55 before), garble 8.84 (devq 10.79); greedy 8/8 + 8/8 identical on both (greedy never uses
the op). Note: compare greedy only between runs with the same thinking setting (milestone.sh / serve.sh = thinking
off; card_bench.sh = the shipped default, thinking on for Ornith / 122B). Chain `runs/tk32_ports.sh`: Coder-Next 2-die
op test + p300 check, then Ornith (package name now ornith-1.5-35b-a3b) and Coder-Next repackaged, p300 card rows
re-measured + p300x2 spot rows; then cards, push into the fast repos. The 122B has no 2-die profile (unaffected).
Results (2026-10-27): Coder-Next 2-die op test exact (75968 columns, 96 vs 667 us); served p300 (`runs/card_cnfast3_p300`
vs card_cnfast_p300, as shipped) 1 user 9.15 -> 8.51 ms, 8 users 17.97 -> 17.30, real text 109.0 -> 117.1 tok/s, garble
9.61 (`runs/cntk_p300`), greedy 8/8 + 8/8 identical; p300x2 spot rows unchanged (5.60 / 12.66 / 52.7 ms). Ornith p300
(`runs/card_fast3_p300`) 1 user 7.91 ms, 2 / 4 / 8 users 10.39 / 12.62 / 17.35 ms, real text 125.6, 16-32 users and
TTFT unchanged; p300x2 spot 4.57 ms / real text 217.4 / 8 users 9.71 (9.44 before) / 32 users 24.2. Op time by rows on
4 dies (`runs/rtk_time.sh`, 16-bit vs 32-bit kernels): 95 / 100 / 161 / 281 vs 98 / 105 / 170 / 318 us at 1 / 2 / 4 / 8
rows; +38 us at 8 rows (0.4 % of the step) - not worth keeping a 16-bit path. Published 2026-10-27 into the existing
fast repos: Ornith tt-model/ornith-1.5-35b-a3b-blackhole-fast:a7e5b6904100 (name kept: the user wants the plain name
only at a later repackage), Coder-Next tt-model/qwen3-coder-next:e92083dded8f; 122B tt-model/qwen3-5-122b-a10b:4973b43968a8
(2026-10-27 01:05). Cards keep the four-die rows of the previous revision with a note giving this revision's spot rows.
Tool note: bench.sh / greedy.py default MODEL_ID to Ornith: export MODEL_ID for other models (realtext.py now asks the
server).
**MTP pass on the fused build** (`runs/mtpst_k3`: Ornith p300x2, step-7 env + QWEN35_MTP=3, real text 1 user, greedy):
real text 80.8 tok/s (plain fused 217.9); 3.06 tokens per pass (hist a = 0..3: 10 / 20 / 20 / 46 of 96), 32.7 ms per pass
(device steps). Stage medians: verify pass (4 rows, unfused GDN prefill-form rows path) 21.2 ms; commit (re-runs the
GDN rows for a + 1 accepted rows when a < 3, ~52 % of passes) 10.6 ms; draft 0.82 ms x 3; head rows 0.83 ms. A plain
fused 4-user step is 7.2 ms, so the verify is ~3x what fused rows cost, and the commit exists only because the pass
keeps no per-row GDN state. Fused verify target: 4 rows through the fused step (attention: rows as users sharing the
page table at consecutive positions; GDN: one op looping the R tokens over the held state and writing the state after
each row, so commit = pick a snapshot) ~ 7-9 ms + drafts 2.5 + rows 0.8 -> ~11-13 ms per 3.06 tokens = 235-280 tok/s
vs 218 plain (+8..+28 %, greedy acceptance; sampled text accepts less). First check before any kernel: the verify
with the fused GDN op chained once per row (upper bound on the GDN cost).
**Fused MTP verify: measured, not built** (2026-10-27, user go-ahead for the check). One die, Ornith TP4 GDN shapes,
traced replays (`tests/bench_gdn_rows_chain.py`): GDN decode ops over 4 users 40.5 us, 1 user 34.9 us; 4 sequential
rows of one user chained on the B=1 op (row slice, op, state + conv-carry snapshot after rows 0..2, concat) 271.3 us
(+231 us per GDN layer, x 30 = +6.9 ms per pass); commit = restore one snapshot 23.0 us (x 30 = 0.7 ms, ~52 % of
passes). Served p300x2 128/128 (card_fast_p300x2): 1 / 2 / 4 users 4.55 / 6.69 / 7.18 ms. Per-pass budget at the
measured real-text greedy acceptance (3.06 tokens per pass), vs plain fused 217.9 tok/s:
- chained GDN (the upper bound): 7.18 + 6.9 verify + 0.4 commit + 2.46 drafts + 0.83 head rows + 5.2 host = 23.0 ms
  -> 133 tok/s (-39 %); with no host time 17.8 ms -> 172 tok/s (-21 %).
- a multi-token GDN op (rows looped in the kernel on the 32 cores one user gets, ~4 x 25 us per row of compute: about
  +2 ms per pass): 18.2 ms -> 168 tok/s (-23 %); host time cut to 1 ms: 14.0 ms -> 219 tok/s (+0.5 %).
- ceiling (GDN rows at the 4-user cost, no host time): 10.9 ms -> 282 tok/s (+29 %), which needs the multi-token
  GDN op AND drafts + accept on device in one trace (the 5.2 ms is three draft read-backs, the verify read-back and the
  serving loop); sampled requests accept fewer drafts than greedy.
Decision (the >= 15 % bar agreed before the check): MTP stays off in the fast builds; no new GDN op.
Next: hub smoke tests of all fast repos (the last step). Ornith's plain package name at its next repackage.
After the releases: the fused MTP verify (user: consider it then). Left alone on purpose (user, 2026-10-26): the
2-chip fabric boundary, conv folded into the in-proj, sampler all-gathers, 8-row boundary, one-die fused MoE.
Box note: background chains die when the Claude session disconnects (nohup / setsid too): launch them with
`systemd-run --user --unit=<name> ...` (see runs/*.sh).

### MTP parameters (user request 2026-10-23: decide k, which batch sizes run MTP, and the resulting throughput)

Knobs (Ornith / 122B / GLM share tt/mtp_serving.py): `QWEN35_MTP=k` drafts per pass (default 3); `QWEN35_MTP_ADAPTIVE=1`
(default) = MTP only while ONE request is live, plain batched decode otherwise; `=0` = MTP over every live slot
(traced multi-user step, slots x (k + 1) <= 32, so max-num-seqs <= 32 / (k + 1)); measured slower than plain decode at
2-4 users before the fusion, which made multi-row verify passes much cheaper. Sweep (`runs/mtp_sweep.sh`, p300x2,
the shipped fast builds): MTP off; adaptive k = 1..4 (real text 1 user, 128/128, acceptance); MTP-all k = 1 / 2 / 3
(16 / 10 / 8 slots) at 2 / 4 / 8 (/ 16) concurrent real-text streams vs plain fused decode at the same concurrency
(`realtext.py CONC=U`). Queue: `ornith_mtp.sh`, `q122_mtp.sh` (GLM after its build is chosen). Decision rule: per
batch size the setting with the highest aggregate real-text tok/s, provided per-user tok/s does not drop; if MTP-all
wins up to N users, the serving change is an adaptive threshold (MTP while <= N requests live) instead of one.

### Run queue (2026-10-23)

`~/ornith-fusion/runs/run_queue.sh`: after the 122B card chain (package + card rows + scorecard subsets), in order:
q122_longctx, q122_gpqa (done; 122B published), c30_chain (done; Coder-30B published), glm_chain, ornith_p150_chain, cn_hub_smoke,
q122_hub_smoke, then the MTP sweeps; gdn_op (step 4 op test + layer tests + profile) runs right after c30_chain
(ornith_mtp, q122_mtp). User, 2026-10-23: finish the scorecards and the fused-kernel work first, MTP tuning after;
multi-user MTP only if the sweep shows a large gain (MTP and chunked prefill are exclusive in the plugin, so each
profile picks one; that choice stays).

## Plan (in order; each step ends with a measurement)

Re-plan 2026-10-25 (from the step 7 measurements; 1 user 5.55 ms after the fabric boundary):
- a. Sampler: rows top-k + tie-break skip (done, -0.32 ms per step); left: the two all-gathers + draw into the op
  (~0.1 ms) and a faster scan (both RISCs + merge tree, ~0.05 ms).
- b. Consumer fusion (done for the MoE->mixer boundary: -0.3 ms); the mixer->MoE boundary into the router (bf16 1.2 MB:
  less to hide) is next on this line.
- c. Fabric boundary for 2..8 rows (gpt-oss helpers / row-tile layout): ~10 us per layer at 8 users.
- d. MoE gate|up split-pair (8 users: gate|up 97.5 us vs ~62 us floor), GDN head V-split (1 user), router.
- e. MTP verify on the fused path (gpt-oss DFlash: 1.7-2.1x at batch 1) once b is in.
Earlier list (kept for the record):

1. **MoE quick wins.** (DONE: fused top-k + 1D router, MoE 177 -> 137 us, TPOT 19.20 -> 17.90 ms; the gate|up / shared
   core-count knobs did not help, see above.) Settings and program configs only, in `tt/moe.py` / `tt/optimized_decoder.py`:
   - Put the router matmul on a sane config: 1 MB should take 3–5 µs, not 25.
   - Give the gate/up SparseMatmul as many cores as the down projection (or more).
   - Spread the shared expert over more cores.
   - Replace the 7-op top-k chain with one fused router top-k. Check the merged PR #58472, "Fused GPT router on Blackhole P150 and partial batches".
   - Expected: about 3 ms of 19, roughly +15% single-user. It also carries over to every model sharing `Qwen35MoE`.
   - Verify with `run_layer_prof.sh` (MoE segment), then the 4-chip MTP-off bench row, then the greedy check.
2. **Fused MoE decode kernel.** (v1 DONE: 38 us, 3 ops, in-place streaming; TPOT 13.97 ms. Follow-ups above.) Target
   about 30 µs per layer, down from 176.
   - Router with top-k computed inside the matmul writer, then streaming gate/up + SiLU + down over the 8 selected experts plus the shared expert, in one or two ops.
   - Sources: tt-metal PR #60161 (gpt-oss fused decode, merged 2026-10-16) and the `nd.py` ND-sharded experts in the `jashansinghTT/gpt-oss-120b-blackhole-fast` package.
   - Adapt to 256 experts top-8 plus a shared expert.
   - Two weight modes:
     - (a) A duplicate decode-optimal copy. 4 chips only: about +8.5 GiB/chip at bfp8, which shrinks the KV pool from about 15 to about 6 GiB/chip.
     - (b) In-place streaming of the existing layout. Required for 2 chips and for the 122B.
   - Start by measuring what (b) achieves. gpt-oss's 1-chip path reached 81% of bandwidth on its existing layout without duplicating.
3. **Fused boundary op.** All-reduce + residual + RMSNorm in one op, as in #60161. Target about 30 µs per layer, down from 103.
4. **GDN mixer in 2–3 kernels.** In-proj → conv + recurrence (the ops on this branch) → norm·gate → out-proj. Target about 45 µs, down from 206. Add a bf16-state option so MTP doesn't regress.
5. **Attention decode fusion** for the 10 full-attention layers (RoPE, QK norm, KV write, SDPA, gate).
6. **LM head:** streaming over the 248k vocab, with top-k sampling on device.
7. **MTP verify on the fused kernels.** A 2-row variant of the step. Without it, MTP slows a fused model down; gpt-oss measured 0.947× until its check ran on the fused path.
8. **Package as a new tt-model profile/repo** (new name). Run the scorecard evals and compare with the baseline numbers above.

**Projected result:** the sum of the targets is about 4.5–5 ms per token on 4 chips, roughly 200–220 tok/s single user before MTP. Re-profile after each step and re-plan from the numbers, not from this projection.

**Gate after steps 1–3:** single-user TPOT on p300x2 with MTP off should be at least 30% below 19.15 ms. If it isn't, stop and re-profile before going further.
