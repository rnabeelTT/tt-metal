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
  prefill identical. Served runs next (`runs/q122/`, `runs/q122_new`, `runs/q122_pub`).

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
  (op tests, decode vs HF fused off / on, package, card rows both bundles).

### Run queue (2026-10-23)

`~/ornith-fusion/runs/run_queue.sh`: after the 122B chain, in order: q122_card_chain (package + card rows + scorecard
subsets), c30_chain, glm_chain, ornith_p150_chain, cn_hub_smoke, gdn_prof_chain (step 4 numbers).

## Plan (in order; each step ends with a measurement)

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
