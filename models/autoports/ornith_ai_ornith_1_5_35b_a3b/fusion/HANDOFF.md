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
- `segments.py <csv>`: the same windows split into boundary / mixer / MoE.
  - The op index ranges are hard-coded for the current op sequence; re-derive them after any change.

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

## Plan (in order; each step ends with a measurement)

1. **MoE quick wins.** Settings and program configs only, in `tt/moe.py` / `tt/optimized_decoder.py`:
   - Put the router matmul on a sane config: 1 MB should take 3–5 µs, not 25.
   - Give the gate/up SparseMatmul as many cores as the down projection (or more).
   - Spread the shared expert over more cores.
   - Replace the 7-op top-k chain with one fused router top-k. Check the merged PR #58472, "Fused GPT router on Blackhole P150 and partial batches".
   - Expected: about 3 ms of 19, roughly +15% single-user. It also carries over to every model sharing `Qwen35MoE`.
   - Verify with `run_layer_prof.sh` (MoE segment), then the 4-chip MTP-off bench row, then the greedy check.
2. **Fused MoE decode kernel.** Target about 30 µs per layer, down from 176.
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
