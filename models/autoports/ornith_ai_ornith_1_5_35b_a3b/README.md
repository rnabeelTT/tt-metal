# Ornith-1.5-35B-A3B on Tenstorrent Blackhole (QB2, 2× p300c)

Bring-up of `ornith-ai/Ornith-1.5-35B-A3B` — a 256-expert / top-8 MoE with a hybrid
attention stack — onto tt-metal, packaged for `tt-model` v5.1 container serving.

> **OPEN BLOCKER: decode is not reproducible.** At temperature 0 the same prompt
> returns a different completion most runs (8 runs → 6 distinct answers, one
> degenerate). A routed expert bank returns a different result from bitwise-identical
> activations, routing weights and expert indices; the router and everything upstream
> are exact. Not a batching bug — batch 1 is affected, and batch 1 is what ships.
> See [NONDETERMINISM.md](NONDETERMINISM.md) and
> `tools/repro_nondeterminism.py`. **Do not advertise reproducible greedy output
> until this is fixed.**

## Running it

```bash
./run.sh                 # every stage that has not passed, detached in tmux 'ornith'
./run.sh --from 05       # start at a stage
./run.sh --only 06 --force
./status.sh              # dashboard, safe any time
tmux attach -t ornith    # watch; Ctrl-b d to leave it running
```

Stages are `scripts/NN-*.sh`; each writes `status/NN-*` and `logs/NN-*.log`.

## What the model is

| | |
|---|---|
| 40 layers | **30 Gated DeltaNet + 10 gated full attention** (every 4th) |
| MoE | 256 routed experts, **top-8**, `moe_intermediate=512`, + 1 gated shared expert |
| Attention | 16 q / 2 kv heads, `head_dim=256`, partial RoPE 0.25, m-RoPE, output gate |
| Context | 262144 — cheap here: only 10 of 40 layers hold a KV cache, so **20 KB/token** |

Built on `models/demos/blackhole/qwen36` (the Qwen3.5/3.6 hybrid port), which supplies
everything except the MoE. The autoport is thin: a layer subclass that swaps the feed
forward, a model subclass that names it, the MoE itself, and a weight remap.

## Design decisions that are not obvious

**The fused `moe_compute` pipeline is unusable on 4 chips.** It targets Galaxy meshes —
every shipped config is 8×4/16×4/16×8, and its own test skips with *"FABRIC_1D only valid
for (8,1) BH LB mesh"*. Measured here with random weights: ~FLT_MAX garbage on (1,4) and
(4,1), and an unsupported `ND_SHARDED` reduce-scatter on (2,2). So the experts run
unfused. That turned out to *help* precision: the fused path stores weights in a packed
ring layout costing a measured **1.9×** (bfp4 0.53×, bfp8 1.01×, bf16 1.90× of raw bf16
bytes), which is what would have put bf16 out of reach on every board.

**Multi-chip is expert-parallel, not tensor-parallel.** qwen36's TP path assumes the
dense Qwen3.5 weight layout; this checkpoint ships q with its output gate pre-fused as
one `[8192, 2048]` tensor and stores GDN's `qkv` separately from `z`/`a`/`b` where the TP
fusions want a folded `qkvzab`. Attention+GDN are 1.03B of 34.8B parameters, so TP was
saving ~1.5 GB/chip. They are **replicated** instead and expert parallelism — 92.5% of
the weights — carries the memory win alone.

**Experts are batched, not looped.** Decode was dispatch-bound: 3492 ms/token over
~30,700 op dispatches is ~114 µs each, with the devices idle at 59 W. Concatenating
gate/up on the output dim and running down as one batched matmul gives ~12 ops per layer
instead of ~768 → TTFT 3292→543 ms, decode 0.29→3.44 tok/s. `ORNITH_MOE_KERNEL=loop`
selects the per-expert reference the batched path was validated against.

## Measured

**The model serves.** vLLM 0.26.0 + the Tenstorrent plugin, OpenAI-compatible, on one
Blackhole P150 (stage 11 PASS).

| mesh | experts | MoE block | TTFT @128 | prefill | decode |
|---|---|---|---|---|---|
| P150 (1 chip) | bfp4 | 5/5 | 363.7 ms | 352.0 tok/s | 29.48 tok/s |
| P300 (2 chips) | bfp4 | 5/5 | 365.0 ms | 350.7 tok/s | 28.91 tok/s |
| P150x4 (4 chips) | bfp4 | 5/5 | 370.0 ms | 346.0 tok/s | 28.78 tok/s |

Per-stream numbers are identical across meshes ON PURPOSE: the model is replicated, so
each die runs a complete independent copy. Decode is also independent of context length
(24.1 tok/s at ISL 2048 on the pre-optimization build), which is what a 3:1
linear-attention stack should give.

**Accuracy, stated honestly: top-1 agrees on 3/3 prompts, logit PCC does not clear the
gate.** Against the HF reference on P150 at bfp4: PCC 0.968 / 0.942 / 0.934 with 6/8 of
the reference top-8 surviving, against `test_model.py`'s 0.95 bar for this dtype. That
test has never passed at bfp4 (there is no `status/10-full-model`), and the loss is NOT
in the routed experts:

* the MoE block itself measures **0.99976** PCC against a 0.99978 ceiling, and
* raising the down projection to bfp8 (`bfloat4_bfloat8`, 23.9 GB, still one chip)
  moves PCC *within noise* -- 0.935 / 0.963 / 0.936, one prompt better and one worse --
  for 6 GB of DRAM.

So the error accumulates in the non-MoE path (GDN, attention, norms). Every optimization
in this round left the logits BIT-IDENTICAL, so none of it moved accuracy.

**Extra dies buy concurrency, and it scales.** One instance per die -- launch them
with `tools/serve_lanes.sh start 4`. End to end through vLLM, one stream per lane,
256-token completions, measured on this QB2:

| lanes (dies) | 1 | 2 | 4 |
|---|---|---|---|
| aggregate | 26.07 tok/s | 54.47 tok/s | **91.24 tok/s** |
| per stream | 29.05 | 28.40 | 25.89 |
| TTFT (median) | 397 ms | 404 ms | 420 ms |

3.50x at four dies. That is the payoff for choosing data parallelism over expert
parallelism -- and, per the cost model below, the only way extra dies can help this
model at all.

Two caveats worth carrying into any quote of these numbers. Aggregate is total tokens
over wall clock, so it includes TTFT and comes in under per-stream x lanes; the same
four lanes measure 80.63 tok/s at 64-token completions, where a fixed ~400 ms prefill
is amortized over a quarter as many tokens. And per-stream decays 29.05 -> 25.89 as
lanes are added. That is HOST contention between four vLLM processes, not the dies,
which are independent -- and it is proportionally worse than the 25.22 -> 23.71 this
table recorded before this round of optimization, because the device got faster while
the host did not. A host with more cores should recover most of it.

Decode is context-independent, which is what a 3:1 linear-attention stack should give:

| ISL | TTFT | prefill | decode |
|---|---|---|---|
| 128 | 380 ms | 336.8 tok/s | 24.22 tok/s |
| 512 | 1368 ms | 374.2 tok/s | 24.19 tok/s |
| 2048 | 5392 ms | 379.8 tok/s | 24.11 tok/s |

The 3.45 -> 24.2 tok/s decode is `ttnn.sparse_matmul` in INDEXED/GATHER mode; the
543 -> 380 ms TTFT is the prefill mask fix. Both are in `src/tt/moe.py`, whose class
docstring carries the measurements. Prefill throughput is flat in prompt length, so
TTFT is linear with no cliff out to 2048.

Served through vLLM (one lane, `max_num_seqs=1`): 417 ms median TTFT, 23.9 tok/s per
stream. Sending 2 requests to ONE lane still queues them (1909 ms TTFT, flat aggregate)
because `max_num_seqs` is pinned to 1 by the GDN external-state assert -- concurrency
comes from lanes, not from that number, and the lane table above is what scales.

**The tt-model v5.1 package is built and all three profiles answer** (`p150`, `p300`,
`p300x2`), verified by stage 12 against the real container image.

### Running lanes

    source ~/ornith-bringup/env.sh; activate_venv
    export ORNITH_WEIGHTS=<weights>
    tools/serve_lanes.sh start 4      # one replica per die, ports 8100..8103
    tools/serve_lanes.sh wait  4      # blocks until every lane answers /v1/models
    tools/serve_lanes.sh status
    tools/serve_lanes.sh stop

Lane `i` serves on port `8100 + i` (`LANE_BASE_PORT` to move them); logs and pidfiles
go to `/tmp/ornith-lanes` (`LANE_RUN_DIR`). There is deliberately no load balancer in
the script -- put your own in front, or fan out client-side the way the bench does:

    tools/serve_bench.py --mesh p150x4 --concurrency 4 --max-tokens 256 \
      --base $(tools/serve_lanes.sh urls 4) --out lanes.json

Three things the script handles that are easy to miss by hand, each of which is a
hard startup failure rather than a slowdown:

* **The single-die mesh descriptor is mandatory.** One die is a PARTIAL p300 board, so
  `TT_VISIBLE_DEVICES` alone makes tt-metal call the QB2 a CUSTOM cluster and refuse to
  start. `TT_MESH_GRAPH_DESC_PATH` must name `p150_mesh_graph_descriptor.textproto`.
* **`EXTRA_MODELS_DIR` must point at the PARENT of this bundle** (`models/autoports`),
  because the plugin discovers models by listing subdirectories for a
  `vllm_metadata.json`. `tt-model` sets it from the manifest; a bare `vllm serve` does
  not, and fails with "No TT model architecture is registered for model: <path>".
* **Stopping cleanly matters.** A half-closed process leaves `/dev/shm/tt_device_*`
  behind and the next open dies with "Sysmem mapped at unexpected NOC address"; `stop`
  waits for the processes to actually go away.

The equivalent by hand, per die, is:

    TT_VISIBLE_DEVICES=<die> \
    TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
    EXTRA_MODELS_DIR=$TT_METAL_HOME/models/autoports \
    vllm serve <weights> --served-model-name ornith-1.5-35b-a3b \
      --max-num-seqs 1 --block-size 64 --max-model-len 4096 \
      --additional-config '{"tt": {"trace_region_size": 1073741824}}' --port 810<die>

## Packaging (tt-model v5.1)

`tt-model package --container` builds an OCI image (Ubuntu 22.04, Python 3.12, the
tt-metal tree, vLLM 0.26.0 empty-target, the plugin, and this model's code) and
`tt-model serve` runs it. Verified per profile by sending a real completion:

| profile | chips | verdict |
|---|---|---|
| **p150** | 1 | **ANSWERED** -- the package serves |
| p300 | 2 | fails at `fabric_firmware_initializer.cpp:271` |
| p300x2 | 4 | fails in a slice: "Ends 2048 must be <= the shape of the tensor 32" |

p300's failure is the per-process fabric link training documented above: the host tests
absorb it with an in-process warm-up, and the plugin's device-open path does not. That
is plausibly a small fix (warm the full mesh before the plugin opens its own), and is
separate from -- and hit BEFORE -- the replicated-attention dispatch cost that makes
4-chip prefill impractical.

**Profile values must be MEASURED, not derived.** All three container failures before
these were fields declared from arithmetic rather than from a run:

| field | declared | verified | failure |
|---|---|---|---|
| `max_model_len` | 131072 | 4096 | OOM on a 4.3 GB allocation |
| `max_num_seqs` | 32 | 1 | `recurrent_state shape mismatch: [32,...] != [1,...]` |
| `trace_region_size` | default 50 MB | 1 GB | trace buffers do not fit |

Each was a flag typed by hand to make the host server work and never carried into the
manifest, so the package described a configuration nobody had run.

## Why decode costs what it does, and what would actually make it faster

Decode is **dispatch-bound**. One program costs ~8.4 us no matter what it computes, so
decode time is essentially `op_count x 8.4 us` and nothing else. Three independent
measurements say so:

* **A program has a floor, and it is most of the cost.** Traced, a `multiply` costs
  5.45 us at 8 KB, 5.46 at 64 KB, 6.36 at 512 KB -- flat until the tensor gets large.
  (An earlier eager measurement suggested size never matters at all; that was wrong,
  because eager time is dominated by host dispatch, which is size-independent by
  construction. The traced numbers above are the real ones.)
* **Size matters only at the top end.** The 1 MB GDN recurrent state costs 10.35 us
  per elementwise op; a quarter of it, which is what TP=4 would give each die, costs
  5.50 -- back at the floor.
* **Marginal cost is flat.** Building partial models, 12 GDN layers cost 12.456 ms and
  24 cost 23.315 -- 8.41 us per program across the 1291 programs each layer adds.
* **Issue configuration is irrelevant.** One command queue against two: 36.38 vs
  36.39 ms.

Where the time goes, from models built out of layer subsets (`tools/layer_sweep.py`):

| | per layer | x count | share |
|---|---|---|---|
| GDN (linear-attention) layers | 905 us | 30 -> 27.15 ms | 74.6% |
| full-attention layers | 764 us | 10 -> 7.64 ms | 21.0% |
| fixed (lm_head + embedding + final norm) | | 1.59 ms | 4.4% |

That model predicts 36.38 ms against 36.38 measured, and two independent fits agree on
the 1.59 ms intercept.

### What this rules out

**Tensor parallelism buys very little here.** TP divides the DATA per op, not the
NUMBER of ops. Only a handful of ops per layer are big enough for size to matter (the
1 MB state ops, ~6 of 109), so TP would shave those from 10.35 to 5.50 us and leave the
other hundred at the floor -- while adding two all-reduces per layer. The per-op excess
over the floor is mostly op TYPE (matmul, layernorm, tilize) and CORE UTILISATION, and
neither is something another die fixes. This is also why Ornith-1.5 is not comparable to
the dense 30-35B models in the fleet table: those read tens of GB per token and are
bandwidth-bound, so TP scales them nearly linearly (ornith-1.0: 11.6 -> 33.3 -> 43.0
tok/s on 1/2/4 chips). An A3B activates ~3B parameters and runs out of ops, not
bandwidth. This is
why extra dies serve extra USERS instead (see the lane table above). It is also why
Ornith is not comparable to a dense-attention MoE like Qwen3-Coder, whose decode is
made of a few large matmuls per layer rather than ~111 small ops.

Also ruled out, each with a measurement rather than an opinion: raising expert
precision (bfp8 down projection moved PCC within noise), cutting prefill op count
(prefill is compute-bound -- fusing its SwiGLU changed TTFT by less than 1%), and
command-queue tuning.

### What is left

`tools/op_profile.py --sequence` prints the ordered device-op sequence. After the conv
fix a GDN layer is **111 programs**: BinaryNg 24, Slice 17, ReshapeView 16, Unary 11,
Matmul 10.5, Copy 9, LayerNorm 5.5.

The `ReshapeView`s are not free views. TILE layout pads the last two dims, so moving the
head axis from position 2 to position 1 -- `[1,1,32,128]` to `[1,32,1,128]`, which the
delta rule does for q, k and v -- turns 1x4 tiles into 32x4 and is a real copy. They are
structural to expressing a per-head recurrence as batched matmuls.

**The step change would be a fused GDN decode kernel.** tt-metal has fused kernels for
the CHUNKED prefill variant (`gated_delta_attn_seq`, `chunk_gated_delta_rule`) but none
for the T=1 recurrence, so the decode path is ~111 Python-composed ops per layer where a
kernel would be a handful. At 8.4 us per program, collapsing the GDN attention's ~58
programs to ~10 would take decode from 34.3 ms to roughly 22 ms (~45 tok/s) -- and it
would help every mesh equally, since none of this is a parallelism problem.

## Batch > 1 decode: three of four blockers cleared, one left

This is the biggest unused resource on a multi-die board. Per-stream decode is
dispatch-bound and does not improve with more dies, so what extra silicon can buy is
more USERS -- and today that is one sequence per die (`max_num_seqs: 1`), four on a QB2.
Batch 4 on a single die already measured **8.84 tok/s per user eager against 14.47 at
batch 1**, i.e. ~2.4x the aggregate, so the payoff is real.

Cleared:

* **The batch-1 assert in `set_external_state`** hardcoded a leading 1 and rejected a
  perfectly good `[B, ...]` state (`recurrent_state shape mismatch: [4, 32, 128, 128] !=
  [1, 32, 128, 128]`). Everything downstream reads B off the tensor -- the delta rule is
  `B = q.shape[0]` throughout -- so only the assert required batch 1. Trailing dims are
  still checked.
* **The MoE token count** was `x.shape[-2]`, which is T. The decoder hands decode a
  `[B, T, H]` tensor, so at batch 1 that equals the token count by coincidence and at
  batch 4 it is 1 while there are 4 tokens. Now the product of every leading dim.
* **The MoE output shape** was normalised to `[1, T, H]` instead of the caller's
  `[B, 1, H]`, which made the shared-expert add fail two calls later.

Left: a **`[B, T, H]` vs `[1, B, H]` convention collision**. One layer's attention output
comes back transposed against its residual, and the add broadcasts `[4,1,H]` against
`[1,4,H]` into `[4,4,H]`; the next layer's attention then reads `T = x.shape[1] = 4`,
decides it is a prefill, and dies in a concat. The MoE is not the source -- it receives
and returns `[4,4,H]` consistently. Fixing it means settling that convention across the
GDN module, the full-attention module and the layer, which is a shared-code change that
needs per-user numerical validation and vLLM slot management behind it.

## The batch>=4 cross-user failure is not a batching bug

`tools/bench_model.py --batch B` feeds every user the same token from the same zeroed
state and asserts every user's logits are IDENTICAL. It fails from B=4. The failure is
real but the diagnosis "batched decode is wrong" is not: the decode graph is not
bit-reproducible, and the cross-user check is simply the first thing that notices.

Five identical runs on P150, same token, same position, GDN recurrent + conv state and
paged KV freed and reallocated (i.e. zeroed) before each run:

| batch | cross-user max\|logit diff\| per run            | same step re-run |
|-------|-------------------------------------------------|------------------|
| 1     | n/a                                             | **1.0156**       |
| 2     | 0, 0, **1.508**, **0.579**, 0                   | **1.5078**       |
| 4     | **0.598, 1.062, 1.031, 0.940, 0.804**           | **1.3750**       |

The right-hand column is the one that settles it. At **batch 1** -- the configuration
that ships -- running the SAME decode step five times from a freshly zeroed state gives
logits up to 1.0156 apart and flips the argmax (8,8,8,8,25). Nothing is batched there
and nothing is compared across users, so nothing catches it. Batch 2 is not "correct"
either; it is clean in 3 of 5 runs. Batch 4 fails in 5 of 5 because a batch-B step makes
B sequentially different expert calls per layer instead of one.

The same batch-4 configuration is clean on one mesh and dirty on another in the same
sitting -- P300 measured 0.000e+00 and P150x4 measured 7.188e-01 on the runs recorded
here -- which is the same stochastic behaviour, not a per-die difference: the model is
replicated, so each die runs the identical graph with a different allocation history.

### Where it enters

A per-layer probe (`attention_norm`, attention, recurrent state, MoE, residual, per user)
puts the first divergence in the **MoE feed-forward, always**. Up to that layer every
attention output, every GDN recurrent state and every norm output is bit-identical across
users; the MoE output differs by ~1 ULP and the residual stream carries it from there
(layer 27 `ff`=2.4e-4 -> layer 39 out=1.4e-1 -> logits ~1).

Inside `_decode_gather`, at the first diverging call, `x_u` and `ids_u` read back
bit-identical for all four users and the FIRST `ttnn.sparse_matmul` (gate_up) already
differs -- user 0 alone, users 1..B-1 identical to each other. Against an fp32 torch
reference built from the same eight expert weight blocks, both answers are accurate and
about one ULP apart:

    u0:       mean err 0.006729   PCC 0.999825358
    u1,u2,u3: mean err 0.006538   PCC 0.999846578

So this is rounding, not corruption. The `argmax` 25-vs-8 in the harness output is a
bf16 tie: users 1..3 get logit[8] == logit[25] == 11.0 and `argmax` returns the lower id,
user 0 gets 11.5 vs 10.8125. The top-5 ids are the same set for every user.

### What it is not (each ruled out by measurement, not by reading)

* **Not stale GDN state, and not prefill.** It reproduces with NO prefill at all, with
  every layer's recurrent state and fused conv state read back as exactly 0.0 before the
  step. `--batch 4` diverges from a cold, verified-zero state.
* **Not the sdpa_decode Q layout or the page table.** The harness' cross-user check runs
  the EAGER, non-paged decode; the attention layers' outputs are bit-identical across
  users at and before the first diverging layer.
* **Not `ttnn.sparse_matmul`.** A plain `ttnn.matmul` against a fixed persistent weight,
  called at the same point in the same loop, shows the same cross-user spread
  (4.297e-02 where the sparse one showed 4.297e-02).
* **Not the operands.** Rebuilding `x_u` and `ids_u` through the host, explicitly zeroing
  the input's implicit tile padding, and pinning every operand AND the output to the same
  tensor objects for all B iterations each still diverge, bit-for-bit as before.
* **Not dispatch or the program cache.** `ORNITH_CQS=1`, a `synchronize_device` before
  every call, a DRAM instead of L1 output, and `disable_and_clear_program_cache()` all
  leave it in place.

What DOES move it is allocation history. Reversing the per-user loop order, running the
MoE twice per layer, or adding one warm-up batch-1 pass each give exactly 0.0, and the
first diverging layer walks (20 -> 23 -> 27 -> 36) with the op sequence. The model is
deterministic for a FIXED allocation sequence -- two `pytest` processes produce the same
PCC to 16 digits -- and varies when that sequence changes, which is what a free/allocate
between runs and what B sequential expert calls per layer both do.

That places the remaining fault below this model, in how a matmul's result depends on
device state that is not its operands. Do not "fix" it here by reordering the user loop
or inserting warm-up calls: those are the perturbations above, they hide the cross-user
check without making the arithmetic reproducible, and batch 1 stays as unreproducible as
it is today.

### One real bug found on the way, and fixed

`Qwen36GatedDeltaNet.set_external_state` installed fresh recurrent and conv buffers but
left `split_conv_state` -- the per-tap decode conv window DERIVED from the previous
buffers -- in place, and `free_kv_caches` never released it. `gdn/decode.py` rebuilds
that window only "when it is None", so after the documented new-generation reset
(`free_kv_caches()` + `allocate_kv_caches()`) the recurrent and fused conv states read
back as exactly zero while **all 30 GDN layers still held the previous sequence's conv
window** (absmax ~30), and one split conv state per layer leaked per generation. Now
dropped in `drop_derived_conv_state()`, which the three sites that already did this by
hand share. Verified: 30 stale layers before, 0 after. This is NOT the cause of the
cross-user failure -- that reproduces with no prefill and a verified-zero state -- but it
is a real cross-sequence state leak in the serving path.

### Throughput, P150, ISL 128, pure `execute_trace`

| batch | ms/token | steps/s | aggregate tok/s | with host readback |
|-------|----------|---------|-----------------|--------------------|
| 1     | 29.03    | 34.45   | 34.5            | 29.30 tok/s        |
| 2     | 44.07    | 22.69   | 45.4            | 16.29 steps/s      |
| 4     | 73.12    | 13.68   | 54.7            | 9.25 steps/s       |
| 8     | 131.64   | 7.60    | 60.8            | 5.10 steps/s       |

Aggregate rises with batch and per-user rate falls faster than linearly, because the
gather path bills per (user, expert) slot -- each slot is its own M=1 matmul and nothing
amortises across users. Unchanged by the conv-state fix.

## Measured negatives (do not re-try these without new information)

### ttnn.experimental.moe_compute (the fused MoE op) is correct here but ~3x slower

Re-tested after single-device support landed. It WORKS at this model's shape, which is
worth saying because that shape is far outside the one the upstream tests cover -- every
single-card config there uses 16 experts per device, and this model replicates all 256
on one die:

    experts/device 256, hidden 2048, intermediate 512, top-8, tokens 1 and 32
    -> passes, combine PCC ~0.991, on Blackhole
    (tests/ttnn/nightly/.../test_moe_compute_single_card.py, 35/35 also pass as shipped)

But it is not faster for this deployment. Measured at 256 experts/device, 1 token:

    moe_compute (fused, eager+synced)                        257.9 us
    this port's WHOLE MoE block, 35 ops, traced              217.0 us
    the routed-expert part it would replace (2 sparse_matmul) 79.0 us

~3x, and the reason is structural rather than a tuning gap: `moe_compute` is built for
expert-parallel meshes where a die owns ~16 experts and the kernel sweeps them, whereas
`sparse_matmul` in INDEXED/gather mode visits exactly the 8 experts the token selected
no matter how many are resident. A single die holding all 256 is the worst case for the
former and the best case for the latter. It should win back its advantage on a mesh
where experts are actually sharded -- which is not how this model is deployed (see the
serve profiles).

### Quantising further does not pay

The routed experts are already bfp4, which is more aggressive than the Qwen3-Coder
precision policy (bfp8 dense + bfp4 routed). Going further was measured, not assumed:

* **lm_head bfp4**: 28.99 -> 28.56 ms, 1.5%, for PCC 0.968/0.942/0.934 ->
  0.960/0.934/0.927. The head is already at 351 GB/s, so halving the bytes does not
  halve the time, and PCC is the thing this model has least of. Rejected; the knob is
  `ORNITH_LM_HEAD_DTYPE` if the trade is ever worth it.
* **bfloat8_b expert activations** (which Ornith-1.0's policy ships): slightly SLOWER,
  32.52 vs 32.14 ms. Those intermediates are 8-32 KB and traced op cost is flat below
  ~256 KB, so halving them buys nothing and costs a conversion.



- **Tracing does not speed up decode**: 3.40 tok/s traced vs 3.45 eager. Decode is
  sparsity-bound (all 256 local experts read per step, 62 GB/s effective, ~12% of peak),
  not dispatch-bound. Batching the experts had already removed the op-count problem.
- **The GDN decode L1 footprint does not scale with batch**: module-managed state needs
  1,684,608 B of circular buffers against a 1,572,864 B L1, IDENTICALLY at batch 4 and
  batch 8. So batching is blocked by that kernel's footprint, not by the batch size.
  `ORNITH_BATCHED_GDN=1` enables the batched-state path -- it gets as far as GDN decode
  at batch 8, where the stock path cannot even allocate, but does not fit L1.
- **An early "3.44 tok/s decode" figure was a one-token PREFILL.** Without a paged KV
  cache the attention module silently takes its concat-prefill branch for T=1.

The MoE block matches the HF module to **PCC 0.99976**, against a measured bf16 ceiling
of 0.99978 -- the port is essentially exact, and the whole-model figures above are
accumulated bf16/bfp4 error across 40 layers, not a defect.

4-chip prefill is dominated by small attention/RoPE ops dispatching to four devices,
because attention is replicated. Correct, but not practical without tracing or a
different split.

## Changes OUTSIDE this model that ship with the image

Four, all in trees the package already carries (`source.code` covers the tt-metal ones;
`runtime.plugin.path` carries the plugin). Each is a general fix, not an Ornith hack.

**tt-metal**

* `models/demos/blackhole/qwen36/tt/model.py` -- take the parallelism width from
  `args.num_devices` instead of `mesh_device.get_num_devices()`. A REPLICATED model on a
  multi-die mesh has `args.num_devices == 1`, and reading the mesh directly sharded the
  embedding and vocab-split the LM head anyway; the resulting all-gathers then died in
  `control_plane.cpp` ("Trying to get un-initialized fabric context"). Unchanged for a
  tensor-parallel model, where the two agree.
* `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py` --
  `_seq_memory_config` now budgets BYTES rather than sequence length. Calibrated to
  reproduce the old 512-token threshold exactly at a 2048-wide GDN, so nothing narrower
  changes; Ornith is 4096 wide, where the old rule put ISL 512 in L1 and broke prefill.

**vllm-tt-plugin**

* `platform.py` -- the "TT standard DP does not support MoE models yet" guard is now
  opt-in via `model_capabilities['supports_standard_dp']`. The guard protects SHARDED
  MoE models, whose experts no single rank could hold; a replicated one is not that
  case, and standard DP is how it uses extra dies.
* `model_runner.py` -- `_kv_cache_shape` no longer divides `num_kv_heads` by the device
  count for models declaring `model_capabilities['kv_heads_replicated']`. Replicated
  attention puts every KV head on every die, and the mismatch surfaced far away as
  `paged_fill_cache geometry mismatch: cache has kv_heads=1 ... input view is kv_heads=2`.

Both plugin flags are declared in `src/tt/ornith_vllm.py`.

## Hardware and framework gotchas found here

### ttnn.sparse_matmul (the MoE path) -- five ways to lose

1. **A wrong `nnz` is a DEVICE HANG, not an error.** It is a compile-time arg: the
   in0-mcast receivers loop exactly `nnz` times while the sender multicasts once per
   *actual* non-zero sparsity entry, so any mismatch deadlocks in `noc_semaphore_wait`,
   silently, unless the watcher is on (tt-metal #45943, #45052). You cannot compute a
   safe static value for decode: routing weights are a softmax over top-k and the small
   ones FLUSH TO ZERO in bfloat16, so the true count is often < top_k. Pass `nnz=None`
   or use gather, which has no nnz at all.
2. **`in0_block_w=1` costs 6.7x** and is what every upstream caller ships (gemma4,
   gpt_oss, and the Qwen3-Coder reference on this box). Measured at gate_up's decode
   shape: 489.7 us at 1, 257.9 at 2, 104.6 at 8, **73.4 at 64**. Snap it to a divisor
   of Kt -- the kernel asserts `Kt % in0_block_w == 0`.
3. **`num_blocks_total <= num_cores_available` has nothing to do with experts.** It is
   `ceil(Mt/per_core_M) * ceil(Nt/per_core_N)` -- M and N tiles only; the expert axis
   is a batch loop. 256 experts on a 64-core grid is fine, and chunking to "fix" this
   solves a problem that does not exist.
4. **`out_subblock_w` must divide `out_block_w`**, or the kernel bakes
   `in1_num_subblocks == 0`, never pushes cb_out, and the in1 writer parks forever on
   `cb_out.wait_front()`. Another silent hang. `per_core_N` also needs CEILING
   division. Both are documented in `models/demos/gpt_oss/tt/experts/config.py`.
5. **INDEXED/GATHER mode (`indices=`) is 10x the sparsity scan and flat in expert
   count** -- 79 us per expert pair at 64, 128 or 256 experts/die, against 260/439/800
   for the scan. It exists in current tt-metal and not in the reference image, whose
   tt-metal predates it (that image's ABI also differs: its sparsity is `[1,1,batch,E]`
   with `nnz = top_k*batch > E`, which the current contract forbids -- do not port its
   call shapes verbatim). Gather on ONE die beats the scan on FOUR.

### Two things that look like hangs and are not

* **A cold kernel compile.** The first `sparse_matmul` of a given shape on an empty
  `TT_METAL_CACHE` took over 7 minutes at 100% host CPU, and 0.4 s once warm. Budget
  for it in timeouts before reaching for the watcher.
* **`ttnn.experimental.fast_reduce_nc` pads the reduced M axis to a full tile.** It
  returns `[1,1,32,H]` for a one-token step, so the next reshape back to `[1,1,T,H]`
  fails a volume check somewhere unrelated. `ttnn.sum(..., keepdim=True)` gives the
  clean shape. Prefill never notices because its T is already a multiple of 32.

### L1 exhaustion surfaces several ops away from its cause

ttnn defaults to L1, and a big interleaved tensor SUCCEEDS there: `[1, 256, 32, 2048]`
is 32 MB, which is only ~300 KB of each bank's 1.4 MB spread over 110 banks. Nothing
complains. What breaks is the *next* op that needs statically allocated circular
buffers -- for the GDN prefill kernel, ~1.36 MB of them in a 1.5 MB L1 -- which dies
with `statically allocated circular buffers clash with L1 buffers on core range
[0-0 - 3-1]`. Dumping L1 with `ttnn.dump_device_memory_state` between stages shows it
completely free, because the offender is transient inside one forward pass. Place large
intermediates in DRAM explicitly.

Same class of bug upstream: `_seq_memory_config` in
`models/experimental/gated_attention_gated_deltanet` thresholded on SEQUENCE LENGTH
(`<= 512 -> L1`), which mis-sizes for a wide GDN -- Ornith is 32 v-heads x 128 = 4096
wide, so ISL **512 was the worst possible case**, the largest sequence still forced
into L1. Changed to a BYTE budget calibrated to reproduce the old threshold exactly at
2048 width, so narrower models are unaffected. Before: ISL 128 worked, 512 died.


- **A mesh-sharded ttnn tensor keeps its GLOBAL logical shape.** Only *non-sharded* dims
  can be indexed uniformly across devices. Bit twice: chunking the weight packer, and
  indexing experts.
- **The first multi-chip fabric open in each PROCESS fails** ethernet router sync; the
  next succeeds. It is link training, does not survive process exit (so a preflight
  script cannot fix it), and warming the **full** mesh is what trains every link — a
  2-chip warm-up only trains that pair.
- **`ttnn.all_reduce` returns full width; `tt_all_reduce` reduce-scatters** on a (1,N)
  mesh. Same name, different contract.
- **The weight cache ignores the mesh mapper on reload** — change a sharding decision and
  you must delete the cache, or `as_tensor` silently replays the old layout.
- **`ttnn.reshape` returns a view sharing its source buffer**; deallocating it frees the
  source.
- **`ttnn.linear` defaults to L1.** A couple of unfreed L1 tensors per layer starves the
  GDN op, which needs ~1.36 MB of the 1.5 MB L1 for circular buffers — and it fails in
  the *next* layer's attention, naming neither the leak nor the module.
- **`TTMoEGate` preallocates per-core L1 buffers.** One gate per layer is fine at 8
  layers and fatal by 16; only the router weight is per-layer, so one gate is shared.
- **The GDN chunk-prefill kernel needs T ≥ 128, and T a multiple of 32.** Below that the
  single-device path is *silently wrong* while the TP path errors out.
- `l1_small_size`, `num_command_queues: 2` and `trace_region_size` are required device
  params, not tuning. Without a trace region, trace capture fails and traced decode
  silently falls back to eager.
- **A hard-killed process leaves a UMD chip lock held by a dead PID** in
  `/dev/shm/TT_UMD_LOCK.CHIP_IN_USE_*`, and the next run blocks forever with no timeout.
  The preflight clears them when no device job is running.
- **`ttnn.all_reduce` deadlocks** in the full model on a (1,4) mesh (40 collectives per
  forward) while passing an isolated one-collective test. All-gather + local sum is the
  reliable equivalent.
- **Replicating attention across a mesh costs 4x dispatch for identical work.** Fine for
  memory, bad for latency: multi-chip prefill is dominated by small attention/RoPE ops,
  not by the MoE.

## Layout

```
env.sh            paths, HF token from the `hf auth login` cache
run.sh status.sh  pipeline driver and dashboard
src/              authored model code, rsynced into tt-metal by stage 03
  tt/moe.py           the MoE seam (batched + loop kernels)
  tt/model_config.py  per-mesh precision and parallelism policy
  tt/weight_mapping.py  MoE→qwen36 checkpoint remap
  tt/layer.py tt/model.py  6-line subclasses
  tt-model.yaml       v5.1 container package: one package, three serve profiles
  tests/              MoE block, layer, full model
scripts/lib/      reference generation, benchmarks, dtype decision, preflight
references/       cached HF reference (generated once; a 35B reference costs ~70 GB RAM)
artifacts/        per-stage evidence, perf tables
```
