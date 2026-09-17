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

## What is verified to serve CORRECT output, per hardware type

Re-verified 2026-09-16 by reading the completion TEXT on every configuration, not by
checking that a request returned something. All three target hardware types serve, and
the unit is ONE DIE -- bigger boards scale by running more replicas, not by opening a
wider mesh.

| hardware | dies | how | verified |
|---|---|---|---|
| **p150** | 1 | the tt-model container package, profile `p150` | **correct text** |
| **p150x2** (= p300) | 2 | `tools/serve_lanes.sh start 2` | **correct text on BOTH dies** |
| **p150x4** (= p300x2) | 4 | `tools/serve_lanes.sh start 4` | **correct text on ALL FOUR dies** |

Sample, four lanes, same prompt "The capital of France is", temperature 0:

    lane 0  " Paris. Paris is the largest city in France, and one of the largest in
              Europe. It is located on the River Seine, in the region of Ile-de"
    lane 1  " Paris, a city renowned for its iconic landmarks such as the Eiffel Tower,
              the Louvre Museum, and the Notre-Dame Cathedral. Paris is also a"
    lane 2  (identical to lane 1)
    lane 3  " Paris. Paris is a beautiful city located on the Seine River. It is home to
              famous landmarks like the Eiffel Tower, the Louvre Museum, and Notre"

All four are correct; the differences in phrasing are the open non-determinism
(NONDETERMINISM.md), since each lane is an independent process. Two lanes on one board
returned BYTE-IDENTICAL 32-token completions, which is consistent with the low per-step
flip rate measured there -- divergence needs enough steps to catch.

**Why lanes rather than one wide mesh.** A single instance that opens the whole board is
one replica and gains no throughput; per-die replicas measure 26 -> 54 -> 91 tok/s
aggregate on 1/2/4 dies. And the one-big-mesh `p300` / `p300x2` serve profiles are
currently blocked anyway: the fabric fix carries them through engine init and prefill,
then multi-die DECODE hangs. So lanes are both the faster and the working multi-die
deployment.

**What is NOT yet true, so nobody has to discover it the hard way:** greedy output is not
reproducible run to run (see NONDETERMINISM.md -- open, characterised, not fixed);
`max_num_seqs` is 1, so concurrency comes from lanes rather than batching; and on-device
sampling is unavailable at this vocabulary on a single die, which is what blocks the TTI
release stage.

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

### Final numbers, per hardware type

The number to quote depends on what you are selling. Per-stream latency is flat across
all three because the model is replicated; throughput is what scales with dies.

| hardware | dies | best deployment | aggregate | per stream | TTFT | batch |
|---|---|---|---|---|---|---|
| **p150** | 1 | 1 instance | **26.07 tok/s** | 29.05 tok/s | 397 ms | 1 |
| **p300** | 2 | `serve_lanes.sh start 2` | **54.47 tok/s** | 28.40 tok/s | 404 ms | 1 |
| **p300x2** | 4 | `serve_lanes.sh start 4` | **91.24 tok/s** | 25.89 tok/s | 420 ms | 1 |

Served end to end through vLLM, one stream per lane, 256-token completions; 3.50x at
four dies. Aggregate carries TTFT, so it sits under per-stream x lanes -- the same four
lanes measure 80.63 at 64-token completions, where a fixed ~400 ms prefill is amortised
over a quarter as many tokens. Quote the completion length with the number.

Running ONE instance that opens the whole board instead of one per die gives a single
replica and no throughput gain -- that is what the `p300`/`p300x2` tt-model profiles do,
and it is why `tools/serve_lanes.sh` exists.

### Tensor + expert parallelism: 3.1x prefill (opt-in, measured 2026-09-17)

Prefill was the weak metric (see the fleet comparison below). It is now fixable, and the
reason it was thought impossible was wrong on every count -- see the corrected TP section
further down and commit `d4d43bcd44b`.

    ORNITH_TP=1                  tensor-parallel attention + GDN
    ORNITH_TP=1 ORNITH_MOE_EP=1  and expert-parallel experts   <- the configuration that pays

| ISL | single chip | `ORNITH_TP=1` | **TP + EP (4 dies)** |
|---:|---|---|---|
| 128 | 351 tok/s, 364 ms | 367 (1.04x) | **684 tok/s, 187 ms (1.95x)** |
| 4096 | 396 tok/s, 10.34 s | 478 (1.21x) | **1230 tok/s, 3.33 s (3.10x)** |
| 16384 | 381 tok/s, 43.0 s | 468 (1.23x) | **1198 tok/s (3.15x)** |

**TP ALONE IS WORTH ONLY ~1.2x. You need EP with it.** TP parallelises the non-expert
1.03B -- 3% of the model -- while the experts are 92.5% and stay replicated unless
`ORNITH_MOE_EP=1`. This also reframes ornith-1.0's 3.29-3.51x prefill figure: that is a
TP=4 **and** EP=4 number, so 3.10x is the like-for-like comparison, not 1.21x.

At 1230 tok/s pf@4K this port passes qwen3-coder-30b-a3b's 1,016. TTFT at 4K drops from
10.3 s to 3.3 s.

`FABRIC_1D` (line) is sufficient for everything -- per-layer all-gather, the GDN and
attention reduce-scatters, fused `matmul_reduce_scatter`, and the EP all-reduce. RING was
never needed here.

**Why it is still OPT-IN, and what to settle before promoting it.** Two things:

* **TP vs single-chip logits measure PCC 0.991686** (TP+EP 0.991554), top-1 matching.
  That is ~100x more error than ornith-1.0 reports against its own single-chip baseline
  (0.999892) and is UNEXPLAINED. Dtypes and the gather arrangement are ruled out by
  inspection, leaving row-parallel K-split reduction order and matmul blocking
  accumulating over 40 layers. Note the accuracy gate against the HF reference does NOT
  get worse -- TP scores 0.98410 on prompt 0 against single-chip's 0.97174 -- so
  single-chip is not automatically the more correct of the two, and part of that 0.9917
  may be single-chip error rather than TP error.
* **EP has had far less exercise in this port than replication**, and the 3.1x depends on
  it.

Decode is deliberately NOT the reason to use TP: it is dispatch-bound here, and
ornith-1.0 measures only 1.65-1.69x on decode from the same TP=4/EP=4.

Also open: `prefill_tp` / `_prefill_paged_tp` are single-shot and exceed L1 above ~8K
(5.23 MB of circular buffers in a 1.5 MB L1 at 16384). The 16384 row above goes through
the existing chunked TP path, which works; only that demo entry point lacks chunking.

### Perf across context length, all three meshes (measured 2026-09-16)

Possible for the first time: the old `max_model_len: 4096` made every column past 4K
unmeasurable. `tools/bench_model.py --isl 128 4096 16384 65536 --decode-steps 24`.

decode is tok/s, "traced" includes the host readback and is the comparable figure;
"pure" is device-only `execute_trace`.

| ISL | | P150 | P300 | P150x4 |
|---|---|---|---|---|
| **128** | TTFT / prefill | 363 ms / 353 tok/s | 364 ms / 352 | 366 ms / 349 |
| | decode traced / pure | **33.6 / 40.5** | 32.2 / 39.2 | 29.6 / 39.2 |
| **4096** | TTFT / prefill | 10.4 s / 394 tok/s | 10.4 s / 394 | 10.4 s / 394 |
| | decode traced / pure | **33.0 / 39.9** | 32.3 / 38.7 | 32.2 / 38.7 |
| **16384** | TTFT / prefill | 43.0 s / 381 tok/s | 43.1 s / 381 | 43.0 s / 381 |
| | decode traced / pure | **32.1 / 38.7** | 31.1 / 38.0 | 30.9 / 38.0 |
| **65536** | TTFT / prefill | 185 s / 354 tok/s | 185 s / 355 | 187 s / 351 |
| | decode traced / pure | **29.3 / 34.7** | 27.9 / 35.1 | 28.0 / 35.1 |

**Decode barely degrades with context, and that is the architecture paying off.** P150
pure-trace goes 40.5 -> 34.7 across a 512x context increase, i.e. **-14% from 128 to 64K**.
Only 10 of 40 layers hold a KV cache and the 30 GDN layers' recurrent state is
context-independent, so there is very little for long context to slow down. Against the
fleet at 64K: ornith-1.0 -10%, ours -13%, qwen3-coder-30b-a3b -21%, muse-glimmer -27%.

**PREFILL IS THE WEAK METRIC, and it is the one to fix next.** Prefill throughput is flat
at ~350-394 tok/s at every length, which is 2.6x to 22x slower than the fleet:

| model | pf@4K tok/s |
|---|---|
| muse-glimmer-30b | 8,666 |
| qwen3.8-27b | 5,937 |
| ornith-1.0-35b | 2,342 |
| qwen3-coder-30b-a3b (closest arch peer) | 1,016 |
| **Ornith-1.5-35B-A3B (this port)** | **394** |

The consequence is TTFT, which is essentially all prefill: **0.36 s at 128 tokens, 10.4 s
at 4K, 43 s at 16K, 185 s at 64K.** Three minutes to first token at 64K context is a
usability problem, and it is the single largest gap to the fleet -- larger, proportionally,
than the decode gap that earlier optimisation rounds targeted. Prefill is compute-bound
here (the README's own earlier note), so unlike decode it is NOT a dispatch-count problem
and the fused-GDN-decode work would not touch it.

Extra dies do not help any of this: the model is replicated, so each die runs a complete
independent copy and per-stream numbers are near-identical across meshes. Extra dies buy
REPLICAS (see the lanes table), not speed.

### Device-level numbers (near-identical across meshes)

| mesh | experts | TTFT @128 | prefill | decode (traced) | decode (pure trace) |
|---|---|---|---|---|---|
| P150 (1 chip) | bfp4 | 359.5 ms | 356.0 tok/s | 33.75 tok/s | **40.50 tok/s (24.69 ms)** |
| P300 (2 chips) | bfp4 | 365.6 ms | 350.2 tok/s | 32.69 tok/s | 39.18 tok/s (25.52 ms) |
| P150x4 (4 chips) | bfp4 | 373.5 ms | 342.7 tok/s | 31.92 tok/s | 39.18 tok/s (25.52 ms) |

Re-measured on 2026-09-16 with `tools/bench_model.py --isl 128 --decode-steps 32` on all
three meshes. **The numbers this table used to carry (29.03 ms / 34.45 tok/s) were stale
by 16%.** Eight hand-fusion commits on this branch walked pure-trace decode 28.97 ->
24.69 ms/token and none of them updated this file; each commit message records its own
step. P150 is now faster than the other two meshes because two of those fusions -- the
fused RoPE op (`ea58fce14f4`) and the L2-rescale-into-gamma fold (`d69c283c82d`) -- are
gated to a single-device mesh, since with the fused RoPE in the traced decode both P300
and P150x4 hang in `read_decode_output` inside `prime_decode_trace`. That gate is
deliberate and those meshes keep the manual sequence bit-for-bit.

Three numbers, three measurement points: `pure execute_trace` is device only, `traced`
adds the host readback, and the per-stream figure in the table above is end to end
through vLLM. The served one is the honest one to advertise.

Per-stream numbers are near-identical across meshes ON PURPOSE: the model is replicated,
so each die runs a complete independent copy, and extra dies add replicas rather than
speed. The one real difference is the 24.69 vs 25.52 ms above, which is the two
single-device-gated fusions, not a parallelism effect. Decode is also independent of
context length (24.1 tok/s at ISL 2048 on the pre-optimization build), which is what a
3:1 linear-attention stack should give.

**Accuracy, stated honestly: top-1 agrees on 3/3 prompts, and logit PCC clears the gate
on 2 of 3.** Against the HF reference on P150 at bfp4, with the precision policy now
shipped (non-MoE path hifi2, shared expert lofi): PCC **0.9717 / 0.9713 / 0.9433**,
top-8 overlap 6/8 8/8 6/8, against `test_model.py`'s 0.95 bar for this dtype. Prompt 2
still misses. The previous policy (non-MoE lofi, shared expert hifi4) gave
0.9681 / 0.9421 / 0.9344 and cleared the bar on none of the three; see the precision
table in `tt/model_config.py` for the five configurations measured and the one real
trade-off among them (lofi/lofi has better top-8 agreement, 8/7/8, and worse PCC).

**More precision will not close prompt 2, and the reason is the router.** The binding
constraint is discrete expert-selection agreement with HF, not arithmetic. Hooking the
router and recording selected indices shows that changing ANY fidelity setting flips
11-16% of all expert slots and alters the expert set for 59-69% of token-decisions,
concentrated in layers 23-32 -- the same band as the largest per-layer PCC moves.
Top-8-of-256 is a discontinuous selection, so bf16 noise flips near-ties and the model
draws a DIFFERENT expert, which is a bounded-but-large output change that no matmul
precision removes. That single mechanism accounts for the fidelity non-monotonicity, the
per-prompt inconsistency, hifi4 buying nothing over hifi2, and the fact that everything
is nonetheless bitwise reproducible for a fixed configuration. Worth measuring next: the
HF rank-8 vs rank-9 score-gap distribution.

The loss is NOT in the routed experts:

* the MoE block itself measures **0.99976** PCC against a 0.99978 ceiling, and
* raising the down projection to bfp8 (`bfloat4_bfloat8`, 23.9 GB, still one chip)
  moves PCC *within noise* -- 0.935 / 0.963 / 0.936, one prompt better and one worse --
  for 6 GB of DRAM.

So the error accumulates in the non-MoE path (GDN, attention, norms). Every optimization
in this round left the logits BIT-IDENTICAL, so none of it moved accuracy.

### Where the error actually comes from, measured per layer

That "accumulates in the non-MoE path" was an inference by elimination. It is now
measured, and it holds -- with a specific culprit and a specific non-culprit.

Feeding every decoder layer the HF reference's EXACT residual stream (teacher forcing, so
each layer's error is its own with nothing inherited) gives end-to-end logit PCC
**0.99934 / 0.99898 / 0.99891**, against 0.968 / 0.942 / 0.934 free-running. All 40
layers land between 0.9935 and 0.9995. **No layer is broken and there is no step at any
layer or layer type: the free-running gap is pure accumulation over 40 layers.**

**GDN is the lever.** Per layer, a Gated DeltaNet layer contributes ~2.5x the error of a
full-attention one -- mean contribution PCC 0.9841-0.9852 (n=30) against 0.9937-0.9944
(n=10), relative error 0.164 vs 0.113 -- and with three times the count it carries
roughly **80% of the total injected error**. The only outlier is layer 4 (GDN) at ~2x the
median GDN contribution on all three prompts; a factor, not a step.

Beware the free-running view here, which says the opposite: it shows ~+0.008 PCC lost per
FULL layer and ~0 per GDN layer. That is confounded -- a full-attention layer follows
three GDN layers and inherits their error. Trust the teacher-forced numbers.

**Prompt 2 is an ill-conditioned POSITION, not a defect.** It is the prompt that misses
the gate, and it diverges from layer 0 rather than from somewhere mid-stack (last-position
relative error 0.122 after layer 0, against 0.039 and 0.059). Yet teacher-forced, prompt
2's layers are the MOST accurate of the three. The mechanism is signal strength: its
last-position residual norm is the smallest through the first third of the stack (0.327 at
the embedding and 1.414 at L10, against 0.392/2.034 and 0.493/1.994), so the same absolute
layer noise is ~30% more RELATIVE error. Its final token is `:` -- a low-information
position where the reference itself is near-tied over whitespace. Note this is a
conditioning effect in the MODEL, not a defect in the metric: adding bf16-scale noise to
the reference logits leaves PCC at 1.0000 on all three prompts over 5 seeds, so the 0.93
is a real deviation.

**The LM head and final norm are not worth touching.** Logit PCC comes out 0.012-0.019
HIGHER than the PCC of the hidden state that produced it, on every prompt at both
fidelities (e.g. prompt 1 at lofi: final-norm 0.92487 -> logits 0.94212). The
248320-wide projection averages the 2048-dim error down. Improving them cannot help.

**One thing still unexplained.** 16% relative error on a single layer's own contribution
is far more than bf16 arithmetic accounts for (~0.4%) and more than the MoE's measured
share (~2% equivalent). So a specific low-precision step inside the GDN path is likely
and has not been found. Two alignment traps for whoever looks, both of which produced
spurious results first time round: `output_hidden_states` returns 41 tensors whose LAST
entry is post-final-RMSNorm rather than the layer-39 output, and this checkpoint's
RMSNorm is `x/rms(x) * (1 + gamma)`, not `* gamma`.

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
| **p150** | 1 | **SERVES, AND THE ANSWERS ARE CORRECT** (re-verified 2026-09-16) |
| p300 | 2 | engine init fixed; multi-die DECODE hangs -- use `serve_lanes.sh` |
| p300x2 | 4 | same: init and prefill work, decode hangs |

Re-verified on 2026-09-16 by rebuilding the package and reading the COMPLETION TEXT:

    "The capital of France is"  -> " Paris, a city renowned for its iconic landmarks
                                    such as the Eiffel Tower, the Louvre Museum, and
                                    the Notre-Dame Cathedral..."
    "the three laws of thermodynamics?" -> "1. **First Law (Law of Conservation of
                                    Energy):** Energy cannot be created or destroyed..."

**That check is the point, and the previous one was not good enough.** The earlier
verdict "ANSWERED -- the package serves" was produced by asserting that a request
returned SOMETHING. It did: it returned fluent gibberish, because the image carried the
pre-fix code that freed the GDN conv buffers the decode trace had baked. A package that
answers incorrectly is not a deliverable, so the check now reads the text. (That stage's
own log also ends in `FAIL :: serve verification` despite the status file recording PASS,
so it had never cleanly passed either.)

p300 and p300x2 both get FURTHER than before -- the fabric fix takes them through engine
init and prefill -- but decode then hangs on a multi-die mesh, so neither is a working
serve profile yet. One replica per die via `tools/serve_lanes.sh` remains the measured
multi-die deployment.

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

Decode is **dispatch-bound**. A program has a floor that dominates what it computes, so
decode time is essentially `op_count x (per-program cost)` and little else. Three
independent measurements say so.

One refinement, from the eight fusion commits that followed: **~8.4 us is an average over
a mix that includes the 1 MB recurrent-state ops, and it overstates a SMALL program.**
Those commits removed 870 programs for 4.34 ms, i.e. ~5.0 us each, and one of them
measured its own three folded eltwise ops at ~1.5 us. Use ~5-6.5 us when costing a small
eltwise or relayout program and ~10.35 us for anything touching the 1 MB state; ~8.4 us
is still right for a whole-layer average.

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

**Tensor parallelism buys very little FOR DECODE -- and that is not the same as "buys
very little", which is what this section used to claim. Corrected 2026-09-17.**

For DECODE the reasoning holds. TP divides the DATA per op, not the NUMBER of ops. Only a
handful of ops per layer are big enough for size to matter (the 1 MB state ops, ~6 of
109), so TP would shave those from 10.35 to 5.50 us and leave the other hundred at the
floor, while adding two all-reduces per layer. The per-op excess over the floor is mostly
op TYPE and CORE UTILISATION, neither of which another die fixes.

**But that argument was generalised to prefill, where it is wrong, and prefill is this
port's worst metric.** Prefill is COMPUTE-bound (stated a few lines below, in this same
document), so dividing the data per op is exactly what helps it. Measured here: prefill is
flat at ~394 tok/s against 1,016 for qwen3-coder-30b-a3b and 2,342 for ornith-1.0, which
is what makes TTFT 10.4 s at 4K and 185 s at 64K.

**Two factual claims in the old text were also simply wrong, and the counter-example is a
sibling model on this very host.** `tt-model/ornith-1.0-35b-p150`'s own docs say:

* it is NOT dense. It is a **256-routed-expert MoE with the same two layer kinds** --
  `linear_attention` (DeltaNet recurrent + conv state) and `full_attention` -- and the
  same `SparseMatmul` expert gate/up and down path. Calling the fleet's 30-35B models
  "dense" to explain away the comparison was incorrect.
* it runs **`TP = 4` for every dense tensor and `EP = 4` for the 256 routed experts** on
  these same four p300c chips, over `FABRIC_1D_RING`, and reports **3.51x / 3.29x warmed
  2048-token PREFILL** and 1.69x / 1.65x warmed traced decode, at PCC 0.999892 against its
  own single-chip baseline, with batch 32 and 262144 context intact.

Note how well that matches the corrected picture: ~1.7x on decode (small, as the
dispatch-bound argument predicts) and ~3.5x on prefill (large, because prefill is
compute-bound). So the right conclusion is not "TP buys little" but "TP buys little on
decode and a great deal on prefill".

**So replication is a CHOICE here, not an architectural necessity.** It was forced by two
implementation obstacles, both recorded in tt/model_config.py: this checkpoint ships q
with its output gate pre-fused as one [8192, 2048] tensor and stores GDN's qkv separately
from z/a/b where the TP fusions want a folded qkvzab, which surfaces as
"all_gather_minimal_matmul_async inner dimensions must match, K=8192 K_w=2048"; and
`ttnn.all_reduce` deadlocks in the full model on a (1,4) mesh. Neither is a property of
the architecture -- ornith-1.0 has the same architecture and does it -- and the second has
a documented workaround already noted here (all-gather + local sum).

Extra dies still serve extra USERS well (see the lane table), and that remains the right
deployment TODAY. But "extra dies cannot help this model" is not supported, and the
prefill gap is the reason to revisit it.

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

* **The `[B, T, H]` vs `[1, B, H]` convention collision** -- the `sdpa_decode` Q layout,
  which is `[T, B, H, D]`. Fixed by permuting in and back out, in both the paged and
  non-paged decode paths.

**Per-slot GDN state writeback on the replicated path -- DONE.** Batch 4 runs, with
verified cross-user isolation. Four different prompts prefilled into four slots, then
decoded together:

| slot | prompt | continuation |
|---|---|---|
| 0 | `Count upward: 1, 2, 3, 4, 5, 6,` | ` 7, 8, 9, 10` |
| 1 | `The letters in order are: A, B, C, D, E, F,` | ` G, H, I, J, K, L, M` |
| 2 | `Days of the week: Monday, Tuesday, Wednesday,` | ` Thursday, Friday, Saturday, Sunday.` |
| 3 | `Colours of the rainbow: red, orange, yellow,` | ` green, blue, indigo, violet.` |

Each continues its OWN prompt: no state leaks between slots. Reproduce with
`tools/repro_batch_isolation.py`. Five real bugs had to be fixed to get there, each
listed in the commit: prefill writing its B=1 state over every row, decode_paged
accepting only a scalar position, the shared DN zero buffers caching the first shape
they saw, the prefill-trace capture binding batched buffers, and the rebind
deallocating split conv buffers the decode trace had baked addresses for (that last one
served fluent garbage rather than failing).

### Where batched SERVING actually stands

Measured against the shipped batch-1 baseline of 26.7 tok/s aggregate on one die:

| max_num_seqs | concurrency | aggregate | per stream | verdict |
|---|---|---|---|---|
| 1 | 1 | 26.7 tok/s | 28.8 | ships today |
| 4 | 2 | 16.1 tok/s | 9.0 | correct, but SLOWER |
| 8 | 2 | 9.9 tok/s | 5.0 | correct, but slower still |

Batched serving is currently a throughput LOSS, not a win, and the reason is structural:
with decode bucketing off every step computes all B slots, so one active user pays Bx.
Concurrency has to reach ~B before it breaks even (at max_num_seqs=8, eight concurrent
streams would be ~8 x 5 = 40 tok/s against 26.7). Making it pay at low concurrency means
slicing GDN state to the active bucket width, which is the bucketing work below.

For scale, `tt-hous/ornith-1.0-35b-p150` -- a fellow Ornith on this stack -- ships the
same `max_num_seqs: 1` in its manifest but its readiness run used **8**, and got **166
tok/s output throughput on 32 concurrent prompts** on one p150, with 41.3 tok/s
single-stream (TPOT 23.2 ms) against our 29. So this is achievable, and a large part of
the gap is single-stream decode, not batching alone.

`max_num_seqs` still ships at 1 because the SERVING path is not finished:

* **It DEADLOCKS after a batch condense.** Concurrency 2 works in ISOLATION (2 ok / 0
  failed, at both max_num_seqs 4 and 8). Run concurrency 1 first against the same
  server and the next concurrent pair never returns -- **900 s of client patience, 0
  bytes received**, so this is a deadlock and not a slow first compile. No server-side
  exception; the engine simply stops responding, and this is what also leaves the board
  at `Read 0xffffffff` afterwards.

  Located, not yet fixed. Attach with `sudo py-spy dump` to the EngineCore CHILD (not
  the API server, and ptrace_scope=1 means plain py-spy is refused):

        ttnn/decorators.py:1106                 <- inside a ttnn op, 235% host CPU
        _decode_gather            moe.py:663    <- ttnn.concat(outs, dim=2)
        _forward_prefill          moe.py:1100
        prefill_paged             model.py:3380
        prefill_paged_slots_replicated  model.py:1946

  So it is the SECOND prefill, on the per-token gather path, that never completes.

  **One of the five "eliminations" below does not hold.** Hypothesis 3 rests on
  `tools/repro_prefill_after_trace.py` running "the exact serving order" and passing. It
  does not run the exact serving order. Verified by reading it: every prompt is
  **12 tokens** (`p0`, `p1`, `p2` are all `(1, 12)`), and it parks only the DECODE trace
  via `prime_decode_trace` -- it never calls `capture_prefill_trace_chunked`. So it
  supplies neither factor the "a post-park compile clobbers the parked trace" mechanism
  needs, and its PASS eliminates nothing.

  That mechanism is not exotic here: `prefill_paged_slots_replicated` calls
  `self.prefill_paged()`, whose shapes all derive from the real prompt length, so it
  compiles a fresh program set per distinct T AT REQUEST TIME. Its TP sibling
  `prefill_paged_slots` calls `prefill_traced_chunked()` -- the bounded, pre-warmed
  masked-bucket set -- instead, and so does `prefill_chunked_peruser`. The replicated
  path is the only holdout. `warmup_prefill_masked_buckets`' own docstring says why that
  matters: "so a request never compiles after a trace is parked (a post-park compile
  clobbers the trace -> second-request hang, #48536)", and
  `tests/test_prefill.py::test_masked_bucket_after_trace_capture` says "This is the case
  that used to hang -- the eager short path would compile a fresh program per length and
  clobber the parked trace."

  **Measured here, 2026-09-16, and left honestly unresolved.** A standalone repro with
  prompts of three DISTINCT lengths (13/24/31) against identical ones (12/12/12):

  * the length variation is real and visible -- the differing-length second prefill took
    **10.61 s against 0.65 s** on a cold kernel cache, i.e. fresh program sets genuinely
    are built at request time;
  * but it dropped to **1.14 s** on a second run, because ttnn's on-disk kernel cache
    survives process exit. So the per-length cost is a one-time cache miss, not a
    per-process one -- which weakens (does not kill) the concern;
  * and NEITHER arm deadlocked -- but the test is INCONCLUSIVE, not a refutation, because
    `capture_prefill_trace_chunked` failed in every attempt (`concat shapes_match`, then
    `binary_ng` after binding the B=1 scratch), so the chunk-prefill trace was never
    actually parked. The hypothesis remains UNTESTED.

  Worth knowing for whoever finishes this: `_bind_gdn_prefill_scratch_replicated` and
  `_unbind_gdn_prefill_scratch_replicated` ALREADY EXIST (model.py) and are simply not
  used by `prefill_paged_slots_replicated`, which nulls `split_conv_state` and takes the
  eager path instead. The pieces for the swap are in the tree.

  The original five hypotheses, for reference (each reverted rather than shipped on a
  guess, since none fixed it):

  1. *Data-dependent shapes.* Short prompts reach `_decode_gather` because the dispatch
     is `decode = T == 1 or T < self.CHUNK`, so expert shapes follow the prompt length.
     Forcing multi-token prefills onto the fixed-shape `_prefill` path
     (`decode = T == 1`) does NOT fix it. (Also costs TTFT, so not worth keeping.)
  2. *L1 pressure in the concat.* Accumulating the per-token outputs and the concat in
     DRAM for T != 1 does NOT fix it -- so the host blocking at `ttnn.concat` is where
     it waits, not what deadlocks.
  3. *Trace-capture order.* `tools/repro_prefill_after_trace.py` runs the exact serving
     order without vLLM -- prefill slot 0, capture the decode trace, decode, prefill
     slot 1 -- and it PASSES (0.33 s).
  4. *Condense remap and multi-request prefill.* The same repro extended with a real
     slot remap plus a two-slot prefill in one call also PASSES (0.65 s).
  5. *Decode/prefill overlap.* The plugin overlaps decode with prefill
     (`async_decode`/`note_prefill_submitted`), so a replay could be reading GDN buffers
     while prefill rebinds them. Adding `ttnn.synchronize_device` barriers around the
     rebind does NOT fix it.

  Since the model-level repro passes in the serving order, the remaining difference is
  inside the plugin's `execute_model` path. Next step needs either a watcher-enabled
  tt-metal build to name the stuck kernel (`TT_METAL_WATCHER` is compile-time and this
  build ignores it -- no `generated/watcher` is produced), or bisecting the plugin.

  Practical note for whoever picks this up: `ptrace_scope=1` here, so plain `py-spy`
  is refused. Use `sudo -n env "PATH=$PATH" py-spy dump --pid <EngineCore CHILD>`; the
  API server's own stack shows only the asyncio loop.

* Non-streaming requests are correct at concurrency 1 and 2 (checked by hand).
* `serve_bench.py`, which streams, fails at concurrency >= 2. Streaming at
  max_num_seqs > 1 is the next thing to debug.
* **It wedges the board on shutdown.** After a `max_num_seqs=4` serving run the next
  device open fails with `Read 0xffffffff over PCIe ID 0: the board should be reset`,
  with no process holding `/dev/tenstorrent/*`. Recovery is `tt-smi -r`. Seen twice,
  both times immediately after a batched serving run and never after a batch-1 one, so
  budget a reset before trusting any measurement that follows one.
* Decode bucketing has to be off (`TT_DECODE_BUCKETING=0`) or the width-1 bucket runs
  against [B, ...] GDN state and dies in the conv shift
  (`ttnn_gated_deltanet.py:145`, "[1,1,8192] does not match [4,1,8192]"). With it off
  every step computes all B slots, so one active user pays 4x: 9.4 tok/s per stream
  against 29. Slicing GDN state per bucket width is what removes that.

So the model-level win is real and measured -- per-die aggregate, pure `execute_trace`:
B=1 34.5, B=2 45.4, B=4 54.7, B=8 60.8 tok/s -- but it is not yet collectable through
the server. Historical note, the blocker that started this:

    TT_FATAL: Input tensor shape Shape([1, 1, 8192]) does not match
              output tensor shape Shape([4, 1, 8192])

vLLM prefills ONE sequence at a time into its decode slot, so the conv state coming back
is `[1, 1, 8192]` while the external buffer is `[B, 1, 8192]`. `gdn/decode.py`'s
in-place branch reshapes the recurrent state to match but copies the conv state whole,
over every row. The fix is a single-row edit that preserves the other (live) rows;
`gdn/tp.py` already implements exactly this for the TP path (`_slice_along`,
`_write_recurrent_state_prefix`, rebuild-by-slice+concat then `ttnn.copy` back to
preserve the trace's baked address), so the pattern exists and needs porting to
`Qwen36GatedDeltaNet`. Batched prefill is also still gated on `num_devices > 1`
(`qwen36_vllm.py:206`), which a replicated model never satisfies.

Worth it: per-die aggregate, pure `execute_trace`, measured -- B=1 34.5, B=2 45.4,
B=4 54.7, B=8 60.8 tok/s. That is +32% to +76% per die, and it multiplies with lanes.

Note the cross-user check failing at B>=4 is NOT a reason to hold this: it is the same
non-determinism that already affects batch 1 (see NONDETERMINISM.md), same magnitudes,
same intermittency. Batching does not make quality worse than what ships today. What
does need real validation before enabling it is cross-user ISOLATION -- that one
sequence's state cannot leak into another's slot -- which is a functional check
(different prompts, coherent independent answers) rather than a bitwise one.

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
  params, not tuning -- with one CAVEAT now measured. The claim that "without a trace
  region, trace capture fails and traced decode silently falls back to eager" is NOT
  true: on a device opened with ttnn's defaults (l1_small_size=0, trace_region_size=0,
  1 command queue) capture SUCCEEDS, returning a real MeshTraceId, and traced decode
  measures ~33 tok/s/user rather than eager's ~3.4. The GDN conv also does not need the
  L1-small reservation at these shapes. Keep passing the documented params anyway --
  every number in this file was measured with 2 command queues and a 1 GB trace region,
  and that is the configuration to reproduce -- but a missing trace region is not the
  silent-fallback trap this used to claim.
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
