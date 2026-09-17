# Ornith-1.5-35B-A3B — handoff, 2026-09-17

## Pick up here

```bash
tmux attach -t ornith          # the verification pass is running in window "verify"
tail -f /home/raahem/ornith-bringup/handoff/log_run_all.txt
```

`run_all.sh` runs three steps unattended, each resetting the dies first and writing its
own log in this directory:

| step | log | what it proves |
|---|---|---|
| `verify_pkg.sh` | `log_verify_pkg.txt` | the packaged container serves **correct text** on p150, and advertises `max_model_len: 262144` |
| `perf_p150.sh` | `log_perf_p150.txt` | TTFT + traced decode, p50 of 3, ornith-1.0's method |
| `verify_lanes.sh` | `log_verify_lanes.txt` | four lanes each answer correctly, one per die |

Everything there exercises the **shipping** configuration. None of it depends on tensor
parallelism.

## What ships, and the numbers

| | 1 die (default `p150`) | 2 dies | 4 dies |
|---|---|---|---|
| deployment | container, `p150` profile | 2 lanes | 4 lanes |
| decode | **32.99 tok/s** (TPOT 30.3 ms) | 28.40/stream | 25.89/stream |
| aggregate | 26.07 | 54.47 | **91.24 tok/s** |
| TTFT | **366 ms** (short), 5.04 s (~1.4K) | 404 ms | 420 ms |
| context | 262144, verified live from the container | | |

Long context on p150: prefill flat **~350-370 tok/s out to ISL 131072**, TPOT 30.3 ->
38.2 ms. Only 10 of 40 layers hold KV, which is why it is flat.

Against the fleet on four dies: ornith-1.0 43.03 tok/s decode / 2342 pf@4K, qwen3-coder
49.2 / 1016. We beat ornith-1.0 **2.1x on aggregate throughput** and **2.8x per die**
(32.99 against its 11.59 on one die); we lose on single-user decode and on prefill
without TP.

Caveat on the 2- and 4-die rows: those aggregates are from 2026-09-16 (256-token
completions, a different harness). Single-die decode measured *higher* tonight (32.99 vs
29.05), so they are probably conservative, but they have not been re-measured with
current code. `verify_lanes.sh` re-confirms correctness, not throughput.

## Tensor parallelism: opt-in, one blocker left, precisely localized

`ORNITH_TP=1`. On four dies it is worth **2.0x prefill at ISL 128 (706 tok/s), 3.1x at
4096 (1241, which passes qwen3-coder's 1016), and TTFT 366 -> 187 ms**, and served
traced decode measured **37.50 tok/s** — faster than one die and 87% of ornith-1.0.

Two fixes got it there, both committed:
* **The ring topology.** `moe.py::_reduce` was calling `ttnn.all_reduce(...,
  topology=Topology.Linear)` with no `num_links` on hardware whose descriptor declares
  `dims: [2,2]`, `channels { count: 2 }` — a line on a ring over one link. ornith-1.0 had
  measured and documented `Ring` + `num_links=2` as better at every traced shape. Worth
  ~6x. This is the single most valuable thing found in this bring-up.
* **A `getattr` guard** in `prefill_masked_bucket`: the GDN conv re-seed is a
  single-device construct and `TPGatedDeltaNet` has no `split_conv_state`, which killed
  the engine on the first TP request while warmup passed clean.

**THE ONE REMAINING BLOCKER.** The TP GDN depthwise conv1d cannot fit L1 during the vLLM
prefill warmup:

```
gdn/tp.py:351 _conv1d_prefill
  Statically allocated circular buffers on core range [0-0 - 10-5] grow to
  1684544 B which is beyond max L1 size of 1572864 B      (deficit 111,680 B)
```

What is **ruled out by measurement** — the byte count is *identical*, to the byte, in
every one of these, which is how we know it is one fixed CB configuration and not a
sizing problem:

* `max_model_len` 4096 vs 2048
* with and without an `l1_small_size: 24576` reservation
* capturing the chunk-prefill trace vs warming the masked buckets only
* masked-bucket set capped to 1024 vs the full `(128,256,512,1024,2048)`
* warmup chunk 2048 vs 1024

So the next step is to look at the CB config inside `_conv1d_prefill` for the TP head
width (`qkv_dim_tp`), not at chunk/bucket/context knobs. Two disproven attempts (a bucket
cap and a chunk halving) were reverted rather than left in shared code.

`trace_mode: decode_only` lets a 4-die TP server *start* (it skips the chunk-trace
capture) and is how the 37.50 was measured — but it leaves prefill unwarmed, so a
post-park compile then deadlocks request 2 in that same conv1d. Both paths lead to the
same op. Reproduce with:

```bash
/home/raahem/ornith-bringup/handoff/tp_stall.sh     # stalls on request 2, dumps py-spy stacks
```

## Also open

* **Prompt 2 accuracy**: logit PCC 0.94334 against the 0.95 bfp4 bar (top-1 3/3, other
  two prompts pass). Settled that no precision knob closes it — nine configs measured,
  the shipped default is the maximum. The one reducible lead is the GDN block at 10.0%
  median relative error vs full attention's 2.4%, with 34-45% at L4.
* **Decode is not bit-reproducible** (`NONDETERMINISM.md`) — open, characterised,
  accepted for release.
* **`max_num_seqs: 1`**; concurrency comes from lanes. Batched serving deadlocks in the
  same compile-after-trace family (`tools/repro_prefill_after_trace.py`).
* **4-lane simultaneous startup flaked twice** at the end of the session (1 of 4 ready
  inside the 1800 s budget, the others parked at `tt/worker.py`'s DP-binding line with no
  error). 4/4 has worked repeatedly, including earlier the same evening, and the lane that
  came up had the new parser flags, so this is a startup flake to investigate rather than a
  regression. Recovery: `serve_lanes.sh stop`, `tt-smi -r`, start again.

## Not done deliberately

**Not pushed.** `tt-model.yaml` still has `repo: PLACEHOLDER/ornith-1.5-35b-a3b`. Set the
repo, then:

```bash
tt-model push /home/raahem/ornith-bringup/artifacts/pkgbuild-0557/ornith-1.5-35b-a3b --private
```

**Package with `--out` OUTSIDE the tt-metal tree** or the build never finishes — the
packager stages into `build/` inside the model dir while `source.code` includes that
dir's parent, so it copies its staging directory into itself ~40 levels deep until
"File name too long". `build/` is gitignored, so `git status` stays clean while 1.7 GB
accumulates.

**The plugin is packaged from `/home/raahem/vllm-tt-plugin`** (`runtime.plugin.path`), and
two capabilities this model needs live on branch `ornith-1.5-replicated-moe-capabilities`
(`5961e48`): `kv_heads_replicated` and `supports_standard_dp`. They were previously
UNCOMMITTED, so the image depended on a dirty working tree. Merge or cherry-pick them.

## Hardware hygiene that cost real time

* **Never `kill -9` a process holding the mesh.** It wedges `MeshCommandQueue` for every
  later run and the next failure looks like a model bug — a cached weight upload stalled
  six minutes in `enqueue_write_shards_nolock`, and `test_layer.py` went from 16 minutes
  to 17 seconds after `tt-smi -r`. Use `resetdev.sh`, which SIGTERMs holders, waits, and
  refuses to reset while any process still holds a die (checked through `/proc`, not a
  lock file — a stale lock and a live run are indistinguishable).
* **`tt-smi -r` on live dies destroys that process's work silently.** It cost ~8 minutes
  of an agent's run here.
