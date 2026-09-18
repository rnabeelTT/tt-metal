# Ornith-1.5-35B-A3B — handoff, 2026-09-18

## State

Shipping configuration is measured, correct, rebuilt and verified. `repo:` is set.
Nothing has been pushed: the box has no Hugging Face credentials at all, so the push is
yours.

**Rebuilt and re-verified 2026-09-18** (`handoff/build_pkg.sh`, then
`handoff/verify_final.sh`), bundle at `artifacts/pkgbuild-final/ornith-1.5-35b-a3b`,
image `tt-model/ornith-1.5-35b-a3b:fb4b1ebf493b`:

| gate | result |
|---|---|
| image provenance | plugin `5961e48`, repo `raahemnabeel/ornith-1.5-35b-a3b` |
| container serves correct text on p150 | pass, `max_model_len: 262144` |
| p150 perf through the container | TTFT **367.5 ms**, decode **32.77 tok/s**, TPOT 30.5 ms |
| four lanes, one per die | **4/4 ready**, all four correct, `finish_reason: stop` |

The rebuild mattered for more than the numbers: the previously built image labelled its
plugin `bef89e42`, the commit BEFORE the replicated-MoE capabilities, because those were
sitting uncommitted in the plugin working tree. They are now committed and
fast-forwarded onto the plugin's `main` (`5961e48`), and the image records it. The older
image `tt-model/ornith-1.5-35b-a3b:82768e358642` still carries `PLACEHOLDER` and the old
plugin label; delete it so nobody serves it by accident.

```bash
hf auth login
tt-model push /home/raahem/ornith-bringup/artifacts/pkgbuild-final/ornith-1.5-35b-a3b --private
```

## The numbers

Re-measured 2026-09-18 with `tools/serve_bench.py`, one stream per lane, 256-token
completions, **all three rows in one pass against one set of four lanes** so they are
directly comparable. 12/12 requests served, none failed. Every lane's text was checked
through the reasoning parser before any timing was taken.

| | p150 (1 die) | 2 dies | 4 dies |
|---|---|---|---|
| deployment | 1 instance | `serve_lanes.sh start 2` | `serve_lanes.sh start 4` |
| **aggregate** | **31.66 tok/s** | **61.55 tok/s** | **109.03 tok/s** |
| per stream | 33.15 | 32.35 | 28.52 |
| TTFT (median) | 394 ms | 395 ms | 410 ms |
| TTFT (p99) | 394 ms | 398 ms | 418 ms |

3.44x at four dies. At 128-token completions the same lanes give 30.35 / 59.06 / 102.10 —
shorter completions amortize the fixed ~400 ms prefill over fewer tokens, so **always
quote the completion length with the number**.

Every one of these beats what the package previously recorded (26.07 / 54.47 / 91.24
aggregate, 29.05 / 28.40 / 25.89 per stream). Those were from 2026-09-16 on a retired
harness and were flagged as un-re-measured; they were conservative.

Through the rebuilt container on one die, p50 of 3, ornith-1.0's method: **decode 32.77
tok/s (TPOT 30.5 ms), TTFT 367.5 ms** short / 5.03 s at ~1.4K. Context **262144**, verified
live from the container. Prefill flat ~350-370 tok/s out to ISL 131072 — only 10 of 40
layers hold KV, which is why it is flat.

### Reading this against the fleet table

The fleet latency sweep measures **Users = 1** — single-user decode per mesh. This model
is replicated, so extra dies serve extra USERS and do not make one stream faster. In the
fleet's frame this port is ~33 tok/s at every mesh, not a scaling curve.

| dies | ornith-1.0 | this port |
|---|---|---|
| 1 | 11.59 | **33.15** (2.86x) |
| 4, one user | **43.03** | 33.15 one lane / 37.50 with opt-in TP |
| 4, four users | 43.03 | **109.03 aggregate** (2.53x) |

Do not quote 28.52 as the single-user four-die number. It is per-stream with four streams
running, and the gap from 33.15 is host contention between four vLLM processes. One user
on a four-die box runs one lane and gets 33.15 with three dies idle.

## What was wrong, and is now fixed

**The 4-lane "startup flake" was neither flaky nor mysterious.** All four lanes failed
identically at exactly 600 s with `TimeoutError: Timed out waiting for engine core
processes to start (VLLM_ENGINE_READY_TIMEOUT_S)`. The previous handoff recorded them as
"parked at tt/worker.py's DP-binding line with no error"; the error was in every log.

The cause was in `serve_lanes.sh`, introduced by the parser fix (28147b0b5a0), which
inserted a comment block between the env assignments and the command they prefix:

```bash
TT_VISIBLE_DEVICES="$i" \
TT_MESH_GRAPH_DESC_PATH="$MESH_DESC" \
# --reasoning-parser / --tool-parser ARE NOT OPTIONAL, ...
nohup vllm serve "$WEIGHTS" \
```

A `#` line after a trailing backslash **ends the continuation**. Bash parses the two
assignments as a complete command, setting them as plain shell variables in the launching
shell and never exporting them; `nohup vllm serve` then runs as a separate command with
neither in its environment. Every lane lost its die pinning and its single-die descriptor,
so all four tried to open the whole 4-die mesh at once and starved until vLLM gave up.
That is also why earlier attempts saw "1 of 4 came up" — whichever lane won the mesh.

Fixed in a34886f299d, with a `/proc/<pid>/environ` guard that refuses to continue if a
lane did not inherit its pinning. The guard **polls**: prefix assignments are applied at
exec, not at fork, so an immediate check false-fails a good launch — which it did on its
first run.

`verify_lanes.sh` also swallowed its own failure (it returned the exit status of a final
`echo`), which is how `run_all.sh` printed `exit=0` over a 0/4 result. Also fixed.

## The single biggest win available, and exactly what blocks it

**5.83 ms/token — 19% of decode — never touches the dies.** Measured on p150:

    device model (tools/layer_sweep.py)   24.67 ms
    served TPOT                           30.50 ms
    SERVING OVERHEAD                       5.83 ms

That is the full `[1, 1, 248320]` logits readback and its per-step sync. ornith-1.0's
served 23.3 ms on FOUR dies essentially equals its own device model, so it carries no
equivalent tax — and our SINGLE-die device compute (24.67 ms) is within 6% of their
four-die total (23.30 ms). The deficit a user sees is almost entirely this overhead, not
compute, not op count, not collectives, and not the architecture.

On-device sampling removes it. Measured on the TP path, it is worth 2.4 ms
(28.53 -> 30.57 tok/s). On one die it would be worth the lot.

**Two things that do NOT block it, both checked:**
* `ttnn.topk`'s 64K width limit. `TTSampling.multi_step_reduction` cuts the padded vocab
  into chunks and is enabled precisely when the mesh is `[1, 1]` (`tt_sampling.py:192`) —
  it exists FOR the single-die case. 248320 pads to 262144 = 4 chunks of exactly 65536,
  tile-aligned; `TTSampling.num_single_device_vocab_splits` accepts it.
* The vLLM plugin. It only reads the `supports_sample_on_device` capability, which we
  already declare. The "certified TP topology (1x4 or 1x8)" wording in tt-model.yaml was
  describing qwen36's OWN gate, not a plugin rule.

**What DOES block it:** `TTSampling.max_batch_size` is floored at 32 for tile alignment
(`tt_sampling.py:229`), so the sampler's parameter tensors are 32 rows while single-die
decode emits one:

    TT_FATAL: Shape of Output tensor Shape([1, 248320]) provided does not match the
    broadcasted output shape Shape([32, 248320])

**And the fix is smaller than that error suggests.** Tracing it: the crash is the additive
invalid-vocab mask, built once at `max_batch_size` rows
(`tt_sampling.py:_create_invalid_vocab_mask` -> `build_invalid_vocab_mask(..., self.max_batch_size)`)
and added to the logits in four places. The mask is **row-invariant** — `vocab_padding.py`
builds `torch.zeros(1, 1, max_batch_size, padded_vocab)` and sets the same padded tail on
every row, because it marks padded VOCAB columns, not batch slots. So a 32-row mask against
1-row logits is both the crash and pure waste, and cutting the mask to the logits' row
count is exact rather than an approximation.

A working patch is four call sites plus a cached `_rows_of(mask, logits)` helper that
returns the mask untouched when the counts already agree (so every currently-working model
is unaffected) and otherwise slices once per row count at warmup. It was written and
reverted rather than shipped: with the single-die gate still closed nothing in the shipped
profiles exercises it, and an unexercised change has no business in a release that is
push-ready. Re-apply it together with the gate in `qwen36/tt/model.py` (allow `(1, 1)` when
`TTSampling.num_single_device_vocab_splits(padded_vocab)` is not None) and gate the pair on
`tests/test_model.py` PCC, because a wrong sampler produces plausible text that smoke tests
pass.

Worth more than anything else outstanding: it is the difference between 33.21 tok/s served
and the 40.5 tok/s the dies are already capable of.

## What TP already shards, and the one tensor it does not

Measured from the per-device shapes the 4-die run actually loads, so this is what the
hardware holds rather than what the code intends:

| tensor | global | per die at TP=4 | |
|---|---|---|---|
| `qkvzab.il` fused in-projection | 2048 x 12352 | 2048 x **3088** | quartered |
| `out` row-parallel out-projection | 4096 x 2048 | **1024** x 2048 | quartered |
| routed experts `up_gate` | [1, **256**, 2048, 1024] | [1, **64**, ...] | EP=4 |
| routed experts `down` | [1, **256**, 512, 2048] | [1, **64**, ...] | EP=4 |
| norms (`weight_distributed`) | 64 | **16** | quartered |
| **shared expert** `sh0` / `sh1` / `sh2` | 2048 x 512, 512 x 2048 | **unchanged** | **REPLICATED** |

So the hidden and intermediate dimensions are already quartered; the shared expert is the
only tensor left at full width on every die. ornith-1.0 does shard its equivalent
(`shared_in` 1056 -> 288 per device).

Worth roughly **0.52 ms/token**: three matmuls of ~2 MB at bf16 is ~6 MB/layer, ~17 us at
this board's measured bandwidth, so quartering saves ~13 us/layer x 40 layers. That is 1.5%
of the 33.7 ms TP TPOT -- 29.65 -> about 30.1 tok/s. It is worth twice as much to
ornith-1.0, whose shared expert is 1056 wide against this model's 512.

Not done: 1.5% did not justify changing shared TP code against an already-verified
package. It is the cheapest remaining TP item if someone wants it.

**And one thing NOT to try.** Running the routed experts DENSE over each die's 64-expert
shard instead of gathering the routed top-8 would be ~32x MORE expert math per die at
decode, not less: top-8 of 256 puts about two experts on each die, and dense would compute
all sixty-four. ornith-1.0 is not dense either -- its own decode profile shows
`SparseMatmul (routed experts)` at 19.99% / 24.46% of the window, running at `active=4`
per die. Their 564 us/layer comes from quartering TENSOR dimensions, which is already done
above, not from densifying experts.

## The one unexplained TP number, and a dead end to not re-walk

**Our TP ops cost ~46% more than ornith-1.0's at the same data size, and nothing here
explains why.** Backing the collectives out of the measured per-layer times:

    ornith-1.0 4-die   564 us / 98 ops                      = 5.75 us/op
    us 4-die           734 us - 92.6 us collectives = 641 us
                       641 us / 76 ops                      = 8.43 us/op

Both sides are quartered -- our per-device shapes are verified above -- so this is not a
sharding gap.

**CAUSE FOUND: the decode matmul SHAPES are TP-aware and the CORE COUNTS are not.**

`tp_common.create_matmul_1d_decode_progcfg(m, k, n, num_cores, ...)` computes
`per_core_n = ceil(n_tiles / (cols * rows))`. In `model_config.py` the `n` passed in is
per-device (`hidden_dim // tp`, `attn_qkv_fused_dim_tp`, ...) but `num_cores` is a
HARDCODED CONSTANT -- 44, 64, 33 -- chosen by `test_mlp_matmul_sweep` at SINGLE-CHIP
widths, as the quoted timings show (42.8 us, ~59 us, ~63 us are full-width numbers). At
TP=4 the width is quartered and the core budget never moves, so the grid is wrong in both
directions at once:

| projection | N/die | tiles | cores now | per_core_n | at 110 cores |
|---|---|---|---|---|---|
| `gdn_qkvz` | 3088 | 97 | 44 | **3** | 1 |
| `attn_qkv` | 2304 | 72 | 64 | **2** | 1 |
| `mlp_w2` | 2048 | 64 | 33 | **2** | 1 |
| `mlp_w1/w3` | 128 | 4 | 44 | 1 | 1 (40 cores idle either way) |

Blackhole's worker grid is 11 x 10 = **110 cores**. We use 33-64 of them and run
`per_core_n` of 2-3; ornith-1.0 spreads the same class of op over all 110 and deliberately
targets `per_core_n = 1` ("at 110 realised cores the 80 output tiles give one tile per
core", and a larger subblock needing fewer cores measured SLOWER on their ladder). Summed
across the four dense decode projections we run 8 units of serialised per-core work where
110 cores would give 4.

Their own doc names the same trap from the other side: "that measurement is at single-chip
widths, and no DRAM-sharded candidate exists at the per-device multichip widths in any
artifact." They re-ran a 9-target x 6-cap ladder AT PER-DEVICE WIDTHS. This port never has.

**The fix is a re-sweep, not a rewrite.** `num_cores` is already a parameter; only the
constants are wrong for TP. Sweep them at the per-device widths in the table (and pass
`grid_w=self.decode_grid_w` on `attn_qkv`, which still uses the legacy cols<=8 shaping),
then select per `tp` rather than per model. `mlp_w1/w3` wants the opposite treatment: at
N=128/die it has four output tiles and 40 of its 44 cores sit idle paying multicast setup,
so it should shrink, not grow.

Sizing it honestly: dense `Matmul` is 14.82% of ornith-1.0's linear-decode window, so if
ours is comparable (~110 us of the 734 us layer) and per_core_n halves, this is roughly
50-70 us/layer = **2-2.8 ms/token**, taking 4-die decode from 33.7 ms to about 31 ms
(~32 tok/s). Real and low-risk, but not the whole 200 us/layer -- the remainder is still
unaccounted, and activation memory configs on the sharded path are the next suspect.

It needs per-op DEVICE timing under TP, which is still partly blocked: full-attention
layers under TP fail in `slice.cpp:121` (`input_rank == begins.size()`), the sixth bug in
that family and the only one left unfixed. GDN layers sweep clean.

**A dead end, stated so it is not re-walked.** There is no "fast-path dispatch queue" that
contiguous strides unlock. The 5.45 / 5.75 / 8.4 us figures that suggest one are a SIZE
curve plus an average, not two dispatch modes: `multiply` measures 5.45 us at 8 KB, 5.46 at
64 KB and 6.36 at 512 KB, and the ~8-10 us numbers are whole-layer averages that include
the 1 MB GDN recurrent-state ops at 10.35 us each. Re-aligning strides cannot move ops onto
a queue that does not exist; the 46% above is real work, wherever it turns out to live.

## The sixth TP bug: diagnosed, NOT fixed, and it blocks the profiling above

`TPAttention._qkv` assumes a rank-4 activation and builds four-element slice bounds:

    models/demos/blackhole/qwen36/tt/attention/tp.py:192-199
        sh = list(qkv.shape)
        _qkv3_mc = ttnn.L1_MEMORY_CONFIG if sh[2] > tpc.TILE_SIZE else ttnn.DRAM_MEMORY_CONFIG
        qkv3 = ttnn.slice(qkv, (0, 0, 0, 0), (sh[0], sh[1], sh[2], qkv3_dim), memory_config=_qkv3_mc)
        gate = ttnn.slice(qkv, (0, 0, 0, qkv3_dim), (sh[0], sh[1], sh[2], qkv3_dim + gate_dim))

Reached with a rank-3 `qkv` it fails in validation:

    TT_FATAL: Input rank 3 and begins 4 must have the same size
    slice.cpp:121: input_rank == begins.size()
    via layer.py:238 forward -> attention/tp.py:428 forward_prefill -> :198 _qkv

Reproduce: `ORNITH_TP=1 python3 tools/layer_sweep.py --mesh P150x4 --spec full --n 4`
(full log: `handoff/decompose/P150x4_full_n4.log`). GDN layers under TP sweep clean; only
full-attention layers hit this.

**Same family as the five fixed in 7dbdd2de25c** -- a hardcoded rank that happens to hold
on the vLLM server's path and not on any other. The server serves all 40 layers under TP
perfectly well, so this is not a serving defect; it is the last thing keeping standalone
tools off the TP full-attention path.

**The fix is rank-agnostic bounds**, not a special case: build begins/ends from `len(sh)`
so both rank-3 and rank-4 work, e.g. zeros for begins and `sh[:-1] + [qkv3_dim]` for ends,
and take the sequence extent as `sh[-2]` rather than `sh[2]`. Note `sh[2]` is wrong twice
over on a rank-3 input -- it selects the hidden dim, so even the L1-vs-DRAM memory-config
choice on the line above is made on the wrong axis.

**Why it matters beyond tidiness:** it is what blocks per-op DEVICE timing for
full-attention layers under TP, which is exactly what is needed to explain the 8.43 vs
5.75 us/op gap above. Fix this and both `tools/op_profile.py` and `tools/layer_sweep.py`
can decompose the whole TP model instead of GDN only.

Not fixed here because the shipped package is built and verified from this tree and the
change cannot be exercised without a rebuild cycle; tree and image were kept identical
instead. It is a small, well-understood change for whoever picks this up.

## Still open, all pre-existing and none blocking the shipping config

* **Tensor parallelism is opt-in and has one blocker left**, precisely localized. The TP
  GDN depthwise conv1d wants 1684544 B of CBs against 1572864 B of L1 (deficit 111,680 B)
  at `gdn/tp.py:351 _conv1d_prefill`. The byte count is *identical* across `max_model_len`
  4096/2048, with/without `l1_small_size`, trace-capture vs buckets-only, bucket caps, and
  chunk 2048 vs 1024 — so it is the CB config at the TP head width (`qkv_dim_tp`), not a
  knob. Reproduce with `handoff/tp_stall.sh`. Worth 2.0-3.1x prefill and 37.50 tok/s
  served decode if closed.

  **The named knob to try first is `act_block_h_override`, and it has not been tried.**
  `ttnn.Conv1dConfig` is an alias for `Conv2dConfig`
  (`ttnn/ttnn/__init__.py:612`), which carries `act_block_h_override`
  (`conv2d_nanobind.cpp:320`) — the canonical control for precisely this
  "statically allocated circular buffers grow to N beyond max L1" failure. Every knob
  tried so far (`max_model_len`, `l1_small_size`, trace capture, bucket caps, chunk size)
  is OUTSIDE the conv, which is exactly why the byte count never moved: none of them
  reaches the activation block that sizes the CB. The deficit is 111,680 B of 1,684,544,
  6.6%, so a modest reduction should clear it. `_conv1d_prefill` builds its `conv_cfg` at
  `models/demos/blackhole/qwen36/tt/gdn/tp.py:323` with only `weights_dtype` and
  `shard_layout` set. Second lead if that costs too much throughput: `weights_dtype`
  is `bfloat16` there and the non-TP path tolerates less. Both are untested — no hardware
  time was spent on TP this session.
* **Prompt 2 accuracy**: logit PCC 0.94334 against the 0.95 bfp4 bar (top-1 3/3; the other
  two prompts pass). Nine precision configs measured; the shipped default is the maximum.
  The reducible lead is the GDN block at 10.0% median relative error vs full attention's
  2.4%.
* **Decode is not bit-reproducible** (`NONDETERMINISM.md`) — characterised, accepted.
* **`max_num_seqs: 1`**, a hard limit from the GDN external-state batch-1 assert.
  Concurrency comes from lanes.

## Hardware hygiene that cost real time

* **Never `kill -9` a process holding the mesh.** It wedges `MeshCommandQueue` for every
  later run and the next failure looks like a model bug. Use `handoff/resetdev.sh`, which
  SIGTERMs holders, waits, and refuses to reset while any process still holds a die
  (checked through `/proc`, not a lock file).
* **`tt-smi -r` on live dies destroys that process's work silently.**
* **Package with `--out` OUTSIDE the tt-metal tree** or the build never finishes: the
  packager stages into `build/` inside the model dir while `source.code` includes that
  dir's parent, so it copies its staging directory into itself until "File name too long".
  `build/` is gitignored, so `git status` stays clean while 1.7 GB accumulates.

## Scripts

All in `/home/raahem/ornith-bringup/handoff/`, deliberately outside the session
scratchpad.

| script | what it does |
|---|---|
| `measure_lanes.sh` | the table above: 4 lanes once, measured at 1/2/4 bases |
| `build_pkg.sh` | rebuild the container package into `artifacts/pkgbuild-final` |
| `verify_final.sh` | image provenance + serves-correct-text + p150 perf + 4 lanes |
| `verify_pkg.sh` / `perf_p150.sh` / `verify_lanes.sh` | the three individual gates |
| `resetdev.sh` | the only safe way to reset the dies |
| `tp_stall.sh` | reproduce the TP conv1d blocker |
