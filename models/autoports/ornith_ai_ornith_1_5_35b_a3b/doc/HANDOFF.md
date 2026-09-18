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
