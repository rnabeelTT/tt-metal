# GLM-5.3-Flash (E192) decode performance (stage 12)

Stage 12 makes three changes to the multi-slot decode step. Changes 1 and 2 are on by default and give bit-identical
outputs. Change 3 (per-row DSA buckets) is opt-in: the sparse-regime gate showed that it changes outputs (see
"Sparse-regime gate"). A later addendum (commit 685db8afb95) makes one user on a multi-slot build faster, with
bit-identical outputs (see "One user on a multi-slot build"); commit 613e4802772 does the same for the 32-slot wide
profile (see "Wide profile: tight sorted-dispatch bound"). This page gives the
defaults, the switches, the measured numbers and their scope, and the items that stage 12 rejected or did not start.

Commits: d86fe3c874d (MoE sentinel, persistent all-gather), ad427402ba2 (per-row DSA decode buckets), 685db8afb95
(one user on a multi-slot build), 613e4802772 (tight sorted-dispatch bound).
Hardware: p300x2, mesh (1, 4), TP4. Analysis that chose these items: the stage-12 decode gap report (GLM vs
Qwen3.5-122B), ranked items 2, 6 and 7.

## What changed

1. **Once-per-step MoE inactive sentinel** (`tt/model.py`, `GLM53_SENT_ONCE`). The term `(1 - active) * 65535` marks
   the inactive slot rows for the routed experts. Before stage 12, each of the 42 MoE layers computed it. Now
   `DeviceDecodeInputs.refresh` computes it once per step (DSA input `inactive_sent`), and every MoE layer reads it.
2. **Persistent all-gather with ping-pong buffers** (`tt/multichip_decoder.py`, `MeshCCL`, `DEFAULT_MC.ag_persistent`).
   The decode all-gathers write into preallocated DRAM buffers. There is one buffer per (call site, shape, dtype,
   layer parity). Layers L and L+1 use different buffers. A buffer read at layer L is written again at layer L+2,
   after every die issued its layer-(L+1) collectives. Stage 7 did not adopt `ag_persistent`, because one buffer per
   site was shared by all 45 layers and the persistent all-gather skips its start barrier. The parity buffers remove
   that reuse (history: `../optimized_full_model/README.md`, persistent all-gather section).
3. **Per-row DSA decode buckets on multi-slot builds** (`tt/dsa_buckets.py`, `DsaBuckets` row mode). Stage 10 turned
   the DSA width buckets off at S >= 2, because the batch-max bucket made one request's selection depend on the
   other slots (served check t23). In row mode, each active row selects its top-512 pools at its own bucket width.
   A step runs the selection at every width between the smallest and the largest bucket of its active rows. Then
   each row keeps the ids of its own bucket. Thus the selection of a row does not depend on the other rows.
   - Bucket contexts: 16,384, then T (the build's context). Bucket width W_b = T_b / 4 + 32 pools. A row at
     position p needs (p + 1) // 4 pools. Edge of the first bucket: p <= 16,514 selects at 4,128 pools;
     p >= 16,515 selects at the full width (`tests/test_dsa_buckets_host.py::test_bucket_boundary_at_16384_context`).
   - Each (lo, hi) bucket range is one point trace per DSA layer. 2 widths give 3 ranges (11 x 3 point traces).
   - Trace region on p300x2 (16 slots, 243,712): 88.1 MiB of 96 MiB (83.9 MiB without the DSA points).
   - 1-slot builds keep the stage-10 buckets (4,096 x 2^k, then T). The block-sharded layout (p300x2-1m) takes none.

## Defaults and switches

| Switch | Default | Effect |
| --- | --- | --- |
| `GLM53_SENT_ONCE` | `1` | `0`: the sentinel is computed in each MoE layer (stage 11). |
| `GLM53_MC=ag_persistent=0` | `ag_persistent=True` in `DEFAULT_MC` | `0`: per-call collective outputs (stage 11). |
| `GLM53_DSA_ROW_BUCKETS` | `0` | `1`: row mode (opt-in, outputs differ in the sparse regime). Any other value: no row mode. With auto buckets, S >= 2 then takes no DSA buckets (stage 11). With an explicit `GLM53_DSA_BUCKETS` list, S >= 2 then runs the batch-max bucket (not batch invariant; diagnostics only). |
| `GLM53_DSA_ROW_CONTEXTS` | `16384` | Comma list of row-mode contexts below T. Each must be a multiple of 128 and >= 2,048 (AssertionError otherwise). More widths need more trace region: see "Rejected". |
| `GLM53_DSA_BUCKETS` | `auto` | `0` / `off`: no DSA decode buckets at any slot count. |
| `GLM53_DSA_BUCKET_MIN_T` | `65536` | Builds with a smaller context take no DSA decode buckets. |
| `GLM53_KDA_ACTIVE` | `1` | `0`: the KDA recurrence computes every slot (before commit 685db8afb95). See "One user on a multi-slot build". |
| `GLM53_MOE_TIGHT_DISPATCH` | `1` | `0`: the sorted dispatch pads for all E experts (before commit 613e4802772). See "Wide profile: tight sorted-dispatch bound". |
| `GLM53_MOE_BUCKET_PATH` | `family` | `family`: every MoE row bucket takes the S bucket's family, except the 1-row bucket of the pairs family, which takes the T = 1 gather path (bit-identical). `pairs`: every bucket on pairs (before 685db8afb95). |

## Evidence

All paths are under `~/glm53-flash-bringup/`.

| Item | Path |
| --- | --- |
| Standalone A/B, 16 slots, 243,712: base (stage 11), cd (sentinel + ag_persistent), all (cd + row buckets) | `artifacts/s12/ab1/{base,cd,all}.pt`, `base_vs_cd.json`, `base_vs_all.json` |
| Build and timing logs (base, cd, the 3-width overflow; then 2 widths) | `logs/s12/dec_ab1.log`, `logs/s12/dec_ab2.log` |
| Served sweep, p300x2, before / after | `artifacts/s12/served/sweep_base.json`, `sweep_rowbk.json` |
| Served batch invariance (text and random-id prompts: solo, next to a long request, solo again) | `artifacts/s12/served/served_inv_rowbk.json` |
| Router precision check (rejected item) | `artifacts/s12/router_bf16_check.log` |
| top-k tie-order probes | `logs/s12/topk_probe.log`, `logs/s12/topk_chunk_probe.log` |
| One user on a multi-slot build: A/B, invariance, served, release gate | `one_user/` in this directory (`ship_vs_new.json`, `ab1cd_vs_ship.json`, `moe_t1_compare.json`, `standalone_invariance.json`, `served_inv_kact.json`, `sweep_kact.json`, `tf_release_record.json`, `wide_tight_dispatch.json`); packaged runs `scorecard_r7/`, `scorecard_r8/`; raw logs `logs/s12/dec_ab3.log`, `moe_t1.log`, `served_kact.log`, `tf_release_kact.log`; Tracy `artifacts/s12/tracy{,_kact}_s{1,16}` |

Correctness (dense regime, positions about 300; for the sparse regime see the gate below):
- base vs cd and base vs all: bit-identical logits, top-1 29 / 29 rows (1 and 16 active). These positions are below
  `DENSE_MAX_POSITION` (2,050), where DSA reads every pool, so this check does not test the selection.
- Served invariance (p300x2, measured with the opt-in row buckets on, `served_inv_rowbk.json`): identical tokens solo, next to a long request, and solo again (`solo_repeat_equal`,
  `concurrent_equal`, `long_still_decoding`: true). Commit ad427402ba2 also reports a standalone check: a target
  request alone vs next to a 70K filler was bit-identical, with the mixed bucket range included.

## Measured numbers

Standalone token-out (16-slot build, 243,712, shared 1,024-block pool):

| Active rows | Stage 11 | cd (sentinel + ag_persistent) | all (cd + row buckets) |
| ---: | ---: | ---: | ---: |
| 1 | 75.0 ms | 74.6 ms | 69.9 ms |
| 16 | 177.8 ms | 176.8 ms | 172.1 ms |
| Trace region | 86.6 MiB | 83.9 MiB (the sentinel's saving; ag_persistent does not change it) | 88.1 MiB |

Served TPOT, p300x2, greedy, same host load (stage 12 = with the opt-in row buckets on; the shipped default is the cd
column above, about 0.5 % faster than stage 11; the release scorecard measures it):

| Workload (input / output, users) | Stage 11 | Stage 12 |
| --- | ---: | ---: |
| 128 / 128, 1 user | 75.32 ms | 69.72 ms |
| 128 / 128, 16 users | 178.03 ms | 172.03 ms |
| 1024 / 256, 16 users | 180.72 ms | 173.93 ms |

Release scorecard on the r6 bundle (shipped defaults, row buckets off; `scorecard_r6/`): 128 / 128 at 1 user 74.80 ms, at 16 users 177.01 ms (77.8 output tokens/s); 1024 / 256 at 16 users 178.83 ms. TTFT 270 ms (128), 4.93 s (8,192), 93.5 s (131,072).

Release verify of the r6 bundle on `p300x2-1m` (warm, 2026-10-24, `~/glm53-flash-bringup/logs/11/verify-staged-p300x2-1m-r6.chain.log`):
- Healthy after 166 s, served `max_model_len` 1,048,576: the ping-pong all-gather buffers fit next to the full
  1,048,704-token sharded DSA pool.
- All smokes pass, and a 262,144-token prompt answers coherently in 582 s.
- The script verdict is FAIL on one gate only: `SMOKE_BURST 1 ... 0/1 identical to solo greedy`. Stage 11 had the
  same result (`doc/tti_release/RUN_NOTES.md`, warm verify p300x2-1m). It is the documented block-placement numerics
  of the sharded layout, not a stage-12 change. Classified PASS with that exception, as in stage 11.

Scope of these numbers:
- **All timed steps were all-short** (positions about 300, every row in the 16,384 bucket). Such a step runs the
  plain narrow selection (range (0, 0)).
- **Mixed batches are not measured.** A step with short and long rows (range (0, 1)) computes both selections plus
  about 8 element-wise ops per DSA layer (typecast, compare, multiply, add). It may be slower than stage 11, which ran
  one full-width selection. The sparse-regime gate below measures long prompts.
- **Host enqueue time rose** (segmented replay: 54 segments plus point traces instead of 43): 8.2 -> 20.3 ms at
  1 active, 19.4 -> 49.9 ms at 16 active (19.2 ms after cd). The step is device-bound, so the enqueue time is hidden
  in these runs. A faster device step or a slower host can expose it.
- The router bf16 check and the trace sizes are p300x2 numbers. p150x4 is derived, not run.

## Sparse-regime gate

Scripts `scripts/s12/gate.sh` and `gate2.sh` (review #2: a base with `ag_persistent` really off) on the committed code (snapshot of ad427402ba2), 16 slots, 243,712, 1 and 16 active,
6 greedy decode steps after the prompt. Evidence: `artifacts/s12/gate_p{2100,4000}/base_vs_{cd,all}.json`,
`artifacts/s12/gate_tf1_{off,on}.json`, `logs/s12/gate.chain.log`.

| Check | Result |
| --- | --- |
| stage-11 paths (`GLM53_MC=ag_persistent=0 GLM53_SENT_ONCE=0`) vs cd (sentinel + ag_persistent), prompt 2,100 | bit-identical, top-1 29 / 29 (`base_mc0_vs_cd.json`) |
| same, prompt 4,000 | bit-identical, top-1 29 / 29 (`base_mc0_vs_cd.json`) |
| ag_persistent on, sentinel off vs cd, prompts 2,100 / 4,000 (first gate run: its base left `DEFAULT_MC` on) | bit-identical, top-1 29 / 29 (`logs/s12/gate.chain.log`; gate2 overwrote that run's `base_vs_cd.json`) |
| base vs all (+ row buckets), prompt 2,100 | **not identical**, top-1 24 / 29, max \|d\| 0.13 .. 14.2 logits |
| base vs all, prompt 4,000 | **not identical**, top-1 23 / 29, max \|d\| 0.28 .. 13.7 logits |
| 1 slot (4,096-token build: replicated layout, no DSA buckets), ag_persistent off vs on, 966 teacher-forced rows | bit-identical. The p300x2-long bucket path and the sharded p300x2-1m layout are covered by the served release checks (scorecard p300x2-long smokes, p300x2-1m verify), not by this A/B |
| prompt 12,000 | not run: the test's 1,024-block shared pool is too small for 16 x 12,000 tokens |

Cause: on the device, the bf16 top-k selects a different top-512 set at a different width
(`logs/s12/topk_probe.log`: set_equal False between 4,128 and 16,416+ pools; neither width matches the fp32 set).
Row mode selects at 4,128 pools where stage 11 selects at the full width, so a row in the sparse regime
(positions 2,051 .. 16,514) attends to other pools. The difference shows already at the first decode step
(max |d| about 1 logit). Neither selection is the exact one, but the shipped evals (stage 11) ran on the full-width
selection.

Decision: row mode is opt-in (`GLM53_DSA_ROW_BUCKETS=1`, default `0`). The release keeps the full-width selection
and ships changes 1 and 2 only. To adopt row mode later: show that its quality is equal (for example a long-context
teacher-forcing check against the HF reference at positions 2,051 .. 16,514, or the release evals with row mode on).

## Rejected

**Router through `TTMoEGate` (gap report item 4).** The gate op computes the top-8 expert set in bf16. On 4,096 rows
per MoE layer (42 layers), its set matched the fp32 router set on only 90.9 % to 96.6 % of the rows (mean 95.1 %; so
about 4.9 % of the tokens route to a different expert set). The fp32 router matches the HF set on 99.98 % to 100 %
of the rows. The saving (estimated ~2.5 ms per step) does not justify a routing change of that size. Evidence:
`artifacts/s12/router_bf16_check.log`.

**3 bucket widths (row mode).** 3 widths give 6 ranges per DSA layer. The decode capture took 95.6 MiB (+11.7 MiB
over the capture without DSA points). With the sampler traces it overflowed the 96 MiB p300x2 trace region
(`logs/s12/dec_ab1.log`). 2 widths (3 ranges, +4.2 MiB) fit. `tests/test_dsa_buckets_host.py::
test_row_dsa_buckets_fit_trace_region` checks every multi-slot profile against its trace region with these measured
costs (`tt/dsa_buckets.py` `row_dsa_trace_estimate`, `serving_profiles.json` `trace_budget`).

## Not started (larger items)

| Item | Estimate | Effort |
| --- | --- | --- |
| Per-width whole-stack decode traces (Qwen `decode_buckets`): run KDA / DSA / mHC / router / shared expert on the active prefix only | 1 user: ~52-55 ms per step (from 74.8 ms shipped) | 3-5 days; trace region per width must be measured first |
| Single-read MoE for 8-16 rows that is exact vs the pairs path (each active expert read once per step) | 16 users: ~77 -> ~120 tok/s | 3-5 days; must stay bit-exact for served batch invariance |

These estimates come from the gap report (made before stage 12), not from a measurement. The 1-user estimate
included the full-width DSA selection. The opt-in row buckets remove it at short positions; with them on, the
remaining saving can be smaller.

Publish (2026-10-24): the r6 bundle (image `8770c660cc9f`, built from `a6ceb44bd5b`; later commits change comments,
docs and the model card only) is on the Hub at `raahemnabeel/glm-5.3-flash-e192-blackhole` revision `301499e6`
(public, catalog; 876 files, no LFS mismatch). Hub verify on a freshly reset p300x2 (`tt-model pull` + `serve`,
`~/glm53-flash-bringup/logs/11/verify-hub-p300x2-r6.chain.log`): healthy after 233 s, burst 16 of 16 identical to
solo greedy, 262,080-token prompt answered in 219 s, VERIFY_RESULT PASS. The image was already on the host, so the
pull checked the Hub files and did not download the image again.

Release accuracy gate on the shipped code (2026-10-24, snapshot of 8a51274702a, runtime code equal to the published
image; shipped defaults, no `GLM53_*` overrides; 1 slot x 243,712, the shape of `p300x2-long`; `tf_release/`):

| Reference | Teacher forcing top-1 / top-5 / top-100 | Prefill check top-1 / top-5 / top-100 |
| --- | --- | --- |
| p0 | 0.92 / 1.00 / 1.00 | 0.92 / 1.00 / 1.00 |
| p1 | 0.95 / 1.00 / 1.00 | 0.95 / 1.00 / 1.00 |
| p2 | 0.92 / 1.00 / 1.00 | 0.91 / 1.00 / 1.00 |

PASS (stage-8 gate: min top-1 >= 0.91, top-5 and top-100 = 1.00). The values equal the stage-8 and stage-10 runs.
The record's `trace_verified` is false because the check expects 2 replays per step, but the decode trace replays
in segments (DSA and MoE point traces). The replay count (2,376 for 99 steps) equals the stage-10 run.

## One user on a multi-slot build (commit 685db8afb95)

Problem: the default profile `p300x2` (16 slots) decoded one user at 74.8 ms per token (served), the 1-slot build
`p300x2-long` at 49.4 ms. The decode trace runs all 16 slot rows. Stage 10 already moved the routed experts to row
buckets. A Tracy profile of the shipped code (reduced model, layers 0, 3, 4; 1-slot build vs 16-slot build with 1
active slot; `~/glm53-flash-bringup/artifacts/s12/tracy_s{1,16}`) showed the rest of the gap (+1.19 ms per 3 layers):
- MoE 1-row bucket: the pairs path ran its two grouped sparse matmuls on 8 cores (141 + 104 us). The 1-slot build runs
  the T = 1 gather path on 32 / 64 cores (86 + 40 us).
- KDA recurrence (`kda.recurrent_decode`): 173 us at 16 slots vs 60 us at 1 slot. The op computes every (slot, head):
  fp32 state read, step, state write.
- KDA conv (`kda_decode_conv`): 93 us vs 40 us. The reader copied the four fp32 tap rows into every slot row with
  scalar L1 writes, per tile.

Changes (all bit-identical for the active rows):
1. **MoE**: in the pairs family, the 1-row bucket takes the T = 1 gather path (`moe_bucket_paths`, "auto").
   Reduced model, 1 and 2 active: 19 / 19 rows bit-identical to the pairs bucket (`one_user/moe_t1_compare.json`).
2. **KDA recurrence**: new optional `active` input (one UINT32 flag per slot, row-major, written by the host only when
   slots start or stop). The reader lists the active slots' heads and deals them over all cores (active head j on
   core j % cores). It passes each core's count to the compute kernel through a CB. The writer writes the active
   heads' state and output, and zero output rows for the inactive heads. An inactive slot's state is not read or
   written. Switch: `GLM53_KDA_ACTIVE` (default 1; 0 = every slot computed). Op tests:
   `tests/test_kda_recurrent_decode.py::test_kda_recurrent_decode_active*` (bit-identical active rows, inactive
   state unchanged, flag changes under a captured trace).
3. **KDA conv**: the reader copies the taps into the active rows only. The compute kernel already selects the
   history and a zero output for inactive rows, so their taps are never used.

| Measurement | Before | After |
| --- | ---: | ---: |
| Full model token-out, 16 slots x 243,712, 1 active (`one_user/ship_vs_new.json`, `ab1cd_vs_ship.json`) | 74.6 ms | 61.6 ms |
| Full model token-out, 16 active | 176.8 ms | 176.8 ms |
| Served TPOT, p300x2, 128 / 128, 1 user (`one_user/sweep_kact.json`) | 74.8 ms | 61.8 ms |
| Served TPOT, p300x2, 128 / 128, 16 users | 177.0 ms | 177.0 ms |
| Decode trace region (served capture) | 83.9 MiB | 81.3 MiB |

Correctness:
- Full model, 16 slots, 1 and 16 active: logits bit-identical to the shipped code, 29 / 29 rows. The conv change alone
  (stage-12 `cd` record vs `ship`): 29 / 29 bit-identical.
- Batch invariance (standalone, `one_user/standalone_invariance.json`): a target alone vs next to a 70K filler, 1 and
  2 active: identical, max |d| 0.
- Served invariance, p300x2 (`one_user/served_inv_kact.json`): solo repeat, concurrent with a long request, and solo
  again are equal (text and random-id prompts). PASS.
- Release accuracy gate (1 slot x 243,712, `one_user/tf_release_record.json`): top-1 0.92 / 0.95 / 0.92 (teacher
  forcing), 0.92 / 0.95 / 0.91 (prefill check), top-5 and top-100 1.00. PASS, equal to the shipped code.

Notes:
- `GLM53_MOE_BUCKET_PATH` with a per-bucket list (for example `8:sorted`) gives unnamed buckets the family, so the
  1-row bucket goes back to pairs. Both paths are bit-identical; only the speed changes.
- `tt/moe.py` has an inherited Qwen3.5 comment: the width-1 gather path was intermittently non-deterministic on the
  Qwen 48-layer model. It does not apply here: the 1-slot GLM profiles ship on this path, and on p300x2 the served
  solo repeats and the concurrent runs are equal (`one_user/served_inv_kact.json`).
- The recurrence op's reader and writer each read the slot flags. They agree because the host writes the flags only
  between programs on the one command queue. A flag write that overlaps a running program (for example from a
  second command queue) could make their head counts differ and stop the op.
- Inactive slots' KDA output rows are now zeros (before: q . S of the unchanged state). Every later op works per row
  or skips inactive rows, so this does not reach active rows (full-model logits bit-identical).

Release package r7 (image `9b84677f4961`, built from `685db8afb95`; `scorecard_r7/`, 2026-10-25; logs
`~/glm53-flash-bringup/logs/s12/chain_r7.log`, `logs/11/score-*`, `logs/11/verify-staged-p300x2{,-1m}-r7.chain.log`):

| Profile, row | r6 (shipped) | r7 |
| --- | ---: | ---: |
| p300x2, 128 / 128, 1 user, TPOT | 74.80 ms | 61.85 ms |
| p300x2, 8,192 / 256, 1 user, TPOT | 78.56 ms | 65.60 ms |
| p300x2, 131,072 / 256, 1 user, TPOT | 79.32 ms | 66.38 ms |
| p300x2, 128 / 128, 16 users, TPOT | 177.01 ms | 177.01 ms |
| p300x2, 1024 / 256, 16 users, TPOT | 178.83 ms | 178.83 ms |
| p300x2, CI 100 / 100, 32 concurrent, mean TPOT | 177.0 ms | 176.9 ms |
| p300x2-wide, 128 / 128, 1 user, TPOT | 124.12 ms | 117.49 ms |
| p300x2-wide, 128 / 128, 32 users, TPOT | 165.05 ms | 165.05 ms |
| p300x2-wide, 1024 / 256, 32 users, TPOT | 165.6 ms | 165.5 ms |
| p300x2-long, 128 / 128, 1 user, TPOT | 49.35 ms | 49.45 ms |

Checks on r7, all on the packaged bundle (`tt-model serve`):
- Staged verify, p300x2 (fresh reset): PASS; burst 16 / 16 identical to solo greedy
  (`scorecard_r7/verify_staged_p300x2_smoke.json`).
- p300x2: preemption load test t2 PASS (`preempt_t2_p300x2.json`); CI burst 64 / 64 successful.
- p300x2-wide (B = 32): smoke PASS with the 159,680-token prompt, burst 16 / 16 identical to solo greedy; served
  batch invariance PASS (`invariance_p300x2-wide.json`).
- p300x2-long: smoke PASS with the 344,000-token prompt (`smoke_p300x2-long.json`).
- p300x2-1m: every row equal to the r6 run (same text). The burst row is 0 / 1, the documented block-placement
  exception of stage 11 (`verify_staged_p300x2-1m_smoke.json`, r6: `logs/11/staged-p300x2-1m-r6-smoke.log`).
- Op tests (`tests/test_kda_recurrent_decode.py`, `test_kda_decode_conv*.py`, incl. H = 16 at B = 2 and 32):
  25 passed (`~/glm53-flash-bringup/logs/s12/op_tests2.log`).

### Wide profile: tight sorted-dispatch bound (commit 613e4802772)

Problem: after the changes above, `p300x2-wide` (32 slots) still decoded one user at 117.5 ms (served). At 32 slots
every MoE row bucket runs the expert-sorted family (stage 10, batch invariance; see doc/optimized_vllm/README.md, "One
routed-path family per build"). A Tracy profile (reduced model, 32-slot build, 1 active, vs the 1-slot build;
`~/glm53-flash-bringup/artifacts/s12/tracy_kact5_s32`) gave +3.2 ms per 3 layers, nearly all in the sorted MoE path of
the 2 MoE layers: gather rows 595 us, untilize 548 us, the two grouped sparse matmuls 527 + 391 us, elementwise ops,
gather-add, dispatch. These ops scale with the dispatched row count. `dispatch_rows` sized the per-expert padding for
all E = 192 experts: 6,016 rows (188 groups) for the 8 real rows of one token.

Change: a call with T x k rows uses at most min(E, T x k) experts, so the bound is now
roundup(T x k, 32) + roundup(31 x min(E, T x k), 32) + 32 (`GLM53_MOE_TIGHT_DISPATCH`, default 1; 0 = the E bound).
The dispatch op's validation accepts this bound. The row layout does not change (experts ascending, each padded to
whole groups, then the trailing padding group), and every row is computed alone, so the results are bit-identical.
Dispatched rows at 1 / 2 / 4 / 8 / 16 rows: 320 / 576 / 1,056 / 2,080 / 4,128 (before about 6,000). When T x k >= E
(the 32-row bucket, every prefill chunk) the bound does not change. Host test: `tests/test_moe_dispatch_host.py`
(random and worst-case routings fit; unchanged bound at full width).

| Full model, 32 slots x 159,744, 160 MiB trace region (`one_user/wide_tight_dispatch.json`) | Before | After |
| --- | ---: | ---: |
| Token-out, 1 active | 117.7 ms | 77.4 ms |
| Token-out, 2 active | 117.9 ms | 80.5 ms |
| Token-out, 8 active | 128.3 ms | 108.8 ms |
| Token-out, 32 active | 157.7 ms | 157.6 ms |
| Decode trace region | 100.8 MiB | 90.3 MiB |

Correctness: logits bit-identical, 67 / 67 rows (1, 2, 8 and 32 active). The `p300x2` (16 slots, pairs family),
`p300x2-long` and `p300x2-1m` decode paths do not use the sorted path at T x k < E, so they do not change.

Release package r8 (image `cfa09cc4402d`, built from `613e4802772`; `scorecard_r8/`, 2026-10-25; logs
`~/glm53-flash-bringup/logs/s12/chain_r8.log`, `logs/s12/scorecard-r8.chain.log`, `logs/11/verify-staged-p300x2-r8.chain.log`):
- p300x2-wide served TPOT, 128 / 128: 1 user 117.49 -> 77.28 ms (8.5 -> 12.9 tokens/s); 32 users 165.05 -> 165.01 ms;
  1024 / 256 at 32 users 165.5 -> 165.52 ms.
- p300x2-wide smoke PASS (159,680-token prompt; same text as r7), burst 16 / 16 identical to solo greedy; served batch
  invariance PASS (`scorecard_r8/invariance_p300x2-wide.json`).
- Staged verify, p300x2 (fresh reset): PASS, burst 16 / 16 (`scorecard_r8/verify_staged_p300x2_smoke.json`).

The 1-slot profiles (`p300x2-long`, `p300x2-1m`) do not get faster: their decode already runs one slot. The remaining
gap of the 16-slot build at 1 user (61.8 vs 49.4 ms) is mostly the full-width DSA selection (the opt-in per-row
buckets remove about 5 ms) and width-16 relayouts in every layer (about 15-25 us each).
