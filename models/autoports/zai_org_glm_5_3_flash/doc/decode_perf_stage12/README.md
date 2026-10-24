# GLM-5.3-Flash (E192) decode performance (stage 12)

Stage 12 makes three changes to the multi-slot decode step. Changes 1 and 2 are on by default and give bit-identical
outputs. Change 3 (per-row DSA buckets) is opt-in: the sparse-regime gate showed that it changes outputs (see
"Sparse-regime gate"). This page gives the
defaults, the switches, the measured numbers and their scope, and the items that stage 12 rejected or did not start.

Commits: d86fe3c874d (MoE sentinel, persistent all-gather), ad427402ba2 (per-row DSA decode buckets).
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

Correctness (dense regime, positions about 300; for the sparse regime see the gate below):
- base vs cd and base vs all: bit-identical logits, top-1 29 / 29 rows (1 and 16 active). These positions are below
  `DENSE_MAX_POSITION` (2,050), where DSA reads every pool, so this check does not test the selection.
- Served invariance (p300x2): identical tokens solo, next to a long request, and solo again (`solo_repeat_equal`,
  `concurrent_equal`, `long_still_decoding`: true). Commit ad427402ba2 also reports a standalone check: a target
  request alone vs next to a 70K filler was bit-identical, with the mixed bucket range included.

## Measured numbers

Standalone token-out (16-slot build, 243,712, shared 1,024-block pool):

| Active rows | Stage 11 | cd (sentinel + ag_persistent) | all (cd + row buckets) |
| ---: | ---: | ---: | ---: |
| 1 | 75.0 ms | 74.6 ms | 69.9 ms |
| 16 | 177.8 ms | 176.8 ms | 172.1 ms |
| Trace region | 86.6 MiB | 83.9 MiB | 88.1 MiB |

Served TPOT, p300x2, greedy, same host load (stage 12 = with the opt-in row buckets on; the shipped default is the cd
column above, about 0.5 % faster than stage 11; the release scorecard measures it):

| Workload (input / output, users) | Stage 11 | Stage 12 |
| --- | ---: | ---: |
| 128 / 128, 1 user | 75.32 ms | 69.72 ms |
| 128 / 128, 16 users | 178.03 ms | 172.03 ms |
| 1024 / 256, 16 users | 180.72 ms | 173.93 ms |

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

Script `scripts/s12/gate.sh` on the committed code (snapshot of ad427402ba2), 16 slots, 243,712, 1 and 16 active,
6 greedy decode steps after the prompt. Evidence: `artifacts/s12/gate_p{2100,4000}/base_vs_{cd,all}.json`,
`artifacts/s12/gate_tf1_{off,on}.json`, `logs/s12/gate.chain.log`.

| Check | Result |
| --- | --- |
| base vs cd (sentinel + ag_persistent), prompt 2,100 | bit-identical, top-1 29 / 29 |
| base vs cd, prompt 4,000 | bit-identical, top-1 29 / 29 |
| base vs all (+ row buckets), prompt 2,100 | **not identical**, top-1 24 / 29, max \|d\| 0.9 .. 7.9 logits |
| base vs all, prompt 4,000 | **not identical**, top-1 23 / 29, max \|d\| 0.3 .. 11.2 logits |
| 1 slot (p300x2-long / -1m shape), ag_persistent off vs on, 966 teacher-forced rows | bit-identical |
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
| Per-width whole-stack decode traces (Qwen `decode_buckets`): run KDA / DSA / mHC / router / shared expert on the active prefix only | 1 user: ~52-55 ms per step (from 69.9 ms) | 3-5 days; trace region per width must be measured first |
| Single-read MoE for 8-16 rows that is exact vs the pairs path (each active expert read once per step) | 16 users: ~77 -> ~120 tok/s | 3-5 days; must stay bit-exact for served batch invariance |

These estimates come from the gap report (made before stage 12), not from a measurement. The 1-user estimate
included the full-width DSA selection, which stage 12 already removed at short positions; the remaining saving can
be smaller.
