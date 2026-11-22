# MiniMax-M2.7 hybrid: where the time goes and what to build next

Status 2026-11-18, v1.11: image d5585647f6c2 from 94fdaf7b131, on HF as raahemnabeel/minimax-m2.7-hybrid-qb2.
Measurements and run directories are in `HANDOFF.md` under the dated entries this doc cites. Keep this doc current
when an item lands or a number changes.

## 1. Bottleneck today (single user, 128K profile)

One speculative pass at short context is ~30 ms and yields ~2.3 tokens; served greedy decode is ~83 tok/s.

| part | ms per pass | limited by |
| --- | --- | --- |
| drafter | 4.5-4.8 | device |
| verify, device compute | ~16.5 | device DRAM bandwidth: fused experts at 400-440 of ~490 GB/s per chip |
| CPU experts (15-20 per pass) | ~5 | host DRAM bandwidth: the int4 kernel reads ~40 GB/s of a 45-54 GB/s ceiling (DDR5 at 3600 MT/s) |
| host round trips (18 hybrid layers) | ~5 | PCIe latency: ~22 us reads, ~40-50 us pinned writes per hybrid layer |

The host round-trip figure is inferred, not timed directly: the profiled host wait minus the HYB_SKIP=1 A/B (CPU
experts off: verify 22-23 vs 27-29 ms).

PCIe bandwidth is not a limit. Copying more experts was flat or slower in every sweep, because the copies read the
same host RAM the CPU kernel needs. Host CPU compute is not a limit either: the kernel is no faster when its weights sit
in L3, and going from 6 to 8 threads gains about 7%.

At 44.7K tokens the pass is ~47 ms and yields 1.9 tokens:
- attention 12.5 ms;
- other device work ~11 ms;
- host wait ~16 ms. Elastic context leaves 22 hybrid layers instead of 18, with ~36 CPU experts per pass.

Sources: HANDOFF "DEVICE PROFILE v1.7", "MOE EXPERT BANDWIDTH", "LONG-CONTEXT DECODE", "Host findings 2026-11-08",
and the 2026-11-08 quality entry for the HYB_SKIP A/B.

## 2. Long-context attention (next single-user lever)

**Problem.** The verify pass's SDPA is the shared-KV rows op (qwen36 `attn_decode/rows_sdpa.py`, on from 4096 tokens
via QWEN35_ROWS_SDPA_SHARED_FROM). At 44K it costs 126 us per layer, against 22 us at short context. It reads ~24 MB
of KV per layer per die, but only at ~230 GB/s, against the chip's ~490 GB/s.

**Per-call cost**, one chip, 64-128-token chunks:
- ~400 GB/s marginal from 16K to 64K tokens;
- plus ~50-60 us fixed.

**Per core**, the flash-decode loop costs ~10 us of compute (QK, softmax, PV) per 128-token chunk, with the reader at
~3.5 GB/s per core. A six-round tree reduction over 55 cores adds ~30 us.

**Size of the prize.** At ~490 GB/s the SDPA would be ~50 us per layer at 44K: about -4.8 ms per pass, ~10% of decode
at 45K. At 85K it is about -10 ms (~15%), and it grows with context. The prefix-cache resend path gains the same.

**Approaches**, cheapest first:
1. Cheaper cross-core reduction: fewer rounds, or a two-level reduce. Estimated ~3-4% at 44K; not pursued on
   2026-11-16.
2. More bytes in flight per core: double-buffer the reader, larger KV chunks per core, or more cores per bank, so the
   compute loop is no longer the pacing step. This is a kernel change in `attn_decode/kernels`.
3. Cheaper per-chunk math: approximate exp or lower fidelity in QK / PV. This needs the quality gate: forced log-probs
   vs ref_fb2 (`cache/cmp_ref.py`) and the long-text top-1 / KL check (HANDOFF entry for the shared-KV rows op).
4. Fewer KV bytes: bfp4 KV, or K in bfp8 and V in bfp4. Quality-gated, and it also frees KV memory, which keeps more
   layers full at long context (fewer hybrid layers, fewer CPU experts).

**How to measure.**
- Device profile of the verify pass at 44K and 85K, as in HANDOFF "LONG-CONTEXT DECODE" (scratch p19).
- The served card's 44.7K and 85K rows (`serve/card_bench.py`).
- Bit-exactness is not expected for approaches 3 and 4; approaches 1 and 2 should be exact.

## 3. Multi-user serving (batching)

Today: `--max-num-seqs 1`, and the DFlash step is single-user (`dflash_serve.py` raises on any slot but 0).

### Stage 1 measurement (2026-11-18, HANDOFF "MULTI-USER COST STUDY")

Verify-pass cost vs rows. The rows are a user's drafts, or foreign tokens injected with DFLASH_BENCH_FOREIGN.

| verify pass | path | ms | distinct experts / hybrid layer | CPU experts / pass |
| --- | --- | --- | --- | --- |
| 4 rows, one user | serving (rows ops, fused boundary) | 24.5 | 21.5 | 17 |
| 4 rows, one user | stock attention, unfused | 32.2 | 21.5 | 17 |
| 8 rows, one user's drafts | stock, unfused | 43.1 | 32.1 | 30 |
| 8 rows, 7 foreign tokens | stock, unfused | 43.3 | 37.3 | 37 |
| 8 rows, 2 texts x 4 | stock, unfused | 46.2 | 36.6 | 47 |

Foreign tokens placed in one shared context raise the union far less than truly independent users would: 37 vs
~56 at 8 rows. Routing follows the context, so these costs are a floor.

### Expert union and misses by configuration

From routing data (`runs/m27/ref_held/routing.npz`) and the serving residency plan (159 static experts per hybrid
layer). "Static misses" are before the LRU refills, which today catch ~80% for one user.

| users x rows | rows | distinct experts / layer | static misses / pass |
| --- | --- | --- | --- |
| 1 x 4 (today) | 4 | 24 | 92 |
| 2 x 4 | 8 | 45 | 183 |
| 4 x 4 | 16 | 81 | 342 |
| 8 x 4 | 32 | 134 | 608 |
| 4 x 1 | 4 | 30 | 115 |
| 8 x 1 | 8 | 56 | 224 |

### Speculation and batching

One user's 4 consecutive tokens share most experts, so a 4-row verify costs far less than 4 passes; speculation
spends that spare capacity on tokens. Independent users share few experts, so every extra row brings new expert reads
and new CPU misses. Two consequences:
- **The CPU wall.** 8 x 4 means 110+ CPU experts per pass even at today's refill rate, 25+ ms of host-RAM-bound
  kernel time.
- **Rows stop being free.** A spec row yields ~0.6 tokens (2.4 per 4 rows); a plain row yields 1. Once rows cost real
  work, plain rows win.

Estimates from the measured per-expert and per-row costs (not measured end to end):

| setup | total tok/s | per user |
| --- | --- | --- |
| 1 user, spec (today) | 83 | 83 |
| 2 users x spec | ~90-115 | ~45-55 |
| 4 users x spec | ~100-120 | ~25-30 |
| 4 users, no spec | ~120-150 | ~30-38 |
| 8 users x spec | ~115-135 | ~15-17 |
| 8 users, no spec | ~145-180 | ~20 |

**Policy.** Use spec for 1-2 active requests and plain batched rows above that. The biggest unknown is the attention
cost per user: the estimates assume one attention call per user per layer, and an op that batches users would lower
every multi-user row.

### Blockers found

- **8 rows fail in the fused fabric boundary.** DFLASH_FB=1, `fabric_boundary/ops.py`: TT_THROW in the circular
  buffer config.
- **8 rows fail in `rows_sdpa_decode`.** Rows x q heads per kv head must fit one tile, and 8 x 6 = 48 > 32. M2.7's
  limit is 5 rows. Multi-user verify needs per-user attention calls in any case, each user's rows against their own
  KV and page table.

### Work list (estimates)

1. **No-spec batching, about 1-2 weeks.** Only when 3+ requests are active; one row per user through the stock batched
   paged decode attention (already in qwen36), with HYB_DECODE_ROWS = users and per-row sampling. Admission: the
   fixed 64K layout, or the elastic short layout while the total context fits; refuse a switch while other users are
   active.
2. **Multi-user DFlash for 2 users, about 1-2 weeks.**
   - per-slot drafter state (window, taps, refills);
   - per-user verify attention;
   - an 8-row fused boundary;
   - per-user acceptance and sampling.
3. **Prefill interleaving, about 1 week.** Chunked prefill alternating with decode steps. A 45K prompt is ~20 s of
   prefill, which would otherwise stall every other user.
4. **Per-user refill budget.** A shared LRU thrashes between users, so give each user part of the per-pass budget.

**Decide first** whether the box sees concurrent load: several people, or an agent issuing parallel requests. For a
single person at the desk, batching makes each request slower.

## 4. Drafter (DFlash) acceptance

nvidia/MiniMax-M2.7-DFlash, at 3 drafts per pass:

| source | prompts | tokens per pass |
| --- | --- | --- |
| NVIDIA's card | MT-Bench | 3.05 |
| this server | MT-Bench, 80 first turns | 2.47 |
| CPU oracle, bf16 drafter on BF16 target taps | 4 MT-Bench answers x 1024 tokens | 2.42-2.45 |

The device path loses nothing against the exact reference. NVIDIA's number does not reproduce outside their stack.
Their target was vLLM FP8 w8a8, trained at 4096 tokens with chat_template_train.jinja.

Checked equal to ours:
- tap layers;
- mask token;
- block size;
- the module math (Model-Optimizer `modeling_dflash.py` vs `dflash_ref.py`).

Levers left:
- **More drafts:** k 3 -> 5 gives +6.5% AL, but verify cost grows (DK=4 was a loss before), and the rows SDPA limit is
  5 rows.
- **Draft trees:** need ~8 verify rows, which brings the blockers above.
- **Fine-tuning the drafter** on this setup's target hidden states. Needs NVIDIA GPUs and a license check (NVIDIA
  Software and Model Evaluation License). Realistic gain +10-20% decode, unproven.

## 5. Smaller items

- **Host DRAM speed.** DDR5 at 3600 MT/s, rated 5600. A BIOS change would speed the CPU experts directly; BIOS changes
  are excluded on this box.
- **Pinned row writes** cost ~40-50 us inside the decode loop vs ~10 us idle (cause not found; HANDOFF host findings).
  At 18-42 hybrid layers per pass, that is the PCIe-latency share above.
- **Router matmul** (~11 us per full layer): qwen36's boundary consumer fusion is one-row only, and M2.7 needs FP32
  logits.
- **Sampled decode** is ~20% slower than greedy at short context: candidate traces (~710 us each on one die).
  `ttnn.topk` past 32768 segfaults on mesh tensors; one 65536-wide chunk per die would halve it.
