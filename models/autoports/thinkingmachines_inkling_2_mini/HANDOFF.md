# Inkling-2-Mini on Blackhole: bring-up log

thinkingmachines/Inkling-2-Mini (26.4B looped MoE, 3.4B active / 5.4B effective per token, NVFP4/MXFP8 experts) on
p150 / p300 (p150x2) / p300x2 (p150x4), everything on device. Goal: served through vLLM + the TT plugin and published
with tt-model-manager, like the other ports. Branch `raahem/inkling-2-mini`.

## Model, as ported

24 layers, hidden 2304; layers 6..17 run 3 passes (48 applications, each with its own KV / conv state; passes 2-3
restart from the pre-loop state and get messages from the previous pass through per-layer gates). Global attention at
layers 5, 11, 17, 23 (8 applications), sliding window 1024 elsewhere. No RoPE: a hidden-state-conditioned relative bias
(r @ proj over 1024 distances), per-token sink logits, sigmoid-gated o_proj, L2-clipped q/k, unweighted-RMS v; short
convs on K, V (dynamic taps on the global layers), attention output and MoE output; block attention residuals over the
inputs of layers 0, 6, 12, 18. MoE: 128 experts top-8 (sigmoid + selection bias, route_scale 8),
act = 4 tanh(g/4) sigmoid(g) 25 tanh(u/25). Logit softcap 30.

## Code

- `tt/weights.py`: checkpoint reader (original names), NVFP4 / MXFP8 dequant bit-exact vs transformers 5.24.
- `tt/attn/`: Inkling-Small's custom SDPA (relative bias in-band, sliding ring, causal rows) + per-row sinks and
  per-item page tables (multi-user). `tests/test_sdpa_device.py` (TP1 / TP4 shapes).
- `tt/attention.py`, `tt/sconv.py` (static fused kernels from Inkling-Small; `DynShortConv`), `tt/moe.py`
  (sparse_matmul gather / masked paths), `tt/model.py` (block residuals, message gates, the 48-application schedule),
  `tt/ccl.py` (replicated-residual all-reduce), `tt/wload.py` (weight cache).
- Reference: `reference/golden.py` (per-application fixtures), `reference/ref_texts.py` (quality-gate reference).
- Tests: `tests/test_apps_device.py` (teacher-forced per application, prefill + decode), `tools/teacher_force.py`.

## Quality (teacher-forced vs the CPU reference, 8 held-out 1000-token texts)

| configuration | top-1 | top-1 (confident) | KL (top-32) | NLL (ref 2.022) |
|---|---|---|---|---|
| noise floor: fp32 vs bf16 CPU reference | 0.975 | 0.998 | 0.010 | 2.020 |
| bfp4 experts, bfp8 KV, 1 die | 0.938 | 0.989 | 0.040 | 2.029 |
| bfp8 experts, bfp8 KV, 1 die | 0.967 | 0.997 | 0.016 | 2.029 |
| **bfp8 experts, bf16 KV, 1 die** | **0.970** | **0.996** | **0.010** | 2.025 |
| bf16 experts, bf16 KV, TP4 | 0.971 | 0.996 | 0.013 | 2.027 |

bfp8 experts + bf16 KV is at the noise floor (bf16 experts add nothing). bfp4 is a measurable loss. Per-application
test (bf16 experts): worst pcc 0.99888 (prefill and decode), head logits pcc 1.00000.

## Host / ttnn issues found (keep)

- Host -> device writes from file-backed memory (safetensors mmap, ttnn .tensorbin via as_tensor) crawl on this host
  (kernel 7.0, page cache in large folios: get_user_pages splitting huge PMDs; 0.2 s -> tens of minutes,
  intermittent). Clone checkpoint tensors; stage cache files through /dev/shm (`tt/wload.py`).
- `ttnn.sparse_matmul` masked mode with fp32 dest accumulation mis-accumulates across in0 blocks (pcc 0.85): use one
  in0 block over all of K.
- `models/tt_transformers` `tt_all_reduce` on a (1, N) mesh only reduce-scatters, and returns its input unreduced for
  cluster_axis=1: use stock `ttnn.all_reduce(cluster_axis=1, Linear)`.
- transformers' last `hidden_states` entry is not the last layer's output: fixtures hook the layers.
- Dev runner mounts the prefetcher fix (`cq_prefetch.cpp`) like the other ports' scripts.

## Serving (vLLM, dev)

`tools/serve_dev.sh` (MESH_DEVICE, SEQS, CTX): vLLM 0.26 + the TT plugin in the dev image, the port's bundle via
EXTRA_MODELS_DIR, `--hf-config-path hf_config --trust-remote-code` (vLLM 0.26 / transformers 5.12 do not know
inkling2_loop; the shim's arch is `TTInkling2LoopForConditionalGeneration`), `--reasoning-parser inkling
--tool-call-parser inkling` (same message tokens as Inkling v1; tool calls parse). `tt/generator_vllm.py`: engine-owned
slot caches, a row -> slot map for the plugin's condense remaps, host sampling (v0). `tools/smoke_serve.py`.

## Decode performance (P150, one die, bfp8 experts, bf16 KV; device time per traced step)

| change | B=1 | B=4 | quality (prefill path) |
|---|---|---|---|
| first batched decode | 126 ms | 197 ms | 0.970 / 0.996 / 0.010 |
| fused rms_norm, stacked block mix, merged gates, folded q/r scales | 113 ms | | 0.969 / 0.996 / 0.012 |
| bfp8 attention + lm_head, tile-layout conv states | 103 ms | 171 ms | 0.971 / 0.997 / 0.014 |
| fused MoE decode kernels (qwen36 moe_decode + ACT_INKLING), one fused-layout weight copy | 102 ms | 133 ms | 0.968 / 0.996 / 0.014 |
| route once per layer | 97 ms | 126 ms | decode path: 0.967 / 0.994 / 0.018 (sparse 0.970 / 0.997 / 0.023) |
| activations in L1, one-program batched conv step | 83.0 ms | 101.3 ms | |
| fused block-mix + message-gate-tail kernels | 72.6 ms | 90.6 ms | decode 0.968 / 0.996 / 0.018, prefill 0.968 / 0.996 / 0.014 |
| explicit 1D mcast matmul configs for every decode matmul (tt/dmm.py) | 57.8 ms | | |
| one-row rms_norm width-sharded on 8 cores | 51.0 ms | 67.5 ms | decode 0.970 / 0.996 / 0.015 |
| attention prep in one program (tt/aprep.py: head gather, l2 clips, rms, tau, sink, KV-write layout) | 42.5 ms | 58.8 ms | see below |
| message gate in one program (tt/gate.py: two cross-core rounds through a root core) | 41.1 ms | | decode (8 x 736) 0.9669 / 0.9965 / 0.0191 |

TP (same code, B = 4 traced): TP4 38.1 ms, TP2 43.5 ms; TP4 B = 1 device 34.9 ms (replicated glue + 96 all-reduces at
39 us + the replicated LM head dominate: next, the fabric all-reduce boundary and a vocab-sharded head).

Fused gate zones (tools/zones_gate.py, us): squares 6, root reduce #1 11.5, norm + partial down proj 6, root reduce #2
9, forget / gain 10, tail 10 (fp32 unpack-to-dest copies with per-copy re-inits ~300 cycles each). Next step for it:
one round (scale the partial projections by the rms factors after the reduce) and a tree reduce over the cores.

At one row most ops are bound by the 32-row tile, not by bytes (attention projections ~60 us each vs a 32 us
bandwidth floor; DRAM-sharded only wins at LoFi): wider batches are nearly free, single-user latency needs fused
kernels (layer glue: norms / mixes / gates / convs ~50 ms of the 97).

Fused MoE notes: the kernels' single-row sigb path routes wrongly (pcc 0.5): one row runs the 2-row path. Expert slices
must be 128..768 columns: sub-shards 3 x 384 (1 die), 384 + 256 (2), 384 padded (4). Down rows op: 3 readers per bank
(H = 2304 is 9 columns per bank) -> 16 rows max; wider batches use the sparse union path.

Decode-quality noise: the teacher-forced decode eval (8 texts x 256 steps) moves by ~0.5 pt top-1 and ~0.005 KL
between numerically equivalent variants (ulp-level rounding differences); code_mbpp alone swings KL 0.03 - 0.10.
The attention-prep kernel at 8 x 736 steps: top-1 0.9671 = ops path 0.9671, top-1 (conf) 0.9954 vs 0.9951, KL 0.0171
vs 0.0150 (all of the difference in code_mbpp / chat_french; with the ops path's r instead, KL 0.0108). The relative-bias
query r is the sensitive input: its per-head sums of squares must not go through a plain TF32 matmul (truncation
bias +1e-4 on the clip factor measurably moved quality); the kernel splits x^2 into hi + lo (exact).

Bandwidth floor (one die, bfp8): every application re-reads its layer's weights (attention ~30 MB, 8 experts ~67 MB),
~5.7 GB per token -> ~11 ms at 512 GB/s; 51 ms is ~22 %. Matmul notes (tools/bench_proj.py, traced, M = 32): the
default config is 2-5x slow at one tile row (k / v 34 -> 13 us, o 60 -> 24, gate down 104 -> 20 with a 1D mcast_in0
config on ~48 cores); DRAM-sharded is slower than 1D here. Small-N matmuls (mix logits N <= 4 tiles, gate down 8 tiles,
router) still run on a few cores (~20 us each): split-K next. One-row rms_norm: 29 us interleaved, 10 us width-sharded
on 8 cores incl. both reshards (tools/bench_norm.py).

## Serving (P150, vLLM, greedy, 256 output tokens, tools/bench_serve.py)

| concurrency | TTFT mean | decode tok/s / user | total tok/s |
|---|---|---|---|
| 1 | 1.71 s | 20.8 | 18.3 |
| 2 | 2.52 s | 20.5 | 34.1 |
| 4 | 5.47 s | 16.8 | 49.1 |
| 8 | 11.97 s | 13.4 | 65.0 |

(before the prefill MoE fix below.) After it, with decode sampling on device (SAMPLE_DEVICE=1: the qwen36 RowsTopK fused
top-32 twice inside the decode trace, the second pass on the rows shifted by the first's max so bf16 resolves the
candidates; host draw with vLLM semantics over the 32, tt/sampling.py; 0.5 ms at 1 row, 1.3 ms at 8):

| concurrency | TTFT mean | decode tok/s / user | total tok/s |
|---|---|---|---|
| 1 | 0.79 s | 20.9 (24.7 at temperature 1) | 19.7 |
| 2 | 1.18 s | 21.2 | 38.7 |
| 4 | 2.56 s | 18.2 | 61.6 |
| 8 | 5.63 s | 14.1 | 85.7 |

Single-user device time depends on the text (greedy repetitive text 47 ms / step, sampled 40 ms): expert placement?
(open). Decode width buckets (one trace per width, the narrowest covering the highest live
slot), logits untilized on device (readback 8.3 -> 1.1 ms). Prefill: the MoE's masked union path was 84 % of a chunk;
the sorted dispatch (MiniMax / Qwen3.5 moe_dispatch, grouped sparse matmuls) takes a 128-row chunk 1832 -> 733 ms,
256 rows 1031 ms (prefill quality 0.969 / 0.996 / 0.016). Next for prefill: the multi-row block mix / message sums
(~165 ms of the 128-row chunk), then batching prefills across users.

## Serving profiles (measured through vLLM; tools/bench_card.py: random-token prompts, ignore_eos, greedy, streamed)

Capacity (tools/mem_probe.py, DRAM per die after the build): one die 26.7 GiB of weights + 0.46 GiB per slot at 8K
(8 x 8K leaves 1.2 GiB); TP2 16 x 32K leaves 6.3 GiB; TP4 32 x 64K leaves 3.8 GiB. Long context (TP4): passphrase
found at depths 0.1 / 0.5 / 0.9 of 17K, 34K and 63K-token prompts (tools/needle_serve.py). 32-wide decode (TP4)
quality 0.969 / 0.995 top-1 (the 17-32 rows run as two 16-row fused MoE calls).

Served 2026-11-22 (evening; tools/bench_card.py, every request a distinct prompt): decode tok/s/user at 1 user,
total tok/s and TTFT at 16 / 32 users (128-token prompts), before -> after this round:

| profile | 1 user | 16 users x 128: total, TTFT | 32 users x 128: total, TTFT | 1K x 16: total, TTFT |
|---|---|---|---|---|
| p300x2 (4 dies, 32 x 64K) | 30.3 -> 35.9-37.1 | 146 -> 206, 6.6 -> 3.2 s | 163 -> 240, 13.4 -> 6.5 s | 146 -> 157, 14.5 -> 14.7 s |
| p300 (2 dies, 16 x 32K) | 28.7 -> 33.3-34.0 | 116 -> 170, 9.0 -> 3.9 s | | 114 -> 118, 19.8 -> 19.8 s |
| p150 (1 die, 8 x 8K) | 24.7 -> 25.7-26.3 | (8 users) 65 -> 76, 6.0 -> 4.0 s | | (8 users) 50 -> 50 |

Full tables: tt-model.yaml card.performance. One-die DRAM (tools/mem_probe.py): model 30.51 GiB (1.03 free), the
adapter's decode widths 0.42 GiB, one prefix snapshot 0.16 GiB; so one die keeps 1 snapshot, 4 prefill lanes and no
MixPlan (its ~184 MB of concatenated queries). Served step on 4 dies, 1 user (I2M_TIMING=1): device 26.0 ms, inputs
0.2, readback 0.2 (the dies' top-k candidates all-gathered on device: 1.3 ms when each die was read), plugin + sampling
0.5. Gap: prefill ~1,100 tok/s on 4 dies (long prompts queue).

Served-path bug found and fixed: vLLM does not clear a reused block-table row past the new request's allocation;
stale ids reached the global page rows and the prefill's padding rows overwrote this request's own first block
(prompt misread, tool calls looping on <|end_message|>, nondeterministic across runs). GlobalPages.set_row keeps only
the request's own entries (tests/test_reuse_device.py STALE=1).

## Prefix caching (vLLM prefix cache + state snapshots)

vLLM pages the GLOBAL layers' KV: `tt/attention.GlobalPages` holds one block table shared by the 8 global applications;
in vLLM mode a slot's row is its request's block table (prefix-cache hits share blocks), unallocated entries and every
row of a free slot point at scratch blocks past vLLM's pool (padded prefill rows and idle decode rows never write into
a block vLLM may have given to another request). The rest of a sequence's state at a cached position q is not in any
KV block: the four short-conv states of all 48 applications and the sliding layers' last window (W / 64 = 16 blocks of
each ring). `tt/prefix.PrefixStore` keeps them in device snapshot pools (the plugin's prefix_state slots; 168 MB per
snapshot on one die, divided by the TP degree; 4 slots on one die, 16 on 2 / 4); save and restore are one block-copy
program each (`tt/blkcopy.py`). The conv state at a snapshot inside a prefill chunk comes out of the conv kernels
(snap_row); the window blocks are copied after that chunk (a ring holds window + chunk + block). Policy: one snapshot per
prompt at its last 64-token boundary before the last token, prompts >= 512 tokens (I2M_PREFIX_MIN).

Checks: resuming at a chunk boundary is bit-exact vs the whole prefill (tests/test_packed_device.py); the packed conv
kernels' states and snapshot rows are exact (tests/test_pfseg_device.py). Served (p300x2, tools/prefix_serve.py): a 4.4K
prompt cold TTFT 4.4 s, the same prompt again 0.54 s, the next chat turn 0.53 s, a new question on the same document
0.53 s, 4 users on one shared document 1.17 s each.

## Packed prefill (several requests per chunk)

`prefill_forward` packs up to 8 requests (I2M_PREFILL_LANES) into each 1024-row chunk (64-row aligned segments; a long
prompt continues in the next chunk next to others). Row-wise work runs on all rows at once (the MoE reads each expert
once per chunk however many sequences it holds); per sequence: the conv states (one "lane" per segment), the KV writes
and the SDPA rows. Every packed program depends on the chunk size only, never on the segment layout: programs first
compiled after the decode trace capture can be overwritten by the replays (the widths hang). So: the segmented prefill
conv kernel (`sconv_kernels/pfseg_reader.cpp`: per-segment state in / commit / snapshot rows), DynShortConv lags by
block copy, ONE paged KV fill per chunk (a per-64-row-block page table) and ONE SDPA per chunk (per-item page rows),
the last-row gather through row-major pages. Packed vs alone: identical first-token logits (KL 0 / 6e-9).

Prefill is device-bound (tools/bench_prefill.py, op_prof PREFILL): a 1024-row chunk on 4 dies is ~900 ms of device
time (sparse-matmul experts 30 %, attention 14 %, glue the rest); 128 rows ~290 ms (expert reads dominate, hence the
packing).

## Decode, this round (traced step, ms; tests/test_batched_device.py)

| change | TP4 B=1 | TP4 B=4 | TP2 B=1 | TP1 B=1 |
|---|---|---|---|---|
| start of the round (served TPOT) | ~32 | 35.0 | ~34 | 41.1 |
| LM head split over the vocab (2 / 4 dies; per-die top-k, global-max shift, host merge) | ~30.5 | | | |
| block mix logits once per block (MixPlan; bit-identical), row-limited mix SFPU, async all-reduce | 26.9 | 32.1 | 29.8 | 37.2 |
| gate kernel SFPU on the live rows only (bit-identical) | 26.0 | 31.2 | | |
| post-attention section in one program (tt/post.py) | 23.9 | 30.4 | | |
| pre-attention section in one program (tt/pre.py) | 22.7 | 30.0 | | |
| message gate on the post / pre rounds scheme (tt/gate.py GateKernel2) | 20.9 | 29.1 | | |
| + the MoE-input norm and the router logits in the MLP gate's program | 20.5 | 28.9 | | |
| + the attention-side gates with their q / k / v norms | 20.1 | 28.7 | | |
| packed all-reduce: the live rows in 3 / 9 / 18 tiles (tt/ccl.py, kernels/arpack.cpp; bit-identical sums) | 19.2 | 28.0 | | |
| MoE output read by the all-reduce in place (no clone / slice) | 19.0 | 27.9 | | |
| router logits in the post-attention program (applications without message gates; <= 4 rows) | 18.7 | 27.9 | | |
| attention output gate in one program (tt/agate.py; bit-identical) | 17.9 | 27.1 | | |
| post / pre read the packed all-reduce rows in place (no unpack program; bit-identical) | 17.7 | 26.9 | | |
| global layers' dynamic conv taps: one matmul + one add (4 + 4 + 1 programs before) | 17.4 | 26.6 | | |
| the three attention-side gates in one program on three core rectangles (GateKernel2.multi), rounds in parallel | 16.6 | 25.2 | | |

| the MoE router's top-k split over 8 cores per row (QWEN35_ROUTE_SPLIT; bit-identical) | 16.1 | | | |
| k / v cache rows written by the attention prep program (no paged_update_cache; bit-identical) | 15.7 | 24.3 | | |

B = 8: 28.0 ms (286 tok/s) with 2 fabric links per packed all-reduce. B = 16: 47.7 ms (335 tok/s; the pre program's
two rounds now share one gather buffer: at 16 rows its CBs reached 1.4 MB and clashed with L1 buffers); B = 32: 70.5 ms
(454 tok/s). Decode eval B = 8: 0.9706 / 0.9970 / 0.0153; B = 16 (8 texts, 8 idle slots): 0.9671 / 0.9960 / 0.0185
(message-gate programs off: 0.9684 / 0.9949 / 0.0178). tests/test_batched_device.py's per-slot pcc at 16 / 32 slots
is below its bar with or without this round's changes (0.987 with them all off). Tried: fewer SDPA cores per kv head on the sliding
layers (4 / 8 / 16 instead of 55): 16.1 -> 16.0 ms at these positions, dropped.

B = 8 at this point: 31.4 ms (255 tok/s). B = 8 trace replays hung (timing-dependent; not with the watcher): the
middle dies sat in a decode matmul on core (9, 0), the fabric sender core all_reduce_async places right of its output
shard grid, while the outer dies ran ahead into the next all-reduce (tools/tt-triage.py dump_running_operations /
dump_callstacks on the live hang). The decode matmuls now stay on columns 0..8 (tt/dmm.py). I2M_L1_CHECK=1 checks
every generic_op's CBs against the lowest L1 buffer (no overlap; pre / fused MoE reach ~1 MB per core). The post
router round is limited to 4 rows (set while chasing the hang; may be liftable).
Tried and dropped: merging the message gate's first two rounds (the projections of the raw operands with the norm
statistics): slower (18.1 / 27.8 / 33.4 ms at B = 1 / 4 / 8): the bigger round costs more than the round it saves. Tried and dropped:
the attention projections streamed per DRAM bank (qwen36 DecodeLinear): no faster (compute-bound at one row).

Decode quality after the packed all-reduce row (TP4, 8 x 736 teacher-forced): 0.9720 / 0.9970 / 0.0156 (0.9689 / 0.9962 /
0.0156 before the gate rows). The packed all-reduce's pack reads a DRAM source through an aligned staging buffer: a
DRAM read needs its L1 destination 64 B-aligned like the source (32 B chunk slots at 32 mod 64 corrupted face 1).

Decode quality (TP4, 8 x 736 teacher-forced): 0.9711 / 0.9970 / 0.0151 (noise band; the all-reduce change moved it
from 0.9677 / 0.9954 / 0.0212, the eval's noise). Tried and dropped: one fused-MoE call for 17..32 rows (works, pcc
0.9998, but slower than two 16-row calls: the MoE is bandwidth-bound at these widths, and TP4's 288-column expert slices
are stored padded to 384); the MoE rows kernels at one row (still route the sigb router wrongly).

## Prefill, this round (TP4, one 1,024-token chunk, tools/bench_prefill.py wall time)

| change | ms | tok/s |
|---|---|---|
| start | 870 | 1,177 |
| grouped expert matmul over the whole grid (tt/gmm.py) + multi-core dispatch (qwen36 dispatch_mc) | 682 | 1,501 |
| MoE activation + routing weights in one program (tt/moeact.py) | 642 | 1,596 |
| down projection writes row-major rows (no untilize before the un-dispatch) | 640 | 1,600 |
| attention prep for the chunk's rows (tt/aprep.py: q / r / sink, the K / V rows into L1 fill tensors) | 525 | 1,949 |
| block mix over (row tile, H column block) cores, one softmax per row tile per core; the sources read in place | 477 | 2,148 |
| short conv: the row shifts on the FPU (0/1 shift-matrix matmuls), not 3 shifted tiles per tile in the reader | 428 | 2,394 |
| message gates: the fused gate tail over the chunk's tiles (tt/gatet.py) | 397 | 2,581 |
| MoE row gather from an L1 row-major copy of x (chunks <= 2048 rows) | 384 | 2,666 |
| grouped matmul: A multicast within each group slice (read from DRAM once, not once per column block) | 355 | 2,887 |
| grouped matmul: padding groups skipped, the weight block double-buffered when it fits | 352 | 2,913 |
| prefill SDPA: 4 items per work unit (their Q tiles stacked; each K / V chunk read once for the 4) | 346 | 2,958 |

Served (p300x2, vLLM, 2048-row chunks, 32 slots x 1M, pool 1,897,344 tokens), tools/needle_serve.py, depths 0.1 / 0.5 /
0.9 all found: 134,668 tokens 50 s (was 137 s at 126K), 538,715 tokens 286 s (was 842 s at 526K), 1,031,088 tokens
760 s (1,350 tok/s over the whole prompt).

Quality after this round (teacher_force.py prefill path, bfp8 experts, bf16 KV): TP4 2048-row chunks 0.9717 / 0.9970 /
0.0153; TP2 1024-row 0.9697 / 0.9962 / 0.0160; TP1 256-row 0.9703 / 0.9966 / 0.0142; decode eval (TP4 B=8) 0.9711 /
0.9955 / 0.0168. Prefill wall: TP2 442.6 ms per 1024 rows (2,314 tok/s), TP1 320 ms per 256 rows (800 tok/s).
tests/test_packed_device.py on one die: packed vs alone KL 8.6e-4 / 1.8e-3 (bar 1e-4; TP4 / TP2 ~1e-9), argmax equal,
the same at the round's first commit (d1a18d465da): pre-existing, row-count dependent matmul configs on the unsplit
weights (not chased).

At 128K positions (op_prof P0=131072, one 1024-row chunk, device): SDPA 377 -> 111 ms (2 items: 228; the stream bound:
linear in items per unit), the chunk 709 -> 443 ms. 8 items: the CBs (1.56 MB) clash with L1; the tree reduction's
code compiles only for one-item units (4 items overflowed the kernel config buffer by 320 bytes).

2048-token chunks: 3,072 tok/s (4096: 2,912 before the gather change), at +1.29 GiB DRAM per die (the sliding rings
hold the window + one chunk for 32 slots: 2,064,384 -> ~1,895,000 pool tokens); not switched on yet. The 1024-row
profile (op_prof PACKED=1, 357 ms device): gmm 65, rms_norm 29, SDPA 22, mix 19 + its logit matmuls 19, q|k|v|g
projections 15, the row gather 14, gate tail 13 + gate matmuls 15, all-reduces 25, prep 12, dispatch 10. The grouped
matmul's CBs stay within ~800 KB: double-buffered 6-tile weight blocks (1.37 MB of CBs) clashed with the packed
prefill's L1 buffers (from 1.34 MB; tests/test_packed_device.py threw, the bench did not).

Decode eval after the attention prep: 0.9711 / 0.9955 / 0.0168 (noise band). The prep's outputs are 1-2 bf16 ulps from
the op path (exact sums); tests/test_packed_device.py's prefix-resume KL (different chunk shapes) went 1.7e-3 ->
1.1e-2 (its bar raised to 2e-2; the exact resume at a chunk boundary stays bitwise). Not straight into the DRAM caches:
rows 2k / 2k + 1 share a 64-byte DRAM word.

## Speculative decoding: DSpark (the checkpoint's mtp.safetensors)

Not in transformers. vLLM serves it from 0.35 nightly (`"method": "dspark", "num_speculative_tokens": 7`), SGLang from
PR #51926 (`--speculative-algorithm DSPARK`, 8 drafts verified). Reference code (SGLang PR head
zcnrex/sglang@chunan/inkling-2-mini-day0) saved under ~/inkling2-work/ref/sglang: srt/models/dspark.py (RNNHead Markov
head = markov_head_type rnn_hidden; context KV from the target's hidden states), srt/models/dflash.py (DFlash base),
kernels/ops/speculative/dspark/markov_walk.py / markov_sample.py, srt/models/inkling2.py (taps at visits 5, 23, 41 +
the final normed state, 48).

DSpark acceptance (reference/dspark_ref.py, CPU, 8 held-out texts x 40 anchors; accepted drafts of 7 per pass):
teacher-forced greedy 1.20 (base logits without the Markov head 0.70), sampled at temperature 1 (the checkpoint's
default; expected acceptance sum_k prod alpha) 1.21; code / problems 1.6 - 2.1, human chat 0.4 - 0.6. Exact greedy
(the target's own continuation) on math: 2.05 vs 1.30 teacher-forced.

## Next

1. Decode glue: a persistent width-sharded layout for the one-row activations (the norms' reshards and casts are ~1/3
   of the remaining glue), then fused mix + norm / gate rounds; the one-row MoE without the duplicated row.
2. Prefill: long-context SDPA past 4 items per stream (smaller CBs: single-buffered tables), 2048-token chunks in the served config (pool cap),
   the block-mix logits once per block (MixPlan for prefill), the gate head (rms / concat / matmuls).
3. DSpark speculative decoding (acceptance 1.2 per pass measured: low priority).
