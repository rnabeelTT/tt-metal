# Hybrid MoE prototype — handoff

Read this first when resuming. Code: `~/FORK-tt-metal/models/experimental/hybrid_moe` on branch
`raahem/qwen35-397b-hybrid-moe` (fork rnabeelTT/tt-metal, based on `fusion_all`); `~/hybrid-moe-poc` is a symlink to it.
Run data (`runs/`, `logs/`, `cache/`, `ref397/`) is gitignored. Formal plan: `docs/PLAN.md`; deviations under **Deviations**.
The plugin fork `~/FORK-vllm-tt-plugin` has the same branch (no changes yet; the demo uses the image's plugin).

## Current focus: MiniMax-M2.7 hybrid + DFlash (updated 2026-11-08)
Goal: no skipping, a >=100K-context profile (128K target), max tok/s; shorter-context faster profiles allowed.
Port: `models/autoports/minimaxai_minimax_m2_7` (from GLM-4.5-Air; full-width QK-norm in qwen36 `tp.py`, sharded
multi-core for decode rows). Run: `run_m27_dflash.sh <out> [-e ...]` (env CTX, FULL, LIB, PROMPTS, NEW).
Best config (128K): `FULL=20 LIB=/hyblib/minimax_m27_int4_s152 run_m27_dflash.sh <out> -e HYB_LRU=32
-e HYB_FETCH_FRAC=0.3 -e HYB_FETCH_MAX=2 -e HYB_DEV_EXPERTS=/hyblib/minimax_m27_dev_s152 -e HYB_STREAM_SLOTS=64`.

Final benchmark (2026-11-08, `runs/m27/final_131072|65536|32768`, current code: rows-ops attention, fused
drafter, FETCH_FRAC 0.3, DK=3). Steady-state tok/s, no skipping (plain hybrid decode 17.2), 128 tokens, held-out
prompts zh / fr / es / Rust, plus an 8K-token code prompt:
| profile | full layers | verify ms/pass | zh | fr | es | Rust | short mean | 8K prompt |
|---|---|---|---|---|---|---|---|---|
| 128K | 20 | ~46.8-50.4 | 40.2 | 44.3 | 41.7 | 44.9 | 42.8 | 29.4 |
| 64K  | 36 | ~41-44 | 49.4 | 50.4 | 43.6 | 45.3 | 47.2 | 35.3 |
| 32K  | 44 | ~38-40 | 46.7 | 56.8 | 52.8 | 50.3 | 51.6 | 40.9 |
(profile = `CTX=131072 FULL=20` stream slots 64 / `CTX=65536 FULL=36` / `CTX=32768 FULL=44` stream slots 128;
draft ~5.1 ms, stage ~1.3 ms per pass.) vLLM serving at 128K (`serve/bench_final_128k.txt`, 8 prompts, 128 tokens):
**35.5 tok/s mean decode, TTFT 1.11 s** (30.8-44.7 per prompt; includes the API/scheduler overhead).

vLLM serving: `serve/serve_m27.sh` (env CTX 131072 / 65536 / 32768 picks FULL 20 / 36 / 44; SPEC=1 DFlash; DK drafts).
The port's `generator_vllm` loads an external single-user speculative step through `QWEN35_EXT_SPEC=module:factory`
(+ `QWEN35_EXT_SPEC_K`) with MtpServing's interface; `dflash_serve.py` `DFlashServing` is that step (and the
standalone runner's). The plugin registers the port from `EXTRA_MODELS_DIR` (its `vllm_metadata.json`). Greedy only.
Bench: `python serve/bench.py 128 8000`; chat: `serve/chat.py "prompt"`.

Pieces (all traced, per-prompt re-capture):
- DFlash (`dflash_tt.py` `DFlashDrafterPaged`, TP, paged K/V + paged SDPA-decode; 6-7 ms/draft); drafter context
  window 1024 with refills (its acceptance falls with longer context: `window_oracle.py`); verify = the port's
  4-row decode, tap layers all-gathered on device. ~2.0-2.6 tokens/pass.
- `HYB_FULL_LAYERS` (`sim_full_layers.py`): layers that miss most hold all 256 experts (stock fused MoE).
- `hybrid/fetch.py` (`HYB_LRU`): FreeToken-style split; fetched experts go zero-copy from pinned host memory
  (`hybrid/pin/pinx.cpp`, tt-metal experimental PinnedMemory; build: `hybrid/pin/build.sh` in an ubuntu:22.04
  container) straight into their LRU slots (`pinx_write_at`, HYB_FETCH_DIRECT; before: staging + 6 in-trace
  `paged_fill_cache` per layer and pass); the CPU computes the rest on a worker thread. The per-layer crafted rows /
  host partial uploads also go zero-copy (`pin.PinnedRows`, HYB_PIN_UPLOADS).
- Host step findings (2026-11-07, bench_pinx.py + timers): a Python worker thread's GIL hand-offs stalled the main
  thread's device calls (pinned write 18 us -> 5 ms with a busy Python thread) -> CPU experts on hostk's C launcher
  thread (HYB_HOSTK_ASYNC); with the int4 kernel streaming host DRAM a pinned write's enqueue takes ~390 us instead of
  ~15 us -> fetch writes are enqueued before the CPU job starts; routing / crafting in C (HYB_CROUTE). 128K verify
  ~56 -> ~50 ms/pass. CPU experts run at the host DRAM limit (~40 GB/s, 0.23 ms/expert; more threads +7%).
- Fetch policy sweep (128K, runs/m27/r128k_sweep2, POC_FETCH_SWEEP): sync f=0.3 ~= f=0.5 (0.3 slightly better,
  now the serving default); async CQ1 fetch (HYB_FETCH_ASYNC, experts resident from the next pass, the CPU computes all
  this pass's misses) is SLOWER (verify 53-67 ms): device hits do not rise with more warming (~1300 of 1344 picks per
  pass either way) and the CPU work doubles; CPU and PCIe share the same ~40 GB/s of host DRAM.
- Rows-mode attention through fused generic ops (2026-11-07, `QWEN35_ROWS_ATTN_OPS`, default 1): the qwen36
  attn_decode prep (RoPE only: NORM=0, PARTS=2, no gate) / post kernels replace ~35 small ops per layer; one
  `paged_fused_update_cache` per row (V on core row 1). Layer PCC 0.99999; in-process A/B at 128K
  (runs/m27/r128k_ab, POC_AB_ROWS_OPS): device t_read 30.1 -> 26.7 ms/pass, verify ~52 -> ~44.5 ms, es/Rust 38 -> 42
  tok/s. Greedy text diverges after a few tokens (near-ties), outputs coherent.
- Drafter projections (2026-11-08, DFLASH_FUSED default 1): q|k|v and gate|up fused per die, fc column-split +
  all-gather, explicit 1D program configs (bench_drmm.py: default linear 70-240 GB/s, 1D ~320-350). Traced drafter
  pass 5.7 -> 4.1 ms (dbg_dflash_paged: same 46/48 agreement); 128K (runs/m27/r128k_drf) draft 6.5 -> 5.1 ms/pass,
  fr / es / Rust 44.3 / 40.6 / 42.2 tok/s.
- 128K memory (2026-11-08): FULL=20 leaves ~1.2 GiB/die after warm-up; the startup peak is the all-logits prefill
  warm-up's vocab all-gather (0.4-0.8 GB). FULL=28 OOMs there; FULL=24 fits with QWEN35_WARM_ALL_LOGITS=0 (0.7 GiB
  left, 8K prompt OK) but gains only ~1-2% verify (runs/m27/r128k_f24) -> 128K stays FULL=20 (keeps all-logits).
- Host findings (2026-11-08): DDR5 runs at 3600 MT/s (4 x 64 GB, 2 DIMMs/channel; rated 5600): measured read ceiling
  ~45-54 GB/s, the int4 kernel gets ~40 -> a BIOS memory-speed change would speed the CPU experts directly. Small
  device reads cost ~22 us (bench_read.py) and pinned row writes ~10 us idle (bench_put.py) but ~40-50 us inside the
  decode loop (t_up_w; cause not found: not hostk cache pollution, not worker spin).
- DK=4 (5 verify rows in the 8-row kernels: HROWS=8 DK=4 run_m27_dflash.sh; the pad rows repeat the last row, KV
  write masked): 2.3-2.5 tokens/pass vs 2.1-2.4 at DK=3 but verify +6 ms -> 37.9 / 40.0 / 42.9 tok/s, no gain; DK=3.
- Quality (2026-11-08, runs/m27/tf_final, 128K profile settings: FULL=20, s152 library, prefill streaming; teacher
  forced over the 8 held-out 1000-token texts vs the CPU reference, compare_m27.py): top-1 0.848 / confident 0.933 /
  KL 0.196 (before the slot-0 fix and the session's changes, runs/m27dev/tf_c192: 0.850 / 0.933 / 0.183).
- Host int4 library mlocked (`HYB_MLOCK`, `--ulimit memlock=-1`): swapped library pages made the CPU path 4x slower.
  Without a memlock limit (tt-model's container, Docker's default 8 MiB) the lock is best-effort: one warning, the
  pages are pre-faulted instead (HYB_MLOCK=strict fails instead). pinx pinning goes through the KMD and needs no
  memlock. Paths: HYB_LIB / HYB_DEV_EXPERTS / HYB_PLAN take hf://<org>/<repo>/<path> (hybrid/hfpath.py, pinned by
  TT_AUXILIARY_WEIGHTS); the package (autoports/minimaxai_minimax_m2_7/tt-model-hybrid.yaml) reads them from
  raahemnabeel/minimax-m2.7-hybrid-qb2-experts.
- OPEN (2026-11-14, tt-model package e.g. 5e77f21b0271): after a deep elastic round trip (all 24 layers to hybrid and
  back, HYB_ELASTIC_PREWARM=1) short decode stays ~12% slower until restart (bench 49.3 / 51.9 vs 59.2 / 58.2 after a
  restart without it; a 45K request's 4-layer round trip leaves it at 58.6 / 56.2). The engine also keeps ~7 GB more
  host RAM after promoting back. A served ~100K request (19 layers, keep=True) does not do it: bench 54.0 / 56.2 after.
- `hybrid/stream.py`: prefill streams missed experts to the device (8K prompt prefill ~11 s).
- Prefill speed-up (2026-11-09, `bench_prefill.py` / `run_prefill_bench.sh`, `runs/m27/pf_full2`, 128K profile,
  NS=128): 4096-token segments (`QWEN35_PREFILL_SEGMENT`), sentinel ids for the picks outside a staging round (no row,
  zero-filled skip-padding output), up to 3 staging rounds from 1 token (`HYB_STREAM_ROUNDS=3 HYB_STREAM_MIN=1`: no
  CPU experts in prefill), copies on CQ1 (`HYB_STREAM_CQ1`), staging slots keyed by (layer, expert) and reused, and
  layer-major groups of 8 segments (`QWEN35_PREFILL_GROUP`, `model.prefill_group`: ~85% of the copies reused).
  Prefill s, old -> new: 512 1.54 -> 1.26, 2K 2.64 -> 1.84, 8K 10.6 -> 4.24, 32K 44.9 -> 15.8. Top-1 equal at every
  length; the new path is run-to-run identical (the old one is not: CPU int4 vs device bfp4 per miss, 8K self-PCC
  0.96, a 512-token top-1 flip), and its logits differ from the old by about the old path's own run-to-run spread.
  Served (`serve_m27.sh` now sets these; DFlash taps read only the last-window segments): TTFT 6.7K 4.9 s, 26.6K
  14.6 s, 77K 42.6 s (before: 21K 31 s, 98K 144 s); 539 tokens 1.6 s; decode 25-33 tok/s; tool calls / reasoning OK.
  The all-logits LM head runs in row slices (the 4096 bucket's vocab all-gather OOMed at 128K; 1024 rows since
  2026-11-10).
  Next: the remaining hybrid cost is mostly `t_read` (the per-layer routing readback that blocks the host).
- Prefill round 2 (2026-11-10, full model, 128K profile; `runs/m27/merge_full`, `pfetch_full`, `pfetch_lo`, `determ`):
  - Merged routed call (`HYB_STREAM_MERGE=1`, default): the resident experts and the first staging round in one
    sorted-dispatch call. 8K 4.24 -> 3.82 s, 32K 15.7 -> 13.6 s.
  - Next-layer prefetch (`HYB_PF_PREFETCH_MIN`, serve default 1024; NS=212 = two banks over alternating hybrid
    layers, hottest experts first): a layer's first step copies the next hybrid layer's non-resident experts on CQ1.
    2K 1.80 -> 1.70, 4K 2.31 -> 2.02, 8K 3.87 -> 3.44, 32K 13.5 -> 13.1 s. Every variant is run-to-run identical;
    logits across configs differ only by the staging layout's combine order.
  - The all-logits LM head now runs in 1024-row slices (`_ALL_LOGITS_ROWS`): its 2048-row gather (819 MB/die)
    OOMed the warm-up at NS=212.
  - CPU split of the cheapest missed experts (`HYB_PF_SPLIT`): no gain on the full model (the device waits on the
    CPU); default off.
  - Quality: TF vs the CPU reference (`runs/m27/tf_new`, `tf_pf`) top-1 0.8475 / confident 0.935 / KL 0.192
    (`tf_final` 0.848 / 0.933 / 0.196); prefetch on vs off is bit-identical there.
  - Prefix caching (`QWEN35_PREFIX_CACHING`) plus DFlash feature reuse (`dflash_serve.feature_start`: the previous
    request's host features cover the window before the cached prefix, so a hit resumes at vLLM's cached position).
    Served (`serve/prefix_test.py`): 8.3K cold 5.1 s, resend 1.37 s, turn2 1.28 s; 35.9K cold 13.5 s, resend
    1.15 s, turn2 1.37 s; DFlash 1.5-2.6 tokens/pass, 21-36 tok/s.
- Decode round (2026-11-10/11), fixed-text benchmark (`POC_FORCE=<result.json>`: every config decodes the recorded
  tokens of `runs/m27/fb_rec`, so speed is not confounded by what the model writes; noise +-1-3%):
  - `HYB_STREAM_MIN=4` (serve default): the new prefill path's 1-token streamed experts (device bfp4) made the prompt's
    hidden states noisier and DFlash accepted ~6% less (37.9 vs 41.0 tok/s short mean); keeping few-token experts on
    the CPU int4 path restores it (41.8) and keeps most of the prefill gain (8K 5.1 s).
  - Rows-mode MoE down op `QWEN35_MOE_DN_READERS=4 QWEN35_MOE_DN_CHUNK=4` (`bench_moe_rows.py`: 157 -> 137 us per
    call at 4 rows, union 19): 41.8 -> 43.3 tok/s (`fb_g64`).
  - Drafter window `DFLASH_WINDOW=256` on the 8K prompt: 27.2 -> 35.2 tok/s, 1.79 -> 2.08 tokens/pass (`fb_win`; 512:
    30.6).
  - No gain: gs64 int4 library (CPU expert time unchanged), pinned in-place uploads, explicit full-layer router config,
    fetch fraction 0.2 / 0.5, the per-request re-plan (below). Put microbench (`bench_put.py`): a pinned put costs
    ~13 us hot but 22-31 us after 300 us of other work (cache-cold dispatch path), whatever the CPU load.
  - Host jitter: some passes (often the first prompt after startup) run 10-15% slower on the host; kcompactd scans
    ~2M pages/s during decode (`vm.compaction_proactiveness`, a machine-wide setting, left alone).
  - Device profile (`runs/m27/prof_now`): hybrid attention segment 36 programs / ~257 us (4 per-row KV updates, 4 CCLs
    with dispatch gaps), hybrid MoE ~180-230 us (gate|up 16 cores, down 24), a whole layer ~610 us.
- DEMO BUILD v1.3 (tag m27-poc-v1.3): + fetch slack controller by default in serving (HYB_FETCH_POLICY=auto: the
  per-pass fetch budget moved by measured pass time; fb_auto: short / 8K = fixed 15% / 2; el_long_auto: 40K context 3%
  faster per pass with ~27 vs 39 host experts / pass). Served: 8-prompt bench 60.1 tok/s, TTFT 0.63 s; 45K prompt
  switch 6.2 s, TTFT 25.2 s, 32.5 tok/s; prefix hit 2.0 s; back to short 6.9 s.
  - Device jobs take ~/.cache/hybrid-moe/ttdev.lock (run_in_image.sh / run_m27_dflash.sh) and wait while the server
    runs; a validation that starts the server holds the lock itself. A shared-KV multi-row decode attention is being
    built separately (branch raahem/m27-rows-sdpa, worktree ~/m27-rows-sdpa), gated on op PCC, reference-model KL and
    40K-context speed.
- DEMO BUILD v1.2 (tag m27-poc-v1.2, same worktree): + QK-norm fusion on by default (QWEN35_QKNORM_FUSED; vs the
  BF16 reference KL 0.043 / top-1 0.898 = stock; fixed text short +2.2%, 8K +2.8%). Served: elastic switches 8.1 / 5.2
  s, 8-prompt bench 56.5 tok/s, TTFT 0.61 s (unforced serving numbers move +-5% with the generated text).
  - Long context: bench_sdpa_rows.py - the 4 verify rows each read the whole KV (paged SDPA decode 1 vs 4 users: 34 vs
    75 us at 8K, 85 vs 267 us at 40K: ~11 ms / pass at 40K); the chunked prefill SDPA is no substitute (0.9-4.5 ms).
    Next: a shared-KV multi-row decode (patched sdpa_decode kernels: rows stacked as GQA heads, per-row causal limits,
    including the chunk-straddle case).
- Round 6 (2026-11-13) - DEMO BUILD v1.1 (tag m27-poc-v1.1, worktree ~/m27-poc-v1: `cd ~/m27-poc-v1/models/
  experimental/hybrid_moe && serve/serve_m27.sh`; the dev tree ~/FORK-tt-metal is separate; copy hybrid/pin/pinx.so
  into a new worktree - it is git-ignored):
  - Gradual elastic demotion (HYB_ELASTIC_GRADUAL=1 default): a request demotes ceil((need - 32K) / 4096) of the 24
    layers, least valuable first. 45K-token prompt served: switch 7.4 s (4 layers; was 37.2 s with all 24), TTFT 30.5
    s (61.6), decode 32.8 tok/s (28.6); back to short 5.7 s (15.9). runs/m27/el_grad1 vs el_grad0 (40K): 31.8 vs 27.6.
  - HYB_PINX_BGWRITE=1 default (fetch copies on pinx's worker thread): +3-4% (fb_bg1 vs fb_bg0), identical log-probs.
  - QWEN35_KV_DEVICE_ZEROS=1 default: bit-identical (lp_kvz), faster startup / pool growth.
  - Served 8-prompt bench: 58.6 tok/s mean, TTFT 0.63 s (runs/m27/... worktree serve/server_m27.log).
  - Tried, no gain (kept opt-in): HYB_FETCH_POLICY=dyn (budget + value-aware admission) and larger fixed fractions,
    with and without the background writer (fb_dsweep, fb_fsweep, fb_fsweep2): every setting that fetches more is
    flat or slower - the fetch's device-side / host-DRAM cost cancels the CPU work it saves; 0.15 / 2 stays.
  - QWEN35_QKNORM_FUSED=1 did not take effect in this build (lp_qk identical to stock); worth <= 1%, not pursued.
  - Device profile op names: the profiler's runtime ids do not map through mesh_workloads_log.yaml (cache/
    prof_names2.py); kernel fusion needs that mapping first.
- Round 5b (2026-11-12, late) - QUALITY FIX + ELASTIC SERVING:
  - LOST WRITES: hybrid/pin zero-copy writes (pinned host region read over the NOC) silently drop some writes on this
    QB2 (cache/test_pinx_write2.py: 5-15% of slot writes per die on 3 of 4 dies, deterministic; the copy path never
    failed). Every LRU expert fetch, the prefill stream uploads and the image-cache loads used them; refilled LRU slots
    held stale weights (HYB_FETCH_VERIFY=<layer>: 20 of 24 wrong). Pinx now writes through the copy path
    (pinx_write_at2; HYB_PINX_ZEROCOPY=1 restores zero-copy; rebuild pinx.so with hybrid/pin/build.sh).
  - Decode-path quality vs the BF16 CPU reference (ref_m27.py on the forced texts, cache/cmp_ref.py): 128K profile
    top-1 0.814 / KL 0.256 before -> 0.902 / 0.045 after (= fetch off: 0.903 / 0.040), speed unchanged (+1-3 ms/pass).
    Not the CPU int4 experts: per-expert output error int4 0.119 vs device bfp4 0.177 (cache/expert_err.py).
  - METHOD: judge numerics changes against the reference model, not against the baseline run (the baseline-relative
    KL of the 32K vs 128K profiles was 0.22: different, equally valid rounding). The decode path is bit-deterministic
    run to run, so a nonzero baseline-relative KL only says "different".
  - Elastic serving (default in serve_m27.sh at CTX 131072; ELASTIC=0 = static 128K): runs/m27/serve_elastic2.out:
    short 55.8 tok/s TTFT 0.65 s; 45K-token prompt switches to long (37 s: layer builds 15, pin 10, KV 9) then 28.6
    tok/s; resent = prefix hit TTFT 2.5 s; back to short 15.9 s; 8-prompt bench 55.3 tok/s mean, TTFT 0.69 s.
  - Host RAM: ~143 GB is lost to pages leaked by today's OOM-killed runs (not freed by a driver reload; a reboot
    would). Normal exits do not leak (unaccounted memory flat across a run). Never run two big jobs at once.
  - Lazy layer state dicts (QWEN35_LAZY_STATE, default 1): warm load 1:05.
- Round 5 (2026-11-12):
  - Weight-image cache (`HYB_WCACHE`, `hybrid/wcache.py`, the port's `moe.EXPERT_WEIGHT_HOOK`; serve default
    `/hyblib/wcache_m27`): load 21:57 -> 5:21, warm-run tokens identical to the cold run (`fb_ctl_off` vs
    `fb_ctl_off_w`). Images are keyed per layer (slot ids, shapes, dtypes, TP, port): a config change converts once.
  - `kv_rows_update` (one-program multi-row K / V write; QWEN35_KVROWS / DFLASH_KVROWS default 1): exact (greedy tokens
    equal, `fb_ctl_kv`), drafter -0.3-0.4 ms/pass. Clean 128K baseline `fb_base`: short 55.8, 8K 41.6 tok/s.
  - Decode-path quality A/B: `DFLASH_LP_DIR` saves a forced run's per-token verify log-probs, `cache/cmp_lp.py`
    compares two runs. The stock decode is bit-deterministic run to run (`lp_off` vs `lp_off2`: KL 0, top-1 100%).
  - QWEN35_QKNORM_FUSED: PARKED (off). One eager layer matches (`dbg_qkfused.py`, PCC 0.99999) but the traced model
    moves log-probs by KL 0.15 (NLL 0.456 -> 0.546, `lp_fuse`); worth ~1%, cause not found.
  - Old greedy reference `fb_rec` differs from today's outputs from the first tokens: settings changed since (fetch
    fraction 0.3 -> 0.15 moves experts between the device bfp4 and the CPU int4 path), not a regression.
  - 32K profile with today's code (`fb_32k`): short 66.1, 8K 49.4 tok/s (+18% over 128K). -> ELASTIC profile
    (`hybrid/elastic.py`, HYB_ELASTIC_DEMOTE = the 24 layers full at 32K but hybrid at 128K, QWEN35_KV_TOKENS=32768,
    max_seq_len 131072): per request (begin_prefill) the layers are rebuilt full or hybrid through the decoder layer's
    own _build_feed_forward (weights from the image cache) and the KV pool resized (generator.resize_kv; the page
    table keeps the 128K width, columns past the pool -> scratch block); fetch.finalize_more pins the demoted layers'
    host library. Exactness tests: `lp_el_s` vs `lp_32`, `lp_el_l` (HYB_ELASTIC_FORCE=long) vs `lp_base`.
  - Queue scripts: wait on the previous queue's PID (`kill -0`), never `pgrep -f` a path (a launcher shell whose
    command line contains the pattern blocks it forever; races started prof4 and the debug run together).
- Round 4 (2026-11-12): socket-mode device profile (`runs/m27/prof3`, profiler on, one verify pass 52.7 ms): the
  device waits 23.0 ms in the CPU-partial recv and 3.7 ms in the crafted-rows recv (42 hybrid layers); a full layer is
  ~450 us (attention ~220 us in ~36 programs of 5-20 us, MoE ~230 us). The CPU kernel is DRAM-bound: hostk 42 GB/s at
  6-8 threads vs 47.6 GB/s peak read (`cache/bench_hostk.py`; DDR5 at 3600 MT/s), and the device's attention time
  leaves host DRAM idle. `HYB_PREFETCH=1` (serve default): at each hybrid layer the two prefetch threads score the next
  hybrid layer's router on this layer's MoE input (C, `hostk_touch_predict`, ~120 us) and, after the partial is sent,
  read the predicted host experts' int4 rows so the next CPU experts read L2/L3 (`cache/bench_touch.py`: 3 experts
  0.64 -> 0.33-0.44 ms). Cache contents only, no numerics change. Fixed text (`fb_pf2`, warm runs only): short 51.1 ->
  52.5 tok/s, CPU wait 5.7 -> 3.3 ms/pass; served 43.2 -> 44.9 tok/s steady, first pass 39.7 -> 42.6, TTFT 0.84 s. Recall 68% / precision 48% at top-8. Dropped (within noise): top-12 /
  16 (`fb_pfk`, recall 82 / 88%), rescaling by the two layers' norm weights (`fb_pfg`, recall unchanged), adding the
  layer's last-pass misses (`fb_pfp`, 72%); a bf16 dot-product (vdpbf16ps) CPU kernel: 2.5e-6 rel. error but no
  faster even L3-resident (the kernel is not compute-bound). A numpy predictor on the main thread cost 18 ms/pass
  (BLAS threads vs the kernel): keep it in C.
  `HYB_SOCK_LATE_CHOOSE=1`: the fetch choice after the rows (async fetch: no effect on this step), lg -> rows 3.6 ->
  1.8 ms/pass, short 52.8 -> 54.5 (`fb_late`). `HYB_SOCK_LATE_STAGE=1`: the fetch copies issued after the partial (the
  idle window) instead of during the CPU experts, short 50.8 -> 52.1 (`fb_lstage`).
  A/B METHOD: a prompt's first run starts with a cold LRU (45-70 vs ~34 CPU experts/pass), so `cache/agg_sweep.py`
  drops it; earlier "ABBA" sweeps credited the cold run to the first value (fixed-text gains above are warm-only).
  Use sweeps like `attr:0,1,0,0,1` (first discarded -> 1,0,0,1).
- Round 3 (2026-11-11): single-trace socket decode (`HYB_SOCKET=1`, serve default): a captured verify pass is ONE
  trace; each hybrid layer sends its routing logits / activation rows to the host (send_async_d2h, chip 0), receives the
  crafted rows on every chip and the CPU partial on chip 0 (recv_async_h2d, host-push L1 FIFOs of 8 pages: 64-page
  FIFOs starved the prefill matmuls' L1). Host handlers run in layer order after the replay is enqueued (`SEG.post`);
  fetch is asynchronous (CQ1, committed before the next replay). Socket programs are warmed eagerly (`_sock_warm`).
  Microbench `bench_sock.py`: 19 us per layer round trip. Fixed text: short 44.7 -> 48.4 tok/s, 8K 31.9 -> 37.0
  (`fb_sock`, `fb_sock2`: socket mode wants 4 host-kernel threads and fetch fraction 0.15). Serving (`serve/bench.py`,
  128K, steady state): 41.5 tok/s, TTFT 0.84 s (was 37.4-39.7, ~1.0 s); verify 39-43 ms/pass.
  Also: NumPy CPU-job prep (`HYB_NP_PREPARE`, +5%), 3 kernel threads in segment mode (+2-3%), `bench_kvupd.py` (the
  4 per-row KV writes 21 us/layer vs 4 us for one call: a merged writer is worth ~1 ms/pass).
  Quality: `runs/m27/tf_min4` 0.8467 / 0.9353 / 0.194 and `tf_min1` 0.8492 / 0.9348 / 0.196 (baseline 0.8475 / 0.9349
  / 0.192).
- Round 2 (2026-11-11), fixed text unless noted:
  - `HYB_EARLY_CPU` (default 1): the CPU experts start before the routing row is built and uploaded. ABBA A/B
    (`runs/m27/fb_early`): CPU wait 6-11 -> 0.2-2.6 ms/pass; short 42.4 -> 44.1 tok/s, 8K 30.9 -> 32.6. The row upload
    now contends with the kernel's DRAM streaming (t_up 4 -> 7-9.6 ms); `HYB.hostk_threads` (hostk_set_active) A/Bs the
    kernel's thread count at run time.
  - Full-layer router: explicit 1D decode program config (moe._router_logits): device profile `runs/m27/prof2`, the
    13-whole-layer segment 7962 -> 6676 us; device per verify pass (profiler on) 33.2 -> 30.7 ms with the MoE down
    tuning.
  - Prefill: stream threshold by length (`HYB_STREAM_MIN=4`, `HYB_STREAM_MIN_LONG=1` from `HYB_STREAM_LONG_FROM=1024`;
    `runs/m27/pf_min`: 16K 8.5 -> 6.5 s, 4K 2.34 -> 2.0 s); the bucket's padding rows no longer route misses to the CPU
    or the staging (`HYB_PF_MASK_PAD`, default 1); prefill chunks over 16 rows use the packed host kernel
    (`HYB_PREFILL_HOSTK_ROWS=16`): 64 tokens 0.98 -> 0.72 s, 256 1.10 -> 0.94 s (`runs/m27/pf_pad`).
  - Runner: `POC_ATTR_SWEEP=attr:v,..` (in-process A/B of a HYB attribute; use ABBA order: a prompt's second run sees
    LRU slots warmed by the first), `POC_WINDOW_SWEEP`.
  - Drafter window 256 confirmed on all prompts (`fb_win2`); serve default.
- Per-request re-plan (2026-11-10, `HYB_REPLAN`, off by default; `sim_replan_m27.py`, `runs/m27/rp0` vs `rp32`):
  after prefill, up to N static experts per hybrid layer the prompt did not route to are swapped for the cold experts
  it routed most to; victims' device bytes go to a pinned store (`pinx_read_at`), and a victim with no int4 row is
  always fetched in decode (never on the CPU). Sim (served policy, LRU 32 + fetch): CPU experts / pass -36% for a
  256-token prompt. Measured (4 x ~50-token prompts): CPU experts 46 -> 25 ms/pass and t_wait 11.5 -> 6.8, but the
  victims' forced fetches (+2.7 ms fetch writes, +2.1 ms readback wait) cancel it: no net gain, plus 0.1-1.6 s of
  swaps before the second token (store reads ~1 ms each until the store is warm). Victims are routed to 12-24 times
  per pass even after 512-token prompts, so the re-plan pays only if victims keep int4 CPU copies (an int4 library
  of the static ranks for the 42 hybrid layers, ~56 GB). Long prompts: "unused by the prompt" leaves no victims; the
  least-mass rule is in the sim only.
- Device profile of one 4-row layer step (fused MoE): ~540 us = attention ~210 (generic rows-mode op chain; the
  Qwen fused attn-decode op does not fit M2.7: per-head norm + gate), MoE kernels ~250 (near DRAM bandwidth for
  up to 32 experts), QK-norm 110 -> ~20 us after the sharded fix. Profiling recipe: TT_METAL_DEVICE_PROFILER=1 with
  a writable /opt/tt-metal/generated mount; aggregate profile_log_device.csv (cache/prof_seq.py; the stock
  process_ops_logs.py breaks on this pandas).
- Trace rules: allocate persistent buffers before the target's capture; no eager device work between replays
  (re-capture per prompt / drafter refill).
- Host knobs: vm.swappiness=10 set at runtime (sysctl, not persistent).
- Bug fixed: prefill dummy picks pointed at slot 0 (scatter could overwrite the slot-0 expert); re-measure quality.

## Goal
Serve Qwen3.5-397B-A17B (FP8 checkpoint, 512 experts/layer, top-10) on the QB2 (4× Blackhole p300c, 127 GiB) at batch 1:
the device holds the shared layers plus a cache of C experts per layer; missed experts are computed on the host CPU
from host RAM and added to the device MoE output. Validate on Qwen3.5-122B first (fits fully → exact reference).

## Approach (prototype = runtime overlay, no tt-metal rebuild)
- Runs inside the 122B tt-model image `tt-model/qwen3-5-122b-a10b:8a750811ffec` (code at `/opt/tt-metal`, image
  source copy in `image_src/`). The old `-v2` image used by the earlier POC is gone.
- `hybrid/` (this dir, mounted at `/poc`) patches the image's autoport at import time:
  - `hybrid/state.py` — streaming state dict: FP8 block dequant (`weight_scale_inv`, 128×128), per-expert fuse, only
    the C resident experts per layer go to the device.
  - `hybrid/patch.py` — `Qwen35MoE.__init__` builds C expert slots (router kept at E); `forward` reads back routing,
    maps ids → slots on the host, zeroes missed experts' scores, adds the host result before the reduce.
  - `hybrid/host.py` — host expert compute (bf16 dequant first; int4 later), per-layer slot maps, stats.
- Env that must stay OFF for the hybrid path (set by `hybrid.install()`): `QWEN35_MOE_FUSED_DECODE`,
  `QWEN35_ROUTER_TOPK_FUSED`, `QWEN35_FB_ROUTER`, `QWEN35_FABRIC_BOUNDARY`, `QWEN35_DECODE_REPLICATED`, `QWEN35_MTP`,
  `QWEN35_VISION`; `QWEN35_PROGRAM_GUARD=0`; decode/prefill traces off (eager) until milestone M3.
- Launch: `./run_in_image.sh <script.py> <out_dir> [-e K=V ...]` (mounts `/hf`, `/cache`, `/poc`, hugepages).

## Milestones
- M0 prerequisites: disk freed (deleted docker volumes Llama-3.1-70B, gpt-oss-120b, DeepSeek-R1-Distill-70B + build
  cache), FP8 397B download (`logs/download_397b.log`, ~58 MB/s, 406 GB).
- M1 122B hybrid eager: C=E exact vs stock; C=50% coherent + hit rate.
- M2 397B hybrid eager: loads within device memory, coherent output.
- M3 dynamic cache (slot promotion) + int4 host kernel.
- M4 per-layer traced decode (speed).
- M5 vLLM serving at batch 1 (demo).

## Status
- Serving (2026-11-05): default context 131072 (MAX_LEN; KV 2049 blocks x 64, fits next to C=256 + MTP head). Tool
  calling on (`--enable-auto-tool-choice --tool-call-parser qwen3_coder`, matches the template's XML tool calls) plus
  `--reasoning-parser qwen3`. Checked: tool call, streamed tool call, tool-result round trip, thinking on/off; needle
  in 21K / 98K / 129K-token prompts found (prefill 31 s / 144 s / 190 s); decode after 34.0 tok/s.
- 4-layer breakdown (2026-11-05, layers 0-3, heldout x2, 256 tokens, traced, hostk, `runs/l4b_*`): decode ms/token
  skip-all-misses 3.1 (hybrid machinery + device only), skip 0.15 3.5 (host experts ~0.1 ms/layer), devcraft 4.2-4.3
  (readback 0.56-0.6 s: the non-blocking read waits longer than the host craft). C=512 takes the unfused stock path
  (3.9 incl. prefill), so it is no longer a like-for-like reference. Takeaway: the remaining time is device layer
  time, not host work; further gains need fewer device ops per layer or more accepted MTP tokens per pass.
- Quality vs the real model (2026-11-05, `ref397_tf.py`, `runs/quality/ref2/compare.json`): CPU reference = HF
  transformers Qwen3_5MoeDecoderLayer in bf16 streamed from the FP8 checkpoint (A_log / GDN norm kept fp32), teacher
  forced over 8 x 1000-token texts (`quality_texts.py`: GSM8K, MBPP, EN/ZH/FR chat, a Python file, a markdown doc,
  LiveCodeBench). Ours: device prefill with all logits (`run_generate.py` POC_TF_TEXTS). Mean top-1 agreement /
  on reference-confident positions / KL: skip 0: 0.900 / 0.945 / 0.102; 0.05: 0.890 / 0.937 / 0.117; 0.1: 0.874 /
  0.920 / 0.160; 0.15 (serving default): 0.838 / 0.889 / 0.266. Math is the worst text (KL 0.27 at skip 0, 0.59 at
  0.15). The skip-0 gap is the 4-bit weights / bfp4 mixer policy, not the hybrid. The multi-turn chat texts are OOD
  for both (the reference itself predicts <|im_end|> mid-turn); agreement there is still valid.
- Served speed at skip 0.05 (MTP k=3, hostk): 17.5 tok/s (vs 34.6 at 0.15); IFEval (non-thinking, greedy, 541):
  see runs/eval/. Host DRAM measured 36-40 GB/s (4 DIMMs at 3600 MT/s), so the host kernel is near the ceiling
  and PCIe shares the same DRAM: zero-skip speed needs fewer bytes per miss or fewer misses.
- HYB_LOWBIT (2026-11-05, default 0): bit-plane library layout (hostk_planarize, in place at load: same bytes) so a
  computed miss whose largest routing weight is below the threshold reads only the top 2 bits of its int4 weights.
  CPU checks (layer 30, 12 experts): bit-plane full read == torch-pack read exactly; hi-only 1.9x faster per
  expert, 0.49 relative output error (skipping = 1.0); prefill's torch path reproduces it (zero lo plane, zero
  point + 1.5 scales) within bf16 (0.4%). Not yet measured on the device (TF fidelity + speed).
- Serving defaults (2026-11-05): HYB_MTP=3 + HYB_SKIP=0.15 + hostk: **34.4-34.9 tok/s** (4 runs, serve/bench.py,
  8 held-out prompts x 128 tokens, TTFT 0.65 s). Same sweep: MTP=3 skip 0.2 35.7 / 36.2 (TF top-1 0.951 vs 0.959
  at 0.15, kept as opt-in); MTP=0 skip 0.15 30.7 / 32.4; MTP=3 skip 0.1 (old default) 27.5 / 28.4. MTP=0 serving
  used to crash in warm-up (prefill bucket traces captured the hybrid MoE): the prefill-trace patch now installs on
  every path. Host RAM peak ~184-191 GB with MTP, ~206 GB without (library loads alongside the model).
- HYB_DEVCRAFT (2026-11-05, default off): crafted rows built on the device (top-k threshold, where, one-hot slot
  matmul, sink logsumexp; `test_devcraft.py`: PCC 1.0 vs host-crafted, 0.997-0.998 vs the sparse path), non-blocking
  logits read, experts queued behind it. Served k=3: 25.2 / 25.7 tok/s, 110 ms/pass (vs 95): the ~15 extra small
  device ops per layer cost more than the host round trip they remove ('read' 44 -> 57 ms/pass). The device's own
  per-layer work (~0.75 ms/layer, 'read' ~44 ms/pass) is now the floor.
- Skip sweep 2 (2026-11-05, `runs/full397_skip2/`, standalone no-MTP traced decode with hostk, prompts 12-15): tau /
  TF top-1 vs ref / vs tau 0.1 / ms per token: 0.1 0.969 / 1 / 37.5 (26.7 tok/s); 0.15 0.959 / 0.977 / 29.2;
  0.2 0.951 / 0.977 / 27.9; 0.3 0.955 / 0.961 / 27.2. Text coherent at all four.
- Fused host-expert kernel (2026-11-05, HYB_HOSTK=1 default, `hybrid/hostk/`): one C call per layer (AVX-512, its
  own 4-thread pool, short spin then futex) runs every missed expert on the library's torch CPU int4 pack (decode /
  MTP verify rows <= 16; prefill keeps torch GEMMs). PCC 0.999996 vs the torch path. Host experts per MTP pass
  61 ms -> 17-28 ms; pass 112-156 ms -> 93-96 ms; bench 27.5 / 28.4 tok/s (vs 27.2-27.7): the critical path is now
  the device + per-layer sync ('read' ~44 ms/pass). An 8-thread pool spinning 200k pauses starved the host step
  (23.8-25.0 tok/s). Built on the host by serve_397b.sh (`python -m hybrid.hostk`; the image has no gcc).
- MTP k=5/k=7 (2026-11-05, with the lean head load both now load: peak ~181-182 GB): 20.1 / 23.3 tok/s, below k=3.
- Lean MTP head load (2026-11-05): with MTP on, the resident int4 library copies start only after the MTP head is
  built (patch._install_mtp_head), and state.mtp_state fills the 512-expert stacks in place. Host RAM peak during
  startup 226.6 GB -> 179.7 GB (= steady state; was 28 GB available at the head build, now ~75 GB), same ~25 min
  startup, bench 27.7 tok/s (k=3, 12 threads). This is the headroom k=7 was missing; k=7 not yet retried.
- MTP sweep (2026-11-05, serve/bench.py, 8 held-out prompts x 128 tokens): k=3/8 threads 25.4 tok/s; k=1 21.1;
  k=3/12 threads 27.2 (serving default now); k=3/16 threads 26.2. k=7 (8-row verify): the engine process died
  silently right after the layers loaded (no TT error in the log; host RAM pressure at the same time: Claude Code
  reaped the sweep for low memory) - not retried. Host RAM budget: ~127 GiB resident int4 library + the MTP head's
  bf16 expert stack during its build + checkpoint page cache.
- MTP (2026-11-04): `serve/serve_397b.sh runs/plan397_v4.npz` now defaults to HYB_MTP=3 (k=3 drafts, 4-row verify
  through the fused rows kernels on crafted rows; the MTP head's MoE keeps all 512 experts on device, stock forward).
  2.5-3.4 tokens/pass, 112-156 ms/pass -> 17-26 tok/s (vs 13-20 without MTP). Per pass (HYB_LOG_EVERY=50):
  host experts 61 ms (4-row union misses, ~26 GB/s, DRAM-bound), readback incl. device wait 45 ms, uploads 15 ms.
- Accuracy fix: one-row decode now runs the 2-row fused kernels (`_kernel_rows`); the one-row kernels on crafted
  logits were PCC 0.93-0.97 per layer vs the sparse-matmul hybrid (test_rows.py), the rows kernels 0.997-0.998.
- 2026-11-04 best so far (`runs/full397_v4/`): fused C=256, HYB_SKIP=0.1, plan v4 (59 prompts, `prompts_v2.TRAIN` +
  old), lean host step (numpy routing, single-row RM transfers). 8 diverse held-out prompts (`prompts_v2.HELDOUT`):
  decode 46.6 ms/token mean (21.5 tok/s; 39.9-66.4 ms, best 25 tok/s), TTFT 0.9 s, decode hit 78%, coherent in
  zh/fr/es/code/math/prose. Chinese is the slowest (56% hits).
- Serving (2026-11-04): `serve/serve_397b.sh runs/plan397_v3.npz` (fused C=256, HYB_SKIP=0.1) -> vLLM chat at
  13-17 tok/s decode, TTFT 1.1-1.3 s, coherent (MoE explainer, Canberra, Python palindrome), no server errors.
- Skip sweep (2026-11-04, `runs/full397_skip/`, prompts 12-15, one build): HYB_SKIP drops missed experts whose
  routing weight is below tau. Mean decode ms/token / TTFT / teacher-forced top-1 vs tau=0 (vs ref):
  0: 83.5 / 2.0 s / 1.000 (0.979); 0.02: 82.5 / 1.8 / 0.986; 0.04: 82.7 / 1.7 / 0.984; 0.06: 75.3 / 1.5 / 0.981;
  0.1: 53.2 / 0.9 s / 0.975 (0.961). The prefill-vs-decode path noise floor is ~0.979 (tau=0 vs its own decode
  reference), so tau=0.1 deviates about as much as path numerics. Serving default HYB_SKIP=0.1 (19 tok/s mean,
  23 tok/s best prompt); code default 0.
- Fused MoE decode (2026-11-04, HYB_FUSED=1 default): the production fused gate-up/down kernels run on a crafted
  logits row (see patch._fused_step1) over C=256 slots (255 experts + an all-zero sink). Full 397B, plan v2, held-out
  prompts 12-15, 128 tokens (`runs/full397_fused3/`): decode 67-134 ms/token (7.5-14.9 tok/s), TTFT 1.8-3.5 s,
  decode hit 63-85%, coherent. The crafted row MUST stay in L1 (DRAM garbles logits 16-31 of each tile: BH DRAM
  read alignment). HYB_BOUNDARY (replicated residual + fabric) is off: its full-model test also had the DRAM bug, so
  re-test; FB_CONSUMER clashes with the prefill norm's L1 at hidden 4096.
- Held-out evaluation (2026-11-04): earlier hit rates were in-sample (plan from prompts 0-3, tests on 0-2) and
  included prefill padding rows. Decode-only, held-out: 4-prompt plan 53-69%; 15-prompt plan (`runs/plan397_v2.npz`)
  66/76/85/87% on prompts 12-15 (sim predicted 79%). C=272 (29.3 GiB/die), production attention/GDN/sampler fusions
  on, HYB_OVERLAP=1: 98-188 ms/token incl. prefill over 128 tokens (5-10 tok/s), coherent. `sim_397.py`: dynamic
  slots only pay at >=16 promotions/token (prep ~35 ms each), so they stay off; serving plan = `runs/plan397_v3.npz`
  (19 prompts). Multilingual prompts miss most (translation 66%): add such routing to the plan.
- Speed (2026-11-03): C=256 static, D=0, RAM-resident int4 library -> 110-143 ms/token incl. prefill (~7-9 tok/s),
  hit rate 73-78%, coherent (`runs/full397_c256/`). The dynamic slots' prep threads (FP8 dequant + bfp8 conversion)
  starved the host step (transfers 50-100x slower than `bench_xfer.py`), so they are off; 16 bfp8 dyn slots = 32 bfp4
  static in memory. 27.67 GiB/die allocated. Per token now: readback incl. device wait ~57 ms, upload ~22 ms, host
  experts ~25-50 ms.
- M0 done: 397B FP8 in HF cache (`Qwen/Qwen3.5-397B-A17B-FP8`); int4 host library (group 32, ~7.5 MB/expert, 226 GB)
  at `~/.cache/hybrid-moe/qwen35_397b_int4/layer_<L>.pt` (mounted `/hyblib`), built by `build_int4_library.py`.
- M1 done: full 122B, C=128/256 static (plan `routing/routing.npz`): coherent text (`runs/full122_c128/`).
- M2 in progress: 397B 4-layer smoke OK (C=224); full 60-layer run `runs/full397_c224_d16/` (records routing).
- M3 done (prototype): dynamic slots. `fill_cache` cannot write bfp4 (only fp32/bf16/bfp8), so the dynamic cache is a
  separate bfp8 side tensor of D slots per layer, written with `fill_cache` through a `[D,1,H,W]` reshape view
  (verified aliasing: `test_fill_slot.py`, 1.2 ms/slot). Static set = most-routed experts (HYB_PLAN). Promotions are
  prepared on 2 worker threads (FP8 dequant + TP pack + bfp8 host tensor, ~35 ms) and written at the start of that
  layer's next MoE call. Two routed passes per layer (static bfp4 + dynamic bfp8), then + host result.

- M5 done (works, slow): vLLM OpenAI server `serve/serve_397b.sh runs/plan397_merged.npz` (~17 min to load), then
  `serve/chat.py "prompt"`. Coherent answers; TTFT 15-30 s (eager prefill), decode ~0.58 tok/s. Serve-only env:
  `QWEN35_MAX_TOKENS_ALL_USERS` (KV pool, default = MAX_LEN 8192) and `QWEN35_PREFIX_STATE_SLOTS=0` (122B budgets).
- M4 done (works, slow): full 60-layer 397B traced decode coherent (`runs/full397_trace2/`): ~1.5 s/token, hit rate
  ~67% (C=224 static + 16 dyn). Per token: host compute ~0.55 s, readback ~0.33 s, upload ~0.33 s, promotions ~0.13 s
  -> host-side overheads dominate; device time is small. Speed work after M5.
- M4 mechanism: segmented decode traces, `hybrid/seg.py`. It wraps `ttnn.begin/end/execute/release_trace`.
  While a decode trace is captured, each MoE layer calls `SEG.split(host_step)` right after its router. Replay of the
  generator's trace id runs the real segments in order and calls each layer's host step between them: read back
  ids/scores (and x when there are misses), apply promotions, host-compute misses, then `copy_host_to_device_tensor`
  into per-width persistent buffers (`_bufs`: slot ids, masked scores, dynamic ids/scores, host partial). These
  buffers are allocated in the eager warm-up, before capture. Host steps must NOT allocate device memory: freed
  trace intermediates could be handed out and overwritten. That is why promotions write through preallocated
  staging tensors. Driver: `-e POC_TRACE=1`.

## Deviations from docs/PLAN.md
- Overlay on the 122B image instead of a `qwen_qwen3_5_397b_a17b` autoport in a tt-metal branch (port later).
- No per-miss CPU/PCIe split (q*): every miss is computed on the CPU (int4); promotions to the device are a fixed,
  async rate (`HYB_PROMOTE`, default 2 per layer per token) and only help later tokens. No startup bandwidth probe.
- Two device expert sets: static bfp4 (C, from a routing plan) + dynamic bfp8 (D, LRU), since fill_cache rejects bfp4.

## Key commands
```bash
# 397B, eager, hybrid (static 224 + dynamic 16 per layer), int4 host experts
HF_MODEL=Qwen/Qwen3.5-397B-A17B-FP8 ./run_in_image.sh run_generate.py runs/<name> \
  -e HYB_SLOTS=224 -e HYB_DYN_SLOTS=16 -e HYB_LIB=/hyblib/qwen35_397b_int4 \
  -e HYB_PLAN=/poc/runs/<prev>/routing.npz -e POC_PROMPTS=2 -e POC_NEW_TOKENS=64 [-e POC_LAYERS=0,1,2,3]
```
Device memory: 4-layer 397B at C=224 = 2.36 GiB/die; full model ≈ 26 GiB of 31.8.

## Gotchas
- A mesh tensor's `.shape` is per-device (e.g. W/4 when sharded on the last dim).
- `ttnn.fill_cache` rejects bfp4 caches and inputs.
- The image's prefill trace / program guard must be off (`hybrid.install()` sets env).
- vLLM eager: `--override-tt-config '{"trace_mode":"none"}'`; overlay enters via `serve/sitecustomize.py` (HYB_ENABLE=1).

## Next
- Finish M2 (full 397B coherent), then re-run with HYB_PLAN from its recorded routing.
- M5 vLLM eager serving (sitecustomize); then M4 speed (per-layer traces).
- Later: turn the overlay into a `qwen_qwen3_5_397b_a17b` autoport in `~/FORK-tt-metal` (user's bring-up fork).

## Log
- 2026-11-05: quality vs real 397B measured (skip 0.15 costs ~6 pts top-1); HYB_LOWBIT 2-bit misses added.
- 2026-11-05: served 34.4-34.9 tok/s (MTP k=3, skip 0.15, hostk); MTP=0 serving fixed.
- 2026-11-04: MTP k=3 served: 17-26 tok/s; one-row decode via 2-row kernels (PCC 0.95 -> 0.998).
- 2026-11-04: held-out decode 21.5 tok/s mean (plan v4 + lean host step).
- 2026-11-04: fused MoE kernels + HYB_SKIP=0.1 served through vLLM at 13-17 tok/s (was 0.58).
- 2026-11-03: M5 vLLM serving works (0.58 tok/s). Next: speed (host overheads), hit rate.
- 2026-11-03: M4 full-model traced decode coherent, 1.5 s/token. Next: M5 vLLM serving.
- 2026-11-03: disk freed to 671 GB; 397B FP8 download done (~1 h); image source extracted to `image_src/`.
- 2026-11-03: full 60-layer traced run crashed at capture: L1 CB clash (router scores/ids kept live in L1 per layer
  for the host step). Fix: DRAM copies for the host step. Repo initialised, plan copied to docs/PLAN.md.
- 2026-11-03: overlay written; 122B M1 coherent; int4 library built; dynamic bfp8 side cache working on 4-layer 397B.
