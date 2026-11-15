# Inkling-Small on one QB2: hand-off log

Read this first when resuming. Newest entries at the bottom of **Log**. Branch: `raahem/inkling-small-hybrid` (fork
`rnabeelTT/tt-metal`), cut from `hybrid-moe-starter`. Approach and gates: see **Plan** below; the M2.7 bring-up this
follows is `ADD_A_MODEL.md` + `HANDOFF.md`.

## Status (update every session)

| Phase | State |
|---|---|
| 0. env / disk | done. Watch disk: the expert-image cache `~/.cache/inkling/wcache` grows by ~130 GB per precision profile (a full disk killed the 12:38 queue on 2026-11-14). |
| 1. CPU reference + routing | done. fp32 router as vLLM computes it. Noise floor between two valid references: top-1 0.925 / conf 0.978 / kl32 0.081 (`compare_refs.py`). |
| 2. TT port | done for text: decode, 8-row verify and chunked prefill all match the fixtures per layer (prefill PCC = decode; 1000-token prefill ≈ 4 s warm). |
| 3. hybrid integration | done. **Profile chosen: all bfp8, C=106 slots, S=106 int4 library**: 0.862 / 0.945 / 0.218 (prefill path), at parity with M2.7 (0.848 / 0.933 / 0.196). Decode path matches prefill path. |
| 4. spec decode | native MTP, verify + drafter both traced, exact (greedy byte-identical to plain decode; exact speculative sampling). Dev-harness best k=1: 15.3 tok/s (15.9 with `HYB_HOST_ROUTE=1`, quality-checked). **Not yet in vLLM**: lives in `tools/spec_bench.py`. DSpark CPU reference only. |
| 5. optimisation | chunked prefill done. Decode (all dev-harness, `tools/spec_bench.py`): traced verify + commit + drafter, one-program KV row update, row-major short conv, device-untilized logits, opt-in host routing. Bottleneck: host int4 experts (~55 of ~104 ms per k=1 step; DDR bandwidth-bound). |
| 6. vLLM serving, package, multimodal | **serving done** (fast profile, see Log 2026-11-15 session 2); package build in progress; multimodal not started. |

**Plan to completion (2026-11-15, second session; supersedes the old next-steps list).** The goal is a published
`tt-model` package served by vLLM, with the best context / TTFT / tok/s at M2.7-parity quality.

Diagnosis behind the plan:
- Decode is bound by host int4 expert bandwidth (~36 of ~46 GB/s DDR), so the device slot count decides speed.
  All-bfp8 C=106 misses ~80 experts per token (~1.26 GB of reads); bfp4 C=194 misses ~13 (~0.21 GB). Multi-row
  verify multiplies misses, so a high-miss profile also caps spec decode at k=1.
- C=106 also makes prefill CPU-bound: ~4 s for 1000 tokens. bfp8 device copies for prompt streaming (~160 GB) do not
  fit in host RAM next to the int4 library.
- On quality the profiles are closer than top-1 alone suggests. early_gu4dn8 C=194 is 0.842 / 0.933 / 0.287 (M2.7:
  0.848 / 0.933 / 0.196), and its NLL is closer to the reference than all-bfp8 C=106 (3.136 vs 3.168; ref 3.093).
- A bfp4 profile can use M2.7's tuned fast path (fused kernels, single-trace socket decode, fetch, prefill streaming).
  The LoFi weight rounding is lossless for bfp4. Inkling's fused build and host routing are already in place, but
  that path has never run.

Steps (device jobs one at a time, each watched; CPU / editor work runs alongside):
1. **vLLM serving v1** on plugin `23b0736` (`prefix-state-cache`, the revision M2.7 production runs; the spec shim
   matches it). `tt/generator_vllm.py`, `vllm_metadata.json`, `vllm_bundle/`, HF-config registration in
   `serve/sitecustomize.py`, an MTP spec step (`QWEN35_EXT_SPEC`, built from `tools/spec_bench.py`'s traced loop),
   and `serve/serve_inkling.sh`. Gate: greedy text through the server equals plain decode; chat, reasoning and tools
   work.
2. **Fast profile:** bfp4 + bfp8-ish early layers at C~194 on the fused / socket path. Gates: decode path = prefill
   path (rows consistency), teacher-forced parity, then task evals.
3. **Quality decision by task evals through the server** (GSM8K, IFEval, a LiveCodeBench slice), fast profile vs
   all-bfp8 C=106. If the fast profile falls short: GPTQ-style error-compensated rounding into the bfp4 grid (offline,
   CPU; the device format is unchanged) for the device experts.
4. **Speed in serving:** fetch / LRU slots, prompt streaming (TTFT), MTP k (and DSpark if it accepts more), long
   context (128K default; 256K / 1M if DRAM allows; log_scaling > 128K tested).
5. **Package and publish:** `tt-model-hybrid.yaml` (first-boot expert builds like the M2.7 package), card numbers
   from `vllm bench serve` / `serve/bench.py`, `tt-model package --container`, a clean `tt-model serve` test,
   `tt-model push`.
6. Multimodal (vision / audio encoders on the host) after text ships.

## vLLM serving plan (mapped 2026-11-15 from the M2.7 hybrid port; plugin fork `~/fork-vllm-tt-plugin`)

How M2.7 is served (copy this shape):
- **Registration:** `EXTRA_MODELS_DIR` bundle: a folder with `vllm_metadata.json` `{"arch", "main_class"}`
  (`models/autoports/minimaxai_minimax_m2_7/vllm_metadata.json`, `vllm_bundle/`); plugin `platform.py:1292-1369`
  registers `"TT"+arch`. Inkling's arch is **`InklingForConditionalGeneration`** (`model_type inkling_mm_model`).
- **Adapter:** `MiniMaxM2ForCausalLM` (`minimaxai_minimax_m2_7/tt/generator_vllm.py:181`) <- `Qwen36ForCausalLM`
  (`models/demos/blackhole/qwen36/tt/qwen36_vllm.py:48`) <- `Generator`. Inkling's `tt/generator.py` already re-exports
  M2.7's `Qwen35MoeGenerator` and `tt/model.py` subclasses M2.7's model, so a copied adapter mostly needs the imports,
  `HF_MODEL`, `_KV_LAYERS` / DRAM constants, `context_contract.json` and vision code replaced.
- **Hooks the plugin calls:** `initialize_vllm_model` (classmethod), `allocate_kv_cache`, `prefill_forward`,
  `decode_forward`, `warmup_model_prefill`, `warmup_model_decode` (all captures happen in decode warm-up phase 2),
  `model_capabilities`, optional `get_max_tokens_all_users`, `read_decode_output` / `process_decode_output_host`,
  `decode_input_update_contract = 1`. Plugin docs: `docs/MODEL_INTERFACE.md`, `MODEL_CAPABILITIES.md`.
- **Hybrid overlay:** `models/experimental/hybrid_moe/serve/sitecustomize.py` runs `hybrid.install()` in every vLLM
  process (serve/ first on `PYTHONPATH`); `templates/serve_model.sh` has the docker + `vllm serve` command
  (`--max-num-seqs 1 --block-size 64`, `trace_mode all`, `sample_on_device_mode all`, `fabric_config FABRIC_1D`).

Inkling-specific work:
1. **HF config registration in every vLLM process before `ModelConfig`:** transformers in the image cannot load
   `inkling_mm_model`; call `tt/hf_config_inkling.py register()` from `serve/sitecustomize.py` (or a plugin hook like
   `_register_kolibri1_hf_config`, `platform.py:638-651`). Importing it from generator_vllm is too late.
2. **`tt/generator_vllm.py` + `vllm_metadata.json` + `vllm_bundle/`**, `supports_multimodal = False` (text first).
3. **KV:** only the 7 global layers (5, 11, ..., 41) take vLLM's paged cache (`InklingAttention.bind_cache`,
   `args.inkling_own_kv=False`); sliding layers keep their own ring cache (`tt/attention.py`). Every layer sets
   `is_full_attention=True` (`tt/layer.py:87`), so the inherited `allocate_kv_caches` must NOT be used as is.
   Conv states + rings are per sequence: `model.reset_sequence_state()` on every new prefill; `--max-num-seqs 1`;
   prefix caching off (no conv / ring snapshots) until state snapshots exist.
4. **Spec decode -- the decision to make first:** M2.7 hybrid serves spec via the model-internal step path:
   `QWEN35_EXT_SPEC=module:build` + `QWEN35_EXT_SPEC_K=k` and M2.7's monkeypatch shim `tt/mtp_plugin_shim.py`
   (decode returns host int32 `[B, k+1]`: accepted drafts + verified token, -1 padded). The step object's interface:
   `models/experimental/hybrid_moe/dflash_serve.py:177-512` (`allocate_kv`, `warm`, `capture`, `begin_prefill` /
   `end_prefill`, `step(tok, pos, ..., sampling)`). Wrap the traced loop of `tools/spec_bench.py` (classes `Traced`,
   `DraftTraced`, `SpecSampler`) in such a class. **Blocker:** the shim matches plugin commit `8b1f8b1`
   (`prefix-state-cache`), which the published M2.7 package pins; on fork `main` (`5b5b6f8`) its patched signatures
   are stale (`_build_runner_output` lacks `stage_outputs=`, `_update_request_with_output` lacks `is_stale`).
   Options: (a) serve on the 8b1f8b1 plugin like M2.7; (b) port the shim to `main`; (c) implement `main`'s native
   contract (`docs/SPEC_DECODE_CONTRACT.md`: `spec_plan`, `decode_forward(draft_token_ids=..)` -> `VerifyOutput`,
   `propose_draft_tokens` -> `DraftOutput`; no tt-metal model implements it yet). Ask the owner which plugin
   revision production uses.
5. **Parsers:** check whether the vLLM in the image ships Inkling reasoning / tool parsers; else add entries in the
   plugin's `entrypoints.py:20-29` (`kolibri1_reasoning_parser.py` as the pattern). Chat template takes
   `reasoning_effort`; vocab padded to 201024 (unpadded 200058): mask the tail in device sampling.
6. Serve env for the chosen profile: `HYB_PORT=thinkingmachines_inkling_small HYB_SLOTS=106
   HYB_LIB=.../inkling_int4_s106 HYB_FUSED=0 HYB_DECODE_ROWS=8 HYB_HOSTK_THREADS=6 HYB_WCACHE=...` (same as
   `tools/eval_hybrid.sh`), plus `HYB_HOST_ROUTE=1` (decided 2026-11-15).

Dev-harness invariants a vLLM port must keep (each was a real bug, see Log 2026-11-14/15): run one eager pass at every
row width (1..8) AND the drafter at every width/depth BEFORE any trace capture, then capture every width up front
(tensors allocated after a capture get overwritten by its replays); the drafter uses its own `TT_CCL`; hybrid needs
`HYB_DECODE_ROWS >= verify rows`; read back row-major logits (untilize on device).

## Log

- **2026-11-14:** Plan approved: text first, MTP and DSpark both built and picked by measurement, BF16 as source and
  reference.
  - Branch created and Docker pruned, keeping the newest tag per repo; 729 GB free.
  - `libhostk.so` built, BF16 + DSpark download started, CPU venv and text sets built.
  - `ref_inkling.py` written and validated.
    - Validation is against HF's full-model forward on a tiny random Inkling, chunked and unchunked: max abs error ≤7e-6.
    - One bug found and fixed on the way: in a layer-at-a-time run, the HF mask helpers size masks from the first cache
      layer of each type, so every cache slot is aliased to the current layer's entry.
- **2026-11-14, first real-weight device results** (640 tokens of `code_python_file`, HF fixtures):
  - **Attention** (alone, PCC per token): layer 0 (sliding) worst 0.9995 / mean 0.9998; layer 5 (global) worst 0.9984 /
    mean 0.9995.
  - **Whole layers, contribution PCC** (`out - x_in` vs the reference's): dense layer 0 1.0000; dense layer 1 0.99995.
    - The MoE layers before the router fix below: layer 2 0.9985 mean; layer 5 0.9981 mean, worst 0.9745.
  - **MoE, measured alone** (`tests/test_moe_device.py`, layer 5; "same-routing" means tokens where device and
    reference pick the same 6 experts):

    | Experts | Same-routing PCC, mean | Same-routing PCC, worst |
    |---|---|---|
    | bfp4 | 0.9915 | 0.984 |
    | bfp4 gate/up + bfp8 down | 0.9957 | 0.989 |
    | bfp8 | 0.9993 | 0.9985 |

  - Layer 2 shows the same ranking: 0.9951 / 0.9982 / 0.9996.
  - Routing flips against the reference are now 1 in 640.
  - Policies are in `doc/precision/*.json`; pass one with `QWEN35_PRECISION_CONFIG`.
  - **The reference's router is now fp32**, as vLLM's InklingGate computes it: fp32 logits from the bf16 matmul, then
    fp32 bias, sigmoid and global_scale.
    - Transformers' all-bf16 router flipped 12% of the picks.
    - One more bug: an in-place bf16 cast of the bias in the reference flipped another 5%.
    - The port's router always used fp32. Against the fp32 reference it flips 0.2% (near-ties: logits within about 1e-3).
  - **Next:** the expert precision is the open quality/memory decision. Settle it with `teacher_forced` top-1 through the
    hybrid overlay. Candidates: bfp4 everywhere; gate/up bfp4 + down bfp8; bfp8 for the hot experts in the most
    sensitive layers.
- **2026-11-14, prefill and MTP.**
  - **Prefill.** One attention launch per chunk, using the work-unit SDPA (a unit is one item × one KV head, one core per
    unit). Per layer against the HF fixtures, prefill delta PCC equals decode: L0 0.99999, L2 0.99864, L5 0.99836.
    640 tokens × 3 layers take 0.06 s, against 3.2 s with rows groups.
    - Sliding-window rings now hold window + prefill chunk + one block, about 100 MB per die over 35 layers at
      2048-token chunks, because a chunk's launch needs every row's window live at once.
    - A fixed K chunk is required in prefill: per-unit dynamic chunk sizes misaligned the CB FIFOs.
  - **MTP** (`mtp_inkling_ref.py`, teacher-forced on 3 held-out texts).
    - Depth-0 drafts match the target's argmax at the next position 0.50 of the time, and match nothing at other offsets
      (about 0.02), so the alignment is verified.
    - The best inputs are the post-final-norm hidden and the backbone-normed embedding. Raw embeddings drop to 0.38,
      matching vLLM's comment.
    - The 8-text run gives an expected 1.42 tokens per pass at k=1 and 1.88 at k=8, and P(accept) falls fast with depth.
    - This is a teacher-forced figure on human text, where the target's own top-1 accuracy is about 0.5. vLLM quotes
      about 0.85 depth-1 acceptance, and DSpark 3.35 acceptance length at block 7, both on generation.
    - **The drafter choice needs a device spec-decode run on generated text.** The confident-position column of
      `mtp_inkling_ref.py` is a cheaper proxy.
- **2026-11-14, DSpark vs MTP, and the precision findings.**
  - **DSpark** (`dspark_inkling_ref.py`, the same teacher-forced method as MTP).
    - Measured on 4 sampled block starts per held-out text. The SpecForge import is satisfied by registering the
      checkpoint's own `dflash.py`.
    - Acceptance length by convention: raw embedding with logits /16 gives **1.31**; the other three combinations give
      1.06–1.16. Native MTP gives 1.42 at k=1 by the same method.
    - Both are depressed by teacher forcing on human text, and DSpark was trained on the NVFP4 target's hidden states.
    - Decision: MTP first, as planned. Settle it with on-device generation.
  - **Host int4 is more accurate than device bfp4.** On layer 40's cold experts and real inputs, expert-output PCC
    against bf16 is 0.988 for int4 g32 and 0.973 for emulated bfp4.
  - **Late MoE layers are much more sensitive to bfp4.** Layer 40 measures 0.983 / 0.989 / 0.9986 (bfp4 / gu4-dn8 / bfp8)
    against layer 5's 0.9915 / 0.9957 / 0.9993. Late layers carry massive activations, with hidden channel 1974 around
    250K.
  - **Per-layer sweep running.** `tools/sweep_moe_precision.py` writes `~/.cache/inkling/sweep_moe_precision.json`, the
    input to the per-layer precision assignment. Options: bfp8 in the sensitive layers, possibly with fewer slots,
    sending more misses to the more accurate int4 host path.
  - **Plan.**
    - At C=194 (2 shared + 192 routed): routed hit 94.4%, 13.4 misses per token, 211 MB per token from the host.
    - The int4 library for the 64 cold experts per layer is built: `~/.cache/hybrid-moe/inkling_int4_s194`, 38 GB.
- **2026-11-14, first end-to-end result through the hybrid overlay** (`tools/eval_hybrid.sh`).
  - Profile: all-bfp4 device experts, C=194 slots, int4 host library, no prefill streaming, eager decode
    (`HYB_FUSED=0`), on 2 held-out texts.
  - **Results:** top-1 0.874, confident top-1 0.951, kl32 0.141, NLL 2.359 against the reference's 2.349. Gates: 0.90 /
    0.95 / 0.10. M2.7 measured 0.848 / 0.933 / 0.196.
  - **Timing:** a 1000-token prefill takes 4 s once warm. The first model build takes 25 min (194 experts × 40 layers
    converted to bfp4 on the host); `HYB_WCACHE` would cache that.
  - **Per-layer emulation:** bfp4 error grows with depth, from 0.995 at L2 to 0.989 by L12–14 and 0.983 at L40.
    - int4 g32 (the host library) is about as accurate as gu4-dn8.
    - Next: a per-layer precision assignment within the measured DRAM headroom (`teacher_force.py` now prints it).
- **2026-11-14, 8-text bfp4 baseline (C=194).**
  - **Mean:** top-1 0.831, confident top-1 0.927, kl32 0.302. Device NLL is within 0.02–0.12 of the reference on every
    text.
  - **Spread by text:** livecodebench is worst (0.666 / 0.817 / 0.84), then doc_markdown and chat_english; code_mbpp is
    best (0.912 / 0.958).
  - **Per-span:** livecodebench's confident top-1 falls from 0.93 in the first 250 tokens to 0.72–0.83 after them. The
    other texts are roughly flat.
  - **The reference is sound.** Checked on trivial prompts: " Rome" at 0.986; the chat reply opens with
    `<|content_text|>` at 1.0. Its odd confident predictions on livecodebench (literal `\r\n`, LaTeX problem text) are
    the model's real behaviour on off-distribution raw text, and that text dominates kl32.
  - **Use the 8-text numbers as a relative measure between configurations.** For absolute quality, add free-running task
    evals (GSM8K / IFEval).
  - **DRAM after the build:** 27.1 GB allocated per die, 4.6 GB free (0.58 GiB × 8 banks). At C=194, bfp8 for a layer
    costs +0.6 GB per die and gu4-dn8 +0.2 GB per die.
  - **Emulated MoE PCC with bfp4, all 40 layers** (`~/.cache/inkling/emulate_moe_precision.json`): about 0.995 at L2,
    0.984–0.987 for L24–40, back to 0.994 at L41. bfp8 is ≥ 0.9999 everywhere; int4 is 0.993–0.998.
  - **Running now:** layers 24–40 with gu4-dn8 (+3.5 GB per die) plus a bf16 LM head
    (`doc/precision/late_gu4dn8_lmbf16.json`).
- **2026-11-14, calibration and where the drift comes from.**
  - **Noise floor** (`compare_refs.py`): the fp32-router reference against HF's bf16-router reference gives top-1 0.925,
    confident top-1 0.978, kl32 0.081 (livecodebench alone 0.82 / 0.93 / 0.26). The device's 0.831 / 0.927 / 0.302 is a
    real gap.
  - **Late layers don't help.** Layers 24–40 at gu4-dn8 (+3.5 GB per die) moved the 8-text mean only from 0.831 to 0.834
    top-1 and from 0.302 to 0.290 kl32.
  - **Drift map** (`tools/drift.py`, full hybrid model, math_gsm8k, against the gold residuals).

    | Layers | Cumulative residual PCC | Per-layer contribution PCC, on the device's own input (mean / p01) |
    |---|---|---|
    | 0–1 | 0.99999 | 0.99999 |
    | 2–11 | 0.9998 → 0.9995 | 0.997–0.999 |
    | 12–16 | falling to 0.9934 | from 0.993 to 0.966 (p01 0.84) |
    | 18–40 | 0.99 → 0.95 | 0.89–0.95 (p01 0.2–0.5) |

    - The same layers isolated on the gold input score 0.986–0.999. The step collapse therefore comes from the layers'
      sensitivity to small input drift: near-tie routing flips change whole experts.
    - Once drift builds in L12–16, it amplifies downstream. This is why the late-layer upgrade barely helped.
    - **The lever is the early MoE layers**, which keep the inputs to L12+ close. Running now:
      `doc/precision/early_gu4dn8.json` (L2–17 gu4-dn8, LM head HiFi2).
  - **Attention:** explicit HiFi2 + fp32 accumulation on its matmuls (`ttnn.linear`'s default config is lower fidelity).
    A small gain: L5 attention mean PCC 0.99954 → 0.99963.
- **2026-11-14 evening, precision profiles, first spec decode, and the verify-row bug.**
  - **Profiles** (8-text means, prefill path, top-1 / conf / kl32): early gu4-dn8 at C=194 0.842 / 0.933 / 0.287;
    **all bfp8 at C=106 0.862 / 0.945 / 0.218** (best; excluding livecodebench 0.885 / 0.960 / 0.158); all gu4-dn8 at
    C=150 0.847 / 0.937 / 0.262. More precision with fewer slots wins: the int4 host misses are more accurate than device
    bfp4. Non-expert paths at bf16 (`all_bfp8_mixers_bf16`) were *worse* (0.853 / 0.945 / 0.265; doc_markdown kl32
    0.215 → 0.357): one of those bf16 paths is suspect, bisect queued (`all_bfp8_kvccl_bf16`, `all_bfp8_weights_bf16`).
    The runtime's mixer fidelity is hifi2 | lofi only.
  - **Decode path = prefill path** (`EVAL_PATH=rows EVAL_ROWS=1`, 3 texts): 0.906 / 0.970 / 0.094 vs prefill
    0.912 / 0.972 / 0.098 on the same texts.
  - **Verify-row bug, fixed** (`1043efe8881`): in eager hybrid decode the host computes missed experts only for the
    first `HYB_DECODE_ROWS` rows (default 1, meant for padded batch rows). Spec-verify rows 1.. and the prompt's tail
    row groups lost every missed expert (at C=106 a large share of the 6 routed picks). Found with
    `tools/rows_consistency.py` (row 0 bit-exact, rows 1.. logits PCC down to 0.71) and `tools/rows_layer_diff.py`
    (first divergence: the first hybrid MoE layer, rows 1.. only). Inkling's env sets `HYB_DECODE_ROWS=8`; the overlay
    now warns once. After the fix: verify rows and rejected drafts are **bit-identical** to plain decode (PCC 1.00000,
    max diff 0, 3 prompts x 96 tokens). Every multi-row spec number before the fix (acceptance, tok/s) is void.
  - **Spec decode, eager, pre-fix** (for the record only): plain 3.7–4.2 tok/s; MTP k=2–3 6.3–7.6 tok/s at 2.1–2.3
    tokens/step greedy, 1.9–2.1 at temperature 0.7. Exact speculative sampling (`SPEC_TEMP` etc., M2.7 `SpecSampler`)
    runs end to end; the garble score found no real defects (flags were dictionary misses).
- **2026-11-15, traced spec decode working and exact** (profile all bfp8 C=106, LIB_S=106; 3 prompts x 128 tokens;
  `tools/spec_bench.py 3 128` with `SPEC_TRACE=1 SPEC_WARM=16`).
  - **Traced hang fixed** (`a01ab050179`): layers cache device tensors per row count on first use; one allocated after a
    trace capture can land in that trace's freed intermediates and is overwritten on every replay. Fix: one eager pass at
    every width (1..8) before ANY capture, then capture every verify width up front.
  - **Same bug, drafter side** (`076e81b74c9`): acceptance at the largest k of a run fell (k=3 1.77 vs eager 2.29) because
    the drafter's lazily cached tensors sat in the widest trace's region (R=K+1, replayed only at k=K). The drafter is now
    warmed at every width and depth before capture.
  - **Exactness:** greedy texts traced = eager at every k, and every k = plain k=0 (byte-identical).
  - **Greedy, tok/s (tokens per pass):** k=0 9.36 (1.00); k=1 **12.35** (1.77); k=2 11.78 (2.07); k=3 11.02 (2.26).
    Eager for comparison: 4.08 / 5.83 / 5.77 / 5.79.
  - **Sampling** (T=0.7, top-p 0.95, exact speculative sampling): k=0 9.20; k=1 **12.43** (1.80); k=2 11.88 (2.12);
    k=3 11.11 (2.31).
  - **Where a step goes** (k=0, ms): device wait `t_read` ~58 (~1.45 per layer), host experts `t_host` ~33 (80 int4
    experts per step, ~30 GB/s: bandwidth-bound, 8 host threads = 6), uploads ~9. Host experts grow with verify rows
    (80 / 138 / 186 / 228 per pass at k=0..3), which is why k>1 loses: the next lever is host-expert cost per verify row,
    then device time per layer (device profile with a 170 GB container cap: the first profiler run hit host OOM).
- **2026-11-15, decode optimisation round 1** (same benchmark; every change byte-identical across k within a run, quality
  gate re-run where numerics could move).
  - **Short conv** (`tt/sconv.py`): the device profile showed each conv's three shifted taps being tilized from one
    row-major row; a one-row fp32 tilize runs on ONE core (~65 us at the 1024-wide per-die shard). Taps now computed on
    row-major rows in the reference summation order, one tilize per conv: device wait per token 58 -> 44 ms. Gate after:
    0.865 / 0.947 / 0.219 (before 0.862 / 0.945 / 0.218, noise level). One prompt's greedy text flipped at a near-tie
    ("To manage this" -> "To resolve this"): fp32 summation order.
  - **MTP drafter traced** (`SPEC_DTRACE`, default on with `SPEC_TRACE`): draft 14.8 -> 8.3 ms per step at k=1.
  - **Logits untilized on the device before readback**: a host readback of a tile tensor moves all 32 padded rows
    (~4 ms per 201K-vocab bf16 row vs ~0.2 ms row-major). Draft 8.3 -> 5.3 ms, verify -3 ms.
  - **Greedy tok/s now:** k=0 11.17; k=1 **14.76** (1.72 tokens per pass); k=2 14.10; k=3 12.95.
  - **Host side is bandwidth-capped:** hostk int4 runs at ~36 GB/s, the box reads at ~46 GB/s max (4x64 GB DIMMs run
    at 3600 MT/s, rated 5600). 15.7 MB per int4 expert (20% of it scales / zeros). At k=1 host experts are ~55 of
    ~116 ms per step.
- **2026-11-15, decode optimisation round 2** (dev harness `tools/spec_bench.py 3 128`, traced, greedy; texts
  byte-identical across k within every run).
  - **Short-conv output cast before its tilize** (same values): k=1 14.76 -> **15.34** tok/s, texts identical to round 1.
  - **`HYB_HOST_ROUTE=1`** (hybrid, opt-in): host routing from the fp32 router logits; device wait 44 -> 33.5 ms per
    token. k=0 12.24, k=1 **15.93** (1.74 tokens per pass), k=2 14.97, k=3 13.97. Routing weights now fp32 on the host
    instead of bf16 on the device, so texts differ from device routing (2 of 3 prompts). Decode-path gate
    (`EVAL_PATH=rows EVAL_ROWS=3`, 8 texts): host route 0.862 / 0.947 / 0.215 vs device route 0.865 / 0.948 / 0.220
    (top-1 -0.003 at run-to-run noise, kl32 better): **kept**.
  - Device profile after round 1 (`spec_q106_devprof3`, die 0): ~1.12 ms per MoE layer (was ~1.48). Remaining big
    items per layer: the resident-expert sparse matmuls (~130 + ~50 us, real work), the attention kernel pair
    (~70 us), CCL all-gather / reduce-scatter groups (~15-20 us each, several), the router tail (now skipped by
    `HYB_HOST_ROUTE`). ~20 ms per token is still host round trips between trace segments (2 splits per MoE layer).
  - **Memory note:** the DIMMs run at 3600 MT/s (rated 5600, 4 x 64 GB); the host-expert half of a step is
    DDR-bandwidth-bound (hostk ~36 GB/s of ~46 GB/s measured). A BIOS memory-profile change is the owner's call.
  - Handoff to vLLM: see **vLLM serving plan**. Open decision for the owner: which plugin revision production serves
    on (the M2.7 spec shim matches `8b1f8b1`, not fork `main`).

- **2026-11-15, session 2: served through vLLM; the fast profile; 15.9 -> 45.5 tok/s (engine bench), quality at parity.**
  - **Profile change (the main speed lever).** Host int4 bandwidth decides decode: all-bfp8 C=106 misses ~80 experts per
    token, bfp4 C=192 ~17. Fast profile = routed bfp4, layers 2-17 gate/up bfp4 + down bfp8 (`early_gu4dn8`), 192
    slots (fused kernels need a multiple of 32: 2 shared + 189 routed + the zero sink), fused decode, sockets.
  - **Quality (teacher-forced, 8 held-out texts; M2.7 0.848 / 0.933 / 0.196):**

    | path | top-1 | conf | kl32 |
    |---|---|---|---|
    | fused prefill, device route, bf16 expert accumulation | 0.826 | 0.924 | 0.348 |
    | + host fp32 routing (`HYB_PREFILL_HOST_ROUTE`, auto for Inkling) | 0.836 | 0.926 | 0.285 |
    | + FP32 accumulation in the expert matmuls (`QWEN35_MOE_FP32_ACC=1`) | **0.866** | **0.947** | **0.213** |
    | fused decode path (rows of 3, host route) | 0.869 | 0.947 | 0.205 |
    | (all-bfp8 C=106, for reference) | 0.862 | 0.945 | 0.218 |

    The device top-k flips Inkling's near-tied selections (sigmoid + bias ~1e-3 apart); the routed expert matmuls
    accumulated in bf16 (tt-metal #49068 workaround: FP32 DST halves the tiles; the fix caps sub-blocks at 4).
    `QWEN35_MOE_DECODE_FIDELITY=hifi2` made no difference (0.827).
  - **vLLM serving** (plugin 23b0736, same as M2.7 production): `tt/generator_vllm.py` (engine-owned caches, vLLM's
    own Inkling multimodal processor registered on the class, text-only guard), `tt/inkling_serve.py` (the engine),
    `serve/serve_inkling.sh` (PROFILE=fast default), `serve/inkling_{smoke,features,long}.py`. Reasoning split, tool
    calls, clean streaming (block markers staged so vLLM's streamed content path does not leak `<|end_message|>`),
    exact sampling: all pass.
  - **Speed, engine bench (tools/engine_bench.py, 3 short prompts x 128, greedy; dev signal):** traced fused k=2 29.6
    -> sockets 36.1 -> + drafter prompt pass 36.7 -> + fused short-conv kernels **45.5 tok/s** (verify 39.5 ms, draft
    6.8 ms at k=2; k=3 40.3). TTFT ~0.5 s on short prompts. Served (before the conv kernels): ~31-33 tok/s.
  - **Device profile** (tools/layer_prof.py): elementwise / data-movement ops were ~35 of ~40 ms of a 4-row pass; the
    4 short convs per layer were ~15 ops each (75-130 us traced). `tt/sconv_kernels` (one generic_op program for
    rows, one for the commit): 19 / 16 us, state bit-exact, outputs within 1 bf16 ulp.
  - Other fixes: socket logits rows padded to whole tiles (258 + 32 = 290 failed the D2H PCIe alignment); the socket
    path without the PCIe fetch; prefetch for Inkling's router; prefill buckets + valid-row conv commits (no tail row
    groups); log-scaling tau in verify past 128000 positions; `INKLING_PREFILL_CHUNK=1024` (2048-row transients OOMed
    next to 192 slots + MTP).
  - GPTQ into the bfp4 grid (gptq_bfp_proto.py, calibration: ref_inkling REF_MLPIN, 21K tokens): layer 12 MoE PCC
    0.98968 -> 0.99249 (gu4dn8 0.99435). Not needed after the FP32-accumulation fix; kept as the next precision lever.
  - Not done: PCIe fetch / prompt streaming for the mixed-dtype profile (fetch.py keeps one dtype for all layers);
    long-context profiles beyond the tests below; multimodal.

- **2026-11-15, session 2 (cont.): prefill, the production server, and the open streaming hang.**
  - **Prefill is CPU compute-bound** (tools/prefill_bench.py): a 1024-token chunk sends ~40K token-expert pairs (12% of
    picks) to the int4 kernel, ~2 TFLOP, ~2.2 s of ~3.0 s per chunk; 8 or 16 kernel threads gain only 7%. Shipped:
    ~340-400 tok/s (the engine runs the 8-thread pool in prefill, 6 in decode).
  - **PCIe streaming of the missed experts** (implemented, opt-in `STREAM=1` / `HYB_STREAM_SLOTS`): the staging takes the
    first hybrid layer's format, so the device copies are gate/up bfp4 + down bfp8 in every layer
    (`HYB_DEV_POLICY=all_gu4dn8.json`); the prefill runs layer-major in groups of chunks (`INKLING_PREFILL_GROUP`) so a
    staged expert serves the group: 16K 1037 tok/s (128 slots), 32K 1001 tok/s (32 slots, 128K context); quality
    0.867 / 0.949 / 0.211. **Open bug, why it is not shipped:** a streamed long prefill (>= 1024 tokens), then traced
    decode, then a second streamed long prefill hangs the device in the second prefill (the host waits in
    `_route_prefill_host`'s readback). Reproduce outside vLLM: `PF_CAPTURE=1 tools/prefill_bench.py 4096,16384,32768`
    with the stream env of `~/inkling-work/jobs/pf9.sh`. Not the cause: HYB_STREAM_CQ1 (0 hangs too), prefetch
    banking, HYB_PINX_BGWRITE, the split. Never hangs: streaming without traces, one long prefill after traces, CPU
    prefill after traces (the server ran 1K .. 104K back to back).
  - **DRAM leak fixed:** each short conv kept its last prefill chunk's sequence (~10 MB per layer, 420 MB per die).
  - **Next steps:** (1) the streaming hang (likely an interaction between the socket decode traces and the staging
    path's programs; try HYB_SOCKET=0 with streaming, then a device-side hang dump); (2) a faster CPU expert kernel
    (AVX512-BF16 dot products: the activations are bf16 and int4 values are exact in bf16; ~1.5-2x on prefill);
    (3) the prefill short convs through a multi-tile fused kernel; (4) multimodal.
