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
| 3. hybrid integration | done. **Shipped profile: early_gu4dn8** (routed experts bfp4; layers 2-17 gate/up bfp4 + down bfp8), 175 static device slots + 16 LRU fetch slots per layer, int4 host library for the rest. Teacher-forced 0.8726 / 0.9506 / 0.1998 with the shipped prefill config, at parity with M2.7 (0.848 / 0.933 / 0.196). (The all-bfp8 C=106 profile of the first sessions is superseded: see the 2026-11-15 plan.) |
| 4. spec decode | done and served: native MTP, k=2 drafts per verify pass in vLLM (`INKLING_SPEC_K`; verify, commits and drafter traced; greedy byte-identical to plain decode, exact speculative sampling with penalties). k=3 loses (host-expert cost per verify row). DSpark does not beat MTP (CPU reference). Prompt-lookup (suffix) drafts: see the 2026-11-17 session-6 entry. |
| 5. optimisation | done for this round. Decode: fused kernels, host<->device sockets, PCIe fetch of missed experts, traced verify / commits / drafter, device-untilized logits. Prefill: device streaming of missed experts, CPU/stream split, layer-major groups, pipelined hybrid layers, 1536 / 2048-token chunks, engine prefix cache. Bottlenecks now: decode = host int4 experts (DDR bandwidth); prefill = host dispatch (~206 ttnn ops per layer-chunk) + staging memcpy on the 8-core host. |
| 6. vLLM serving, package, multimodal | **published and served** `raahemnabeel/inkling-small-hybrid-qb2` (tt-model, profile p300x2-128k; package of 2026-11-19 session 9 with the prefill attention fix, image / audio inputs): short chat TTFT 0.48 s, 54.6 / 49.8 tok/s greedy / sampled; new prompts prefill ~1,430-1,610 tok/s from 15K to 126K (125.9K in 87 s); resent prompt 1.1-2.0 s, next chat turn 1.9-2.0 s (engine prefix cache); GSM8K-200 97.0%, IFEval-200 87.5 / 89.0; greedy output repeats exactly for the same request sequence; multimodal suite passes through the server. |

**Open items (2026-11-19, end of session 9).** The published package is the deliverable; these are what is left.
1. Prefill of new long prompts: ~1,410-1,570 tok/s through the server (session 9; session 7 ~1,300-1,380) vs M2.7's
   ~2,300. Bound by the staging copies (~3.9 GB per 2048-token chunk through the command queue): faster dispatch only
   moves the wait (session-9 entry). Levers left: zero-copy pinned writes (item 6), or more CPU experts with a CPU
   library as exact as the streamed bfp4 / bfp8 copies (int4 fails the gate; a bfp kernel costs 1.5-1.8x per expert).
   Speed is bimodal per process (~10%); measure with prefill_bench PF_SWEEP.
2. Multimodal: shipped (host-side towers; session 8 commits f46b4c271f5..69c81ae7745).
3. Prefix cache holds one entry (the last long prompt): interleaved users lose the ~2 s next-turn TTFT. Several
   snapshots cost host RAM for the rings (+ device buffers for the convs) per entry.
4. Speculative decode: k=2 MTP. Wider passes lose on host-expert cost per verify row; suffix drafts at k=2 give
   nothing (session 6). A suffix win needs wider verify widths used only on long copy matches (traces for widths up to
   ~8, HYB_ROWS / socket FIFOs and trace region sized for them, DRAM is ~0.7 GiB free at 128K).
5. close_mesh_device hangs at exit after fetch runs (dev tools exit early; tt-model stop / restart is clean).
6. Zero-copy pinned writes lose ~85% of writes (hybrid/pin/test_zerocopy.py); copy path used everywhere.
7. Not yet tested: sustained multi-hour load, several concurrent clients (single-sequence engine: requests queue).
8. Fixed (session 9): served greedy output repeats exactly for the same request sequence. A request can still depend
   on earlier ones (expert placement carried over; HYB_RESET_PER_REQUEST=1 removes it at ~16-18% speed).

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

- **2026-11-15, published.** `tt-model serve raahemnabeel/inkling-small-hybrid-qb2` (public; tt-metal b188a2b, plugin
  23b0736, Inkling-Small @8cc5877). Built from the worktree `/home/raahem/inkling-pkg` (full checkout; build
  `libhostk.so` there first: the manifest's verify step fails without it). Tested from a fresh first boot (int4
  library 5.6 min, device images ~25 min). Warm numbers through `tt-model serve`: TTFT 0.53 s and decode ~40-45 tok/s
  on chat prompts; prefill ~350 tok/s (28K: 79 s); needle found to 104K. Quality: teacher-forced 0.866 / 0.947 /
  0.213; GSM8K-200 97.0%; IFEval-200 88.0 / 90.0 (effort none). Two first-request stalls (75-85 s) after the fresh
  install did not reproduce on a warm server (likely one-time kernel compiles); worth a check on the next install.

- **2026-11-15, session 3: sampling, the streaming hang (root cause), faster CPU prefill kernel.**
  - **Host-sampled requests killed the server.** Requests the plugin samples on the host (min_p, logit_bias, bad
    words, allowed ids, min_tokens, logprobs, structured outputs) made it index the MTP step's token block as logits:
    `IndexError` in `_get_output_tokens`, EngineCore dead, every later request 500. Now `_host_sampled_decode`
    (tt/generator_vllm.py) runs one width-1 verify per step and returns the logits row; the engine adopts the plugin's
    token on the next step (`engine.step_logits`, `adopt_next`). Served: 23-26 tok/s on that path.
  - **Penalties were silently ignored** in the MTP step. tt/sampling.py (`InklingSampler`): presence / frequency /
    repetition penalties on the host, vLLM's formulas, inside exact speculative sampling (row i of a verify pass counts
    the drafts before it; the drafter's q is penalized the same way). tests/test_sampling_host.py checks the formula
    and the emitted distribution against plain penalized sampling.
  - **Structured outputs**: Inkling's tokenizer has no eos_token, so xgrammar guessed its stop token (200028); after a
    complete JSON value the grammar allowed only that, vLLM does not stop on it -> "grammar rejected tokens [200028]".
    The adapter gives xgrammar the model's end-of-turn ids (`_xgrammar_stop_ids`).
  - **The streamed-prefill hang, root cause:** a program compiled after the decode trace captures gets its kernel
    binaries allocated in DRAM while the traces exist, possibly inside a trace's freed intermediates; the replays then
    overwrite the kernels and the next launch of that program hangs. A long streamed prompt compiled the streaming
    path's programs (merged resident + staging call, rounds) after the captures; the second one ran them corrupted.
    Proof: one long streamed prefill BEFORE the captures (prefill_bench PF_PRE=4096) and the sequence that hung
    (4K -> 16K -> 32K with traced decode between) completes. The same hazard existed in the shipped CPU prefill: per
    prompt length, programs with data-dependent shapes compiled after the captures (found with INKLING_PROGLOG=1, a
    ttnn op wrapper that logs every compile after the captures):
    - the host experts' partial expanded with ttnn.embedding over a table of (rows with host results + 1) rows ->
      tables padded to powers of two, every size compiled in the warm-up (`warm_host_rows_dense`);
    - the short conv's prefill commit sliced the 3 state rows at the valid-row offset -> `sconv_kernels/commit_rm.cpp`
      (offset as a runtime argument: generic_op hashes only the runtime-argument count);
    - the last real row's logits slice (offset follows the prompt length) -> every 32-row offset compiled in warm-up;
    - streaming: the merged call / rounds / resident-only calls per bucket and layer dtype only run for some routings
      -> the warm-up runs prompts with every miss streamed and with none streamed, short (single staging) and long
      (prefetch banks), per bucket.
    Safety nets: after every prefill the engine compares the program cache with the count at the captures; if it grew
    it releases the traces, re-captures and re-runs the prompt (`_new_programs`, logged as a warning, capped by
    INKLING_MAX_RECAPTURES); tt/lazy_guard.py frees the prefill's lazily built device caches (dispatch constants, pairs
    masks) built after the captures. Validation (prefill_bench, PF_CAPTURE=1, INKLING_PROGLOG=1, prompts 300 / 777 /
    4097 / 9000 / 16384 with traced decode between): zero programs after the captures, no re-capture, no hang.
    Warm-up 74 s (was 9 s).
  - **CPU prefill kernel** (`hostk_moe_csr`, HYB_HOSTK_CSR=1 default): one call per layer step instead of ~60 16-row
    tiles; each 64-output block of an expert is dequantized once into a bf16 L2 buffer (scale and zero folded in),
    then every row of that expert runs against it with AVX512-BF16 dot products (4 rows x 64 outputs in registers).
    Exact against a float64 reference with the same bf16 rounding (1.3e-5); 0.33% from the fp32 kernel (bf16 weights
    and down-projection input, as on the device). Layer step 70 -> 31 ms (tests/bench_hostk_csr.py); quality gate
    0.868 / 0.950 / 0.209 (fp32 kernel 0.866 / 0.947 / 0.213). Falls back to the fp32 tiles without AVX512-BF16.
  - **Streamed prefill tuned on real text** (prefill_bench PF_TEXT = the held-out documents; synthetic ids
    concentrate the misses on a few experts and overstated it): a 1024-token chunk misses ~840 experts with few tokens
    each. CQ1 copies + prefetch banks (copying a layer's whole non-resident set) 478 tok/s -> CQ0, no banks 693 ->
    split cost HYB_PF_COPY_MS 0.65 -> 4 (more few-token experts on the bf16 CPU kernel, in parallel with the device)
    892 (6: 910, the device starts waiting on the CPU). Quality of the served setting 0.866 / 0.948 / 0.199 (tools/eval_hybrid.sh with the served env). C routing selection (hostk_route_inkling) in
    prefill.
  - **Published again** (`raahemnabeel/inkling-small-hybrid-qb2`, tt-metal 2e8209529fd), measured through `tt-model
    serve`: prefill 507 tok/s at 914 tokens, ~930-980 tok/s from 14K to 104K (104K: 112 s; the previous package ~300 s);
    chat decode 35.3 tok/s greedy / 33.9 sampled, TTFT 0.64 s; every sampling parameter, json_schema, needles to 104K.
    First boot builds the 46 GB of streaming copies in ~7 min (10 s per layer).
  - The retired "quality" profile (106 bfp8 slots) is gone from the README / serve script: the one configuration is at
    parity and ~3x faster.
  - **Open:** DSpark (DFlash-style drafter) vs MTP on generated text (the CPU comparison so far was teacher-forced on
    human text, 1.31 vs 1.42; DSpark's card claims 3.35 at block 7 on generation). Short-prompt TTFT 0.53 -> 0.64 s with
    streaming on (a streaming threshold by prompt length would recover it). tool_choice "required" (vLLM's inkling
    parser leaves the JSON calls in content).

- **2026-11-16, session 4: the decode hang (root cause, tt-metal fix), decode fetch, drafter / LM-head speedups.**
  - **Decode hang, root cause: a race in the CQ prefetcher, fixed in `tt_metal/impl/dispatch/kernels/cq_prefetch.cpp`.**
    A mesh trace buffer is sized for the longest device's trace, so a shorter trace is followed by padding pages that
    `paged_read_into_cmddat_q` may still be prefetching (NOC trid 1) when EXEC_BUF_END is processed. `process_exec_buf_cmd`
    returned without retiring them, so they could land on top of the next host commands fetched into cmddat_q, and
    the prefetcher then executed stale trace bytes as host commands (device 0 CQ0 stuck in relay_paged_packed /
    relay_inline). The fix retires trid 1 before returning (one barrier per replay). Socket decode queues the next
    host commands during a replay, so it hit the race (1 in 4 runs; 9 of 9 with the LRU fetch, which queues more);
    the watcher's slowdown hid it. Found with the prefetcher's cmddat_q dumped via exalens (`~/.cache/inkling/pfregs.py`:
    the stuck bytes were identical across hangs and absent from the host issue queue) and the host sysmem dumped from
    inside the process (`INKLING_MEMDUMP`, `hybrid/pin/memdump.c`). After the fix: every fetch + sockets run clean
    (20+). The package ships the patched source (JIT-compiled at startup; kernel caches rebuild on its hash);
    `tools/run_dev.sh` and `serve/serve_inkling.sh` mount it. Upstream this.
  - **Decode fetch (M2.7's PCIe expert copies)**: 175 static + 16 LRU slots per layer (S=175 libraries), fixed policy
    30% of a layer's misses, at most 2 per layer and step, copied on CQ1 and resident from the next replay; layers 18-41
    fetch their downs as bfp4 own-format copies (HYB_FETCH_DN_EXPERTS). Fetching more was slower (0.6/4, 1.0/8, auto
    budget 16, LRU 32 on S=159); auto (M2.7) ~ equal. Held-out prompts: host experts per step 47.8 -> ~20.
  - **LM head padded to tile-aligned vocab shards** (201024 -> 201088 columns; 50256 per die was half a tile short):
    every `_lm_head` all-gather had taken all_gather_async's composite path (untilize, 5 permutes, concat, tilize).
    Traced draft depth 3.17 -> 1.68 ms; verify faster too. MTP input projection as two matmuls (no concat to 2H).
    C routing (hostk_route_inkling) in decode.
  - **Measured** (vLLM, `serve/card_bench.py`: 8 held-out prompts x2, OSL 128, thinking at the template default):
    shipped package 38.9 / 35.7 tok/s greedy / sampled -> fetch 44.8 / 41.4 -> + LM head / drafter / routing
    **51.9 / 48.2** (M2.7's card on the same harness: 66.8 / 53.9). k=3 47.0 / 43.2 (k stays 2).
    Quality: prefill path 0.873 / 0.951 / 0.200; decode path (EVAL_PATH=rows, fetch 1.0/8 worst case) 0.866 / 0.948 /
    0.203 vs 0.870 / 0.951 / 0.203 without fetch (M2.7 0.848 / 0.933 / 0.196).
  - **DSpark (RadixArk/Inkling-Small-DSpark-Preview) does not beat MTP** (CPU reference, `dspark_inkling_ref.py`): on
    thinking-mode sampled text (the template default effort, temperature 1.0, top-p 0.95: its training setting) 1.75
    tokens per pass at block 7 / 1.86 at block 15, vs MTP 2.18 / 2.52 / 2.78 at k = 2 / 3 / 4. Every convention the
    checkpoint leaves open was tried (tap layers L vs L-1, residual / layer delta / RMS-normed / next-layer-normed
    features, embedding raw / normed, logits x1 / x1/16, Markov on / off); the best is residual taps, raw embedding,
    logits / 16 + Markov. Its card's 3.35 was measured against the NVFP4 target in SGLang; not reproducible here.
  - Tried, no gain: k=3, drafter weights bf16 (identical acceptance), per-die argmax (exact but 404 vs 451 us),
    HYB_PREFETCH off, decode host threads 4 / 8, DEVICE_PULL sockets (recv_async_h2d supports HOST_PUSH only).
  - **Open:** `close_mesh_device` hangs at exit after a fetch run (2 CQs + sockets; flush / sync / pinned release /
    socket release all complete first); serving never closes the mesh mid-run, but check a `tt-model serve` stop and
    restart. Remaining gap to M2.7 is acceptance (MTP ~1.7-2.0 tokens per pass on chat vs DFlash ~2.3) and 40 hybrid
    layers with a host step each (M2.7 at 32K: 18).
  - **Package (built from 6c823ce7497 + card), measured through `tt-model serve` at p300x2-128k:** short chat 48.2 tok/s
    greedy / 47.1 sampled (temperature 1.0, top-p 0.95, top-k 40), TTFT 0.54-0.59 s; prefill 572 tok/s at 3.8K, ~880-910
    tok/s from 15K to 118K (118K: 137 s), decode 41-50 tok/s at those contexts. GSM8K-200 96.5%, IFEval-200 89.0 /
    92.0; every feature check passes (reasoning split, tool call, streaming, json_schema, min_p, logprobs, penalties).
    `tt-model stop` shut down cleanly in 11.7 s and the next `tt-model serve` came up without a board reset (220 s
    warm). DRAM free after warm-up at 128K: 0.74 GiB.

- **2026-11-17, session 5: TTFT (prefix cache, pipelined prefill, longer chunks).**
  - **vs M2.7:** short-prompt TTFT already matched (0.54-0.59 s vs 0.60-0.71 s). Long prompts: M2.7 prefills at
    ~2300 tok/s and serves a resent prompt from vLLM's prefix cache in 1.1-1.8 s; Inkling had no prefix cache (a
    resent 118K prompt took 135 s) and prefilled at ~900 tok/s.
  - **Prefix cache** (tt/inkling_serve.py, INKLING_PREFIX_CACHE=1 default, INKLING_PREFIX_MIN 2048): engine-side,
    vLLM's stays off. The global layers' KV is position-indexed and kept; the per-sequence state at the last chunk
    boundary b before the prompt's end is snapshotted: the 4 short-conv states of every layer (+ MTP depth 0) copied
    into device buffers allocated before the captures (ttnn.copy, the reset program) when the layer's chunk ending at
    b runs (prefill_tokens on_layer; pipelined layers advance the attention-side convs ahead of the MLP conv, hence
    the "attn" / "mlp" parts), and the sliding rings read to host after the prefill (a ring holds window + chunk +
    block, so b's 512-row window is still in it). A prompt whose first b tokens match restores both and prefills from
    b. Invalidated by warm-up / captures (their eager passes write KV) and replaced by every prefill. Served: resent /
    next chat turn 1.3-2.0 s at 4K-118K. First-token top-5 logprobs fresh vs cached differ no more than fresh vs fresh
    (the CPU / device expert split varies run to run; top-1 identical in every check; serve/prefix_check.py).
  - **Prefill profile** (prefill_bench PF_PROF=1, cProfile of the main thread): the host is the bottleneck, not the
    device. Per 16K prompt: ~132K ttnn op calls (206 per layer-chunk, ~34 us of dispatch each: 8 s), the staging
    memcpy 3.5 s, from_torch uploads 2.1 s, reads 1 s. 8-core host: the CPU experts, the memcpy and Python share it.
  - **Pipelined prefill** (INKLING_PREFILL_PIPE=1 default, INKLING_PREFILL_AHEAD 2): a hybrid layer's chunks overlap.
    hybrid/patch.py splits the host-routed prefill MoE into _pf_pre (router logits + non-blocking reads of them and the
    MoE input + event), _pf_plan (select / decide / CPU experts) -> go() (_pf_device: uploads, resident + staged
    experts, shared) -> finish (_pf_finish: CPU rows, all-reduce); tt/layer.py prefill_pre / prefill_moe; tt/model.py
    _prefill_layer_pipelined queues chunk i + 2's attention before planning chunk i and completes chunk i - 1 after.
    Teacher-forced gate bit-identical (0.8726 / 0.9506 / 0.1998 = the shipped config). ~+10% alone (host-bound).
  - **Longer prefill chunks** (per profile in the package): 128K 1536 rows / group 10 (2048 OOMs: a 191 MB dispatch
    buffer; 2048 with 16 slots streams 3x more and is slower), 32K 2048 / group 8 (128 slots). Gate: 1536 = shipped;
    2048 0.867 / 0.946 / 0.205. Served (dev server): 128K profile 3.8K 4.2 s, 15K 12.0 s, 30K 23.5 s (1266 tok/s),
    63K 49.6 s, 118.6K 99.4 s (1193 tok/s; was 137 s); 32K profile 15K 11.2 s (1332 tok/s), 30K 23.8 s.
  - **Tried, no gain:** HYB_PF_COPY_MS 8 / 12 (fewer copies, more CPU: -2%), INKLING_PREFILL_GROUP 32, lookahead 4,
    HYB_PF_BGSTREAM=1 (first staging round copied by pinx's background writer on CQ1: implemented, off; the memcpy
    then competes with the CPU experts: 32K 1172 vs 1335 tok/s), zero-copy pinned writes (still lose ~85% of writes on
    every die after the prefetcher fix: hybrid/pin/test_zerocopy.py; the copy path 0 of 200).
  - **Remaining gap to M2.7's prefill** (~1.2-1.3K vs ~2.3K tok/s): host work per chunk. Next levers: fewer ops per
    layer-chunk (fused conv / MoE glue), traced attention segments, more device-resident experts (DRAM).
  - **Package (built from 21d47b7852f + card), measured through `tt-model serve` at p300x2-128k:** short chat 49.8
    tok/s greedy / 47.0 sampled, TTFT 0.53-0.54 s; prefill 862 tok/s at 3.8K, ~1,140-1,200 tok/s from 15K to 118K
    (118.6K: 104 s, was 137 s); resent prompts 1.3-1.9 s, next chat turn 1.5-2.0 s. GSM8K-200 96.0%, IFEval-200
    90.0 / 90.0; feature checks pass; DRAM free after warm-up 0.72 GiB; ready in 280 s.

- **2026-11-17, session 6: prompt-lookup (suffix) drafts from M2.7 (65f8c420c5f on raahem/qwen35-397b-hybrid-moe);
  status / open items refreshed.**
  - Ported: hybrid/suffix_draft.py (M2.7's SuffixIndex + Chooser, unchanged) and InklingEngine._device_step: the
    continuation of the longest earlier match of the context's last 3+ tokens replaces the k=2 MTP drafts (same
    verify width, point-mass q when sampling) when the request's per-match-length head-to-head tally favours it.
    INKLING_SUFFIX, **default 0**. Also a per-request log line ("Inkling request: tokens/step, suffix steps, verify +
    draft ms") written when the next request starts.
  - Served A/B (dev server, 128K profile, K=2, FETCH=1; jobs/sfx1.sh + ab_suffix.py: 8 workloads x greedy / sampled,
    800 output tokens: code rename / docstrings / small edit, a quote, an explanation, a summary, two chats): the
    suffix was chosen on **0 steps** in every request; tokens/step off vs on within noise (greedy: rename 2.13 / 2.09,
    docstrings 2.23 / 2.24, small edit 2.36 / 2.27, quote 2.32 / 2.24, explain 2.31 / 2.29, summary 1.99 / 1.95, story
    1.74 / 1.74, tech 2.03 / 2.04; outputs differ slightly run to run through the prefill's CPU / device split).
  - Why (offline replay of the recorded greedy outputs through the same lookup, jobs/sfx_sim.py): matches exist (both
    suffix drafts right on 41% of steps for the rename, 59% for the quote, 1-7% for chat), but MTP at k=2 already
    takes 2.1-2.4 tokens per step on exactly those workloads (max 3), so the head-to-head never favours the suffix.
    M2.7's gain comes from DFlash's 16-token blocks, where a long copy span is worth several extra tokens per pass. For
    Inkling that needs wider verify passes on suffix steps only (open item 4).
  - Not republished: the package's behaviour is unchanged (suffix off). Re-served the published package after the A/B.

- **2026-11-17, session 7: prefill host work (9e85a433d56).** Profiled with prefill_bench PF_OPS=1 (new: every ttnn op
  call by op / call site with its host time) and PF_PROF=1 (cProfile + who calls from_torch / to_torch), 16K tokens of
  held-out text, served 128K config (1536-token chunks, group 10). Baseline 14.81 s (1,106 tok/s), 199 ttnn ops per
  layer-chunk.
  - **Short conv in one program per chunk** (tt/sconv_kernels/pf_*.cpp, ShortConv.prefill_commit,
    INKLING_SCONV_PF_KERNEL default 1): the ttnn path ran ~15 ops per conv (65 of the 199). The kernel splits a column's
    row tiles over the grid (the carried 3 rows come from the state for the first tile, else from the previous tile),
    every argument common (the core's work item from its logical coordinates, so the descriptor costs the same for
    any core count), the new state rows written to a side buffer (no core reads what another writes) then copied into
    the state. The SFPU rounds the fp32 sum to bf16 (RNE) before the pack: the packer's own conversion differed from
    ttnn.typecast in ~0.1% of values (one bf16 ulp). tools/sconv_pf_test.py: output and state identical to the ttnn path
    and to the fp32 reference, padded last chunks included; host 390 -> 32 us, device 1000 -> 285 us per hidden-width
    conv. 14.81 -> 13.97 s.
  - **Per-chunk attention uploads shared by every layer** (attention._pf_upload: positions, KV-fill blocks, the global
    page table keyed by its contents; freed when prefill_tokens returns, so none outlives the prefill into trace
    replays) and **the MoE input untilized on the device before its non-blocking read** (HYB_PF_XRM default 1: the
    host untilize of [1536, 4096] was ~2.2 ms per layer-chunk). -> 13.14 s.
  - **SDPA prefill program cached per shape** (inkling_sdpa _PF_CACHE, INKLING_SDPA_PF_CACHE default 1: static parts
    + a custom program hash; tools/sdpa_cache_check.py identical on every prefill case). Within noise (13.07 s): by now
    the host waits on the device (record_event ~2 ms per call: the command queue full of staging copies queued behind
    compute; new _pf_pre timers t_pf_pre_ev / _rd / _ops).
  - Quality gate (128K prefill config, after the conv / uploads / readback changes): 0.8726 / 0.9506 / 0.1998, identical
    to the shipped config.
  - **Tried, rejected:** HYB_STREAM_CQ1=1 (staging copies on CQ1 to overlap compute) 13.76 s (slower); INKLING_PREFILL_AHEAD 3
    no change; HYB_PF_COPY_MS 2 (more streamed) 14.39 s; HYB_PF_COPY_MS 8 (more on the CPU) 12.85 s (+3-5%) but fails
    the gate: 0.8679 / 0.9476 / 0.2120 (int4 CPU experts are less exact than the streamed bfp4 / bfp8 copies).
  - **Package (built from 9e85a433d56), through `tt-model serve` at p300x2-128k:** short chat TTFT 0.43 s (was 0.53),
    49.8 / 47.1 tok/s greedy / sampled; prefill 3.8K 3.7 s, 14.9K 10.8 s (1,376 tok/s), 29.8K 21.5 s (1,383), 62.7K
    46.2 s (1,355), 118.7K 91.2 s (1,302; was 104.1 s); resent 1.3-1.8 s, next turn 1.4 / 1.8 s; feature checks pass;
    DRAM free after warm-up 0.72 GiB (unchanged).
  - **Package evals:** GSM8K-200 96.0% (same); IFEval-200 86.5 / 88.0, rerun 89.0 / 89.0 (earlier packages 90.0 / 90.0,
    89.0 / 92.0). Both new runs had doc 23 in a greedy repetition loop to the 4096-token cap. Investigated: served
    greedy decoding is NOT run-to-run deterministic (two runs of the same package share 7-12 of 200 IFEval
    responses; doc 23 x8 per server start: published package 0 / 1 / 1 loops of 8, new package 3 / 2 / 3, dev server
    at the published commit 2, at HEAD with every new flag off 3 and on 3), so the loop comes and goes with the
    prompt's near-tie. The new code changes no numbers: teacher-forced gate identical, SDPA old vs new module
    bit-identical on every decode / verify / prefill case (jobs/sdpa_decode_ab.py), conv kernel identical, flags-off
    as unstable as on. New flag INKLING_PF_SHARED_UPLOADS (default 1) for the shared attention uploads (bisect use).
  - **Where it stands:** 139 ops per layer-chunk; the main thread's remaining time per 16K: ttnn enqueue ~2.9 s, the
    staging memcpy ~2.7 s (memory-bandwidth bound, single thread; zero-copy would remove it but still loses writes),
    from_torch ~1.1 s (part of it waiting for command-queue space), host routing 0.7 s. Next levers: zero-copy (a
    tt-metal pinned-memory bug), fewer MoE glue ops (~40 per layer-chunk in _routed_sorted / _kernel_dispatch), the
    host-rows upload sharded over the dies (1/4 of the bytes; changes the reduction order: needs the gate).
  - **Open (decode):** the decode / verify fused conv kernel (sconv_kernels/compute.cpp) packs fp32 -> bf16 with the
    packer's rounding, so it likely differs from the ttnn path in ~0.1% of values (one ulp), as the prefill kernel did
    before its RNE typecast. Adding the same typecast changes decode outputs slightly toward the reference: measure
    before shipping.
  - **Open (served determinism):** greedy output through the server varies run to run (same prompt, fresh server or
    not). Not the host expert kernel (hostk.c: every thread owns its output columns, a fixed order). Candidates: the
    decode fetch (CQ1 copies "resident from the next replay"), the prefill CPU / stream split or the staging-slot reuse
    across requests (state carried between requests), socket-mode timing. Pre-existing (the published package does it);
    worth finding: a deterministic server would make evals comparable run to run.
- **2026-11-19, session 9: long-context prefill host work (43d6d35c52e..4ca5ab25fb6).** Default 128K profile config
  (2048-token chunks, group 8, 32 staging slots), 16K of held-out text, prefill_bench as served (PF_CAPTURE=1).
  - **Measure with in-process sweeps.** Prefill speed is bimodal per process (and sometimes switches within one):
    fast ~1,470-1,490 tok/s, slow ~1,330-1,360, every main-thread CPU step ~30% slower in the slow mode at steady
    clocks. Cross-process A/Bs are unreliable; prefill_bench PF_SWEEP (one repeat per value, interleaved; `HYB.<attr>`
    sets the hybrid state directly) gives values that repeat to +-0.5% within a process. Cause not found: not the main
    thread's SMT sibling alone (isolating it changes the mean by 0), not tt-metal's dispatch threads (below).
  - **Kept:** hostk_route_inkling single-pass top-k (routing -35%, same picks); HYB_PF_XPIN (default 1: the router
    logits / MoE input reads land in persistent host tensors viewed through DLPack, no allocation or copy per read);
    a fix to it (6a037826b95: the logits slot was never returned, one pinned [T, E] fp32 host tensor leaked per
    layer-chunk; never in a published package); prefill timers (t_pf_part1, t_pf_merge, per-repeat lines);
    PF_WATCHDOG; readback / upload microbenches. Quality gate on HEAD (2048 / group 8 / 32 slots): 0.8726 / 0.9506 /
    0.1998, identical to the shipped config. 16K: ~1,350 -> ~1,470-1,490 tok/s (fast mode).
  - **Where the time goes (PF_PROF / PF_OPS, 16K, 11.3 s):** staging copies (pinx write_at2, the copy path through
    the command queue) 2.5 s, ttnn op dispatch 3.3 s for 49.6K ops (148 per layer-chunk), from_torch 1.1 s, routing /
    planning ~1 s. Small ops cost 120-500 us and record_event ~2.5 ms because every mesh enqueue hands per-device work
    to tt-metal's dispatch thread pool and sleeps until it is done; the pool's workers are pinned to fixed cores
    (single NUMA node: round robin), which the CPU experts keep busy, so each enqueue pays scheduler wake-ups.
    TT_MESH_PASS_THROUGH_THREAD_POOL=1 (the main thread dispatches itself) or isolating the main thread cut that time
    (t_pf_pre_ev 540 -> 10, t_pf_part1 -35%) but t_stream grows by the same amount: **the ceiling is the staging
    copies (~3.9 GB per chunk through the command queue / host memory bandwidth)**, not dispatch.
  - **Rejected (measured):** prefill segment traces (the attention half of each hybrid layer captured once and
    replayed; work kept in ~/inkling-work/shelved): fixed two hangs (the check mode bypassed hybrid/seg.py's
    segmented-trace wrapper; the shared attention uploads stayed alive across replays) but logits drift beyond eager
    run-to-run noise (mean |d| 0.94 vs 0.17-0.20) and no speedup (the replay ordering serializes what the eager
    pipeline overlaps). Host-rows upload sharded over the dies: no change. Main-thread isolation, passthrough pool:
    no net change. **HYB_PF_COPY_MS:** 9 is +7.5% at 16K (in-process sweep; 4 -> 1,473, 9 -> 1,588) by putting more
    missed experts on the CPU, but int4 CPU experts fail the gate (11-17: 8 -> 0.8679 / 0.9476 / 0.2120); layers
    18-41 only at 14: 0.8697 / 0.9486 / 0.2021, also worse on every metric. **A CPU kernel on the device-format
    bytes** (bfp4 gate/up, bfp8 down, already pinned on the host; decoding verified bit-exact against ttnn.to_torch,
    scratch bench) would make that shift quality-neutral, but it reads 18.3 MB per expert vs 6.6 MB of int4 and costs
    1.5-1.8x per expert at 1-4 rows, which uses up the CPU's slack (~810 of ~1,330 ms per chunk at copy cost 9): not
    pursued on this host.
  - **Package (built from this commit + card), through `tt-model serve` at p300x2-128k:** short chat TTFT 0.47 s, 55.2 /
    50.9 tok/s greedy / sampled (session 7: 49.8 / 47.1; the session-8 decode work ships here); prefill 3.8K 3.7 s,
    14.9K 9.5 s (1,567 tok/s; was 1,376), 29.8K 19.9 s (1,495; was 1,383), 65.9K 45.3 s (1,456), 125.9K 89.3 s
    (1,410; 118.7K took 91.2 s); resent 1.1-2.1 s, next turn 2.1 / 2.2 s (was 1.4 / 1.8: 2048-token chunks redo up to
    2047 tokens); prefix check top-1 identical; multimodal suite and feature checks pass; GSM8K-200 96.0%, IFEval-200
    89.5 / 91.0. The first build failed on an Ubuntu mirror outage (apt in tt-model's runtime stage; retried).
  - **Served non-determinism, root cause (466e6fd01e1):** prefill attention freed its own relative-bias input.
    `rq = ttnn.pad(r, ..)` returns r's buffer in TILE layout ([.., 16] -> [.., 32] fits the tile), so
    `ttnn.deallocate(r)` freed rq, and the SDPA's output, allocated next, landed on it: the kernel wrote attention rows
    over the Rq tiles it was still reading, corrupting rows' bias depending on core timing. Wrong as well as
    non-deterministic on 1K-2K-row chunks (a 1500-token prompt's logits moved by up to 12 between identical prefills;
    the teacher-forced eval never hit it, its allocation order differs). Found by bisection: prefill_bench PF_DET
    (repeated prefills) with INKLING_PF_LAYER_HASH (first layer: 0, dense: not the hybrid), op hashes inside layer 0
    (first op: the SDPA, inputs identical), the SDPA launched twice in the model (differed 1 in 4) but never in
    isolation (tools/sdpa_det_check.py), replayed inputs matching neither model output, then a per-input clean-copy
    swap showing rq's memory changing after the launch. Fixed at all three pad sites (prefill, decode, verify).
    After: repeated prefills bit-identical; tools/determinism_check.py greedy output identical across processes.
  - **Per-request placement reset (d9bd5ab4131, default off since 879e7104f1a):** where an expert runs changes its
    numbers (device bfp4 / bfp8 vs CPU int4), and the LRU slots / staging carry over between requests, so a request's
    output depends on the ones before it. HYB_RESET_PER_REQUEST=1 restores the startup placement per request, but costs
    ~18% prefill / ~16% decode (tools/reset_ab.py: 1,241 vs 1,511 tok/s, 44.0 vs 52.2 tok/s), and the same request
    sequence already repeats exactly without it: off by default.
  - **Package with the fix (d27b89640c7 + card), through `tt-model serve` at p300x2-128k:** short chat 0.48 s, 54.6 /
    49.8 tok/s; prefill 3.8K 1,134 tok/s, 14.9K 1,434, 29.7K 1,612, 65.9K 1,454, 125.9K 1,444 (87.2 s); resent
    1.1-2.0 s, next turn 1.9 / 2.0 s; multimodal suite, feature checks, prefix check pass; GSM8K-200 97.0%, IFEval-200
    87.5 / 89.0; teacher-forced gate unchanged (0.8726 / 0.9506 / 0.1998).
