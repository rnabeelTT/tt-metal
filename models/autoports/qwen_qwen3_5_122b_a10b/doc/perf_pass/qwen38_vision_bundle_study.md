# Study: vision support for Qwen3.5-122B-A10B — the `qwen3.8-flash-next` bundle vs. our own inheritance chain

Read-only study of the vision-relevant files in the extracted `sjettTT/qwen3.8-flash-next_p150x4` bundle
(`vision_reference.py`, `mrope.py`, `vision_splice.py`, `ttnn/vision.py`/`vision_layout.py`/`vision_residency.py`,
`tools/qwen38_vision_inputs.py`, `docs/SERVER.md`, `chat.py`, `checkpoint.py`), cross-referenced against our own
`tt/model.py`, `tt/weights.py`, `tt/model_config.py`, `tt/generator_vllm.py`. The single biggest finding did not come
from the bundle: our `Qwen35MoeModel` subclasses `models/demos/blackhole/qwen36/tt/model.py::Qwen36Model`, which
already carries a complete, in-tree, vLLM-registered vision path (merged PRs #49531, #48861, #51556) for the same ViT
family. That changes the shape of this study from "port the competitor's device code" to "finish wiring plumbing we
already inherit."

## Vision tower architecture — identical ViT, confirmed a third time

The hypothesis holds, and now across three checkpoints, not two. Fetching `Qwen/Qwen3.6-27B`'s `config.json`
(`vision_config`) gives `depth 27, hidden_size 1152, num_heads 16, intermediate_size 4304, patch_size 16,
temporal_patch_size 2, spatial_merge_size 2, in_channels 3, num_position_embeddings 2304` — bit-identical to
Qwen3.5-122B-A10B and Qwen3.8-Flash-Next (already established). Only `out_hidden_size` tracks each text model's own
hidden size: 2560 (3.8-Flash-Next), 3072 (our 122B-A10B), 5120 (3.6-27B). Cosmetic label difference only: the
bundle's `vision_config.model_type` reads `"qwen4_exp"` (`vision_reference.py:98`); Qwen3.6-27B's reads `"qwen3_5"`.
Same tensor names and counts in all three: `VISION_TENSOR_COUNT = 333` / `VISION_TENSOR_BYTES = 897_862_112`
(`vision_reference.py:31-32`, `checkpoint.py:34-35` in the bundle) — our own checkpoint's `model.visual.*` prefix is
already confirmed to carry exactly that set. This is one shared ViT across the whole Qwen3.5/3.6/3.8 line, wired to
whatever text model sits behind it.

## What our own tree already has (not the bundle — `models/demos/blackhole/qwen36`)

`Qwen36Model` (our direct base class) ships:
- A generic TTNN vision tower, `DropInVisionTransformer` (`qwen36/tt/vision/model.py`), parameterized entirely off
  `hf_config.vision_config` (`qwen36/tt/vision/vision_model_config.py:36-49`: `dim = vision_config.hidden_size`,
  `head_dim`, `n_heads`, padded head dim to a tile multiple) plus `patch_merger.py` (row/column-sharded 2x2 merge +
  fc1/fc2, TP-aware) and `vision_attention.py`/`vision_block.py`/`vision_mlp.py`/`vision_distributed_layernorm.py`.
  It borrows shared code from `models/demos/qwen3_vl/tt/common.py` — this ViT was ported once, generically, for the
  Qwen3-VL family and qwen36 reuses it.
- M-RoPE already in `qwen36/tt/rope.py`: `build_request_rope(input_ids, image_grid_thw=, video_grid_thw=)`
  (`rope.py:153-187`) computes `get_rope_index` (3-axis positions + `rope_delta`) and stages a sequence-indexed
  cos/sin table; absent an image/video grid it is a no-op and callers get ordinary 1D RoPE
  (`rope.py:163` `if image_grid_thw is None and video_grid_thw is None: ...`). `mrope_section` comes straight off
  `rope_parameters` (`qwen36/tt/model_config.py:64`: `[11, 11, 10]` default) — the same values already confirmed
  identical across checkpoints.
- The full host-side splice: `get_image_features`/`get_video_features` (`qwen36/tt/model.py:217-305`) run the tower
  and return packed `[num_image_tokens, H]` rows; `_scatter_vision_tokens` / `_set_vision_merge` / `_apply_vision_merge`
  / `_vis_row_offset_for` (`qwen36/tt/model.py:392-507`) place those rows at the `<|image_pad|>` / `<|video_pad|>`
  positions two ways: an on-device scatter for the non-traced paths, and — for the traced/masked-bucket serving
  paths — a persistent `ttnn.where(mask, vision, x)` over buffers allocated once at warmup and updated per request by
  `copy_host_to_device_tensor` (never a per-request program compile, so a captured trace survives an image request).
  This is a materially different (and more house-appropriate) design than the bundle's own splice
  (`vision_splice.py`'s `-1`/`-0.0` sentinel-add trick, see below) and it is already the one our model would use.
- vLLM registration: `qwen36_vllm.py` imports **vLLM's own** `Qwen3_5ProcessingInfo`, `Qwen3VLDummyInputsBuilder`,
  `Qwen3VLMultiModalProcessor` from `vllm.model_executor.models.qwen3_5` (`qwen36_vllm.py:19-24`), registers via
  `@MULTIMODAL_REGISTRY.register_processor(...)` + `SupportsMultiModal` (`qwen36_vllm.py:45-48`), and
  `_compute_vision_tokens`/`_has_visual`/`_gather_user_visual` (`qwen36_vllm.py:156-201`) pull `pixel_values` /
  `image_grid_thw` (or the video equivalents) straight out of the vLLM request kwargs — i.e. vLLM's own HF-processor
  plugin already did the image decode/resize/patchify before the model ever sees it. One modality per request is the
  documented limit (`get_supported_mm_limits` returns `{"image": 1, "video": 1}`, `qwen36_vllm.py:37-42`), because the
  splice keys off one placeholder token id at a time.

## Cross-reference: what our own autoport already carries, dormant

This directly answers the task's "how deliberate was this" question — very:
- `tt/weights.py:19-24, 214-219`: `StreamingStateDict` explicitly keeps `model.visual.*` reachable
  (`visual_keys()`/`visual_tensor()`) with the comment "stays reachable ... **for the later vision enablement**".
  Not silently dropped, not erroring — a named, intentional deferral.
- `tt/model.py:415-465` (`prefill_masked_bucket`, our stage-6/7 fast path override with the persistent KV-scratch
  block and captured bucket traces) already threads `vision_tokens`/`vis_row_offset` parameters through, and its
  docstring says outright: "Falls back to the stock path for multimodal segments" (`model.py:428`). The guard is
  `if vision_tokens is not None or ... : return super().prefill_masked_bucket(...)` (`model.py:432`) — i.e. a vision
  request today would take the slower, inherited `Qwen36Model` path rather than our fast masked-bucket path, and
  that's a deliberate, already-coded fallback, not a gap someone forgot to close.
  every other prefill entry point we did NOT override (`prefill_traced_chunked`, `prefill_paged`,
  `prefill_layer_chunked` in `qwen36/tt/model.py`) already accepts and threads `vision_tokens` generically.
- `tt/generator_vllm.py:413-414`: `if self._has_visual(kwargs, "pixel_values") or self._has_visual(kwargs,
  "pixel_values_videos"): raise NotImplementedError("text-only serving: this bringup does not serve images or
  videos")`. `_has_visual` is inherited from `qwen36_vllm.py:156` — our generator subclasses the same vLLM wrapper
  lineage. This is the single explicit off-switch; removing it (and the analogous `MULTIMODAL_REGISTRY` registration)
  is most of the vLLM-side integration surface.
- `tt/model_config.py` / `tt/precision.py`: neither mentions `vision`/`visual` at all today — no vision weight group,
  no vision dtype policy. `Qwen36ModelArgs` (our base) does already carry `mrope_section`, `image_token_id`,
  `video_token_id` off `hf_config` (`qwen36/tt/model_config.py:64,68-69`), so those fall out for free once our HF
  config is read the normal way (it already is).

## What's near-verbatim copyable (from the bundle)

- `vision_reference.py`'s pure-torch FP32 reference (patch embed, position interpolation, 27 blocks, merger) is a
  clean, dependency-free spec of the math and the checkpoint's tensor names/shapes (`vision_tensor_names`,
  `vision_tensor_shapes`). Useful as an independent numerics oracle for the `DropInVisionTransformer` path above, or
  to unit-test our converted weights — the only delta needed is `out_hidden_size` 2560 -> 3072 and the checkpoint
  "qwen4_exp"/"qwen3_5" `model_type` string check (`vision_reference.py:98`) relaxed to accept ours.
- `mrope.py`'s constants (`IMAGE_TOKEN_ID 248056`, `VIDEO_TOKEN_ID 248057`, `VISION_START/END_TOKEN_ID`,
  `ROPE_THETA 10_000_000.0`, `MROPE_SECTION (11,11,10)`, `ROPE_DIM 64`) all match our checkpoint bit-for-bit (already
  established) and match what `qwen36/tt/rope.py` already derives from `hf_config`. The file itself is redundant
  with the inherited `rope.py::build_request_rope`/`get_rope_index` machinery — not something to port, just a good
  cross-check of the formulas (same `mrope_section`/interleaved layout).
- `tools/qwen38_vision_inputs.py`'s `smart_resize`/pixel-normalization constants (`image_mean/std 0.5`, `min_pixels
  65_536`, `max_pixels 16_777_216`, `LOW_DETAIL_MAX_PIXELS 512*512`) match the checkpoint's `preprocessor_config.json`
  and are the same `Qwen2VLImageProcessorFast` rule vLLM's `Qwen3VLMultiModalProcessor` already runs upstream for us
  — this file's *logic* is portable in spirit but not something we need to write: vLLM's registered processor does
  it before our generator ever sees `pixel_values`.

## What needs a rewrite / new work — and it is not mainly the bundle's device code

- The bundle's `ttnn/vision.py`/`vision_layout.py`/`vision_residency.py` (device-side tower, eager ops, row-bucket
  residency ladder, ASGI-server admission control) is written for a single-stream, non-vLLM server with its own
  program-cache/DRAM-admission discipline (`vision_residency.py:1-30`, `VISION_ROW_BUCKETS` power-of-two padding to
  avoid request-time compiles). None of that residency machinery applies under vLLM, which owns its own scheduler
  and batching; and we already have a device tower (`DropInVisionTransformer`) in our inheritance chain, so this is
  not the thing to port. The bundle is also a materially different split: its patch embed + position-table
  interpolation run **on device** (`vision.py::embed_patches`, `run_image`), where qwen36's tower instead needs an
  **HF reference torch submodule** for those same steps (`qwen36/tt/model.py:217-223`: "It still needs the HF
  reference visual for the patch embed / positional-interpolation steps that are not ported to TT"). That's a real
  gap worth closing eventually (host-CPU patch-embed/pos-embed is extra host latency per image versus the bundle's
  fully-on-device version), but it is orthogonal to bringing vision up at all.
- Real new work, scoped to *our* model:
  1. Instantiate `VisionModelArgs`/`DropInVisionTransformer` against our own checkpoint (`out_hidden_size=3072`, our
     `(1,4)` mesh) and prove it loads against our `StreamingStateDict.visual_tensor()` (or, if `reference_visual`
     truly needs an HF `nn.Module`, prove a small transformers `AutoModel` instantiation of just the ~900 MB vision
     submodule is acceptable under our host-RAM constraints — the whole-checkpoint streaming design in `weights.py`
     exists specifically because the *text* model is 250 GB; the vision submodule alone is not).
  2. Call `init_vision_model()` once at model build, wire `get_image_features()`/`get_video_features()` into our
     generator's prefill path, and delete the `NotImplementedError` in `generator_vllm.py:413-414`.
  3. Register the vLLM multimodal processor on our model class the way `qwen36_vllm.py:36-48` does — check first
     whether vLLM's `vllm.model_executor.models.qwen3_5` module already covers HF `model_type "qwen3_5_moe"` (ours)
     or only the `"qwen3_5"` string qwen36 answers to; if not, a thin `ProcessingInfo` subclass is the fix, following
     `TT_Qwen3_5ProcessingInfo` (`qwen36_vllm.py:36-42`) as the template.
  4. Decide whether a vision request is allowed to fall back to our slow/stock masked-bucket path forever
     (`model.py:432`'s existing guard already makes that safe, just slow) or whether the fast KV-scratch bucket path
     needs the same `_set_vision_merge`/`_apply_vision_merge` splice wired through it — unverified cost without
     running it; "slow-path-only for images" may be an acceptable first cut given vision is a minority of traffic.
  5. Precision/config: add a `vision` weight group to `precision.py`/`model_config.py` (today absent) if the vision
     tower should have its own dtype policy rather than inheriting whatever default `dtype=ttnn.bfloat8_b`
     `init_vision_model()` defaults to (`qwen36/tt/model.py:217`).

## Server/plumbing — vLLM already owns almost all of this layer

The bundle's `docs/SERVER.md` vision section is long (image_url parts, `detail` levels, the eight DRAM row buckets,
42-44s prewarm cost, `vision_unavailable` HTTP 400, video refused in every spelling) but almost none of it is ours to
build: it is the bespoke ASGI server's own admission control and HTTP surface, not something vLLM needs re-litigated.
Under vLLM:
- Image ingestion, `detail` handling, and the pixel-bounds/token-count accounting are vLLM's `Qwen3VLMultiModalProcessor`
  job (via the HF `AutoProcessor`), already demonstrated end-to-end by `qwen36_vllm.py`'s registration — not a file we
  write.
- "Video not supported" and "one modality per request" are choices we would make the same way qwen36 did
  (`get_supported_mm_limits` returning `{"image": 1, "video": 1}` — or `{"image": 1}` only, if we want to defer video
  specifically, which the user's framing suggests: text-only was deliberate, vision-only-no-video would be a
  reasonable next increment).
- What the bundle's residency section *does* usefully warn about, in spirit: warm-up/compile cost for a new device
  program shape. Our traced/captured-bucket serving path will need its own warm-up hook analogous to
  `vision_warm_hook` (`vision_residency.py:370-389`) if the fast path ever needs to prewarm a vision-tower program
  shape before serving — but this is a much smaller problem for us than for the bundle, since `DropInVisionTransformer`
  already exists as a captured/TT-idiomatic module rather than a bespoke eager-ops tower with its own bucket ladder.
- Digest/prefix-reuse keying (`vision_splice.py`'s `image_digests`/`Qwen38VisionPrompt.spans`) is vLLM's own
  prefix-caching problem once registered as multimodal, not ours to reimplement — though note
  `generator_vllm.py:405` already documents that *this* autoport doesn't support chunked/cached-prefix prefill at
  all (`NotImplementedError` above `start_pos != 0`), so prefix reuse for images is moot until that lands regardless.

## Effort estimate

The MTP study bucketed ~60% blueprint / ~20% near-copyable / ~20% new work. Vision looks different — more is
*already present*, just not switched on:

- **~35% already done, just needs enabling**: the ViT port, M-RoPE, and host-side splice all exist in our own
  inheritance chain (`Qwen36Model`, `qwen3_vl` common code) and are exercised today by `qwen36`'s own vLLM path.
  Wiring `init_vision_model()`, deleting the `NotImplementedError`, and registering the vLLM processor is mostly
  plumbing, not invention.
- **~30% adapt**: proving the inherited tower against our checkpoint's `out_hidden_size`/mesh shape, resolving the
  `reference_visual` weight-loading question against our streaming-checkpoint convention, confirming vLLM's
  `qwen3_5` processor module actually answers for our `model_type "qwen3_5_moe"`, and precision/config wiring
  (`precision.py`/`model_config.py` currently have zero vision awareness).
- **~20% genuinely new**: extending our fast KV-scratch masked-bucket/traced prefill path to carry the splice (today
  it deliberately punts to the slow stock path on any vision request), plus whatever GDN-hybrid-specific interaction
  the masked-bucket fallback needs proven correct (GDN state reset timing, `_build_request_rope` interaction with
  our own `_reset_gdn_state_for_new_sequence()`, `model.py:453-454`).
- **~15% out of our control / bundle-not-applicable**: the bundle's bespoke server admission/residency ladder and
  ASGI plumbing, which does not transfer to vLLM at all and was mostly useful here as a design cross-check, not as
  code.
- The bundle itself (`vision_reference.py`, `mrope.py`) contributes mainly as an **independent numerics oracle** and
  a confirmation that the three checkpoints' vision towers and M-RoPE constants agree — genuinely useful, but a
  small slice of the total effort compared to what our own `qwen36` sibling already provides.

## Open questions (need running code, not just reading)

- Is `qwen36`'s vision path validated end-to-end on Blackhole beyond `demo/vision_demo.py`? The PR titles
  (#49531/#48861/#51556) suggest it shipped and was optimized, but this study did not run anything, so numerics/perf
  for *our* checkpoint's `out_hidden_size=3072` variant are unverified.
- Does `init_vision_model(reference_visual=None)`'s fallback (`VisionModelArgs.reference_vision_model`,
  `qwen36/tt/vision_model_config.py:139`) actually instantiate a full HF `transformers` vision submodule via
  `from_pretrained`, and if so, does that conflict with our "never materialize the full checkpoint" streaming
  convention (`weights.py` docstring) — or is the vision submodule (~900 MB) simply small enough that it doesn't
  matter? Unverified without running it against our checkpoint directory.
  - Related: can `StreamingStateDict.visual_tensor()` feed that reference module directly (bf16 tensors, right
    names) instead of a `transformers.AutoModel` load, avoiding a second weight-loading convention entirely? Not
    something this read-only pass could test.
- Does `vllm.model_executor.models.qwen3_5`'s `Qwen3_5ProcessingInfo` (used by `qwen36_vllm.py`) work unmodified for
  our HF `model_type "qwen3_5_moe"`, or does the processing-info class key off `model_type` in a way that needs a
  thin override? Would need to read the installed vLLM's `qwen3_5.py` (outside this study's file list) to confirm.
- Cost of the deliberate fallback in `model.py:432` (vision request -> slow stock `Qwen36Model.prefill_masked_bucket`
  instead of our fast KV-scratch path) is unmeasured; whether that's acceptable for a served vision request's TTFT
  is a performance question, not a reading one.
- Numerics gap between the bundle's fully-on-device patch-embed/position-interpolation and `qwen36`'s host-CPU
  version for those same steps: both should be mathematically equivalent (same formulas per `vision_reference.py`),
  but nothing here proves the `DropInVisionTransformer` path matches to the PCC bar our other components hold to —
  needs the actual PCC harness (`debug=True` path already exists at `qwen36/tt/model.py:309-312`) run against our
  checkpoint.
- Whether vision plus our hybrid GDN/MoE decoder layers interacts correctly at all (GDN recurrent state reset timing
  around `_build_request_rope`, MoE routing on image-token rows) is unverified — the bundle's own MoE/MTP files were
  explicitly out of scope for this study, and the inherited splice was written against `Qwen36Model`'s plain
  attention + dense/MoE-FFN layers, not against our specific hybrid layer stack.
