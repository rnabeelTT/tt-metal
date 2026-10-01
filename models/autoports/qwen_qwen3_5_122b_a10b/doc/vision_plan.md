# Vision (image / video input) for Qwen3.5-122B-A10B on p300x2 - plan (2026-10-01)

Scope: serve `image_url` (and video) content parts through the vLLM `Qwen3_5MoeForConditionalGeneration` interface in
the production server (32 slots, MTP on). Reuse, in this order: (1) the in-tree qwen36 vision path (same model
classes we subclass: `models/demos/blackhole/qwen36/tt/vision/` tower = the Qwen3-VL ViT port, the trace-safe splice
`_set_vision_merge` / `_apply_vision_merge`, the per-request M-RoPE `_build_request_rope` + `rope_delta`), (2) the
qwen3.8-flash-next bundle's vision code (`~/qwen35-122b-bringup/references/qwen38_flash_next/{ttnn/vision*.py,
vision_splice.py, mrope.py, tools/qwen38_vision_inputs.py}`) for DRAM residency admission and image-input handling
ideas - its classes are tied to Qwen3.8 (hidden 2560) and its own server, so it is a reference, not a port source.

Checkpoint vision config: ViT depth 27, hidden 1152, 16 heads, patch 16, spatial merge 2, temporal patch 2, out
3072, `deepstack_visual_indexes` empty (no per-layer deepstack adds), image_token 248056 / video 248057, M-RoPE
interleaved sections [11, 11, 10] on the 0.25 partial rotary.

## What exists already in our path
- `Qwen35MoeModel.prefill_masked_bucket(..., vision_tokens, vis_row_offset)` calls `_apply_vision_merge` inside the
  bucket body (so a captured bucket trace contains the where-select over persistent buffers) and clears the splice
  mask per segment; with `vision_tokens` it currently FALLS BACK to the stock eager path (no persistent splice
  buffers are allocated on the mesh: qwen36 allocates them only on the single-device chunked trace).
- `tt/weights.py::StreamingStateDict.visual_keys() / visual_tensor()` expose the 333 `model.visual.*` tensors.
- The vLLM adapter registers `Qwen3VLMultiModalProcessor` with image / video limits 0 and raises on pixel inputs.

## Work items
V1 Tower on the mesh. Build the HF visual module STANDALONE (`Qwen3_5MoeVisionModel(config.vision_config)` with the
   streaming `model.visual.*` tensors; never `from_pretrained` of the 250 GB checkpoint) and hand it to
   `init_vision_model(reference_visual=..., vision_args=VisionModelArgs(...))` (the tower keeps the HF module for the
   patch embed / position interpolation on host). Unit test: TT tower vs the HF visual on a synthetic 448x448 image
   (PCC), weights bfp8; DRAM: ~0.9 GB bf16 -> ~0.45 GB bfp8 replicated or TP-fractured (as the qwen36 port does).
V2 Persistent splice buffers per bucket (`[1,1,bucket,dim/TP]` rows + `[1,1,bucket,1]` mask, TP layout) allocated
   BEFORE the captures; `prefill_masked_bucket` with `vision_tokens` stages them (`_set_vision_merge` with the
   segment's `vis_row_offset`) and replays the bucket trace instead of falling back; the per-request M-RoPE table
   (`_build_request_rope(token_ids, vision_tokens)` from the stashed image grid) feeds the bucket's cos/sin staging
   (`_rope_tp_cos_sin_torch` is already M-RoPE aware). Multi-segment images (placeholders spanning 2048-token
   segments) use the row offset as qwen36 does.
V3 vLLM adapter: real `get_supported_mm_limits`, `get_multimodal_embeddings` / `embed_input_ids` as the qwen36 class,
   per-request `pixel_values` + `image_grid_thw` out of the prefill kwargs (`_gather_user_visual`) ->
   `model.get_image_features` -> `vision_tokens` + offsets into `prefill_serving` for that slot; the MTP head's
   in-trace prompt rows are unaffected (they read the spliced hidden).
V4 Decode positions per slot: `prepare_decode_inputs_host` adds ONE scalar `rope_delta`; a 32-slot server needs a
   per-slot delta vector recorded at the end of each request's prefill (the device feedback loop advances rot idxs
   by one per step, so the delta survives); the MTP step's `d_rot / p_rot / r_rot` writes add the slot's delta too.
V5 Validation: HF `Qwen3_5MoeForConditionalGeneration` CPU reference on one image prompt (mmap-backed, background);
   served image request top-k agreement + caption sanity; the readiness suite with `limit_mm_per_prompt` set; the
   context contract and KV pool re-measured with the tower resident; bundle `tt-model.yaml` limits + smoke image.

## Order and gates
V1 (tower PCC) -> V2 + V4 on the reduced model (text prompts unchanged bit for bit; an image prompt runs) -> V3 served
on the reduced model -> V5 on the full model -> bundle.
