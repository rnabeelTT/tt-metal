# Inkling-Small on one QB2: hand-off log

Read this first when resuming. Newest entries at the bottom of **Log**. Branch: `raahem/inkling-small-hybrid` (fork
`rnabeelTT/tt-metal`), cut from `hybrid-moe-starter`. Approach and gates: see **Plan** below; the M2.7 bring-up this
follows is `ADD_A_MODEL.md` + `HANDOFF.md`.

## Status (update every session)

| Phase | State |
|---|---|
| 0. env / disk | done. 729 GB were free after the Docker prune; `libhostk.so` is built; the BF16 checkpoint is downloading. |
| 1. CPU reference + routing | `ref_inkling.py` is self-tested, and it records routing, top-32, gold residuals, taps and `final_hidden`. Text sets are built. It is queued to run once the download finishes. |
| 2. TT port | decode path written: attention, layer, MoE router, model, weights. The SDPA kernel passes on device (PCC 0.9996–0.9998 vs fp32, decode and 8-row verify, up to 128K). sconv passes on device. Layer tests against HF fixtures are pending: they need weights. |
| 3. hybrid integration | overlay, int4/dev builders, plan and sim support for Inkling are written but not run. Next: `make_inkling.sh plan int4 dev`, then `tools/teacher_force.py` with `HYB_ENABLE=1`. |
| 4. spec decode | rows mode (verify) is written. `mtp_inkling_ref.py` (CPU acceptance of the native MTP) is written. DSpark is not started. |
| 5. optimisation | not started. The prefill kernel (band-bias chunked SDPA, see Plan) is the first item. Prefill currently runs as rows groups of 8: exact but slow. |
| 6. multimodal, package | not started |

**Next steps, in order:**
1. Layer device tests on the fixtures. Commands are below; the gate is delta PCC > 0.995 mean.
2. `ref_inkling.py` on the train set, then `make_inkling.sh plan int4 dev`.
3. `teacher_force.py` through the hybrid overlay. This is the quality gate: top-1 ≥ 0.90, confident top-1 ≥ 0.95, kl32 ≤ 0.10.
4. `mtp_inkling_ref.py`: decide how many MTP depths to run.
5. The prefill kernel.

### Design decisions taken (and why)

- **Attention bias as tile matmuls.** In decode and verify, `bias = Σ_rows Rq_row @ T[phase][m]`, where T is a pre-tilized
  per-layer table (`tt/relbias.py`). The data-movement cores do no per-element work, and verify rows each get their own
  phase. The QK matmul adds the bias block, with the causal/window mask folded into it.
- **Sliding-window layers use a ring KV through the page table.** Logical block b maps to physical block b mod n_ring,
  where n_ring = 512/64 + 1. The SDPA reads only the window, through the stock helper's window workload. KV for the 35
  sliding layers is constant (about 1 MB per layer per die) at any context length.
- **The 2 shared experts are experts 256 and 257.** Inkling's router weights them in the same 8-way normalisation as the
  6 routed picks, so the MoE is "8 weighted experts of 258".
  - The shared experts are pinned resident in the hybrid plan.
  - route_scale × global_scale is folded into every expert's down projection (`tt/weights.py expert_scale`), so router
    weights sum to 1 everywhere: device router, host routing, crafted rows, int4 library and device images.
  - `tests/test_router_host.py` proves this equals HF `InklingMoE`.
- **Model-level folds at load** (`tt/weights.py`): the embedding table carries embed_norm, the final norm carries the
  1/16, and the dense layers' down projection carries their global_scale.
- **One ranking for the hybrid.** `plan_from_routing.py <routing> <out> <sim plan>` embeds the sim's `order`, and the
  overlay uses it as is. The resident set therefore equals the complement of the int4 library's cold set.

## The model, as it matters for the port

From `config.json` and transformers 5.17 `models/inkling/modeling_inkling.py` (the reference we match):

- **Shape:** 42 layers, H 4096, vocab 201,024 (only the first 200,058 logits are real).
  - Attention: 32 Q heads, 8 KV heads, head_dim 128.
  - **Dense MLP** in layers 0–1: I 16384, times a learned `global_scale`.
  - **MoE** in layers 2–41: 256 routed experts (I 2048, top-6) plus 2 shared experts (I 2048).
- **Router**, 258 rows (256 routed + 2 shared):
  - Selection: `sigmoid(logits[:256]) + bias`, top-6.
  - Weights: `softmax(logsigmoid([6 chosen logits, 2 shared logits]))` × 8 (`route_scale`) × `global_scale`.
  - The last 2 weights are the shared-expert gammas, multiplied inside the shared SwiGLU before `down`.
- **Attention has no RoPE.** Layer types:
  - `hybrid_sliding` layers (window 512): all except 5, 11, 17, 23, 29, 35, 41.
  - `hybrid` (global) layers: those 7.
- **Attention internals:**
  - q and k are RMS-normalised per head (after k's sconv); the scale is 1/head_dim.
  - **Relative bias** `bias[h, q, k] = (r_proj(x_q).view(heads, 16) @ proj[16, E])[h, q-k]` for 0 ≤ q-k < E, and 0 otherwise.
  - E = 512 for sliding layers (the whole window) and 1024 for global layers.
  - **Global layers only:** q and the bias are multiplied by `tau = 1 + 0.1*ln(max(1, (pos+1)/128000))`. This is a no-op below 128K.
- **sconv**: `y = x + causal_depthwise_conv(x, w[C, 4])` in fp32. There are 4 per layer:
  - `k_sconv` and `v_sconv` (C = 1024), applied right after the k/v projections and before k_norm and the KV cache;
  - `attn_sconv` on `o_proj`'s output, before the residual add;
  - `mlp_sconv` on the MLP/MoE output (shared experts included), before the residual add.
  - Decode state is the last 3 inputs per conv.
- **Embedding:** `embed_norm(embed(ids))` (RMSNorm). `embed` and `unembed` are separate tensors.
- **Output:** `final_norm(x) / 16 @ unembed.T`, truncated to 200,058.
- **Checkpoint layout:** keys are `model.llm.layers.N.{attn.wq_du, wk_dv, wv_dv, wr_du, wo_ud, ...}`.
  - Gate/up pairs are **row-interleaved**: `w13`, with gate on the even rows. See `ref_inkling.deinterleave`.
  - The sconv weights are `[C, 4]`.
- **MTP:** `mtp.safetensors`, 8 chained layers, each a full block (attention + sconv + dense MLP). Layers 1 and 3 are global, the rest sliding.

Consequences for the port:
- **KV is small.** Only 7 layers hold full KV, about 15 KB/token in bfp8, so 1M tokens is about 16 GB.
- **The new device work:**
  - the band-limited relative bias, in decode, verify and prefill;
  - 4 stateful fp32 convs per layer, which spec verify must roll back;
  - the router normalisation;
  - the shared experts.

## Traps found (read before touching the reference or the config)

- **transformers 5.17 resolves the expert width wrongly.** `InklingTextConfig` moves the checkpoint's
  `dense_intermediate_size` (16384) into `intermediate_size`, overwriting the checkpoint's `intermediate_size` (2048, the
  expert width). It then leaves `moe_intermediate_size` at its class default of 3072, so HF modules built from that
  config have wrong-shaped experts.
  - The real shapes are `w13 [256, 4096, 4096]` and `w2 [256, 4096, 2048]`.
  - `ref_inkling.config()` and `tt/weights.load_config()` pin `moe_intermediate_size=2048, intermediate_size=16384`.
- **The serving image's transformers (5.12) has no Inkling.**
  - `tt/hf_config_inkling.py` vendors the 5.17 config classes and `register()`s them with AutoConfig.
  - It also extends `ALLOWED_LAYER_TYPES` with `hybrid_sliding`, which 5.12's validator rejects otherwise.
  - Device tests compare against host-made fixtures (`fixtures_inkling.py`), not live HF modules.
- **sconv weights are `[C, 1, 4]`** in the checkpoint, not `[C, 4]`.

## Environment notes

- The HF cache is at **`HF_HOME=/opt/tenstorrent/models`**, not `~/.cache/huggingface`.
  - The starter's docker scripts mount `~/.cache/huggingface`; mount `/opt/tenstorrent/models` instead.
  - Host scripts need `HF_HUB=/opt/tenstorrent/models/hub`.
- **CPU venv** (torch 2.9 cpu, transformers 5.17): `~/inkling-work/venv`.
- **Sparse checkout:** `git sparse-checkout add <dir>` before using a directory that isn't present yet.

## Commands

```bash
cd models/experimental/hybrid_moe
export HF_HUB=/opt/tenstorrent/models/hub PY=~/inkling-work/venv/bin/python
# text sets (done; 8 held-out + 21 train, 1000 tokens each, Inkling chat template)
TT_SRC=$PWD/../../autoports TOK_REPO=thinkingmachines/Inkling-Small $PY quality_texts.py runs/inkling/texts_held.json
TT_SRC=$PWD/../../autoports TOK_REPO=thinkingmachines/Inkling-Small $PY quality_texts.py runs/inkling/texts_train.json train
# the reference's chunked path vs HF's full forward (tiny random model, no checkpoint)
$PY ref_inkling_selftest.py
# CPU reference (hours): routing.npz + ref_<name>.npz; REF_GOLD=1 also writes per-layer fixtures for the port's tests
REF_GOLD=1 REF_TAPS=5,11,23,29,35 $PY ref_inkling.py runs/inkling/texts_held.json runs/inkling/ref_held
$PY ref_inkling.py runs/inkling/texts_train.json runs/inkling/ref_train
```

### Prefill kernel plan (Phase 5, item 1)

Prefill currently runs as rows groups of 8. That is exact and reuses the verify path, but every group re-reads the
global layers' whole KV, and host op launches scale with T/8. The proper kernel:

1. **Precompute G.** Per query row and head, `G = r_q @ proj_rev / scale`, the relative logits over distance stored
   reversed. That is one ttnn matmul per layer per chunk: [T, NH, E].
2. **Base.** Copy the stock chunked SDPA prefill kernels (ttnn/.../transformer/sdpa/device/kernels) in their
   **use_provided_mask + chunked** configuration. In that configuration the reader streams one mask tile per
   (q chunk, k chunk) and compute adds it.
3. **Synthesise instead of read.** The reader no longer reads mask tiles from DRAM; it builds them.
   - Band k-tiles: row i takes the 32 contiguous G values at offset E-1-(d0+i) (a skewed copy), with -inf past the
     causal edge or outside the sliding window.
   - Far k-tiles: zeros.
4. **Port only that factory slice to python.** Port the slice of `sdpa_program_factory.cpp` for that configuration as a
   generic_op, as `rows_sdpa.py` did for decode.

Estimated bias-synthesis cost: about 1 ms per layer per 4K tokens.

**DSpark (Phase 4).** `RadixArk/Inkling-Small-DSpark-Preview` is downloaded and has its reference code (`dspark.py`,
`dflash.py`).
- It has 0.9B params: 5 Qwen3 layers (64 heads / 16 KV heads, head_dim 64, FFN 8192) with YaRN RoPE (theta 8e6,
  factor 128).
- Block 15, mask token 200064, taps [5, 11, 23, 29, 35], plus a Markov head (rank 256) and a confidence head.
- The model card reports a mean acceptance length of 3.35 at block 7, measured against the NVFP4 target.

### Device tests (in the image, via the port's runner)

```bash
cd models/autoports/thinkingmachines_inkling_small
tools/run_dev.sh python models/autoports/thinkingmachines_inkling_small/tests/test_sdpa_device.py      # kernel vs fp32
DEV_ENV="-e INK_LONG=1" tools/run_dev.sh python .../tests/test_sdpa_device.py                            # + 128K cases
tools/run_dev.sh python .../tests/test_sconv_device.py
tools/run_dev.sh python .../tests/test_attention_device.py /cache/fixtures/layer5.pt 5                   # HF fixtures
tools/run_dev.sh python .../tests/test_layer_device.py /cache/fixtures/layer0.pt 0
tools/run_dev.sh python .../tests/test_layer_device.py /cache/ref/held/gold.pt 17                        # gold residuals
# host-only tests (CPU venv): relbias, sconv, router formulation
PYTHONPATH=$REPO ~/inkling-work/venv/bin/python -m pytest -q --noconftest --import-mode=importlib .../tests/test_*_host.py
```
(`/cache` in the container = `~/.cache/inkling` on the host: fixtures in `fixtures/`, the reference in `ref/`.)

## Log

- **2026-11-14:** Plan approved: text first, MTP and DSpark both built and picked by measurement, BF16 as source and
  reference.
  - Branch created and Docker pruned, keeping the newest tag per repo; 729 GB free.
  - `libhostk.so` built, BF16 + DSpark download started, CPU venv and text sets built.
  - `ref_inkling.py` written and validated.
    - Validation is against HF's full-model forward on a tiny random Inkling, chunked and unchunked: max abs error ≤7e-6.
    - One bug found and fixed on the way: in a layer-at-a-time run, the HF mask helpers size masks from the first cache
      layer of each type, so every cache slot is aliased to the current layer's entry.
