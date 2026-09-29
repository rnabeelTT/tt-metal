# Qwen3.5-122B-A10B on 4 Blackhole dies — handoff

Text-only TTNN port of `Qwen/Qwen3.5-122B-A10B` (snapshot `dc4d348443bc740c68e2d77492492c11606384d5`) for the (1,4)
Blackhole mesh (p300x2 = 2 x p300c, or p150x4). Served through vLLM 0.26.0 and the Tenstorrent plugin, packaged as a
`tt-model` container bundle. Branch `qwen3_5_122b_a10b` (local commits per stage, never pushed; SHAs in section 6).

## 1. What is served

| item | value |
|---|---|
| architecture name | `Qwen3_5MoeForConditionalGeneration` (text path; image / video limits 0) |
| adapter | `tt/generator_vllm.py::Qwen35MoeForConditionalGeneration` (subclass of the qwen36 `Qwen36ForCausalLM`) |
| mesh | (1,4), `MESH_DEVICE=P300x2` or `P150x4`, `fabric_config FABRIC_1D`, `l1_small_size 24576`, trace region 768 MiB |
| parallelism | TP4 mixers (KV heads replicated per device pair), routed experts TP on the intermediate dim (all 256 experts per die at width 256), vocab-sharded LM head, on-device sampling |
| precision | `doc/datatype_sweep/selected_precision_config.json`: routed experts bfp4 LoFi, mixers bfp4 LoFi, shared expert bfp8, router bf16, LM head bfp4, KV bfp8, GDN state bf16 |
| serving | `--max-num-seqs 32 --max-model-len 262144 --block-size 64`, `sample_on_device_mode all`, `--reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice`, `--default-chat-template-kwargs '{"enable_thinking": false}'` (the validated release configuration); thinking on per request with `chat_template_kwargs {"enable_thinking": true}` (reasoning in the message's `reasoning` field, needs thousands of tokens) |
| decode | one trace per power-of-two width 2..32 (`tt/decode_buckets.py`; width 1 not served, see section 4) |
| KV pool | 1,820,160 tokens shared by the live users (one user reaches 262144; 32 users share ~57K each) |

Measured on this host (stage 10, `doc/optimized_vllm/README.md`; stage 11 release report rows agree):

| workload | result |
|---|---|
| 1 user, 128 / 128 tokens | TTFT 426 ms (`vllm bench serve`, stage 10) / 394-398 ms (TTI release rows), TPOT 37.7 ms (26.5 t/s/u) |
| 32 users, 128 / 128 tokens | TPOT 125-133 ms, 288 total tok/s (release report), CI burst 129 tok/s |
| per-width TPOT 1 / 2 / 4 / 8 / 16 / 32 users | 37.7 / 39.4 / 51.8 / 93.1 / 107.6 / 125.1 ms |
| long prompt, 1 user | TTFT 133 s at 131072 tokens, 283 s at 262080 tokens; decode unchanged |
| Hub bundle (final image `36b6aa8bb6e4`, thinking off by default), served from the Hub on a reset device (`doc/tti_release/RUN_NOTES.md` round 3) | healthy in 12.8 min; `tt-model curl` answers with defaults; thinking opt-in works; 32 concurrent 126.1 tok/s; 131072-token prompt OK |
| accuracy (TTI ci-nightly subsets, non-thinking greedy, graded vs the bf16 HF checkpoint in the same setting) | ifeval 92.66 vs HF 89.91 (published 93.4 is thinking mode); gpqa_diamond 77.5 vs HF 80.0 (a pass by one item under TTI's sample-count rule, both budget-cut at 4096 tokens; published 86.6 is thinking mode): stage-11 status `release-readiness-ci-subset-pass`; full sets non-thinking 87.99 / 68.69 (`doc/tti_release/RUN_NOTES.md`) |

## 2. Run it

From the checkout (the readiness runner starts vLLM with the plugin):
```bash
cd ~/tt-metal && source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD:$PYTHONPATH ARCH_NAME=blackhole MESH_DEVICE=P300x2
export HF_MODEL=Qwen/Qwen3.5-122B-A10B QWEN35_PROGRAM_GUARD=1 QWEN35_TRACE_REGION_MIB=768
export TT_MODEL_BRINGUP_ROOT=~/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.3
export TT_AUTODEBUG_ROOT=~/.claude/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.5
eval "$(python "$TT_MODEL_BRINGUP_ROOT/scripts/environment.py")"
python -m readiness_check.run_vllm_server --model-dir models/autoports/qwen_qwen3_5_122b_a10b --hf-model $HF_MODEL \
  --mesh-device P300x2 --stages serve --max-num-seqs 32 --max-model-len 262144 --block-size 64 --port 8131 \
  --tt-config '{"trace_region_size": 805306368, "l1_small_size": 24576, "fabric_config": "FABRIC_1D"}' \
  --additional-server-args "--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml --default-chat-template-kwargs '{\"enable_thinking\": false}'"
```
From the bundle (`tt-model` 0.1.0, `~/tt-model-manager/.venv/bin/tt-model`):
```bash
tt-model serve raahemnabeel/qwen3-5-122b-a10b-blackholex4 --profile p300x2   # or --profile p150x4
tt-model curl "Say hello in one sentence."   # thinking off by default; add --chat_template_kwargs '{"enable_thinking": true}' --max-tokens 4096 for thinking
```
Rebuild the bundle: `tt-model package --container models/autoports/qwen_qwen3_5_122b_a10b/tt-model.yaml --out <dir outside the tree>`
(the build takes ~1 h on this host; shipped files must be world-readable; see `doc/tti_release/RUN_NOTES.md` "container bundle").

The weights are a pointer (`weights.repo` / `revision` in `tt-model.yaml`); the first start converts them into the TT cache
(`TT_CACHE_PATH`, ~20 GB per die, one-time ~40 min); later starts reach "Application startup complete" in ~9-10 min (layer load ~8 min + trace warm-up; `logs/11/vllm-server.log` 10:56:31 -> 11:05:48).

Env knobs the runtime reads (all optional; defaults are the served configuration): `QWEN35_TRACE_REGION_MIB` (must equal the
TT config's `trace_region_size`), `QWEN35_PROGRAM_GUARD` (assert no program compiles after the traces are parked),
`QWEN35_DECODE_BUCKETING` (default on), `QWEN35_DECODE_MIN_WIDTH` (default 2), `QWEN35_SEEDED_FULL_WIDTH` (default 1),
`QWEN35_MOE_GATHER_MAX_TOKENS` (default 4), `QWEN35_PREFILL_TRACE_MAX_BUCKET`.

## 3. Checks

| check | command | expected |
|---|---|---|
| adapter unit tests (device) | `pytest models/autoports/qwen_qwen3_5_122b_a10b/tests/test_generator_vllm.py -x` | 6 passed in ~3 min (reduced 2-layer model, one GDN + one attention layer, real weights; `logs/10/adapter-test12.log`) |
| readiness serving suite | `run_vllm_server --stages serve,sampling,qualitative,benchmark --sampling-profile full ...` | sampling 73 passed / 4 skipped, qualitative 6 / 6 |
| teacher forcing | `python -m readiness_check.run_teacher_forcing --model-dir ... --mesh-device P300x2 --fabric-config FABRIC_1D` | top-5 >= 98 %, top-100 = 100 % (stage 8 numbers in `doc/datatype_sweep/`) |
| stage gates | `MODEL_DIR=models/autoports/qwen_qwen3_5_122b_a10b bash $TT_MODEL_BRINGUP_ROOT/prompts/model_bringup_multigoal/{06,07,09,10,11}-*.check.sh` | exit 0 each |
| TTI release | `doc/tti_release/RUN_NOTES.md` (needs the local TTI patch `doc/tti_release/tti_local_integration.patch` on v0.22.0) | run 11: Acceptance PASS under FUNCTIONAL, evals 2 / 2 vs the same-setting HF reference, spec tests 22 / 22; status `release-readiness-ci-subset-pass` |

## 4. Known limits and decisions

- **Width-1 decode is not served.** A single live user runs the width-2 trace (37.7 ms per token) because the width-1
  step is intermittently non-deterministic on device (greedy re-runs fork on 48 layers; bisected in
  `doc/optimized_vllm/README.md` section 3, not the MoE gather, not attention / GDN padding). Widths >= 2 are
  bit-reproducible. The width-1 trace measures 25.1 ms: about 12 ms per token is left on the table for one user.
- **Seeded requests run at full width (32).** Cross-width numerics differ (gather vs masked MoE, SDPA grid), so a batch
  with any `seed` runs the width-32 trace to keep seeded outputs reproducible regardless of co-tenants.
- **Prefill is serialised** at ~1000 tok/s per admission (traced chunk buckets 128..2048: 7.8 s at 8192 tokens, 133 s at 131072); 32 simultaneous long prompts wait minutes
  for TTFT (release report rows). Concurrent prefill is the next serving lever.
- **Thinking mode loops on constraint-heavy prompts** under greedy decoding and under the model card's coding preset
  (temperature 0.6, no presence penalty): 20-35 % of IFEval-style prompts never close the think block within 16K tokens and
  end in verbatim two-line repetition; the bf16 HF reference on CPU shows the same dithering (`doc/tti_release/RUN_NOTES.md`
  run 7). Answers that do finish score at the published level (94.35 % on the non-empty IFEval rows). The card's GENERAL
  thinking preset (temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5) removes the loops: in RUN_NOTES run 8a all
  16 probe prompts finish with no repetition. The bundle serves thinking OFF by default because a plain request gets the
  model's `generation_config.json` decoding (temperature 0.6, no presence penalty), which is the looping one; clients
  enabling thinking must pass the general preset and a budget of thousands of tokens.
  Whether the port's precision policy changes the loop frequency was NOT measured: the alternative-policy servers (stage-7
  baseline, GDN state fp32, KV bf16) did not start under vLLM in the one attempt (RUN_NOTES run 7b) — a follow-up.
- **Accuracy vs the published numbers.** The model card's IFEval 93.4 / GPQA-Diamond 86.6 are thinking-mode scores; the
  served non-thinking configuration scores 87.99 / 68.69 on the full sets (0.942 / 0.793 of them). Against the bf16 HF
  checkpoint in the same setting the port is within TTI's tolerance (ci-nightly subsets: 92.66 vs 89.91 on IFEval; 77.5 vs
  80.0 on GPQA, one item above the threshold; both sides are cut at the 4096-token budget on 13-18 of the 40 answers, so the
  non-thinking GPQA numbers are budget-limited), which is what TTI's run-11 release report grades. A thinking-mode GPQA full set with the card's 81920-token budget is not feasible on
  this hardware (~6 tok/s/user at 32 users); the thinking-mode IFEval full set with the general preset scores 87.25 (92.0 on
  the answered rows).
- **Vision is not served** (section 5). MTP head (`mtp.*`) is dropped.
- **Plugin registration** is via `vllm_metadata.json` under `EXTRA_MODELS_DIR` (checkout: `models/autoports`; bundle:
  `vllm_bundle/`). The plugin checkout `~/vllm-tt-plugin` carries an unpushed local commit 92403f3 that adds the same
  registration to `platform.py`; upstream it or keep using `EXTRA_MODELS_DIR`.
- **TTI v0.22.0 does not know this model**: the local patch (impl spec, dev catalog template, eval config, spec-test matrix)
  is recorded in `doc/tti_release/`, not pushed.

## 5. Vision enablement path (not done in this bring-up)

The checkpoint's vision tower (`model.visual.*`, 333 tensors, Qwen3-VL style ViT) is loadable:
`tt/weights.py::StreamingStateDict.visual_keys()` / `visual_tensor()` expose it without loading the text weights. To serve
images / videos:
1. `tt/generator_vllm.py`: replace `TT_Qwen3_5MoeProcessingInfo.get_supported_mm_limits` (`{"image": 0, "video": 0}`) with
   real limits, drop the `NotImplementedError` in `prefill_forward` for `pixel_values` / `pixel_values_videos`, and
   implement `get_multimodal_embeddings` / `embed_input_ids` (the vLLM `Qwen3_5MoeForConditionalGeneration` interface).
2. Vision encoder on device: reuse `models/demos/qwen3_vl/tt/` (the Qwen3-VL ViT port) with this checkpoint's
   `vision_config` (hidden 1152, depth 27, 16 heads, patch 16, spatial merge 2, merger to hidden 3072;
   `deepstack_visual_indexes` is empty in this checkpoint, so no per-layer deepstack residual add is needed: the merged
   visual embeddings replace the image-placeholder token embeddings before layer 0).
3. Position ids: the text path uses 1-D RoPE positions (`mrope_section [11, 11, 10]`, `mrope_interleaved`, reduces to 1-D for text). Images need
   the 3-D m-RoPE index (t, h, w) per token: compute on host in the adapter (`get_rope_index` from HF) and pass a
   `[3, T]` position tensor into the prefill rotary setup (`models/demos/blackhole/qwen36/tt/attention` rotary tables);
   decode continues from the max text position as HF does.
4. Memory: the ViT weights are ~0.9 GB bf16 (TP4: ~0.25 GB per die); the KV pool shrinks accordingly. Re-run the
   context-contract numbers (`doc/context_contract.json`).
5. Validation: HF `Qwen3_5MoeForConditionalGeneration` image reference (single image, 1-3K visual tokens), then the
   readiness serving suite with `limit_mm_per_prompt` set; add an image request to the smoke.

## 6. Where things are

| item | location |
|---|---|
| code | `models/autoports/qwen_qwen3_5_122b_a10b/tt/` (`model.py`, `generator.py`, `generator_vllm.py`, `decode_buckets.py`, `moe.py`, `moe_dispatch.py`, `weights.py`, `precision.py`, stage decoders) |
| shared code touched | `models/demos/blackhole/qwen36/`, `models/experimental/gated_attention_gated_deltanet/`, `models/tt_transformers/`, `models/common/` (per-stage commits list the hunks) |
| tests | `models/autoports/qwen_qwen3_5_122b_a10b/tests/` |
| per-stage evidence | `doc/{functional_decoder,fused_decoder,optimized_decoder,multichip_decoder,optimized_multichip_decoder,full_model,optimized_full_model,datatype_sweep,vllm_integration,optimized_vllm,tti_release}/` |
| bundle manifest | `models/autoports/qwen_qwen3_5_122b_a10b/tt-model.yaml`; HF repo `raahemnabeel/qwen3-5-122b-a10b-blackholex4` (private) |
| bring-up workspace | `~/qwen35-122b-bringup/` (scripts per stage, logs, STATUS.md, staged bundle under `pkgbuild/`) |
| stage commits | 1-5: see `git log --oneline qwen3_5_122b_a10b`; 6 `215075683d2`, 7 `96badd2f6fc`, 8 `e8de3cdd6ba`, 9 `0e57369c7a6`, 10 `191f7f672f3`, 11 = the commit that adds this file (`git log -1 -- models/autoports/qwen_qwen3_5_122b_a10b/doc/HANDOFF.md`) |
