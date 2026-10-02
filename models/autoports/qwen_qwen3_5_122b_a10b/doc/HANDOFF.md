# Qwen3.5-122B-A10B on 4 Blackhole dies — handoff

Text-only TTNN port of `Qwen/Qwen3.5-122B-A10B` (snapshot `dc4d348443bc740c68e2d77492492c11606384d5`) for the (1,4)
Blackhole mesh (p300x2 = 2 x p300c, or p150x4). Served through vLLM 0.26.0 and the Tenstorrent plugin, packaged as a
`tt-model` container bundle. Branch `qwen3_5_122b_a10b` (local commits per stage, never pushed; SHAs in section 6).

## 1. What is served

| item | value |
|---|---|
| architecture name | `Qwen3_5MoeForConditionalGeneration` (text + images; image limit `QWEN35_VISION_MAX_IMAGES` = 4 per request, video 0; §5) |
| adapter | `tt/generator_vllm.py::Qwen35MoeForConditionalGeneration` (subclass of the qwen36 `Qwen36ForCausalLM`) |
| mesh | (1,4), `MESH_DEVICE=P300x2` or `P150x4`, `fabric_config FABRIC_1D`, `l1_small_size 24576`, trace region 768 MiB |
| parallelism | TP4 mixers (KV heads replicated per device pair), routed experts TP on the intermediate dim (all 256 experts per die at width 256), vocab-sharded LM head, on-device sampling |
| precision | `doc/datatype_sweep/selected_precision_config.json`: routed experts bfp4 LoFi, mixers bfp4 LoFi, shared expert bfp8, router bf16, LM head bfp4, KV bfp8, GDN state bf16 |
| serving | `--max-num-seqs 32 --max-model-len 262144 --block-size 64`, `sample_on_device_mode all`, `--reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice`, `--default-chat-template-kwargs '{"enable_thinking": true}'` (the validated release configuration); thinking on per request with `chat_template_kwargs {"enable_thinking": true}` (reasoning in the message's `reasoning` field, needs thousands of tokens) |
| profiles (run 10+) | `--profile p300x2` interactive default (thinking on, MTP, vision) · `p300x2-burst` (batched short-prompt prefill on: 32-request burst TTFT 3.8 s vs 6.3 s; seeded / greedy consistency across batch compositions not guaranteed) · `p300x2-longctx` (chunked prefill, MTP off: prefill steps shared between prompts) · `p300x2-text` (no vision tower, 28472 KV blocks) · `p150x4` (default on four p150) - one image, launch-time env/args (tt-model.yaml `serve_profiles`) |
| conformance on the bundle (run 9, thinking-off requests) | readiness `sampling` stage (73 plugin parameter tests) against the bundle server: 72 passed, 1 `test_bad_words` false positive ("_hello" is a single token the bad-words blocker cannot match; `doc/perf_pass/work_log.md` 02:16); with the `-burst` profile 12 tests fail (first-token numerics vary with the batch composition) |
| decode | one trace per power-of-two width 2..32 (`tt/decode_buckets.py`; width 1 not served, see section 4) |
| KV pool | 1,820,160 tokens shared by the live users (one user reaches 262144; 32 users share ~57K each) |

Measured on this host (stage 10, `doc/optimized_vllm/README.md`; stage 11 release report rows agree):

| workload | result |
|---|---|
| SHIPPED configuration (MTP k=3 adaptive on, vision on, expert-sorted decode MoE from 16 rows; served 2026-10-01 21:47, `logs/13/serve-sorted16-cfgfix.log`) | readiness streaming benchmark 128 / 128: 1 user TTFT 320 ms, TPOT 20.6 ms = 48.4 t/s/u (MTP; the readiness suite's own single-user run earlier the same day: 16.6 ms, acceptance-dependent); 8 users TPOT 63.9 ms, 103 tok/s; 32 users TTFT 6.3 s, TPOT 110.2 ms, 201 tok/s (release: 37.7 / 93.1 / 125-131 ms). Full sweep of the deployed bundle (input 128 .. 130816 x 1 .. 32 users): `doc/perf_pass/deployed_sweep.md` |
| 1 user, 128 / 128 tokens | TTFT 408 ms (p50), TPOT 32.7 ms (30.6 t/s/u) on the width-1 trace (perf pass, `doc/perf_pass/`); stage 10/11: 394-426 ms, 37.7 ms on the width-2 trace |
| 32 users, 128 / 128 tokens | TPOT 125-131 ms, 288 total tok/s (release report), CI burst 129 tok/s (unchanged by the perf pass: weight-bandwidth bound) |
| per-width TPOT 1 / 2 / 4 / 8 / 16 / 32 users | 32.7 / 38.1 / 46.7 / 67.7 / 107.0 / 125.2 ms (perf pass; stage 10: 37.7 / 39.4 / 51.8 / 93.1 / 107.6 / 125.1) |
| long prompt, 1 user (stage 12, group-parallel MoE prefill) | TTFT 4.0 s at 8192, 16.4 s at 32768, 72.3 s at 131072 tokens (stage 11: 7.8 / 31.4 / 132.6 s); decode unchanged; `doc/optimized_prefill/` |
| Hub bundle, revision `511541bd2499` (2026-10-02 02:45; tt-metal `da20251f443`; FINAL: served image-size fix, profiles p300x2 / -burst / -longctx / -text + the p150x4 four, thinking ON by default, MTP k=3 adaptive, batched prefill opt-in) | fresh-device verification `doc/tti_release/bundle_run10/`: healthy 11 min, chat (reasoning 595 tokens) / tool call / curl, 2381-token prompt, 32-request burst 189 tok/s, 131K prompt 44.9 s, 194K 74.0 s, 1080p / 4 MP photo / two-image requests served; sampling stage 71 / 73 with thinking on (bad_words false positive + one empty-content budget case), 72 / 73 thinking off (run 9) |
| Hub bundle, revision `832a9834` (2026-10-01 22:08; image `e096e047ba10`, tt-metal `aa559ead231`: MTP default-on, vision on, resident MoE weights + moe_gather_rows + row-major down projection, expert-sorted decode from 16 rows, fabric-aware CCL topology; thinking off), verified from the staged package on a reset device (`doc/tti_release/bundle_run6/verify.log`) | healthy in ~11 min; chat, tool call, tt-model curl; 32-request burst 188 tok/s; 131072-token prompt 45.0 s; 194119-token prompt 73.8 s; image request == HF reference. Earlier revisions: `668d1f0b` (2026-10-01 14:41), stage-11 `36b6aa8bb6e4` image |
| accuracy (TTI ci-nightly subsets, non-thinking greedy, graded vs the bf16 HF checkpoint in the same setting) | ifeval 92.66 vs HF 89.91 (published 93.4 is thinking mode); gpqa_diamond 77.5 vs HF 80.0 (a pass by one item under TTI's sample-count rule, both budget-cut at 4096 tokens; published 86.6 is thinking mode): stage-11 status `release-readiness-ci-subset-pass`; full sets non-thinking 87.99 / 68.69 (`doc/tti_release/RUN_NOTES.md`) |

## 2. Run it

**The box must be a ring, or the model runs Linear.** A p300x2 is a ring of four dies only when BOTH board-to-board
ethernet links are up; the engine's fabric discovery prints the check: `intra-mesh degree histograms mesh0 {2:4}` is
a ring, `{1:2, 2:2}` is a line (one inter-board link missing or down - seen 2026-10-01 on another p300x2, firmware
19.15.0, where the stock Ring choice died at the first all-gather with `TT_FATAL ... Could not find any forwarding
direction from src (M0, D0) to dst (M0, D3)`). Since commit 0b48a2312d2 the model's collectives pick Ring / Linear
from the fabric the process actually got (`ttnn.get_usable_topology`; `QWEN35_CCL_TOPOLOGY=ring|linear|auto`), so a
line-cabled box serves (slower collectives) instead of crashing; restore the second link for ring performance. Also:
one engine per box - tt-model's "host ready: tt devices" check does not detect a running engine (`docker ps`,
`fuser /dev/tenstorrent/*` before `tt-model serve`).

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
  --additional-server-args "--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml --default-chat-template-kwargs '{\"enable_thinking\": true}'"
```
From the bundle (`tt-model` 0.1.0, `~/tt-model-manager/.venv/bin/tt-model`):
```bash
tt-model serve raahemnabeel/qwen3-5-122b-a10b-blackholex4 --profile p300x2   # or --profile p150x4
tt-model curl "Say hello in one sentence."   # thinking ON by default (reasoning in the `reasoning` field); add --chat_template_kwargs '{"enable_thinking": false}' for a direct answer
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
- **Width-1 decode was non-deterministic until the perf pass**: the fused router gate op (`generalized_moe_gate`,
  single-row decode only) forked greedy re-runs; it is off by default now (functional two-op router), a single user runs
  the width-1 trace (32.7 ms per token), and served seeded / greedy completions reproduce at every width (`doc/perf_pass/`).
  The pairs decode MoE path (widths 1-8) cut the 8-user step 93 -> 68 ms.
- **Seeded requests run at full width (32).** Cross-width numerics differ (gather vs masked MoE, SDPA grid), so a batch
  with any `seed` runs the width-32 trace to keep seeded outputs reproducible regardless of co-tenants.
- **Prefill is serialised** per admission at ~2000 tok/s (stage 12: group-parallel `ttnn.sparse_matmul` factory over all
  110 cores + matmul counting sort; traced chunk buckets 128..2048: 4.0 s at 8192 tokens, 72 s at 131072; before stage 12
  ~1000 tok/s). 32 simultaneous long prompts still wait for each other's prefill. Chunked / batched prefill (interleaving
  prefill segments with decode steps, as the gpt-oss-120b p150x4 bundle does) is the next serving lever; the remaining
  per-layer MoE cost at 2048 tokens is dispatch bookkeeping (elementwise ops on the one-hot, three 1-core scatters).
- **Thinking mode loops on constraint-heavy prompts** under greedy decoding and under the model card's coding preset
  (temperature 0.6, no presence penalty): 20-35 % of IFEval-style prompts never close the think block within 16K tokens and
  end in verbatim two-line repetition; the bf16 HF reference on CPU shows the same dithering (`doc/tti_release/RUN_NOTES.md`
  run 7). Answers that do finish score at the published level (94.35 % on the non-empty IFEval rows). The card's GENERAL
  thinking preset (temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5) removes the loops: in RUN_NOTES run 8a all
  16 probe prompts finish with no repetition. UPDATE 2026-10-02 (owner decision): the bundle serves thinking ON by default from run 10 (`--default-chat-template-kwargs '{"enable_thinking": true}'`); clients turn it off per request with `{"enable_thinking": false}`, which is the configuration every gate below was measured in. Until run 9 the bundle served thinking OFF by default because a plain request gets the
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

### Perf pass turn 5 (2026-09-30): long-prompt prefill

- The expert-sorted prefill path (T >= 128) now builds its dispatch with ONE device kernel,
  `ttnn.experimental.moe_sorted_dispatch` (new op under `ttnn/cpp/ttnn/operations/experimental/moe_sorted_dispatch/`;
  bit-identical to the former matmul counting sort), the grouped sparse matmul skips the zero-fill of its output and
  skips the static padding groups (`MatmulMultiCoreReuseProgramConfig.skip_padding_groups`, new field). 48-layer traced
  prefill at 2048 / 8192 tokens: 985 / 3959 -> 645 / 2611 ms (`doc/perf_pass/README.md` turn 5).
- Knobs (defaults are the shipped values): `QWEN35_DISPATCH_MODE=kernel` (`counting_mm` restores the stage-12 path),
  `QWEN35_SKIP_PAD_GROUPS=1`, `QWEN35_UNDISPATCH_KERNEL=1` (the fused un-dispatch kernel `moe_gather_add_rows`; 0 restores
  the k gather + add ops), `QWEN35_PREFILL_SEGMENT=2048` (do not raise: whole-layer programs fail above 2048 rows),
  `QWEN35_SORTED_GROUP_ROWS=32` (64 measured slower).
- Building: the new ops are part of the `ttnn` target (`ninja -C build_Release ttnn`); the linked library is
  `build_Release/ttnn/_ttnn.so` (not `build_Release/lib/`), and Python loads `ttnn/ttnn/_ttnn.so`: replace it with
  copy + atomic `mv` while device jobs run. Kernels compile on first use.
- MTP (speculative decoding with the checkpoint's multi-token-prediction head; `doc/perf_pass/mtp_plan.md`,
  `doc/perf_pass/adaptive_mtp_plan.md`): ON BY DEFAULT since 2026-10-01 (`QWEN35_MTP` = k drafts per pass, default 3;
  `QWEN35_MTP=0` disables; `QWEN35_CHUNKED_PREFILL=1` without an explicit `QWEN35_MTP` turns it off) at ANY
  `--max-num-seqs`.
  ADAPTIVE (default with > 1 slot): the single-user MTP step runs whenever exactly one request is live, the plain
  batched decode otherwise, the request's state handed over at every switch (GDN state into a persistent B = 1 MTP
  scratch, head hidden rows and head K/V kept by two small hooks inside every plain decode step: +0.4 ms at 8 users);
  requests with penalties or logprobs stay on the plain path. Measured (8 slots, full model): c = 1 TPOT 18.7 ms =
  53.6 t/s/u (plain 29.9), TTFT p50 344 ms (plain 261: +~10 ms phase enter, + the head's prompt rows), concurrent
  texts identical to the plain server's; c = 8 TPOT 60.0 ms (plain 59.6). `QWEN35_MTP_ADAPTIVE=0` = the batched
  multi-user step over every live slot (slower than plain at 2-4 users). The plugin shim (`tt/mtp_plugin_shim.py`)
  runs under vLLM's async scheduling: with chunked prefill off it never classifies a cached non-resumed row as prompt
  work (two in-flight multi-token steps leave up to 2k + 1 tokens outstanding; the width bound misread that as a
  chunked continuation and crashed the engine). Temperature 0 = greedy accept; temperature > 0 =
  EXACT speculative sampling (drafts sampled from the head, rejection-sampling accept with the request's temperature /
  top-k / top-p; a request with 0 < top_k <= 32 - the checkpoint's generation default is top_k 20 / top_p 0.95 - is
  served from device top-k candidates with no logits read-back; penalties are not applied in the MTP step). Knobs:
  `QWEN35_MTP_GDN_ROWS=32` (GDN block rows of the verify pass; 128 = the prefill-form norm path),
  `QWEN35_MTP_MOE_ROWS=pairs` (`sorted` is slower), `QWEN35_MTP_KV_ROWMASK=1`, `QWEN35_MTP_TRACED_PREFILL=1` (the
  head's prompt rows run inside the traced prefill bucket programs; 0 = eager bucket prefill + a separate head pass).
  The plugin runs with an import-time shim (`tt/mtp_plugin_shim.py`) for a variable number of tokens per step; the KV
  pool shrinks by the head's K/V (13th KV layer + 1.5 GiB). Measured (full model, c = 1, 2026-10-01): TPOT 18.6 ms =
  53.7 t/s/u vs 31.3 plain; code prompts 1.8x, prose ~1.06x end to end (sampled T = 0.7: 51.6 / 28.2 tok/s); TTFT p50 306 ms (plain 258-274). Per-request
  `MTP_STATS` lines in the server log give passes, tokens per pass, acceptance histogram and ms per pass.
  Standalone tools: `tools/mtp_traced_run.py` (traced step, per-program walls; `--temperature/--top-k/--top-p` time the
  sampled loop), `tools/mtp_pass_check.py` (verify pass vs decode, row by row), `tools/mtp_generate_full.py` (eager
  loop), `tools/mtp_acceptance.py`, `tools/mtp_traced_multi_run.py` (multi-user traced step).
- Host-sampled requests (logprobs, min_p, structured outputs, logit bias): their prefill arrives WITHOUT device
  sampling params; the row is final unless the chunked-prefill shim's mask says otherwise (a heuristic that tied
  "final" to the presence of sampling params broke every such request between the chunked-prefill turn and
  2026-10-01 08:40 - zero logits in plain mode, an engine crash in MTP mode). Any change to the prefill contract must
  be exercised with a logprobs request (`EXTRA=1` in `scripts/14-serve-mtp.sh`).
- Chunked prefill (turn 7, opt-in): `QWEN35_CHUNKED_PREFILL=1` + vLLM `--enable-chunked-prefill --max-num-batched-tokens
  2048`; intermediate chunks run whole 2048-token segments from the request's own processed end with the GDN state
  carried per request (keyed by its first KV block), the final chunk samples; validated with the plugin's needle test
  (4 concurrent, ragged boundaries). The plugin never mixes prefill and decode in one step, so this bounds step length
  and shares steps between prompts; it does not shorten running users' inter-token gap. `doc/perf_pass/chunked_prefill_plan.md`.
- Grouped sparse matmul (prefill sorted dispatch, decode pairs path): RESIDENT weights - a core that owns consecutive
  groups of the same expert reads the expert's [K, N] block once (`TT_SPARSE_GROUPED_RESIDENT_IN1=0` opts out; the
  factory streams K-blocks when the block does not fit the 1.3 MB per-core CB budget, e.g. bfp8 gate/up). Measured
  at 2048 tokens: gate/up sparse matmul 2.37 -> 1.43 ms, down 1.65 -> 1.20 ms per layer (-1.4 ms of ~9.7). Kernels:
  `kernels/compute/bmm_grouped_resident_in1.cpp`, `reader_writer_bmm_tile_layout_in1_indexed.cpp` (RESIDENT_IN1),
  tests `tests/test_sparse_matmul_grouped.py::test_grouped_sorted_runs_match_torch`.
- Decode MoE path by width: gather (<= 4 rows), PAIRS (one group per (token, expert) pair, grouped sparse matmul) up to
  `QWEN35_MOE_PAIRS_MAX_TOKENS` = 16 (2026-10-01: -17 % / -9 % attention / GDN layer step at width 16 vs the masked
  path), masked (dense 256-expert glue) at 32 rows where 256 pair groups lose by 22-32 %.
- Decode router: `QWEN35_ROUTER_DRAM_SHARDED=1` (default) runs the [32, 3072] x [3072, 256] router as a DRAM-sharded
  decode matmul (71 -> 25 us per layer at width 8; 0 restores the stock linear).
- Batched multi-user MTP (built, measured negative at 2-4 users; `QWEN35_MTP_ADAPTIVE=0` selects it instead of the
  default adaptive mode) and the levers that would flip it: `doc/perf_pass/mtp_plan.md`.
- Next levers (not done): compact-layout decode MoE for 8-32 rows (gather / un-dispatch kernels; the same work as the
  prefill levers "row gather" + "untilize before un-dispatch"), one weight pass per expert in the grouped sparse
  matmul, a fused GDN decode step, MTP commit / draft overlap on the second command queue.

## 5. Vision (images) — served behind `QWEN35_VISION=1`

The checkpoint's vision tower (`model.visual.*`: Qwen3-VL-style ViT, depth 27, hidden 1152, patch 16, spatial merge 2,
merger to the 3072-wide text hidden) runs on the mesh; image content parts of chat requests are served when the engine
starts with `QWEN35_VISION=1`. Text-only is still the default of the checkout (the bundle sets the env, see §2).

| piece | where | notes |
|---|---|---|
| tower | `tt/vision.py` + the in-tree port `models/demos/blackhole/qwen36/tt/vision/` | HF `Qwen3_5MoeVisionModel` built standalone from the 333 streamed `model.visual.*` tensors (never `from_pretrained` of the 250 GB checkpoint); `DropInVisionTransformer` keeps the patch embed + position interpolation on host, 27 blocks + merger on device, TP4 (hidden-fractured output); weights bfp8 (`QWEN35_VISION_DTYPE=bf16` builds too but its merger `as_tensor` takes minutes on this host; PCC gain 0.001) |
| tower accuracy | `tests/test_vision_tower.py` | PCC vs the HF visual 0.9857 (448x448, 196 tokens) / 0.9868 (896x672, 588 tokens); sampled-row min 0.96. Required the SDPA windows fix in the shared port (`cu_window_seqlens=[0, real, padded]`, explicit `is_causal=False`): the padded patch sequence (2048 multiples) leaked into the real patches' non-causal attention (PCC 0.88 before) |
| splice | `Qwen35MoeModel.prefill_masked_bucket` (persistent bucket path) | image rows placed at the placeholder positions on host, staged into fixed `[1,1,2048,H/TP]` + mask buffers (`_alloc_vision_merge_buffers` before the captures), the bucket body's `where(mask, vision, text)` selects them (identity for text segments) — the bucket traces stay valid; images whose placeholders span 2048-token segments use the per-segment row offset |
| positions | `_build_request_rope(token_ids, vision_tokens)` staged once over the whole prompt (`rope_staged`); decode: per-slot M-RoPE delta vector (`set_slot_rope_delta`), used by `prepare_decode_inputs_host` and the MTP step's rot idxs | text requests keep delta 0 |
| adapter | `tt/generator_vllm.py` | `get_supported_mm_limits` = {image: `QWEN35_VISION_MAX_IMAGES` (4), video: 0}; `prefill_forward` turns vLLM's per-user `pixel_values` / `image_grid_thw` into tower rows (`_vision_for_user`) and hands `vision=[...]` to `prefill_serving`; tower warm-up at 448 / 896 px before the captures; videos raise `NotImplementedError` |
| MTP | unchanged | the MTP head reads the spliced hidden; adaptive single-user MTP serves image requests like text ones |

Limits: images only (no video); an image prompt is prefilled whole (no chunked-prefill continuation of an image
prompt — chunked prefill is opt-in and off with MTP anyway); at most `QWEN35_VISION_MAX_IMAGES` images per request
(vLLM's `--limit-mm-per-prompt` must agree); the tower runs eagerly (not traced) per image, padded to 2048-patch
multiples (448 px = 784 patches -> 2048; 896 px = 3136 -> 4096; 1920x1080 = 8160 -> 10240). Image resolution is capped
at `QWEN35_VISION_MAX_PIXELS` (default 4194304 = 4 MP = 16384 patches = 4096 image tokens per image; the checkpoint's
processor default is 16 MP) by `TT_Qwen3_5MoeProcessingInfo.get_hf_processor`, and `_warm_vision` compiles the tower for
EVERY padded bucket up to the cap (2048 .. 18432, nine eager passes at start-up). Why: the tower's programs are keyed
by the padded patch count, so an image in an unwarmed bucket compiles after the traces are parked and the program guard
stops the engine - the served failure of 2026-10-01 (a photo of 47k patches against a warm-up of 2048 / 4096 only:
`programs compiled after the trace captures (1894 -> 1923)`). Coverage test: `tests/test_vision_sizes.py`.

Validation so far (2026-10-01): tower PCC above; reduced 2-layer server with `QWEN35_VISION=1` serves text, logprobs,
penalty, min_p and a 448x448 image chat request (220 prompt tokens, 196 image tokens) with the program guard quiet;
HF CPU reference for the same image prompt (`~/qwen35-122b-bringup/scripts/16-hf-image-ref.py`, greedy, thinking off):
"The image features a red square in the upper left and a yellow circle in the lower right, both set against a solid
blue background." The full-model served comparison is logged in `doc/perf_pass/work_log.md` (search `IMAGE_AGREEMENT`).

## 6. Future scope (capped 2026-10-01 23:15)

The model is production-ready at bundle run 7 (image-size fix). What remains is optimization, documented here with the
evidence gathered so far; none of it blocks serving.

| item | evidence | next step |
|---|---|---|
| Sampler cost at full occupancy | temperature-1 decode at 32 users 129.7 ms per step vs 110.2 ms greedy (`doc/perf_pass/deployed_sweep.md`, pass 2) | Tracy of one sampled step at 32 rows; top-k pre-selection on the vocab-sharded logits before the gather |
| Batched short-prompt prefill | OPT-IN (`QWEN35_BATCHED_PREFILL=1`): served A/B 2026-10-02 TTFT 32 users 6.3 -> 3.8 s, 8 users 1.83 -> 1.27 s, decode unchanged; with it on, 12 / 73 conformance tests failed on bundle run 8 (seeded reproducibility and greedy consistency across batch compositions: the batched pass's first-token numerics differ from a single admission, PCC 0.9995, and the sample comes from logits row i) (`doc/perf_pass/work_log.md` 01:34) | admission-independent first-token numerics (one kernel path for single and batched, row-0 RNG via a row select) + MTP head rows for batched admissions, then default on |
| GDN slot write per admission | 29 ms of the 238 ms fixed cost per 128-token prefill (`doc/perf_pass/work_log.md` 2026-10-01 prefill cost) | write the B=1 scratch straight into the slot inside the bucket trace (one-hot row select) |
| Width-1 plain decode | 29.9 ms per token, ~90 small ops per layer; MTP hides most of it | router chain fusion (~5 ms), then a fused GDN decode kernel |
| Long-context decode at high occupancy | 32 users x 32K: 425 ms per token (sweep pass 1) | Tracy at that point first (paged-attention reads vs prefill interleaving) |
| MTP draft length | served sweep 2026-10-02 at one user: k=2 23.9 ms, k=3 19.6-20.6 ms, k=4 24.0 ms TPOT (`doc/perf_pass/work_log.md` 00:26) | k = 3 stays (`QWEN35_MTP=3`) |
| Multi-user MTP | measured 2026-10-02 (4 slots, `QWEN35_MTP_ADAPTIVE=0`): TPOT c=1 48.5 / c=2 46.7 / c=4 58.7 ms vs adaptive 19.6 / 37 / 44.5 ms - loses at every concurrency (`doc/perf_pass/work_log.md` 00:55) | adaptive stays; revisit only with a cheaper multi-row verify pass |
| Plugin asks (not in this tree) | chunked prefill is disabled for any model emitting more than one token per step; no prefill / decode alternation; prefix caching off | vllm-tt-plugin changes; the GDN carried-state primitive (`_pending_prefill`) is ready for prefix-boundary snapshots |

## 7. Where things are

| item | location |
|---|---|
| code | `models/autoports/qwen_qwen3_5_122b_a10b/tt/` (`model.py`, `generator.py`, `generator_vllm.py`, `decode_buckets.py`, `moe.py`, `moe_dispatch.py`, `weights.py`, `precision.py`, stage decoders) |
| shared code touched | `models/demos/blackhole/qwen36/`, `models/experimental/gated_attention_gated_deltanet/`, `models/tt_transformers/`, `models/common/` (per-stage commits list the hunks) |
| tests | `models/autoports/qwen_qwen3_5_122b_a10b/tests/` |
| stage-12 prefill evidence | `doc/optimized_prefill/` (README, work_log, Tracy before/after CSVs, readiness prefill checks, served before/after) |
| perf-pass evidence (stage 13) | `doc/perf_pass/` (README, work_log, width-1 investigation artifacts, served per-width runs) |
| per-stage evidence | `doc/{functional_decoder,fused_decoder,optimized_decoder,multichip_decoder,optimized_multichip_decoder,full_model,optimized_full_model,datatype_sweep,vllm_integration,optimized_vllm,tti_release}/` |
| bundle manifest | `models/autoports/qwen_qwen3_5_122b_a10b/tt-model.yaml`; HF repo `raahemnabeel/qwen3-5-122b-a10b-blackholex4` (private) |
| bring-up workspace | `~/qwen35-122b-bringup/` (scripts per stage, logs, STATUS.md, staged bundle under `pkgbuild/`) |
| stage commits | 1-5: see `git log --oneline qwen3_5_122b_a10b`; 6 `215075683d2`, 7 `96badd2f6fc`, 8 `e8de3cdd6ba`, 9 `0e57369c7a6`, 10 `191f7f672f3`, 11 = the commit that adds this file (`git log -1 -- models/autoports/qwen_qwen3_5_122b_a10b/doc/HANDOFF.md`) |
