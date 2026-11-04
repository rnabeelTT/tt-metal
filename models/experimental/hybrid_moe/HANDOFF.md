# Hybrid MoE prototype — handoff

Read this first when resuming. Code: `~/FORK-tt-metal/models/experimental/hybrid_moe` on branch
`raahem/qwen35-397b-hybrid-moe` (fork rnabeelTT/tt-metal, based on `fusion_all`); `~/hybrid-moe-poc` is a symlink to it.
Run data (`runs/`, `logs/`, `cache/`, `ref397/`) is gitignored. Formal plan: `docs/PLAN.md`; deviations under **Deviations**.
The plugin fork `~/FORK-vllm-tt-plugin` has the same branch (no changes yet; the demo uses the image's plugin).

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
- 2026-11-04: held-out decode 21.5 tok/s mean (plan v4 + lean host step).
- 2026-11-04: fused MoE kernels + HYB_SKIP=0.1 served through vLLM at 13-17 tok/s (was 0.58).
- 2026-11-03: M5 vLLM serving works (0.58 tok/s). Next: speed (host overheads), hit rate.
- 2026-11-03: M4 full-model traced decode coherent, 1.5 s/token. Next: M5 vLLM serving.
- 2026-11-03: disk freed to 671 GB; 397B FP8 download done (~1 h); image source extracted to `image_src/`.
- 2026-11-03: full 60-layer traced run crashed at capture: L1 CB clash (router scores/ids kept live in L1 per layer
  for the host step). Fix: DRAM copies for the host step. Repo initialised, plan copied to docs/PLAN.md.
- 2026-11-03: overlay written; 122B M1 coherent; int4 library built; dynamic bfp8 side cache working on 4-layer 397B.
