# Bringing a new MoE model up with the hybrid overlay

Starting point: branch `hybrid-moe-starter` (fork `rnabeelTT/tt-metal`). It is the code that serves
[raahemnabeel/minimax-m2.7-hybrid-qb2](https://huggingface.co/raahemnabeel/minimax-m2.7-hybrid-qb2) (MiniMax-M2.7,
230B MoE, ~56 tok/s on one QB2, 32K / 64K / 128K profiles), plus copy-and-edit templates. Everything below uses M2.7 as
the worked example. Deep history and measurements: `HANDOFF.md`.

**The idea in one paragraph.** The model does not fit in the QB2's 4 x 32 GB of DRAM. Each MoE layer keeps its most-used
experts on the device (bfp4); the rest live on the host as int4 (computed by an AVX-512 CPU kernel) and as bfp4 copies
(streamed over PCIe when a prefill or a hot miss needs them). Some layers keep all experts ("full"); the more full layers,
the faster decode but the less DRAM is left for KV, which is what the 32K / 64K / 128K profiles trade. A speculative
drafter (DFlash) verifies 4 tokens per pass for about the cost of one, which is what makes decode usable.

## 0. Workflow

```bash
git fetch https://github.com/rnabeelTT/tt-metal.git hybrid-moe-starter
git switch -c hybrid/<your-model> FETCH_HEAD        # one branch per model, off the starter
tt-model pull raahemnabeel/minimax-m2.7-hybrid-qb2  # the serving image the dev scripts run in (or build your own)
```

Keep model-specific files in your branch. A fix that helps every model (in `hybrid/`, the build scripts, the
templates): cherry-pick it onto `hybrid-moe-starter` so the next person gets it.

Files that matter (the rest of `hybrid_moe/` is benchmarks and debugging history):

| Path | What |
|---|---|
| `hybrid/` | the runtime overlay: `patch.py` (installs into the autoport), `fetch.py` / `stream.py` (PCIe), `hostk/` (CPU int4 kernel), `pin/` (pinned zero-copy), `wcache.py` (device-image cache), `elastic.py` (128K), `hostmem.py` (RAM check), `hfpath.py` (`hf://` paths) |
| `serve/sitecustomize.py` | installs the overlay in every vLLM process when `HYB_ENABLE=1` |
| `dflash_serve.py`, `dflash_tt.py` | the DFlash drafter + verify loop (plugged in via `QWEN35_EXT_SPEC=dflash_serve:build`) |
| `quality_texts.py`, `ref_m27.py`, `sim_m27.py`, `sim_full_layers.py`, `build_int4_library.py`, `build_dev_experts.py` | the offline artifact pipeline |
| `serve/serve_m27.sh`, `serve/bench.py`, `serve/elastic_test.py` | dev serving and the two tests every number here comes from |
| `templates/` | `build_artifacts.sh`, `serve_model.sh`, `tt-model-hybrid.yaml`: copy these |
| `../../autoports/minimaxai_minimax_m2_7/` | the M2.7 port the overlay patches, and its `tt-model-hybrid.yaml` |

## 1. Will it fit? (do this before anything else)

From the model's `config.json`:

```
expert params   = MoE layers x experts per layer x 3 x hidden x moe_intermediate
device share    ~ 117 GB / (expert params x 0.5625 B)           # bfp4; M2.7: 225B -> 127 GB -> ~79% on device
host RAM        ~ 11 GiB + off-device expert params x 1.19 B     # int4 + pinned bfp4 copy; M2.7: 1.75 GB / hybrid layer
```

Rules of thumb: device share above ~55% gives usable decode; host RAM must fit in what is *free* on the machine
(this QB2 host has 249 GB, shared). Smaller experts with a skewed router (a few experts take most of the routing mass)
are best. `sim_m27.py` tells you the real hit rate once you have reference routing (step 4).

| Model | Expert params | On device | Host RAM | Drafter | Notes |
|---|---|---|---|---|---|
| MiniMax-M2.7 (done) | 225B | ~79% | 43-85 GB | DFlash | reference point |
| Ornith-1.5-397B | 386B | ~54% | ~225 GB as is: needs an int4-only "lean" host mode | DFlash (same format) | same arch as the Qwen3.5 port the overlay started on |
| Inkling-Small | ~258B | ~81% | ~60 GB | DSpark (unread) | gpt-oss-style SWA + sink attention |
| MiniMax-M3.1-Flash | ~220B total | good | OK | DFlash2 (needs new drafter code) | new sparse attention |

## 2. Drafter

`dflash_tt.py` implements z-lab's `DFlashDraftModel` (config: `dflash_config.target_layer_ids`, `mask_token_id`,
`block_size`). Same-format drafters (e.g. `ornith-ai/Ornith-1.5-397B-DFlash`) need only:
- `dflash_serve.load_drafter`: the target / drafter repo ids (`hfpath.snapshot(...)`) and the target's embedding
  table (it borrows `ref_m27.get("model.embed_tokens.weight")`: point it at your reference script);
- `dflash_tt.py` reads `block_size` from the top level; some drafters put it inside `dflash_config`.

`DFlash2DraftModel` (conv + selector: MiniMax-M3.1-Flash, Beam) and `DSparkDraftModel` (Inkling) are different
architectures: new drafter code. Bring the model up with `SPEC=0` first.

## 3. The port

The overlay patches an autoport with the **Qwen3.5 autoport layout** (`models/autoports/<name>/tt/` with
`model_config.py`, `moe.py`, `moe_dispatch.py`, `optimized_decoder.py`, `generator.py`, `model.py`); the M2.7 port was
derived from the Qwen3.5-122B one. If your model has no such port, that comes first (and is most of the work). Then the
model-specific spots in the overlay:
- `hybrid/patch.py` `load_state_dict` (`if PORT == ...`) and `hybrid/state.py` `hybrid_fused_state_dict`: how the
  checkpoint's expert tensors are named / laid out;
- `build_int4_library.py` `HYB_LIB_KEYS` (`qwen35` | `minimax_m2`): add your checkpoint's expert key style;
- the router: `router_kind` on the model args (`softmax` | `sigmoid_bias`);
- `ref_m27.py` -> `ref_<tag>.py`: CPU BF16 reference (HF decoder layers, streamed one layer at a time); it records
  per-token routing and is the quality yardstick;
- `sim_m27.py` -> `sim_<tag>.py`: set `H`, `I` and `E` at the top.

## 4. Build the artifacts

```bash
cp templates/build_artifacts.sh build_<tag>.sh   # edit the CHANGE ME block
./build_<tag>.sh texts && ./build_<tag>.sh ref     # ref: hours on CPU (REF_THREADS)
./build_<tag>.sh plan                              # prints hit rates (runs/<tag>/sim_<tag>.txt) + full-layer sets
./build_<tag>.sh int4 && ./build_<tag>.sh dev      # dev needs the device, nothing else on it
```

Needs a venv with torch (cpu), transformers, safetensors, numpy, pandas, pyarrow, and these datasets in the HF cache:
`openai/gsm8k`, `google-research-datasets/mbpp`, `CohereForAI/aya_dataset`, `livecodebench/code_generation_lite`.
M2.7 numbers: plan held-out hit rate 0.864 at 192 slots; int4 library 36 GB and device copies 33 GB (42 layers).

## 5. Serve and measure

```bash
cp templates/serve_model.sh serve/serve_<tag>.sh   # edit the values listed in its header
SPEC=0 serve/serve_<tag>.sh                         # first without the drafter; logs: serve/server_<tag>.log
python3 serve/bench.py 128 8000 <label>             # 8 short prompts: mean decode tok/s and TTFT
python3 serve/elastic_test.py --port 8000           # ~45K-token prompt, prefix-cache resend, switch back
```

M2.7 for comparison: short ~56-60 tok/s / 0.63 s TTFT; 45K prompt ~23 s TTFT, ~31 tok/s. Check the `[hybrid] host RAM`
line in the log against the container's real use (`docker stats`). Judge quality against the CPU reference
(`compare_m27.py`, teacher forced), not against an earlier run.

## 6. Package and publish

1. Upload the expert copies + plans as a public model repo (`int4_s<S>/`, `dev_s<S>/`, `plans/`, the base model's
   LICENSE + a README), e.g. with `HfApi().upload_large_folder(...)` (hard-link the files into a staging dir, no copy).
2. `cp templates/tt-model-hybrid.yaml ../../autoports/<your port>/` and edit what its header lists; pin the experts
   repo revision; commit (a dirty tree still packages, but marks it).
3. ```bash
   tt-model package --container models/autoports/<port>/tt-model-hybrid.yaml
   tt-model serve build/<name>/tt_kernel_manifest.json --profile <p> --detach   # test every profile
   tt-model push build/<name> --public          # --publish also lists it in the catalog
   docker builder prune -f                      # each package regrows ~100 GB of build cache
   ```

## 7. Gotchas we hit (2026-11-14)

- **Submodules**: a stray `umd` / `tracy` bump on the branch breaks the image build (`tlb_manager.hpp` not found).
  `git diff <base> -- tt_metal/third_party` before packaging.
- **Native libs** `hybrid/pin/pinx.so` and `hybrid/hostk/libhostk.so` are built on the host (the image has no
  compiler: `python -m hybrid.hostk`, `hybrid/pin/build.sh`) and gitignored. The package copies them from the
  working tree, so build them before `tt-model package` and check they load in the image.
- **int4 group size**: the CPU kernel needs a group size dividing 64; the builder defaults to 32. `hostk_moe returned -1`
  = mismatch.
- **Shared-KV rows SDPA** (`QWEN35_ROWS_SDPA_SHARED_FROM`) stays 0: with elastic resizing it hits
  `!is_capturing_trace` (its program depends on the KV pool shape).
- **`HYB_ELASTIC_PREWARM`** stays off: the full 24-layer round trip at startup leaves short decode ~12% slower.
- **No `--ulimit memlock`** needed: the KMD pins without memlock accounting; mlock of the int4 library is best-effort.
- **Two copies of the serve settings** (`serve_<tag>.sh` and the yaml's `serve.env`): change one, change both.
- **Disk**: the first boot writes ~110-140 GB of device images into `~/.cache/tt-model/<name>/weights`.
