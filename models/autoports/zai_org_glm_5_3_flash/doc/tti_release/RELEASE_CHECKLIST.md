# GLM-5.3-Flash (E192) on p300x2: stages 9-11 device queue and release checklist

Status 2026-10-18: host work done, no device job ran for stages 9-11. Template: the GLM-4.5-Air release
(`models/autoports/zai_org_glm_4_5_air`, `~/glm45-air-bringup`, Hub `raahemnabeel/glm-4.5-air-blackhole`) and the
Qwen3.5-122B TTI run (`models/autoports/qwen_qwen3_5_122b_a10b/doc/tti_release/RUN_NOTES.md`).

Conventions:

- `S=~/glm53-flash-bringup/scripts`. Every device script below takes a ticket in the FIFO device queue
  (`$S/run-locked.sh`: `tt-smi -r` before and after). Do not run `tt-smi -r` by hand while a ticket is live.
- Serving numbers come from `serving_profiles.json` only. After a change, run
  `python -m models.autoports.zai_org_glm_5_3_flash.tools.serving_profiles check` (also a host test).
- Durations are estimates. Basis: warm weight load 110 s (weight cache), cold load about 25 min; stage-6 1-slot
  capacity probe (104K-token prefill 170 s); block-sharded probe (281K-token prefill 427 s); GLM-4.5-Air server boot
  about 10 min; decode about 53 ms per token at 1 slot.

## A. Device steps, in order

| # | Step | Command | Estimate | Pass condition |
|---|---|---|---:|---|
| A0 | Preconditions (host) | stage 6/7 rework and stage 8 flip committed; `pytest -q $MD/tests/test_generator_vllm_host.py` | 1 min | 19 passed; `serving_profiles check` OK |
| A1 | Adapter validation, reduced model (layers 0, 3, 4), no vLLM | `SKIP_SERVE=1 $S/09-validate-adapter.sh` | 15 min | `VALIDATE_RESULT PASS`: warm-up phases, program guard, prefill greedy == host argmax (lengths 37 / 300 / 2051), 24 traced steps (sync + async read), remap, lockstep 0/64 |
| A2 | G9 capacity probes, full model, selected policy | `INGEST=1 $S/09-g9-probe.sh 1`, then `16`, then `32` | 25 + 25 + 20 min | probe exit 0, `copy_accuracy` 1.0; `serving_profiles.json` `measured` has 1 / 16 / 32; `tt-model.yaml` re-rendered; profiles `status: measured` |
| A3 | Reduced vLLM server smoke (inner loop, never evidence) | `PROFILE_NAME=reduced NOSTAGES=1 SMOKE=1 BURST=4 $S/09-serve.sh` | 20 min | server healthy; `SMOKE_RESULT PASS`; non-aligned lengths 1 .. 4223 answer |
| A4 | Full model, default profile: readiness smoke + served smokes | `PROFILE_NAME=p300x2 SPROFILE=smoke SMOKE=1 $S/09-serve.sh` | 60 min | sampling smoke passes; qualitative verdict OK; `SMOKE_GREEDY_REPEAT True`; burst = solo greedy; page growth over 3 blocks; logprobs / min_p / repetition-penalty requests answer |
| A5 | Full sampling profile (stage-9 evidence) | `PROFILE_NAME=p300x2 SPROFILE=full $S/09-serve.sh` | 1.5-2 h | full plugin suite; only reproducibility-only failures allowed (vllm-integration skill); `readiness_vllm-p300x2` snapshot in `logs/09/` |
| A6 | Stage 10 sweep, default profile | `PROFILE_NAME=p300x2 NOSTAGES=1 SWEEP=1 SWEEP_ARGS="--users 1,2,4,8,16 --inputs 128,1024,4096,16384,65536,131072,262144" $S/09-serve.sh` | 1-1.5 h | every row completes; `artifacts/s10/sweep_p300x2.json` |
| A7 | Wide profile: concurrency 32 | `PROFILE_NAME=wide SPROFILE=smoke SMOKE=1 BURST=32 SWEEP=1 SWEEP_ARGS="--users 1,8,16,32 --inputs 128,1024,16384,131072" $S/09-serve.sh` | 1.5 h | smokes pass at 32 concurrent; 32/32 burst streams equal their solo controls |
| A8 | Long profile: largest context | `PROFILE_NAME=long NOSTAGES=1 SMOKE=1 SMOKE_ARGS="--long 364000" SWEEP=1 SWEEP_ARGS="--users 1 --inputs 128,4096,65536,262144,364288" $S/09-serve.sh` | 1.5 h | 364,000-token prompt answers coherently; sweep rows complete (numbers: use the measured `max_model_len` after A2) |
| A9 | Async decode (optional for release) | write `tests/test_generator_vllm.py` stale-input cases first (DESIGN.md 8.4); then `ASYNC=1 PROFILE_NAME=p300x2 SPROFILE=smoke SMOKE=1 $S/09-serve.sh` | 1 h + test writing | stale token / position / page-table cases pass; served smokes equal the `ASYNC=0` controls. Else keep `supports_async_decode` False |
| A10 | TTI smoke, then release (external server, no Docker) | B6 first. Server: `PROFILE_NAME=p300x2 HOLD=1 NOSTAGES=1 $S/09-serve.sh` (port 8131). Client: see A10 commands below | smoke 15 min; release ci-nightly 3-6 h | `run.py` exit 0; run spec `docker_server=false`, `impl.code_path=models/autoports/zai_org_glm_5_3_flash`; rows classified |
| A11 | Package the bundle (no device; Docker) | `$S/11-package.sh` | 45-90 min | `package exit=0`; staged `pkgbuild/build/glm-5.3-flash-e192-blackhole/` |
| A12 | Fresh-device verification, staged bundle, cold caches | B1 first. `COLD=1 $S/11-verify.sh ~/glm53-flash-bringup/pkgbuild/build/glm-5.3-flash-e192-blackhole/tt_kernel_manifest.json p300x2` | 2-3 h (+ the 330 GB download if the parent snapshot is not cached) | `VERIFY_RESULT PASS`: served `max_model_len` = `serving_profiles.json`; smokes; long prompt; first-boot time recorded |
| A13 | Staged bundle, other profiles (warm caches) | `$S/11-verify.sh <staged manifest> p300x2-long`, then `p300x2-wide` | 45 + 45 min | `VERIFY_RESULT PASS` each |
| A14 | From the Hub, after the owner pushes | `$S/11-verify.sh raahemnabeel/glm-5.3-flash-e192-blackhole p300x2` | 45 min | `VERIFY_RESULT PASS` |

Total device time: about 16-20 h (A1-A10, A12-A14). A2 must run before A4: the profiles must be measured before the
stage-9 evidence runs. A1 + A3 can run as one job: `$S/09-validate-adapter.sh` (validation, then the reduced server).

A10 commands (TTI client, from the Qwen3.5-122B run; tag v0.22.0 = `~/tt-inference-server` HEAD `9ef9d660`):

```bash
cd ~/tt-inference-server
git apply ~/tt-metal/models/autoports/zai_org_glm_5_3_flash/doc/tti_release/tti_local_integration.patch
export HF_HUB_OFFLINE=0 SERVICE_PORT=8131
python3 run.py --dev-mode --model zai-org/GLM-5.3-Flash --device p300x2 --workflow benchmarks --service-port 8131 \
  --no-auth --skip-system-sw-validation --disable-trace-capture --limit-samples-mode smoke-test
python3 run.py --dev-mode --model zai-org/GLM-5.3-Flash --device p300x2 --workflow release --service-port 8131 \
  --no-auth --skip-system-sw-validation --limit-samples-mode ci-nightly
```

The 09-serve.sh server serves the names `$GLM53_CKPT` and `zai-org/GLM-5.3-Flash`. Before the release run, check the
written run spec under `workflow_logs/run_specs/` (`docker_server=false`, `service_port=8131`, `impl.code_path`).

## B. Release checklist

Mark each line with the evidence path when done.

B1. Disk. The fresh-device check needs the parent snapshot (about 330 GB, all 288 experts) and the bundle's own
weight cache (about 124 GB: the cache key has a digest of the checkpoint index, so the parent snapshot does not reuse
`~/.cache/glm53_flash`). Need: about 460 GB free. Free on 2026-10-18: 295 GB. Large items: HF cache of other models
(Ornith 262 GB, Qwen3.5-122B 234 GB, Qwen3-Coder-Next 149 GB, Qwen3-Coder-30B 57 GB), `~/.cache/glm53_flash` 124 GB
(the bring-up cache; keep it until stage 10 ends), `~/.cache/tt-metal-cache` 79 GB. The owner decides what to remove.

B2. Code. Commit the C++ ops (kda / dsa / mhc, fp32 kernel fix) and the autoport before A11: `tt-model package`
builds the working tree. `11-package.sh` prints the number of dirty files; it must be 0 for the release build.

B3. Serving profiles. All four profiles `status: measured` (A2). `default_profile` decided from A6 (DESIGN.md Q4). Then
`serving_profiles.py refresh -w`, `render-yaml -w`, `check`.

B4. Context contract. `doc/context_contract.json` `supported_context` = `serving_profiles.supported_context` (the
1-slot measured value), `capacity_evidence` = the A2 1-slot probe, `capability_reduction.to` the same value. The
adapter warns while they differ; the stage-9 gate compares the served `max_model_len` with the contract.

B5. Precision. The server log shows `precision .../selected_precision_config.json` (adapter line) and the policy
readback of `tt/precision.py`. The stage-8 flip may change the `GLM53_PRECISION` default; the adapter sets it
explicitly, so the served policy is `combo_a_kv` in both cases.

B6. TTI integration. Apply `tti_local_integration.patch` (checked with `git apply --check` on v0.22.0). Fill the
eval references or record why none exists (part D). After A2, make the dev-catalog `max_concurrency` /
`max_context` / `trace_region_size` equal the default profile.

B7. Weights pin. `tt-model.yaml` `weights.revision` = `serving_profiles.json` `weights_revision` =
`91b69215804f2d2036a8867f6ce44848e278dcda` (the parent commit of the E192 build). `--revision` and
`GLM53_WEIGHTS_REVISION` are rendered from the same value.

B8. Permissions. `11-package.sh` runs `chmod -R o+rX` on the shipped trees (GLM-4.5-Air pitfall: 0660 files are not
readable in the image).

B9. Model card (`tt-model.yaml` `card`; `tt-model package` renders it to the Hub README). Replace every TODO:

| Card section | Status | Source |
|---|---|---|
| What this is (E192, parent + mask, text only) | done | - |
| Use with your client (thinking always on, parsers) | done; check `reasoning_effort` behaviour in A4 | A4 smoke |
| First start, disk and time | estimates; replace with A12 measured first-boot time and disk | A12 |
| Profiles | rendered from `serving_profiles.json`; estimates until A2 | A2 |
| Precision | done (stage 8 `combo_a_kv`) | `doc/datatype_sweep/` |
| Accuracy: teacher forcing, 3 AIME24 prompts | done (stage 8) | `doc/datatype_sweep/README.md` |
| Accuracy: long-context gate | TODO | stage 8 long-context TT-vs-TT gate |
| Accuracy: vLLM conformance | TODO | A5 |
| Accuracy: TTI evals (ci-nightly subset, label it so) | TODO | A10 |
| Performance tables (latency batch 1, concurrency) | TODO | A6-A8: `python3 ~/glm45-air-bringup/scripts/card_tables.py "p300x2 (4 dies)=$A/sweep_p300x2.json" ...` |
| Limitations | done; re-check after A6 (decode width) and A9 (async) | - |

B10. Package, verify, publish. A11 -> A12 (cold) -> A13. The owner pushes:
`tt-model -v push ~/glm53-flash-bringup/pkgbuild/build/glm-5.3-flash-e192-blackhole` (private), then `--publish` /
the catalog when the owner decides (GLM-4.5-Air: private first, public + catalog later). Then A14.

B11. Copy-back. `doc/tti_release/`: release report markdown, report data JSON, run spec, runtime model spec, small
benchmark JSONs, `RUN_NOTES.md` with the commands, versions, server mode, reset actions, the line
`Autoport implementation check: models/autoports/zai_org_glm_5_3_flash ... matched`, release status and the row
classification (`fixed` / `issue-waived` / `readiness-fail`). Do not copy weights, caches, `.env`, tensor dumps or raw
profiler CSVs. Remove `.env` from the TTI checkout and revert the patch after the run.

B12. Stage reviews. Stage 9, 10 and 11 each close with a `stage-review` clean-pass and a local commit.

## C. Open requests to other owners

R1 (generator owner, `tt/generator.py`). `BUDGET` holds stage-7 constants: `free_after_load_bytes` 1,984,045,568 and
`block_bytes.replicated` 1,531,904 (bf16 latent). The selected policy measures 3,047,399,936 B free and 811,008 B per
block (fp8 latent, bfp4 projections). Effects on the standalone / readiness path: `plan_context(1)` gives a default
context of 102,400 instead of about 364K; a larger request selects the `sharded` layout (which `Glm53Model` refuses with
an fp8 latent: "the block-sharded DSA cache supports the bf16 latent only") or raises ValueError. Request: take the budget from
`tt/serving_profiles.py` (`load()["budget"]`, `pool_blocks(data, S)`), or at least derive `block_bytes` from the
active latent dtype and never select `sharded` when the latent is not bf16. Serving is not affected: the adapter passes
`budget_check=False` and `dsa_layout` from the data file and checks the fit itself.

R2 (context-contract owner, `doc/context_contract.json`). After A2: set `supported_context`, `capacity_evidence` and
`capability_reduction` from the 1-slot G9 probe (`serving_profiles.supported_context`). Now: 104,448 (stage 6, bf16
latent); the data file estimate: 364,544.

R3 (generator owner). `TRACE_REGION_SIZE` default (64 MiB) duplicates `serving_profiles.json` `trace_region_bytes`.
Request: read the default from the data file, so the readiness tools and the server open the same trace region.

R4 (precision owner, `tt/precision.py`). The flip: `selected_precision_config.json` says "GLM53_PRECISION unset = this
file", `tt/precision.py` still means "unset = stage-7 policy". The adapter sets the path explicitly. Request: confirm
that the weight-cache key is the same for the explicit path and for the flipped default (else the bundle cache and the
bring-up cache differ in name only).

R5 (generator owner, test). `Glm53Generator(defer_caches=True, budget_check=False, dsa_layout="replicated")` with the
fp8 latent is the serving build. The G9 probe (`tools/memory_probe.py capacity`) builds `budget_check=False` without
`dsa_layout`; with `GLM53_DSA_SHARD` unset it takes `replicated` too. Keep that default, or pass `dsa_layout` from the
data file in the probe.

## D. Open decisions

- Default profile (DESIGN.md Q4): decide from A6 / A7.
- TTI accuracy reference: no score of the E192 model exists (the published 288-expert scores do not apply). Options:
  (a) a same-request reference on the ci-nightly subset (HF on CPU is not practical for thinking-length outputs),
  (b) record the TTI accuracy rows as informational, with this reason in `RUN_NOTES.md`; the release status then
  cannot be `release-readiness-pass` without an owner waiver.
- MTP (DESIGN.md Q6): not served.
