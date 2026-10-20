# GLM-5.3-Flash (E192): vLLM integration design (stage 9)

Status (2026-10-18): design and code complete on the host; no device job ran yet. The generator serving API of
section 9 exists in `tt/generator.py` and the adapter uses it. Host tests: `tests/test_generator_vllm_host.py`, 19 passed.
The serving profiles are data: `serving_profiles.json` (section 6). Device steps that remain:
`doc/tti_release/RELEASE_CHECKLIST.md` part A.

Decisions (coordinator, 2026-10-17): bundle weights = parent `zai-org/GLM-5.3-Flash` + shipped `tt/e192_mask.json`;
keep `serve_overlay`; supported context = the 1-slot value, per-profile limits as evidence; default profile after the
G9 measurement; keep `--hf-overrides`; MTP decided in stage 7; keep the template's default reasoning effort.

Model: `zai-org/GLM-5.3-Flash` with the E192 expert subset, on a (1, 4) Blackhole mesh (p300x2), tensor parallel 4.
Autoport: `models/autoports/zai_org_glm_5_3_flash`. Workspace: `~/glm53-flash-bringup`.

## 1. Decisions

| Topic | Decision |
|---|---|
| Adapter | `tt/generator_vllm.py`, class `Glm5NextForConditionalGeneration`. It only translates the plugin interface. It calls `Glm53Generator` (`tt/generator.py`). |
| Registration | Bundle file `vllm_bundle/glm_5_3_flash/vllm_metadata.json` (`EXTRA_MODELS_DIR`). The plugin registers `TTGlm5NextForConditionalGeneration`. The server starts with `--hf-overrides '{"architectures": ["TTGlm5NextForConditionalGeneration"]}'`. No plugin source change. |
| Precision | The stage-8 policy `combo_a_kv` (`doc/datatype_sweep/selected_precision_config.json`). The adapter sets `GLM53_PRECISION` to it when unset. fp8 e4m3 DSA latent: 6,336 B per token per die. |
| Block size | 128 tokens. The DSA pool keys of one block are one tile row. The adapter refuses other values. |
| Cache ownership | vLLM owns the block allocation (block ids, block tables, `num_blocks`). The model allocates the DSA pools for exactly that `num_blocks` in `allocate_kv_cache` and returns them. vLLM passes that object back on every call. |
| KDA state | Model-owned, constant size, one row per decode slot (as the Qwen3.5 GDN state). No vLLM KV spec. |
| Slots | The plugin gives a state slot per request. The adapter keeps a virtual-to-physical slot map. A plugin `slot_remap` changes the map, not the device state. |
| New request | Prefill writes the slot's KDA state and the DSA open-pool state from the request's own prompt. No separate reset. |
| Sampling | The generator's split sampling: model trace, then sampler trace. The sampler writes the token into the persistent token buffer. Per-request parameters go to slot rows. Multi-die lockstep is required (section 7.3). |
| Host sampling | Compatibility mode only. The plugin selects it for logprobs (4 dies), `min_p`, `logit_bias`, bad words, structured output. |
| Prefill | Whole prompts. No vLLM chunked prefill and no prefix caching in stage 9. The model chunks internally (128 / 2048). Any prompt length is valid. |
| Async decode | Implemented in the adapter (split submit / read / host format). `supports_async_decode` stays `False` until the device stale-input tests pass (`GLM53_VLLM_ASYNC_DECODE=1` turns it on for those tests). |
| Context | DRAM limits it. `serving_profiles.json` holds the numbers (section 6); estimate with the stage-8 policy: 364,544 tokens at 1 slot, 282,624 at 16, 190,464 at 32 (G9 measures them). |
| Bundle | `tt-model.yaml` from the GLM-4.5-Air template. Four profiles, rendered from `serving_profiles.json`. Weights: the parent `zai-org/GLM-5.3-Flash` at the pinned commit + `tt/e192_mask.json` (section 3.2). Weight cache on (`GLM53_CACHE_DIR`). |

## 2. Host environment (checked 2026-10-17)

| Item | Value |
|---|---|
| vLLM | 0.26.0 in `~/tt-metal/python_env` (`VLLM_TARGET_DEVICE=empty` build) |
| Plugin | `~/vllm-tt-plugin`, editable, HEAD `92403f3` (local commit that registers Qwen3.5-122B) |
| transformers | 5.12.1 in `python_env`. It does NOT know `glm5_next` (needs >= 5.16). |
| Serving overlay | `~/glm53-flash-bringup/serve_overlay`: transformers 5.19.0, tokenizers 0.23.3, huggingface_hub 2.2.0 (`pip install --target`, made by this stage). Put it first on `PYTHONPATH`. |
| tt-model-manager | `~/tt-model-manager/.venv/bin/tt-model` (schema 5.1, as GLM-4.5-Air) |

Host checks that passed with the overlay (no device):

1. vLLM `ModelConfig` accepts the E192 checkpoint with the `--hf-overrides` architecture: runner `generate`, not multimodal.
2. The plugin registers `TTGlm5NextForConditionalGeneration` from `EXTRA_MODELS_DIR`.
3. The tokenizer loads. The chat template renders (`[gMASK]<sop>...<|assistant|><think>`).
4. Parsers resolve: tool parser `glm47`, reasoning parser `glm45`.
5. `tests/test_generator_vllm_host.py`: 19 passed on 2026-10-18 (adapter against a fake generator, serving profiles,
   manifest sync, precision policy, parent-checkpoint overlay).

Findings that change the plan:

- Without the overlay, vLLM stops: "model type `glm5_next` ... Transformers does not recognize this architecture".
- vLLM 0.26 has no class for `Glm5NextForConditionalGeneration`. `ModelConfig` rejects the bare name. With the
  `--hf-overrides` name, `ModelConfig` inspects the TT class. The class therefore has the four vLLM model-protocol
  stubs (`__init__(vllm_config, prefix)`, `embed_input_ids`, `forward`, `compute_logits`). The plugin never calls them.
- vLLM's default KV spec for this config is wrong (`use_mla False`, 64 KV heads of 64). The adapter defines
  `get_kv_cache_spec`: one full-attention group, 1 head of 512 (the latent row), block 128.

### 2.1 Plugin contract used (revision 92403f3)

Call order: `init_device` (calls `get_max_tokens_all_users`, sizes the pool) -> `load_model`
(`initialize_vllm_model`) -> `allocate_kv_cache(shape, dtype, num_layers)` -> warm-up phase 1
(`warmup_model_prefill(enable_trace=False)`, `warmup_model_decode(enable_trace=False)`) -> phase 2 (both with
`enable_trace=True`) -> serving.

- Prefill kwargs: `tokens [N, L_pad]`, `page_table [N, n]`, `kv_cache`, `prompt_lens`, `start_pos`, `empty_slots`
  (state slot per request), `sampling_params` (only for device sampling).
- Decode kwargs: `tokens [B, 1]`, `start_pos [B]` (-1 = padding row), `page_table [B, n]`, `sampling_params`,
  `prompt_tokens` / `output_tokens` (penalties), `slot_remap` (full-width permutation: row i reads slot `remap[i]`),
  contract v1 commands `reload_inputs`, `reload_page_table`, `reload_sampling_params`, `reset_sampling_state`,
  `enable_trace`, `read_from_device`.
- Output: device sampling -> tokens indexed by row; host sampling -> logits `[rows, 1, V]`.
- Block 0 of vLLM's pool is the null block. vLLM never gives it to a request. Block-table padding is 0.
- `plan_decode_reload` sets `reload_inputs` after a prefill, a layout change or a sampling-mode change. It sets
  `reload_page_table` when the submitted block tables differ.
- Logprobs on 4 dies always go to host sampling (`check_perform_device_sampling`).

## 3. Registration and launch

### 3.1 Architecture name

The checkpoint says `Glm5NextForConditionalGeneration`. The bundle registers `TTGlm5NextForConditionalGeneration`.
Start the server with:

```text
--hf-overrides '{"architectures": ["TTGlm5NextForConditionalGeneration"]}'
```

The plugin keeps a name that starts with `TT`. The worker then finds `get_kv_cache_spec` on the TT class.
Alternative (not used): add a bare-name alias in `platform.py::register_tt_models()`. That changes the plugin.

### 3.2 Model path

`initialize_vllm_model` sets `GLM53_CKPT` (`resolve_checkpoint`):

1. `GLM53_CKPT` if set (the bring-up host: the E192 build).
2. Else the served model path when it is a local directory (the E192 build, or a parent snapshot that has
   `e192_mask.json`).
3. Else the local HF snapshot of the repo id at `GLM53_WEIGHTS_REVISION` (the bundle pins the parent commit
   `91b69215804f2d2036a8867f6ce44848e278dcda`, the commit of every stage), plus an overlay directory under
   `GLM53_CKPT_OVERLAY_DIR`: symlinks to the snapshot files and `tt/e192_mask.json`. The parent snapshot has 288
   experts and no mask; `CheckpointReader` reads the 192 kept experts and router rows through the mask. The adapter
   refuses a mask made for another parent.

The bundle passes `--revision <commit>` to vLLM too, so the config and the tokenizer come from the same snapshot.
Not run on device yet: the parent-snapshot path (tt-model verify, `RELEASE_CHECKLIST.md` A9). The weight-cache key has a
digest of the checkpoint index, so the parent snapshot builds its own cache (about 25 min, 124 GB) on the first start.

## 4. DSA paged cache and vLLM's block table

Model layout (per DSA layer, replicated on every die, `tt/dsa.py`):

- latent `[1, 1, NB*128, 512]` bf16, block b = rows `[128 b, 128 b + 128)`;
- pooled keys, 32 per block (one tile row);
- page table `[S, T/128 + 1]`; the extra column is the indexer's one-tile overrun and points at a dummy block.

Mapping:

1. vLLM gives `num_blocks` blocks of 128 tokens. The model allocates `NB = num_blocks + 1`. Block `num_blocks` is the
   model's dummy block. vLLM never uses that id.
2. All 11 DSA layers use the same block ids (one vLLM KV group, one block table). A token's latent and pool key land in
   the same block id in every DSA layer.
3. The adapter pads every block-table row to `T/128` columns with 0. 0 is vLLM's null block. No request owns it. So a
   padded column can only write into the null block (the pool keys of a prefill's padding rows) and is only read
   under the causal limit mask. `complete_page_table` puts the dummy block in the overrun column.
4. `T` (the model's `max_context`) is `max_model_len` rounded up to 128.
5. Page growth: vLLM appends a block when a request crosses a 128-token boundary. The adapter gives the full table on
   every decode. The generator compares it on the host and copies it to the device only when a row changed (about once
   per 128 tokens per request). The plugin's `reload_page_table` is a hint only.
6. No prefix caching: a block is never shared between requests.

Cost note (stage 10): every DSA layer has its own device copy of the same page table. One shared device tensor saves 10
host-to-device copies per change.

## 5. KDA state, DSA open-pool state and slots

Per slot and per die: 34 KDA layers x (recurrent `[16, 128, 128]` fp32 + 3 conv rows), about 35.2 MiB; plus the DSA
open-pool accumulators (`acc_m`, `acc_den`, `acc_num`, `[1, 1, S, 128]` fp32 per DSA layer).

### 5.1 Virtual slot map

The plugin moves state with `slot_remap` after a batch condense. Moving 35 MiB per slot through 45 layers is slow. The
adapter keeps `SlotMap.phys` (virtual slot -> model slot) instead:

- prefill into vLLM slot v -> model slot `phys[v]`;
- decode row r -> model slot `phys[r]` (after the step's remap);
- `slot_remap` -> `phys = [phys[remap[i]] for i]` (always a permutation).

No device state moves. The device token, position and sampler seed of a request stay in its model slot. The generator
gets the row-to-slot list on every call (`slots=`). Sampling parameters and penalty history go from row order to slot
order on the host (`serving_state.rows_to_slots`, `rows_to_slots_tokens`).

### 5.2 New request in a slot

- KDA: `Glm53Model.prefill` starts from a zero state and `commit_prefill` writes the slot's recurrent and conv rows.
  The previous occupant's state is overwritten.
- DSA open pool: a prompt that ends inside a pool writes the slot's accumulators. Else the first decode step is at a
  pool start (`p % 4 == 0`) and the step resets them.
- DSA cache: the new request gets new blocks. Old rows are not visible (causal limit by position).
- Sampler: `apply_prefill_state` sets the slot's params, seed and penalty history.

## 6. max_num_seqs and max_model_len against the DRAM budget

One data file, one rule. Data: `serving_profiles.json` (next to `tt-model.yaml`). Rule: `tt/serving_profiles.py` (no
ttnn import). Readers: the adapter (`get_max_tokens_all_users`, `initialize_vllm_model`, `allocate_kv_cache`),
`tools/serving_profiles.py` (ingest / refresh / render-yaml / check / get), the workspace scripts (`09-serve.sh`,
`09-g9-probe.sh`, `11-verify.sh`), the host test and the bundle's `verify` step. No other file holds a profile number.

Rule, per die (the DSA pool is replicated on every die, the KDA state is per slot):

```text
pool_blocks(S)   = measured[S].pool_blocks                               (a completed capacity probe at S slots), else
                   floor((free_after_load - (kda_slots(S) - 2) * kda_bytes_per_slot - reserve) / dsa_bytes_per_block)
vLLM num_blocks  = pool_blocks - 1          (get_max_tokens_all_users = (pool_blocks - 1) * 128 - 128 * S;
                                             the plugin adds 128 * S and rounds up; the model adds its dummy block)
max_model_len(S) = largest L, a multiple of context_granularity(L), with ceil(L / 128) + (S - 1) + 2 <= pool_blocks(S)
model context T  = max_model_len rounded up to 128, then to context_granularity (functional_decoder asserts it)
```

`kda_slots(S) = S + S % 2` (MultichipKDA pads an odd batch). `free_after_load` is measured with 2 KDA slots (a 1-slot
build pads to 2).

Budget now (2026-10-18; status PARTIAL, `serving_profiles.json` `budget`):

| Item | Value | Source |
|---|---:|---|
| free DRAM per die after load (64 MiB trace region, 2 KDA slots) | 3,047,399,936 B | combo_a_kv 1-slot capacity probe, bind stage |
| DSA bytes per 128-token block, 11 layers | 811,008 B (6,336 B per token) | same probe |
| KDA state per slot | 35,651,584 B | stage-6 memory budget |
| reserve (prefill / warm-up / decode workspace) | 734,003,200 B (700 MiB) | stage-6 rule: 551 MB measured + margin; not re-measured for combo_a_kv |

Profiles (ESTIMATES until G9 ingests a completed probe per slot count):

| Profile | max_num_seqs | pool blocks | max_model_len |
|---|---:|---:|---:|
| `p300x2-long` | 1 | 2,852 | 364,544 |
| `p300x2` (default), `p150x4` | 16 | 2,237 | 282,624 |
| `p300x2-wide` | 32 | 1,533 | 190,464 |

G9 (device, `RELEASE_CHECKLIST.md` A2): `scripts/09-g9-probe.sh S` runs `tools/memory_probe.py capacity` at S slots on
the current tree with the selected policy, then `tools/serving_profiles.py ingest` + `refresh` + `render-yaml`. A
measured entry wins over the rule for its slot count.

The adapter builds the generator with `budget_check=False` and `dsa_layout=serving_profiles.json dsa_layout`: its own
check (`serving_profiles.check_fit`) uses the measured data. `tt/generator.BUDGET` still holds the stage-7 constants
(bf16 latent: 1,531,904 B per block, 1,984,045,568 B free) and would refuse or mis-route the stage-8 contexts (request
R1, `doc/tti_release/RELEASE_CHECKLIST.md` part C).

Context contract: the advertised context is 1,048,576. The supported context is the 1-slot value of the data file
(`serving_profiles.supported_context`). `doc/context_contract.json` still says 104,448 (stage-6 bf16 latent); the
adapter logs a warning while they differ. Update the contract from the G9 1-slot probe (request R2, same file).

Ways to more context (not stage 9 or 10): the block-sharded latent (`GLM53_DSA_SHARD=1`, bf16 only today: fp8 + shard
is not implemented), sharded pool keys, a smaller KDA state.

## 7. Sampling

### 7.1 Device path (`sample_on_device_mode: all`)

- Decode: model trace -> logits `[1, 1, S, V/4]` -> sampler trace (top-k per die, gather, `ttnn.sampling`) -> token
  into `tok_buf`, which the next model trace reads. No host argmax and no logits readback.
- Per-step state before the sampler replay (`SlotSamplingState.apply_decode`): params when the plugin reloads them,
  penalty history on `reset_sampling_state`, seed registration and position alignment, `get_new_values`.
- Prefill: the model returns the last row's logits on the device, copied to all S rows. The sampler samples with the
  request's params broadcast to all rows. The generator reads row `slot`. The sampler writes a new tensor, never
  `tok_buf` (it holds the live slots' tokens).
- The device sampler supports `top_k <= 32` (`max_top_k=32`).

### 7.2 Host path

The plugin selects host sampling for a step when one request needs it. Then the adapter returns host logits
(`[N, 1, V]` prefill, `[B, 1, V]` decode) and the plugin samples. The decode trace is the same; only the sampler trace
is skipped and the logits go to the host (about 10 MB per step at 32 rows).

### 7.3 Multi-die lockstep (required)

`ttnn.sampling` runs on every die. Each die writes its own token into its `tok_buf`. The host reads die 0. Free-running
device PRNGs drift apart with context length (Qwen3.5-122B: 162 of 600 sampled steps at 95K context; Qwen3-Coder-Next:
196 of 400 at 20K). Then die 0's token goes to the user while the other dies continue from a different token.

`serving_state.install_lockstep_sampling` ports the Coder-Next fix: identical fresh seeds before every draw, and an
all-gather of the token row with a copy of die 0's row into every die's buffer. Install it in `Glm53Model._make_sampler`
before any precompile or capture (change M3). Test: read `tok_buf` from every die after each sampled step at long
context (`ttnn.get_device_tensors`) and compare.

## 8. Decode loop, async decode and stale inputs

### 8.1 One step

1. Apply `slot_remap` to the slot map.
2. Rows -> model slots. Page table (padded) -> generator (compare; copy on change).
3. If `reload_inputs`: positions and tokens come from the host, merged with the device values (8.2).
4. Sampler state (7.1).
5. Replay the model trace and the sampler trace with `blocking=False`.
6. `read_from_device=False`: return the device token buffer. `read_decode_output(async_read=True)`: non-blocking copy
   of die 0's shard plus an event. `process_decode_output_host`: slot order -> row order.

### 8.2 Host-authoritative reload with a device that is one step ahead

Under async scheduling the host's last token and position of a running request can lag the device by one step. On a
reload, `merge_async_ahead` keeps the device pair of a slot when the device position equals the host position or the
host position + 1, unless the slot was prefilled since the last decode. This is the same rule as
`tt_transformers` `Generator.decode_forward`. It costs two small blocking reads, only on reload steps.

### 8.3 Page growth

The skill warns that older plugin revisions do not flag page growth. Revision 92403f3 compares the submitted tables
(`_page_tables_changed`). The adapter does not depend on it: it always hands over the table and the generator compares.

### 8.4 Enabling `supports_async_decode`

Set it only after these device tests pass (section 12): changed / unchanged token, changed / unchanged position,
changed / unchanged page table, page growth across 3 blocks with 4 concurrent requests against solo controls, a remap
after a request finishes, a prefill between two decodes of other requests.

## 9. Required generator / model API changes

Status 2026-10-18: G1-G8 and M1-M5 exist in `tt/generator.py`, `tt/model.py`, `tt/dsa.py` (the stage-6/7
agent). G9 is `tools/memory_probe.py capacity` + `tools/serving_profiles.py`. Open requests: `doc/tti_release/RELEASE_CHECKLIST.md`
part C.

The adapter calls exactly these. `G` = `tt/generator.py`, `M` = `tt/model.py` (or `tt/dsa.py`).

G1. `Glm53Generator(mesh, *, max_batch_size, max_seq_len, defer_caches=False, **model_kw)`.
`defer_caches=True`: load weights and allocate KDA slot state; do not allocate the DSA pools, do not install the default
page table, do not warm, do not capture. `defer_caches=False` keeps today's behaviour.

G2. `bind_kv_cache(num_blocks: int) -> list`. Allocate every DSA layer's pools for `num_blocks + 1` blocks (dummy =
`num_blocks`), build `DeviceDecodeInputs`, set `self.kv_cache` to the returned list (one entry per DSA layer, e.g.
`(latent, pooled)`), set an all-null page table. Every later call accepts `kv_cache is self.kv_cache`.

G3. `warm_programs()` (phase 1, no capture, idempotent). Compile with every slot inactive and a warm-up page table on
free blocks (ids 1..34):
- prefill of lengths 100, 2050 and 4224 into one slot. That covers 128-flash, 2048-flash, 128-dense, 2048-sparse and
  128-sparse DSA paths, and KDA first-chunk and carried-state chunks;
- the prefill head and the prefill sampler (7.1);
- one eager decode step, the host-logits readback;
- the sampler with `precompile(all_configs=True)`;
- then `reset_state()` and positions all -1.

G4. `capture_traces()` (phase 2, idempotent): the model trace, then one sampler trace per variant (penalties off / on,
force-argmax off / on), all on `logits_out` and `tok_buf`. No capture may happen later (the 2026-10-16 hang).

G5. `prefill_forward(tokens, *, page_table, kv_cache, prompt_lens, slots, sampling_params=None)`.
`page_table [N, T/128]` (row per request). `sampling_params`: list of one `SamplingParams` per request, or `None`.
Returns int64 tokens `[N]` (device sampling) or host logits `[N, 1, V]`. Marks the slots as fresh for G6.

G6. `decode_forward(..., serving=True, ...)` dispatches to `_decode_serving`. Without `serving=True` the call is the
standalone path, unchanged. Serving keywords: `(tokens, start_pos, *, page_table, kv_cache, enable_trace=True, slots, sampling_params=None,
reload_inputs=True, reload_sampling_params=False, reset_sampling_state=False, fresh_slots=(), prompt_tokens=None,
output_tokens=None, read_from_device=True, page_table_changed_hint=True)`.
`tokens`, `start_pos`, `page_table` in row order; `slots[r]` = model slot of row r; `sampling_params`, `prompt_tokens`,
`output_tokens` already in slot order (S entries). Returns: host logits `[B, V]` (no `sampling_params`), host tokens
`[B]` in row order (`read_from_device=True`), or an opaque device handle.

G7. `read_decode_output(handle, async_read=True) -> (host, [event])` and
`process_decode_output_host(host, slots) -> int64 [B]` in row order (die 0's token row).

G8. `program_cache_entries() -> int` (program-guard evidence).

G9. A probe tool (`tools/serving_memory_probe.py`) that builds the served configuration for a given `max_num_seqs`,
runs the warm-up and the capture, and records free DRAM per die (`ttnn.get_memory_view`) for `serving_tp4` (section 6).

M1. `Glm53Model(..., allocate_dsa=True)` + `allocate_dsa_caches(num_blocks)` (count includes the dummy block).
`GlmDSA.allocate_decode_state(batch, num_blocks=None)` sets `NB` / dummy before the first allocation; a pool smaller
than S full contexts starts with an all-dummy page table instead of failing the default-table assert.

M2. `Glm53Model.prefill(ids, slot, *, all_logits=False, device_logits=False)`. `device_logits=True` returns the last
row's logits on the device as `[1, 1, S, V/4]` (row copied to all S rows), not host logits.

M3. `Glm53Model(..., lockstep_sampling=False)`: True installs `serving_state.install_lockstep_sampling` right after the
`SamplingGenerator` is built and turns seed salting off. The generator sets it in `defer_caches` mode only, so the
standalone (readiness) path keeps today's sampler. Note: this mesh has `allow_force_argmax` off, so with lockstep on
even greedy decode runs the die-0 broadcast (one extra all-gather of S tokens per step).

M4. `Glm53Model.set_page_table` must accept `[S, T/128]` tables whose unused entries are 0 (vLLM null block). It does
today; keep the "dummy only in the overrun column" rule.

M5. Positions near `max_model_len`: `GlmDSA._decode_host_inputs` asserts `p < T`. vLLM never schedules
`p >= max_model_len <= T`. No change; keep the assert.

The serving helpers for G5 / G6 / M3 are drafted in `tt/serving_state.py` (`SlotSamplingState`, `merge_async_ahead`,
`install_lockstep_sampling`, `rows_to_slots`, `rows_to_slots_tokens`). Use them from `tt/generator.py`.

## 10. Prefill

- Whole prompts per plugin call; requests in one call run one after the other into their own slots.
- `start_pos != 0` raises `NotImplementedError` (`supports_chunked_prefill` and `supports_prefix_caching` are False).
- The model chunks a prompt into 2048-token pieces, the last on the smallest of (128, 2048) that holds it, and masks the
  padding rows. Any length from 1 to `max_model_len` works. The smoke uses 1, 3, 127, 129, 2049, 2051, 2381, 4223.
- A long prompt blocks decode for its prefill time (estimate 1.5-2 min at 128K). Stage 10: scheduler-level chunked
  prefill. It needs `prefill(ids_chunk, slot, start, final)` that resumes the KDA state from the slot's decode state
  and the DSA open pool, with chunk starts that are multiples of 128.
- Stage 7 adds prefill bucket traces. Then capture them in phase 2 after the decode traces (G4).

## 11. Bundle (tt-model-manager)

From the GLM-4.5-Air template (`models/autoports/zai_org_glm_4_5_air/tt-model.yaml`, Hub
`raahemnabeel/glm-4.5-air-blackhole`):

```text
models/autoports/zai_org_glm_5_3_flash/
  tt-model.yaml                         manifest; profile lines rendered from serving_profiles.json
  serving_profiles.json                 the serving profiles (section 6)
  vllm_metadata.json                    copy of the bundle metadata (as GLM-4.5-Air)
  vllm_bundle/glm_5_3_flash/vllm_metadata.json   arch + main_class for EXTRA_MODELS_DIR
  tt/generator_vllm.py                  adapter
  tt/serving_profiles.py                sizing rule (no ttnn import)
  tt/serving_state.py                   generator-side serving helpers
  tt/e192_mask.json                     expert mask (applied to the parent snapshot)
  tools/serving_profiles.py             ingest / refresh / render-yaml / check / get
  doc/context_contract.json             read by the adapter (advertised context, contract check)
  doc/datatype_sweep/selected_precision_config.json   the combo_a_kv policy (GLM53_PRECISION)
```

- Weights: `zai-org/GLM-5.3-Flash` at `91b69215804f2d2036a8867f6ce44848e278dcda` (about 330 GB, all 288 experts).
- Profiles: `p300x2` (default, 16 slots), `p300x2-long` (1), `p300x2-wide` (32), `p150x4` (= `p300x2`, not run).
- Serve args: `--hf-overrides` (TT class), `--revision <commit>`, tool parser `glm47`, reasoning parser `glm45`.
- Env: `GLM53_CACHE_DIR=/weight-cache/glm53_flash` and `GLM53_CKPT_OVERLAY_DIR=/weight-cache/glm53_ckpt_overlay`
  (tt-model mounts `~/.cache/tt-model/<name>/weights` at `/weight-cache`), `GLM53_WEIGHTS_REVISION`,
  `GLM53_TRACE_REGION` (= `trace_region_size`), `GLM53_PROGRAM_GUARD=1`, `VLLM_ENGINE_READY_TIMEOUT_S=7200` (cold first
  start: weight conversion about 25 min, then kernel compile and warm-up).
- Runtime overrides pin transformers 5.19.0, tokenizers 0.23.3, huggingface_hub 2.2.0 (glm5_next).
- The image builds the working tree, so the new C++ ops (`kda.recurrent_decode`, `dsa_decode_scores`, the mHC ops, the
  fp32 kernel fix) ship with it. Commit them before `tt-model package`.
- `tt-model` parses the manifest (`load_container_manifest(check_sources=True)`: OK, 2026-10-18) and composes the four
  serve commands (checked with `VllmPluginLauncher.serve_argv`).

Release steps: `doc/tti_release/RELEASE_CHECKLIST.md`.

## 12. Commands

Host test (no device):

```bash
source ~/glm53-flash-bringup/env.sh; cd ~/tt-metal
python -m pytest -q models/autoports/zai_org_glm_5_3_flash/tests/test_generator_vllm_host.py
python -m models.autoports.zai_org_glm_5_3_flash.tools.serving_profiles check
```

Device steps, in order, with commands and durations: `doc/tti_release/RELEASE_CHECKLIST.md` part A. Every device script
goes through `scripts/run-locked.sh` (FIFO ticket queue, reset before and after).

The server command inside `09-serve.sh` (numbers from `serving_profiles.json`, here the default profile):

```bash
PYTHONPATH=$BRINGUP_ROOT/serve_overlay:$PYTHONPATH EXTRA_MODELS_DIR=$TT_METAL_HOME/$MD/vllm_bundle \
GLM53_CACHE_DIR=~/.cache/glm53_flash GLM53_TRACE_REGION=67108864 \
python -m vllm.entrypoints.openai.api_server --model "$GLM53_CKPT" --served-model-name "$GLM53_CKPT" zai-org/GLM-5.3-Flash \
  --tokenizer "$GLM53_CKPT" --hf-overrides '{"architectures": ["TTGlm5NextForConditionalGeneration"]}' \
  --block-size 128 --max-num-seqs 16 --max-model-len 282624 --port 8131 \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
  --additional-config '{"tt": {"sample_on_device_mode": "all", "trace_region_size": 67108864, "l1_small_size": 24576, "fabric_config": "FABRIC_1D"}}'
```

Device tests to write for stage 9 (`tests/test_generator_vllm.py`, reduced model 0,3,4, no vLLM): construction through
`initialize_vllm_model`, `allocate_kv_cache` with 64 blocks, the two warm-up phases, program guard; greedy prefill token
== host argmax of the prefill logits; traced decode == eager decode for 32 steps; stale inputs (8.4); slot remap with
4 requests against solo controls; lockstep (all dies same token, sampled, T=0.8, 400 steps). Most of these are in
`~/glm53-flash-bringup/scripts/09_validate_adapter.py` today (eager control and solo controls still TODO there).

## 13. Open questions

Closed by the coordinator (2026-10-17 / 18): Q1 weights = parent + mask (pinned commit); Q2 keep the serving overlay;
Q3 supported context = the 1-slot value, per-profile limits as evidence; Q5 keep `--hf-overrides`; Q7 keep the
template's default reasoning effort.

Open:

Q4. Default profile: 16 slots (the draft) or fewer. One decode trace width serves all slots, so a 16-slot build is
slower per user also with one live request (stage 7: S=8 build 112.8 ms per step against 56.9 ms at S=1 without
`GLM53_SKIP_INACTIVE`). Decide from the stage-10 sweep (`RELEASE_CHECKLIST.md` A6).

Q6. MTP (1 DSA layer, 288 experts, about 0.7 GiB per die) trades directly against the DSA pool. Not served now.

Q8. Disk for the fresh-device verification: the parent snapshot (about 330 GB) + the bundle's weight cache (about
124 GB) need about 460 GB; the host has 295 GB free (2026-10-18). Free space first (`RELEASE_CHECKLIST.md` B1).
