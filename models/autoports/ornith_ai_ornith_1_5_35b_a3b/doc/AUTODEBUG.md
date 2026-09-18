# AUTODEBUG: vLLM serves gibberish while the model's own generator is fluent

Inspection-only. No hardware was used. Every claim carries file:line evidence and is
labelled **PROVEN** (the source shows it), **INFERRED** (follows from source plus a
runtime property I cannot observe statically), or **UNSETTLED**.

The vLLM plugin source is present on this machine at
`/home/raahem/vllm-tt-plugin/src/vllm_tt_plugin/`, so the plugin's call contract and
warmup order are proven facts here, not assumptions.

---

## TL;DR — one bug, one line

**The serving path deallocates the GDN causal-convolution state buffers whose addresses
the decode trace has baked in. It does this on every prefill, including the first, and
nothing ever re-creates them.** All 30 Gated DeltaNet layers therefore run their causal
conv window out of freed DRAM for the whole generation — and write back into it —
while the 10 full-attention layers, the paged KV cache, the decode position, the token
feed, the rope, the logits readback and the GDN *recurrent* state are all correct.

The divergence:

```
models/demos/blackhole/qwen36/tt/model.py:2895      dn.drop_derived_conv_state()
```

reached on every request via `prefill_masked_bucket` → `_reset_gdn_state_for_new_sequence()`
→ `_reset_dn_state_inplace()` (`model.py:2374` → `2881` → `2895`).
`drop_derived_conv_state()` is `ttnn.deallocate` per split-conv buffer
(`gdn/gated_deltanet.py:120-123`), and `ttnn.deallocate` defaults to `force=True`
(`ttnn/ttnn/operations/core.py:521-529`) — a real free.

This codebase already names this exact failure, in the sibling path that guards against it:

> `model.py:1957-1961` — "NOT set_external_state: it calls drop_derived_conv_state(),
> which **DEALLOCATES the split conv buffers. The captured decode trace has their
> addresses baked in, so freeing and lazily reallocating them makes the replay read
> whatever now lives there — the model serves fluent-looking garbage rather than
> failing.**"

and again in the README, as already-lived history:

> `README.md:423-425` — "…the rebind deallocating split conv buffers the decode trace had
> baked addresses for (**that last one served fluent garbage rather than failing**)."

The `max_num_seqs > 1` serving path (`prefill_paged_slots_replicated`, `model.py:1893-1970`)
implements the guard. **The `max_num_seqs == 1` path the failing server uses
(`Qwen36ForCausalLM.prefill_forward`, `qwen36_vllm.py:218-243`) has no equivalent.** The
working driver `tt/generator.py` escapes it for an unrelated reason: it captures the
decode trace *after* prefill, so the buffers it bakes are the live post-prefill ones.

**Scorecard against the five questions:** Q1 and Q5 are the root cause. **Q2 does not
apply — `prefill_paged` is not the path taken, and the documented "T ≥ 128 / silently
wrong" hazard turns out not to exist on the single-device path at all (§7.2).** Q3 and
Q4 are clean: position, token feed, rope and logits are all correct in this
configuration. Exactly one finding survives verification.

---

## 1. What actually ran — PROVEN from the plugin source

`model_runner.py:1989-1996` builds the prefill kwargs and passes
`"enable_trace": self.trace_mode in ["all"]`. The failing run logs `trace_mode=all`
(`readiness_vllm/server.log:51`), so **`enable_trace=True` reaches `prefill_forward`**.
Hence `qwen36_vllm.py:232` (`use_trace=kwargs.get("enable_trace", False)`) is **True**,
and `generator_interface.py:97-98` routes to `model.prefill_traced_chunked`. **The
untraced `model.prefill_paged` branch (`generator_interface.py:105`) never runs in this
configuration.**

The live call site is `model_runner.py:2035` (not 2041), because
`request_specific_rope = bool(model_config.uses_mrope)` is True — the checkpoint carries
`rope_parameters.mrope_section = [11, 11, 10]`.

Route for a 5-token prompt: `prefill_traced_chunked` → `chunk_size = 2048`
(`model.py:2536`) → `num_full = 5 // 2048 = 0` → the `num_full == 0` branch
(`model.py:2559-2581`) → `prefill_masked_bucket(actual_len=5, chunk_start=0)` →
`bucket = _mask_bucket_for(5) = 128` (`_PREFILL_MASK_BUCKETS = (128, 256, 512, 1024, 2048)`,
`model.py:2192`). Tokens are **not** padded by the plugin
(`input_tokens = ...[:, :max_prefill_tokens]`, `max_prefill_tokens = 5`,
`model_runner.py:1225-1228`); the pad to 128 happens model-side at `model.py:2381-2385`.

Warmup order, `model_runner.py:2651-2663`, verbatim and complete:

| # | line | call | effect on this model |
|---|---|---|---|
| 1 | 2652 | `warmup_model_prefill(enable_trace=False)` | **no-op** — `qwen36_vllm.py:381-382` returns when `not enable_trace` |
| 2 | 2653 | `warmup_model_decode(enable_trace=False)` | eager compile decode at pos 0; creates `_bucket_trace_store[1]`; builds the split-conv buffers for the first time |
| 3 | 2656-7 | `already_warmed_up_prefill = False` | re-arms step 4 |
| 4 | 2661 | `warmup_model_prefill(enable_trace=True)` | `capture_prefill_trace_chunked` — binds the external GDN buffers, sets `_chunk_inplace_state=True`, `_init_dn_zero_buffers()`, warms every masked bucket and fill width, parks the chunk trace (`model.py:1204-1243`) |
| 5 | 2663 | `warmup_model_decode(enable_trace=True)` | **captures the decode trace** (`generator.py:2162-2189`) |

So the decode trace is captured at step 5, **before any real prefill has ever run**, at
`start_pos = 0`, `token id = 0`, and an all-zero page table
(`models/common/warmup/warmup_utils.py:91-95`). `max_batch_size = max_num_seqs = 1`, so
`warmup_decode_buckets` short-circuits without a width sweep
(`generator_interface.py:16-17`) and `can_sample_on_device=False` makes
`_create_sampling_params` return `[None]` — exactly one `decode_forward` per phase.

**The warmup-captured trace is the one replayed for real requests.** `warmup_model_decode`
calls `self.decode_forward` (`warmup_utils.py:133`), which is the adapter's override, so
warmup populates `self._bucket_trace_store[1]` — the same store the request path reads at
`qwen36_vllm.py:362-369`. I checked this specifically because a key mismatch would have
silently discarded the warmup trace and caused a recapture at the first real decode,
which would have *hidden* this bug. It is consistent; there is no mismatch (§7.6).

Ordering is also correct for the external buffers: step 4's rebind
(`dn.recurrent_state = ext_rec; dn.fused_conv_state = ext_conv`, `model.py:1204-1212`)
happens **before** the step-5 decode capture, so the decode trace bakes the post-rebind
addresses — the same ones prefill writes. **There is no pre-/post-rebind address split
for the recurrent or fused state.** The primary hypothesis in the brief ("buffers rebound
out from under a decode trace") is therefore refuted for those two, and confirmed for the
third piece of state nobody was watching.

---

## 2. The causal chain, link by link

Constants: `linear_conv_kernel_dim = 4` → `K − 1 = 3` taps.
`q_dim + k_dim + v_dim = 2048 + 2048 + 4096 = 8192`. 30 GDN layers.

### 2.1 The decode conv window lives ONLY in `split_conv_state` — PROVEN

`set_external_state` sets `use_inplace_state = True` (`gdn/gated_deltanet.py:111`), which
selects `_causal_conv1d_decode_t1_split_inplace`
(`models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py:619-627`):

```python
# ttnn_gated_deltanet.py:178-193
out = ttnn.multiply(x, weight_taps[kernel_size - 1], ...)
for k in range(kernel_size - 1):
    out = ttnn.addcmul(out, conv_state_list[k], weight_taps[k], ...)   # READS split buffers
...
for k in range(kernel_size - 2):
    ttnn.copy(conv_state_list[k + 1], conv_state_list[k])              # WRITES split buffers
ttnn.copy(x, conv_state_list[kernel_size - 2])                         # WRITES split buffers
```

`conv_state_list` is `gdn.split_conv_state` (`gdn/decode.py:102`). **`fused_conv_state`
(the external `ext_conv`) is never read or written during decode.** The split buffers are
*owned clones*, not views — `ttnn.clone` at `gdn/state.py:34` — so writing `ext_conv`
does not update them.

The recurrent state is the opposite and is fine: the decode kernel writes into the state
buffer and returns it (`ttnn_delta_rule_ops.py:518` returns `state_buffer`), so
`gdn.recurrent_state = new_state` at `gdn/decode.py:131` is an identity rebind and
`ext_rec`'s baked address stays live.

### 2.2 The 90 buffer addresses are baked into the decode trace — PROVEN

`split_conv_state` is built in **host Python**, guarded on being `None`:

```python
# gdn/decode.py:46-51
if T == 1 and getattr(gdn, "split_conv_state", None) is None:
    ...
    if gdn.fused_conv_state is not None:
        split_fused_conv_state(gdn)          # gdn/state.py:26-36 -- ttnn.clone x (K-1)
```

The capture is two phases. `_prepare_decode_trace_text` runs an **executed, eager**
compile forward first (`models/tt_transformers/tt/generator.py:2057-2065` →
`_decode_forward_no_trace_text:1985-2037`), which finds `split_conv_state is None` and
allocates 3 taps × 30 layers = **90 buffers**. `_record_decode_trace_text` then opens
`ttnn.begin_trace_capture` and runs the forward again (`generator.py:2115-2139`); this
time the guard is False, so **no build op is recorded** — the trace merely reads and
writes the buffers the compile pass allocated. Their addresses are baked:

> `tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrestness.md:39` —
> "The commands contain the addresses of the operands… Whichever addresses were read from
> and written to during capture will be read from and written to again. **No allocator
> operations or checks are performed.**"

(The capture pass itself does not touch device state: `record_begin` enables sysmem bypass,
`tt_metal/distributed/fd_mesh_command_queue.cpp:1348`. So capture records without
dispatching.)

`split_conv_state` is `None` entering that compile pass because step 4 dropped it four
times over — `model.py:1210`, `1215`, each dummy `prefill_masked_bucket` inside
`warmup_prefill_masked_buckets` (`model.py:1230`), and `1233`.

Size, for the "is this memory plausibly reused?" question: each tap is logical
`[1, 1, 8192]`, which in `TILE_LAYOUT` pads to `[1, 32, 8192]` bf16 = **512 KB**.
3 × 30 = **45 MB**, freed and re-requestable on every single prefill.

### 2.3 Every prefill frees them — PROVEN

Both single-device B=1 prefill entry points re-zero GDN at sequence start:
`prefill_masked_bucket` at `model.py:2372-2374` (short prompts — all four reported
lengths) and `prefill_traced_chunked` at `model.py:2602` (long prompts). With
`_chunk_inplace_state=True` (set at step 4, `model.py:1211`),
`_reset_gdn_state_for_new_sequence` (`model.py:2877-2881`) dispatches to:

```python
# model.py:2885-2895, _reset_dn_state_inplace
ttnn.copy(self._dn_zero_recurrent, dn.recurrent_state)   # zeroes ext_rec IN PLACE   (correct)
ttnn.copy(self._dn_zero_conv, dn.fused_conv_state)       # zeroes ext_conv IN PLACE  (correct)
# split_conv_state rebuilt lazily on first decode.
dn.drop_derived_conv_state()                             # <-- FREES the 90 BAKED BUFFERS
```

The comment on line 2894 is the load-bearing wrong assumption. It is true only while no
decode trace is parked.

`ttnn.copy` is `copy(src, dst)` with `dst` modified in place — confirmed from
`ttnn/cpp/ttnn/operations/data_movement/copy/copy.hpp:12` and
`copy_nanobind.cpp:47-59` — so the zeroing and the write-back directions in this file
are all correct. The only thing wrong is the free.

### 2.4 Nothing re-creates them, because replay runs no host Python — PROVEN

Replay is `ttnn.execute_trace` and nothing else (`generator.py:2255-2256`). The lazy
rebuild at `gdn/decode.py:46-51` is host Python inside `recurrent_forward`, which a replay
never executes. The only three callers of the address-preserving refresh
`_restore_split_conv_from_fused()` are all off this path:

* `model.py:1969` — `prefill_paged_slots_replicated`, `max_num_seqs > 1` only
* `model.py:3614` — `_restore_deltanet_states`, reached only from `prime_decode_trace`, which is **demo/generator-only** by its own docstring (`generator_interface.py:108-131`)
* `gdn/gated_deltanet.py:175` — `remap_slots`, batched condense only

So on the `max_num_seqs == 1` serving path the post-prefill conv window is written
correctly into `ext_conv` (`gdn/decode.py:127`) and then **read by nothing**, while the
traced decode reads and writes a window in freed DRAM.

### 2.5 What the model therefore computes

| component | state at the first decode step | correct? |
|---|---|---|
| paged KV cache, 10 attention layers | filled by prefill via `paged_fill_cache`, real blocks only | ✅ |
| GDN recurrent state, 30 layers | written in place into `ext_rec`, the baked address (`gdn/decode.py:120-122`) | ✅ |
| decode position / token / rope / page table | re-uploaded every step (§4 Q3) | ✅ |
| prefill logits (output token 1) | masked-bucket prefill, numerically sound (§7.2) | ✅ |
| **GDN causal-conv window, 30 layers × 3 taps** | **freed DRAM** | ❌ |

The causal conv is the model's *short-range* mechanism — a depthwise FIR over the last 3
timesteps of q/k/v. Corrupt it and you get token-level repetition and echo, while
long-range content (correct attention KV plus a correct recurrent state) still leaks
through as plausible-looking fragments. The corrupted conv output then feeds the
recurrent state, which is a running accumulation, so damage compounds every step.

That predicts the observed shape exactly: **output token 1 correct (it comes from
prefill), then progressive collapse into adjacent duplication.** For
`' Paris isThe capital of of France'`, `' Paris'` is the prefill argmax; everything after
it is decode. It also predicts the specific symptom the brief calls most informative —
adjacent duplication — better than any position/token/prefill hypothesis, all of which
are independently cleared below.

---

## 3. Why (A) is fluent and (B) is not — the single difference

`tt/generator.py` captures the decode trace **after** prefill, at the real post-prefill
state:

```python
# tt/generator.py:497-508
self.reset()
logits = self._prefill_last_logits(ids, prompt_len)     # leaves split_conv_state None
...
if enable_trace:
    self._ensure_decode_trace(fed, pos)                 # capture happens HERE
```

At that capture `split_conv_state is None`, so the compile pass rebuilds it **from the
correct post-prefill `ext_conv`**, and *those* addresses are baked. The trace then reads
and writes live, correct buffers for the whole generation. **PROVEN.**

The vLLM path captures at warmup instead. Note the two independent faults this creates:
even if nothing freed the buffers, they would still hold the conv window of a dummy
2048-token all-zeros prefill (the last thing `capture_prefill_trace_chunked` wrote into
`ext_conv` at `gdn/decode.py:127`). **The finding therefore does not depend on proving
DRAM reuse** — stale-but-intact is also wrong. That is what makes it robust, and §6/E4
turns the distinction into a prediction.

### Proof by contrast: the batched sibling already implements the fix

`prefill_paged_slots_replicated` (`model.py:1893-1970`) is the same author solving the
same problem for `max_num_seqs > 1`:

```python
# model.py:1933-1938
# Hold on to the split conv buffers rather than letting prefill drop them: the
# decode trace replays against these exact addresses.
prev_split = [getattr(dn, "split_conv_state", None) for dn in dn_layers]
for dn in dn_layers:
    dn._chunk_inplace_state = False
    dn.split_conv_state = None  # prefill must not touch the decode buffers
...
# model.py:1963-1969  (finally:)
    dn.split_conv_state = split
    if split is not None:
        dn._restore_split_conv_from_fused()
```

So the contract is documented, the fix pattern exists in-tree in three places, and the
B=1 serving path never got it.

---

## 4. The five questions

**Q1 — After prefill, is the GDN recurrent and conv state at the addresses the trace
baked?** **Recurrent: yes. Fused conv: yes, but read by nothing at decode. Split conv:
NO — freed.** `_reset_dn_state_inplace` preserves the recurrent and fused addresses by
zeroing in place (`model.py:2892-2893`) and the prefill forward writes back in place
(`gdn/decode.py:120-127`). The one thing between prefill and the first replay that frees a
baked buffer is `drop_derived_conv_state()` at `model.py:2895`. **PROVEN.**

**Q2 — What does `prefill_paged` do for T = 5, 81, 157, 309, and does the
T ≥ 128 / T % 32 == 0 rule explain 157?** **It is not the path taken, so it explains
nothing about this failure — and the constraint itself does not exist on the
single-device path.** `enable_trace=True` is passed (`model_runner.py:1993`), routing to
`prefill_traced_chunked` → `prefill_masked_bucket`, where every prompt runs at a bucket of
128/256/512 with the GDN masked back to the exact `valid_len`. Separately, and
contradicting `README.md:815-816`: the single-device GDN chunk prefill goes through
`chunk_gated_delta_rule_seq_adapter`, which **zero-pads T up to a multiple of 128**
(`ttnn_delta_rule_seq.py:461-500`) and crops the output back (`:745-747`); the zero pad is
an exact identity for the delta rule (`g = 0` ⇒ flat decay, `β = 0` ⇒ `k_beta = v_beta = 0`
⇒ `L = I` ⇒ zero state contribution). The `T % 32 == 0` `TT_FATAL` is on the **TP/fused**
kernel only (`chunk_gated_delta_rule.cpp:288-290`). **So T = 5, 81, 157 and 309 would all
be numerically correct even on `prefill_paged`.** Full arithmetic and the exhaustive
rule enumeration are in §7.2. **No rule derived from the chunking arithmetic reproduces
157-passes / 5,81,309-fail, and I am not forcing one** — see §8 for what does explain it.

**Q3 — Is the decode position correct? Is the trace's position input updated per step?**
**Yes, and every input is refreshed every step, doubly forced.** `reload_inputs` is
unconditionally True because (i) `not on_device_sampling` with host sampling
(`generator.py:1840`) and (ii) the model sets
`_tt_vllm_always_refresh_decode_trace_inputs = True` at `model.py:64` — "*RoPE is
host-recomputed each step, so refresh all decode trace inputs.*" That takes the **full**
refresh branch at `generator.py:2232-2240`, which re-runs `prepare_decode_inputs_host` and
`copy_host_to_device` over all four tuple entries (tokens, `cur_pos`, the host-packed
rope, page table) into the baked buffers before `execute_trace`. The partial
page-table-only branch at `generator.py:2241-2251` (`DECODE_PAGE_TABLE_INPUT_IDX`) is
**unreachable** for this model. The position value itself is correct: for a T-token prompt
the first decode gets `start_pos = T` (`positions_np = num_tokens - 1 = 6 - 1 = 5` for
T=5, `model_runner.py:1231-1235`), and it is consumed as a *tensor* runtime arg
(`update_idxs_tensor` / `cur_pos_tensor`, `ttnn_gated_attention.py:443-465`), not a baked
kernel constant. **Not a stale-position or off-by-one bug. PROVEN.**

**Q4 — Is the token fed to step N+1 the model's own argmax from step N, or a stale
readback?** **Fresh, and properly synchronised.** The plugin calls
`decode_forward(read_from_device=False)` (`async_decode.py:867`), so `read_decode_output`
is never used; the single readback is `finalize_decode` →
`process_decode_output_host` → `Qwen36Model.process_output_decode` →
`ttnn.to_torch(tt_out)` (`model.py:3570`), which lowers to
`from_device(blocking=True)` on **CQ0** (`ttnn/cpp/ttnn/operations/core/core.cpp:55-57`)
— the same queue as `execute_trace(..., cq_id=0, blocking=False)`
(`generator.py:2256`), so the read is ordered behind trace completion by the queue.
Host argmax at temperature 0 (`model_runner.py:2186`, `2301`), written back into
`token_ids_cpu` (`:2483-2490`), which is exactly what the next step reads as `tokens`. The
async-ahead device-token merge (`generator.py:1867-1938`) is gated on
`on_device_sampling`, which is False; `supports_async_decode: False` is honoured by the
platform (`platform.py:1807-1821`). **Not a stale-logits bug. PROVEN.**

**Q5 — Does warmup leave state the first real request inherits?** **Yes — and this is the
root cause, though not in the "contaminating values" sense the question anticipates.**
Warmup leaves no bad *values* that survive: it advances the GDN state twice at position 0
(once per plugin phase), and the first prefill then zeroes `ext_rec`/`ext_conv` in place
and overwrites both correctly. Its KV writes go to physical block 0 via the all-zero
warmup page table, and any block a request maps is rewritten by prefill. What warmup
leaves is a **trace whose conv-state addresses point at buffers allocated before any
prefill existed, which the first prefill then frees.** That is why the very first request
is already broken and why this is not cross-request contamination. **PROVEN.**

---

## 5. The minimal fix

Three options. All follow the pattern already proven in-tree at `model.py:1956-1969`,
`model.py:3606-3614` and `gdn/gated_deltanet.py:170-175`. I did not edit any source.

**Fix A — narrowest, model-level, fixes every driver. Recommended.** Two edits, ~8 lines,
no allocation, every baked address preserved.

*(i)* Make the in-place reset refresh the derived window instead of freeing it. Because
`ext_conv` has just been zeroed on the line above, restoring from it *is* zeroing the
split buffers in place — the intended semantics, minus the free:

```python
# model.py:2885-2895, _reset_dn_state_inplace
ttnn.copy(self._dn_zero_recurrent, dn.recurrent_state)
ttnn.copy(self._dn_zero_conv, dn.fused_conv_state)
if dn.split_conv_state is not None:
    dn._restore_split_conv_from_fused()   # keep the trace-baked addresses
else:
    dn.drop_derived_conv_state()          # nothing derived yet
```

*(ii)* Re-seed the window after prefill, at the end of `prefill_masked_bucket` (after
`_forward_prefill_chunk_masked`, which has just written the real window into `ext_conv`
via `gdn/decode.py:127`), so decode sees the prompt's conv taps rather than zeros:

```python
for layer in self.layers:
    if not layer.is_full_attention and layer.attention.split_conv_state is not None:
        layer.attention._restore_split_conv_from_fused()
```

Both halves are required. Without (i) the addresses are already gone and (ii) is a no-op
on `None`; without (ii) decode starts from a zero conv window — less catastrophic, still
wrong. Leave `_chunk_inplace_state` alone; it is correct for the traced prefill path.

**Fix B — adapter-level, no model change.** Wrap the B=1 prefill in
`Qwen36ForCausalLM.prefill_forward` (`qwen36_vllm.py:218-243`) exactly as
`prefill_paged_slots_replicated` wraps the batched one: stash `split_conv_state`, set the
attribute to `None` so the reset's drop cannot reach it, and in a `finally` restore the
list and call `_restore_split_conv_from_fused()`. Lower blast radius; leaves the trap in
place for the next caller.

**Fix C — not recommended alone.** Force the decode trace to be captured after the first
prefill instead of at warmup (i.e. make the B=1 path behave like `prime_decode_trace`).
This is what accidentally keeps driver (A) fluent, but it costs a capture on the first
request and does **not** fix prompts 2..N (§7.3).

---

## 6. Cheapest hardware experiments, in order

Standing caveat from `NONDETERMINISM.md`: greedy output on this port varies run to run at
the argmax level (a same-step MoE-bank race, separately localised and unrelated to this
finding). The fluent/gibberish distinction is far above that noise floor, so these
experiments are safe — just do not read token-level agreement into them.

**E1 — Two prompts through the working generator, one process. No vLLM, ~2 minutes.
This is the decisive test.** Build `OrnithGenerator`, call `generate()` twice.
Prediction: **prompt 1 fluent, prompt 2 gibberish** — `reset()` frees the baked buffers
(`tt/generator.py:573-575` → `model.py:2895`) and `_decode_trace_primed` prevents a
recapture (`generator.py:535-536`). If prompt 2 is fluent, this diagnosis is wrong and I
would want to know immediately.

**E2 — Print the buffer addresses. ~5 minutes, zero risk.** For one GDN layer, log
`[b.buffer_address() for b in dn.split_conv_state]` at three points around the first
request: just after the Phase-2 decode capture, just after `_reset_dn_state_inplace`, and
at the first decode step. Expected: three addresses, then `None`, then still `None`. That
single observation converts the last inferred edge of the chain into a direct observation.

**E3 — `TT_METAL_TRACE_ALLOC_TRACKING=1`.** The framework has a checker for exactly this
class of fault: `verify_before_replay` (`ttnn/ttnn/unsafe_allocation_tracker.py:67`, wired
at `ttnn/ttnn/__init__.py:161`). The 90 split buffers are allocated while the chunked
prefill trace is already live and are **not** covered by `_mark_trace_buffers_corruptible`
(which annotates only `device_inputs` and `tt_out_trace`, `generator.py:2080`, `2140`), so
the tracker should fire. Note it fires on the **prefill** trace's first replay, i.e. you
need a prompt > 2048 tokens to trip it.

**E4 — Apply Fix A and re-run the four completion prompts plus the six chat prompts.**
Expect all ten to become coherent.

**E5 — Separate "freed and reused" from "stale but intact" (this is what explains 157).**
Comment out only the `ttnn.deallocate` inside `drop_derived_conv_state` — leak the
buffers, keep the addresses live — while still setting `split_conv_state = None`.
Prediction: output becomes **fluent but off-prompt for every prompt**, the same flavour as
the 157-token case, because decode then runs from the warmup dummy prefill's conv window
rather than from reclaimed memory. That confirms the §8 reading of the length dependence.

---

## 7. Other potential issues — real, but they do NOT explain this failure

Each was checked as a candidate and demoted with a reason.

**7.1 `prefill_paged` is unsafe on the serving binding — three faults, all latent.**
Reached only when `enable_trace` is falsy (`generator_interface.py:97-105`), which this
run is not. `model.py:3364` `self.reset_state(batch_size=B)`:
(a) reallocates `recurrent_state` at a new address (`gdn/state.py:23`), unbinding the layer
from `ext_rec` — survivable only because `model.py:3419` happens to copy the result back
in the correct direction;
(b) frees the split-conv buffers (`gdn/gated_deltanet.py:189`) — the §2.3 fault again;
(c) with `_chunk_inplace_state=True`, the binding the traced path requires, the next GDN
prefill forward reaches `gdn/decode.py:124-127` with `fused_conv_state is None` and **dies
in `ttnn.copy(new_fused_conv, None)`**. The author documents (c) verbatim at
`model.py:1925-1931`. Practical consequence: **`prefill_paged` would crash, not serve
gibberish**, in the current serving configuration — independent confirmation that the
traced path is the one that ran. Worth fixing before anything routes there.

**7.2 `README.md:815-816` is wrong about the single-device GDN prefill.** It claims
"*The GDN chunk-prefill kernel needs T ≥ 128, and T a multiple of 32. Below that the
single-device path is silently wrong while the TP path errors out.*" The second clause is
correct: `chunk_gated_delta_rule.cpp:288-290` is a `TT_FATAL` on `T % C != 0` with
`C = 32`, reached only from `gdn/tp.py:551` (`num_devices > 1`). **The first clause has no
support in the single-device call chain.** That chain is `model.py:3379-3380` →
`layer.py:267-270` (`chunk_size=128` default) → `gdn/decode.py:56` →
`ttnn_gated_deltanet.py:865-877` → `chunk_gated_delta_rule_seq_adapter`, which pads and
crops as described in Q2. Instantiated:

| | T=5 | T=81 | T=157 | T=309 |
|---|---|---|---|---|
| `pad_len = (128 − T % 128) % 128` | 123 | 47 | 99 | 75 |
| `L`, `num_chunks` | 128, 1 | 128, 1 | 256, 2 | 384, 3 |
| output crop `o[:, :T, :]` | ✓ | ✓ | ✓ | ✓ |
| conv state slice `x_padded[T : T+3]` | `x[2..4]` ✓ | `x[78..80]` ✓ | `x[154..156]` ✓ | `x[306..308]` ✓ |
| verdict | correct | correct | correct | correct |

The only C++ constraint is on the *chunk* size, not T:
`gated_delta_attn_device_operation.cpp:43-50` `TT_FATAL(C == 128)`. Every candidate rule
(`T ≥ 128`, `T % 32`, `T % 64`, `T % 128`, "no remainder chunk", `num_chunks == 1`, the L1
threshold at T ≤ 128) fails against at least one of the four observations; the two that fit
157 alone (`num_chunks == 2`, `T % 32 == 29`) have no mechanism in the kernel and are
overfits. The masked-bucket path used in the failing run always feeds the GDN a multiple
of 128 anyway (`model.py:2192-2200`), including the `((length+127)//128)*128` fallback.
**Fix the README; add an assert or an auto-reroute in `prefill_paged` if you want the TP
constraint enforced uniformly, but the silent-wrongness claim should not be propagated.**

**7.3 The working generator is broken from its second prompt onward — same root cause.**
`tt/generator.py:554-577` `reset()` frees the baked buffers, via
`_reset_gdn_state_for_new_sequence()` (line 573 → `model.py:2895`) and again explicitly at
line 575, while `_decode_trace_primed` stays True so `_ensure_decode_trace` returns early.
The readiness runners drive one prompt per process, which is why evidence (A) is clean.
**This is E1**, the cheapest confirmation of the headline. Fix A closes it too.

**7.4 The MoE `T < 32` seam is not reached here.** `moe.py` dispatches
`decode = T == 1 or T < self.CHUNK` (CHUNK=32), and the `_decode_gather` combine uses
`ttnn.sum` (which re-enables `fp32_dest_acc_en` by omission) while the prefill combine
uses `fast_reduce_nc` — the asymmetry a previous investigation flagged. A raw 5-token
prefill *would* hit it, but the masked-bucket path runs every prompt at T = bucket ≥ 128,
so **no prompt in the failing run reaches the MoE decode seam during prefill.** Cleared
for this failure; the asymmetry remains worth resolving on its own merits.

**7.5 Phase 1 of the plugin warmup compiles no prefill program for this model.**
`model_runner.py:2652` calls `warmup_model_prefill(enable_trace=False)`, which
`qwen36_vllm.py:381-382` makes a no-op — so the plugin's own stated invariant ("Phase 1:
compile all code paths", `model_runner.py:2619-2635`) is unmet for prefill. Harmless in
practice because `capture_prefill_trace_chunked` runs its own warmups before
`begin_trace_capture` (`model.py:1214-1230`) and the decode trace is captured afterwards.
Worth tidying.

**7.6 Decode bucketing swaps the trace store at B=1 — checked, consistent.**
`qwen36_vllm.py:362-369` replaces `self.trace_ids_decode` with `_bucket_trace_store[B]`. I
checked this closely because a mismatch would have discarded the warmup trace and forced a
recapture at the first real decode — which would have masked the headline bug and created
a different one (an unguarded double forward advancing the non-idempotent GDN state, the
hazard `generator_interface.py:116-122` warns about). Warmup reaches `decode_forward`
through `warmup_utils.py:133`, so warmup and requests share `store[1]`. **No bug.**
Recorded because it is load-bearing for the diagnosis.

**7.7 Two redundant GDN advances at position 0 during warmup.** Phase 1's eager decode and
Phase 2's compile pass each execute a real forward at pos 0 with token 0; the capture pass
itself does not (bypass mode, `fd_mesh_command_queue.cpp:1348`). Phase 2's compile pass is
unnecessary at B=1 — the `max_batch_size > 1` path avoids it with
`skip_trace_precompile=True` (`generator_interface.py:44-52`). Harmless because the first
prefill re-zeros `ext_rec`/`ext_conv` in place. Minor waste, not a correctness issue.

**7.8 Masked-bucket and paged-fill program coverage is complete.** I checked the
"request-time compile clobbers the parked trace" trap (tt-metal #48536) as an alternative
explanation for a non-monotonic length dependence, and it is closed: all five buckets are
warmed (`model.py:2443-2456`); `_warmup_paged_fill_widths` warms every fill width 1..N per
bucket (`model.py:2486-2508`), covering the 1/2/3/5-block fills that T = 5/81/157/309 need;
and the short-prompt path pads or clips the page table to the warmed width
(`model.py:2566-2578`). **Cleared.**

**7.9 Doc bug: the `prefill_dispatch` block-count warning is stale.**
`generator_interface.py:86-88` says "*the masked path writes K/V for the full bucket, so
the page_table must map enough blocks to cover the rounded-up bucket length (<= 2048 -> 32
blocks of 64)*". `_forward_prefill_chunk_masked` no longer does that: `model.py:2227-2230`
clamps the fill to `blkN = num_blocks_in_seq(chunk_start + valid_len, block_size)` — 1
block for a 5-token prompt, matching the 1 block vLLM allocates. The full page table is
still handed to SDPA, but the trailing zeros point at vLLM's reserved null block
(`vllm/v1/core/block_pool.py:188-191`) and causal masking means the selected row
(`actual_len - 1`) never reads past the real prompt. **Doc bug, not a serving bug.**

**7.10 Minor: the prefill-warmup page table contains out-of-range block ids.**
`qwen36_vllm.py:388-393` builds `torch.arange(ceil(4097/32)*32) = arange(4128)`, so ids
4097..4127 index past a 4097-block cache. Harmless — warmup only touches blocks 0..31, and
requests are padded to that width with zeros — but it is an out-of-range id sitting in a
device buffer.

**7.11 Test gap.** Nothing in `models/demos/blackhole/qwen36/tests/` exercises the serving
order "capture the decode trace at warmup → run a real prefill → replay". The closest
artefact, `tools/repro_prefill_after_trace.py`, structurally cannot reproduce it: it
captures via `prime_decode_trace` (*after* prefill, line 61) and prefills through
`prefill_paged_slots_replicated` (line 53) — the one path that already saves and restores
the split buffers. That is why the model-level repro passes while serving fails. A
regression test asserting that each `split_conv_state` buffer address is unchanged across
a prefill, once a decode trace is parked, would have caught this and would catch a
regression of Fix A.

---

## 8. What I could not settle

* **Whether the freed 45 MB is actually reused by the prefill that freed it**, and hence
  whether the corruption is reclaimed-memory garbage or merely the stale warmup window.
  **INFERRED, and it is the one open edge of the chain.** Both sub-cases are wrong and both
  are fixed by Fix A, so the headline does not depend on it — but it is what would explain
  the *flavour* of each failure, and it is my best account of the non-monotonic length
  dependence that no chunking rule explains (§7.2): what lands in those addresses depends
  on the prefill's allocation pattern, which depends on the bucket (128/256/512) and the
  prompt length, non-monotonically. On that reading T=157 landed in the "stale but intact"
  regime — fluent but off-prompt, which is this codebase's own named signature for reading
  wrong state — and the others in the "reused" regime. **E2 and E5 settle it.**
* **The exact token-level onset.** With 3 taps the garbage window shifts out over three
  decode steps while poisoning the recurrent state permanently, so the onset should be
  within the first few tokens and the damage permanent. Consistent with "the first token or
  two are often correct"; I cannot pin the index from source.
* **The six chat completions all beginning with `"The"`.** I treated this as a possible
  counter-signal that prefill is *also* broken, since prefill correctness is central to
  this diagnosis. It is consistent with correct prefill: these are chat-templated prompts
  for a thinking model, the assistant turn opens inside a reasoning block, and
  `"The user is"` — visible verbatim in the prompt-4 output — is the canonical opening of a
  Qwen reasoning trace, so `"The"` is a plausible top-1 for all six rather than evidence of
  a context-independent prefill. **Consistent, not proven.** To settle it, compare the
  server's prefill logits against `run_prefill_check`'s 0.891 top-1 on the same prompts; if
  prefill top-1 holds up under the server, prefill is exonerated outright.
* **Whether any *second* independent defect is hiding behind this one.** Every other
  candidate I could construct from the five questions was traced and cleared (§4, §7), so I
  expect Fix A to restore coherent serving outright — but the remaining logit-PCC weakness
  this port already documents (decode-path PCC 0.75–0.94) will still cap agreement with HF,
  and the MoE nondeterminism will still make greedy output run-to-run variable.
