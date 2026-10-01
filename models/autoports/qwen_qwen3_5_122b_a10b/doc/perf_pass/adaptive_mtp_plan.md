# Adaptive MTP: speculative decoding on by default in a multi-user server (2026-10-01)

Single-user MTP beats plain decode (53.7 vs 31.3 t/s/u greedy; sampled within 7-9 ms per pass of greedy); the batched
multi-user step loses at 2-4 users (its pass costs like a width-8 / 16 decode step). A server runs with
`--max-num-seqs 32`. Adaptive MTP = the single-user MTP step whenever EXACTLY ONE request is live, the plain batched
traced decode otherwise, with the request's state handed over at every switch. Nothing changes for the plain path when
`QWEN35_MTP` is unset.

## State that crosses a switch (slot s)
| state | plain decode holds it in | MTP step holds it in | enter (plain -> MTP) | leave (MTP -> plain) |
|---|---|---|---|---|
| GDN recurrent state + conv taps | batched buffers row s | a dedicated persistent B=1 "MTP scratch" (bound onto the GDN layers for the phase) | slice row s -> copy into the scratch; rebuild conv_carry | `write_slot(s, scratch)` (slice_write) |
| main-model paged KV | paged, by page-table row | same (the pass writes per-row K/V through the slot's row) | `set_page_table(row)` | - |
| last token / position | device decode inputs (the device feeds its sampled token back) | runner `_tok` / `_pos` (host) | from the plugin (host authoritative) | the next plain step is forced host-authoritative (`reset_batch`; slot marked freshly prefilled so the stale device token is not kept) |
| head hidden of the last token | `hid_prev` [1,1,Bmax,H]: written INSIDE every plain decode step (post hook: final-normed hidden rows) | runner `d_hid` | `d_hid <- hid_prev[s]` (one-hot pick) | `hid_prev[s] <- d_hid` (one-hot row replace) |
| head paged KV (every past position) | written INSIDE every plain decode step (pre hook: K/V rows of (token in, hid_prev) at the step's positions; the head's mixer + attention norm + QKV projection + RoPE + paged update, no SDPA / MoE) | the step's `head_rows` / draft writes | - | - |
| sampler state (seeds, penalties) | on-device sampler per slot | SpecSampler per request (no penalties) | fresh SpecSampler | `reset_batch` recounts penalties from the plugin's output tokens |

The head's prompt rows always run inside the traced prefill (in_trace), for every request, so the head KV is complete
from position 0 for every slot at all times; a request that spends part of its life under plain decode drafts as well
as one that never did. Cost of the hooks on a plain step: ~1 extra small layer-fragment per step (mixer + norm + fused
QKV + 2 paged updates), expected ~0.3-0.5 ms at width 32.

## Switch rules (generator_vllm._adaptive_decode)
0. a prefill ran since the phase's last step (`prefill_forward` only SUSPENDS the phase: unbinds the scratch, keeps
   its content): the phase's request finished -> `drop` (no write-back: the common sequential case, saves ~36
   slot writes of TTFT), it is still the only live row -> `resume` (rebind), it has company -> `leave` (write-back).
1. `slot_remap` present: leave the phase (if any), remap GDN slots, permute `hid_prev` rows.
2. exactly one live row s: enter(s) if not in phase for s (token / position: the device decode inputs when they are
   one step ahead of the host view, else the host's); run the MTP step; return its 1..k+1 ids on row s.
3. otherwise: leave the phase (if any); plain decode, forced host-authoritative right after a leave with the
   model's own last token / position substituted for the left slot (the host view may lag under async scheduling).

## Warm-up / capture order (all programs before any trace)
`_mtp.warm()` (MTP scratch bound: step programs; also enter/leave for every slot) -> plain eager decode at every width
(hooks inside -> their programs at every width) -> traces: plain decode traces at every width (hooks captured) ->
MTP traces (scratch bound) -> prefill bucket traces.

## Validation
Reduced model served with `--max-num-seqs 4`: two concurrent requests (plain path, hooks on), then a single request
(MTP phase), then a request admitted during a phase (leave -> prefill -> plain) and the survivor after one finishes
(enter mid-request): temperature-0 texts must equal the single-user served texts. Then the full model: the c = 1
benchmark (MTP numbers) and the c = 8 / 32 benchmarks (plain numbers must not move beyond the hook cost).
