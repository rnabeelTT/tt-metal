# Skill stage-gate status

What the `tt-model-bringup` gates actually say about this autoport, as measured, with the
exact command to reproduce each. Written 2026-09-16 on a QB2 (2x Blackhole p300c, 4 dies).

Run any gate with:

```bash
export TT_MODEL_BRINGUP_ROOT=~/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.3
export PYTHONPATH=$TT_MODEL_BRINGUP_ROOT/runtime
export MODEL_DIR=models/autoports/ornith_ai_ornith_1_5_35b_a3b
bash "$TT_MODEL_BRINGUP_ROOT/prompts/model_bringup_multigoal/<gate>.check.sh"; echo "EXIT=$?"
```

Exit 0 passes, 1 is advisory, 2 is critical.

| gate | exit | state |
|---|---|---|
| 06-full-model | **0** | **PASS** |
| 07-optimized-full-model | **0** | **PASS** |
| 09-vllm | 2 | blocked — see below |
| 10-optimized-vllm | 2 | blocked — see below |
| 11-tti-release | 2 | not started (`doc/tti_release/` absent) |

All five reported CRITICAL before this session, for one structural reason: the autoport
exposed no `tt/generator.py`, so **none of the packaged readiness runners could execute
against it** and no generation evidence existed anywhere. That file now exists and
implements the full `readiness_check.contract.Generator` contract.

## 06 / 07 — PASS

Two criteria, both met on real artifacts:

* **Non-degenerate free-running generation.** `readiness_autoregressive/` holds the
  unedited output of a real device run via `readiness_check.run_autoregressive`:
  256 tokens, greedy, fluent and on-prompt. The checker measures
  `adjacent_duplication` 0.0 against a 0.10 critical bar and
  `trigram_loop_fraction` 0.0159 against a 0.50 advisory bar.
* **Context contract.** `doc/context_contract.json` records target = supported = 262144
  and the gate agrees: *"Context contract OK: target=262144, supported=262144 (full HF
  context)."*

Note on the artifact, so it is not over-read: only 10 of 256 tokens match the HF
completion. That is expected rather than alarming — greedy free-running amplifies a
single disagreement forever, and this port's decode-path logit PCC is 0.75-0.94. Both
completions are coherent and on-topic; that is what the artifact supports, not
token-level agreement.

Supporting runner results, same session, single chip:

| runner | result |
|---|---|
| `run_prefill_check` | top-1 0.891 (57/64), top-5 1.000, top-100 1.000 |
| `run_teacher_forcing` | top-1 0.859 (55/64), top-5 0.984, TTFT 388 ms, decode 28.67 tok/s/user via the TRACED path |

The 28.67 tok/s/user matches the traced-with-host-readback figure rather than eager
(~3.4), which is independent evidence the trace really ran rather than silently falling
back.

## 09 / 10 — blocked by a REAL readiness gap, not by missing paperwork

`readiness_check.run_vllm_server` **enforces on-device sampling** and says why:

> On-device sampling is enforced (`sample_on_device_mode: all`). A model that cannot
> serve sampling from the device is not production-ready, so the readiness check fails
> fast there rather than papering over it with host-side sampling.

This model cannot do it. The plugin refuses, with the numbers:

```
RuntimeError: on-device sampling requires a certified TP topology (1x4 or 1x8) with at
most 65536 logits/device; got mesh=(1, 1), vocab=248320, logits/device=248320.
```

It cannot be configured around either: the runner merges `--tt-config` OVER
`{"sample_on_device_mode": "all"}` so the key cannot be removed, and the plugin asserts
the value is `all` or `decode_only` whenever the key is present, so it cannot be unset to
null. Host sampling requires the key to be **absent**. The runner therefore structurally
cannot drive this model, and it is behaving as designed.

**The fix is known and is real work.** Vocab-sharding the LM head puts 248320/4 = 62080
logits per die, which fits, and would unlock both on-device sampling and ~1.15 ms of the
head's own compute. This port's README records that path as correct standalone but
hanging in-model, with the flag left off (commit `179daf406f6`), because it needs working
collectives for the logit all-gather. That is the blocker for stages 09/10.

**What exists instead, and how to read it.** The qualitative artifact was produced by the
runner's own `qualitative` stage against a real server running HOST sampling, so the
gate's actual criterion — non-degenerate generated text at temperature 0.0 and 0.7 — is
genuinely evaluated on real completions. Nothing is hand-written or synthesised. But the
runner's on-device-sampling requirement is **UNMET**, so stages 09 and 10 should not be
reported as clean passes on their own terms.

## 11 — not started

`doc/tti_release/` does not exist. Stage 11 depends on stages 09/10 being complete, so it
is gated behind the on-device-sampling work above.

## Stages with no `.check.sh`

Stages 01-05 and 08 have no runner-side gate script; their evidence is the model's own
test suite plus the README. Current state of that suite on P150:

| test | result |
|---|---|
| `tests/test_layer.py` | 3 passed, 1 skipped (the skip is the full-attention paged case, which needs a page table a bare layer does not build) |
| `tests/test_moe_block.py` | MoE block PCC 0.99976 against a 0.99978 bf16 ceiling |
| `tests/test_model.py` | top-1 3/3; logit PCC 0.9717 / 0.9713 / 0.9433 against a 0.95 bfp4 bar — **clears it on 2 of 3 prompts** |

`test_layer.py`'s T=256 Gated DeltaNet case had been failing at every math fidelity
before this session (statically allocated circular buffers clashing with L1 buffers by
5120 bytes); stage 05 had recorded a PASS without it passing. Fixed in `5925b92cfbd`.

`test_model.py` has never passed outright, and will not be closed by tuning precision:
the binding constraint on the remaining prompt is discrete router expert-selection
agreement with HF, not arithmetic. See the README's accuracy section.

## Open blockers, unchanged by this session

1. **Decode is not bit-reproducible** (`NONDETERMINISM.md`). Still open. Three more
   hypotheses were ruled out by measurement this session, including the Blackhole
   matmul-throttle mechanism, which is recorded there in full so it is not re-derived.
2. **Batched serving deadlocks**, so `max_num_seqs` ships at 1 and concurrency comes from
   lanes rather than from batching.
3. **4-chip decode hangs.** Engine init and prefill now work on a 4-die mesh (the fabric
   fix in `d430d70c898`), but decode goes silent. One-big-mesh multi-die serving is
   therefore still not viable; `tools/serve_lanes.sh` remains the measured multi-die
   deployment.
4. **On-device sampling** — the 09/10 blocker above.
