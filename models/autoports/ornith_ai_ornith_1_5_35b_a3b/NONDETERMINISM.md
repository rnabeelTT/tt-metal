# Decode is not reproducible

**Status: open, not fixed. Do not advertise reproducible greedy output.**
The fault is not in this model's routing or buffer management; it is in the expert
compute below the Python level. Evidence and the exact reproducers are below.

## What a user sees

Served through vLLM at temperature 0 with a fixed seed, the same prompt returns a
different completion most runs. Eight runs of one prompt gave **six distinct answers**,
one degenerate ("positive integers" in place of an answer). Greedy decoding is supposed
to be reproducible, so this is a release blocker.

Bitwise, batch 1, GDN + KV state zeroed, same token at the same position: logits differ
between trials by up to ~1.5 and the argmax flips between two near-tied ids.

## The one fact that localises it

Instrumenting every routed-expert bank to record what goes in and what comes out, over
three independent runs:

| | x_in | scores | indices | bank_out |
|---|---|---|---|---|
| run 1, first divergence at L15 | 0 | 0 | 0 | 3.66e-04 |
| run 2, first divergence at L20 | 0 | 0 | 0 | 1.22e-04 |
| run 3, first divergence at L27 | 0 | 0 | 0 | 1.53e-04 |

**The expert bank receives bitwise-identical activations, routing weights AND expert
indices, and returns a different result.** The router is exonerated: routing is
bit-identical every time. So is everything upstream — in every run the same layer's
attention/GDN output was bitwise identical, so the error is *injected* in the
feed-forward, not propagated into it. The layer index differs every run (L00, L02, L04,
L09, L11, L15, L16, L20, L21, L27, L31, L36 observed), so it is not one bad layer: it is
whichever expert bank first loses the race.

Injection magnitude is ~1e-4 to 4e-3, amplified through the residual stream to ~1.5 at
the logits, which is enough to flip an argmax between near-tied ids.

## Minimal reproducer

Call the same bank **twice in a row** inside one decode step with the identical operand
tensors — same process, same step, nothing in between:

    trial 0: 0/40 banks differed
    trial 1: 1/40 banks differed   (L11, diff 6.10e-05)
    trial 2: 1/40 banks differed   (L06, diff 2.44e-04)
    trial 3: 2/40 banks differed   (L21, diff 4.88e-04)
    ...5 of 8 trials showed at least one

One op, one set of inputs, two different answers, milliseconds apart. Rare per call
(~1-2 in 40), but a 40-layer model generating hundreds of tokens hits it constantly.

## Ruled out by measurement

| Hypothesis | Verdict | Evidence |
|---|---|---|
| The router / top-k selection | **Ruled out** | scores and indices bitwise identical, 3/3 runs |
| Prefill | **Ruled out** | prefill is bitwise deterministic, 6/6 |
| Our buffer management in `_decode_gather` | **Ruled out** | holding every intermediate alive to end of step still flips, 3 reps |
| Command-queue count | **Ruled out** | 1 and 2 both flip |
| Reading uninitialised L1 | **Ruled out** | filling free L1 with 0/7/-7/1000 before each trial changed nothing |
| `ttnn.sparse_matmul` standalone | Deterministic in isolation | 8/8 identical, synthetic operands, empty device — so context matters, not the op alone |
| A host/device timing race | **Ruled out** | a full `ttnn.synchronize_device` after each of the 40 layers does NOT suppress it, 0/3 runs |
| The MoE's high-fidelity matmuls (router / shared expert) | **Ruled out** | `ORNITH_MOE_FIDELITY=lofi` takes them off HiFi entirely and b2b still flips (1 of 3 reps dirty, L24 diff 6.87e-05) -- the same rate as the hifi4 baseline (1 of 3, L36 diff 1.95e-03) |
| Blackhole's dynamic matmul throttle corrupting HiFi matmuls | **Ruled out** | see below -- the guard is bit-identical on both light and sustained load |
| Uninitialised L1 at device init | **Ruled out** | `TT_METAL_CLEAR_L1=1`, which zeroes the whole per-core L1 including the throttle word, still flips (1 of 3 reps, L16 diff 3.05e-05) |

## The Blackhole matmul-throttle hypothesis, and why it is not the answer

A source audit proposed a specific and attractive mechanism, and it is worth recording in
full because the DEFECT IS REAL even though it is not this model's problem.

On Blackhole, `matmul_block` routes unconditionally through
`matmul_block_math_dynamic_throttle` (`tt_metal/hw/inc/api/compute/matmul.h:246-266`).
That function reads a word the ARC firmware writes and, when its low bit is set,
reprograms the matmul MOP for `MM_THROTTLE_MAX = 5` while passing `MATH_FIDELITY` through
**with no guard**. The LLK's own test refuses to exercise that combination
(`tt_metal/tt-llk/tests/python_tests/test_matmul_custom.py:216-228`):

    "throttle levels 4/5 do not increment the fidelity phase per LLK;
     only correct for LoFi (single-phase)"
    "for a high-fidelity (multi-phase) matmul this collapses the extra phases
     and yields ~half the result"

ttnn's main matmul kernel (`bmm_large_block_zm_fused_bias_activation.cpp`) uses
`matmul_block`, so every high-fidelity matmul on this architecture is exposed. Since the
throttle word is power/thermal state, that would be intermittent and immune to
synchronisation -- exactly this document's signature.

**It is not what is happening here.** Guarding the branch so only single-phase fidelity
takes it (a one-line `if constexpr`; `matmul.h` is JIT-compiled, so no tt-metal rebuild is
needed) changes NOTHING, measured twice:

* a standalone 512x2048x2048 bf16 matmul at LoFi / HiFi2 / HiFi4 -- **bit-identical**
  before and after, same checksum, PCC and mean error at all three fidelities;
* the full 40-layer model's prefill logits at lofi and hifi2 -- **bit-identical** before
  and after, all six PCCs matching to 5 decimals. This is the sustained-load case, where
  an idle board's failure to draw enough power is not an excuse.

So the firmware never asserts the bit destructively on this part, and the hypothesis is
refuted for this model on both light and sustained load. Report the `matmul_block` defect
upstream on its own merits; do not spend time on it here.

One caveat on the guard's SHAPE, for whoever writes that upstream fix: the fidelity test
must sit inside `TRISC_MATH`. `MATH_FIDELITY` is part of the compute kernel's generated
prelude and is not declared for the unpack/pack RISCs, which also compile that header, so
a naive guard fails to build -- silently, if the caller only greps for its expected
output. Those cores are unaffected either way, because both arms of the branch contain
nothing but `MATH()` calls.

What the hunt DID turn up is a real second-order defect, and it is not throttle-related:
with `fp32_dest_acc_en=False`, HIGHER math fidelity is measurably WORSE than LoFi past
K~1024 (1.7x worse at K=2048, with a ~2.8% scale bias), because the extra fidelity passes
accumulate into a bf16 destination. With `fp32_dest_acc_en=True` fidelity behaves as
intended (HiFi4 ~16x better than LoFi, error flat in K). See the README's accuracy
section; it is a precision-policy bug, fully deterministic, and unrelated to this
document's non-determinism.

## What the flip rate actually is, and where it is NOT

Measured 2026-09-16, through the traced paged decode path (tt/generator.py), all on P150.
These narrow the search a lot, and one of them corrects a tempting misreading.

**Prefill is exact.** Six runs of the same prompt give ONE distinct post-prefill logit
fingerprint, and the step-0 token (the argmax of those logits) is identical across 16
runs. So nothing before the first decode step contributes. This is stronger than the
"prefill is bitwise deterministic, 6/6" already recorded above: it is the full logit
vector, not just the sampled token.

**The first decode step is where divergence begins.** Every diverging run of a 32-token
generation first differs at step 1, never later. Not at step 5, not at step 20.

**The per-step flip rate is LOW.** Repeating prefill + exactly one decode step 16 times in
one process gives the same step-1 token 13-15 times in a row, with the odd flip. Across
six FRESH processes the sixth generation was identical in all six.

**But a long generation still diverges, and this is the trap.** Repeating a 48-token
greedy completion 8 times in one warmed process gives 6 distinct completions. A low
per-step flip rate is not a low per-GENERATION flip rate: ~48 steps x 40 layers is enough
opportunity that some step flips nearly every time, and greedy then amplifies it forever.

    DO NOT conclude from the one-step data that "the model is deterministic once warmed".
    That was concluded here and it is WRONG. One decode step is mostly stable; a real
    generation is not. Always grade on a full-length completion.

**The bank-level b2b reproducer is no longer a valid proxy.** Matching decode's
expert-combine DEST precision to prefill's took b2b from 7-of-16 dirty reps to 0-of-16
(p ~ 4e-5) while repeated greedy completions stayed unreproducible -- 7 of 8 distinct
before, 6 of 8 after. So b2b and the user-visible symptom have DIFFERENT sources, and
only the b2b one is fixed. Grade future work on repeated greedy completions.

**Ruled out as the cause:** the lazy construction of the derived conv window.
`restore_split_conv_from_fused` returns early when `split_conv_state is None`, so
generation 1 skipped every in-place restore and let `gdn/decode.py` build the window
lazily -- a different code path from generation N, at addresses nothing had baked. That
IS a real hazard and is now fixed (the in-place reset materialises the window instead),
but it is NOT this bug: with it fixed, six fresh processes still split 5x/1x on the first
generation's step-1 token, and in-process greedy went only 7-of-8 to 6-of-8 distinct.

So the remaining source is a rare, genuine per-step non-determinism in the decode graph,
below the expert bank (which b2b now exonerates) and after prefill (which is exact).

## A warning about instrumented measurements

This effect is suppressed or distorted by observation, so several plausible-looking
results are artifacts. Two that were believed and then disproved by repetition:

* "Reading every sub-block back suppresses it (4/4 clean)" — **did not replicate.** With
  the maximal readback set (attention + feed_forward, all 40 layers) it still flips 0/3.
  The original clean run was intermittency.
* "Double-calling every op shows which one is non-deterministic" — **confounded.** Every
  op double-called shows 0 mismatches in 3183 calls, because double-calling everything is
  itself a perturbation. Only the *bank-level* double call reproduces.

**It is intermittent: a clean run is not evidence of a fix.** One six-trial run came out
6/6 identical and a rerun of the identical script flipped. A later sitting was 43/43
non-deterministic with zero clean runs of any configuration. Always repeat.

## Reproducers

`tools/repro_nondeterminism.py` ships both experiments above:

    python tools/repro_nondeterminism.py --mode trials   # identical inputs, different output
    python tools/repro_nondeterminism.py --mode b2b      # same bank called twice in a row

Repeat each several times -- the effect is intermittent and a clean run proves nothing.

## What is NOT the fix

A warm-up pass, an extra synchronisation, or an extra MoE call must not be shipped as a
fix. They change timing and allocation enough to hide the symptom in some runs while
leaving the arithmetic just as unreproducible — and the suppression does not even
replicate. This needs a real repair below the model.

## Note on the AUTODEBUG.md source audit

A source-level audit (untracked `AUTODEBUG.md` at the repo root) independently cleared
the GDN state handling, the MoE decode buffer path, and the trace/program-cache
machinery, and reviewed commit c5bd58989c6 as complete and correct. That much is
consistent with the measurements here. Its *causal story*, however, was built on the
premise that "every known suppressor adds serialisation", which the 43-run replication
above disproves — so its ranked hypotheses should be re-derived from the evidence in
this document rather than taken as-is.
