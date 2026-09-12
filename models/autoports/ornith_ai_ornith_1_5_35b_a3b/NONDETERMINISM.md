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
