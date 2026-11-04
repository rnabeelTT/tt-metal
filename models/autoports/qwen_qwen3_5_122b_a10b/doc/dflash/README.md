# DFlash 2 speculative decoding for Ornith-1.5-122B (branch `ornith122_dflash`)

Goal: higher single-user decode throughput on p300x2 with unchanged output quality, using the public DFlash 2 drafter
`incoai/Ornith-1.5-122B-DFlash2` (6 layers, hidden 3072, block 16, 8 target context layers [1, 7, 14, 20, 26, 32, 39, 45],
non-causal sliding-window attention 2048, two-tap dynamic grouped convs, top-16 candidate selector; reference math:
github.com/z-lab/dflash `dflash/model.py` `DFlash2DraftModel`).

## Pieces

| file | what |
|---|---|
| `tt/dflash_ref.py` | torch reference of the drafter (host): context K/V, block hidden, selector path |
| `tt/dflash_device.py` | the drafter on the device (replicated bfp8 weights, K/V rings of 2112 slots, fixed 32-row programs) |
| `tt/dflash_traced.py` | one-user traced round: MTP verify pass (rows mode, + context-layer capture) / commit, drafter ctx / draft |
| `tt/dflash_serving.py` | serving: the adaptive MTP plumbing (lone request -> DFlash phase), prefill context in the bucket traces, acceptance guard |
| `tools/dflash_acceptance.py` | eager acceptance on the device target with the host drafter (+ greedy agreement) |
| `tools/dflash_traced_run.py` | the traced device loop: tokens per round, ms per token, per-program walls |

Env: `QWEN35_DFLASH=1` (needs `QWEN35_MTP=0`, off with chunked prefill), `QWEN35_DFLASH_BLOCK` (16),
`QWEN35_DFLASH_MIN_TAU` (2.5 tokens / round) and `QWEN35_DFLASH_GUARD_ROUNDS` (8): the acceptance guard.

## Measurements

### Acceptance (eager, greedy, thinking on, 96 new tokens; `runs/dflash/acc1.log`)

Device target (default env), host drafter (`tools/dflash_acceptance.py`), block 16:

| prompt | tokens / round | equal to plain greedy |
|---|---|---|
| code (LRU cache) | 6.25 | first 42 (then a near-tie rewording) |
| math (quadratic) | 8.08 | first 34 |
| prose (product description) | 4.13 | first 67 |
| mean | 6.15 | |

The divergences are bf16 near-ties between the 16-row verify pass and the one-row decode (different kernels: GDN prefill
form, multi-row MoE); both continuations read correctly (e.g. "To achieve O(1) for both..." vs "The standard approach is...").
The drafter is therefore correct (a broken drafter accepts ~0 drafts) and its published acceptance (MT-Bench 5.4, code
6.4-8.6 on SGLang) holds on this target.
