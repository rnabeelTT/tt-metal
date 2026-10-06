# Qwen3.5-122B-A10B on 4 Blackhole dies (p300x2 / p150x4)

TTNN port of `Qwen/Qwen3.5-122B-A10B` (256-expert MoE, 36 Gated DeltaNet + 12 gated attention layers, vision tower)
served through vLLM and the Tenstorrent plugin on a (1,4) mesh: text and image chat, tool calls, 262144-token context,
up to 32 concurrent sequences, speculative decoding from the checkpoint's MTP head. The engineering record is
`doc/HANDOFF.md`; this file is the operator's view.

## Serve it

```bash
tt-model serve raahemnabeel/qwen3-5-122b-a10b-blackholex4-v2 --profile p300x2   # private Hub bundle, one image
tt-model curl "Say hello in one sentence."
```

From this checkout instead of the bundle: `doc/HANDOFF.md` §2 (the readiness runner command and the env knobs).

Thinking is ON by default (reasoning in the message's `reasoning` field, answer in `content`). Turn it off per request
with `"chat_template_kwargs": {"enable_thinking": false}` for short, direct answers; give thinking requests thousands of
tokens of `max_tokens`. Images go in as `image_url` content parts (up to 4 per request, resized to at most 4 MP).

## Fixed defect (v2 repo): dies drawing different tokens with on-device sampling

`raahemnabeel/qwen3-5-122b-a10b-blackholex4-v2` is the first revision with this fix; the original
`raahemnabeel/qwen3-5-122b-a10b-blackholex4` repo is kept as it was.

The device sampler runs `ttnn.sampling` on every die of the mesh; each die draws from its own hardware
PRNG, writes its token into *its* persistent decode token input, and the host reads die 0's copy. The
unseeded seed manager leaves those PRNGs free-running after the first step, and over a long context the
dies drift apart, so the client receives die 0's word while the model continues from the other dies'
word. Found on the sibling Qwen3-Coder-Next port (`models/autoports/qwen_qwen3_coder_next/doc/text_quality/README.md`),
which shows it from ~8K tokens of context; this model needs a longer context before the dies drift.
Measured here with the real model on p300x2 (default sampling `temperature 0.6, top_p 0.95, top_k 20`,
token buffer of every die read after each decode step):

| context | sampler | steps where a die disagrees with die 0 | ms / step |
|---|---|---|---|
| 20K | stock | 0 / 400 | 31.34 |
| 20K | fixed | 0 / 400 | 31.36 |
| 95K | stock | **162 / 600** | 32.91 |
| 95K | fixed | **0 / 600** | 32.86 |

Fix (`tt/model.py`, `_install_lockstep_sampling`, same code as the Coder-Next port): identical fresh
seeds pushed before every draw, and after the draw die 0's token row is all-gathered and copied into
every die's token buffer, so the token fed back is the token the client receives whatever the dies
drew. `QWEN35_SAMPLER_LOCKSTEP=0` restores the stock sampler for A/B measurement only.

## Profiles: which one for which use case

One image serves every profile; a profile is a launch configuration (environment + server arguments). Each `p300x2-*`
profile exists as `p150x4-*` for four p150 boards (the identical (1,4) mesh; untested on this host).

| profile | use case | what it changes | what to expect |
|---|---|---|---|
| `p300x2` (default) | Interactive assistants and agents: one or a few users who want the fastest answer; images and tool calls | nothing: MTP speculative decoding at one user, vision tower resident, thinking on | 1 user: first token 0.32 s, ~50 tok/s; 8 users: 1.8 s, 15.7 tok/s each; 32 users: 6.3 s, 9.1 tok/s each (201 aggregate) |
| `p300x2-burst` | Many short prompts arriving together: chat front-ends with dozens of users, batch classification / extraction jobs | `QWEN35_BATCHED_PREFILL=1`: up to four short prompts (<= 512 tokens) prefilled per pass | 32-request burst first token 3.8 s instead of 6.3 s, 8 users 1.27 s instead of 1.83 s, decode unchanged. A request's first token can depend on what arrived with it (seeded reproducibility and greedy consistency across batch compositions not guaranteed; 12 of 73 conformance tests fail in this mode) |
| `p300x2-longctx` | Document and codebase work: very long prompts (tens of thousands of tokens) mixed with short questions | `QWEN35_CHUNKED_PREFILL=1`, `QWEN35_MTP=0`, `--enable-chunked-prefill --max-num-batched-tokens 2048` | prefill runs in 2048-token chunks shared between prompts, so a short question is not stuck behind a 100K-token upload; a lone user decodes at ~30 tok/s (no MTP); running users still pause while prefill chunks run (the plugin never mixes prefill and decode in one step) |
| `p300x2-text` | Text-only deployments with long conversations at high occupancy | `QWEN35_VISION=0`, image limit 0 | vision tower not loaded: 28472 KV blocks of 64 tokens instead of 22725 (1.8 M pooled tokens for 32 users); image requests rejected at the front end |

Numbers: `doc/perf_pass/deployed_sweep.md` (full sweep of the deployed bundle), `doc/HANDOFF.md` §1 (served table),
`doc/perf_pass/work_log.md` (every measurement with its log). Settings that were measured and rejected (MTP draft
length 2 / 4, the multi-user MTP step, thinking off as the default) are in `doc/HANDOFF.md` §6.

## Knobs (environment, read by the runtime)

| knob | default | meaning |
|---|---|---|
| `QWEN35_MTP` | 3 | MTP drafts per speculative pass; 0 disables speculative decoding |
| `QWEN35_MTP_ADAPTIVE` | 1 | MTP step only when exactly one request is live (0 = the batched multi-user step, measured slower) |
| `QWEN35_BATCHED_PREFILL` | 0 | batched short-prompt prefill (the `-burst` profiles) |
| `QWEN35_CHUNKED_PREFILL` | 0 | scheduler-chunked prefill with carried GDN state (the `-longctx` profiles; needs the vLLM flags above) |
| `QWEN35_VISION` / `QWEN35_VISION_MAX_IMAGES` / `QWEN35_VISION_MAX_PIXELS` | 1 / 4 / 4194304 | image support, images per request, resolution cap (every padded patch bucket up to the cap is warmed at start-up) |
| `QWEN35_DECODE_SORTED_MIN_ROWS` | 16 | decode MoE: expert-sorted path from this many rows (pairs path below) |
| `QWEN35_TRACE_REGION_MIB` | 768 | must match the TT config's `trace_region_size` |
| `QWEN35_PROGRAM_GUARD` | 1 in the bundle | assert that no program compiles after the traces are parked |

## Layout

`tt/` runtime (`model.py`, `generator.py`, `generator_vllm.py`, `prefill_batched.py`, `mtp_*.py`, `moe*.py`, `vision.py`),
`tests/` device tests, `doc/` per-stage evidence and the handoff, `tt-model.yaml` the bundle manifest (profiles, card).
