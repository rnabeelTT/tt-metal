# Qwen4 bring-up kit

Everything gathered before Qwen4's release (2026-11-12) so a port can start the day the weights appear. If you are an
agent picking this up: read the **Playbook** first, run `triage.py` on the new repo, then follow the branch it
recommends.

What is known: Qwen4 is not out. **Qwen3.8-Flash-Next** (`Qwen/Qwen3.8-Flash-Next`, `model_type: qwen4_exp`,
125B-A6B + 51B n-gram table + vision) is Qwen's own "experimental preview of the architecture that will underpin
Qwen4". Transformers >= 5.16.1 has its reference code (`transformers/models/qwen4_exp`). Samuel Jett's full
Blackhole port of it is on tt-metal main: `models/demos/blackhole/qwen38_flash_next` (#58690, #64184). The other
Qwen3.8 releases (27B dense, 2.4T-A95B) are the old `qwen3_5` architecture.

## Contents

| file | purpose |
|---|---|
| `triage.py` | **Run first.** Compares a new HF repo (config, tensor names, shapes from the safetensors headers; no weight download) against Flash-Next, sizes it per profile, and recommends a starting point. Standard library only. |
| `sizing.py` | Parameters, device bytes per part, fit per profile (p150 / p300 / p300x2 / galaxy), and where the PLE table lives. Calibrated within 0.01 % on both Flash-Next (360.0 GB) and Qwen3.8-27B (55.6 GB). |
| `mini_reference.py` | Random-weight `qwen4_exp` checkpoint of any size, from the HF reference, with the released tensor names, plus golden prefill/decode logits. Use it for device unit tests before real weights exist, and to test sizes Qwen has not released. |
| `ple_nvme_probe.py` | Times the per-token PLE row reads from NVMe with no page cache. |
| `flash_next_config.json`, `flash_next_tensors.json` | Baseline for `triage.py` (refresh with `triage.py --snapshot Qwen/Qwen3.8-Flash-Next`). |
| `qwen38_27b_config.json` | Dense calibration point for `sizing.py`. |
| `tests/test_prep.py` | Host-only tests of the above: `python -m pytest models/demos/blackhole/qwen4_prep/tests -q --noconftest`. The mini-reference test needs torch + transformers. |

CPU environment for `mini_reference.py` and the tests. The repo's `python_env` works too if its transformers is
>= 5.16.1:

```bash
uv venv q4env --python 3.12
VIRTUAL_ENV=$PWD/q4env uv pip install --index-url https://download.pytorch.org/whl/cpu torch
VIRTUAL_ENV=$PWD/q4env uv pip install "transformers>=5.16.1" safetensors pytest
```

## Playbook (when a Qwen4 checkpoint appears)

### 1. Triage (minutes)

```bash
python3 models/demos/blackhole/qwen4_prep/triage.py Qwen/<new-repo> --json /tmp/q4_triage.json
```

Read three things in the output:
- `model_type`. If it is `qwen3_5*`, this is the old architecture: serve it with our `qwen36` stack, as for Qwen3.5 /
  Qwen3.6 / Qwen3.8-27B, and stop here.
- The NEW / GONE / RESHAPED lines. RESHAPED-only means the same blocks at new sizes. NEW tensors mean new math.
- The sizing table, which shows the profiles that fit and where the PLE table goes.

### 2. Pin down the math (an hour)

Install the Transformers version that knows the new `model_type`. Diff its modeling file against
`qwen4_exp/modeling_qwen4_exp.py` (5.16.1). Then rebuild a random checkpoint at the new sizes:
`mini_reference.py --preset tiny --set key=value ...` (or add a preset; for a new `model_type`, change the import
in `mini_reference.py`). The reference math as of `qwen4_exp`, with the details that are easy to get wrong:

- **Residual stream**: the embedding is repeated `hc_count` times (width `W = hc_count * hidden`). Every RMSNorm in
  this model multiplies by `(1 + w)`, because the weights are zero-initialized. `hc_norm`, `norm_key`, `norm_query`
  and `norm_conv` are *grouped* RMSNorms, one group per `hidden` slice. There is no final RMSNorm:
  `hyper_connection_mixer` (a Gated Residual without inject) makes the final hidden state.
- **Gated Residual** (before each mixer and each MoE): `n = hc_norm(x)`;
  `g = sigmoid(up(silu(down(n) / hc)))`; block input `= mean over branches of (g * n)`; after the block,
  `x = x + out * 2 * sigmoid(inject(n) / hc)` per branch. The residual uses the *un-normed* `x`.
- **PLE** (n-gram embedding, `ple_layer_ids` are 1-based: `[2]` means checkpoint `layers.1`, applied *before* that
  layer's GR read):
  - The ids are SplitMix64-derived odd multipliers. Bigram ids are `(x*m0) ^ (prev*m1)` and trigram ids xor in
    `prev2*m2`, both taken mod a per-head prime and added to a per-head offset. The primes are the next primes after
    `ngram_vocab_size_base - 1`, indexed globally across PLE layers. The history resets at EOS, and PLE ids are
    masked to EOS on padding.
  - Rows are `embed_dim / ((n-1) * heads_per_ngram)` wide, and the table is split into `split_ngram_parts` shard
    tensors.
  - `gate = sum(norm_key(key_proj(e)) * norm_query(x)) / sqrt(hidden)` per branch, followed by a signed sqrt
    (`sign * sqrt(|gate|)`); `v = sigmoid(gate) * value_proj(e)`.
  - Output: `v + silu(dilated_conv(norm_conv(v)))`, with a depthwise kernel of `ple_conv_kernel_size` and dilation
    `ngram_size`. The conv state is `(k-1) * ngram_size` tokens.
  - Upstream's per-token form: `qwen38_flash_next/ttnn/ple.py: ngram_token_ids_decode_step`.
- **QSA**: `index_qk_proj` gives `indexer_n_heads` query heads plus one shared key head (`indexer_head_dim`). The
  query gets `q_layernorm` + RoPE. Raw keys are cached per token. Complete blocks of `compress_ratio` tokens are
  mean-pooled, then `k_layernorm`, then RoPE at the block's first position.
  `score = sum_heads(relu(q . k_block)) / sqrt(d)`; the top `budget / ratio` blocks plus the incomplete tail block
  are selected. Main attention is gated attention over the selected tokens only. The gate is packed in `q_proj`,
  `[q | gate]` per head: `out * sigmoid(gate)`. Partial RoPE (`partial_rotary_factor`) with interleaved mRoPE.
  Up to `budget` tokens of context this is exactly dense causal attention.
- **MoE**: softmax router, top-k, renormalized (`norm_topk_prob`); shared expert scaled by
  `sigmoid(shared_expert_gate(x))`. Experts are stored fused: `experts.gate_up_proj [E, 2*I, H]`,
  `experts.down_proj [E, H, I]`. The HF config *requires* `num_experts > 0`, so a dense Qwen4 would be a new
  `model_type`.
- **GDN**: identical tensors and math to Qwen3.5 (`in_proj_qkv/z/a/b`, conv k=4, `A_log`, `dt_bias`, gated norm).
- **MTP**: `fc_embedding` / `fc_hidden` with pre-norms, plus one QSA + MoE + GR layer and its own mixer.

### 3. Choose the base

| situation | base | why |
|---|---|---|
| Same blocks, 4 Blackhole chips (p150x4 / p300x2), sizes TP4-divisible | **4a**: upstream `qwen38_flash_next` | QSA / GR / PLE / MTP / BF4 experts, acceptance gates and serving are done; only the pins need relaxing |
| p150 or p300 (1-2 chips), or widths that break the upstream literals | **4b**: our `qwen36` stack plus ported blocks | `qwen36` is already parametric (dense at TP 4/8, MoE at TP 1-4, our fused decode kernels, vLLM + prefix caching) |
| New `model_type` / new tensors | step 2 first, then 4a or 4b | |

### 4a. Upstream port: what to relax

The port refuses any other checkpoint by design. These are the pin sites, relative to
`models/demos/blackhole/qwen38_flash_next/`:

- `config.py:14` `CONFIG_SHA256`, and every `_expect(...)` in `Qwen38Config.from_checkpoint` (from `config.py:107`).
  `config.py:15` `LAYER_PATTERN` assumes 48 layers.
- `checkpoint.py:25-29`: `INDEX_SHA256`, `EXPECTED_SHARDS = 131`, `EXPECTED_TENSORS = 1658`.
- `config.py:226` `Qwen38Placement`: exactly 4 devices on a 1x4 mesh. `qsa_kv_device_groups` hard-codes 2 KV heads
  across 4 devices.
- `TP_SIZE = 4` in `ttnn/contracts.py:22` (`MESH_SHAPE`), `ttnn/{builder,gr,final_mixer,layer,embedding,gdn,mtp,ple}.py`
  and `tt/{gr,moe,model,ple,qsa,gdn}.py`. `ROUTED_EXPERTS = 512` in `ttnn/moe.py:56` and `ttnn/builder.py:105`.
  `tools/live_decode_construction.py:44`.
- About 240 literal `2560` / `640` across `ttnn/` and `ttnn/fused/*/__init__.py`. **640 is both hidden / TP and
  the expert width**, so check each site by meaning, not by value. Find them with
  `grep -rn '\b2560\b\|\b640\b\|\b10240\b' ttnn`.
- `ttnn/bf4.py:626`: the BF4 expert-cache layout is keyed by `(hidden, intermediate, experts, experts_per_device)`.
- External dependencies: open `moe_compute` PRs #59945 (expert rows; the upstream perf numbers need it) and #60469
  (packed routing for single-call prefill slabs).
- Hardware: p300x2 has a mesh descriptor (`tools/qb2_p300_1x4_line_mesh_graph_descriptor.textproto`) but is untested
  upstream.

### 4b. `qwen36` stack: what to add

On this box the stack lives at `~/tt-metal-bringups` (branch `ornith_fusion`), `models/demos/blackhole/qwen36/tt/`.
Its engineering record is `models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/HANDOFF.md`. Add three modules,
using the upstream port as the reference implementation and its fused kernels as starting points:

| block | upstream reference (`qwen38_flash_next/`) | notes |
|---|---|---|
| Gated Residual | `tt/gr.py` (torch), `ttnn/gr.py`, `ttnn/fused/gr_read`, `gr_write`, `gr_recip_last`, `gr_fold` | Replaces `addnorm` / `decode_boundary` / `fabric_boundary`, our most-tuned path. The residual is 4x wider. GR weights are about 13M params/layer at Flash-Next sizes (about 14 MB/token/layer in BF8, half of the 10 active BF4 experts' bytes), so try BF4 GR weights. |
| QSA | `tt/qsa.py`, `ttnn/qsa.py`, `ttnn/fused/qsa_block`, `qsa_rows` | Uses `ttnn.experimental.indexer_score_dsa` and `ttnn.experimental.topk_large_indices` (both on main). Bring up dense attention first: it is exact up to `budget` tokens of context. |
| PLE | `tt/ple.py`, `ttnn/ple.py` (`Qwen38ResidentPLELookup`), `ttnn/fused/ple` | Host lookup + device projection/conv. If `sizing.py` says the table fits on device, a device gather (`ttnn.embedding` over a DRAM-sharded table, hashing on host) avoids the per-token upload. |
| MoE | our `moe_decode` | 512 experts / top-10 / shared expert is Qwen3-Coder-Next's routing. Expert width 640 = 20 tiles (we have run 512 and 1024). Loader: split the fused `gate_up_proj`. |
| GDN, attention projections, RoPE | our `gdn_decode`, `attn_decode`, `rope` | Unchanged layout. `GdnDecode` asserts at most 32 V heads per device. |

### 5. Gates (same bar as our published ports)

1. Each block against `mini_reference.py` golden data at the new sizes, then against the HF CPU reference on real
   weights (layer by layer; `qwen38_flash_next/tools/qwen38_reference_corpus.py` scores top-1 / KL columns).
2. Greedy acceptance against CPU records (upstream: `--acceptance --require-json-96`).
3. Accuracy: GSM8K and teacher-forced top-1 agreement. Upstream Flash-Next with BF4 experts gets 86.4 % GSM8K and
   95.2 % top-1 (KL 0.053).
4. Performance: report every change as before -> after per metric (TTFT, TPOT at 1 / 32 users, tok/s).

### 6. Serving

vLLM via our plugin (prefix caching: `prefix_state_cache` capability, see HANDOFF "Prefix caching"), packaged as a
tt-model bundle like our `*-blackhole-fast` repos. Speculation: use the model's MTP head for MoE variants
(upstream uses MTP 4). DFlash-style drafters pay on dense models (below) but broke even on our 122B MoE, where an
8-row verify costs about 3.5 decode steps.

## Measured facts and numbers to beat

**The PLE table's size depends on the n-gram base, not the model size.** Its parameter count is about
`ngram_vocab_size_base x ple_embed_dim` (Flash-Next: 20M x 2560 = 51.2B). `sizing.py --scenarios`, with experts BF4,
dense BF8, PLE bf16, 32k context, 1 user and 3 GB reserved per chip:

| model (guesses marked) | weights on device | p150 (29 GB usable) | p300 (58 GB) | p300x2 / p150x4 (116 GB) |
|---|---|---|---|---|
| Flash-Next (released) | 75.6 GB | no | no | fits; PLE on host |
| guess: 35B-A3B-class + qwen4 blocks, base 5M | 22.5 GB | fits; PLE on host | fits; PLE on device (20.5 GB) | fits; PLE on device |
| guess: 35B-A3B-class, base 20M | 22.5 GB | fits; PLE on host | fits; PLE on host | fits; PLE on device |
| guess: 27B dense + qwen4 blocks (not a valid `qwen4_exp`) | 33.8 GB | no | fits | fits |

**Host RAM is not needed for the PLE table.** Upstream's read pattern (WILLNEED + one pread per row) on a cold 50 GB
file, with the page cache evicted (`ple_nvme_probe.py --mode willneed`):

| rows per pass | p50 | p99 |
|---|---|---|
| 16 (one decode token) | 0.23 ms | 0.35 ms |
| 80 (MTP verify, k = 4) | 0.58 ms | 0.73 ms |

**Reference numbers:**
- Upstream Flash-Next, p150x4, 1 user, MTP 4: 54-58 tok/s decode (44 at 255k), prefill 1.3-1.6k tok/s.
- Our Qwen3.5-122B-A10B (no MTP), p300x2: 7.5 ms/step for 1 user, 45 ms/step at 32 users.
- Our Ornith-35B, p300x2: 4.6 ms/step for 1 user.
- mando2222's Qwen3.8-27B (dense) bundle, p300x2: 46.4 tok/s plain, 87.0 tok/s with DFlash2 (greedy).

## Borrowable from mando2222/qwen3.8-27b-dflash2-vision-p300x2-q4kv-v6-thin-cp4

A fork of our `qwen36` tree (all shared kernels byte-identical) serving the dense Qwen3.8-27B on p300x2 under vLLM.

- **DFlash2 speculative decoding** (`dflash2*.py`, `spec_decode.py`, drafter `incoai/Qwen3.8-27B-DFlash2`):
  46.4 -> 87.0 tok/s, 1 user, greedy. This is the template for a dense variant.
- **BF4 paged KV**: `QWEN_SDPA_KV_DTYPE=bf16|bf8|bf4` (`attention/tp.py`, `qwen36_vllm.py`). Small and generic;
  matters for models with many attention layers at long context.
- **Thinking budget** (`think_budget.py`): vLLM API middleware that closes runaway reasoning and continues into the
  answer. Model-agnostic.
- **GDN spec-verify kernels** (`qwen38_v6_ops.gdn_spec_step`): overlap with our `gdn_decode` rows mode. A/B before
  choosing.
- Not needed: their prefix caching (GDN state as vLLM `MambaSpec` blocks of 4096). Ours ships with 2048-token
  snapshots.
