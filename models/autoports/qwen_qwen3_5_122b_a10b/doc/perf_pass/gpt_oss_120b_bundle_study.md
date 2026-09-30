# Study: `tt-hous/gpt-oss-120b-p150x4` bundle (TP vs EP for MoE)

Read-only study of the served decoder path (`code/models/autoports/openai_gpt_oss_120b/tt/{model,multichip_decoder,
fused_decoder}.py`, `models/demos/gpt_oss/tt/experts/operations.py`) inside the `tt-hous/gpt-oss-120b-p150x4` HF repo, on
4x p150 (QuietBox2). gpt-oss-120b: 36 layers, hidden 2880, 128 experts top-4 (~5B active params), alternating
sliding-window/full attention, GQA.

## MoE parallelism: Tensor Parallel, EP pinned to 1 (not Expert Parallel)

The repo ships a *generic* shared MoE module (`models/common/modules/moe/tt_moe_decode.py`) that does support EP
(`ttnn.experimental.all_to_all_dispatch_metadata`, `expert_mapping: sequential`) — but gpt-oss-120b does not use that
path in its served config. Its own decoder (`tt/model.py:602-636`) sets
`mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp, ep=1, sp=1), prefill=ModeConfig(tp=tp, ep=1, sp=1))`
with `tp = mesh_device.shape[1]` (4). Experts are **TP-fractured**: every chip holds all 128 experts, each expert's
weight sharded on its own hidden/intermediate width across the 4 chips (`multichip_decoder.py` docstring: "`sparse_matmul`
experts with TP-fractured weights and ring reductions"; `_DOWN_SUBBLOCK_WIDTH_BY_TP = {2: 1, 4: 3}` keys sub-block width
by TP degree). Decode dispatch is a local indexed `ttnn.sparse_matmul(..., indices=expert_indices_rm,
is_input_b_sparse=True)` per chip (`fused_decoder.py:789-846`) — no cross-chip token routing; combine is
`ttnn.all_reduce`/`all_gather` (`experts/operations.py:56-106`), same CCL shape as a dense TP matmul.

**This is structurally identical to our own Qwen3.5-122B-A10B choice**: TP4 on the intermediate dim, all 256 experts
resident on every chip, sparse_matmul gather-mode dispatch, no all-to-all. Two independent autoports converged on the
same answer for the same reason (see judgment below).

## Perf (README, QuietBox2/4xP150, batch=1 decode)
TPOT 15.3-20.1 ms (49.8-65.4 tok/s/u), essentially flat across ISL 128 -> 130,816; TTFT scales 247 ms -> ~30 s with
prompt length; burst 230-560 tok/s at 32 concurrent users depending on ISL. None of this is attributable to TP vs EP
per se — it mostly reflects gpt-oss's much smaller active-parameter footprint (~5B active vs our ~10B active) and
top-4-of-128 vs our top-k-of-256. The flat TPOT-vs-ISL is consistent with TP: decode weight-read cost is bounded by
top-k regardless of routing pattern, since every chip always holds a slice of whichever expert gets picked.

## Judgment: TP was the right call here, on the same logic as ours
128 experts / top-4 on 4 chips is a bad EP target: at batch 1-2 a decode step touches at most ~4-8 distinct experts
total, so an EP4 split (32 experts/chip) leaves most chips idle most steps and makes load a function of *which*
experts got routed, plus pays all-to-all latency every step for no benefit at this batch size. TP sidesteps this:
every chip participates in every step uniformly, and CCL cost (allreduce/allgather) is fixed and data-independent.
Contrast `qwen3.8-flash-next_p150x4`'s EP4 choice (512 experts, top-10, `doc/perf_pass/qwen38_bundle_study.md`): more
experts get touched per token there, and their design targets throughput/server-scale batches where dispatch overhead
amortizes and disjoint-expert-per-chip avoids diluting every chip's weight stream across a 512-wide pool. Rule of
thumb these three autoports collectively confirm: **EP wins when (experts touched per step) is large relative to
chip count; TP wins when it's small** — 256-of-4 and 128-of-4 land on TP, 512-of-4 (with top-10, i.e. more experts
touched per token) lands on EP.

## Actionable diffs vs. our branch
- gpt-oss-120b splits compute fidelity by phase (HiFi2 prefill / LoFi decode attention projections,
  `doc/datatype_sweep/selected_precision_config.json`) — worth a quick check whether our `tt/precision.py` policy
  already does this per-phase split or only per-tensor; if only per-tensor, phase-splitting attention fidelity is a
  small lever we haven't tried.
- Its MoE dispatch is a plain indexed `sparse_matmul` + `ttnn.embedding`-gathered bias, with no counting-sort or
  gather-add-combine fusion visible — our qwen3.8-derived prefill dispatch levers (kernel-side counting sort,
  gather-add un-dispatch fusion, already shipped) are more advanced than what gpt-oss-120b's bundle does. Nothing to
  pick up there.
- It has a "FullLocal" calibrated-layer `moe_compute` allowlist gated on exact checkpoint hash for batch<=2
  (`fused_decoder.py:65-85`) — a checkpoint-specific speed hack, not portable and not worth chasing.
- Precision policy (bfp4 experts, bfp8 attention/KV) matches ours closely; no lever there.

## Method note
Read via `huggingface_hub` with `HF_HUB_OFFLINE=0` (env.sh sets it to 1 by default); repo is a ~1000-file snapshot of
their internal `models/` tree, not just this one autoport — pulled only the gpt-oss-specific files plus the shared
`modules/moe`, `modules/attention`, `tt_ccl.py` files, not the whole tree.
