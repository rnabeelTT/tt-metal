# Stage 12 work log — prefill parallelization

- 2026-09-29 ~12:00 UTC: user request: serialized prefill makes long prompts infeasible; parallelize across the cores; perf
  tests + correctness only (no accuracy evals). Reference pulled: tt-hous/gpt-oss-120b-p150x4 (chunked prefill / batched
  prefill capabilities; its experts use `ttnn.sparse_matmul(..., indices=...)`).
- Tracy before (`scripts/12-tracy-prefill.sh`, `12-tracy-moe.sh`): whole 2048 layer capture truncated by the profiler
  buffer (43 ops) -> phased MoE test with `ttnn.ReadDeviceProfiler` flushes between phases (`test_perf_moe_prefill`).
  Result: MoE block ~53 ms device per layer at 2048 tokens; dispatch bookkeeping ~35 ms, sparse matmuls 12 ms on 32 cores.
- Implemented the grouped sparse matmul factory + indexed in1 writer kernel; `select_program_factory` for the sparse op;
  matmul counting sort. Build: `ninja ttnn` in build_Release (~20 s), then copy `build_Release/ttnn/_ttnn.so` over the stale
  `ttnn/ttnn/_ttnn.so` that Python imports (otherwise the old library runs: "std::get: wrong index for variant").
- 14:03 first device run: op returned the rank-6 indexed shape ([1,G,1,G,M,N], the decode cross-product semantics) ->
  compact [1,G,M,N] spec + validation for the grouped mode; G > E allowed in grouped mode.
- 14:11 gu64 passes (PCC vs torch 0.99996). 14:14 gu512 hangs. Bisect (`12-grouped-case.py`, one process per case +
  reset): 130 groups hang (some cores own 2 groups), 64 pass -> `split_work_to_cores` column-major core groups vs row-major
  runtime-arg assignment. Fixed (row_wise split + `core_group_1.contains(core)`); 14:41 130 groups pass.
- 14:44 unit suite: gu64/gu512/dn512/gu37_bfp8 PCC pass (bar 0.9995 abs + grouped >= mcast vs torch); perf gate/up 12.02 ->
  3.61 ms, down 37.86 -> 2.52 ms; mesh case pass. `test_dispatch_modes_equivalent[t500]` failed: Bk selection required a
  divisor of Rp (4000 rows) -> pad Rp to nb*Bk, slice back; T = 513/1000/1023 added to the unit test and 513/1000 to the
  mesh prefill-vs-HF test.
- 14:45-14:52 chain1 (`scripts/12-chain1.sh`, `logs/12/chain1.log`): mesh layer suite with the grouped path 25/25
  (`test_mc_prefill_decode_against_hf` 5..2048 both kinds, `test_mc_chunked_prefill_matches_hf`, `test_mc_shipped_policy_pcc`,
  `test_mc_prefill_after_traced_decode`, `test_mc_real_weight_against_hf`, `test_mc_dispatch_counting_matches_sort`).
  GDN prefill T=2048 PCC 0.99849 (stage-5 value 0.9985), attention 0.99998.
- 14:53-14:55 chain1b: `test_dispatch_modes_equivalent` t256/t500/t513/t1000/t1023/t2048 pass (counting_mm == counting_h
  bit-exact); mesh `test_mc_dispatch_counting_matches_sort` 512..2048 + prefill-vs-HF seq513 / seq1000 both kinds: 9/9.
- 14:55 chain2: Tracy after (MoE phases, whole 2048 layer both kinds). chain3 queued: 48-layer prefill bucket bench,
  prefill-check on p2048 / p8192 / aime24 refs, served after, served before (grouped off).
- 14:55-15:00 chain2 Tracy after: MoE block device 52.98 -> 23.24 ms per layer (sparse matmuls 12.16 -> 4.68 ms on 110
  cores; permute+cumsum 25.0 -> 0.27 ms); phase host walls dispatch 19.2 -> 6.7, gate/up 8.7 -> 2.9, down 3.8 -> 2.75 ms;
  whole-layer untraced wall GDN 109.5 -> 89.6 ms, attention 106.8 -> 86.2 ms.
- 15:01-15:12 chain3 prefill bucket bench, 48 layers, traced (3 reps each): 128 tok 343.8 ms (stage 7: 344.0), 161 tok
  344.9 ms (555.8), 2048 tok 988.5 ms (1934.8; 2072 tok/s), 4097 tok 2208.7 ms, 8192 tok 3977.5 ms (2060 tok/s).
  Program cache 910 entries after warm-up, unchanged by the bucket captures.
- 15:12-15:22 prefill-check book_p2048 (2048-token prompt through the grouped path, 48 layers): top1 0.99 / top5 1.00 /
  top100 1.00 (bars: top5 >= 0.98, top100 = 1.0). p8192 and aime24 running.
- 15:22-15:42 prefill-check book_p8192 (4 sorted segments): top1 0.95 / top5 1.00 / top100 1.00; aime24 chat (masked
  path, < 512 tokens): 0.97 / 1.00 / 1.00 = stage-7 values. chain4 queued: p8192 with QWEN35_SORTED_GROUPED=0 (control).
- 15:41-15:57 served after (`12-serve-chain.sh` TAG=after, `logs/12/serve-chain-after.log`): c1 128/128 TTFT p50 423 ms,
  TPOT 37.7 ms; c32 TTFT p50 12.12 s, TPOT 132.1 ms; long prompts TTFT 8192: 4.01 s, 32768: 16.4 s, 131072: 72.3 s
  (stage 11: 7.8 / 31.4 / 132.6 s). 131072 greedy text identical to stage 10. served before (grouped off) running.
- 15:57-16:13 served before A/B (grouped off, counting_mm on): c1 TTFT p50 372 ms; c32 12.05 s; long TTFT 8192: 5.33 s,
  32768: 21.6 s, 131072: 93.5 s -> decomposition: counting_mm 132.6 -> 93.5 s, grouped matmuls 93.5 -> 72.3 s. Texts identical.
- 16:14-16:26 chain4 control: prefill-check p8192 with the grouped path off: top1 0.95 / top5 1.00 / top100 1.00 = the
  grouped result. Stage closed at the layer, full-model and served levels; commit, container rebuild, private push, Hub verify next.
- 16:23-16:31 commit 6da58d94a83 (amended to drop overwritten readiness_vllm working files); `tt-model package` rebuilt the
  image in 5 min on the warm cache: image 5eba8ab305a3, manifest tt_metal sha 6da58d94a83; 16:32 pushed private (Hub
  revision 1e17833e). 16:33-16:47 first Hub verification served the OLD image 36b6aa8bb6e4: `tt-model serve --refresh`
  compares the installed revision with the Hub tip through a 3 s network call, and the script inherited HF_HUB_OFFLINE=1
  (its export line carried a `HF_HUB3_OFFLINE` typo from stage 11), so the refresh returned nothing and the installed
  stage-11 revision was served (8K prompt 8.3 s, 128K 132.7 s = stage-11 timings; functional checks passed on that image).
  Fixed the export; rerun queued behind the stage-13 probes.
- 17:18-17:33 Hub verification rerun (`logs/12/verify-hub-run2.log`, HF_HUB_OFFLINE=0): `tt-model serve --refresh` moved
  the install cb1f080e -> 1e17833e and started image 5eba8ab305a3; healthy in 10.7 min; /v1/models 262144; chat, tool call
  (get_weather Paris), tt-model curl default / thinking on (603 reasoning chars) / thinking off, non-aligned 2381-token
  prompt, 32-request burst 126.3 tok/s, 8192-token prompt and 131072-token prompt 72.9 s wall for 16 new tokens (old
  image: 132.7 s), completion text identical. Stage 12 closed end to end.
