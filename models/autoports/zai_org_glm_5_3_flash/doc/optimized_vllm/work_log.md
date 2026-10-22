# Stage 10 (optimized vLLM) work log

Tickets run through `~/glm53-flash-bringup/scripts/run-locked.sh`, each from a pinned snapshot (`s10_snap*`: git
archive of HEAD + the stage-10 files, md5s in the snapshot). Scripts: `~/glm53-flash-bringup/scripts/s10_*.sh`. Logs:
`~/glm53-flash-bringup/logs/s10/`. Records: `~/glm53-flash-bringup/artifacts/s10/`.

| Ticket | What | Result |
|---|---|---|
| probe1 | DSA selection cost by width; top-k invariance; small-trace replay | width cost 0.084 -> 0.537 ms per layer (4,128 -> 70,176 pools); top-k tie order width-dependent |
| t2 | reduced, buckets on vs off at the same T | top-1 89/91, 13/20 (tie noise); tracker clean |
| t3, t3b, t9, t10 | exactness against the `T_b` build | bit-identical one request per process (fused + classic modes, 1 and 8 slots); t3 multi-request mismatch = harness prompt seeding |
| t4 | 16K: single trace vs segments | +0.1 to +0.45 ms -> `GLM53_DSA_BUCKET_MIN_T` = 65,536 |
| t5 | full model 280,576: DSA buckets | token-out 55.07 -> 50.10 ms at pos 300; teacher forcing identical |
| t6, t16 | vLLM adapter validator | PASS (t16 on HEAD + the P1/P2 set) |
| t7 | pool-key alias | bit-identical; 8.40 -> 4.20 MB per layer |
| t8 | 16K build with the 280K row block | 4,448 ms (row block not the TTFT cause) |
| t11 | three requests in one process | bit-identical 27/27 |
| t12 | Tracy, 16-slot vs 1-slot (reduced) | MoE pairs glue +1.3 ms per MoE layer, KDA +167 us per layer |
| t13 | MoE row buckets, reduced | pairs exact; 7.01 -> 5.42 ms at 1 active |
| t14 | prefill views, full model 243,712 | TTFT 8K 5,611 -> 4,376 ms; teacher forcing identical |
| t15, t15b | MoE row buckets, full model 16 slots | t15: 64 MiB trace region overflow -> fit_row_buckets; t15b: 106.3 -> 72.9 ms at 1 active, exact |
| t16 | vLLM adapter validator on HEAD + the P1/P2 set | PASS, PROGRAM_GUARD 929 = 929 |
| t17 | async stale-input test (4 slots, buckets active) on 0a2663d0b6b | ASYNC_RESULT PASS |
| (host) | slot-aware trace-region fit (G9 32-slot capture out of memory) | identical lists for every measured config; 32 slots fit; committed 044dbeaa3fd |
| t18 | wide batch invariance (32 slots): off / old / family / pairs | old not invariant at 32 active; family (sorted) invariant, 118.6 / 123.8 / 147.4 ms at 1 / 8 / 32 active, TF 93 |
| t19 | dealloc fixes (moe_opt pairs 1-row ids, dsa_shard q_rm view) + family rule | probe DEALLOC_CHECK clean in 8 / 32 replicated and 8 / 1 sharded; invariance at 32 and 8 slots bit-identical |
| t20 | DSA bucket batch invariance (reduced, 2 slots): auto / classic-only / off | auto not invariant (max logit delta 1.73); classic-only invariant, +0.7 to +1.3 ms below 65,536 |
| t21 | capacity re-probe, pool sized for ~700 MiB free after the bind (cf41be52370 + fallback) | copy 1.0: p300x2 16 x 262,144, wide 32 x 159,744, long 1 x 344,064 (was 239,616 / 137,216 / 325,632) |
| t22 | proof of classic-only buckets at S >= 2 on 3993d5eecef | reduced S = 2 / 16 bit-identical; served smokes and burst 8/8 pass; served 4K solo vs next to a 70K decode differs |
| t23 | served diagnostic on 3993d5eecef (p300x2, text + random prompts) | solo repeat always equal; all-off and DSA-buckets-off identical; default and async off differ: DSA buckets cause it |
| t24 | served check on the fallback (no auto DSA buckets at S >= 2) | text and random: solo repeat and concurrent equal |
| stage-9 s9_final / wide-r2 | served after numbers | p300x2 1 user TPOT 119.27 -> 69.91 ms; wide on the fixed code: burst 32/32 = solo, 1 user 120.9 ms, 32 users 160.1 ms |

Commits: 86403d19830 (DSA width buckets, pool-key alias), 0a2663d0b6b (MoE row buckets, prefill views, shared iotas,
per-profile trace region), 51997a179ba (docs), 044dbeaa3fd (slot-aware trace-region fit), 0bcf89f7fd6
(batch-invariant MoE buckets, dealloc fixes; t18, t19), 36f3dd30b61 (stage 9: wide final on the fixed code), 3993d5eecef
(review #1: classic-only DSA buckets at S >= 2, probe sizing), 344195b5ef5 (review #1 P2-1 / P2-2: no DSA decode
buckets at S >= 2, t21 capacities).

The combined stage-9 serving ticket (s9_final) and the wide re-run supplied the served after values. Stage-10
review #2: CLEAN PASS.
