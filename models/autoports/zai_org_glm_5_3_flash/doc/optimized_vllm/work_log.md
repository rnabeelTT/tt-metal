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

Commits: 86403d19830 (DSA width buckets, pool-key alias), 0a2663d0b6b (MoE row buckets, prefill views, shared iotas,
per-profile trace region).

Pending: the combined stage-9 serving ticket (served before/after per profile, CI burst, sampling tests, qualitative
outputs, G9 re-probe), then the stage review.
