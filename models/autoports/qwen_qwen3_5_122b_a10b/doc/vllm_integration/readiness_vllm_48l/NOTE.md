First passing 48-layer run (`--max-num-seqs 32 --max-model-len 262144`, all four stages, full sampling profile;
`logs/09/vllm-48l.log`). `server.log` of THIS run was overwritten by the held server of the direct checks before the snapshot
(the misattributed copy was deleted after review round 2); its warm-up figures are in work_log.md (00:50 grep). The final evidence run
`readiness_vllm_48l-final/` carries a matching server.log.
