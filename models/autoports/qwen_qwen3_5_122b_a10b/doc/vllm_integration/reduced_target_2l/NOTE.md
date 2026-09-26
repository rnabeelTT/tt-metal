Reduced 2-layer serving target (QWEN35_LAYER_INDICES=0,3; layers 0 and 3 of 48) used only for the adapter bring-up loop:
`run_vllm_server` full sampling profile (73 passed / 4 skipped) + both benchmark profiles at max_num_seqs 32 / max_model_len 4096
(`logs/09/vllm-2lfull.log`). Its text is meaningless by construction (2 of 48 layers) and its qualitative outputs were removed
from the tree so the stage check scans only the 48-layer evidence; its benchmark numbers are NOT the model's performance.
