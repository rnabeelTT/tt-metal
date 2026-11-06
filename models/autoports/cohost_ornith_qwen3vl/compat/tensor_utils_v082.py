

# --- appended by the co-host image build (compat/tensor_utils_v082.py) ---
# The Qwen3-VL vision code was written on tt-metal v0.82, whose tensor_utils has this helper; this v0.79 tree
# doesn't. Upstream it changes the config only when TT_DRAM_SHARDED_MULTICORE=1 opts into a multi-core pipeline
# that this ttnn lacks, so here it keeps the config and refuses the opt-in.
def with_dram_sharded_cores_per_bank(program_config, m: int, k: int, n: int, num_cores: int):
    import os

    if os.environ.get("TT_DRAM_SHARDED_MULTICORE", "0") not in ("", "0"):
        raise RuntimeError("TT_DRAM_SHARDED_MULTICORE=1 needs tt-metal v0.82's multi-core DRAM-sharded matmul")
    return program_config
