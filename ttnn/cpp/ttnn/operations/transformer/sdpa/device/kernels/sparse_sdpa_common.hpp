// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// Host/device constants shared by the sparse-SDPA factory and JIT kernels. Keep this header free of host-only and
// kernel-only dependencies so both compilation paths use the same protocol and packed-cache geometry.
namespace sparse_sdpa {

// Uniform dispatch values precede tensor-accessor metadata in each kernel's common arguments.
namespace reader_common_arg {
enum : uint32_t { Q_ADDRESS, KV_ADDRESS, INDICES_ADDRESS, KV_BATCH_PAGE_OFFSET, END };
}
namespace writer_common_arg {
enum : uint32_t { OUTPUT_ADDRESS, KV_ADDRESS, KV_BATCH_PAGE_OFFSET, ATTENTION_SINK_ADDRESS, END };
}

constexpr uint32_t SCALE_BLOCK_WIDTH = 128;
constexpr uint32_t PACKED_FIELD_ADDRESS_UNIT_BYTES = 16;
constexpr uint32_t CB_PAGE_ALIGNMENT = 16;
constexpr uint32_t CB_DOUBLE_BUFFER_DEPTH = 2;

constexpr uint32_t align_cb_page(uint32_t bytes) {
    return ((bytes + CB_PAGE_ALIGNMENT - 1) / CB_PAGE_ALIGNMENT) * CB_PAGE_ALIGNMENT;
}

constexpr uint32_t scale_block_count(uint32_t latent_width) { return latent_width / SCALE_BLOCK_WIDTH; }

constexpr uint32_t scaled_kv_scale_bytes(uint32_t latent_width) {
    return scale_block_count(latent_width) * sizeof(float);
}

constexpr bool scaled_kv_rope_offset_is_aligned(uint32_t latent_width) {
    return (latent_width + scaled_kv_scale_bytes(latent_width)) % PACKED_FIELD_ADDRESS_UNIT_BYTES == 0;
}

constexpr uint32_t packed_kv_payload_bytes(uint32_t k_dim, uint32_t latent_width) {
    return latent_width + scaled_kv_scale_bytes(latent_width) + (k_dim - latent_width) * sizeof(uint16_t);
}

// Reader -> writer dual-NoC gather request. SCALE_DST_L1 is used only by the scaled-cache path.
namespace gather_request {
constexpr uint32_t BASE = 0;
constexpr uint32_t SPLIT = 1;
constexpr uint32_t IS_LAST = 2;
constexpr uint32_t DST_L1 = 3;
constexpr uint32_t SCALE_DST_L1 = 4;
constexpr uint32_t WORD_COUNT = 5;
constexpr uint32_t PAGE_BYTES = align_cb_page(WORD_COUNT * sizeof(uint32_t));
}  // namespace gather_request

// Reader -> compute token control message.
namespace control_message {
constexpr uint32_t ACTIVE_CHUNKS = 0;
constexpr uint32_t VALID_KEYS = 1;
// Key split (kv_split > 1) only: this core's chunk range [CHUNK_BEGIN, CHUNK_END) of the token's ACTIVE_CHUNKS,
// its role, and the token's participant count (cores that hold >= 1 chunk; the reducer merges that many partials).
constexpr uint32_t CHUNK_BEGIN = 2;
constexpr uint32_t CHUNK_END = 3;
constexpr uint32_t ROLE = 4;
constexpr uint32_t PARTS = 5;
constexpr uint32_t WORD_COUNT = 6;
constexpr uint32_t PAGE_BYTES = align_cb_page(WORD_COUNT * sizeof(uint32_t));
}  // namespace control_message

constexpr uint32_t ACK_PAGE_BYTES = CB_PAGE_ALIGNMENT;

// Key split: a token's active chunks are spread over KV_SPLIT cores (one group per token). Every participant
// exports its unnormalized flash state (max, partial row-sum, out) into slot j of the group reducer's receive
// buffers and bumps the reducer's arrival semaphore; the reducer (j = 0) merges them in one matmul pass
// (out = sum_j diag(exp(m_j - M)) @ out_j, FP32 DEST) and normalizes. Idle cores (j >= participants) skip it.
namespace split_role {
constexpr uint32_t IDLE = 0;
constexpr uint32_t WORKER = 1;
constexpr uint32_t REDUCER = 2;
}  // namespace split_role
// Reader -> writer per-token split control (role, participants, local chunk count).
namespace split_message {
constexpr uint32_t ROLE = 0;
constexpr uint32_t PARTS = 1;
constexpr uint32_t LOCAL_CHUNKS = 2;
constexpr uint32_t WORD_COUNT = 3;
constexpr uint32_t PAGE_BYTES = align_cb_page(WORD_COUNT * sizeof(uint32_t));
}  // namespace split_message
constexpr uint32_t SPLIT_ARRIVAL_SEMAPHORE = 0;

// Chunk range of participant j (of p) over n chunks: the remainder goes to the highest j, so the reducer (j = 0)
// carries the fewest chunks and starts its merge early.
constexpr uint32_t split_chunk_begin(uint32_t n, uint32_t p, uint32_t j) {
    const uint32_t base = n / p, rem = n % p, lo = p - rem;
    return j * base + (j > lo ? j - lo : 0);
}
constexpr uint32_t split_chunk_end(uint32_t n, uint32_t p, uint32_t j) { return split_chunk_begin(n, p, j + 1); }

// Compile-time argument positions. The factory checks the vector size against each END value before appending
// TensorAccessorArgs, while the kernels use the same names to decode the stream.
namespace reader_ct_arg {
enum : uint32_t {
    H = 0,
    S,
    TOPK,
    K_CHUNK,
    K_DIM,
    CB_Q_RM,
    CB_K_RM,
    CB_MASK_PART,
    CB_IDX,
    CB_CTRL,
    Q_ELEM_BYTES,
    KV_ELEM_BYTES,
    IDX_ELEM_BYTES,
    BLOCK_CYCLIC,
    BC_CHUNK_LOCAL,
    BC_SP,
    BC_SHARD_STRIDE_GAP,
    BC_SLAB_STRIDE_GAP,
    SCALED_KV,
    LATENT_WIDTH,
    CB_K_SCALE_BCAST,
    PACKED_ROW_BYTES,
    CB_KREQ,
    CB_KACK,
    KV_SPLIT,
    CB_SPLIT_CTL,
    CB_RMAX,
    CB_RSUM,
    CB_ROUT,
    V_DHT,
    END,
};
}  // namespace reader_ct_arg

namespace writer_ct_arg {
enum : uint32_t {
    H = 0,
    S,
    V_DHT,
    CB_OUT_RM,
    CB_SCALE,
    CB_COL_IDENTITY,
    CB_NEGINF,
    OUT_ELEM_BYTES,
    BLOCK_CYCLIC,
    BC_CHUNK_LOCAL,
    BC_SP,
    BC_SHARD_STRIDE_GAP,
    BC_SLAB_STRIDE_GAP,
    SCALED_KV,
    K_DIM,
    KV_ELEM_BYTES,
    CB_IDX,
    CB_KREQ,
    CB_KACK,
    PACKED_ROW_BYTES,
    USE_ATTENTION_SINK,
    CB_ATTENTION_SINK,
    CB_SINK_SCRATCH,
    KV_SPLIT,
    CB_SPLIT_CTL,
    CB_SPLIT_DONE,
    CB_RMAX,
    CB_RSUM,
    CB_ROUT,
    CB_MAX_A,
    CB_MAX_B,
    CB_SUM_A,
    CB_SUM_B,
    CB_OUT_A,
    CB_OUT_B,
    CB_IDENTITY,
    END,
};
}  // namespace writer_ct_arg

namespace compute_ct_arg {
enum : uint32_t {
    H = 0,
    DHT,
    V_DHT,
    SKT,
    SCALE,
    CB_Q_RM,
    CB_Q_IN,
    CB_K_RM,
    CB_K_IN,
    CB_NEGINF,
    CB_MASK_PART,
    CB_SCALE,
    CB_QK_IM,
    CB_MAX_A,
    CB_MAX_B,
    CB_SUM_A,
    CB_SUM_B,
    CB_OUT_A,
    CB_OUT_B,
    CB_CORR,
    CB_OUT_IM,
    CB_OUT_RM,
    CB_CTRL,
    CB_COL_IDENTITY,
    CB_RECIP_SCRATCH,
    SCALED_KV,
    CB_K_ROPE_RM,
    CB_K_SCALE_BCAST,
    CB_K_LATENT_TILE,
    CB_K_ROPE_TILE,
    MATH_APPROX_MODE,
    QUERY_SUBBLOCK,
    PACKED_ROW_BYTES,
    USE_ATTENTION_SINK,
    CB_ATTENTION_SINK,
    KV_SPLIT,
    CB_SPLIT_DONE,
    CB_RMAX,
    CB_RSUM,
    CB_ROUT,
    CB_GMAX,
    CB_CORR_ALL,
    CB_DIAG,
    CB_IDENTITY,
    END,
};
}  // namespace compute_ct_arg

}  // namespace sparse_sdpa
