// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// Fused host MoE experts on torch's CPU int4 pack (torch._convert_weight_to_int4pack_for_cpu, innerKTiles 1):
// a weight [N, K] is stored as N/64 blocks of K*32 bytes; byte k*32 + j of block b holds rows 64b+j (low nibble) and
// 64b+32+j (high nibble) at column k. scales_and_zeros [K/gs, N, 2] bf16 (scale, zero): w = (q - 8) * scale + zero.
//
// hostk_moe: out[R, H] += sum_e W[e, r] * down_e(silu(gate_e(x_r)) * up_e(x_r)) over the rows r with W[e, r] != 0,
// for one layer's missed experts in one call. Phase 1 splits (expert, 64-wide block of the I gate/up rows), so each
// unit has its gate and up rows together and writes the activation directly; phase 2 splits 64-wide blocks of the H
// outputs, so every unit owns its output columns across all experts. A persistent pthread pool (spin, then futex)
// runs the units; the calling thread takes part. Built by hostk.py (gcc -O3 -mavx512f -mavx512bw).
//
// Bit-plane layout (hostk_planarize, HYB_LOWBIT): each 64-row block's K*32 bytes become a hi plane (K*16 bytes: the
// top 2 bits of q) followed by a lo plane (K*16: the bottom 2 bits). Byte j of a plane's 16 bytes at column k holds
// rows j, 16+j, 32+j, 48+j at bits 0-1, 2-3, 4-5, 6-7. A full read reproduces q = 4h + l exactly; a hi-only read
// (low-weight misses) streams half the weight bytes and uses q ~ 4h + 1.5, the middle of the dropped bits' range.
#define _GNU_SOURCE
#include <immintrin.h>
#include <linux/futex.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#define MAXR 64  // rows per hostk_moe call (decode: 1-4; prefill: 64-row tiles of (row, expert) pairs)
#define TMAXM 16  // rows of the predict-then-touch router scoring (hostk_touch_predict)
#define MAXE 640
#define MAXT 32
#ifndef HOSTK_PF
#define HOSTK_PF 0  // software prefetch distance in bytes ahead of the weight stream (0: off)
#endif
#ifndef HOSTK_PF_HINT
#define HOSTK_PF_HINT \
    _MM_HINT_T0  // _MM_HINT_NTA: the streamed weights bypass L2 / L3 (no eviction of the host step's data)
#endif
#define SPIN 2000  // ~50 us of pause before futex sleep: a longer spin starves the host step's other threads

typedef struct {
    int R, H, I, gs, E;
    const float* X;              // [R, H]
    const int64_t* rows;         // [E] row of each expert in the library tensors
    const float* W;              // [E, R]
    const int8_t* lo;            // [E] 1: hi plane only (NULL: all full)
    int planar;                  // library in the bit-plane layout
    const uint8_t *gu_p, *dn_p;  // library bases
    const uint16_t *gu_sz, *dn_sz;
    float* out;  // [R, H]
    // per expert: active rows (compacted)
    int ne[MAXE];
    int er[MAXE][MAXR];
    float* xs_gu;  // [E][MAXR][H/gs]  group sums of the active x rows
    float* A;      // [E][MAXR][I]     activations
    float* xs_dn;  // [E][MAXR][I/gs]  group sums of A
    // bf (hostk_set_bf16): experts with more than 4 rows in a call take the AVX512-BF16 dot products, on bf16 copies
    // of their inputs (X: exact, the host rows come from bf16; A: rounded, as the device path's activations)
    int bf;
    uint16_t* Xb;  // [R][H]         bf16 copy of X
    uint16_t* Ab;  // [E][MAXR][I]   bf16 copy of A
    int phase, units;
} job_t;

static job_t J;
static int g_bf = 0;  // hostk_set_bf16
static int g_nt = 0;
static _Atomic int g_active = 0;  // workers with an index >= this skip the phases (hostk_set_active); 0 = all
static pthread_t g_th[MAXT];
static _Atomic int g_gen = 0, g_next = 0, g_done = 0;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static size_t g_scratch_sz = 0;
static float* g_scratch = NULL;

static inline float bf16f(uint16_t v) {
    union {
        uint32_t u;
        float f;
    } c = {.u = (uint32_t)v << 16};
    return c.f;
}

// acc[r][0..63] += sum_k w[n0+.., k] x_r[k] over K for one 64-row block; rows in chunks of 4 (x rows beyond nr = 0).
// xs[r][g] = sum of x_r over group g. w: the block's K*32 bytes; sz: scales_and_zeros + n0*2 ([K/gs, N, 2]).
static void gemv_block_many(
    const uint8_t* w, const uint16_t* sz, int N, int K, int gs, const float* const* x, const float* const* xs, int nr,
    int layout, float out[][64]);
#define BF16_FN __attribute__((target("avx512bf16")))  // the BF16 path only runs when hostk_set_bf16(1) (CPU checked)
BF16_FN static void gemv_block_bf(
    const uint8_t* w, const uint16_t* sz, int N, int K, int gs, const uint16_t* const* xb, const float* const* xs,
    int nr, int layout, float out[][64]);
BF16_FN static void to_bf16(const float* x, uint16_t* y, size_t n);

static inline uint16_t f2bf(float f) {  // round to nearest even (finite inputs)
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

static inline float bf2f(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static void gemv_block(
    const uint8_t* w,
    const uint16_t* sz,
    int N,
    int K,
    int gs,
    const float* const* x,
    const float* const* xs,
    int nr,
    int layout,  // 0: torch pack, 1: bit planes (full), 2: hi plane only
    float out[][64],
    const uint16_t* const* xb) {  // bf16 rows (J.bf, nr > 4) or NULL
    if (nr > 4 && xb) {  // prefill tiles: AVX512-BF16 dot products over the bf16 rows
        gemv_block_bf(w, sz, N, K, gs, xb, xs, nr, layout, out);
        return;
    }
    if (nr > 4) {  // prefill tiles: each group's weights decoded once (gemv_block_many), not once per 4 rows
        gemv_block_many(w, sz, N, K, gs, x, xs, nr, layout, out);
        return;
    }
    const int G = K / gs;
    for (int r0 = 0; r0 < nr; r0 += 4) {
        const int rc = nr - r0 < 4 ? nr - r0 : 4;
        __m512 tot[4][4];
        for (int r = 0; r < 4; r++) {
            for (int i = 0; i < 4; i++) {
                tot[r][i] = _mm512_setzero_ps();
            }
        }
        for (int g = 0; g < G; g++) {
            __m512 acc[4][4];
            for (int r = 0; r < 4; r++) {
                for (int i = 0; i < 4; i++) {
                    acc[r][i] = _mm512_setzero_ps();
                }
            }
            const uint8_t* p = w + (size_t)g * gs * (layout ? 16 : 32);
            const uint8_t* pl = w + (size_t)K * 16 + (size_t)g * gs * 16;
            const __m512i m15 = _mm512_set1_epi32(15), m3 = _mm512_set1_epi32(3);
            for (int k = 0; k < gs; k++, p += layout ? 16 : 32, pl += 16) {
#if HOSTK_PF > 0
                if (!(k & 1)) {  // one 64-byte line every two k (32 bytes per k in the torch pack)
                    _mm_prefetch((const char*)(p + HOSTK_PF), HOSTK_PF_HINT);
                    if (layout == 1) {
                        _mm_prefetch((const char*)(pl + HOSTK_PF), HOSTK_PF_HINT);
                    }
                }
#endif
                __m512 f[4];
                if (layout == 0) {
                    __m512i v0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)p));
                    __m512i v1 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(p + 16)));
                    f[0] = _mm512_cvtepi32_ps(_mm512_and_si512(v0, m15));
                    f[1] = _mm512_cvtepi32_ps(_mm512_and_si512(v1, m15));
                    f[2] = _mm512_cvtepi32_ps(_mm512_srli_epi32(v0, 4));
                    f[3] = _mm512_cvtepi32_ps(_mm512_srli_epi32(v1, 4));
                } else if (layout == 1) {
                    __m512i vh = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)p));
                    __m512i vl = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)pl));
                    for (int i = 0; i < 4; i++) {
                        __m512i h = _mm512_and_si512(_mm512_srli_epi32(vh, 2 * i), m3);
                        __m512i l = _mm512_and_si512(_mm512_srli_epi32(vl, 2 * i), m3);
                        f[i] = _mm512_cvtepi32_ps(_mm512_or_si512(_mm512_slli_epi32(h, 2), l));
                    }
                } else {
                    __m512i vh = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)p));
                    for (int i = 0; i < 4; i++) {
                        f[i] = _mm512_cvtepi32_ps(_mm512_and_si512(_mm512_srli_epi32(vh, 2 * i), m3));
                    }
                }
                const int kk = g * gs + k;
#define ROW(r)                                                                        \
    if (rc > r) {                                                                     \
        __m512 xb = _mm512_set1_ps(x[r0 + r][kk]);                                    \
        for (int i = 0; i < 4; i++) acc[r][i] = _mm512_fmadd_ps(f[i], xb, acc[r][i]); \
    }
                ROW(0) ROW(1) ROW(2) ROW(3)
#undef ROW
            }
            const uint32_t* s32 = (const uint32_t*)(sz + (size_t)g * N * 2);
            for (int i = 0; i < 4; i++) {
                __m512i u = _mm512_loadu_si512((const void*)(s32 + i * 16));
                __m512 s = _mm512_castsi512_ps(_mm512_slli_epi32(u, 16));
                __m512 z = _mm512_castsi512_ps(_mm512_and_si512(u, _mm512_set1_epi32((int)0xFFFF0000u)));
                // w = (q - 8) s + z; hi only: q ~ 4h + 1.5 -> w = h (4 s) + z - 6.5 s
                __m512 zz = _mm512_fnmadd_ps(_mm512_set1_ps(layout == 2 ? 6.5f : 8.0f), s, z);
                if (layout == 2) {
                    s = _mm512_mul_ps(s, _mm512_set1_ps(4.0f));
                }
                for (int r = 0; r < rc; r++) {
                    tot[r][i] = _mm512_fmadd_ps(s, acc[r][i], tot[r][i]);
                    tot[r][i] = _mm512_fmadd_ps(zz, _mm512_set1_ps(xs[r0 + r][g]), tot[r][i]);
                }
            }
        }
        for (int r = 0; r < rc; r++) {
            for (int i = 0; i < 4; i++) {
                _mm512_storeu_ps(out[r0 + r] + i * 16, tot[r][i]);
            }
        }
    }
}

// gemv_block for many rows: per group, the gs x 64 weights are decoded once into an L1 buffer (fp32, q - as stored)
// and every 4-row chunk FMAs from it; the per-group scale / zero step is the same as gemv_block's, in the same order,
// so the result is bit-identical to gemv_block's.
static void gemv_block_many(
    const uint8_t* w,
    const uint16_t* sz,
    int N,
    int K,
    int gs,
    const float* const* x,
    const float* const* xs,
    int nr,
    int layout,
    float out[][64]) {
    const int G = K / gs;
    float wf[64 * 64] __attribute__((aligned(64)));  // gs <= 64
    for (int r = 0; r < nr; r++) {
        memset(out[r], 0, 64 * sizeof(float));
    }
    const __m512i m15 = _mm512_set1_epi32(15), m3 = _mm512_set1_epi32(3);
    for (int g = 0; g < G; g++) {
        const uint8_t* p = w + (size_t)g * gs * (layout ? 16 : 32);
        const uint8_t* pl = w + (size_t)K * 16 + (size_t)g * gs * 16;
        for (int k = 0; k < gs; k++, p += layout ? 16 : 32, pl += 16) {
            __m512 f[4];
            if (layout == 0) {
                __m512i v0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)p));
                __m512i v1 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(p + 16)));
                f[0] = _mm512_cvtepi32_ps(_mm512_and_si512(v0, m15));
                f[1] = _mm512_cvtepi32_ps(_mm512_and_si512(v1, m15));
                f[2] = _mm512_cvtepi32_ps(_mm512_srli_epi32(v0, 4));
                f[3] = _mm512_cvtepi32_ps(_mm512_srli_epi32(v1, 4));
            } else if (layout == 1) {
                __m512i vh = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)p));
                __m512i vl = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)pl));
                for (int i = 0; i < 4; i++) {
                    __m512i h = _mm512_and_si512(_mm512_srli_epi32(vh, 2 * i), m3);
                    __m512i l = _mm512_and_si512(_mm512_srli_epi32(vl, 2 * i), m3);
                    f[i] = _mm512_cvtepi32_ps(_mm512_or_si512(_mm512_slli_epi32(h, 2), l));
                }
            } else {
                __m512i vh = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)p));
                for (int i = 0; i < 4; i++) {
                    f[i] = _mm512_cvtepi32_ps(_mm512_and_si512(_mm512_srli_epi32(vh, 2 * i), m3));
                }
            }
            for (int i = 0; i < 4; i++) {
                _mm512_store_ps(wf + k * 64 + i * 16, f[i]);
            }
        }
        const uint32_t* s32 = (const uint32_t*)(sz + (size_t)g * N * 2);
        __m512 sv[4], zv[4];
        for (int i = 0; i < 4; i++) {
            __m512i u = _mm512_loadu_si512((const void*)(s32 + i * 16));
            __m512 s = _mm512_castsi512_ps(_mm512_slli_epi32(u, 16));
            __m512 z = _mm512_castsi512_ps(_mm512_and_si512(u, _mm512_set1_epi32((int)0xFFFF0000u)));
            zv[i] = _mm512_fnmadd_ps(_mm512_set1_ps(layout == 2 ? 6.5f : 8.0f), s, z);
            sv[i] = layout == 2 ? _mm512_mul_ps(s, _mm512_set1_ps(4.0f)) : s;
        }
        for (int r0 = 0; r0 < nr; r0 += 4) {
            const int rc = nr - r0 < 4 ? nr - r0 : 4;
            __m512 acc[4][4];
            for (int r = 0; r < 4; r++) {
                for (int i = 0; i < 4; i++) {
                    acc[r][i] = _mm512_setzero_ps();
                }
            }
            for (int k = 0; k < gs; k++) {
                const int kk = g * gs + k;
                const float* wk = wf + k * 64;
                __m512 f0 = _mm512_load_ps(wk), f1 = _mm512_load_ps(wk + 16), f2 = _mm512_load_ps(wk + 32),
                       f3 = _mm512_load_ps(wk + 48);
#define ROWM(r)                                         \
    if (rc > r) {                                       \
        __m512 xb = _mm512_set1_ps(x[r0 + r][kk]);      \
        acc[r][0] = _mm512_fmadd_ps(f0, xb, acc[r][0]); \
        acc[r][1] = _mm512_fmadd_ps(f1, xb, acc[r][1]); \
        acc[r][2] = _mm512_fmadd_ps(f2, xb, acc[r][2]); \
        acc[r][3] = _mm512_fmadd_ps(f3, xb, acc[r][3]); \
    }
                ROWM(0) ROWM(1) ROWM(2) ROWM(3)
#undef ROWM
            }
            for (int r = 0; r < rc; r++) {
                for (int i = 0; i < 4; i++) {
                    __m512 o = _mm512_loadu_ps(out[r0 + r] + i * 16);
                    o = _mm512_fmadd_ps(sv[i], acc[r][i], o);
                    o = _mm512_fmadd_ps(zv[i], _mm512_set1_ps(xs[r0 + r][g]), o);
                    _mm512_storeu_ps(out[r0 + r] + i * 16, o);
                }
            }
        }
    }
}

BF16_FN static void to_bf16(const float* x, uint16_t* y, size_t n) {  // n % 16 == 0, round to nearest even
    for (size_t i = 0; i < n; i += 16) {
        _mm256_storeu_si256((__m256i*)(y + i), (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(x + i)));
    }
}

// gemv_block_many with AVX512-BF16: per group, the gs x 64 weights (q, exact in bf16) are decoded once into an L1 buffer
// of (k, k+1) pairs per column, and each row's (x_k, x_k+1) bf16 pair is broadcast into vdpbf16ps (two products per lane
// per instruction, fp32 accumulation). The scale / zero step is gemv_block's.
BF16_FN static void gemv_block_bf(
    const uint8_t* w,
    const uint16_t* sz,
    int N,
    int K,
    int gs,
    const uint16_t* const* xb,
    const float* const* xs,
    int nr,
    int layout,
    float out[][64]) {
    const int G = K / gs;
    __m512i wb[32 * 4] __attribute__((aligned(64)));  // [gs / 2][4]: 32 bf16 = 16 columns x (k, k+1)
    for (int r = 0; r < nr; r++) {
        memset(out[r], 0, 64 * sizeof(float));
    }
    const __m512i m15 = _mm512_set1_epi32(15), m3 = _mm512_set1_epi32(3);
    // lanes 2c, 2c+1 <- column c of k (lo half) and of k+1 (hi half)
    const __m512i pidx = _mm512_set_epi16(
        31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8, 23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);
    for (int g = 0; g < G; g++) {
        const uint8_t* p = w + (size_t)g * gs * (layout ? 16 : 32);
        const uint8_t* pl = w + (size_t)K * 16 + (size_t)g * gs * 16;
        __m512 prev[4];
        for (int k = 0; k < gs; k++, p += layout ? 16 : 32, pl += 16) {
            __m512 f[4];
            if (layout == 0) {
                __m512i v0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)p));
                __m512i v1 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(p + 16)));
                f[0] = _mm512_cvtepi32_ps(_mm512_and_si512(v0, m15));
                f[1] = _mm512_cvtepi32_ps(_mm512_and_si512(v1, m15));
                f[2] = _mm512_cvtepi32_ps(_mm512_srli_epi32(v0, 4));
                f[3] = _mm512_cvtepi32_ps(_mm512_srli_epi32(v1, 4));
            } else if (layout == 1) {
                __m512i vh = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)p));
                __m512i vl = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)pl));
                for (int i = 0; i < 4; i++) {
                    __m512i h = _mm512_and_si512(_mm512_srli_epi32(vh, 2 * i), m3);
                    __m512i l = _mm512_and_si512(_mm512_srli_epi32(vl, 2 * i), m3);
                    f[i] = _mm512_cvtepi32_ps(_mm512_or_si512(_mm512_slli_epi32(h, 2), l));
                }
            } else {
                __m512i vh = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)p));
                for (int i = 0; i < 4; i++) {
                    f[i] = _mm512_cvtepi32_ps(_mm512_and_si512(_mm512_srli_epi32(vh, 2 * i), m3));
                }
            }
            if (k & 1) {
                for (int i = 0; i < 4; i++) {
                    __m256i lo = (__m256i)_mm512_cvtneps_pbh(prev[i]);
                    __m256i hi = (__m256i)_mm512_cvtneps_pbh(f[i]);
                    __m512i both = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
                    _mm512_store_si512(wb + (k >> 1) * 4 + i, _mm512_permutexvar_epi16(pidx, both));
                }
            } else {
                for (int i = 0; i < 4; i++) {
                    prev[i] = f[i];
                }
            }
        }
        const uint32_t* s32 = (const uint32_t*)(sz + (size_t)g * N * 2);
        __m512 sv[4], zv[4];
        for (int i = 0; i < 4; i++) {
            __m512i u = _mm512_loadu_si512((const void*)(s32 + i * 16));
            __m512 s = _mm512_castsi512_ps(_mm512_slli_epi32(u, 16));
            __m512 z = _mm512_castsi512_ps(_mm512_and_si512(u, _mm512_set1_epi32((int)0xFFFF0000u)));
            zv[i] = _mm512_fnmadd_ps(_mm512_set1_ps(layout == 2 ? 6.5f : 8.0f), s, z);
            sv[i] = layout == 2 ? _mm512_mul_ps(s, _mm512_set1_ps(4.0f)) : s;
        }
        for (int r0 = 0; r0 < nr; r0 += 4) {
            const int rc = nr - r0 < 4 ? nr - r0 : 4;
            __m512 acc[4][4];
            for (int r = 0; r < 4; r++) {
                for (int i = 0; i < 4; i++) {
                    acc[r][i] = _mm512_setzero_ps();
                }
            }
            for (int kp = 0; kp < gs / 2; kp++) {
                const int kk = g * gs + 2 * kp;
                const __m512i* wk = wb + kp * 4;
                __m512bh f0 = (__m512bh)_mm512_load_si512(wk), f1 = (__m512bh)_mm512_load_si512(wk + 1),
                         f2 = (__m512bh)_mm512_load_si512(wk + 2), f3 = (__m512bh)_mm512_load_si512(wk + 3);
#define ROWB(r)                                                                         \
    if (rc > r) {                                                                       \
        uint32_t pr;                                                                    \
        memcpy(&pr, xb[r0 + r] + kk, 4);                                                \
        __m512bh xp = (__m512bh)_mm512_set1_epi32((int)pr);                             \
        acc[r][0] = _mm512_dpbf16_ps(acc[r][0], f0, xp);                                \
        acc[r][1] = _mm512_dpbf16_ps(acc[r][1], f1, xp);                                \
        acc[r][2] = _mm512_dpbf16_ps(acc[r][2], f2, xp);                                \
        acc[r][3] = _mm512_dpbf16_ps(acc[r][3], f3, xp);                                \
    }
                ROWB(0) ROWB(1) ROWB(2) ROWB(3)
#undef ROWB
            }
            for (int r = 0; r < rc; r++) {
                for (int i = 0; i < 4; i++) {
                    __m512 o = _mm512_loadu_ps(out[r0 + r] + i * 16);
                    o = _mm512_fmadd_ps(sv[i], acc[r][i], o);
                    o = _mm512_fmadd_ps(zv[i], _mm512_set1_ps(xs[r0 + r][g]), o);
                    _mm512_storeu_ps(out[r0 + r] + i * 16, o);
                }
            }
        }
    }
}

static void unit(int u) {
    const int H = J.H, I = J.I, gs = J.gs;
    const size_t gu_bytes = (size_t)2 * I * H / 2, dn_bytes = (size_t)H * I / 2;
    const size_t gu_sz_n = (size_t)(H / gs) * 2 * I * 2, dn_sz_n = (size_t)(I / gs) * H * 2;
    if (J.phase == 1) {
        const int nb = I / 64, e = u / nb, j = u % nb, nr = J.ne[e];
        const int64_t row = J.rows[e];
        const uint8_t* wb = J.gu_p + row * gu_bytes;
        const uint16_t* sb = J.gu_sz + row * gu_sz_n;
        const float *x[MAXR], *xs[MAXR];
        const uint16_t* xbr[MAXR];
        const int bf = J.bf && nr > 4;
        for (int r = 0; r < nr; r++) {
            x[r] = J.X + (size_t)J.er[e][r] * H;
            xbr[r] = bf ? J.Xb + (size_t)J.er[e][r] * H : NULL;
            xs[r] = J.xs_gu + ((size_t)e * MAXR + r) * (H / gs);
        }
        const int lay = J.planar ? (J.lo && J.lo[e] ? 2 : 1) : 0;
        float g[MAXR][64], up[MAXR][64];
        gemv_block(wb + (size_t)j * H * 32, sb + (size_t)j * 64 * 2, 2 * I, H, gs, x, xs, nr, lay, g, bf ? xbr : NULL);
        gemv_block(
            wb + (size_t)(nb + j) * H * 32, sb + (size_t)(I + j * 64) * 2, 2 * I, H, gs, x, xs, nr, lay, up,
            bf ? xbr : NULL);
        for (int r = 0; r < nr; r++) {
            float* a = J.A + ((size_t)e * MAXR + r) * I + j * 64;
            for (int t = 0; t < 64; t++) {
                a[t] = g[r][t] / (1.0f + expf(-g[r][t])) * up[r][t];
            }
            if (bf) {  // the down projection reads A in bf16: its group sums are of the rounded values
                uint16_t* ab = J.Ab + ((size_t)e * MAXR + r) * I + j * 64;
                for (int t = 0; t < 64; t++) {
                    ab[t] = f2bf(a[t]);
                    a[t] = bf2f(ab[t]);
                }
            }
            float* xd = J.xs_dn + ((size_t)e * MAXR + r) * (I / gs) + j * 64 / gs;
            for (int q = 0; q < 64 / gs; q++) {
                float s = 0;
                for (int t = 0; t < gs; t++) {
                    s += a[q * gs + t];
                }
                xd[q] = s;
            }
        }
    } else {
        const int hb = u;
        float y[MAXR][64];
        for (int e = 0; e < J.E; e++) {
            const int nr = J.ne[e];
            const int64_t row = J.rows[e];
            const float *x[MAXR], *xs[MAXR];
            const uint16_t* xbr[MAXR];
            const int bf = J.bf && nr > 4;
            for (int r = 0; r < nr; r++) {
                x[r] = J.A + ((size_t)e * MAXR + r) * I;
                xbr[r] = bf ? J.Ab + ((size_t)e * MAXR + r) * I : NULL;
                xs[r] = J.xs_dn + ((size_t)e * MAXR + r) * (I / gs);
            }
            gemv_block(
                J.dn_p + row * dn_bytes + (size_t)hb * I * 32,
                J.dn_sz + row * dn_sz_n + (size_t)hb * 64 * 2,
                H,
                I,
                gs,
                x,
                xs,
                nr,
                J.planar ? (J.lo && J.lo[e] ? 2 : 1) : 0,
                y,
                bf ? xbr : NULL);
            for (int r = 0; r < nr; r++) {
                const int rr = J.er[e][r];
                const float wv = J.W[(size_t)e * J.R + rr];
                float* o = J.out + (size_t)rr * H + hb * 64;
                for (int t = 0; t < 64; t++) {
                    o[t] += wv * y[r][t];
                }
            }
        }
    }
}

static void run_units(void) {
    for (;;) {
        int u = atomic_fetch_add(&g_next, 1);
        if (u >= J.units) {
            break;
        }
        unit(u);
    }
}

static void* worker(void* arg) {
    const int tid = (int)(intptr_t)arg;
    int seen = atomic_load(&g_gen);
    for (;;) {
        int spins = 0;
        while (atomic_load_explicit(&g_gen, memory_order_acquire) == seen) {
            if (++spins < SPIN) {
                _mm_pause();
            } else {
                syscall(SYS_futex, &g_gen, FUTEX_WAIT_PRIVATE, seen, NULL, NULL, 0);
            }
        }
        seen = atomic_load(&g_gen);
        const int active = atomic_load(&g_active);
        if (!active || tid < active) {
            run_units();
        }
        atomic_fetch_add_explicit(&g_done, 1, memory_order_release);
    }
    return NULL;
}

static void run_phase(int phase, int units) {
    J.phase = phase;
    J.units = units;
    atomic_store(&g_next, 0);
    atomic_store(&g_done, 0);
    atomic_fetch_add_explicit(&g_gen, 1, memory_order_release);
    syscall(SYS_futex, &g_gen, FUTEX_WAKE_PRIVATE, 1 << 30, NULL, NULL, 0);
    run_units();
    while (atomic_load_explicit(&g_done, memory_order_acquire) < g_nt - 1) {
        _mm_pause();
    }
}

// Threads that run the kernel's phases from now on (1..nthreads of hostk_init; 0 = all): the others still wake and
// report done, so a phase never waits on them. Returns the previous setting.
int hostk_set_active(int n) { return atomic_exchange(&g_active, n < 0 ? 0 : n); }

// 1: experts with more than 4 rows in a call use the AVX512-BF16 path (gemv_block_bf); returns the previous setting.
int hostk_set_bf16(int on) {
    int prev = g_bf;
    g_bf = on ? 1 : 0;
    return prev;
}

int hostk_init(int nthreads) {
    if (g_nt) {
        return g_nt;
    }
    if (nthreads < 1) {
        nthreads = 1;
    }
    if (nthreads > MAXT) {
        nthreads = MAXT;
    }
    g_nt = nthreads;
    for (int t = 1; t < nthreads; t++) {
        pthread_create(&g_th[t], NULL, worker, (void*)(intptr_t)t);
    }
    return g_nt;
}

// Prefetch (HYB_PREFETCH): a few dedicated threads read given byte ranges (the next layer's predicted missed experts)
// while the host would otherwise idle, so hostk_moe later reads them from L2 / L3 (a victim cache on Zen: lines a
// toucher loads spill there from its L2) instead of DRAM. hostk_moe sets g_tabort: the touchers stop at the next 64
// KiB chunk. Only the cache contents change, never a result.
#define TMAXR 256
#define TCHUNK 65536
static pthread_t g_tth[MAXT];
static int g_tn = 0;
static _Atomic int g_tgen = 0, g_tnext = 0, g_tdone = 0, g_tabort = 0;
static int g_tcnt = 0, g_tchunks = 0;
static const uint8_t* g_tptr[TMAXR];
static int64_t g_tlen[TMAXR], g_tfirst[TMAXR + 1];
static volatile uint64_t g_tsink;

static void touch_units(void) {
    __m512i acc = _mm512_setzero_si512();
    for (;;) {
        if (atomic_load_explicit(&g_tabort, memory_order_relaxed)) {
            break;
        }
        int c = atomic_fetch_add(&g_tnext, 1);
        if (c >= g_tchunks) {
            break;
        }
        int r = 0;
        while (g_tfirst[r + 1] <= c) {
            r++;
        }
        int64_t off = (int64_t)(c - g_tfirst[r]) * TCHUNK, end = off + TCHUNK;
        if (end > g_tlen[r]) {
            end = g_tlen[r];
        }
        const uint8_t* p = g_tptr[r];
        for (int64_t o = off; o + 256 <= end; o += 256) {
            acc = _mm512_xor_si512(acc, _mm512_loadu_si512((const void*)(p + o)));
            acc = _mm512_xor_si512(acc, _mm512_loadu_si512((const void*)(p + o + 64)));
            acc = _mm512_xor_si512(acc, _mm512_loadu_si512((const void*)(p + o + 128)));
            acc = _mm512_xor_si512(acc, _mm512_loadu_si512((const void*)(p + o + 192)));
        }
    }
    g_tsink += (uint64_t)_mm512_reduce_add_epi64(acc);
}

// hostk_touch_predict: the touchers first score the next layer's router on this layer's MoE input (units of 8
// experts), the last one to finish selects (sigmoid + bias top-k per row, experts with a host pointer, more rows
// first, up to the byte cap) and publishes the ranges; then all of them touch.
typedef struct {
    const float* x;  // [m, H]
    int m, H, E, k;
    const uint16_t* W;     // [E, H] bf16 router weight
    const float* bias;     // [E]
    const uint64_t* eptr;  // [E, 4] addresses of gu_p / gu_sz / dn_p / dn_sz rows (0: resident, never touched)
    const int64_t* elen;   // [4] bytes of each
    int64_t cap;           // byte cap
    int32_t* pred;         // [E] out: 1 = predicted host expert (touched or over the cap)
} pred_t;
static pred_t g_p;
static int g_tmode = 0;  // 0: ranges, 1: predict then touch
static _Atomic int g_tunit = 0, g_tgated = 0, g_tready = 0;
static float g_z[TMAXM * 1024];

static void gate_units(void) {
    const int H = g_p.H, m = g_p.m, nu = (g_p.E + 7) / 8;
    for (;;) {
        if (atomic_load_explicit(&g_tabort, memory_order_relaxed)) {
            return;
        }
        int u = atomic_fetch_add(&g_tunit, 1);
        if (u >= nu) {
            return;
        }
        for (int e = u * 8; e < u * 8 + 8 && e < g_p.E; e++) {
            const uint16_t* w = g_p.W + (size_t)e * H;
            __m512 acc[TMAXM];
            for (int r = 0; r < m; r++) {
                acc[r] = _mm512_setzero_ps();
            }
            for (int h = 0; h < H; h += 16) {
                __m512 wf = _mm512_castsi512_ps(
                    _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(w + h))), 16));
                for (int r = 0; r < m; r++) {
                    acc[r] = _mm512_fmadd_ps(wf, _mm512_loadu_ps(g_p.x + (size_t)r * H + h), acc[r]);
                }
            }
            for (int r = 0; r < m; r++) {
                g_z[r * g_p.E + e] = _mm512_reduce_add_ps(acc[r]);
            }
        }
    }
}

static void select_ranges(void) {
    const int E = g_p.E, k = g_p.k;
    int cnt[1024] = {0};
    for (int r = 0; r < g_p.m; r++) {
        float sc[1024];
        for (int e = 0; e < E; e++) {
            sc[e] = 1.0f / (1.0f + expf(-g_z[r * E + e])) + g_p.bias[e];
        }
        for (int j = 0; j < k; j++) {
            int b = -1;
            for (int e = 0; e < E; e++) {
                if (sc[e] > -1e30f && (b < 0 || sc[e] > sc[b])) {
                    b = e;
                }
            }
            if (b < 0) {
                break;
            }
            sc[b] = -1e31f;
            cnt[b]++;
        }
    }
    int n = 0;
    int64_t tot = 0, per = g_p.elen[0] + g_p.elen[1] + g_p.elen[2] + g_p.elen[3];
    for (int c = g_p.m; c >= 1 && n + 4 <= TMAXR; c--) {
        for (int e = 0; e < E && n + 4 <= TMAXR; e++) {
            if (cnt[e] != c || !g_p.eptr[(size_t)e * 4]) {
                continue;
            }
            g_p.pred[e] = 1;
            if (tot + per > g_p.cap) {
                continue;
            }
            tot += per;
            for (int q = 0; q < 4; q++) {
                g_tptr[n] = (const uint8_t*)(uintptr_t)g_p.eptr[(size_t)e * 4 + q];
                g_tlen[n] = g_p.elen[q];
                g_tfirst[n + 1] = g_tfirst[n] + (g_tlen[n] + TCHUNK - 1) / TCHUNK;
                n++;
            }
        }
    }
    g_tcnt = n;
    g_tchunks = (int)g_tfirst[n];
}

static void* toucher(void* arg) {
    (void)arg;
    int seen = atomic_load(&g_tgen);
    for (;;) {
        while (atomic_load_explicit(&g_tgen, memory_order_acquire) == seen) {
            syscall(SYS_futex, &g_tgen, FUTEX_WAIT_PRIVATE, seen, NULL, NULL, 0);
        }
        seen = atomic_load(&g_tgen);
        if (g_tmode == 1) {
            gate_units();
            if (atomic_fetch_add(&g_tgated, 1) == g_tn - 1) {
                if (!atomic_load(&g_tabort)) {
                    select_ranges();
                }
                atomic_store_explicit(&g_tready, 1, memory_order_release);
            } else {
                while (!atomic_load_explicit(&g_tready, memory_order_acquire)) {
                    _mm_pause();
                }
            }
        }
        touch_units();
        atomic_fetch_add_explicit(&g_tdone, 1, memory_order_release);
    }
    return NULL;
}

int hostk_touch_init(int n) {
    if (g_tn) {
        return g_tn;
    }
    if (n > MAXT) {
        n = MAXT;
    }
    for (int t = 0; t < n; t++) {
        pthread_create(&g_tth[t], NULL, toucher, NULL);
    }
    g_tn = n;
    atomic_store(&g_tdone, n);  // idle: hostk_touch waits for every toucher to report done
    return g_tn;
}

// Starts reading n byte ranges (returns at once): -1 too many ranges, -4 not initialized. A touch still running is
// stopped first (its ranges are the stale prediction).
int hostk_touch(int n, const uint64_t* ptrs, const int64_t* lens) {
    if (!g_tn) {
        return -4;
    }
    if (n > TMAXR) {
        return -1;
    }
    atomic_store(&g_tabort, 1);
    while (atomic_load_explicit(&g_tdone, memory_order_acquire) < g_tn) {
        _mm_pause();
    }
    g_tmode = 0;
    g_tcnt = n;
    g_tfirst[0] = 0;
    for (int i = 0; i < n; i++) {
        g_tptr[i] = (const uint8_t*)(uintptr_t)ptrs[i];
        g_tlen[i] = lens[i];
        g_tfirst[i + 1] = g_tfirst[i] + (lens[i] + TCHUNK - 1) / TCHUNK;
    }
    g_tchunks = (int)g_tfirst[n];
    atomic_store(&g_tnext, 0);
    atomic_store(&g_tdone, 0);
    atomic_store(&g_tabort, 0);
    atomic_fetch_add_explicit(&g_tgen, 1, memory_order_release);
    syscall(SYS_futex, &g_tgen, FUTEX_WAKE_PRIVATE, 1 << 30, NULL, NULL, 0);
    return 0;
}

// Starts the predict-then-touch job (returns at once): -1 bad sizes, -4 not initialized. x, W, bias, eptr, elen and
// pred must stay alive until the next hostk_touch* / hostk_moe call; pred is zeroed here.
int hostk_touch_predict(
    const float* x,
    int m,
    int H,
    int E,
    int k,
    const uint16_t* W,
    const float* bias,
    const uint64_t* eptr,
    const int64_t* elen,
    int64_t cap,
    int32_t* pred) {
    if (!g_tn) {
        return -4;
    }
    if (m > TMAXM || E > 1024 || H % 16 || k > E) {
        return -1;
    }
    atomic_store(&g_tabort, 1);
    while (atomic_load_explicit(&g_tdone, memory_order_acquire) < g_tn) {
        _mm_pause();
    }
    memset(pred, 0, sizeof(int32_t) * (size_t)E);
    g_p = (pred_t){x, m, H, E, k, W, bias, eptr, elen, cap, pred};
    g_tmode = 1;
    g_tcnt = 0;
    g_tchunks = 0;
    g_tfirst[0] = 0;
    atomic_store(&g_tunit, 0);
    atomic_store(&g_tgated, 0);
    atomic_store(&g_tready, 0);
    atomic_store(&g_tnext, 0);
    atomic_store(&g_tdone, 0);
    atomic_store(&g_tabort, 0);
    atomic_fetch_add_explicit(&g_tgen, 1, memory_order_release);
    syscall(SYS_futex, &g_tgen, FUTEX_WAKE_PRIVATE, 1 << 30, NULL, NULL, 0);
    return 0;
}

// Returns 0, or a negative code: -1 bad shapes, -2 too many rows/experts, -3 out of memory, -4 not initialized.
int hostk_moe(
    int R,
    int H,
    int I,
    int gs,
    const float* X,
    int E,
    const int64_t* rows,
    const float* W,
    const int8_t* lo,
    int planar,
    const uint8_t* gu_p,
    const uint16_t* gu_sz,
    const uint8_t* dn_p,
    const uint16_t* dn_sz,
    float* out) {
    if (!g_nt) {
        return -4;
    }
    if (H % 64 || I % 64 || 64 % gs || H % gs) {
        return -1;
    }
    if (R > MAXR || E > MAXE) {
        return -2;
    }
    if (E == 0) {
        return 0;
    }
    atomic_store(&g_tabort, 1);  // a running prefetch (hostk_touch) yields the DRAM to the experts
    pthread_mutex_lock(&g_mu);
    size_t need = (size_t)E * MAXR * ((size_t)H / gs + I + (size_t)I / gs);
    need += ((size_t)R * H + (size_t)E * MAXR * I) / 2 + 32;  // Xb, Ab (bf16), alignment slack
    if (need > g_scratch_sz) {
        free(g_scratch);
        g_scratch = malloc(need * sizeof(float));
        g_scratch_sz = g_scratch ? need : 0;
        if (!g_scratch) {
            pthread_mutex_unlock(&g_mu);
            return -3;
        }
    }
    J.R = R, J.H = H, J.I = I, J.gs = gs, J.E = E, J.X = X, J.rows = rows, J.W = W, J.lo = lo, J.planar = planar;
    J.gu_p = gu_p, J.gu_sz = gu_sz, J.dn_p = dn_p, J.dn_sz = dn_sz, J.out = out;
    J.xs_gu = g_scratch;
    J.A = J.xs_gu + (size_t)E * MAXR * (H / gs);
    J.xs_dn = J.A + (size_t)E * MAXR * I;
    J.bf = g_bf;
    J.Xb = (uint16_t*)(J.xs_dn + (size_t)E * MAXR * (I / gs));
    J.Ab = J.Xb + (size_t)R * H;
    if (J.bf) {  // H % 64 == 0: whole vectors
        to_bf16(X, J.Xb, (size_t)R * H);
    }
    const int G = H / gs;
    for (int e = 0; e < E; e++) {
        int n = 0;
        for (int r = 0; r < R; r++) {
            if (W[(size_t)e * R + r] != 0.0f) {
                J.er[e][n] = r;
                float* xs = J.xs_gu + ((size_t)e * MAXR + n) * G;
                const float* x = X + (size_t)r * H;
                for (int g = 0; g < G; g++) {
                    float s = 0;
                    for (int t = 0; t < gs; t++) {
                        s += x[g * gs + t];
                    }
                    xs[g] = s;
                }
                n++;
            }
        }
        J.ne[e] = n;
    }
    run_phase(1, E * (I / 64));
    run_phase(2, H / 64);
    pthread_mutex_unlock(&g_mu);
    return 0;
}

// In place, between torch's int4 pack and the bit-plane layout (inverse = 1: back to torch's): nblocks 64-row blocks
// of K*32 bytes each. Returns 0, or -3 out of memory.
int hostk_planarize(uint8_t* p, int64_t nblocks, int K, int inverse) {
    uint8_t* tmp = malloc((size_t)K * 32);
    if (!tmp) {
        return -3;
    }
    for (int64_t b = 0; b < nblocks; b++) {
        uint8_t* w = p + (size_t)b * K * 32;
        for (int k = 0; k < K; k++) {
            uint8_t q[64];
            if (!inverse) {
                for (int j = 0; j < 32; j++) {
                    q[j] = w[(size_t)k * 32 + j] & 15;
                    q[32 + j] = w[(size_t)k * 32 + j] >> 4;
                }
                for (int j = 0; j < 16; j++) {
                    uint8_t h = 0, l = 0;
                    for (int i = 0; i < 4; i++) {
                        h |= (q[16 * i + j] >> 2) << (2 * i);
                        l |= (q[16 * i + j] & 3) << (2 * i);
                    }
                    tmp[(size_t)k * 16 + j] = h;
                    tmp[(size_t)K * 16 + (size_t)k * 16 + j] = l;
                }
            } else {
                for (int j = 0; j < 16; j++) {
                    const uint8_t h = w[(size_t)k * 16 + j], l = w[(size_t)K * 16 + (size_t)k * 16 + j];
                    for (int i = 0; i < 4; i++) {
                        q[16 * i + j] = (((h >> (2 * i)) & 3) << 2) | ((l >> (2 * i)) & 3);
                    }
                }
                for (int j = 0; j < 32; j++) {
                    tmp[(size_t)k * 32 + j] = q[j] | (q[32 + j] << 4);
                }
            }
        }
        memcpy(w, tmp, (size_t)K * 32);
    }
    free(tmp);
    return 0;
}

// ---- routing + crafted rows for the hybrid's host step 1 (one C call each instead of ~20 numpy calls per layer) ----
// Top-k of m rows of router logits lg [m, ld] over E experts: sigmoid_bias = 1 selects on sigmoid(z) + bias and weighs
// by the selected sigmoids normalized; else selects on z and weighs by the softmax over the selected logits.
// Writes ids [m, k] (int64) and sc [m, k] (float32); the order within a row is the selection order (largest first).
int hostk_route(
    const float* lg, int m, int ld, int E, int k, int sigmoid_bias, const float* bias, int64_t* ids, float* sc) {
    if (k <= 0 || k > 64 || E > 4096) {
        return -1;
    }
    float key[4096], val[4096];
    for (int r = 0; r < m; r++) {
        const float* z = lg + (int64_t)r * ld;
        for (int e = 0; e < E; e++) {
            if (sigmoid_bias) {
                val[e] = 1.0f / (1.0f + expf(-z[e]));
                key[e] = val[e] + bias[e];
            } else {
                val[e] = z[e];
                key[e] = z[e];
            }
        }
        int sel[64];
        for (int j = 0; j < k; j++) {  // k passes of argmax (k = 8 over E = 256: cheaper than a sort)
            int b = -1;
            float bv = -INFINITY;
            for (int e = 0; e < E; e++) {
                if (key[e] > bv) {
                    bv = key[e];
                    b = e;
                }
            }
            sel[j] = b;
            key[b] = -INFINITY;
        }
        float s = 0.0f, mx = -INFINITY;
        if (!sigmoid_bias) {
            for (int j = 0; j < k; j++) {
                mx = fmaxf(mx, val[sel[j]]);
            }
        }
        for (int j = 0; j < k; j++) {
            float w = sigmoid_bias ? val[sel[j]] : expf(val[sel[j]] - mx);
            sc[r * k + j] = w;
            s += w;
        }
        for (int j = 0; j < k; j++) {
            sc[r * k + j] /= s;
            ids[r * k + j] = sel[j];
        }
    }
    return 0;
}

// The fused kernels' crafted logits rows row [m, C + 32]: filler at every slot, 0 in the 32 tail columns except the
// shared-expert gate logit lg[r, E] at column C, ln(score) at each resident pick's slot (slot_of[id] >= 0) and
// ln(missed mass) at the sink C - 1 when the row has misses. Returns the number of resident picks.
int hostk_craft(
    int m,
    int k,
    const int64_t* ids,
    const float* sc,
    const int32_t* slot_of,
    int C,
    float filler,
    const float* lg,
    int ld,
    int E,
    float* row) {
    int hits = 0;
    const int W = C + 32;
    for (int r = 0; r < m; r++) {
        float* o = row + (int64_t)r * W;
        for (int c = 0; c < C; c++) {
            o[c] = filler;
        }
        for (int c = C; c < W; c++) {
            o[c] = 0.0f;
        }
        o[C] = lg[(int64_t)r * ld + E];
        float miss = 0.0f;
        for (int j = 0; j < k; j++) {
            int s = slot_of[ids[r * k + j]];
            if (s >= 0) {
                o[s] = logf(sc[r * k + j]);
                hits++;
            } else {
                miss += sc[r * k + j];
            }
        }
        if (miss > 1e-6f) {
            o[C - 1] = logf(miss);
        }
    }
    return hits;
}

// ---- asynchronous hostk_moe: a C launcher thread runs the call, so the hybrid's host step needs no Python worker
// thread (a Python thread holding the GIL delayed every ctypes / ttnn call of the main thread by up to the 5 ms
// switch interval: pinned writes 18 us -> 5 ms in bench_pinx.py). One call in flight at a time.
typedef struct {
    int R, H, I, gs, E, planar;
    const float* X;
    const int64_t* rows;
    const float* W;
    const int8_t* lo;
    const uint8_t *gu_p, *dn_p;
    const uint16_t *gu_sz, *dn_sz;
    float* out;
} moe_args_t;
static moe_args_t g_a;
static pthread_t g_launch;
static int g_launch_up = 0, g_pending = 0, g_busy = 0, g_rc = 0;
static pthread_mutex_t g_am = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_ac = PTHREAD_COND_INITIALIZER;

static void* launcher(void* arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&g_am);
        while (!g_pending) {
            pthread_cond_wait(&g_ac, &g_am);
        }
        g_pending = 0;
        g_busy = 1;
        moe_args_t a = g_a;
        pthread_mutex_unlock(&g_am);
        int rc = hostk_moe(
            a.R, a.H, a.I, a.gs, a.X, a.E, a.rows, a.W, a.lo, a.planar, a.gu_p, a.gu_sz, a.dn_p, a.dn_sz, a.out);
        pthread_mutex_lock(&g_am);
        g_rc = rc;
        g_busy = 0;
        pthread_cond_broadcast(&g_ac);
        pthread_mutex_unlock(&g_am);
    }
    return NULL;
}

// Same arguments as hostk_moe; returns at once (0), or -5 when a call is still in flight, -6 if the thread failed.
int hostk_moe_async(
    int R,
    int H,
    int I,
    int gs,
    const float* X,
    int E,
    const int64_t* rows,
    const float* W,
    const int8_t* lo,
    int planar,
    const uint8_t* gu_p,
    const uint16_t* gu_sz,
    const uint8_t* dn_p,
    const uint16_t* dn_sz,
    float* out) {
    pthread_mutex_lock(&g_am);
    if (!g_launch_up) {
        if (pthread_create(&g_launch, NULL, launcher, NULL) != 0) {
            pthread_mutex_unlock(&g_am);
            return -6;
        }
        g_launch_up = 1;
    }
    if (g_pending || g_busy) {
        pthread_mutex_unlock(&g_am);
        return -5;
    }
    g_a = (moe_args_t){R, H, I, gs, E, planar, X, rows, W, lo, gu_p, dn_p, gu_sz, dn_sz, out};
    g_pending = 1;
    pthread_cond_broadcast(&g_ac);
    pthread_mutex_unlock(&g_am);
    return 0;
}

// Blocks until the asynchronous call finished; returns its hostk_moe code.
int hostk_moe_wait(void) {
    pthread_mutex_lock(&g_am);
    while (g_pending || g_busy) {
        pthread_cond_wait(&g_ac, &g_am);
    }
    int rc = g_rc;
    pthread_mutex_unlock(&g_am);
    return rc;
}

// ---- int4 library -> device bfp4_b tile images (prefill streaming of experts with no device-format copy) ----
// One 16-value bfp4_b face row: the weights of rows n0 .. n0+15 at column k of a library matrix [N, K] (torch int4
// pack or bit planes, see the top of this file), dequantized w = (q - 8) s + z (fp32, unfused, as torch), rounded to
// `sig` significant bits (build_dev_experts.round_sig), then tt-metal's bfp packing (blockfloat_common.cpp,
// convert_u32_to_bfp<Bfp4_b>: shared exponent = the row's max, mantissa shifted by the difference, rounded to nearest
// even to 3 bits, saturated, sign kept only on a nonzero mantissa). Writes the shared exponent and 8 bytes of nibbles
// (value j in bits 4 (j % 2) of byte j / 2).
__attribute__((optimize("fp-contract=off"))) static void bfp4_face_row(
    const uint8_t* w,
    const uint16_t* sz,
    int N,
    int K,
    int gs,
    int planar,
    int sig,
    int k,
    int n0,
    uint8_t* exp_out,
    uint8_t* mant_out) {
    const uint8_t* blk = w + (size_t)(n0 / 64) * K * 32;
    const int r0 = n0 % 64;
    __m512i q;
    if (!planar) {
        __m512i v = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(blk + (size_t)k * 32 + (r0 % 32))));
        q = r0 < 32 ? _mm512_and_si512(v, _mm512_set1_epi32(15)) : _mm512_srli_epi32(v, 4);
    } else {
        const int m = r0 / 16;
        __m512i vh = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(blk + (size_t)k * 16)));
        __m512i vl = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(blk + (size_t)K * 16 + (size_t)k * 16)));
        __m512i h = _mm512_and_si512(_mm512_srli_epi32(vh, 2 * m), _mm512_set1_epi32(3));
        __m512i l = _mm512_and_si512(_mm512_srli_epi32(vl, 2 * m), _mm512_set1_epi32(3));
        q = _mm512_or_si512(_mm512_slli_epi32(h, 2), l);
    }
    __m512i u = _mm512_loadu_si512((const void*)(sz + ((size_t)(k / gs) * N + n0) * 2));
    __m512 s = _mm512_castsi512_ps(_mm512_slli_epi32(u, 16));
    __m512 z = _mm512_castsi512_ps(_mm512_and_si512(u, _mm512_set1_epi32((int)0xFFFF0000u)));
    __m512 qf = _mm512_cvtepi32_ps(_mm512_sub_epi32(q, _mm512_set1_epi32(8)));
    __m512 wv = _mm512_add_ps(_mm512_mul_ps(qf, s), z);
    __m512i b = _mm512_castps_si512(wv);
    if (sig < 24) {
        const int drop = 24 - sig;
        b = _mm512_and_si512(_mm512_add_epi32(b, _mm512_set1_epi32(1 << (drop - 1))), _mm512_set1_epi32(~((1 << drop) - 1)));
    }
    __m512i e = _mm512_and_si512(_mm512_srli_epi32(b, 23), _mm512_set1_epi32(0xff));
    const uint32_t shared = (uint32_t)_mm512_reduce_max_epu32(e);
    __m512i mant = _mm512_or_si512(_mm512_and_si512(b, _mm512_set1_epi32(0x7fffff)), _mm512_set1_epi32(0x800000));
    mant = _mm512_srlv_epi32(mant, _mm512_sub_epi32(_mm512_set1_epi32((int)shared), e));  // counts >= 32 give 0
    __m512i rv = _mm512_and_si512(mant, _mm512_set1_epi32(0x1fffff));
    __m512i m3 = _mm512_srli_epi32(mant, 21);
    __mmask16 up = _mm512_cmpgt_epu32_mask(rv, _mm512_set1_epi32(0x100000)) |
                   (_mm512_cmpeq_epu32_mask(rv, _mm512_set1_epi32(0x100000)) &
                    _mm512_test_epi32_mask(m3, _mm512_set1_epi32(1)));
    m3 = _mm512_mask_add_epi32(m3, up, m3, _mm512_set1_epi32(1));
    m3 = _mm512_min_epu32(m3, _mm512_set1_epi32(7));
    __mmask16 zero = _mm512_cmpeq_epi32_mask(e, _mm512_setzero_si512());
    m3 = _mm512_mask_mov_epi32(m3, zero, _mm512_setzero_si512());
    __mmask16 nz = _mm512_test_epi32_mask(m3, m3);
    __m512i sgn = _mm512_maskz_and_epi32(nz, _mm512_srli_epi32(b, 28), _mm512_set1_epi32(8));
    __m128i nib = _mm512_cvtepi32_epi8(_mm512_or_si512(sgn, m3));  // 16 bytes, one nibble each
    // pairs (lo, hi) -> lo + 16 hi
    __m128i pr = _mm_maddubs_epi16(nib, _mm_set1_epi16(0x1001));
    __m128i packed = _mm_packus_epi16(pr, pr);
    _mm_storel_epi64((__m128i*)mant_out, packed);
    *exp_out = (uint8_t)shared;
}

// Four face rows at once: library rows n0 .. n0 + 63 (n0 % 64 == 0, one library block) of column k, as bfp4_face_row
// would write them for n0, n0 + 16, n0 + 32, n0 + 48 (same bytes): one load of the packed weights, the four shared
// exponents reduced together (6 lane shuffles instead of 4 full reductions).
__attribute__((optimize("fp-contract=off"))) static void bfp4_face_rows4(
    const uint8_t* w,
    const uint16_t* sz,
    int N,
    int K,
    int gs,
    int planar,
    int sig,
    int k,
    int n0,
    uint8_t* const exp_out[4],
    uint8_t* const mant_out[4]) {
    const uint8_t* blk = w + (size_t)(n0 / 64) * K * 32;
    __m512i q[4];
    if (!planar) {
        __m512i v0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(blk + (size_t)k * 32)));
        __m512i v1 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(blk + (size_t)k * 32 + 16)));
        q[0] = _mm512_and_si512(v0, _mm512_set1_epi32(15));
        q[1] = _mm512_and_si512(v1, _mm512_set1_epi32(15));
        q[2] = _mm512_srli_epi32(v0, 4);
        q[3] = _mm512_srli_epi32(v1, 4);
    } else {
        __m512i vh = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(blk + (size_t)k * 16)));
        __m512i vl = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(blk + (size_t)K * 16 + (size_t)k * 16)));
        for (int m = 0; m < 4; m++) {
            __m512i h = _mm512_and_si512(_mm512_srli_epi32(vh, 2 * m), _mm512_set1_epi32(3));
            __m512i l = _mm512_and_si512(_mm512_srli_epi32(vl, 2 * m), _mm512_set1_epi32(3));
            q[m] = _mm512_or_si512(_mm512_slli_epi32(h, 2), l);
        }
    }
    const uint16_t* zp = sz + ((size_t)(k / gs) * N + n0) * 2;
    __m512i b[4], e[4];
    for (int m = 0; m < 4; m++) {
        __m512i u = _mm512_loadu_si512((const void*)(zp + 32 * m));
        __m512 s = _mm512_castsi512_ps(_mm512_slli_epi32(u, 16));
        __m512 z = _mm512_castsi512_ps(_mm512_and_si512(u, _mm512_set1_epi32((int)0xFFFF0000u)));
        __m512 qf = _mm512_cvtepi32_ps(_mm512_sub_epi32(q[m], _mm512_set1_epi32(8)));
        __m512 wv = _mm512_add_ps(_mm512_mul_ps(qf, s), z);
        b[m] = _mm512_castps_si512(wv);
        if (sig < 24) {
            const int drop = 24 - sig;
            b[m] = _mm512_and_si512(
                _mm512_add_epi32(b[m], _mm512_set1_epi32(1 << (drop - 1))), _mm512_set1_epi32(~((1 << drop) - 1)));
        }
        e[m] = _mm512_and_si512(_mm512_srli_epi32(b[m], 23), _mm512_set1_epi32(0xff));
    }
    __m512i A = _mm512_max_epu32(_mm512_shuffle_i32x4(e[0], e[1], 0x44), _mm512_shuffle_i32x4(e[0], e[1], 0xEE));
    __m512i B = _mm512_max_epu32(_mm512_shuffle_i32x4(e[2], e[3], 0x44), _mm512_shuffle_i32x4(e[2], e[3], 0xEE));
    __m512i C = _mm512_max_epu32(_mm512_shuffle_i32x4(A, B, 0x88), _mm512_shuffle_i32x4(A, B, 0xDD));
    C = _mm512_max_epu32(C, _mm512_shuffle_epi32(C, (_MM_PERM_ENUM)0x4E));
    C = _mm512_max_epu32(C, _mm512_shuffle_epi32(C, (_MM_PERM_ENUM)0xB1));  // 128-bit lane m: shared exponent of row m
    uint32_t sh[16];
    _mm512_storeu_si512((void*)sh, C);
    for (int m = 0; m < 4; m++) {
        __m512i S;
        switch (m) {
            case 0: S = _mm512_shuffle_i32x4(C, C, 0x00); break;
            case 1: S = _mm512_shuffle_i32x4(C, C, 0x55); break;
            case 2: S = _mm512_shuffle_i32x4(C, C, 0xAA); break;
            default: S = _mm512_shuffle_i32x4(C, C, 0xFF); break;
        }
        __m512i mant = _mm512_or_si512(_mm512_and_si512(b[m], _mm512_set1_epi32(0x7fffff)), _mm512_set1_epi32(0x800000));
        mant = _mm512_srlv_epi32(mant, _mm512_sub_epi32(S, e[m]));  // counts >= 32 give 0
        __m512i rv = _mm512_and_si512(mant, _mm512_set1_epi32(0x1fffff));
        __m512i m3 = _mm512_srli_epi32(mant, 21);
        __mmask16 up = _mm512_cmpgt_epu32_mask(rv, _mm512_set1_epi32(0x100000)) |
                       (_mm512_cmpeq_epu32_mask(rv, _mm512_set1_epi32(0x100000)) &
                        _mm512_test_epi32_mask(m3, _mm512_set1_epi32(1)));
        m3 = _mm512_mask_add_epi32(m3, up, m3, _mm512_set1_epi32(1));
        m3 = _mm512_min_epu32(m3, _mm512_set1_epi32(7));
        __mmask16 zero = _mm512_cmpeq_epi32_mask(e[m], _mm512_setzero_si512());
        m3 = _mm512_mask_mov_epi32(m3, zero, _mm512_setzero_si512());
        __mmask16 nz = _mm512_test_epi32_mask(m3, m3);
        __m512i sgn = _mm512_maskz_and_epi32(nz, _mm512_srli_epi32(b[m], 28), _mm512_set1_epi32(8));
        __m128i nib = _mm512_cvtepi32_epi8(_mm512_or_si512(sgn, m3));
        __m128i pr = _mm_maddubs_epi16(nib, _mm_set1_epi16(0x1001));
        __m128i packed = _mm_packus_epi16(pr, pr);
        _mm_storel_epi64((__m128i*)mant_out[m], packed);
        *exp_out[m] = (uint8_t)sh[4 * m];
    }
}

// One matrix [R, C] (row i = column k(i) of the library matrix, its 16-wide column blocks c0 -> library rows n0(c0))
// as bfp4_b tiles in row-major tile order (576 bytes each: 4 x 16 exponents, then the 4 faces' nibbles).
static void bfp4_matrix(
    const uint8_t* w,
    const uint16_t* sz,
    int N,
    int K,
    int gs,
    int planar,
    int sig,
    int R,
    int C,
    int k0,
    const int* n0_of_cb,
    uint8_t* out) {
    // Column blocks outer in runs that share one 64-row library block (4 consecutive 16-row blocks), output rows inner:
    // the library block is then read sequentially (k-major inside it) instead of one 16-byte piece of each of up to 64
    // blocks per output row, which left the kernel waiting on DRAM (~10 ms per expert per thread before).
    const int ct_n = C / 32, ncb = C / 16;
    for (int cb0 = 0; cb0 < ncb;) {
        int run = 1;
        while (cb0 + run < ncb && run < 4 && n0_of_cb[cb0 + run] == n0_of_cb[cb0] + 16 * run &&
               n0_of_cb[cb0 + run] / 64 == n0_of_cb[cb0] / 64) {
            run++;
        }
        const int four = run == 4 && n0_of_cb[cb0] % 64 == 0;
        for (int i = 0; i < R; i++) {
            const int rt = i / 32, ti = i % 32, fr = ti / 16, ri = ti % 16;
            if (four) {
                uint8_t* eo[4];
                uint8_t* mo[4];
                for (int j = 0; j < 4; j++) {
                    const int cb = cb0 + j, ct = cb / 2, fc = cb % 2, f = fr * 2 + fc;
                    uint8_t* tile = out + ((size_t)rt * ct_n + ct) * 576;
                    eo[j] = tile + f * 16 + ri;
                    mo[j] = tile + 64 + f * 128 + ri * 8;
                }
                bfp4_face_rows4(w, sz, N, K, gs, planar, sig, k0 + i, n0_of_cb[cb0], eo, mo);
                continue;
            }
            for (int cb = cb0; cb < cb0 + run; cb++) {
                const int ct = cb / 2, fc = cb % 2, f = fr * 2 + fc;
                uint8_t* tile = out + ((size_t)rt * ct_n + ct) * 576;
                bfp4_face_row(
                    w, sz, N, K, gs, planar, sig, k0 + i, n0_of_cb[cb], tile + f * 16 + ri, tile + 64 + f * 128 + ri * 8);
            }
        }
        cb0 += run;
    }
}

// Expert `row` of a layer's int4 library -> its device-format copy, the layout of build_dev_experts.py: per die d,
// gate|up as [H, 2 Nl] = [up_d | gate_d] and down as [Nl, H] (rows d Nl ..), each bfp4_b tiles; out = [gu die 0 .. tp-1
// | dn die 0 .. tp-1]. Returns the bytes written, or -1 on bad shapes.
int64_t hostk_to_bfp4(
    int H,
    int I,
    int gs,
    int tp,
    int planar,
    int sig,
    const uint8_t* gu_p,
    const uint16_t* gu_sz,
    const uint8_t* dn_p,
    const uint16_t* dn_sz,
    int64_t row,
    uint8_t* out) {
    if (H % 64 || I % (64 * tp) || 64 % gs || tp < 1 || tp > 8) {
        return -1;
    }
    const int Nl = I / tp;
    const size_t gu_bytes = (size_t)2 * I * H / 2, dn_bytes = (size_t)H * I / 2;
    const size_t gu_sz_n = (size_t)(H / gs) * 2 * I * 2, dn_sz_n = (size_t)(I / gs) * H * 2;
    const uint8_t* wg = gu_p + row * gu_bytes;
    const uint16_t* sg = gu_sz + row * gu_sz_n;
    const uint8_t* wd = dn_p + row * dn_bytes;
    const uint16_t* sd = dn_sz + row * dn_sz_n;
    const size_t ng = (size_t)(H / 32) * (2 * Nl / 32) * 576, nd = (size_t)(Nl / 32) * (H / 32) * 576;
    int n0[8192 / 16];
    if (2 * Nl / 16 > 512 || H / 16 > 512) {
        return -1;
    }
    for (int d = 0; d < tp; d++) {
        for (int cb = 0; cb < 2 * Nl / 16; cb++) {  // [up_d | gate_d]: library rows I + d Nl .. (up), d Nl .. (gate)
            const int c0 = cb * 16;
            n0[cb] = c0 < Nl ? I + d * Nl + c0 : d * Nl + (c0 - Nl);
        }
        bfp4_matrix(wg, sg, 2 * I, H, gs, planar, sig, H, 2 * Nl, 0, n0, out + d * ng);
    }
    for (int cb = 0; cb < H / 16; cb++) {
        n0[cb] = cb * 16;
    }
    for (int d = 0; d < tp; d++) {
        bfp4_matrix(wd, sd, H, I, gs, planar, sig, Nl, H, d * Nl, n0, out + tp * ng + d * nd);
    }
    return (int64_t)(tp * (ng + nd));
}
