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

#define MAXR 16
#define MAXE 640
#define MAXT 32
#define SPIN 2000  // ~50 us of pause before futex sleep: a longer spin starves the host step's other threads

typedef struct {
    int R, H, I, gs, E;
    const float* X;              // [R, H]
    const int64_t* rows;         // [E] row of each expert in the library tensors
    const float* W;              // [E, R]
    const uint8_t *gu_p, *dn_p;  // library bases
    const uint16_t *gu_sz, *dn_sz;
    float* out;  // [R, H]
    // per expert: active rows (compacted)
    int ne[MAXE];
    int er[MAXE][MAXR];
    float* xs_gu;  // [E][MAXR][H/gs]  group sums of the active x rows
    float* A;      // [E][MAXR][I]     activations
    float* xs_dn;  // [E][MAXR][I/gs]  group sums of A
    int phase, units;
} job_t;

static job_t J;
static int g_nt = 0;
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
static void gemv_block(
    const uint8_t* w,
    const uint16_t* sz,
    int N,
    int K,
    int gs,
    const float* const* x,
    const float* const* xs,
    int nr,
    float out[][64]) {
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
            const uint8_t* p = w + (size_t)g * gs * 32;
            const __m512i m15 = _mm512_set1_epi32(15);
            for (int k = 0; k < gs; k++, p += 32) {
                __m512i v0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)p));
                __m512i v1 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(p + 16)));
                __m512 f[4];
                f[0] = _mm512_cvtepi32_ps(_mm512_and_si512(v0, m15));
                f[1] = _mm512_cvtepi32_ps(_mm512_and_si512(v1, m15));
                f[2] = _mm512_cvtepi32_ps(_mm512_srli_epi32(v0, 4));
                f[3] = _mm512_cvtepi32_ps(_mm512_srli_epi32(v1, 4));
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
                __m512 zz = _mm512_fnmadd_ps(_mm512_set1_ps(8.0f), s, z);  // z - 8 s
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
        for (int r = 0; r < nr; r++) {
            x[r] = J.X + (size_t)J.er[e][r] * H;
            xs[r] = J.xs_gu + ((size_t)e * MAXR + r) * (H / gs);
        }
        float g[MAXR][64], up[MAXR][64];
        gemv_block(wb + (size_t)j * H * 32, sb + (size_t)j * 64 * 2, 2 * I, H, gs, x, xs, nr, g);
        gemv_block(wb + (size_t)(nb + j) * H * 32, sb + (size_t)(I + j * 64) * 2, 2 * I, H, gs, x, xs, nr, up);
        for (int r = 0; r < nr; r++) {
            float* a = J.A + ((size_t)e * MAXR + r) * I + j * 64;
            for (int t = 0; t < 64; t++) {
                a[t] = g[r][t] / (1.0f + expf(-g[r][t])) * up[r][t];
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
            for (int r = 0; r < nr; r++) {
                x[r] = J.A + ((size_t)e * MAXR + r) * I;
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
                y);
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
    (void)arg;
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
        run_units();
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
        pthread_create(&g_th[t], NULL, worker, NULL);
    }
    return g_nt;
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
    pthread_mutex_lock(&g_mu);
    size_t need = (size_t)E * MAXR * ((size_t)H / gs + I + (size_t)I / gs);
    if (need > g_scratch_sz) {
        free(g_scratch);
        g_scratch = malloc(need * sizeof(float));
        g_scratch_sz = g_scratch ? need : 0;
        if (!g_scratch) {
            pthread_mutex_unlock(&g_mu);
            return -3;
        }
    }
    J.R = R, J.H = H, J.I = I, J.gs = gs, J.E = E, J.X = X, J.rows = rows, J.W = W;
    J.gu_p = gu_p, J.gu_sz = gu_sz, J.dn_p = dn_p, J.dn_sz = dn_sz, J.out = out;
    J.xs_gu = g_scratch;
    J.A = J.xs_gu + (size_t)E * MAXR * (H / gs);
    J.xs_dn = J.A + (size_t)E * MAXR * I;
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
