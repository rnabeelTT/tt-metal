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

#define MAXR 16
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
    int phase, units;
} job_t;

static job_t J;
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
        const int lay = J.planar ? (J.lo && J.lo[e] ? 2 : 1) : 0;
        float g[MAXR][64], up[MAXR][64];
        gemv_block(wb + (size_t)j * H * 32, sb + (size_t)j * 64 * 2, 2 * I, H, gs, x, xs, nr, lay, g);
        gemv_block(wb + (size_t)(nb + j) * H * 32, sb + (size_t)(I + j * 64) * 2, 2 * I, H, gs, x, xs, nr, lay, up);
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
                J.planar ? (J.lo && J.lo[e] ? 2 : 1) : 0,
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
static float g_z[MAXR * 1024];

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
            __m512 acc[MAXR];
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
    if (m > MAXR || E > 1024 || H % 16 || k > E) {
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
