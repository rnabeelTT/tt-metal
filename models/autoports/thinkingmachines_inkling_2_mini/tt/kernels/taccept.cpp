// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Speculative acceptance on device (tt/sampling.DeviceAccept), at the end of the verify trace: the verify pass's top-K
// candidates (DeviceTopK.device: one group of R <= 8 rows, NT tiles per row as tsample.cpp; merged over the dies the
// same way), the pass's tokens [anchor, drafts] and the drafter's per-step candidates (cand_id / cand_lg: q) ->
// the emitted tokens and the drafts kept, written into the next drafter's inputs (in_anchor = last emitted token,
// in_n = kept + 1), the next verify's accept row (acc[0] = kept) and a small result row the host reads (no host turn
// between the verify and the next drafter).
// Rule (HostSampler.spec at temperature T, no penalties / top-k / top-p: tt/sampling.device_ok): draft j kept with
// probability min(1, p_j / q_j); first rejection: a draw from max(p - q, 0) renormalised (p when that is 0); all
// kept: a draw from row J. Greedy (mode 0): kept iff the draft is row j's argmax (first max in merge order).
// Integer arithmetic only (the RISC-V has no FPU): logits -> Q16, weights w = 2^-(d * a) in Q24 (a = log2 e / T in
// Q24, 2^-f from two 256-entry Q30 tables), probabilities Q31 via one reciprocal per row, draws by inverse CDF on
// host-written Q31 uniforms. tt/sampling.accept_ref is the same arithmetic in Python (bitwise).
//
// params (uint32 [1, 64]): [0] mode (0 greedy, 1 sampled, 2 off), [1] J (live drafts), [2] a_p, [3] a_q,
// [8 + j] accept uniform of draft j, [16 + j] sample uniform of row j.
// out (uint32 [1, 32]): [0 .. 7] emitted tokens (0xFFFFFFFF pad), [8] kept, [16 .. 16 + R) the pass's tokens.
//
// runtime args: idx, val, tok, qid, qlg, prm, tab, out, acc, anchor, n

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

inline int32_t fx16(uint32_t u) {  // fp32 bits -> Q16 (truncated toward 0, |v| clamped to 2^30 / 2^16)
    const uint32_t e = (u >> 23) & 255u;
    if (e == 0) {
        return 0;
    }
    const uint32_t m = (u & 0x7FFFFFu) | 0x800000u;
    const int32_t sh = static_cast<int32_t>(e) - 134;
    uint32_t mag;
    if (sh > 6) {
        mag = 1u << 30;
    } else if (sh >= 0) {
        mag = m << sh;
    } else if (sh > -32) {
        mag = m >> (-sh);
    } else {
        mag = 0;
    }
    return (u >> 31) ? -static_cast<int32_t>(mag) : static_cast<int32_t>(mag);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t R = get_compile_time_arg_val(1);
    constexpr uint32_t NT = get_compile_time_arg_val(2);
    constexpr uint32_t VP = get_compile_time_arg_val(3);
    constexpr uint32_t V = get_compile_time_arg_val(4);
    constexpr uint32_t IB = get_compile_time_arg_val(5);
    constexpr uint32_t DM = get_compile_time_arg_val(6);  // the drafter's d_max (rows of cand_id / cand_lg)
    constexpr uint32_t K = 32;
    constexpr auto i_args = TensorAccessorArgs<7>();
    constexpr auto v_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto qi_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();
    constexpr auto ql_args = TensorAccessorArgs<qi_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<ql_args.next_compile_time_args_offset()>();
    constexpr auto tb_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<tb_args.next_compile_time_args_offset()>();
    constexpr auto a_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto an_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto n_args = TensorAccessorArgs<an_args.next_compile_time_args_offset()>();

    // L1 layout (64 B aligned pieces)
    const uint32_t li = get_write_ptr(cb);
    const uint32_t lv = li + NT * 1024 * IB;
    const uint32_t lt = lv + NT * 2048;
    const uint32_t lp = lt + 64;
    const uint32_t ltb = lp + 256;
    const uint32_t lqi = ltb + 2048;
    const uint32_t lql = lqi + DM * 128;
    const uint32_t lo = lql + DM * 128;
    const uint32_t la = lo + 128;
    const uint32_t lan = la + 128;
    const uint32_t ln = lan + 64;
    const uint32_t lc = ln + 64;

    const auto s_p = TensorAccessor(p_args, get_arg_val<uint32_t>(5), 256);
    noc_async_read(s_p.get_noc_addr(0), lp, 256);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* prm = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lp);
    const uint32_t mode = prm[0];
    if (mode == 2) {
        return;
    }
    const uint32_t J = prm[1] < R - 1 ? prm[1] : R - 1;
    const uint32_t a_p = prm[2], a_q = prm[3];

    const auto s_i = TensorAccessor(i_args, get_arg_val<uint32_t>(0), 1024 * IB);
    const auto s_v = TensorAccessor(v_args, get_arg_val<uint32_t>(1), 2048);
    const auto s_t = TensorAccessor(t_args, get_arg_val<uint32_t>(2), 4 * R);
    const auto s_qi = TensorAccessor(qi_args, get_arg_val<uint32_t>(3), 128);
    const auto s_ql = TensorAccessor(ql_args, get_arg_val<uint32_t>(4), 128);
    const auto s_tb = TensorAccessor(tb_args, get_arg_val<uint32_t>(6), 2048);
    const auto s_o = TensorAccessor(o_args, get_arg_val<uint32_t>(7), 128);
    const auto s_a = TensorAccessor(a_args, get_arg_val<uint32_t>(8), 128);
    const auto s_an = TensorAccessor(an_args, get_arg_val<uint32_t>(9), 4);
    const auto s_n = TensorAccessor(n_args, get_arg_val<uint32_t>(10), 4);
    for (uint32_t t = 0; t < NT; ++t) {
        noc_async_read(s_i.get_noc_addr(t), li + t * 1024 * IB, 1024 * IB);
        noc_async_read(s_v.get_noc_addr(t), lv + t * 2048, 2048);
    }
    noc_async_read(s_t.get_noc_addr(0), lt, 4 * R);
    if (mode == 1) {
        noc_async_read(s_tb.get_noc_addr(0), ltb, 2048);
        for (uint32_t j = 0; j < J; ++j) {
            noc_async_read(s_qi.get_noc_addr(j), lqi + j * 128, 128);
            noc_async_read(s_ql.get_noc_addr(j), lql + j * 128, 128);
        }
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(li);
    volatile tt_l1_ptr uint16_t* ids16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(li);
    volatile tt_l1_ptr uint16_t* vals = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(lv);
    volatile tt_l1_ptr uint32_t* tok = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lt);
    volatile tt_l1_ptr uint32_t* tab = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ltb);
    volatile tt_l1_ptr uint32_t* qid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lqi);
    volatile tt_l1_ptr uint32_t* qlg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lql);
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lo);
    volatile tt_l1_ptr uint32_t* acc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(la);
    uint32_t* cid = reinterpret_cast<uint32_t*>(lc);
    uint32_t* ckey = reinterpret_cast<uint32_t*>(lc + NT * K * 4);
    uint32_t* oid = reinterpret_cast<uint32_t*>(lc + NT * K * 8);
    int32_t* ofx = reinterpret_cast<int32_t*>(lc + NT * K * 8 + K * 4);
    uint32_t* pw = reinterpret_cast<uint32_t*>(lc + NT * K * 8 + K * 8);    // p, Q31
    uint32_t* rs = reinterpret_cast<uint32_t*>(lc + NT * K * 8 + K * 12);   // residual, Q31
    uint32_t* qw = reinterpret_cast<uint32_t*>(lc + NT * K * 8 + K * 16);   // q, Q31

    auto e2 = [&](uint32_t dlt, uint32_t a) -> uint32_t {  // 2^-(dlt * a), dlt Q16, a Q24 -> Q24
        const uint64_t t = (static_cast<uint64_t>(dlt) * a) >> 24;
        const uint64_t n = t >> 16;
        if (n >= 25) {
            return 0;
        }
        const uint32_t f = static_cast<uint32_t>(t) & 0xFFFFu;
        const uint64_t v = (static_cast<uint64_t>(tab[f >> 8]) * tab[256 + (f & 255u)]) >> 30;
        return static_cast<uint32_t>(v >> (6 + n));
    };
    // row r's merged top K (order: logit descending, ties lowest id; tsample.cpp) -> oid / ofx (Q16)
    auto merge = [&](uint32_t r) {
        for (uint32_t t = 0; t < NT; ++t) {
            for (uint32_t c = 0; c < K; ++c) {
                const uint32_t e = ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
                cid[t * K + c] = (IB == 4 ? ids[t * 1024 + e] : static_cast<uint32_t>(ids16[t * 1024 + e])) + t * VP;
                uint32_t u = static_cast<uint32_t>(vals[t * 2048 / 2 + e]) << 16;
                u = (u << 1) == 0 ? 0 : u;
                ckey[t * K + c] = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
            }
        }
        if constexpr (NT > 1) {
            for (uint32_t t = 0; t < NT; ++t) {
                uint32_t* ks = ckey + t * K;
                uint32_t* is = cid + t * K;
                for (uint32_t c = 1; c < K; ++c) {
                    const uint32_t kc = ks[c], ic = is[c];
                    uint32_t d = c;
                    while (d > 0 && (ks[d - 1] < kc || (ks[d - 1] == kc && is[d - 1] > ic))) {
                        ks[d] = ks[d - 1];
                        is[d] = is[d - 1];
                        --d;
                    }
                    ks[d] = kc;
                    is[d] = ic;
                }
            }
        }
        uint32_t head[NT];
        for (uint32_t t = 0; t < NT; ++t) {
            head[t] = t * K;
        }
        for (uint32_t j = 0; j < K; ++j) {
            uint32_t b = 0xFFFFFFFFu;
            for (uint32_t t = 0; t < NT; ++t) {
                const uint32_t q = head[t];
                if (q == (t + 1) * K) {
                    continue;
                }
                if (b == 0xFFFFFFFFu || ckey[q] > ckey[b] || (ckey[q] == ckey[b] && cid[q] < cid[b])) {
                    b = q;
                }
            }
            ++head[b / K];
            oid[j] = cid[b];
            ofx[j] = fx16((ckey[b] & 0x80000000u) ? (ckey[b] & 0x7FFFFFFFu) : ~ckey[b]);
        }
    };
    // first max over the valid candidates (greedy token; the shift for the weights)
    auto best = [&]() -> uint32_t {
        uint32_t b = 0xFFFFFFFFu;
        for (uint32_t j = 0; j < K; ++j) {
            if (oid[j] < V && (b == 0xFFFFFFFFu || ofx[j] > ofx[b])) {
                b = j;
            }
        }
        return b;
    };
    // p (Q31) of the merged row at a_p
    auto probs = [&](uint32_t b) {
        const int32_t m = ofx[b];
        uint64_t Z = 0;
        for (uint32_t j = 0; j < K; ++j) {
            pw[j] = oid[j] < V ? e2(static_cast<uint32_t>(m - ofx[j]), a_p) : 0;
            Z += pw[j];
        }
        const uint64_t rz = (1ull << 62) / Z;
        for (uint32_t j = 0; j < K; ++j) {
            pw[j] = static_cast<uint32_t>((pw[j] * rz) >> 31);
        }
    };
    // inverse-CDF draw over w (Q31 weights, total S > 0) with uniform u (Q31)
    auto draw = [&](const uint32_t* w, uint64_t S, uint32_t u) -> uint32_t {
        const uint64_t tg = (static_cast<uint64_t>(u) * S) >> 31;
        uint64_t c = 0;
        uint32_t last = 0;
        for (uint32_t j = 0; j < K; ++j) {
            if (w[j] == 0) {
                continue;
            }
            c += w[j];
            last = j;
            if (c > tg) {
                return j;
            }
        }
        return last;
    };

    uint32_t k = 0, em[8];
    for (uint32_t j = 0;; ++j) {
        merge(j);
        const uint32_t b = best();
        const bool last = j == J;
        if (mode == 0) {
            const uint32_t g = b == 0xFFFFFFFFu ? 0 : oid[b];
            if (!last && g == tok[j + 1]) {
                em[j] = g;
                continue;
            }
            em[j] = g;
            k = j;
            break;
        }
        probs(b);
        if (last) {
            uint64_t P = 0;
            for (uint32_t i = 0; i < K; ++i) {
                P += pw[i];
            }
            em[j] = oid[draw(pw, P, prm[16 + j])];
            k = j;
            break;
        }
        const uint32_t dj = tok[j + 1];
        // q of draft j over the drafter's candidates (cand_lg: softcapped fp32 logits)
        int32_t mq = fx16(qlg[j * 32]);
        int32_t qf[K];
        for (uint32_t i = 0; i < K; ++i) {
            qf[i] = fx16(qlg[j * 32 + i]);
            mq = qf[i] > mq ? qf[i] : mq;
        }
        uint64_t Zq = 0;
        for (uint32_t i = 0; i < K; ++i) {
            qw[i] = e2(static_cast<uint32_t>(mq - qf[i]), a_q);
            Zq += qw[i];
        }
        const uint64_t rq = (1ull << 62) / Zq;
        uint64_t qd = 0, pd = 0;
        for (uint32_t i = 0; i < K; ++i) {
            qw[i] = static_cast<uint32_t>((qw[i] * rq) >> 31);
            if (qid[j * 32 + i] == dj) {
                qd += qw[i];
            }
            if (oid[i] == dj) {
                pd += pw[i];
            }
        }
        if (pd > 0 && qd > 0 && static_cast<uint64_t>(prm[8 + j]) * qd < (pd << 31)) {
            em[j] = dj;
            continue;
        }
        uint64_t S = 0;
        for (uint32_t i = 0; i < K; ++i) {
            uint64_t pq = 0;
            for (uint32_t c = 0; c < K; ++c) {
                if (qid[j * 32 + c] == oid[i]) {
                    pq += qw[c];
                }
            }
            rs[i] = pw[i] > pq ? static_cast<uint32_t>(pw[i] - pq) : 0;
            S += rs[i];
        }
        if (S > 0) {
            em[j] = oid[draw(rs, S, prm[16 + j])];
        } else {
            uint64_t P = 0;
            for (uint32_t i = 0; i < K; ++i) {
                P += pw[i];
            }
            em[j] = oid[draw(pw, P, prm[16 + j])];
        }
        k = j;
        break;
    }
    for (uint32_t i = 0; i < 32; ++i) {
        out[i] = 0xFFFFFFFFu;
        acc[i] = 0;
    }
    for (uint32_t i = 0; i <= k; ++i) {
        out[i] = em[i];
    }
    out[8] = k;
    for (uint32_t i = 0; i < R; ++i) {
        out[16 + i] = tok[i];
    }
    acc[0] = k;
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lan) = em[k];
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ln) = k + 1;
    noc_async_write(lo, s_o.get_noc_addr(0), 128);
    noc_async_write(la, s_a.get_noc_addr(0), 128);
    noc_async_write(lan, s_an.get_noc_addr(0), 4);
    noc_async_write(ln, s_n.get_noc_addr(0), 4);
    noc_async_write_barrier();
}
