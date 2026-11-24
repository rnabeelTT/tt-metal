// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Decode attention prep (tt/aprep.py) units per user: q head groups (APREP_QN, APREP_HG heads each), r, k head groups
// (APREP_KN of APREP_KG heads), v head groups, sink. Types: 0 q, 1 r, 2 k, 3 v, 4 sink; heads [h0, h1) of the unit.
// (The per-head math is row-local: a group computes its rows exactly as the whole unit would.)

#pragma once

#include <stdint.h>

#ifndef APREP_QN
#define APREP_QN 1
#define APREP_HG 32
#define APREP_KN 1
#define APREP_KG 32
#endif

constexpr uint32_t APREP_NU = APREP_QN + 2 * APREP_KN + 2;

inline void aprep_unit(uint32_t tu, uint32_t nh, uint32_t nkv, uint32_t& type, uint32_t& h0, uint32_t& h1) {
    if (tu < APREP_QN) {
        type = 0, h0 = tu * APREP_HG, h1 = h0 + APREP_HG < nh ? h0 + APREP_HG : nh;
    } else if (tu == APREP_QN) {
        type = 1, h0 = 0, h1 = nh;
    } else if (tu < APREP_QN + 1 + 2 * APREP_KN) {
        const uint32_t i = tu - APREP_QN - 1;
        type = 2 + i / APREP_KN, h0 = (i % APREP_KN) * APREP_KG, h1 = h0 + APREP_KG < nkv ? h0 + APREP_KG : nkv;
    } else {
        type = 4, h0 = 0, h1 = nh;
    }
}
