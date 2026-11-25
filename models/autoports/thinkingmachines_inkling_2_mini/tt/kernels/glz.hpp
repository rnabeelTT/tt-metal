// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// point markers for the device profiler (a zero-length zone: its start is the event's time; nothing without it)
#pragma once
#include "tools/profiler/kernel_profiler.hpp"
#define GLZ(n)                 \
    {                          \
        DeviceZoneScopedN(n); \
    }
