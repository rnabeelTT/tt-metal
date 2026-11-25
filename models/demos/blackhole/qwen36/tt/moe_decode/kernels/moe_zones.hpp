// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// MOE_MARK(name): a device-profiler point (an empty zone) when the rows ops are built with QWEN35_MOE_ZONES=1.
#pragma once
#ifdef MOE_ZONES
#include "tools/profiler/kernel_profiler.hpp"
#define MOE_MARK(name) \
    { DeviceZoneScopedN(name); }
#else
#define MOE_MARK(name)
#endif
