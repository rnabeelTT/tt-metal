# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device time of a callable's programs (die 0), from the device profiler (TT_METAL_DEVICE_PROFILER=1,
TT_METAL_PROFILER_MID_RUN_DUMP=1, TT_METAL_PROFILER_CPP_POST_PROCESS=1); None without it."""

import os

import ttnn


def devtime(mesh, f, reps=5):
    """median over reps of the summed device FW duration (us) of the programs f() launches."""
    if os.environ.get("TT_METAL_DEVICE_PROFILER") != "1":
        return None
    import ttnn._ttnn.profiler as P

    def read():
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        data = P.get_latest_programs_perf_data()
        if not data:
            return 0.0
        tot = 0.0
        for prog in data[sorted(data)[0]]:
            res = prog.program_analyses_results
            k = "DEVICE FW DURATION [ns]" if "DEVICE FW DURATION [ns]" in res else next(iter(res))
            tot += res[k].duration / 1e3
        return tot

    f()
    read()
    ts = []
    for _ in range(reps):
        f()
        ts.append(read())
    return sorted(ts)[len(ts) // 2]
