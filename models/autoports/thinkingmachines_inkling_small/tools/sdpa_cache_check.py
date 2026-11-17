# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""inkling_sdpa's prefill program cache (INKLING_SDPA_PF_CACHE): every prefill case of tests/test_sdpa_device.py run
with the cache off, on (first build) and on again (a cache hit with fresh buffers at other addresses); the results
(PCC vs the reference, max error) must be identical.

usage (in the image): tools/run_dev.sh python models/autoports/thinkingmachines_inkling_small/tools/sdpa_cache_check.py
"""

import os
import sys

import ttnn

from models.autoports.thinkingmachines_inkling_small.tests import test_sdpa_device as T
from models.autoports.thinkingmachines_inkling_small.tt.attn import inkling_sdpa as S


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=24576)
    bad = 0
    try:
        for i, (W, E, p0, n) in enumerate(T.PREFILL):
            pos = list(range(p0, p0 + n))
            res = []
            for on in (False, True, True):
                S._PF_CACHE_ON = on
                res.append(T.run_case(dev, W, E, pos, seed=100 + i, items_r=8))
            same = res[0] == res[1] == res[2]
            bad += not same
            print(
                f"W={W} E={E} p0={p0} rows={n}: off {res[0]} | on {res[1]} | hit {res[2]} -> "
                f"{'SAME' if same else 'DIFF'}",
                flush=True,
            )
        print(f"RESULT {'PASS' if bad == 0 else 'FAIL'}", flush=True)
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
