# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Closing the mesh after a hybrid run without hanging the process.

With the hybrid engine up (socket-mode boundary, async expert fetch on CQ1, pinned host regions) a bare
close_mesh_device can block forever: queued CQ1 copies and the pinned regions' destructors still talk to the mesh, and
the process sits in futex waits with its results already printed (an eval held the board 1 h 50 m that way). close()
first quiesces in order (pinned-copy flush, device sync, fetch release, sockets dropped), logging each step so a stall
names itself, then closes the mesh; a watchdog thread ends the process with os._exit if the whole sequence exceeds
`timeout` seconds. Call it only once the results are written: the backstop skips the remaining teardown.
"""

import os
import sys
import threading
import time


def _drop_sockets(hyb):
    import gc

    if hyb is not None and getattr(hyb, "socks", None) is not None:
        hyb.socks = None
        gc.collect()


def _stop_prefetch(mesh):
    pf = sys.modules.get("models.autoports.thinkingmachines_inkling_2_mini.tt.prefetch")
    if pf is not None:
        pf.stop(mesh)


def quiesce(mesh):
    import ttnn

    hyb = getattr(sys.modules.get("hybrid.patch"), "HYB", None)
    f = getattr(hyb, "fetch", None) if hyb is not None else None
    for name, fn in (
        ("pinx flush", lambda: f.px.flush() if f is not None and getattr(f, "px", None) is not None else None),
        ("synchronize", lambda: ttnn.synchronize_device(mesh)),
        ("tensor prefetcher stop", lambda: _stop_prefetch(mesh)),
        ("fetch release", lambda: f.release() if f is not None else None),
        ("sockets release", lambda: _drop_sockets(hyb)),
    ):
        t0 = time.time()
        print(f"[exit] {name} ...", flush=True)
        try:
            fn()
        except Exception as e:  # noqa: BLE001  keep closing
            print(f"[exit] {name} failed: {e}", flush=True)
        print(f"[exit] {name} done ({time.time() - t0:.1f} s)", flush=True)


def close(mesh, timeout=None, code=0):
    """Quiesce, close the mesh, disable fabric; os._exit(code) if that takes over `timeout` s (INKLING_EXIT_TIMEOUT,
    default 90)."""
    import ttnn

    timeout = float(timeout if timeout is not None else os.environ.get("INKLING_EXIT_TIMEOUT", "90"))
    done = threading.Event()

    def _backstop():
        if not done.wait(timeout):
            print(f"[exit] teardown exceeded {timeout:.0f} s: forcing process exit", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(code)

    threading.Thread(target=_backstop, daemon=True, name="exit-backstop").start()
    quiesce(mesh)
    t0 = time.time()
    print("[exit] close_mesh_device ...", flush=True)
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    print(f"[exit] closed ({time.time() - t0:.1f} s)", flush=True)
    done.set()
