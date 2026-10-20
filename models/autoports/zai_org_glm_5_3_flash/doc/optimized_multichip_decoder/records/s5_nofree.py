"""Run a module with ttnn.deallocate forced to force=False (process-wide), logging the call count at exit.
usage: python s5_nofree.py <module> [args...]"""
import atexit
import runpy
import sys

import ttnn

_orig = ttnn.deallocate
_n = [0]


def _dealloc(t, *a, **kw):
    _n[0] += 1
    kw["force"] = False
    if a:  # positional force
        a = a[1:]
    return _orig(t, *a, **kw)


ttnn.deallocate = _dealloc
atexit.register(lambda: print(f"S5_NOFREE ttnn.deallocate calls (forced force=False): {_n[0]}", flush=True))
mod = sys.argv[1]
sys.argv = [mod] + sys.argv[2:]
runpy.run_module(mod, run_name="__main__", alter_sys=True)
