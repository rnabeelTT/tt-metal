# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One process per die for multi-die serving.

Running every die's engine as a thread of the server process serialises them on the GIL: the eager vision path and
the host staging of each pass are Python-heavy, and four dies in one interpreter measured no faster than one. Here
each die is a spawned process that sees only its own chip (``TT_VISIBLE_DEVICES``), opens it as a one-die mesh with
the p150 mesh graph, and runs a ``DieEngine``; the server-side ``ProcEngine`` forwards calls over a pipe. A die
process is exactly the single-p150 configuration, so every profile runs the same per-die code.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import traceback

CALL_TIMEOUT_S = float(os.environ.get("QWEN3VL_RERANKER_CALL_TIMEOUT_S", "900"))
WARM_TIMEOUT_S = float(os.environ.get("QWEN3VL_RERANKER_WARM_TIMEOUT_S", "3600"))


def _p150_mesh_graph() -> str:
    return os.path.join(
        os.environ.get("TT_METAL_HOME", "/opt/tt-metal"),
        "tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto",
    )


def _worker(die: int, conn, traced: bool):
    os.environ["QWEN3VL_RERANKER_IN_DIE_PROC"] = "1"
    os.environ["TT_VISIBLE_DEVICES"] = str(die)
    # several die processes share the host: keep each one's torch / OpenMP pool small
    threads = int(os.environ.get("QWEN3VL_RERANKER_DIE_THREADS", "2"))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    import torch

    torch.set_num_threads(threads)
    os.environ["TT_MESH_GRAPH_DESC_PATH"] = os.environ.get("QWEN3VL_RERANKER_DIE_MESH_GRAPH", _p150_mesh_graph())
    try:
        from models.autoports.qwen_qwen3_vl_reranker_8b.tt.engine import build_engines, close_engines

        engines = build_engines((1, 1), traced=traced)
        eng = engines[0]
        eng.die = die
        conn.send(("ok", None))
    except Exception:
        conn.send(("err", traceback.format_exc()))
        return
    try:
        while True:
            cmd, arg = conn.recv()
            try:
                if cmd == "warm":
                    res = eng.warm()
                    res["die"] = die
                elif cmd == "score":
                    res = eng.score(arg)
                elif cmd == "stats":
                    res = eng.stats()
                elif cmd == "close":
                    close_engines(engines)
                    conn.send(("ok", None))
                    return
                else:
                    raise ValueError(cmd)
                conn.send(("ok", res))
            except Exception:
                conn.send(("err", traceback.format_exc()))
    except (EOFError, KeyboardInterrupt):
        close_engines(engines)


class ProcEngine:
    """``ScoringEngine`` proxy for a die process. Calls are serialised by the server's per-die worker thread."""

    def __init__(self, die: int, traced: bool = True):
        self.die = die
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker, args=(die, child, traced), name=f"reranker-die{die}", daemon=True)
        self.proc.start()
        status, err = self.conn.recv()
        if status != "ok":
            raise RuntimeError(f"die {die} failed to start:\n{err}")
        self._max = None

    def alive(self) -> bool:
        return self.proc.is_alive()

    def _call(self, cmd, arg=None, timeout=None):
        if not self.proc.is_alive():
            raise RuntimeError(f"die {self.die} process is not running (exit code {self.proc.exitcode})")
        timeout = timeout or CALL_TIMEOUT_S
        self.conn.send((cmd, arg))
        if not self.conn.poll(timeout):
            # a die that does not answer is hung: stop it, so /health reports it and dispatch skips it
            self.proc.kill()
            raise RuntimeError(f"die {self.die} did not answer {cmd!r} within {timeout:.0f} s; stopped it")
        try:
            status, res = self.conn.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError(f"die {self.die} process died during {cmd!r}: {exc}") from exc
        if status != "ok":
            raise RuntimeError(f"die {self.die} {cmd} failed:\n{res}")
        return res

    def max_tokens(self) -> int:
        from models.autoports.qwen_qwen3_vl_reranker_8b.tt.engine import MAX_TOKENS

        return MAX_TOKENS

    def warm(self) -> dict:
        return self._call("warm", timeout=WARM_TIMEOUT_S)

    def score(self, pairs):
        return self._call("score", list(pairs))

    def stats(self) -> dict:
        return self._call("stats")

    def close(self) -> None:
        if self.proc.is_alive():
            try:
                self._call("close", timeout=120)
            except Exception:
                pass
            self.proc.join(timeout=60)
            if self.proc.is_alive():
                self.proc.terminate()


def build_proc_engines(n: int, traced: bool = True) -> list[ProcEngine]:
    import concurrent.futures as cf

    with cf.ThreadPoolExecutor(n) as ex:
        return list(ex.map(lambda d: ProcEngine(d, traced), range(n)))
