# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Segmented traces: a captured trace is cut into segments with a host callback between them.

Wraps ttnn.begin_trace_capture / end_trace_capture / execute_trace / release_trace. While a capture is open, code
inside it may call ``SEG.split(fn)``: the current real trace is ended, a new one begins, and ``fn`` is registered to
run between the two at replay. ``execute_trace`` on the id returned by ``begin_trace_capture`` replays the segments in
order, calling each callback after its segment (callbacks may read back device tensors, which blocks until the
segment finished, and write into buffers allocated BEFORE the capture; they must not allocate device memory).
``SEG.post(fn)`` registers a host handler WITHOUT cutting the trace: at replay it runs after its segment has been
enqueued, before that segment's split callback (the socket-mode host steps: the device blocks on socket receives
inside the segment until the handler has written the data). A capture with post handlers is kept as a segmented trace
even when it has a single segment; ``pre_replay`` (when set) runs before such a replay is enqueued.
"""

import os
import time

import ttnn

_orig = {
    "begin": ttnn.begin_trace_capture,
    "end": ttnn.end_trace_capture,
    "exec": ttnn.execute_trace,
    "release": ttnn.release_trace,
}


def _key(tid):
    try:
        hash(tid)
        return tid
    except TypeError:
        return repr(tid)


class SegTrace:
    def __init__(self):
        self.active = None
        self.traces = {}  # key(first real id) -> [(real id, fn | None)]
        self.stats = {"replays": 0, "segments": 0}
        self.log_every = int(
            os.environ.get("HYB_LOG_EVERY", "0")
        )  # serving diagnostics: host-step stats every N replays
        self.on_log = None
        self.host_st = None
        self.pre_replay = None  # called before a replay of a trace with post handlers (HYB_SOCKET: fetch commit)

    @property
    def capturing(self):
        return self.active is not None

    def begin(self, mesh, cq_id=0):
        real = _orig["begin"](mesh, cq_id=cq_id)
        self.active = {"mesh": mesh, "cq": cq_id, "segs": [], "cur": real, "vid": real, "posts": []}
        return real

    def post(self, fn):
        a = self.active
        if a is None:
            raise RuntimeError("SEG.post outside a trace capture")
        a["posts"].append(fn)

    def split(self, fn):
        a = self.active
        if a is None:
            raise RuntimeError("SEG.split outside a trace capture")
        _orig["end"](a["mesh"], a["cur"], cq_id=a["cq"])
        a["segs"].append((a["cur"], fn, a["posts"]))
        a["posts"] = []
        a["cur"] = _orig["begin"](a["mesh"], cq_id=a["cq"])

    def end(self, mesh, tid, cq_id=0):
        a = self.active
        if a is None or _key(tid) != _key(a["vid"]):
            return _orig["end"](mesh, tid, cq_id=cq_id)
        _orig["end"](mesh, a["cur"], cq_id=cq_id)
        a["segs"].append((a["cur"], None, a["posts"]))
        if len(a["segs"]) > 1 or any(p for _, _, p in a["segs"]):
            self.traces[_key(a["vid"])] = a["segs"]
        self.active = None

    def execute(self, mesh, tid, cq_id=0, blocking=False):
        segs = self.traces.get(_key(tid))
        if not segs:
            return _orig["exec"](mesh, tid, cq_id=cq_id, blocking=blocking)
        if self.stats.get("t_first") is None:
            self.stats["t_first"] = time.time()  # first traced decode step: everything before it is prefill (TTFT)
        st = self.host_st  # optional timing sink (patch.HYB.st): enqueue vs callback wall time
        if self.pre_replay is not None and any(p for _, _, p in segs):
            self.pre_replay()
        for real, fn, posts in segs:
            t0 = time.perf_counter()
            _orig["exec"](mesh, real, cq_id=cq_id, blocking=False)
            t1 = time.perf_counter()
            for p in posts:
                p()
            if fn is not None:
                fn()
            if st is not None:
                st["t_exec"] = st.get("t_exec", 0.0) + t1 - t0
                st["t_cb"] = st.get("t_cb", 0.0) + time.perf_counter() - t1
        self.stats["replays"] += 1
        self.stats["segments"] += len(segs)
        if self.log_every and self.stats["replays"] % self.log_every == 0 and self.on_log is not None:
            self.on_log(self.stats["replays"])
        if blocking:
            ttnn.synchronize_device(mesh)

    def release(self, mesh, tid):
        segs = self.traces.pop(_key(tid), None)
        if not segs:
            return _orig["release"](mesh, tid)
        for real, _, _ in segs:
            _orig["release"](mesh, real)


SEG = SegTrace()


def install():
    ttnn.begin_trace_capture = lambda mesh, cq_id=0: SEG.begin(mesh, cq_id=cq_id)
    ttnn.end_trace_capture = lambda mesh, tid, cq_id=0: SEG.end(mesh, tid, cq_id=cq_id)
    ttnn.execute_trace = lambda mesh, tid, cq_id=0, blocking=True: SEG.execute(
        mesh, tid, cq_id=cq_id, blocking=blocking
    )
    ttnn.release_trace = lambda mesh, tid: SEG.release(mesh, tid)
