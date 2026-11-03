# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Segmented traces: a captured trace is cut into segments with a host callback between them.

Wraps ttnn.begin_trace_capture / end_trace_capture / execute_trace / release_trace. While a capture is open, code
inside it may call ``SEG.split(fn)``: the current real trace is ended, a new one begins, and ``fn`` is registered to
run between the two at replay. ``execute_trace`` on the id returned by ``begin_trace_capture`` replays the segments in
order, calling each callback after its segment (callbacks may read back device tensors, which blocks until the
segment finished, and write into buffers allocated BEFORE the capture; they must not allocate device memory).
"""

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

    @property
    def capturing(self):
        return self.active is not None

    def begin(self, mesh, cq_id=0):
        real = _orig["begin"](mesh, cq_id=cq_id)
        self.active = {"mesh": mesh, "cq": cq_id, "segs": [], "cur": real, "vid": real}
        return real

    def split(self, fn):
        a = self.active
        if a is None:
            raise RuntimeError("SEG.split outside a trace capture")
        _orig["end"](a["mesh"], a["cur"], cq_id=a["cq"])
        a["segs"].append((a["cur"], fn))
        a["cur"] = _orig["begin"](a["mesh"], cq_id=a["cq"])

    def end(self, mesh, tid, cq_id=0):
        a = self.active
        if a is None or _key(tid) != _key(a["vid"]):
            return _orig["end"](mesh, tid, cq_id=cq_id)
        _orig["end"](mesh, a["cur"], cq_id=cq_id)
        a["segs"].append((a["cur"], None))
        if len(a["segs"]) > 1:
            self.traces[_key(a["vid"])] = a["segs"]
        self.active = None

    def execute(self, mesh, tid, cq_id=0, blocking=False):
        segs = self.traces.get(_key(tid))
        if not segs:
            return _orig["exec"](mesh, tid, cq_id=cq_id, blocking=blocking)
        for real, fn in segs:
            _orig["exec"](mesh, real, cq_id=cq_id, blocking=False)
            if fn is not None:
                fn()
        self.stats["replays"] += 1
        self.stats["segments"] += len(segs)
        if blocking:
            ttnn.synchronize_device(mesh)

    def release(self, mesh, tid):
        segs = self.traces.pop(_key(tid), None)
        if not segs:
            return _orig["release"](mesh, tid)
        for real, _ in segs:
            _orig["release"](mesh, real)


SEG = SegTrace()


def install():
    ttnn.begin_trace_capture = lambda mesh, cq_id=0: SEG.begin(mesh, cq_id=cq_id)
    ttnn.end_trace_capture = lambda mesh, tid, cq_id=0: SEG.end(mesh, tid, cq_id=cq_id)
    ttnn.execute_trace = lambda mesh, tid, cq_id=0, blocking=True: SEG.execute(mesh, tid, cq_id=cq_id,
                                                                               blocking=blocking)
    ttnn.release_trace = lambda mesh, tid: SEG.release(mesh, tid)
