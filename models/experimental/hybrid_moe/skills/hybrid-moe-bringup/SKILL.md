---
name: hybrid-moe-bringup
description: Bring up, serve, measure and package a large MoE model with the hybrid CPU+TT expert overlay (models/experimental/hybrid_moe) on a QB2. Use when starting a new model from the hybrid-moe-starter branch, building its expert artifacts, serving it under vLLM, judging its quality, or publishing it with tt-model.
---

# Hybrid MoE bring-up

The procedure is `models/experimental/hybrid_moe/ADD_A_MODEL.md`: read it first and follow its numbered steps
(fit check, drafter, port, artifacts, serve, package). `HANDOFF.md` holds the measurements and dead ends behind every
default; search it before trying an idea, since many have already been measured and dropped.

Rules that cost hours when skipped:

1. **Check the device before device work.** A bare `ttnn.open_mesh_device(ttnn.MeshShape(1, 4), ...)` must open and
   close. If it hangs in "Starting devices in cluster", another process holds the boards: find it, and ask the
   machine's owner before stopping anything you did not start.
2. **Quality is judged against the CPU BF16 reference** (`ref_<tag>.py`, teacher forced with `compare_m27.py`: KL and
   top-1), never against an earlier TT run. A speed change that moves those numbers is a bug until shown otherwise.
   Speculative decoding must not change the output: compare SPEC=1 against SPEC=0 text.
3. **One device job at a time.** Wait on the previous job's PID (`kill -0`); kill by PID, never `pkill -f <pattern>`
   (it matches your own shell). Stop any watchdog when serving goes idle.
4. **Two copies of the serve settings**: `serve/serve_<tag>.sh` and the yaml's `serve.env`. Change both.
5. **Before `tt-model push`**: build `hybrid/hostk/libhostk.so` and `hybrid/pin/pinx.so` in the packaging worktree (they
   are gitignored), check `git diff <base> -- tt_metal/third_party` is empty, and serve the built package for every
   profile (`tt-model serve ... --profile <p>`) with `serve/card_bench.py` before publishing.
6. **Host RAM**: compare `MemAvailable` with the `[hybrid] host RAM` line before long-context runs; leaked pinned memory
   returns only on a reboot.

Report each measured number with its command, profile and context length; say what was not measured.
