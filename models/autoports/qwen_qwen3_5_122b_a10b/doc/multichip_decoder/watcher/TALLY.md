# Watcher processes on the (1,4) mesh — teardown form vs watcher dump timing

All 14 watcher-enabled processes (all watcher features on, logger Warning) ran model kernels or CCLs watcher-clean (0 bad
watcher.log lines, no tripped model-kernel assert after the reduce-scatter writer fix). Every process failed at the teardown
of the fabric routers on device 0 eth core (0,9) / virtual (29,25), in one of two forms:
(i) "Timed out while waiting for active ethernet core 29-25 to become active again" (host side, `MetalContext::teardown`),
(ii) the watcher's "subordinate_erisc detected invalid NOC command buffer state before starting the next kernel ...
Current kernel: fabric_erisc_router.cpp" — only when a 10 s watcher dump was in flight during the teardown window.
`MetalContext::teardown` detaches the watcher before the RISC-firmware teardown that waits for the eth core, so form (ii) can
only be observed when a dump starts inside that window; otherwise only form (i) is observable.

| process (`~/qwen35-122b-bringup/artifacts/`) | test / workload | test result | pytest call s | last dump (watcher clock) | form |
|---|---|---|---|---|---|
| 04-watcher-1 | real_weight gdn_37 | PASSED | 13.64 | #2 completed 13.13 s | (i) |
| 04-watcher-2 | real_weight attn_37 | PASSED | 27.16 | #4 STARTED 33.19 s, never completed | (ii) |
| 04-watcher-3 | batched_decode_b32 attn | PASSED | 31.08 | #4 completed 33.22 s | (i) |
| 04-watcher-4 | chunked_prefill gdn (4429) | PASSED | 55.27 | #6 completed 53.28 s | (i) |
| 04-watcher-5 | chunked_prefill attn (4429) | PASSED | ~40 | #5 STARTED 43.24 s, never completed | (ii) |
| 04-watcher-6 | traced_decode gdn | PASSED | 23.28 | #3 completed 23.11 s | (i) |
| 04-watcher-probe2 | real_weight gdn_37 (fixed kernel) | PASSED | 29.61 | #4 completed 33.23 s | (i) |
| 04-watcher-bisect-1 | real_weight attn_37 (repeat) | PASSED | 13.37 | #2 completed 13.08 s | (i) |
| 04-watcher-bisect-2 | real_weight gdn_37 (repeat) | PASSED | 13.61 | #2 completed 13.15 s | (i) |
| 04-watcher-bisect-3 | real_weight attn_37 (repeat) | PASSED | 13.58 | #2 completed 13.26 s | (i) |
| 04-watcher-bisect-4 | traced_decode attn | PASSED | 23.71 | #3 completed 23.21 s | (i) |
| 04-watcher-control | 20 plain all-gathers, no model code | ok | – | #1 completed 1.36 s | (i) |
| 04-watcher-attn-ccl-control | attention CCL pattern, 20 rounds | ok | – | #1 completed 1.36 s | (i) |
| 04-watcher-attn-ccl-control-100 | attention CCL pattern, 100 rounds | ok | – | #1 completed 1.33 s | (i) |

Reading: form (ii) occurred in 2 of the 2 processes whose dump tick fell 0-4 s into teardown (at or after the PASSED report) (the aborted "Dump #4 at
33.188s" / "Dump #5 at 43.244s" lines in `watcher_2.tail.txt` / `watcher_5.tail.txt`), and in 0 of the 12 processes without a
tick there (the repeats ran with warm kernel caches, 13-14 s instead of 27 s, so their teardown fell between ticks; the
controls had a single tick at 1.3 s). The NOC command-buffer state the check reads on the erisc is programmed only by code
running on that erisc (the fabric router, on both erisc and subordinate_erisc per the dumps), never by model worker kernels.
The message is `DebugAssertNCriscNOCPacketTagClearedTripped`, raised by the erisc firmware after `kernel_main()` returns
(`tt_metal/hw/firmware/src/tt-1xx/active_erisck.cc:51`; the dumps show "Last waypoint: SEW,NKFW"), and the router kernel
returns only when the fabric is torn down at mesh close. Both forms are therefore the fabric router not exiting cleanly under eth-core watcher instrumentation at teardown — not
attention-specific and not model-specific. `TT_METAL_WATCHER_DISABLE_ETH=1` (`04-watcher-noeth`): exit 0, clean teardown.
Every non-watcher run (final suite 35/35, perf, sweeps) tears down cleanly. To be reported upstream with these logs.
Files here: `pytest_<i>.txt` (filtered pytest output), `watcher_<i>.tail.txt` (last 120-200 watcher.log lines),
`pytest_probe2.txt`, `pytest_bisect-<j>.txt`, `control.txt`, `attn-ccl-control*.txt`.
