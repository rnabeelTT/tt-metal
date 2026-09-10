#!/usr/bin/env bash
# Device preflight: make the boards ready before a stage touches them.
#
# Two things this handles, both learned the hard way on this QB2:
#
# 1. FABRIC WARM-UP. The first multi-chip mesh opened with FABRIC_1D after a board reset
#    ALWAYS fails ethernet router sync ("Fabric Router Sync: Timeout ... handshake"); the
#    second and every subsequent open succeeds. It is link training, not a fault. Absorb
#    it here rather than letting it fail a real stage and look like a model bug.
#
# 2. WEDGED BOARDS. Killing a run mid-weight-upload leaves the NOC hung ("NOC0 is hung on
#    PCIe device ID N") and every later stage fails against dead hardware. Detect and
#    reset, so an unattended overnight pipeline recovers instead of burning the night.
#
# usage: device_preflight.sh [mesh ...]     (default: all three targets)
# 3. STALE DEVICE LOCKS. UMD takes a robust mutex per chip in /dev/shm. A hard-killed
#    process leaves it held by a dead PID, and the next run blocks forever on
#    "Waiting for lock 'CHIP_IN_USE_0_PCIe' which is currently held by thread TID: <dead>".
#    Nothing times out; the pipeline just stops.
# Standalone: this ships inside the model, so it must not depend on the bring-up
# workspace. Honour an already-sourced env when there is one, else fall back.
: "${TT_METAL_HOME:=$HOME/tt-metal}"
MESHES=("$@"); [ ${#MESHES[@]} -eq 0 ] && MESHES=(P150 P300 P150x4)

# Only safe when nothing is actually using the devices -- check before removing, or a
# concurrent run loses its mutex.
if pgrep -f "bench_model.py|pytest.*(test_model|test_layer|test_moe)" >/dev/null; then
    echo "  another device job is running; leaving UMD locks alone"
elif ls /dev/shm/TT_UMD_LOCK.CHIP_IN_USE_* >/dev/null 2>&1; then
    echo "  clearing stale UMD chip locks (no device job running)"
    rm -f /dev/shm/TT_UMD_LOCK.CHIP_IN_USE_*
fi

activate_venv || exit 1

python - "${MESHES[@]}" <<'PY'
import sys
import ttnn

SHAPES = {"P150": (1, 1), "P300": (1, 2), "P150x4": (1, 4)}
meshes = [m for m in sys.argv[1:] if m in SHAPES]
need_fabric = any(SHAPES[m][0] * SHAPES[m][1] > 1 for m in meshes)
if need_fabric:
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)

bad = []
for m in meshes:
    rows, cols = SHAPES[m]
    # Two attempts: the first absorbs link training on a cold fabric.
    for attempt in (1, 2):
        try:
            md = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
            n = md.get_num_devices()
            ttnn.close_mesh_device(md)
            print(f"  {m:8s} ({rows},{cols}) ready, {n} device(s)"
                  + ("  [warmed on retry]" if attempt == 2 else ""))
            break
        except Exception as e:
            first = str(e).splitlines()[0][:90]
            if attempt == 1:
                print(f"  {m:8s} cold-open failed, retrying: {first}")
            else:
                print(f"  {m:8s} STILL FAILING: {first}")
                bad.append(m)
if need_fabric:
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
sys.exit(2 if bad else 0)
PY
rc=$?
if [ $rc -ne 0 ]; then
    echo "!! preflight failed — resetting boards and retrying once"
    tt-smi -r >/dev/null 2>&1
    exec "$0" "${MESHES[@]}"   # one reset+retry; a second failure is a real fault
fi
echo "preflight OK"
