# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Make MESH_DEVICE actually select the mesh for this suite.

The repo-root `mesh_device` fixture takes its shape from `request.param` and otherwise
falls back to the WHOLE system mesh. On this box that is 4 chips, so a run launched as
`MESH_DEVICE=P150 pytest ...` silently gets a (1,4) mesh, loads the 4-chip MoE config,
and reports results under the wrong board name. Every stage script selects the target
board through MESH_DEVICE, so that fallback would quietly invalidate the per-mesh
evidence this bring-up is built on.

This parametrizes the fixture indirectly from the env var so the requested board is
the board that opens.
"""
import os

import pytest

import ttnn

# Names come from tt_transformers' device-name map for Blackhole (1/2/4 chips).
#
MESH_SHAPES = {"P150": (1, 1), "P300": (1, 2), "P150x4": (1, 4)}


@pytest.fixture(scope="session", autouse=True)
def _warm_fabric():
    """Absorb the first multi-chip fabric open of the process.

    On this QB2 the first mesh opened with FABRIC_1D in a given PROCESS fails ethernet
    router sync ("Fabric Router Sync: Timeout ... handshake"); the second and all later
    opens succeed. It is link training, and it does not carry across processes -- warming
    the fabric from a separate preflight script does nothing for pytest. Doing it here, in
    a session fixture that runs before any mesh_device, turns a confusing stage failure
    into a few wasted seconds at startup.

    Only meshes larger than one chip need it, and a failure here is deliberately ignored:
    it is the expected outcome of the warm-up, and a genuinely broken link will fail the
    real device open immediately after with the same message.
    """
    name = os.environ.get("MESH_DEVICE", "P150")
    rows, cols = MESH_SHAPES.get(name, (1, 1))
    if rows * cols > 1 and os.environ.get("ORNITH_MOE_EP") == "1":
        # Warm with the FULL mesh, not the target shape. Opening a 2-chip mesh trains
        # only that pair's links, and on this box a cold (1,2) open fails even on the
        # retry; opening the full (1,4) once trains every link and a subsequent (1,2)
        # then succeeds first time. Measured: cold (1,4) OK -> (1,2) OK, whereas
        # cold (1,2) -> fail, retry -> OK.
        try:
            full = ttnn.get_num_devices()
        except Exception:
            full = max(r * c for r, c in MESH_SHAPES.values())
        rows, cols = 1, max(full, rows * cols)
        # NOTE: leaving the fabric set process-wide instead (and dropping fabric_config
        # from device_params) was tried and is WORSE -- it breaks the 4-chip mesh. The
        # per-open fabric_config is what P150x4 needs.
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
            md = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
            ttnn.close_mesh_device(md)
        except Exception:
            pass
        finally:
            # Hand the fabric back disabled: the real device open configures it itself
            # through device_params, and leaving it set here breaks the 4-chip mesh.
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    yield


def pytest_generate_tests(metafunc):
    if "mesh_device" not in metafunc.fixturenames:
        return
    # An explicit @pytest.mark.parametrize("mesh_device", ..., indirect=True) on a test
    # wins; only fill in the default when the test did not ask for a specific shape.
    if any(m.args and m.args[0] == "mesh_device" for m in metafunc.definition.iter_markers("parametrize")):
        return
    name = os.environ.get("MESH_DEVICE", "P150")
    if name not in MESH_SHAPES:
        raise pytest.UsageError(f"MESH_DEVICE={name!r} is not one of {sorted(MESH_SHAPES)}")
    metafunc.parametrize("mesh_device", [MESH_SHAPES[name]], indirect=True, ids=[name])

    # Fabric only where there is actually something to talk to. A single chip has no
    # collectives, and forcing FABRIC_1D on it makes device open fail in ethernet
    # router sync ("Fabric Router Sync: Timeout ... handshake") rather than anywhere
    # near the code under test.
    if "device_params" in metafunc.fixturenames and not any(
        m.args and m.args[0] == "device_params" for m in metafunc.definition.iter_markers("parametrize")
    ):
        # The GDN prefill depthwise conv1d needs an L1 small-allocator region; without
        # it the model forward dies with "statically allocated circular buffers clash
        # with L1 buffers", pointing at the GatedDeltaAttnSeq op rather than at the
        # missing device parameter. qwen36's own tests pass the same constant.
        from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE

        # Match the qwen36 text demo's known-good device parameters. This is not
        # cargo-culting: l1_small_size alone was not enough -- the GDN seq op still
        # failed with "circular buffers clash with L1 buffers", and the demo's
        # num_command_queues / trace_region_size are part of the L1 layout it was
        # validated against.
        params = {
            "l1_small_size": GDN_CONV1D_L1_SMALL_SIZE,
            "num_command_queues": 2,
        }
        if MESH_SHAPES[name][0] * MESH_SHAPES[name][1] > 1:
            # Fabric ONLY under expert parallelism. Replicated experts (the default,
            # see _SparseExperts) make the model collective-free on every mesh, and
            # asking for a fabric you never use is not free: the cold (1,2) open
            # trains ethernet links and fails often enough that P300 needed the
            # warm-up dance above, and a fabric left half-configured surfaces later
            # as "Unsupported fabric config" from ccl_common. No collectives, no
            # fabric, no failure mode.
            if os.environ.get("ORNITH_MOE_EP") == "1":
                params["fabric_config"] = ttnn.FabricConfig.FABRIC_1D
            params["trace_region_size"] = 1024 * 1024 * 1024
        metafunc.parametrize("device_params", [params], indirect=True, ids=[""])
