# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Re-export of the shared qwen35 sorted-dispatch module (the hybrid overlay resolves it by this port's name)."""
from models.autoports.minimaxai_minimax_m2_7.tt.moe_dispatch import *  # noqa: F401,F403
from models.autoports.minimaxai_minimax_m2_7.tt.moe_dispatch import dispatch_rows  # noqa: F401
