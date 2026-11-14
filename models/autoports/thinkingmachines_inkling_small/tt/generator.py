# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Re-export of the shared qwen35 generator module (the M2.7 port's): the hybrid overlay resolves it by this port's name
and patches methods of its classes. (Inkling's own MTP drafter will live in tt/inkling_mtp.py.)"""
from models.autoports.minimaxai_minimax_m2_7.tt.generator import *  # noqa: F401,F403
