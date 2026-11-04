# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL-Embedding-8B and Qwen3-VL-Reranker-8B as two co-host app members that share one 4-chip mesh with an LLM.

The two checkpoints have the same architecture (Qwen3-VL-8B: every tensor has the same shape), so one text tower is
built over the whole mesh with each weight *sharded by chip*: chips listed as ``embed`` in ``QWEN3VL_COHOST_LANES``
hold the embedder's tensor, chips listed as ``rerank`` hold the reranker's (default ``embed,embed,rerank,rerank``).
Every chip runs the same program on its own weights and its own inputs, so one pass serves embedding requests on
chips 0-1 and rerank requests on chips 2-3 at once. Per chip this costs ONE 8B copy (about 8.1 GiB in bfp8), which
is what fits next to Ornith-1.5-122B; two replicated copies (16 GiB per chip) would not.

Why not one 1x1 submesh per model: a submesh gets its own allocator, blind to the parent mesh's allocations, so a
submesh model next to an LLM on the parent mesh would be handed DRAM addresses the LLM already uses. Sharding by
chip on the parent mesh keeps one allocator for everything.

Modules: ``split`` (the chip-sharded scorer), ``engine`` (traced passes + the cross-lane combiner), ``member`` (the
co-host ``AppMember`` factories ``embed`` and ``rerank`` and their HTTP apps).
"""
