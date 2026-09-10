# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ornith-1.5-35B-A3B text model.

tok_embeddings -> 40 x OrnithDecoderLayer -> RMSNorm -> LM head, i.e. the qwen36
assembly with an MoE feed-forward in every layer. The hybrid state is what makes this
model cheap at long context: only the 10 full-attention layers hold a KV cache
(20 KB/token, so 256K is 5.4 GB), while the 30 Gated DeltaNet layers carry a
fixed-size recurrent state of ~67 MB per sequence regardless of length.
"""
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.layer import OrnithDecoderLayer
from models.demos.blackhole.qwen36.tt.model import Qwen36Model


class OrnithModel(Qwen36Model):
    """Ornith-1.5-35B-A3B on Blackhole."""

    DECODER_LAYER_CLS = OrnithDecoderLayer

    def allocate_kv_caches(self, kv_cache_shape, dtype, batch_size=1):
        """Allocate paged KV plus a BATCHED Gated DeltaNet state.

        qwen36's single-device path builds a `[batch_size, ...]` recurrent state and then
        hands it to `set_external_state`, which asserts `[1, ...]` -- so batch_size > 1
        fails outright ("recurrent_state shape mismatch: [32, 32, 128, 128] !=
        [1, 32, 128, 128]"). Its tensor-parallel path avoids this by letting the module
        own its state (`attention.B = batch_size`), which is why batched serving works
        there and not here.

        The underlying state already supports a batch: `init_recurrent_state` allocates
        `[B, num_v_heads, head_k_dim, head_v_dim]`. Only the external-buffer hand-off is
        batch-1. So this calls `reset_state(batch_size)` and lets each DeltaNet layer own
        its state, exactly as the TP path does.

        This matters more for this model than for a dense one. Without sparse expert
        dispatch the MoE reads every local expert once per STEP, so at batch 1 that cost
        is paid for a single token; batching is what amortizes it, and measured serving
        at max_num_seqs=1 is flat at 3.3 tok/s aggregate no matter the concurrency.

        MEASURED RESULT: this does NOT work, and is off by default. Module-managed GDN
        state takes a different decode kernel configuration whose statically allocated
        circular buffers need 1,684,608 B per core against a 1,572,864 B L1 -- and that
        figure is IDENTICAL at batch 4 and batch 8, so it is the state-management path
        that does not fit, not the batch size. The batch-1 external-buffer path is the
        one that serves.

        Kept behind ORNITH_BATCHED_GDN=1 because it does clear the real structural
        blocker: with it, the model reaches GDN decode at batch 8, where the stock path
        cannot even allocate (`recurrent_state shape mismatch` at
        `set_external_state`). Making it usable needs the GDN decode kernel's L1
        footprint reduced, which is kernel work.
        """
        import os

        import ttnn

        if os.environ.get("ORNITH_BATCHED_GDN") != "1":
            return super().allocate_kv_caches(kv_cache_shape, dtype, batch_size)

        assert self._deltanet_external_states is None, "allocate_kv_caches already called; free_kv_caches first"
        kv_caches = []
        for _ in self._attention_layer_indices:
            kv_caches.append(
                [
                    ttnn.zeros(kv_cache_shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device),
                    ttnn.zeros(kv_cache_shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device),
                ]
            )
        self.set_paged_kv_caches(kv_caches)

        for layer in self.layers:
            if not layer.is_full_attention:
                layer.attention.B = batch_size
                layer.attention.reset_state(batch_size)
        # Marker only: the state lives in the modules, not in external buffers, so
        # free_kv_caches has nothing of its own to release.
        self._deltanet_external_states = []
        return kv_caches
