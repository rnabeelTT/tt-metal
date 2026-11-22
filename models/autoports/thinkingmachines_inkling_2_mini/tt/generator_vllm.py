# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM adapter for thinkingmachines/Inkling-2-Mini (text) on a (1, N) Blackhole mesh (N = 1, 2, 4: tensor parallel).

The engine owns every cache (tt/attention.AttnCache with one slot per decode row: the global layers' paged KV, the
sliding layers' rings, the conv states), so the KV allocation vLLM asks for is a token budget only and vLLM's page
table is not used. Requests are prefilled into a physical slot in chunks (one sequence at a time on the lane) and then
decode together in one traced batched step over every slot (tt/batched.py).

vLLM rows vs physical slots: the plugin condenses its persistent batch (a finished request's row is refilled with the
last live one and the move is reported as ``slot_remap``). Device state never moves: the adapter keeps a row -> slot
map, updated by each remap, and permutes the per-step inputs / outputs.

Sampling: on the host (the step returns the next-token logits); prefix caching and chunked prefill are off (v0).

Config: vLLM 0.26 / transformers 5.12 do not know ``inkling2_loop``; serve with ``--hf-config-path`` pointing at this
port's ``hf_config/`` (a flat config with an auto_map, ``--trust-remote-code``). The tokenizer and the chat template come
from the checkpoint; vLLM's ``inkling`` reasoning / tool parsers apply (same message tokens as Inkling v1).
"""

from __future__ import annotations

import os
import time

import torch
import ttnn
from loguru import logger

from models.autoports.thinkingmachines_inkling_2_mini.tt.batched import BatchedDecoder, TracedDecoder
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model

HF_MODEL = "thinkingmachines/Inkling-2-Mini"
DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
EXPERT_DT = os.environ.get("I2M_EXPERT_DT", "bfp8")
KV_DT = os.environ.get("I2M_KV_DT", "bf16")
PREFILL_CHUNK = int(os.environ.get("I2M_PREFILL_CHUNK", "256"))
TRACE = os.environ.get("I2M_TRACE", "1") == "1"


class Inkling2ForCausalLM:
    """vLLM bridge (HF arch Inkling2LoopForConditionalGeneration, text only)."""

    supports_multimodal = False
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": False,
        "supports_chunked_prefill": False,
    }

    def __init__(self, model: Inkling2Model = None, mesh=None, B: int = 1, max_seq_len: int = 0, vllm_config=None,
                 prefix: str = ""):
        # (vllm_config / prefix: vLLM's registry inspects the class for its model protocol; the TT plugin builds the
        # adapter through initialize_vllm_model)
        self.m, self.mesh, self.B, self.max_seq_len = model, mesh, B, max_seq_len
        self.dec = BatchedDecoder(model, B)
        self.tr = TracedDecoder(self.dec) if TRACE else None
        self.row2slot = list(range(B))  # vLLM row -> physical slot
        self.V = model.V
        self.vocab = int(model.cfg.vocab_size)
        self.N = mesh.get_num_devices()
        self._captured = False
        self.already_warmed_up_prefill = False
        self._decode_steps = 0

    # vLLM's text-generation protocol (registry inspection only; the TT plugin drives prefill_forward / decode_forward)
    def embed_input_ids(self, input_ids):
        raise NotImplementedError("TT model: driven through prefill_forward / decode_forward")

    def forward(self, input_ids, positions, *args, **kwargs):
        raise NotImplementedError("TT model: driven through prefill_forward / decode_forward")

    def compute_logits(self, hidden_states):
        raise NotImplementedError("TT model: driven through prefill_forward / decode_forward")

    # ------------------------------------------------------------------ construction
    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len=None, tt_data_parallel=1,
                              optimizations=None, **kwargs):
        if int(tt_data_parallel) != 1:
            raise ValueError(f"{cls.__name__}: tensor parallel only (tt_data_parallel={tt_data_parallel})")
        max_seq_len = int(max_seq_len or 32768)
        B = int(max_batch_size)
        t0 = time.time()
        model = Inkling2Model(mesh_device, max_seq_len=max_seq_len + 64, expert_dtype=DT[EXPERT_DT], kv_dtype=DT[KV_DT],
                              prefill_chunk=PREFILL_CHUNK, slots=B)
        logger.info(f"{cls.__name__}: {B} slots x {max_seq_len} tokens, experts {EXPERT_DT}, KV {KV_DT}, "
                    f"{mesh_device.get_num_devices()} device(s), up in {time.time() - t0:.0f}s (engine-owned caches)")
        return cls(model, mesh_device, B, max_seq_len)

    # ------------------------------------------------------------------ KV: a token budget only
    @classmethod
    def get_max_tokens_all_users(cls, model_name="", num_devices=1, tt_data_parallel=1, max_model_len=None,
                                 max_num_seqs=None, **kwargs):
        n = int(max_model_len or 32768) * int(max_num_seqs or 1)
        return (n + 63) // 64 * 64

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        logger.info(f"KV: vLLM asked for {num_layers} x {tuple(kv_cache_shape)}; the engine keeps its own caches")
        return [(None, None)] * int(num_layers)

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        return [(None, None)] * len(per_layer_specs)

    # ------------------------------------------------------------------ warm-up
    def warmup_model_prefill(self, kv_cache=None, enable_trace=False, *args, **kwargs):
        """Compile every prefill chunk size (multiples of 64 up to the chunk) on slot 0 before any trace capture."""
        if not getattr(self, "_pf_warmed", False):  # (the plugin may call this again after the decode capture)
            self._pf_warmed = True
            t0 = time.time()
            for n in range(64, PREFILL_CHUNK + 1, 64):
                self.m.reset()
                self.m.forward(torch.zeros(n, dtype=torch.long), 0, valid=n, slot=0, last_only=True)
            self.m.reset()
            logger.info(f"prefill warm-up ({PREFILL_CHUNK // 64} chunk sizes) in {time.time() - t0:.0f}s")
        self.already_warmed_up_prefill = True

    def warmup_model_decode(self, *args, **kwargs):
        if not self.already_warmed_up_prefill:
            self.warmup_model_prefill()
        if self.tr is not None and not self._captured:
            t0 = time.time()
            self.dec.set_inputs(torch.zeros(self.B, dtype=torch.long), torch.full((self.B,), -1))
            self.tr.capture()
            self._captured = True
            logger.info(f"decode trace captured ({self.B} slots) in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------ helpers
    def _host_logits(self, t, rows):
        lg = ttnn.to_torch(ttnn.get_device_tensors(t)[0] if self.N > 1 else t).float()
        return lg.reshape(-1, lg.shape[-1])[:rows, : self.V]

    def _apply_remap(self, slot_remap):
        if slot_remap is None:
            return
        remap = [int(v) for v in torch.as_tensor(slot_remap).reshape(-1).tolist()]
        if remap == list(range(len(remap))):
            return
        old = list(self.row2slot)
        new = [old[r] for r in remap]
        # rows whose source moved elsewhere now duplicate a slot: give them the slots no row holds (keeps a permutation)
        seen, dup = set(), []
        for i, s in enumerate(new):
            if s in seen:
                dup.append(i)
            seen.add(s)
        free = [s for s in range(self.B) if s not in seen]
        for i, s in zip(dup, free):
            new[i] = s
        self.row2slot = new

    # ------------------------------------------------------------------ prefill
    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        """Prefill N requests into their rows' slots; returns the last prompt token's logits [N, 1, vocab] (host)."""
        N = int(tokens.shape[0])
        start_pos = kwargs.get("start_pos")
        if start_pos is not None and any(int(v) for v in torch.as_tensor(start_pos).reshape(-1).tolist()):
            raise NotImplementedError("prefix caching / chunked prefill are off (start_pos must be 0)")
        rows = kwargs.get("empty_slots")
        rows = list(range(N)) if rows is None else [int(r) for r in rows]
        plens = [int(prompt_lens[u]) for u in range(N)] if prompt_lens is not None else [int(tokens.shape[1])] * N
        out = torch.full((N, 1, self.vocab), float("-inf"))
        for u in range(N):
            P, slot = plens[u], self.row2slot[rows[u]]
            ids = torch.as_tensor(tokens[u, :P]).to(torch.long)
            t0 = time.time()
            self.m.reset()
            lg = None
            for p0 in range(0, P, PREFILL_CHUNK):
                c = ids[p0 : p0 + PREFILL_CHUNK]
                n = int(c.numel())
                if n % 64:
                    c = torch.cat([c, torch.zeros((-n) % 64, dtype=c.dtype)])
                lg = self.m.forward(c, p0, valid=n, slot=slot, last_only=True, last_row=n - 1)
            self.dec.load_slot(slot)
            out[u, 0, : self.V] = self._host_logits(lg, 1)[0]
            logger.info(f"prefill {P} tokens -> row {rows[u]} slot {slot} in {time.time() - t0:.2f}s")
        return out

    # ------------------------------------------------------------------ decode
    def decode_forward(self, *args, **kwargs):
        tokens = kwargs.get("tokens", args[0] if args else None)
        start_pos = kwargs.get("start_pos", args[1] if len(args) > 1 else None)
        self._apply_remap(kwargs.get("slot_remap"))
        tok = torch.as_tensor(tokens).reshape(-1).to(torch.long)
        pos = torch.as_tensor(start_pos).reshape(-1).to(torch.long)
        R = int(pos.numel())
        tok_s = torch.zeros(self.B, dtype=torch.long)
        pos_s = torch.full((self.B,), -1, dtype=torch.long)
        for r in range(min(R, self.B)):
            if int(pos[r]) >= 0:
                s = self.row2slot[r]
                tok_s[s], pos_s[s] = tok[r], pos[r]
        d = self.tr if (self.tr is not None and self._captured) else self.dec
        d.set_inputs(tok_s, pos_s)
        lg = self._host_logits(d.step(), self.B)  # [B, V] by physical slot
        out = torch.full((R, 1, self.vocab), float("-inf"))
        for r in range(min(R, self.B)):
            if int(pos[r]) >= 0:
                out[r, 0, : self.V] = lg[self.row2slot[r]]
        self._decode_steps += 1
        return out

    def read_decode_output(self, tt_out, async_read=False, **kwargs):
        return (tt_out, []) if async_read else tt_out

    def process_decode_output_host(self, tt_out, is_tokens=False, **kwargs):
        return tt_out


__all__ = ["Inkling2ForCausalLM", "HF_MODEL"]
