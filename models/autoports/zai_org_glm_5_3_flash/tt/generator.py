# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generator of the full GLM-5.3-Flash model (stage 6): the readiness-check / serving contract
(readiness_check.contract.Generator) over tt/model.Glm53Model.

Decode is split into two traces, replayed back to back without host work in between:
  1. the model trace: per-step inputs derived on device from the position tensor (tt/decode_inputs.py), 45 layers,
     head -> logits [1,1,S,V/tp]; positions advanced at its end;
  2. the sampler trace (models.common.sampling.SamplingGenerator, semantically greedy params): token written into the
     persistent token buffer `tok_buf` that the model trace reads (tt_out_tok), so step N's sample is step N+1's input.
The host writes tokens / positions only when it is authoritative: right after a prefill, when teacher forcing overrides
the sampled token, or in the host-sampling compatibility mode (on_device_sampling=False: logits to the host, host
argmax).

Cache ownership: the model owns its caches (DSA latent pool per layer, KDA state per slot). `kv_cache` arguments of the
low-level API are accepted for the contract and must be None or `self.kv_cache`; `page_table` [S, blocks] maps the
DSA latent pool blocks of every slot.
"""

from __future__ import annotations

import os

import torch
from loguru import logger

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.model import Glm53Model, model_from_env

try:
    from readiness_check.contract import Generator
except ImportError:  # the readiness package is optional at import time (unit tests)

    class Generator:  # noqa: D101
        tokenizer = None


L1_SMALL_SIZE = 24576
TRACE_REGION_SIZE = int(
    os.environ.get("GLM53_TRACE_REGION", 64 << 20)
)  # decode needs 46 MiB (doc/full_model/memory_budget.json)


class Glm53Generator(Generator):
    def __init__(
        self, mesh, *, max_batch_size: int = 32, max_seq_len: int = 4096, defer_caches: bool = False, **model_kw
    ):
        """defer_caches=True (vLLM serving, tt/generator_vllm.py): load the weights and the KDA slot state only. The DSA
        pools come from bind_kv_cache(num_blocks) (vLLM's block count); warm-up and trace capture from
        warm_programs() / capture_traces() (the plugin's two warm-up phases). False = the standalone generator."""
        self.mesh = mesh
        self.max_batch_size = max_batch_size
        self.defer_caches = defer_caches
        if defer_caches:
            model_kw.setdefault("allocate_dsa", False)
            model_kw.setdefault("lockstep_sampling", True)
        self.model: Glm53Model = model_from_env(mesh, max_batch=max_batch_size, max_context=max_seq_len, **model_kw)
        self.max_seq_len = max_seq_len
        self.kv_cache = None  # model-owned (see module docstring)
        self.tokenizer = self._load_tokenizer()
        S = max_batch_size
        dsa = next((d.attn for d in self.model.layers if d.attn_kind == "dsa"), None)
        if defer_caches:
            self.page_table = None  # bind_kv_cache installs an all-null table
        else:
            self.page_table = dsa.default_page_table()[:, : max_seq_len // 128] if dsa is not None else None
        if self.page_table is not None:
            self.model.set_page_table(self.page_table)
        self.tok_buf = ttnn.from_torch(
            torch.zeros(1, 1, 1, self.model.SR, dtype=torch.int32),  # sampler width (>= S): the model reads [:S]
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.trace_id = None
        self.logits_out = None
        self.host_trace_id = None
        self._sampler_ready = False
        self.positions = [-1] * S
        # serving state (defer_caches mode)
        self.num_blocks = None
        self._fresh = set()  # slots prefilled since the last serving decode (their host token is authoritative)
        self._serving_decoded = False
        self._sstate = None
        self._warmed_programs = False
        self._program_cache_after_capture = None
        if self.model.sampling is not None:
            from models.autoports.zai_org_glm_5_3_flash.tt.serving_state import SlotSamplingState

            self._sstate = SlotSamplingState(self.model.sampling, self.model.SR)
        if not defer_caches:
            self._warm()

    # ------------------------------------------------------------------------------------------------ setup
    def _load_tokenizer(self):
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(str(self.model.rd.path), trust_remote_code=True)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"tokenizer unavailable ({e})")
            return None

    def _tok_view(self):
        t = ttnn.reshape(self.tok_buf, (1, self.model.SR))
        return t if self.model.SR == self.max_batch_size else ttnn.slice(t, (0, 0), (1, self.max_batch_size))

    def _warm(self):
        """Compile everything before any trace capture, with every slot inactive (no state / cache changes): one eager
        decode step, the head, the sampler programs; then capture the model trace."""
        # one prefill per constructed length first: prefill builds persistent constants lazily (DSA per-length tables,
        # ...). Built after the capture, they land in freed trace-intermediate memory and the next decode replay
        # overwrites them (a prefill after replays then hung, 2026-10-16). Slot 0's state is zeroed again below.
        self.model.choose_moe_split()
        for L, sizes in self._warm_plans():
            self.model.prefill(torch.zeros(L, dtype=torch.int64), 0, sizes=sizes)
        self.model.reset_state()
        self.model.prepare_prefill_traces()  # persistent inputs before any capture
        self.model.inputs.set_positions([-1] * self.max_batch_size)
        logits = self.model.decode_step(self._tok_view())
        if self.model.sampling is not None:
            from models.common.sampling.generator import format_sampling_params
            from models.common.sampling.sampling_params import SamplingParams

            self.model.sampling.reset_sampling_params(
                format_sampling_params(SamplingParams(temperature=0.0, top_k=1, top_p=1.0), self.model.SR)
            )
            # the exact program the sampler trace records (output into the persistent token buffer)
            self.model.sampling.precompile(logits, tt_out_tok=self.tok_buf, all_configs=False)
            self._sampler_ready = True
        ttnn.deallocate(logits)
        ttnn.synchronize_device(self.mesh)
        self.trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        try:
            self.logits_out = self.model.decode_step(self._tok_view())
        finally:
            ttnn.end_trace_capture(self.mesh, self.trace_id, cq_id=0)
        if self.model.sampling is not None:
            # capture the sampler trace on the model trace's logits buffer now: a first capture inside decode_forward,
            # right after a non-blocking model replay, hung the probe test (2026-10-16). Writes a junk token into
            # tok_buf; the first decode step after a prefill always reloads the tokens.
            self.model.sampling.sample(
                self.logits_out, tt_out_tok=self.tok_buf, enable_trace=True, skip_precompile=True
            )
        ttnn.synchronize_device(self.mesh)
        self.model.capture_prefill_traces()  # last: their outputs are consumed by each prefill before any decode replay
        self.model.reset_state()
        self.model.inputs.set_positions([-1] * self.max_batch_size)
        ttnn.synchronize_device(self.mesh)
        logger.info(f"decode traces captured (model + sampler); prefill traces {sorted(self.model._ptraces)}")

    # ------------------------------------------------------------------------------------------------ serving (vLLM)
    def program_cache_entries(self):
        """Compiled programs on the mesh (None if the binding is missing): must not grow after capture_traces()."""
        fn = getattr(self.mesh, "num_program_cache_entries", None)
        try:
            return int(fn()) if fn is not None else None
        except Exception:  # noqa: BLE001
            return None

    def bind_kv_cache(self, num_blocks: int):
        """defer_caches mode: allocate every DSA layer's pools for vLLM's `num_blocks` blocks + the model's dummy block
        (id num_blocks), install an all-null page table (block 0 = vLLM's null block). Returns the cache object that
        vLLM passes back as kv_cache: [(latent, pooled)] per DSA layer."""
        assert self.defer_caches and self.kv_cache is None, "bind_kv_cache: defer_caches mode, once"
        self.num_blocks = int(num_blocks)
        self.kv_cache = self.model.allocate_dsa_caches(self.num_blocks + 1)
        if any(d.attn_kind == "dsa" for d in self.model.layers):
            self.page_table = torch.zeros(self.max_batch_size, self.max_seq_len // 128, dtype=torch.int32)
            self.model.set_page_table(self.page_table)
        self.model.inputs.set_positions([-1] * self.max_batch_size)
        return self.kv_cache

    def _warm_plans(self):
        """Prefill warm-up prompts [(length, chunk sizes)]: every (chunk length, DSA path) pair a prefill chunk plan
        (Glm53Model.chunk_plan) can produce, before any capture (lazily built prefill constants must exist first):
        flash = each length from position 0; dense = a chunk at start n (n < top) and the smallest length after the
        top one ending at position <= 2050; sparse = each length after the top one."""
        lens = list(self.model.prefill_lengths)
        top, T = max(lens), self.max_seq_len
        plans = [(min(n, T), [n]) for n in lens]
        plans += [(2 * n, [n, n]) for n in lens if n < top and 2 * n <= T]
        plans += [(top + 2, [top, lens[0]])] if len(lens) > 1 and top + 2 <= T else []
        plans += [(top + n, [top, n]) for n in lens if top + n <= T]
        return [(L, sz) for L, sz in plans if self.num_blocks is None or -(-L // 128) < self.num_blocks]

    def _warm_lengths(self):
        """Lengths of the warm-up prompts (_warm_plans)."""
        return [L for L, _ in self._warm_plans()]

    def warm_programs(self):
        """Plugin warm-up phase 1 (no capture, idempotent): every prefill path, the prefill sampler, one eager decode
        step, the head and every sampler variant, with every slot left inactive afterwards."""
        if self._warmed_programs:
            return
        assert self.kv_cache is not None or not self.defer_caches, "bind_kv_cache before warm_programs"
        S = self.max_batch_size
        if self.page_table is not None:
            warm_pt = self.page_table.clone()
            nb = warm_pt.shape[1]
            if self.num_blocks is not None:  # free blocks 1.. (no request is live during the warm-up)
                n = min(nb, self.num_blocks - 1)
                warm_pt[0, :n] = torch.arange(1, n + 1, dtype=torch.int32)
            self.model.set_page_table(warm_pt)
        self.model.choose_moe_split()
        for L, sizes in self._warm_plans():
            self.model.prefill(torch.zeros(L, dtype=torch.int64), 0, sizes=sizes)
        logits = None
        if self.model.sampling is not None:
            from models.common.sampling.generator import format_sampling_params
            from models.common.sampling.sampling_params import SamplingParams

            greedy = format_sampling_params(SamplingParams(temperature=0.0, top_k=1, top_p=1.0), self.model.SR)
            self.model.sampling.reset_sampling_params(greedy)
            # the prefill sampler: last-row device logits [1,1,S,V/tp] -> eager sample into a fresh tensor
            lg = self.model.prefill(torch.zeros(min(100, self.max_seq_len), dtype=torch.int64), 0, device_logits=True)
            tt_tok, _ = self.model.sampling.sample(lg, enable_trace=False, count_tokens=False)
            ttnn.deallocate(lg)
        self.model.reset_state()
        if self.page_table is not None:
            self.model.set_page_table(self.page_table)
        self.model.prepare_prefill_traces()  # persistent inputs before any capture
        self.model.inputs.set_positions([-1] * S)
        logits = self.model.decode_step(self._tok_view())
        self.model.logits_to_host(logits)  # host-logits compatibility mode
        if self.model.sampling is not None:
            self.model.sampling.precompile(logits, tt_out_tok=self.tok_buf, all_configs=True)
            self._sampler_ready = True
        ttnn.deallocate(logits)
        self.model.inputs.set_positions([-1] * S)
        ttnn.synchronize_device(self.mesh)
        self._warmed_programs = True

    def capture_traces(self):
        """Plugin warm-up phase 2 (idempotent): the model decode trace, then one sampler trace per variant (penalties
        off / on x force-argmax off / on where allowed; device log-probs are never requested on this mesh). Nothing
        may be captured after this (a later capture after a non-blocking replay hung, 2026-10-16)."""
        if self.trace_id is not None:
            return
        self.warm_programs()
        ttnn.synchronize_device(self.mesh)
        self.trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        try:
            self.logits_out = self.model.decode_step(self._tok_view())
        finally:
            ttnn.end_trace_capture(self.mesh, self.trace_id, cq_id=0)
        sm = self.model.sampling
        if sm is not None:
            ts = sm.tt_sampling
            saved = (sm._penalties_active, ts._force_argmax_sampling)
            try:
                for penalties in (False, True):
                    for fa in (False, True):
                        if fa and not getattr(ts, "_allow_force_argmax_sampling", False):
                            continue
                        sm._penalties_active, ts._force_argmax_sampling = penalties, fa
                        _, slot = sm._trace_slot(penalties, False, fa)
                        if slot["id"] is None:
                            sm.capture_trace(self.logits_out, tt_out_tok=self.tok_buf, skip_precompile=True)
            finally:
                sm._penalties_active, ts._force_argmax_sampling = saved
        ttnn.synchronize_device(self.mesh)
        self.model.capture_prefill_traces()  # last (see _warm)
        self.model.reset_state()
        self.model.inputs.set_positions([-1] * self.max_batch_size)
        ttnn.synchronize_device(self.mesh)
        self._program_cache_after_capture = self.program_cache_entries()
        logger.info(f"serving traces captured (model + sampler variants); programs {self._program_cache_after_capture}")

    def _read_positions(self) -> torch.Tensor:
        t = ttnn.to_torch(ttnn.get_device_tensors(self.model.inputs.pos)[0])
        return t.reshape(-1)[: self.max_batch_size].round().to(torch.int64)

    def _decode_serving(
        self,
        tokens,
        start_pos,
        *,
        page_table=None,
        enable_trace=True,
        slots=None,
        sampling_params=None,
        reload_inputs=True,
        reload_sampling_params=False,
        reset_sampling_state=False,
        fresh_slots=(),
        prompt_tokens=None,
        output_tokens=None,
        read_from_device=True,
        **_,
    ):
        """vLLM decode step (decode_forward(..., serving=True)). Rows -> model slots via `slots`; sampling_params,
        prompt_tokens, output_tokens already in SLOT order. Returns host logits [B, V] (no sampling_params), host
        tokens [B] in row order (read_from_device) or the device token buffer (read_decode_output reads it)."""
        from models.autoports.zai_org_glm_5_3_flash.tt.serving_state import merge_async_ahead

        S = self.max_batch_size
        tokens = torch.as_tensor(tokens).reshape(-1).to(torch.int64)
        start_pos = torch.as_tensor(start_pos).reshape(-1).to(torch.int64)
        slots = list(slots) if slots is not None else list(range(tokens.shape[0]))
        if page_table is not None:
            self._set_page_rows(page_table, slots)  # host compare; device copy only on change
        device = sampling_params is not None and self.model.sampling is not None
        fresh = set(int(s) for s in fresh_slots) | self._fresh
        if reload_inputs:
            pos = torch.full((S,), -1, dtype=torch.int64)
            tok = torch.zeros(S, dtype=torch.int64)
            for r, s in enumerate(slots):
                pos[s] = int(start_pos[r])
                tok[s] = int(tokens[r])
            if device and enable_trace and self._serving_decoded:
                tok, pos = merge_async_ahead(tok, pos, self._read_tokens(), self._read_positions(), fresh)
            pos_l = [int(p) for p in pos.tolist()]
            if pos_l != self.model.inputs.positions:
                self.model.inputs.set_positions(pos_l)
            self._write_tokens([int(t) for t in tok.tolist()])
        if device and self._sstate is not None:
            self._sstate.apply_decode(
                sampling_params,
                list(self.model.inputs.positions) + [-1] * (self.model.SR - S),
                reload_params=reload_sampling_params,
                reset_state=reset_sampling_state,
                host_authoritative=reload_inputs,
                prompt_tokens_slots=prompt_tokens,
                output_tokens_slots=output_tokens,
            )
        if enable_trace:
            ttnn.execute_trace(self.mesh, self.trace_id, cq_id=0, blocking=False)
            logits = self.logits_out
        else:
            logits = self.model.decode_step(self._tok_view())
        self.model.inputs.host_advance()
        self._fresh.clear()
        self._serving_decoded = True
        if device:
            self.model.sampling.sample(logits, tt_out_tok=self.tok_buf, enable_trace=enable_trace)
            if not enable_trace:
                ttnn.deallocate(logits)
            if not read_from_device:
                return self.tok_buf
            out = self._read_tokens()
            return torch.stack([out[s] for s in slots])
        host = self.model.logits_to_host(logits)[0, 0, :, : self.model.vocab]
        if not enable_trace:
            ttnn.deallocate(logits)
        return torch.stack([host[s] for s in slots])

    def read_decode_output(self, handle, async_read=True):
        """Device token buffer -> host copy (non-blocking + event when async_read)."""
        if not async_read:
            return handle.cpu()
        host = handle.cpu(blocking=False)
        return host, [ttnn.record_event(self.mesh, 0)]

    def process_decode_output_host(self, host, slots):
        """Host copy of the token buffer -> int64 tokens in row order (die 0's row: every die holds the same token
        with lockstep sampling)."""
        t = ttnn.to_torch(ttnn.get_device_tensors(host)[0]).reshape(-1)[: self.max_batch_size].to(torch.int64)
        return torch.stack([t[int(s)] for s in slots])

    # ------------------------------------------------------------------------------------------------ state
    def reset(self):
        self.model.reset_state()
        self.positions = [-1] * self.max_batch_size
        self.model.inputs.set_positions(self.positions)
        if self.page_table is not None:
            self.model.set_page_table(self.page_table)

    def _write_tokens(self, toks: list[int]):
        t = torch.zeros(1, 1, 1, self.model.SR, dtype=torch.int32)
        t[0, 0, 0, : len(toks)] = torch.tensor(toks, dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)
            ),
            self.tok_buf,
        )

    def _read_tokens(self) -> torch.Tensor:
        t = ttnn.to_torch(ttnn.get_device_tensors(self.tok_buf)[0])
        return t.reshape(-1)[: self.max_batch_size].to(torch.int64)

    # ------------------------------------------------------------------------------------------------ low level
    def prefill_forward(
        self,
        tokens,
        *,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        return_all_logits=False,
        slots=None,
        sampling_params=None,
        **kw,
    ):
        """tokens [B, L_pad]; prompt_lens: real lengths; slots: decode slot per row (default 0..B-1). Returns logits
        [B, 1, V] (last position) or [B, L, V] (return_all_logits, B = 1). sampling_params (serving): one
        models.common SamplingParams per row -> on-device sampled int64 tokens [B] instead of logits."""
        assert kv_cache is None or kv_cache is self.kv_cache
        tokens = torch.as_tensor(tokens)
        B = tokens.shape[0]
        prompt_lens = list(prompt_lens) if prompt_lens is not None else [tokens.shape[1]] * B
        slots = list(slots) if slots is not None else list(range(B))
        if page_table is not None:
            self._set_page_rows(page_table, slots)
        if sampling_params is not None and self.model.sampling is not None and not return_all_logits:
            return self._prefill_sampled(tokens, prompt_lens, slots, sampling_params)
        outs = []
        for b in range(B):
            ids = tokens[b, : prompt_lens[b]].to(torch.int64)
            self._fresh.add(int(slots[b]))
            out = self.model.prefill(ids, slots[b], all_logits=return_all_logits)
            outs.append(out if return_all_logits else out.reshape(1, -1))
            self.positions[slots[b]] = prompt_lens[b]
        return torch.stack(outs, dim=0)

    def _prefill_sampled(self, tokens, prompt_lens, slots, sampling_params):
        """Serving prefill with on-device sampling: per request, the last row's device logits (copied to every sampler
        row) are sampled eagerly with that request's params set on its slot; the token is row `slot`. The output is a
        fresh tensor: the decode token buffer (live slots' fed-back tokens) is not touched."""
        out = []
        for b, s in enumerate(slots):
            ids = tokens[b, : prompt_lens[b]].to(torch.int64)
            lg = self.model.prefill(ids, int(s), device_logits=True)
            self._sstate.apply_prefill(sampling_params[b], int(s), ids)
            tt_tok, _ = self.model.sampling.sample(lg, enable_trace=False)
            ttnn.deallocate(lg)
            row = ttnn.to_torch(ttnn.get_device_tensors(tt_tok)[0]).reshape(-1)
            out.append(int(row[int(s)]))
            self.positions[int(s)] = int(prompt_lens[b])
            self._fresh.add(int(s))
        return torch.tensor(out, dtype=torch.int64)

    def _set_page_rows(self, page_table, slots):
        pt = torch.as_tensor(page_table).to(torch.int32)
        if self.page_table is None:
            return
        new = self.page_table.clone()
        for r, s in enumerate(slots):
            n = min(pt.shape[1], new.shape[1])
            new[s, :n] = pt[r, :n]
        if not torch.equal(new, self.page_table):
            self.page_table = new
            self.model.set_page_table(new)

    def decode_forward(
        self,
        tokens,
        start_pos,
        *,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        on_device_sampling=True,
        reload_inputs=True,
        slots=None,
        **kw,
    ):
        """One decode step. tokens [B, 1], start_pos [B] (slot rows; -1 = inactive). Returns sampled tokens [B]
        (on-device greedy) or logits [B, V] (host compatibility mode). reload_inputs=False: the device already holds
        the tokens (fed back by the sampler) and positions (advanced by the trace). serving=True: the vLLM step
        (_decode_serving: per-slot sampling params, async-ahead reload, read_from_device=False)."""
        assert kv_cache is None or kv_cache is self.kv_cache
        if kw.pop("serving", False):
            return self._decode_serving(
                tokens,
                start_pos,
                page_table=page_table,
                enable_trace=enable_trace,
                reload_inputs=reload_inputs,
                slots=slots,
                **kw,
            )
        S = self.max_batch_size
        tokens = torch.as_tensor(tokens).reshape(-1).to(torch.int64)
        start_pos = torch.as_tensor(start_pos).reshape(-1).to(torch.int64)
        B = tokens.shape[0]
        slots = list(slots) if slots is not None else list(range(B))
        if page_table is not None:
            self._set_page_rows(page_table, slots)
        if reload_inputs:
            pos = [-1] * S
            tok = [0] * S
            for r, s in enumerate(slots):
                pos[s] = int(start_pos[r])
                tok[s] = int(tokens[r])
            if pos != self.model.inputs.positions:
                self.model.inputs.set_positions(pos)
            self._write_tokens(tok)
        if enable_trace:
            ttnn.execute_trace(self.mesh, self.trace_id, cq_id=0, blocking=False)
            logits = self.logits_out
        else:
            logits = self.model.decode_step(self._tok_view())
        self.model.inputs.host_advance()
        if on_device_sampling and self.model.sampling is not None:
            self.model.sampling.sample(logits, tt_out_tok=self.tok_buf, enable_trace=enable_trace)
            out = self._read_tokens()
            return torch.stack([out[s] for s in slots])
        host = self.model.logits_to_host(logits)[0, 0, :, : self.model.vocab]
        if not enable_trace:
            ttnn.deallocate(logits)
        return torch.stack([host[s] for s in slots])

    # ------------------------------------------------------------------------------------------------ high level
    def prefill_logits(self, prompt_token_ids):
        self.reset()
        ids = torch.tensor(list(prompt_token_ids), dtype=torch.int64).reshape(1, -1)
        return self.prefill_forward(ids, prompt_lens=[ids.shape[1]], return_all_logits=True)

    def generate(
        self, prompt_token_ids, max_new_tokens, *, next_input=None, enable_trace=True, on_device_sampling=True, **kw
    ):
        """Greedy generation (slot 0). next_input(i, pred_i) may override the token fed at step i (teacher forcing).
        Returns the model's own predictions."""
        self.reset()
        on_device_sampling = on_device_sampling and self.model.sampling is not None
        ids = torch.tensor(list(prompt_token_ids), dtype=torch.int64).reshape(1, -1)
        P = ids.shape[1]
        logits = self.prefill_forward(ids, prompt_lens=[P])
        pred = int(torch.argmax(logits[0, -1, : self.model.vocab]).item())  # first token: the prefill logits row
        preds = []
        pos = P
        reload = True
        for i in range(int(max_new_tokens)):
            preds.append(pred)
            if i == int(max_new_tokens) - 1:
                break
            forced = next_input(i, pred) if next_input is not None else None
            tok = pred if forced is None else int(forced)
            reload = reload or tok != pred or not on_device_sampling
            out = self.decode_forward(
                torch.tensor([[tok]]),
                torch.tensor([pos]),
                enable_trace=enable_trace,
                on_device_sampling=on_device_sampling,
                reload_inputs=reload,
            )
            pred = int(out[0]) if on_device_sampling else int(torch.argmax(out[0]).item())
            pos += 1
            reload = False
        if next_input is not None and len(preds) == int(max_new_tokens):
            next_input(len(preds) - 1, preds[-1])
        return preds

    def teardown(self):
        self.model.release_prefill_traces()
        if self.trace_id is not None:
            ttnn.release_trace(self.mesh, self.trace_id)
            self.trace_id = None


def _check_mesh(mesh_device):
    try:
        trace = ttnn.get_memory_view(mesh_device, ttnn.BufferType.TRACE).total_bytes_per_bank
        small = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1_SMALL).total_bytes_per_bank
    except Exception as e:  # noqa: BLE001
        logger.warning(f"memory view unavailable ({e}); assuming the mesh has a trace region")
        return
    if trace <= 0 or small < L1_SMALL_SIZE:
        raise RuntimeError(
            f"mesh opened with trace_region_size={trace} l1_small_size={small}; this model needs a trace region and "
            f"l1_small_size>={L1_SMALL_SIZE}: open it with models/autoports/zai_org_glm_5_3_flash/tools/readiness.py"
        )


def build_generator(model_dir, mesh_device, **kwargs):
    _check_mesh(mesh_device)
    return Glm53Generator(
        mesh_device,
        max_batch_size=int(kwargs.get("max_batch_size", os.environ.get("GLM53_MAX_BATCH", 32))),
        max_seq_len=int(kwargs.get("max_seq_len", os.environ.get("GLM53_MAX_SEQ_LEN", 4096))),
    )
