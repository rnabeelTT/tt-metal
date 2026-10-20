# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests of the vLLM adapter's interface translation (no device, no vLLM server): slot map vs the plugin's
slot_remap, page-table padding, row -> slot sampling params, the async-ahead merge, the serving profiles (sizing rule,
tt-model.yaml in sync, precision policy, parent-checkpoint overlay), and the adapter's calls
into a fake Glm53Generator that implements the API of doc/vllm/DESIGN.md section 9."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import models.autoports.zai_org_glm_5_3_flash.tt.generator as gen_mod
import models.autoports.zai_org_glm_5_3_flash.tt.generator_vllm as gv
from models.autoports.zai_org_glm_5_3_flash.tt import serving_profiles as sp
from models.autoports.zai_org_glm_5_3_flash.tt.serving_state import (
    merge_async_ahead,
    rows_to_slots,
    rows_to_slots_tokens,
)


# ------------------------------------------------------------------------------------------------ pure helpers
def test_slot_map_follows_plugin_remap(expect_error):
    m = gv.SlotMap(4)
    # plugin: request A in slot 0, B in slot 1, C in slot 2; A finishes -> rows [B, C] read slots [1, 2]
    assert m.remap(torch.tensor([1, 2, 0, 3]))
    assert m.rows(2) == [1, 2]  # B's state is still in model slot 1, C's in 2: nothing moved on device
    # a new request is prefilled into vLLM slot 2 (the plugin's free slot after the remap) -> model slot 0 (A's old)
    assert m[2] == 0
    assert not m.remap(torch.arange(4))  # identity: no change
    with expect_error(AssertionError, "not a permutation"):
        m.remap(torch.tensor([0, 0, 1, 2]))


def test_pad_page_table_uses_null_block():
    pt = torch.tensor([[5, 6], [7, 0]], dtype=torch.int32)
    out = gv.pad_page_table(pt, 4)
    assert out.tolist() == [[5, 6, 0, 0], [7, 0, 0, 0]]
    assert gv.pad_page_table(torch.zeros(2, 6, dtype=torch.int32), 4).shape == (2, 4)


def test_rows_to_slots_params_and_tokens():
    sp = SimpleNamespace(
        temperature=[0.7, 0.0],
        top_k=[20, 1],
        top_p=[0.9, 1.0],
        presence_penalty=[0.0, 0.5],
        frequency_penalty=[0.0, 0.0],
        repetition_penalty=[1.0, 1.1],
        seed=[42, None],
        enable_log_probs=[False, False],
        num_logprobs=None,
    )
    out = rows_to_slots(sp, [3, 1], 4)
    assert out.temperature == [0.0, 0.0, 0.0, 0.7]
    assert out.top_k == [1, 1, 1, 20]
    assert out.seed == [None, None, None, 42]
    assert out.presence_penalty[1] == 0.5 and out.repetition_penalty[1] == 1.1
    t = rows_to_slots_tokens(torch.tensor([[1, 2], [3, -1]]), [3, 1], 4)
    assert t.tolist() == [[-1, -1], [3, -1], [-1, -1], [1, 2]]


def test_merge_async_ahead():
    host_tok = torch.tensor([10, 20, 30, 0])
    host_pos = torch.tensor([5, 7, 9, -1])
    dev_tok = torch.tensor([11, 21, 99, 0])
    dev_pos = torch.tensor([6, 7, 3, -1])  # slot 0 one ahead, slot 1 equal, slot 2 stale (previous occupant)
    tok, pos = merge_async_ahead(host_tok, host_pos, dev_tok, dev_pos, fresh_slots=[1])
    assert tok.tolist() == [11, 20, 30, 0]  # slot 1 fresh -> host; slot 2 disagrees -> host
    assert pos.tolist() == [6, 7, 9, -1]


# ------------------------------------------------------------------------------------------------ serving profiles
def _data(**budget):
    d = sp.load()
    d["budget"] = {**d["budget"], **budget}
    d["measured"] = {}
    return d


def test_serving_profiles_rule(expect_error):
    """The sizing rule of tt/serving_profiles.py, by hand: pool = (free - extra KDA slots - reserve) / block bytes."""
    d = _data(
        free_after_load_bytes=3_047_399_936,
        kda_bytes_per_slot=35_651_584,
        dsa_bytes_per_block=811_008,
        reserve_bytes=734_003_200,
    )
    assert sp.pool_blocks(d, 1) == (3_047_399_936 - 734_003_200) // 811_008 == 2852
    assert sp.pool_blocks(d, 2) == sp.pool_blocks(d, 1)  # 1 slot is padded to 2 KDA slots: free_after_load has both
    assert sp.pool_blocks(d, 3) == sp.pool_blocks(d, 4)  # odd S pads to S + 1 (MultichipKDA)
    assert sp.pool_blocks(d, 32) == (3_047_399_936 - 30 * 35_651_584 - 734_003_200) // 811_008
    for S in (1, 4, 8, 16, 32):
        L = sp.max_model_len_for(d, S)
        assert L % sp.context_granularity(L) == 0 and sp.model_context(L) == L
        assert sp.blocks_needed(S, L) <= sp.pool_blocks(d, S) < sp.blocks_needed(S, L + 2048)
        # the plugin: num_blocks = ceil((max_tokens_all_users + block * S) / block); the model adds the dummy block
        nb = -(-(sp.max_tokens_all_users(d, S) + 128 * S) // 128)
        assert nb + 1 == sp.pool_blocks(d, S)
    assert sp.supported_context(d) == sp.max_model_len_for(d, 1) == 364544
    with expect_error(ValueError, "largest max_model_len"):
        sp.check_fit(d, 32, sp.max_model_len_for(d, 32) + 2048)


def test_serving_profiles_measured_wins_and_reserve():
    d = _data()
    est = sp.pool_blocks(d, 16)
    d["measured"] = {"16": {"pool_blocks": est - 100, "evidence": "x"}}
    assert sp.pool_blocks(d, 16) == est - 100 and sp.pool_blocks(d, 1) == sp.pool_blocks(_data(), 1)
    assert sp.pool_blocks(d, 16, extra_reserve_bytes=d["budget"]["dsa_bytes_per_block"] * 10) == est - 110
    assert "measured" in sp.pool_source(d, 16) and "budget rule" in sp.pool_source(d, 1)


def test_model_context_granularity_matches_generator():
    for T in (128, 384, 512, 1920, 2048, 8000, 104_448, 282_624, 364_544):
        t = -(-T // 128) * 128
        assert sp.context_granularity(t) == gen_mod.context_granularity(t)
        mc = sp.model_context(T)
        assert mc >= T and mc % 128 == 0 and mc % gen_mod.context_granularity(mc) == 0
    assert sp.model_context(8000) == 8192 and sp.model_context(300) == 384


def test_shipped_data_file_consistent_and_yaml_in_sync():
    """serving_profiles.json validates; tt-model.yaml carries exactly its numbers (render-yaml is a no-op)."""
    from models.autoports.zai_org_glm_5_3_flash.tools import serving_profiles as tool

    d = sp.load()
    assert sp.validate(d) == []
    y = tool.YAML.read_text()
    assert tool.render_yaml(d, y) == y, "tt-model.yaml is out of sync: run tools/serving_profiles.py render-yaml -w"
    import yaml

    m = yaml.safe_load(y)
    default = sp.profile(d)
    assert (m["serve"]["max_num_seqs"], m["serve"]["max_model_len"]) == (
        default["max_num_seqs"],
        default["max_model_len"],
    )
    assert m["serve"]["additional_config"]["tt"]["trace_region_size"] == d["trace_region_bytes"]
    assert int(m["serve"]["env"]["GLM53_TRACE_REGION"]) == d["trace_region_bytes"]
    assert m["weights"]["revision"] == d["weights_revision"] == m["serve"]["env"]["GLM53_WEIGHTS_REVISION"]
    assert d["weights_revision"] in m["serve"]["args"]
    assert m["serve"]["block_size"] == d["block_size"] == gv.BLOCK_SIZE
    assert m["serve"]["env"]["GLM53_CACHE_DIR"].startswith("/weight-cache")  # tt-model's persistent weight mount
    names = {p["name"]: p for p in m["serve_profiles"]}
    for p in sp.profiles(d):
        y_p = names[p["name"]]
        assert y_p.get("max_num_seqs", m["serve"]["max_num_seqs"]) == p["max_num_seqs"]
        assert y_p.get("max_model_len", m["serve"]["max_model_len"]) == p["max_model_len"]
    assert m["default_profile"] == d["default_profile"]
    code = set(m["source"]["code"])
    for rel in ("serving_profiles.json", d["precision"]["config"], "doc/context_contract.json", "tt", "vllm_bundle"):
        assert f"models/autoports/zai_org_glm_5_3_flash/{rel}" in code, rel


def test_tool_ingest_refresh(monkeypatch, tmp_path, expect_error):
    """tools/serving_profiles.py: a completed capacity probe becomes measured[S] (pool incl. the dummy block), refresh
    moves that profile's max_model_len to the measured limit; an incomplete probe is refused."""
    import json

    from models.autoports.zai_org_glm_5_3_flash.tools import serving_profiles as tool

    data_p = tmp_path / "serving_profiles.json"
    data_p.write_text(sp.path().read_text())
    monkeypatch.setenv("GLM53_SERVING_PROFILES", str(data_p))
    d = sp.load()
    probe = {
        "mode": "capacity",
        "S": 16,
        "max_seq_len": 290816,
        "num_blocks": 2200,
        "capped_by_max_seq_len": False,
        "trace_region_bytes": d["trace_region_bytes"],
        "free_after_load_bytes": 2_500_000_000,
        "dsa_bytes_per_block_all_layers": 811_008,
        "prompt_len": 281_000,
        "copy_accuracy": 1.0,
        "after_capture": {"free": 600_000_000, "trace_allocated": 48_000_000},
        "after_prefill": {"free": 590_000_000},
        "prefill_peaks": {"dsa+moe/2048/sparse": {"peak_over_entry_bytes": 300_000_000}},
    }
    cp = tmp_path / "capacity.json"
    cp.write_text(json.dumps(probe))
    tool.main(["ingest", str(cp), "-w"])
    m = sp.load()["measured"]["16"]
    assert m["pool_blocks"] == 2201 and m["free_at_prefill_peak_bytes"] == 300_000_000
    tool.main(["refresh", "-w"])
    p = sp.profile(sp.load(), "p300x2")
    assert p["status"] == "measured" and p["max_model_len"] == sp.floor_to_granularity((2201 - 2 - 15) * 128)
    assert sp.profile(sp.load(), "p300x2-long")["status"] == "estimate"
    del probe["copy_accuracy"]
    cp.write_text(json.dumps(probe))
    with expect_error(SystemExit, "probe not complete"):
        tool.main(["ingest", str(cp)])


def test_selected_precision_policy_loads():
    """The policy the adapter installs (GLM53_PRECISION) parses with tt/precision.py and is the one the data names."""
    from models.autoports.zai_org_glm_5_3_flash.tt.precision import PrecisionOverrides

    d = sp.load()
    o = PrecisionOverrides._load(str(sp.MODEL_DIR / d["precision"]["config"]))
    assert o.id == d["precision"]["id"] and o.kv_latent_dtype == d["precision"]["kv_latent_dtype"]


def test_default_precision_uses_serving_budget(monkeypatch):
    """Stage-8 review #4 P1: the generator's DRAM budget must come from serving_profiles.json for the default precision
    (GLM53_PRECISION unset). If the default policy id and the data file's precision id differ, the generator falls
    back to the stage-7 built-in budget and the default 1-slot context drops (112,640 instead of 280,576)."""
    from models.autoports.zai_org_glm_5_3_flash.tt.precision import PrecisionOverrides

    for k in ("GLM53_PRECISION", "GLM53_DRAM_RESERVE_MIB", "GLM53_BUDGET_FREE_BYTES", "GLM53_DSA_SHARD"):
        monkeypatch.delenv(k, raising=False)
    d = sp.load()
    assert PrecisionOverrides.from_env().id == d["precision"]["id"], "default policy id != serving_profiles.json id"
    budget = gen_mod._budget()
    assert budget["source"].startswith("serving_profiles.json"), budget["source"]
    assert budget["latent_dtype"] == d["precision"]["kv_latent_dtype"]
    monkeypatch.setattr(gen_mod, "BUDGET", budget)
    plan = gen_mod.plan_context(1)  # the standalone default (1,200 MiB reserve)
    assert plan["layout"] == "replicated" and plan["max_seq_len"] >= 280576, plan
    # one sizing rule: at the serving reserve, the generator accepts the data file's 1-slot value
    monkeypatch.setenv("GLM53_DRAM_RESERVE_MIB", str(int(d["budget"]["reserve_bytes"]) >> 20))
    monkeypatch.setattr(gen_mod, "BUDGET", gen_mod._budget())
    gen_mod.plan_context(1, sp.supported_context(d), serving=True)  # must not raise


def test_apply_precision_default(monkeypatch, expect_error):
    d = sp.load()
    monkeypatch.setenv("GLM53_PRECISION", "")
    gv.apply_precision_default(d)
    assert os.environ["GLM53_PRECISION"] == str(sp.MODEL_DIR / d["precision"]["config"])
    monkeypatch.setenv("GLM53_PRECISION", d["precision"]["config"])  # relative to the model dir, any cwd
    monkeypatch.chdir("/")
    gv.apply_precision_default(d)
    assert os.environ["GLM53_PRECISION"] == str(sp.MODEL_DIR / d["precision"]["config"])
    for spelling in ("none", "NONE", "shipped", "off", "0"):  # stage-7 policy: refused on the vLLM path
        monkeypatch.setenv("GLM53_PRECISION", spelling)
        with expect_error(ValueError, "does not serve"):
            gv.apply_precision_default(d)
    monkeypatch.setenv("GLM53_PRECISION", "")  # unset -> the selected policy id of the data file
    gv.apply_precision_default(d)
    assert gv._precision_id() == d["precision"]["id"]


def test_resolve_checkpoint_parent_overlay(monkeypatch, tmp_path, expect_error):
    """Parent snapshot (288 experts, no mask) -> overlay dir of symlinks + tt/e192_mask.json, at the pinned revision."""
    import huggingface_hub

    snap = tmp_path / "hub" / "snapshots" / "91b6921"
    (snap / "mtp").mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model-00001-of-00002.safetensors").write_bytes(b"x")
    seen = {}

    def fake_snapshot_download(repo, revision=None, local_files_only=False):
        seen.update(repo=repo, revision=revision, offline=local_files_only)
        return str(snap)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.delenv("GLM53_CKPT", raising=False)
    monkeypatch.setenv("GLM53_CKPT_OVERLAY_DIR", str(tmp_path / "overlay"))
    monkeypatch.setenv("GLM53_WEIGHTS_REVISION", "91b69215804f2d2036a8867f6ce44848e278dcda")
    root = Path(gv.resolve_checkpoint(SimpleNamespace(_name_or_path="zai-org/GLM-5.3-Flash")))
    assert seen == {
        "repo": "zai-org/GLM-5.3-Flash",
        "revision": "91b69215804f2d2036a8867f6ce44848e278dcda",
        "offline": True,
    }
    assert root == tmp_path / "overlay" / "91b6921" and os.environ["GLM53_CKPT"] == str(root)
    assert (root / "config.json").is_symlink() and (root / "mtp").is_symlink()
    assert (root / "e192_mask.json").read_bytes() == (sp.MODEL_DIR / "tt" / "e192_mask.json").read_bytes()
    monkeypatch.delenv("GLM53_CKPT")
    assert Path(gv.resolve_checkpoint(SimpleNamespace(_name_or_path="zai-org/GLM-5.3-Flash"))) == root  # idempotent
    monkeypatch.delenv("GLM53_CKPT")
    with expect_error(RuntimeError, "was made for"):
        gv.resolve_checkpoint(SimpleNamespace(_name_or_path="someone/other-model"))


def test_get_max_tokens_all_users(monkeypatch, expect_error):
    monkeypatch.delenv("GLM53_MAX_TOKENS_ALL_USERS", raising=False)
    monkeypatch.delenv("GLM53_VLLM_EXTRA_RESERVE_MIB", raising=False)
    d = sp.load()
    cls = gv.Glm5NextForConditionalGeneration
    for p in sp.profiles(d):
        S, L = p["max_num_seqs"], p["max_model_len"]
        t = cls.get_max_tokens_all_users(num_devices=4, max_model_len=L, max_num_seqs=S)
        assert t == sp.max_tokens_all_users(d, S)
        with expect_error(ValueError, "does not fit the DSA pool"):
            cls.get_max_tokens_all_users(num_devices=4, max_model_len=sp.max_model_len_for(d, S) + 2048, max_num_seqs=S)
    with expect_error(ValueError, "serves one"):
        cls.get_max_tokens_all_users(num_devices=8, max_num_seqs=1)


# ------------------------------------------------------------------------------------------------ adapter vs fake generator
class FakeGenerator:
    """Implements the Glm53Generator serving API of DESIGN.md section 9 on the host. Device tokens: the next token
    of slot s at position p is 1000 * (s + 1) + p."""

    def __init__(self, mesh, *, max_batch_size, max_seq_len, defer_caches=False, **kw):
        assert defer_caches
        # the adapter does the DRAM check from serving_profiles.json and fixes the layout the profiles were sized for
        assert kw == {"dsa_layout": sp.load()["dsa_layout"], "budget_check": False}, kw
        self.S, self.T = max_batch_size, max_seq_len
        self.model = SimpleNamespace(layers=[0] * 45, sampling=object())
        self.calls = []
        self.pos = [-1] * self.S
        self.page_rows = {}

    def bind_kv_cache(self, num_blocks):
        self.calls.append(("bind", num_blocks))
        self.kv = [("latent", i) for i in range(11)]
        return self.kv

    def warm_programs(self):
        self.calls.append(("warm",))

    def capture_traces(self):
        self.calls.append(("capture",))

    def program_cache_entries(self):
        return 7

    def prefill_forward(self, tokens, *, page_table, kv_cache, prompt_lens, slots, sampling_params):
        assert kv_cache is self.kv and page_table.shape[1] == self.T // 128
        self.calls.append(("prefill", list(slots), list(prompt_lens)))
        for s, n in zip(slots, prompt_lens):
            self.pos[s] = n
        if sampling_params is None:
            return torch.zeros(len(slots), 1, 8)
        return torch.tensor([1000 * (s + 1) + n - 1 for s, n in zip(slots, prompt_lens)])

    def decode_forward(self, tokens, start_pos, *, slots, sampling_params, reload_inputs, read_from_device, **kw):
        self.calls.append(("decode", list(slots), bool(reload_inputs), kw.get("fresh_slots")))
        if reload_inputs:
            for r, s in enumerate(slots):
                self.pos[s] = int(start_pos[r])
        out = {s: 1000 * (s + 1) + self.pos[s] for s in range(self.S) if self.pos[s] >= 0}
        self.pos = [p + 1 if p >= 0 else p for p in self.pos]
        if sampling_params is None:
            return torch.zeros(len(slots), 8)
        if read_from_device:
            return torch.tensor([out.get(s, 0) for s in slots])
        return dict(out)  # the "device handle"

    def read_decode_output(self, handle, async_read=True):
        return dict(handle), []

    def process_decode_output_host(self, host, slots):
        return torch.tensor([host.get(s, 0) for s in slots])


@pytest.fixture
def adapter(monkeypatch, tmp_path):
    monkeypatch.setattr(gen_mod, "Glm53Generator", FakeGenerator)
    monkeypatch.setenv("GLM53_CKPT", str(tmp_path))
    monkeypatch.setenv("GLM53_FAULTHANDLER", "0")
    monkeypatch.setenv("GLM53_PRECISION", "")  # restored at teardown (the adapter sets it)
    monkeypatch.delenv("GLM53_LAYERS", raising=False)
    mesh = SimpleNamespace(shape=(1, 4))
    a = gv.Glm5NextForConditionalGeneration.initialize_vllm_model(SimpleNamespace(), mesh, 4, max_seq_len=8000)
    assert a.T == 8192  # rounded up to 128, then to the 2048-token prefill chunk (functional_decoder assert)
    assert os.environ["GLM53_PRECISION"].endswith("selected_precision_config.json")
    kv = a.allocate_kv_cache((100, 1, 128, 512), torch.bfloat16, 1)
    a.warmup_model_prefill(kv_cache=kv, enable_trace=False, can_sample_on_device=True)
    a.warmup_model_decode(kv_cache=kv, enable_trace=False, can_sample_on_device=True)
    a.already_warmed_up_prefill = False
    a.warmup_model_prefill(kv_cache=kv, enable_trace=True, can_sample_on_device=True)
    a.warmup_model_decode(kv_cache=kv, enable_trace=True, can_sample_on_device=True)
    assert [c[0] for c in a._gen.calls] == ["bind", "warm", "capture"]
    return a, kv


def _sp(n):
    return SimpleNamespace(temperature=[0.0] * n, top_k=[1] * n, top_p=[1.0] * n, seed=[None] * n)


def test_adapter_prefill_decode_remap(adapter):
    a, kv = adapter
    pt = torch.tensor([[1, 2], [3, 0]], dtype=torch.int32)
    out = a.prefill_forward(torch.zeros(2, 300), pt, kv, [300, 129], empty_slots=[0, 1], sampling_params=_sp(2))
    assert out.tolist() == [[1299], [2128]]
    toks = torch.tensor([[1299], [2128], [0], [0]])
    pos = torch.tensor([300, 129, -1, -1])
    o = a.decode_forward(
        tokens=toks,
        start_pos=pos,
        page_table=pt,
        kv_cache=kv,
        sampling_params=_sp(4),
        reload_inputs=True,
        enable_trace=True,
        read_from_device=False,
    )
    o, ev = a.read_decode_output(o, async_read=True)
    assert a.process_decode_output_host(o).reshape(-1).tolist()[:2] == [1300, 2129]
    assert a._gen.calls[-1][3] == [0, 1]  # both slots were fresh (prefilled since the last decode)
    # request 0 finishes; the plugin condenses: row 0 reads vLLM slot 1
    o = a.decode_forward(
        tokens=torch.tensor([[2129], [0], [0], [0]]),
        start_pos=torch.tensor([130, -1, -1, -1]),
        page_table=pt[1:],
        kv_cache=kv,
        sampling_params=_sp(4),
        slot_remap=torch.tensor([1, 0, 2, 3]),
        reload_inputs=True,
        read_from_device=True,
    )
    assert o.reshape(-1).tolist()[0] == 2130  # model slot 1 still holds the request: no device state moved
    # a new request lands in vLLM slot 1 -> model slot 0 (the free one)
    a.prefill_forward(torch.zeros(1, 50), pt[:1], kv, [50], empty_slots=[1], sampling_params=_sp(1))
    assert a._gen.calls[-1][1] == [0]


def test_adapter_rejects_chunked_prefill_and_foreign_cache(adapter, expect_error):
    a, kv = adapter
    with expect_error(NotImplementedError, "chunked prefill"):
        a.prefill_forward(torch.zeros(1, 256), torch.zeros(1, 2, dtype=torch.int32), kv, [256], start_pos=[128])
    with expect_error(RuntimeError, "not the object allocate_kv_cache returned"):
        a.prefill_forward(torch.zeros(1, 256), torch.zeros(1, 2, dtype=torch.int32), [object()], [256])


def test_adapter_host_logits_mode(adapter):
    a, kv = adapter
    out = a.prefill_forward(torch.zeros(1, 10), torch.ones(1, 1, dtype=torch.int32), kv, [10], empty_slots=[2])
    assert out.shape == (1, 1, 8)
    o = a.decode_forward(tokens=torch.zeros(4, 1), start_pos=torch.tensor([-1, -1, 10, -1]), kv_cache=kv)
    assert isinstance(o, torch.Tensor) and o.shape == (4, 1, 8)


# ------------------------------------------------------------------------------------------------ generator serving API
def test_default_paths_unchanged():
    """The standalone generator / model keep today's defaults (readiness jobs import these files)."""
    import inspect

    import models.autoports.zai_org_glm_5_3_flash.tt.dsa as dsa_mod
    import models.autoports.zai_org_glm_5_3_flash.tt.model as model_mod

    g = inspect.signature(gen_mod.Glm53Generator.__init__).parameters
    assert g["defer_caches"].default is False
    m = inspect.signature(model_mod.Glm53Model.__init__).parameters
    assert m["allocate_dsa"].default is True and m["lockstep_sampling"].default is False
    assert inspect.signature(model_mod.Glm53Model.prefill).parameters["device_logits"].default is False
    assert inspect.signature(dsa_mod.GlmDSA.allocate_decode_state).parameters["num_blocks"].default is None
    assert "_warm" in gen_mod.Glm53Generator.__dict__  # the standalone warm-up + capture path is still there


def _bare_generator(S=4, T=8192, num_blocks=100, lens=(128, 2048)):
    g = object.__new__(gen_mod.Glm53Generator)
    g.max_batch_size, g.max_seq_len, g.num_blocks = S, T, num_blocks
    g.model = SimpleNamespace(prefill_lengths=lens)
    return g


def _dsa_path(start, n, length):
    last = start + length - 1
    return "flash" if start == 0 and last <= 2050 else ("sparse" if last > 2050 else "dense")


def test_warm_plans_cover_dsa_paths():
    """Every (chunk length, DSA path) that a chunk plan can produce is warmed (stage 7: cost-based chunk plans)."""
    assert _bare_generator()._warm_lengths() == [128, 2048, 256, 2050, 2176, 4096]
    assert _bare_generator(T=4096)._warm_lengths() == [128, 2048, 256, 2050, 2176, 4096]
    assert _bare_generator(num_blocks=20)._warm_lengths() == [128, 2048, 256, 2050, 2176]  # 4096 needs 33 blocks
    for lens in ((128, 2048), (128, 512, 2048)):
        g = _bare_generator(T=8192, lens=lens)
        covered = set()
        for L, sizes in g._warm_plans():
            c0 = 0
            for n in sizes:
                ln = min(n, L - c0)
                covered.add((n, _dsa_path(c0, n, ln)))
                c0 += ln
        want = {(n, "flash") for n in lens} | {(n, "dense") for n in lens if n < 2048} | {(n, "sparse") for n in lens}
        assert want <= covered, want - covered


class _Inputs:
    def __init__(self, S):
        self.positions = [-1] * S
        self.sets = 0

    def set_positions(self, p):
        self.positions = list(p)
        self.sets += 1

    def host_advance(self, n=1):
        self.positions = [p + n if p >= 0 else p for p in self.positions]


def test_decode_serving_reload_merge_and_order(monkeypatch):
    import ttnn

    S = 4
    g = _bare_generator(S=S)
    sampled = []
    g.model = SimpleNamespace(
        inputs=_Inputs(S),
        sampling=SimpleNamespace(sample=lambda *a, **k: sampled.append(k)),
        vocab=8,
    )
    g.mesh, g.trace_id, g.logits_out, g.tok_buf = None, 1, "logits", "tok_buf"
    g._fresh, g._serving_decoded, g._sstate = set(), True, None
    g.page_table = None
    written = {}
    monkeypatch.setattr(ttnn, "execute_trace", lambda *a, **k: None)
    g._write_tokens = lambda toks: written.setdefault("tok", toks)
    # device is one step ahead for slot 2 (continuing), slot 0 was just prefilled
    g._read_tokens = lambda: torch.tensor([0, 0, 77, 0])
    g._read_positions = lambda: torch.tensor([0, -1, 11, -1])
    g._fresh = {0}
    out = g._decode_serving(
        torch.tensor([5, 9]), torch.tensor([30, 10]), slots=[0, 2], sampling_params=object(), read_from_device=False
    )
    assert out == "tok_buf" and sampled  # device handle returned, sampler replayed
    assert written["tok"] == [5, 0, 77, 0]  # slot 2 keeps the device's token / position (host lags one step)
    assert g.model.inputs.positions == [31, -1, 12, -1]  # after host_advance

    class _Host:  # read / process: die-0 row, slot -> row order
        pass

    monkeypatch.setattr(ttnn, "get_device_tensors", lambda h: [h])
    monkeypatch.setattr(ttnn, "to_torch", lambda h: torch.tensor([100, 101, 102, 103]))
    assert g.process_decode_output_host(_Host(), [2, 0]).tolist() == [102, 100]
