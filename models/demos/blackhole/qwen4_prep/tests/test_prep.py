# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the Qwen4 prep tools (no device, no network).

    python3 -m pytest models/demos/blackhole/qwen4_prep/tests -q

The mini-reference test needs torch and transformers >= 5.16.1 and is skipped without them.
"""

import json
import sys
from pathlib import Path

import pytest

PREP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PREP))

import sizing  # noqa: E402
import triage  # noqa: E402


def _total_params(config_name: str) -> int:
    arch = sizing.Arch.from_config(json.loads((PREP / config_name).read_text()))
    return sum(sizing.breakdown(arch).values())


@pytest.mark.parametrize(
    "config_name, checkpoint_bytes",
    [
        # model.safetensors.index.json metadata.total_size of the released bf16 checkpoints
        ("flash_next_config.json", 359_999_963_128),  # Qwen/Qwen3.8-Flash-Next (qwen4_exp)
        ("qwen38_27b_config.json", 55_562_855_904),  # Qwen/Qwen3.8-27B (qwen3_5, dense)
    ],
)
def test_sizing_matches_released_checkpoints(config_name, checkpoint_bytes):
    params = _total_params(config_name)
    assert abs(2 * params - checkpoint_bytes) / checkpoint_bytes < 1e-3


def test_ple_table_is_base_times_width():
    arch = sizing.Arch.from_config(json.loads((PREP / "flash_next_config.json").read_text()))
    assert sizing.ple_table_params(arch) == 20_000_000 * 2560


def test_flash_next_fits_four_chips_with_ple_off_device():
    arch = sizing.Arch.from_config(json.loads((PREP / "flash_next_config.json").read_text()))
    parts = sizing.breakdown(arch)
    weights = sum(sizing.device_bytes(arch, parts, sizing.Precision(), mtp=True, vision=True).values()) / sizing.GB
    usable = 4 * (sizing.CHIP_DRAM_GB - sizing.CHIP_RESERVE_GB)
    assert 60 < weights < usable  # ~75.6 GB; the upstream BF4 expert cache alone is ~69 GB
    table = sizing.ple_table_params(arch) * sizing.BYTES["bf16"] / sizing.GB
    assert table > usable - weights


def test_triage_baseline_is_self_consistent():
    config = json.loads((PREP / "flash_next_config.json").read_text())
    tensors = json.loads((PREP / "flash_next_tensors.json").read_text())
    assert triage.diff_config(config, config) == []
    assert triage.diff_tensors(tensors, tensors) == []
    assert tensors["model.language_model.layers.N.self_attn.q_proj.weight"]["shape"] == [12288, 2560]  # gate packed
    assert tensors["model.language_model.layers.N.mlp.experts.gate_up_proj"]["shape"] == [512, 1280, 2560]
    assert triage.pattern("model.language_model.layers.17.ple.ple_embedding.ngram_embedding.shard_3.weight") == (
        "model.language_model.layers.N.ple.ple_embedding.ngram_embedding.shard_K.weight"
    )


def test_triage_flags_dimension_changes():
    config = json.loads((PREP / "flash_next_config.json").read_text())
    changed = json.loads(json.dumps(config))
    changed["text_config"]["hidden_size"] = 2048
    changed["text_config"]["new_block_flag"] = True
    lines = triage.diff_config(changed, config)
    assert any(line.startswith("  CHANGED  hidden_size") for line in lines)
    assert any(line.startswith("  NEW      new_block_flag") for line in lines)


def test_mini_reference_decode_matches_prefill(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("transformers.models.qwen4_exp", reason=f"transformers {transformers.__version__}")
    import mini_reference

    monkeypatch.setattr(
        sys,
        "argv",
        ["mini_reference", "--preset", "tiny", "--prompt-len", "96", "--decode-steps", "4", "--out", str(tmp_path)],
    )
    mini_reference.main()
    out = capsys.readouterr().out
    err = float(out.split("max rel err ")[1].split()[0])
    assert err < 1e-4, out
    assert (tmp_path / "golden.pt").exists() and (tmp_path / "model.safetensors").exists()
