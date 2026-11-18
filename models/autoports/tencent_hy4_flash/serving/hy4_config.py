# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Configuration classes for tencent/Hy4-flash (`model_type` `hy_v4_flash`) for transformers 5.12 / vLLM 0.26.

Neither the image's transformers (5.12.1) nor its vLLM (0.26.0) knows `hy_v4_flash` (support is in transformers PR
50903 and vLLM PR 68296, both unmerged). vLLM resolves the checkpoint's config through `AutoConfig`, so this module
registers a config class under that model type (`register()`, called from `serving/sitecustomize.py` in every
process). The classes only carry the checkpoint's fields; they do not validate or rename anything:

- `HYV4FlashConfig` is the composite config of the checkpoint (`text_config`, `vision_config`, plus the drafter's
  `dfly_*` and `speculative_config` fields at the top level). vLLM reads text dimensions through `get_text_config()`.
- `HYV4FlashTextConfig` holds `text_config` as attributes (every key of the checkpoint's dict is kept).

The serving architecture is text only (`HYV4FlashForCausalLM`, see `vllm_bundle/hy4_flash/vllm_metadata.json`): the
top-level `architectures` is rewritten by `--hf-overrides '{"architectures": ["HYV4FlashForCausalLM"]}'`.
"""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig

MODEL_TYPE = "hy_v4_flash"
TEXT_MODEL_TYPE = "hy_v4_flash_text"
VISION_MODEL_TYPE = "hy_v4_flash_vision"

# Fields PretrainedConfig.__init__ consumes itself; everything else is set as a plain attribute.
_BASE_KEYS = ("bos_token_id", "eos_token_id", "pad_token_id", "tie_word_embeddings", "dtype", "torch_dtype",
              "architectures", "transformers_version")


def _split(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    base = {k: kwargs.pop(k) for k in _BASE_KEYS if k in kwargs}
    return base, kwargs


class HYV4FlashTextConfig(PretrainedConfig):
    """The language backbone's config (the checkpoint's `text_config`), every field kept verbatim."""

    model_type = TEXT_MODEL_TYPE
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, **kwargs: Any) -> None:
        base, rest = _split(dict(kwargs))
        rest.pop("model_type", None)
        for k, v in rest.items():
            setattr(self, k, v)
        # vLLM derives head geometry from these when they exist; MLA layers have no GQA groups.
        if not hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = getattr(self, "num_attention_heads", None)
        base.setdefault("tie_word_embeddings", False)
        super().__init__(**base)

    # vLLM / transformers helpers look these up by name
    @property
    def num_experts(self) -> int | None:
        return getattr(self, "n_routed_experts", None)


class HYV4FlashVisionConfig(PretrainedConfig):
    """The ViT's config (unused while serving is text only), every field kept verbatim."""

    model_type = VISION_MODEL_TYPE
    base_config_key = "vision_config"

    def __init__(self, **kwargs: Any) -> None:
        base, rest = _split(dict(kwargs))
        rest.pop("model_type", None)
        for k, v in rest.items():
            setattr(self, k, v)
        super().__init__(**base)


class HYV4FlashConfig(PretrainedConfig):
    """The checkpoint's composite config: `text_config`, `vision_config` and the top-level drafter fields."""

    model_type = MODEL_TYPE
    sub_configs = {"text_config": HYV4FlashTextConfig, "vision_config": HYV4FlashVisionConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, text_config: dict | PretrainedConfig | None = None,
                 vision_config: dict | PretrainedConfig | None = None, **kwargs: Any) -> None:
        if isinstance(text_config, dict) or text_config is None:
            text_config = HYV4FlashTextConfig(**(text_config or {}))
        if isinstance(vision_config, dict):
            vision_config = HYV4FlashVisionConfig(**vision_config)
        self.text_config = text_config
        self.vision_config = vision_config
        base, rest = _split(dict(kwargs))
        rest.pop("model_type", None)
        for k, v in rest.items():
            setattr(self, k, v)
        for k in ("bos_token_id", "eos_token_id", "pad_token_id"):
            base.setdefault(k, getattr(text_config, k, None))
        base.setdefault("tie_word_embeddings", False)
        super().__init__(**base)

    def get_text_config(self, *args: Any, **kwargs: Any) -> PretrainedConfig:
        return self.text_config

    # Text dimensions at the top level too: some vLLM paths read hf_config directly.
    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") or name in ("text_config", "vision_config"):
            raise AttributeError(name)
        tc = self.__dict__.get("text_config")
        if tc is not None and name in tc.__dict__:
            return tc.__dict__[name]
        raise AttributeError(name)


_registered = False


def register() -> None:
    """Make `AutoConfig` (and with it vLLM's config loader) resolve `hy_v4_flash`. Idempotent."""
    global _registered
    if _registered:
        return
    from transformers import AutoConfig

    for mt, cls in ((MODEL_TYPE, HYV4FlashConfig), (TEXT_MODEL_TYPE, HYV4FlashTextConfig),
                    (VISION_MODEL_TYPE, HYV4FlashVisionConfig)):
        try:
            AutoConfig.register(mt, cls)
        except ValueError:  # already registered (e.g. a transformers that knows the model): keep its class
            pass
    _registered = True
