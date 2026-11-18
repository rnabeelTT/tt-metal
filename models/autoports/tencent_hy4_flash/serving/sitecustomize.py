# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Loaded by every Python process started with this directory on PYTHONPATH (the vLLM API server and its EngineCore
subprocess). Registers the Hy4-flash config (`hy4_config.register`) the moment transformers' auto-config module is
imported, so `AutoConfig` / vLLM's `get_config` resolve `model_type: hy_v4_flash` in every process (the EngineCore
unpickles the config object too). Nothing is imported eagerly: processes that never touch transformers pay nothing.

It also registers the `hy_v4` reasoning and tool parsers (this directory's `hy_v4_reasoning_parser.py`,
`hy_v4_tool_parser.py`, ported from vLLM PR 68296) as lazy vLLM parsers when `vllm.reasoning` / `vllm.tool_parsers`
are imported, so `--reasoning-parser hy_v4 --tool-call-parser hy_v4` work in every process without plugin flags.

With HYB_ENABLE=1 it also installs the hybrid-MoE overlay, as models/experimental/hybrid_moe/serve/sitecustomize.py
does (only one sitecustomize module is ever loaded)."""

import importlib.abc
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _here_on_path() -> None:
    if _HERE not in sys.path:
        sys.path.append(_HERE)


def _register_config(_module) -> None:
    _here_on_path()
    try:
        import hy4_config

        hy4_config.register()
    except Exception as e:  # noqa: BLE001 - surface loudly: without it vLLM cannot read the checkpoint's config
        sys.stderr.write(f"[hy4] config registration FAILED in pid {os.getpid()}: {e!r}\n")
        raise


def _register_reasoning_parser(module) -> None:
    # lazy: the parser module (and its vLLM imports) loads only when "hy_v4" is asked for
    _here_on_path()
    module.ReasoningParserManager.register_lazy_module("hy_v4", "hy_v4_reasoning_parser", "HYV4ReasoningParser")


def _register_tool_parser(module) -> None:
    _here_on_path()
    module.ToolParserManager.register_lazy_module("hy_v4", "hy_v4_tool_parser", "HYV4ToolParser")


# module -> callback run right after that module has executed (each at most once per process)
_HOOKS = {
    "transformers.models.auto.configuration_auto": _register_config,
    "vllm.reasoning": _register_reasoning_parser,
    "vllm.tool_parsers": _register_tool_parser,
}


class _PostImportFinder(importlib.abc.MetaPathFinder):
    """Runs `_HOOKS[name]` once module `name` has executed (wraps its loader's exec_module)."""

    def find_spec(self, fullname, path, target=None):
        hook = _HOOKS.pop(fullname, None)
        if hook is None:
            return None
        sys.meta_path.remove(self)  # resolve the real spec without recursing into ourselves
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            if _HOOKS:
                sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_hook(module):
            exec_module(module)
            hook(module)

        spec.loader.exec_module = exec_and_hook
        return spec


for _name in [n for n in _HOOKS if n in sys.modules]:
    _HOOKS.pop(_name)(sys.modules[_name])
if _HOOKS:
    sys.meta_path.insert(0, _PostImportFinder())

if os.environ.get("HYB_ENABLE") == "1":
    sys.path.insert(0, os.environ.get("HYB_POC_DIR", "/poc"))
    try:
        import hybrid

        hybrid.install()
        sys.stderr.write(f"[hybrid] installed in pid {os.getpid()}\n")
    except Exception as e:  # noqa: BLE001 - surface loudly, then re-raise
        sys.stderr.write(f"[hybrid] install FAILED in pid {os.getpid()}: {e!r}\n")
        raise
