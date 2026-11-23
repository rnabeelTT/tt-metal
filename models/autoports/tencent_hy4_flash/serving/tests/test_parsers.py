# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-flash `hy_v4` reasoning / tool parsers on vLLM 0.26 (CPU only, no device).

Run inside the serving image with this test's `serving/` directory on PYTHONPATH (so `sitecustomize.py` registers the
config and the lazy parsers) and HY4_TOKENIZER pointing at a directory or repo with the checkpoint's tokenizer:

  docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e HF_HUB_OFFLINE=1 --entrypoint "" \\
    -v <serving>:/serving -v <hf cache or tokenizer dir>:/w -e PYTHONPATH=/serving -e HY4_TOKENIZER=/w/... \\
    <image> python -m pytest -q /serving/tests/test_parsers.py

Streaming is checked the way vLLM drives it: the model output is tokenized and fed one token per step (plus a
multi-token-per-step variant, as speculative decoding delivers), and the accumulated deltas must equal the
non-streaming result.
"""

import json
import os

import pytest

TOK = os.environ.get("HY4_TOKENIZER", "tencent/Hy4-flash")
S = ":opensource"
THINK, ETHINK = f"<think{S}>", f"</think{S}>"
TCS, TCE, TC, ETC = f"<tool_calls{S}>", f"</tool_calls{S}>", f"<tool_call{S}>", f"</tool_call{S}>"
AK, EAK, AV, EAV = f"<arg_key{S}>", f"</arg_key{S}>", f"<arg_value{S}>", f"</arg_value{S}>"

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Weather in a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                        "days": {"type": "integer"},
                                                        "detailed": {"type": "boolean"},
                                                        "units": {"type": "object"}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "search", "description": "Web search.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]


@pytest.fixture(scope="module")
def tokenizer():
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer(TOK)


def _request(tools=TOOLS, effort=None, tool_choice="auto"):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    kw = {"chat_template_kwargs": {"reasoning_effort": effort}} if effort else {}
    if tools:
        kw.update(tools=tools, tool_choice=tool_choice)
    return ChatCompletionRequest(model="hy4", messages=[{"role": "user", "content": "hi"}], **kw)


def _steps(tokenizer, text, per_step):
    """(prev_text, cur_text, delta_text, prev_ids, cur_ids, delta_ids) per step, decoding like vLLM's detokenizer
    (full decode of the prefix, special tokens kept: the parsers see them as text)."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    prev_ids, prev_text = [], ""
    for i in range(0, len(ids), per_step):
        delta_ids = ids[i:i + per_step]
        cur_ids = prev_ids + delta_ids
        cur_text = tokenizer.decode(cur_ids, skip_special_tokens=False)
        yield prev_text, cur_text, cur_text[len(prev_text):], prev_ids, cur_ids, delta_ids
        prev_ids, prev_text = cur_ids, cur_text


# ------------------------------------------------------------------------------------------------ registration


def test_lazy_registration():
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    assert "hy_v4" in ReasoningParserManager.list_registered()
    assert ToolParserManager.get_tool_parser("hy_v4").__name__ == "HYV4ToolParser"
    assert ReasoningParserManager.get_reasoning_parser("hy_v4").__name__ == "HYV4ReasoningParser"


def test_config_loads():
    from vllm.transformers_utils.config import get_config

    if not os.path.isdir(TOK):
        pytest.skip("config check needs a local checkpoint directory")
    c = get_config(TOK, trust_remote_code=False)
    t = c.get_text_config()
    assert (t.num_hidden_layers, t.hidden_size, t.n_routed_experts, t.max_position_embeddings) == (42, 3072, 512,
                                                                                                    1048576)


# ------------------------------------------------------------------------------------------------ reasoning


def _rparser(tokenizer, effort=None):
    from vllm.reasoning import ReasoningParserManager

    cls = ReasoningParserManager.get_reasoning_parser("hy_v4")
    return cls(tokenizer, chat_template_kwargs={"reasoning_effort": effort} if effort else {})


@pytest.mark.parametrize("effort", [None, "high"])
def test_reasoning_thinking(tokenizer, effort):
    p = _rparser(tokenizer, effort)
    req = _request(tools=None)
    assert p.extract_reasoning(f"Let me think.{ETHINK}The answer is 391.", req) == ("Let me think.",
                                                                                    "The answer is 391.")
    assert p.extract_reasoning("still thinking", req) == ("still thinking", None)  # truncated
    assert p.reasoning_end_str == ETHINK and p.reasoning_start_str == THINK
    end_id = tokenizer.convert_tokens_to_ids(ETHINK)
    assert p.is_reasoning_end([1, 2, end_id]) and not p.is_reasoning_end([1, 2, 3])
    assert p.extract_content_ids([5, end_id, 7, 8]) == [7, 8]


def test_reasoning_no_think(tokenizer):
    p = _rparser(tokenizer, "no_think")
    req = _request(tools=None)
    assert p.extract_reasoning("The answer is 391.", req) == (None, "The answer is 391.")
    assert p.is_reasoning_end([1])


@pytest.mark.parametrize("per_step", [1, 3, 6])
@pytest.mark.parametrize("effort", ["high", "no_think"])
def test_reasoning_streaming_matches(tokenizer, per_step, effort):
    p = _rparser(tokenizer, effort)
    text = (f"First, 17*20 = 340; then 17*3 = 51. Sum: 391.{ETHINK}**391**: seventeen times twenty-three."
            if effort != "no_think" else "**391**: seventeen times twenty-three.")
    want = p.extract_reasoning(text, _request(tools=None))
    reasoning, content = "", ""
    for args in _steps(tokenizer, text, per_step):
        d = p.extract_reasoning_streaming(*args)
        if d is not None:
            reasoning += d.reasoning or ""
            content += d.content or ""
    assert (reasoning or None, content or None) == want


# ------------------------------------------------------------------------------------------------ tools


def _tparser(tokenizer, tools=TOOLS):
    from vllm.tool_parsers import ToolParserManager

    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest  # noqa: F401

    req = _request(tools=tools)
    return ToolParserManager.get_tool_parser("hy_v4")(tokenizer, req.tools), req


CALL_TEXT = (f"I'll check both.{TCS}{TC}get_weather{AK}city{EAK}{AV}Paris{EAV}{AK}days{EAK}{AV}3{EAV}"
             f"{AK}detailed{EAK}{AV}true{EAV}{AK}units{EAK}{AV}{{\"temp\": \"c\"}}{EAV}{ETC}"
             f"{TC}search{AK}query{EAK}{AV}Paris <events> & \"news\"{EAV}{ETC}{TCE}")
WANT_CALLS = [("get_weather", {"city": "Paris", "days": 3, "detailed": True, "units": {"temp": "c"}}),
              ("search", {"query": 'Paris <events> & "news"'})]


def test_tool_calls_nonstreaming(tokenizer):
    p, req = _tparser(tokenizer)
    r = p.extract_tool_calls(CALL_TEXT, req)
    assert r.tools_called
    assert (r.content or "").strip() == "I'll check both."
    got = [(c.function.name, json.loads(c.function.arguments)) for c in r.tool_calls]
    assert got == WANT_CALLS


def test_no_tool_call_is_content(tokenizer):
    p, req = _tparser(tokenizer)
    r = p.extract_tool_calls("Plain answer, with a < sign and <b>tags</b>.", req)
    assert not r.tools_called and r.content == "Plain answer, with a < sign and <b>tags</b>."


def test_malformed_falls_back_to_content(tokenizer):
    p, req = _tparser(tokenizer)
    bad = f"{TCS}{TC}get_weather{AK}city{EAK}{AV}Paris"  # truncated mid-call (max_tokens)
    r = p.extract_tool_calls(bad, req)
    assert not r.tools_called


@pytest.mark.parametrize("per_step", [1, 2, 6])
def test_tool_calls_streaming_matches(tokenizer, per_step):
    p, req = _tparser(tokenizer)
    content, calls = "", {}
    for args in _steps(tokenizer, CALL_TEXT, per_step):
        d = p.extract_tool_calls_streaming(*args, request=req)
        if d is None:
            continue
        content += d.content or ""
        for tc in d.tool_calls or []:
            c = calls.setdefault(tc.index, {"name": None, "args": "", "ids": []})
            if tc.id:
                c["ids"].append(tc.id)
            if tc.function is not None:
                if tc.function.name:
                    c["name"] = tc.function.name
                c["args"] += tc.function.arguments or ""
    assert content.strip() == "I'll check both."
    got = [(calls[i]["name"], json.loads(calls[i]["args"])) for i in sorted(calls)]
    assert got == WANT_CALLS
    assert all(len(calls[i]["ids"]) == 1 for i in calls)  # one id per call, on its first chunk


def test_reasoning_then_tool_call_streaming(tokenizer):
    """The serving order: the reasoning parser splits off the thinking, the tool parser sees the content part."""
    rp = _rparser(tokenizer, "high")
    text = f"Need the weather.{ETHINK}{CALL_TEXT}"
    reasoning, content = rp.extract_reasoning(text, _request())
    assert reasoning == "Need the weather."
    p, req = _tparser(tokenizer)
    r = p.extract_tool_calls(content, req)
    assert [c.function.name for c in r.tool_calls] == ["get_weather", "search"]
