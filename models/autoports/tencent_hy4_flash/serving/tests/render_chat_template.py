# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Render Hy4-flash chat-template examples with the image's transformers (no device).

usage (inside the serving image, this directory's parent on PYTHONPATH for the config hook):
  python render_chat_template.py <model dir or repo id> [out.txt]

Prints each rendered prompt (repr, so the special tokens and newlines are visible) and its token count, and checks the
facts the parsers and the engine rely on (asserted):
  - the generation prompt ends with `<think:opensource>` when reasoning_effort is high (default) and does
    NOT open a think block for no_think;
  - bos is `<｜hy_start:opensource｜>` (120000), the end of turn is `<｜hy_end:opensource｜>` (120025, the eos);
  - every structural token the parsers use is a single vocabulary token.
"""

import sys

from transformers import AutoTokenizer

STRUCTURAL = [
    "<think:opensource>", "</think:opensource>", "<tool_calls:opensource>", "</tool_calls:opensource>",
    "<tool_call:opensource>", "</tool_call:opensource>", "<arg_key:opensource>", "</arg_key:opensource>",
    "<arg_value:opensource>", "</arg_value:opensource>", "<tool_response:opensource>", "</tool_response:opensource>",
    "<｜hy_start:opensource｜>", "<｜hy_middle:opensource｜>", "<｜hy_end:opensource｜>",
    "<｜reasoning_mode:opensource｜>",
]

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                        "unit": {"type": "string", "enum": ["c", "f"]}},
                       "required": ["city"]},
    },
}]


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "tencent/Hy4-flash"
    out = open(sys.argv[2], "w") if len(sys.argv) > 2 else sys.stdout
    tok = AutoTokenizer.from_pretrained(src)
    vocab = tok.get_vocab()
    for t in STRUCTURAL:
        assert t in vocab, f"{t} is not a single token"
        ids = tok.encode(t, add_special_tokens=False)
        assert len(ids) == 1, (t, ids)
    print(f"structural tokens: {[(t, vocab[t]) for t in STRUCTURAL]}", file=out)
    print(f"bos {tok.bos_token!r} {tok.bos_token_id}  eos {tok.eos_token!r} {tok.eos_token_id}  "
          f"pad {tok.pad_token!r} {tok.pad_token_id}  vocab {len(tok)}", file=out)
    assert tok.bos_token_id == 120000 and tok.eos_token_id == 120025

    user = [{"role": "user", "content": "What is 17 * 23?"}]
    sysu = [{"role": "system", "content": "You are a terse assistant."}] + user
    multi = sysu + [
        {"role": "assistant", "content": "391", "reasoning_content": "17*23 = 17*20 + 17*3 = 340 + 51 = 391."},
        {"role": "user", "content": "And the weather in Paris?"},
    ]
    tool_turn = [
        {"role": "user", "content": "Weather in Paris, in celsius?"},
        {"role": "assistant", "content": "", "reasoning_content": "Call the tool.",
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "get_weather", "arguments": {"city": "Paris", "unit": "c"}}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c": 14, "sky": "overcast"}'},
    ]
    cases = [
        ("user, default effort (high)", user, {}, None),
        ("user, no_think", user, {"reasoning_effort": "no_think"}, None),
        ("system + user, high", sysu, {"reasoning_effort": "high"}, None),
        ("multi-turn, high (earlier reasoning dropped?)", multi, {}, None),
        ("tools, high", user, {}, TOOLS),
        ("tool call + tool response, high", tool_turn, {}, TOOLS),
        ("tool call + tool response, no_think", tool_turn, {"reasoning_effort": "no_think"}, TOOLS),
    ]
    for name, msgs, kw, tools in cases:
        text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True, **kw)
        n = len(tok.encode(text, add_special_tokens=False))
        print(f"\n=== {name}: {n} tokens\n{text!r}", file=out)
        think_open = text.rstrip().endswith("<think:opensource>")
        if kw.get("reasoning_effort") == "no_think":
            assert not think_open, "no_think must not open a think block"
        else:
            assert think_open, "thinking modes end the generation prompt with <think:opensource>"
    print("\nall template checks passed", file=out)


if __name__ == "__main__":
    main()
