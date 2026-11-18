# Hy4-flash serving front-end (vLLM 0.26 + TT plugin)

vLLM 0.26.0 and transformers 5.12.1 (the serving image) predate Hy4-flash: neither knows `model_type: hy_v4_flash`
(transformers PR 50903, vLLM PR 68296, both unmerged on 2026-11-18). This directory supplies what they lack; put it
first on `PYTHONPATH` of the vLLM server (every process loads its `sitecustomize.py`).

| file | what |
|---|---|
| `sitecustomize.py` | post-import hooks, run in every process: registers the config with `AutoConfig` when transformers' auto-config module loads, and the `hy_v4` reasoning / tool parsers as lazy vLLM parsers when `vllm.reasoning` / `vllm.tool_parsers` load. Imports nothing eagerly. Also installs the hybrid-MoE overlay with `HYB_ENABLE=1` (only one `sitecustomize` is ever loaded). |
| `hy4_config.py` | `HYV4FlashConfig` (composite: `text_config`, `vision_config`, drafter `dfly_*` fields at the top) and `HYV4FlashTextConfig`; every checkpoint field kept verbatim. vLLM derives: max_model_len 1,048,576, vocab 120,832, hidden 3072, `is_multimodal_model` False (with a text-only adapter), eos 120025. |
| `hy_v4_reasoning_parser.py` | vLLM PR 68296's parser, imports adapted to 0.26. `reasoning_effort` (chat_template_kwargs, then top level, default `high`) picks thinking vs `no_think`. |
| `hy_v4_tool_parser.py` | vLLM PR 68296's parser (native `<tool_calls>` format, schema-typed values, streaming), 0.26 adapter: no xgrammar structural tag; `tool_choice` required / named take vLLM's generic JSON-schema path. |
| `tests/` | `test_parsers.py` (registration, config, reasoning + tool parsing, streaming at 1 / 2-3 / 6 tokens per step equals non-streaming), `test_template.py` / `render_chat_template.py` (the chat template's structure). |

Serve flags: `--reasoning-parser hy_v4 --tool-call-parser hy_v4 --enable-auto-tool-choice` (tt-model yaml:
`serve.capabilities.reasoning_parser / tool_parser: hy_v4`), `EXTRA_MODELS_DIR=<port>/vllm_bundle`,
`PYTHONPATH=<port>/serving:/opt/tt-metal`. The adapter class must be text only (`supports_multimodal = False`) and
must not inherit a multimodal processor registration that rejects this config (the Inkling port hit that through
the Qwen3.5 adapter base).

Run the tests (no device; the image has no pytest: mount a pure-python copy):

```
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e HF_HUB_OFFLINE=1 --entrypoint "" \
  -v $PWD/serving:/serving -v <dir with pytest, pluggy, iniconfig>:/pt -v ~/.cache/huggingface:/hf -e HF_HOME=/hf \
  -e PYTHONPATH=/serving:/pt -e HY4_TOKENIZER=tencent/Hy4-flash <image> python -m pytest -q /serving/tests
```

## Chat template facts

- Turn = `<｜hy_start:opensource｜>{role}<｜hy_middle:opensource｜>{content}<｜hy_end:opensource｜>`; bos 120000,
  eos / end of turn 120025, pad 120002. The template writes the turn tokens itself; the tokenizer adds no BOS.
- The system turn always exists and ends with `<｜reasoning_mode:opensource｜>reasoning_effort:{high|max|no_think}`.
  Only `high` (default), `max`, `no_think` are accepted; any other value raises in the template (an OpenAI
  `reasoning_effort: "low"/"medium"` request would fail).
- Generation prompt: `...assistant<｜hy_middle:opensource｜><think:opensource>` (thinking) or
  `...<think:opensource></think:opensource>` (no_think). Earlier assistant turns are rendered with an empty think
  block (reasoning dropped) unless tools are present (`preserved_thinking` defaults to true with tools).
- Tool calls: `<tool_calls:opensource><tool_call:opensource>NAME<arg_key:opensource>K</arg_key:opensource>
  <arg_value:opensource>V</arg_value:opensource>...</tool_call:opensource>...</tool_calls:opensource>`; tool results in
  `<tool_response:opensource>...</tool_response:opensource>`. All structural markers are single tokens (120029-120043).
- Recommended sampling (generation_config.json, applied by vLLM by default): temperature 0.9, top_p 1.0, top_k off.
