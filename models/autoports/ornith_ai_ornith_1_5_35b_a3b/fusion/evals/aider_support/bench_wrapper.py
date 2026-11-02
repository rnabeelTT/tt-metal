"""Run aider's benchmark/benchmark.py unchanged, after registering litellm model
metadata (context window) for a model aider does not know.

Without this, aider sees max_input_tokens=0 for an unknown openai/<id> model and
caps chat history at 1024 tokens, so try 2 triggers an extra summarisation call
and a summarised history. With metadata registered it behaves like any
large-context model on the leaderboard (history cap 8192 tokens).

Usage: python bench_wrapper.py <metadata.json> <benchmark.py args...>
"""
import runpy
import sys

from aider import models

meta = sys.argv[1]
loaded = models.register_litellm_models([meta])
print("bench_wrapper: registered model metadata from", loaded, flush=True)
sys.argv = ["/aider/benchmark/benchmark.py"] + sys.argv[2:]
sys.path.insert(0, "/aider/benchmark")
runpy.run_path("/aider/benchmark/benchmark.py", run_name="__main__")
