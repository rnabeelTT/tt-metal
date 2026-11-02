#!/bin/bash
# GSM8K (gsm8k_cot_llama_hf: gsm8k_cot_llama with the namespaced dataset id; the stock task fails to load) on both
# Qwen3-Coder-Next bundles, p300x2 as shipped, same settings as the other evals.
export PATH=$HOME/ornith-fusion/bin:$PATH MODEL_ID=Qwen/Qwen3-Coder-Next
T=$HOME/tt-metal-codernext/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
R=$HOME/ornith-fusion/runs
for cfg in "card_cnnew_p300x2:$HOME/tt-metal-codernext/build/qwen3-coder-next-blackhole-fast/tt_kernel_manifest.json:qwen3-coder-next-blackhole-fast" "card_cnpub_p300x2:raahemnabeel/qwen3-coder-next:qwen3-coder-next-blackhole"; do
  IFS=: read L B N <<< "$cfg"; C=tt-model-$N-p300x2
  docker rm -f $C >/dev/null 2>&1
  tt-model serve "$B" --profile p300x2 --detach --port 20000 --no-update-check > $R/$L/serve_gsm8k.log 2>&1
  t0=$(date +%s); until curl -sf localhost:20000/v1/models >/dev/null || [ $(( $(date +%s) - t0 )) -gt 3600 ]; do sleep 10; done
  echo "$L: up after $(( $(date +%s) - t0 )) s"
  PORT=20000 RUNS=$R $T/eval_coder.sh $L gsm8k_cot_llama_hf 2>&1 | grep -E "^==|\|.*(exact_match)"
  docker rm -f $C > /dev/null
done
echo "cn gsm8k done $(date +%T)"
