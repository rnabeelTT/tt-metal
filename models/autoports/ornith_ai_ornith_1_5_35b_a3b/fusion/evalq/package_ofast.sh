#!/bin/bash
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH
cd $HOME/tt-metal-bringups && tt-model package --container models/autoports/ornith_ai_ornith_1_5_35b_a3b/tt-model-fast.yaml
echo "exit $?"
