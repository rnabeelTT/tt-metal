#!/bin/bash
# Ornith-1.5-122B weights (bf16, ~250 GB) into the HF hub cache, pinned revision.
export PATH=$HOME/.local/bin:$PATH
echo "download start $(date +%T)"
hf download ornith-ai/Ornith-1.5-122B --revision 7681ef608624e7532d8164128bcffdbdbaf1b1af > $HOME/ornith-fusion/runs/o122_download.hf.log 2>&1
echo "download exit $? $(date +%T)"
du -shL $HOME/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-122B/snapshots/7681ef608624e7532d8164128bcffdbdbaf1b1af
