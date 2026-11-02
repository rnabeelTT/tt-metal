#!/bin/bash
# ref_chain.sh: original-card evals in priority order (user asked about Ornith-1.5-122B first, 2026-10-28).
cd $HOME/ornith-fusion/runs
./ref_evals2.sh o122 >> ref_evals2.log 2>&1
./ref_evals.sh orn q122 >> ref_evals.log 2>&1
./ref_evals2.sh cn >> ref_evals2.log 2>&1
