#!/bin/bash
# after the packages are built: pause the eval queue at a run boundary, then gate + publish o122 and q122
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH
R=$HOME/ornith-fusion/runs
while systemctl --user is-active -q pkgfix; do sleep 60; done
grep -q "o122 package exit 0" $R/pkg_mmfix.log && grep -q "q122 package exit 0" $R/pkg_mmfix.log || { echo "packaging failed"; cat $R/pkg_mmfix.log; exit 1; }
touch $R/evalq.pause; echo "pause requested $(date +%T)"
exec $R/mmfix_release.sh o122 q122
