#!/bin/bash
# stage-3 review #4 phase 1: the A/B probe, then the conv-op test (one ticket each)
S=~/glm53-flash-bringup/scripts/s3r4_one.sh
$S ab; $S conv
echo "$(date -Is) phase1 done" >> ~/glm53-flash-bringup/logs/s3r4.status
