#!/bin/bash
# Build pinx.so on the host (the serving image has no compiler) against the image's tt-metalium headers
# (TT_SDK: headers + libtt_metal.so copied out of the image) plus tt_stl / third-party headers that the image lacks.
set -e
cd "$(dirname "$(readlink -f "$0")")"
TT_SDK=${TT_SDK:-/tmp/claude-1003/ttsdk}
INSTALL=${TT_INSTALL:-$HOME/tt-metal-a0fb0661-install/include}
FORK=${FORK_DIR:-$HOME/FORK-tt-metal}
${CXX:-g++-13} -std=c++20 -O2 -fPIC -shared -DSPDLOG_FMT_EXTERNAL \
  -I"$TT_SDK/tt_metal_api" -I"$FORK/tt_stl" -I"$INSTALL" -I"$INSTALL/metalium-thirdparty" \
  -I"$FORK/tt_metal/third_party/umd/device/api" \
  -o pinx.so pinx.cpp -L"$TT_SDK" -l:libtt_metal.so -l:libfmt.so.11 -Wl,-rpath,/opt/tt-metal/build/lib
