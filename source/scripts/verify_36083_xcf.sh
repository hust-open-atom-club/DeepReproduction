#!/bin/bash
set -euo pipefail
ASAN_LIB="$(find /usr -name 'libclang_rt.asan-x86_64.so' 2>/dev/null | head -1 || true)"
if [[ -z "${ASAN_LIB}" ]]; then
  ASAN_LIB="$(find /usr -name 'libasan.so*' 2>/dev/null | head -1 || true)"
fi
echo "ASAN_LIB=${ASAN_LIB}"
export QT_PLUGIN_PATH=/src/kimageformats/build/bin
export LD_LIBRARY_PATH="$(dirname "${ASAN_LIB}"):${LD_LIBRARY_PATH:-}"
export ASAN_OPTIONS=detect_leaks=0:abort_on_error=0:symbolize=1
clang++ -g -O0 -fsanitize=address -shared-libasan -fno-omit-frame-pointer \
  /tmp/qimage_harness.cpp -o /tmp/qimage_harness \
  $(pkg-config --cflags --libs Qt5Gui Qt5Core)
set +e
/tmp/qimage_harness /poc/crafted_bpp5.xcf xcf
echo "exit=$?"
