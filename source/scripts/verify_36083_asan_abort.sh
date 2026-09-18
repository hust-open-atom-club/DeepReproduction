#!/bin/bash
set +e
export QT_PLUGIN_PATH='/src/kimageformats/build/bin'
export ASAN_OPTIONS='abort_on_error=1:symbolize=0:detect_leaks=0:handle_abort=1:handle_segv=1'
for _asan_dir in /usr/lib/llvm-*/lib/clang/*/lib/linux /usr/lib/clang/*/lib/linux; do
  if [[ -f "${_asan_dir}/libclang_rt.asan-x86_64.so" ]]; then
    export LD_LIBRARY_PATH="${_asan_dir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    break
  fi
done
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
clang++ -g -O0 -fsanitize=address -shared-libasan -fno-omit-frame-pointer \
  /tmp/qimage_harness.cpp -o /tmp/qimage_harness \
  $(pkg-config --cflags --libs Qt5Gui Qt5Core)
timeout 90s /tmp/qimage_harness /poc/crafted_bpp5.xcf xcf
echo "exit=$?"
