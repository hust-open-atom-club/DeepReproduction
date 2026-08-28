"""ASan + PIE hardening helpers.

WSL2 kernels with `vm.mmap_rnd_bits=32` (upstream default is 28) can map PIE
executables at randomized bases that collide with ASan's reserved shadow /
allocator regions. The result is a bare SIGSEGV (exit 139) raised during ASan
runtime initialization, before main() runs and before any ASan report.

The fix: compile every ASan-instrumented binary with `-no-pie` so its load
address is fixed and cannot collide with the shadow region.
"""

from __future__ import annotations

import re

# A compiler invocation segment: starts at a compiler token and runs to the
# next shell separator (`;`, `&&`, `|`, newline). The token must be followed
# by whitespace so flags like `-x c++` are not confused with the compiler.
_COMPILER_SEGMENT_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:gcc|cc|clang|g\+\+|c\+\+|clang\+\+)(?=\s)[^;&|\n]*"
)


def ensure_no_pie_for_asan(command: str) -> str:
    """Inject `-no-pie` into compiler invocations that use `-fsanitize`.

    Idempotent: segments that already contain `-no-pie` are left untouched,
    and commands without any `-fsanitize` flag are returned unchanged.
    """

    if not command or "-fsanitize" not in command:
        return command

    def _inject(segment: str) -> str:
        if "-fsanitize" not in segment or "-no-pie" in segment:
            return segment
        out_match = re.search(r"\s(-o\s+\S+)", segment)
        if out_match:
            return segment[: out_match.start(1)] + " -no-pie " + segment[out_match.start(1):]
        return segment + " -no-pie"

    return _COMPILER_SEGMENT_RE.sub(lambda m: _inject(m.group(0)), command)
