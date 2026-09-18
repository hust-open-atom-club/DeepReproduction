"""Stateless log parsing helpers shared by PoC and verify stages.

These functions implement the PoC log contract:
    target_binary=...
    trigger_command=...
    execution_exit_code=<int>
    stdout_begin / stdout_end
    stderr_begin / stderr_end
"""

from __future__ import annotations

import re
from typing import Any

# Cap the observed stdout/stderr carried into downstream stages/artifacts. A
# sanitizer "storm" (e.g. AddressSanitizer:DEADLYSIGNAL recursion) can emit
# gigabytes before the trigger timeout fires; keeping the full block in memory
# and in YAML artifacts is wasteful and can crash PyYAML on huge scalars. The
# crash header + a few stack frames are always at the head, which is all the
# crash-type matching needs.
MAX_OBSERVED_OUTPUT_CHARS = 256 * 1024


def _truncate_observed(text: str) -> str:
    if len(text) <= MAX_OBSERVED_OUTPUT_CHARS:
        return text
    return text[:MAX_OBSERVED_OUTPUT_CHARS]


def extract_block(text: str, begin: str, end: str) -> str:
    """Extract content between two marker lines."""

    pattern = rf"{re.escape(begin)}\n(.*?)(?:\n{re.escape(end)})"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def extract_execution_observation(execution_logs: str) -> dict[str, Any]:
    """Parse the PoC log contract into a structured observation."""

    stdout = _truncate_observed(extract_block(execution_logs, "stdout_begin", "stdout_end"))
    stderr = _truncate_observed(extract_block(execution_logs, "stderr_begin", "stderr_end"))
    if not stderr:
        stderr = _truncate_observed(_extract_outer_stderr(execution_logs))
    exit_code = None
    match = re.search(r"execution_exit_code=(\d+)", execution_logs)
    if match:
        exit_code = int(match.group(1))
    crash_type = classify_observed_crash_type(stdout, stderr, execution_logs)
    return {
        "observed_exit_code": exit_code,
        "observed_stdout": stdout,
        "observed_stderr": stderr,
        "observed_crash_type": crash_type,
    }


def _extract_outer_stderr(execution_logs: str) -> str:
    """Extract stderr-like outer sections when the script contract stderr block is empty."""

    for begin, end in (
        ("[container_run_stderr]\n", None),
        ("=== stderr ===\n", None),
    ):
        start = execution_logs.find(begin)
        if start == -1:
            continue
        content = execution_logs[start + len(begin) :]
        if end and end in content:
            content = content.split(end, 1)[0]
        return content.strip()
    return ""


def match_patterns(haystack: str, patterns: list[str]) -> list[str]:
    """Return the subset of patterns that appear (case-insensitively) in the haystack."""

    lowered = haystack.lower()
    matches = [pattern for pattern in patterns if pattern and pattern.lower() in lowered]
    return sorted(set(matches))


# Concrete ASan/MSan/UBSan bug kinds. Generic labels like "AddressSanitizer" are
# intentionally excluded — they also match noise such as alloc-dealloc-mismatch.
SPECIFIC_SANITIZER_BUG_TOKENS: tuple[str, ...] = (
    "heap-buffer-overflow",
    "stack-buffer-overflow",
    "global-buffer-overflow",
    "heap-use-after-free",
    "stack-use-after-free",
    "use-after-poison",
    "use-after-free",
    "double-free",
    "negative-size-param",
    "alloc-dealloc-mismatch",
    "initialization-order-fiasco",
    "stack-overflow",
)

GENERIC_SANITIZER_LABELS: frozenset[str] = frozenset({"addresssanitizer", "asan"})

# Fatal-but-unspecific labels. ASan always prints ABORTING after a real bug
# report; SEGV/null-deref often come from harvested descriptions. When the
# plan names a concrete sanitizer kind, these must not win classification
# or pattern matching.
WEAK_CRASH_LABELS: frozenset[str] = frozenset(
    {
        "segv",
        "segmentation fault",
        "null dereference",
        "null-dereference",
        "abort",
        "aborting",
        "assert",
    }
)


def classify_observed_crash_type(*texts: str) -> str:
    """Prefer a concrete sanitizer bug over generic abort/SEGV.

    ASan reports end with ``==N==ABORTING``. Scanning ``abort`` first would
    classify every overflow/UAF as abort and fail crash_type_compatible.
    """

    joined = "\n".join(texts or []).lower()
    if not joined.strip():
        return ""
    for token in SPECIFIC_SANITIZER_BUG_TOKENS:
        if token in joined:
            return token
    for marker in ("segmentation fault", "assert", "abort"):
        if marker in joined:
            return marker
    return ""


def ensure_specific_crash_token_in_patterns(
    patterns: list[str] | None,
    crash_type: str | None,
) -> list[str]:
    """Keep the expected sanitizer kind in stderr patterns even if SEGV is listed."""

    items = list(patterns or [])
    tokens = specific_sanitizer_bugs_in([crash_type or ""])
    if not tokens:
        return items
    token = tokens[0]
    if any(token in (item or "").lower() for item in items):
        return items
    return [token, *items]


def drop_weak_crash_labels_when_specific(
    patterns: list[str] | None,
    extra_texts: list[str] | None = None,
) -> list[str]:
    """Drop SEGV/abort/null-deref patterns when a specific sanitizer kind is expected."""

    items = list(patterns or [])
    probe = items + list(extra_texts or [])
    if not specific_sanitizer_bugs_in(probe):
        return items
    kept: list[str] = []
    for item in items:
        lowered = (item or "").strip().lower()
        if not lowered or lowered in WEAK_CRASH_LABELS:
            continue
        if lowered in GENERIC_SANITIZER_LABELS:
            continue
        if specific_sanitizer_bugs_in([item]):
            kept.append(item)
            continue
        if any(
            token in lowered
            for token in ("segv", "segmentation fault", "null dereference", "aborting")
        ):
            continue
        kept.append(item)
    return kept or items


def specific_sanitizer_bugs_in(texts: list[str] | None) -> list[str]:
    """Return specific sanitizer bug tokens mentioned in any of the texts."""

    found: list[str] = []
    parts = [(item or "").lower() for item in (texts or []) if item]
    if not parts:
        return found
    for token in SPECIFIC_SANITIZER_BUG_TOKENS:
        if token in found:
            continue
        if any(token in part for part in parts):
            found.append(token)
    return found


def drop_generic_sanitizer_labels(
    patterns: list[str] | None,
    extra_texts: list[str] | None = None,
) -> list[str]:
    """Drop bare AddressSanitizer/asan labels when a specific bug kind is also listed.

    Bare ``AddressSanitizer`` matches every ASan report, including unrelated
    ``alloc-dealloc-mismatch`` noise from mixed Qt/plugin linkage.
    ``extra_texts`` lets stack-keyword lists inherit the specific-bug signal
    from stderr/crash-type expectations.
    """

    items = list(patterns or [])
    probe = items + list(extra_texts or [])
    if not specific_sanitizer_bugs_in(probe):
        return items
    kept = [item for item in items if (item or "").strip().lower() not in GENERIC_SANITIZER_LABELS]
    return kept or items


def filter_hits_for_specific_sanitizer(hits: list[str], expected_texts: list[str] | None) -> list[str]:
    """Keep pattern hits that name an expected specific sanitizer bug.

    If the plan only expects a generic sanitizer label, hits are returned unchanged.
    """

    expected = specific_sanitizer_bugs_in(list(expected_texts or []))
    if not expected:
        return list(hits or [])
    return [hit for hit in (hits or []) if any(token in (hit or "").lower() for token in expected)]


def haystack_has_specific_sanitizer_bug(haystack: str, expected_texts: list[str] | None) -> bool:
    """True when haystack contains a specific sanitizer bug the plan asked for."""

    expected = specific_sanitizer_bugs_in(list(expected_texts or []))
    if not expected:
        return False
    lowered = (haystack or "").lower()
    return any(token in lowered for token in expected)


def should_outer_asan_preload(run_command: str = "", trigger_mode: str = "") -> bool:
    """Whether the PoC/verify wrapper should LD_PRELOAD ASan around the whole trigger.

    Library/plugin harnesses compile a small loader next to an already-ASan
    ``.so``. Outer preload must not wrap ``clang++``/``timeout``/``bash`` —
    that causes DEADLYSIGNAL storms or SIGILL. Inner ``LD_PRELOAD`` in the
    run command is the same class of bug — skip outer in both cases. When
    outer preload is still needed (cli-file), the trigger fragment starts
    timeout/bash clean and exports LD_PRELOAD only inside ``bash -c``.
    """

    if (trigger_mode or "").strip().lower() == "library-harness":
        return False
    cmd = run_command or ""
    if "LD_PRELOAD" in cmd:
        return False
    if "qimage_harness" in cmd:
        return False
    return True
