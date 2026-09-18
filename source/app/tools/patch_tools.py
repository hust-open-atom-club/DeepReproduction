"""Patch parsing utilities for the knowledge agent."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional

from pydantic import BaseModel, Field


SOURCE_FILE_SUFFIXES = (
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".hh",
    ".rs",
    ".go",
    ".java",
    ".py",
    ".js",
    ".ts",
    ".lua",
)


def find_patch_diff(
    cve_id: str,
    search_roots: Optional[Iterable[str]] = None,
) -> Optional[Path]:
    """Locate patch.diff with configurable search roots.

    Search order:
      1. Each entry in search_roots (in order, if provided)
      2. Default fallback: ["Dataset", "source/Dataset"]

    Default prefixes always remain as a fallback so existing call sites
    (build / poc) keep working when search_roots is None.

    Returns None if no candidate exists.
    """

    candidates: list[str] = []
    if search_roots:
        candidates.extend(search_roots)
    for default in ("Dataset", "source/Dataset"):
        if default not in candidates:
            candidates.append(default)

    for prefix in candidates:
        candidate = Path(prefix) / cve_id / "vuln_data" / "vuln_diffs" / "patch.diff"
        if candidate.exists():
            return candidate
    return None


def score_patch_candidate(
    diff_text: str,
    *,
    url: str = "",
    fixed_ref: Optional[str] = None,
    preferred_files: Optional[Iterable[str]] = None,
) -> int:
    """Rank a unified diff for use as the authoritative verify patch.diff.

    Prefer security-fix commit diffs that touch source files over incidental
    build-system bumps (e.g. ECM/CMake version changes) discovered recursively.
    """

    text = diff_text or ""
    if not text.strip().startswith("diff ") and "@@" not in text:
        return -100

    score = 0
    lowered_url = (url or "").lower()
    fixed = (fixed_ref or "").strip().lower()
    if fixed and fixed in lowered_url:
        score += 100

    files = re.findall(r"^\+\+\+\s+b/(.+)$", text, re.MULTILINE)
    preferred = {item.replace("\\", "/").lstrip("./").lower() for item in (preferred_files or []) if item}
    source_hits = 0
    cmake_only = bool(files)
    for path in files:
        normalized = path.replace("\\", "/").lstrip("./")
        lowered = normalized.lower()
        if preferred and (lowered in preferred or any(lowered.endswith(pref) for pref in preferred)):
            score += 40
        if lowered.endswith(SOURCE_FILE_SUFFIXES):
            source_hits += 1
            cmake_only = False
            score += 20
        if "cmakelists.txt" in lowered or lowered.endswith(".cmake"):
            score -= 5
        else:
            cmake_only = False

    if cmake_only and files:
        score -= 40

    security_tokens = (
        "overflow",
        "underflow",
        "asan",
        "ubsan",
        "bounds",
        "sanitize",
        "null",
        "crash",
        "bpp",
        "return false",
        "static_assert",
    )
    lowered_diff = text.lower()
    score += sum(8 for token in security_tokens if token in lowered_diff)
    if source_hits == 0 and re.search(r"find_package\s*\(\s*ecm", lowered_diff):
        score -= 50
    return score


def should_replace_patch_diff(
    existing_text: str,
    candidate_text: str,
    *,
    candidate_url: str = "",
    fixed_ref: Optional[str] = None,
    preferred_files: Optional[Iterable[str]] = None,
) -> bool:
    """Return True when candidate_text is a better authoritative patch.diff."""

    if not (candidate_text or "").strip():
        return False
    if not (existing_text or "").strip():
        return True
    existing_score = score_patch_candidate(
        existing_text,
        fixed_ref=fixed_ref,
        preferred_files=preferred_files,
    )
    candidate_score = score_patch_candidate(
        candidate_text,
        url=candidate_url,
        fixed_ref=fixed_ref,
        preferred_files=preferred_files,
    )
    return candidate_score > existing_score


_DIFF_GIT_SPLIT_RE = re.compile(r"(?=^diff --git )", re.MULTILINE)
_BINARY_FILES_DIFFER_RE = re.compile(r"^Binary files .+ differ\s*$", re.MULTILINE)
_GIT_BINARY_PATCH_RE = re.compile(r"^GIT binary patch\s*$", re.MULTILINE)
_BINARY_LITERAL_RE = re.compile(r"^literal \d+\s*$", re.MULTILINE)
_BINARY_DELTA_RE = re.compile(r"^delta \d+\s*$", re.MULTILINE)
_DIFF_GIT_PATHS_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)(?:\n|$)")


def is_unapplyable_binary_stub_section(section: str) -> bool:
    """True for placeholder binary hunks that ``git apply`` cannot consume.

    Gate: section reports ``Binary files ... differ`` but lacks a real
    ``GIT binary patch`` / ``literal`` / ``delta`` payload (e.g. qpdf corpus
    stubs that only record ``index`` + ``Binary files /dev/null and b/...``).
    """

    if not section or not _BINARY_FILES_DIFFER_RE.search(section):
        return False
    if _GIT_BINARY_PATCH_RE.search(section):
        return False
    if _BINARY_LITERAL_RE.search(section) or _BINARY_DELTA_RE.search(section):
        return False
    return True


def strip_unapplyable_binary_stub_hunks(diff_text: str) -> tuple[str, list[str]]:
    """Drop unapplyable binary stubs; keep text and full binary patches.

    Returns ``(filtered_diff, dropped_paths)``. When nothing is dropped, the
    original text is returned unchanged (same object identity not guaranteed).
    """

    text = diff_text or ""
    if not text.strip() or "Binary files " not in text:
        return text, []

    parts = _DIFF_GIT_SPLIT_RE.split(text)
    kept: list[str] = []
    dropped: list[str] = []
    for part in parts:
        if not part:
            continue
        if part.lstrip().startswith("diff --git") and is_unapplyable_binary_stub_section(part):
            match = _DIFF_GIT_PATHS_RE.match(part.lstrip())
            dropped.append(match.group(2) if match else "unknown")
            continue
        kept.append(part)

    if not dropped:
        return text, []

    filtered = "".join(kept)
    if filtered and not filtered.endswith("\n"):
        filtered += "\n"
    return filtered, dropped


class PatchSummary(BaseModel):
    """Structured summary extracted from a patch."""

    affected_files: List[str] = Field(default_factory=list, description="Affected file list.")
    changed_functions: List[str] = Field(default_factory=list, description="Function or symbol hints.")
    summary: str = Field(default="", description="Short patch summary.")


class PatchTool:
    """Parse unified diff text into structured hints."""

    _FILE_RE = re.compile(r"^\+\+\+\s+b/(.+)$", re.MULTILINE)
    _FUNCTION_RE = re.compile(r"@@.*?@@\s*(.+)$", re.MULTILINE)

    def parse_diff(self, diff_text: str) -> PatchSummary:
        """Parse diff content and extract lightweight metadata."""

        affected_files = sorted(set(self._FILE_RE.findall(diff_text)))
        changed_functions = [item.strip() for item in self._FUNCTION_RE.findall(diff_text) if item.strip()]
        summary = f"Patch touches {len(affected_files)} file(s)." if affected_files else "Patch metadata unavailable."

        return PatchSummary(
            affected_files=affected_files,
            changed_functions=changed_functions[:20],
            summary=summary,
        )

    def extract_hunks(self, diff_text: str) -> List[str]:
        """Return diff hunks split by hunk header."""

        chunks = re.split(r"(?=^@@)", diff_text, flags=re.MULTILINE)
        return [chunk.strip() for chunk in chunks if chunk.strip().startswith("@@")]
