"""Patch parsing utilities for the knowledge agent."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional

from pydantic import BaseModel, Field


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
