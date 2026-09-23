"""文件说明：verify 失败后的确定性 ref 重选（闭环恢复）。

AGENTS.md 记录的「verify 无闭环」遗留：`post_still_triggered` /
`pre_not_triggered` 目前是终态。本模块实现两类确定性恢复决策：

- ``post_still_triggered``（补丁没修住）→ 推进到下一个候选 fixed_ref。
  候选来源：当前 fixed_ref → knowledge.references 里的 commit 链接 →
  仓库历史挖掘（触及 affected_files、提交信息带 fix/security 指示词）。
- ``pre_not_triggered``（漏洞版不崩）→ OSV 的漏洞区间端点很可能已经
  包含修复（CVE-2021-45926 即此情形），把 vulnerable_ref 回退到当前
  fixed_ref 的父提交重建。

决策是纯函数（不执行任何构建），由 verify_node 调用并写回 AppState，
routers 据此把流程送回 build 节点。
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.schemas.knowledge import KnowledgeModel
from app.schemas.verify_result import VerifyResult


MAX_RECOVERY_ROUNDS = 2

_COMMIT_URL_RE = re.compile(r"/(?:-/)?commit/([0-9a-f]{7,40})")

# Words that make a commit message look like a fix for a memory-safety bug.
_FIX_MESSAGE_RE = re.compile(
    r"\b(fix|fixes|fixed|overflow|out.of.bounds|oob|uaf|use.after.free|"
    r"bounds|fuzz|crash|security|cve-)\b",
    re.IGNORECASE,
)


@dataclass
class RecoveryDecision:
    """一次 ref 重选决策。"""

    kind: str  # "advance_fixed" | "swap_vulnerable_parent"
    new_fixed_ref: Optional[str] = None
    new_vulnerable_ref: Optional[str] = None
    tried_refs: list[str] = field(default_factory=list)
    note: str = ""


def _candidates_from_references(knowledge: KnowledgeModel) -> list[str]:
    """Commit SHAs harvested from knowledge references (GitHub links)."""
    found: list[str] = []
    for url in knowledge.references or []:
        m = _COMMIT_URL_RE.search(url or "")
        if m:
            sha = m.group(1).rstrip(".")
            if sha not in found:
                found.append(sha)
    return found


def mine_fixed_candidates(
    repo_path: Optional[Path], affected_files: list[str], limit: int = 15
) -> list[str]:
    """Mine the repo history for fix-like commits touching affected files.

    Ordered: commits whose message carries fix indicators first (newest
    first within each group). Requires a plain git repo; returns [] on any
    failure (best-effort).
    """
    if not repo_path or not affected_files:
        return []
    try:
        out = subprocess.run(
            ["git", "log", "--all", "--format=%H%x09%s", "-n", str(limit * 3), "--", *affected_files],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    fix_like: list[str] = []
    other: list[str] = []
    for line in out.stdout.splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        sha, subject = parts
        (fix_like if _FIX_MESSAGE_RE.search(subject) else other).append(sha)
    return (fix_like + other)[:limit]


def fixed_ref_candidates(
    knowledge: KnowledgeModel,
    repo_path: Optional[Path] = None,
    tried: Optional[list[str]] = None,
) -> list[str]:
    """Ordered, de-duplicated fixed-ref candidates not yet tried."""
    tried = tried or []
    ordered: list[str] = []
    if knowledge.fixed_ref:
        ordered.append(knowledge.fixed_ref)
    ordered.extend(_candidates_from_references(knowledge))
    ordered.extend(
        mine_fixed_candidates(repo_path, knowledge.affected_files or [])
    )
    unique: list[str] = []
    for sha in ordered:
        if sha and sha not in tried and sha not in unique:
            unique.append(sha)
    return unique


def plan_recovery(
    verify: VerifyResult,
    knowledge: KnowledgeModel,
    repo_path: Optional[Path] = None,
    tried_refs: Optional[list[str]] = None,
) -> Optional[RecoveryDecision]:
    """Decide whether a failed verify result can be retried with new refs.

    Returns None when the failure is not ref-related or no candidate remains.
    """
    tried = list(tried_refs or [])
    reason = (verify.reason or "").lower()

    if "post_still_triggered" in reason:
        # The currently chosen fixed ref has already been disproven by this
        # verify pass — seed the tried set with it.
        effective_tried = list(tried)
        if knowledge.fixed_ref and knowledge.fixed_ref not in effective_tried:
            effective_tried.append(knowledge.fixed_ref)
        candidates = fixed_ref_candidates(knowledge, repo_path, effective_tried)
        if not candidates:
            return None
        nxt = candidates[0]
        return RecoveryDecision(
            kind="advance_fixed",
            new_fixed_ref=nxt,
            tried_refs=effective_tried + [nxt],
            note=(
                "post still triggered: chosen fixed ref does not stop this "
                f"crash; advancing to next candidate {nxt[:12]}"
            ),
        )

    if "pre_not_triggered" in reason:
        # The vulnerable build does not crash: the OSV vulnerable endpoint
        # very likely already contains the fix. Rebuild the vulnerable side
        # at the parent of the current fixed ref (CVE-2021-45926 pattern).
        if not knowledge.fixed_ref or not repo_path:
            return None
        try:
            out = subprocess.run(
                ["git", "rev-parse", "--verify", f"{knowledge.fixed_ref}^"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if out.returncode != 0:
            return None
        parent = out.stdout.strip()
        if not parent or parent == knowledge.vulnerable_ref:
            return None
        return RecoveryDecision(
            kind="swap_vulnerable_parent",
            new_vulnerable_ref=parent,
            tried_refs=tried,
            note=(
                "pre not triggered: vulnerable ref probably already fixed "
                f"(OSV range endpoint); rebuilding vulnerable side at fix parent {parent[:12]}"
            ),
        )

    return None


def apply_decision(
    knowledge: KnowledgeModel, decision: RecoveryDecision
) -> KnowledgeModel:
    """Return a knowledge copy with the swapped refs."""
    updates: dict = {}
    if decision.new_fixed_ref:
        updates["fixed_ref"] = decision.new_fixed_ref
    if decision.new_vulnerable_ref:
        updates["vulnerable_ref"] = decision.new_vulnerable_ref
    if updates:
        return knowledge.model_copy(update=updates)
    return knowledge
