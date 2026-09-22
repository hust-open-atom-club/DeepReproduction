"""文件说明：verify_recovery 闭环恢复决策的单测。

场景取自真实 CVE：
- CVE-2021-45926（mdbtools）：pre_not_triggered → 回退 vulnerable 到
  fix 父提交（OSV 漏洞端点已含修复）。
- CVE-2021-45942（openexr）：post_still_triggered → 推进 fixed 候选。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from app.schemas.knowledge import KnowledgeModel
from app.schemas.verify_result import VerifyResult
from app.stages.verify_recovery import (
    apply_decision,
    fixed_ref_candidates,
    plan_recovery,
)


def _knowledge(**kw) -> KnowledgeModel:
    base = dict(
        cve_id="CVE-2021-45926",
        summary="s",
        vulnerability_type="stack-buffer-overflow",
        repo_url="https://github.com/mdbtools/mdbtools.git",
        vulnerable_ref="e52c47d561a2582aa9a6c6c9a59cf7503b7040fd",
        fixed_ref="373b7ff4c4daf887269c078407cb1338942c4ea6",
        references=[
            "https://github.com/mdbtools/mdbtools/commit/373b7ff4c4daf887269c078407cb1338942c4ea6",
            "https://github.com/mdbtools/mdbtools/commit/9b6b52cc8c5838cffeee9388c04890fe1eb73b52",
        ],
        affected_files=["src/libmdb/money.c"],
    )
    base.update(kw)
    return KnowledgeModel(**base)


def _verify(verdict: str, reason: str) -> VerifyResult:
    return VerifyResult(
        pre_patch_triggered=False,
        post_patch_clean=False,
        verdict=verdict,
        reason=reason,
        confidence="low",
    )


class TestPlanRecovery:
    def test_post_still_triggered_advances_candidate(self):
        k = _knowledge()
        # 373b7ff4 already chosen as fixed_ref; next candidate should be the
        # harvested commit 9b6b52cc8c... from references.
        d = plan_recovery(_verify("failed", "post_still_triggered"), k)
        assert d is not None
        assert d.kind == "advance_fixed"
        assert d.new_fixed_ref == "9b6b52cc8c5838cffeee9388c04890fe1eb73b52"
        assert "9b6b52cc8c5838cffeee9388c04890fe1eb73b52" in d.tried_refs

    def test_post_still_triggered_no_candidates_left(self):
        k = _knowledge(
            references=[],
            fixed_ref="373b7ff4c4daf887269c078407cb1338942c4ea6",
        )
        d = plan_recovery(_verify("failed", "post_still_triggered"), k, tried_refs=["373b7ff4c4daf887269c078407cb1338942c4ea6"])
        assert d is None

    def test_pre_not_triggered_swaps_to_fix_parent(self, tmp_path: Path):
        # Build a tiny git repo: fix commit whose parent differs from the
        # (already-fixed) vulnerable ref — the exact 45926 shape.
        repo = tmp_path / "r"
        repo.mkdir()
        def git(*args):
            subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True)
        git("init", "-q")
        git("config", "user.email", "t@t")
        git("config", "user.name", "t")
        (repo / "money.c").write_text("int a;\n")
        git("add", ".")
        git("commit", "-qm", "base")
        parent = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo), capture_output=True, text=True
        ).stdout.strip()
        (repo / "money.c").write_text("int a; int guard;\n")
        git("commit", "-aqm", "Fix buffer overflow in mdb_numeric_to_string")
        fix = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo), capture_output=True, text=True
        ).stdout.strip()

        k = _knowledge(fixed_ref=fix, vulnerable_ref="e52c47d561a2582aa9a6c6c9a59cf7503b7040fd")
        d = plan_recovery(_verify("failed", "pre_not_triggered: exit 0"), k, repo_path=repo)
        assert d is not None
        assert d.kind == "swap_vulnerable_parent"
        assert d.new_vulnerable_ref == parent

    def test_non_ref_failure_returns_none(self):
        d = plan_recovery(_verify("failed", "patch_apply_failed: exit_code=1"), _knowledge())
        assert d is None

    def test_apply_decision_swaps_fields(self):
        d = plan_recovery(_verify("failed", "post_still_triggered"), _knowledge())
        assert d is not None
        k2 = apply_decision(_knowledge(), d)
        assert k2.fixed_ref == d.new_fixed_ref
        assert k2.vulnerable_ref == "e52c47d561a2582aa9a6c6c9a59cf7503b7040fd"


class TestCandidateMining:
    def test_candidates_dedup_and_order(self):
        k = _knowledge()
        cands = fixed_ref_candidates(k, repo_path=None)
        # fixed_ref first, then reference commits (deduped), repo untouched.
        assert cands[0] == "373b7ff4c4daf887269c078407cb1338942c4ea6"
        assert "9b6b52cc8c5838cffeee9388c04890fe1eb73b52" in cands
        assert len(cands) == len(set(cands))

    def test_tried_refs_excluded(self):
        k = _knowledge()
        cands = fixed_ref_candidates(
            k, repo_path=None, tried=["373b7ff4c4daf887269c078407cb1338942c4ea6"]
        )
        assert cands and "373b7ff4c4daf887269c078407cb1338942c4ea6" not in cands

    def test_repo_mining_prefers_fix_messages(self, tmp_path: Path):
        repo = tmp_path / "r2"
        repo.mkdir()
        def git(*args):
            subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True)
        git("init", "-q")
        git("config", "user.email", "t@t")
        git("config", "user.name", "t")
        (repo / "money.c").write_text("v1\n")
        git("add", "."); git("commit", "-qm", "docs update")
        (repo / "money.c").write_text("v2\n")
        git("commit", "-aqm", "Fix buffer overflow in numeric")
        fix_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo), capture_output=True, text=True
        ).stdout.strip()
        (repo / "money.c").write_text("v3\n")
        git("commit", "-aqm", "typo")
        k = _knowledge(references=[], fixed_ref=None, affected_files=["money.c"])
        cands = fixed_ref_candidates(k, repo_path=repo)
        assert cands, "mining should return candidates"
        assert cands[0] == fix_sha  # fix-like commit ranked first
