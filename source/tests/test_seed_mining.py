"""文件说明：seed_mining 仓库语料挖掘器的单测。

场景取自 CVE-2021-36083（kimageformats）：Dataset 种子池为空时，
从 autotests/read/xcf/ 挖出 .xcf 样本作为权威载荷。
"""

from __future__ import annotations

from pathlib import Path

from app.stages.seed_mining import infer_seed_extensions, mine_repo_seeds


class TestInferExtensions:
    def test_xcf_from_kimageformats_context(self):
        exts = infer_seed_extensions(
            "stack buffer overflow in XCFImageFormat", "out-of-bounds write",
            "https://invent.kde.org/frameworks/kimageformats.git",
            "src/imageformats/xcf.cpp",
        )
        assert ".xcf" in exts

    def test_cil_from_selinux_context(self):
        exts = infer_seed_extensions(
            "use-after-free in CIL compiler", "use-after-free",
            "https://github.com/SELinuxProject/selinux.git",
            "libsepol/cil/src/cil_verify.c",
        )
        assert ".cil" in exts

    def test_abc_from_abcm2ps(self):
        exts = infer_seed_extensions("abcm2ps crash", "out-of-bounds read",
                                     "https://github.com/lewdlime/abcm2ps.git", "draw.c")
        assert ".abc" in exts

    def test_no_match_returns_empty(self):
        assert infer_seed_extensions("generic", "logic bug", "", "") == []


class TestMineRepoSeeds:
    def _make_repo(self, tmp_path: Path) -> Path:
        repo = tmp_path / "repo"
        xcf_dir = repo / "autotests" / "read" / "xcf"
        xcf_dir.mkdir(parents=True)
        (xcf_dir / "big.xcf").write_bytes(b"\x00" * 100)
        (xcf_dir / "small.xcf").write_bytes(b"\x00" * 10)
        (xcf_dir / "notes.txt").write_text("not a seed")
        other = repo / "src"
        other.mkdir()
        (other / "skip.xcf").write_bytes(b"\x00" * 5)  # outside test dirs
        return repo

    def test_mines_test_dirs_only(self, tmp_path: Path):
        repo = self._make_repo(tmp_path)
        seeds = mine_repo_seeds(repo, [".xcf"])
        names = [p.name for p in seeds]
        assert "small.xcf" in names and "big.xcf" in names
        assert "skip.xcf" not in names
        assert "notes.txt" not in names
        # smallest first
        assert names.index("small.xcf") < names.index("big.xcf")

    def test_no_repo_or_no_exts(self, tmp_path: Path):
        assert mine_repo_seeds(None, [".xcf"]) == []
        assert mine_repo_seeds(tmp_path, []) == []
        assert mine_repo_seeds(tmp_path / "missing", [".xcf"]) == []

    def test_per_extension_limit(self, tmp_path: Path):
        repo = tmp_path / "r2"
        d = repo / "test"
        d.mkdir(parents=True)
        for i in range(20):
            (d / f"s{i:02d}.cil").write_bytes(b"\x00" * (i + 1))
        seeds = mine_repo_seeds(repo, [".cil"], limit=8)
        assert len(seeds) == 8

    def test_nested_test_dirs_are_mined(self, tmp_path: Path):
        """36086 回归：selinux 的 .cil 语料在 libsemanage/tests/（二级目录）。"""

        repo = tmp_path / "selinux"
        nested = repo / "libsemanage" / "tests"
        nested.mkdir(parents=True)
        (nested / "test_bool.cil").write_bytes(b"(class CLASS (PERM))" * 10)
        (repo / "libsepol" / "src").mkdir(parents=True)
        (repo / "libsepol" / "src" / "sample.cil").write_bytes(b"(class X (P))")
        seeds = mine_repo_seeds(repo, [".cil"])
        names = [p.name for p in seeds]
        assert "test_bool.cil" in names

    def test_anywhere_fallback_only_when_no_test_hits(self, tmp_path: Path):
        repo = tmp_path / "nofixtures"
        docs = repo / "docs"
        docs.mkdir(parents=True)
        (docs / "example.cil").write_bytes(b"(class A (B))")
        assert [p.name for p in mine_repo_seeds(repo, [".cil"])] == ["example.cil"]
