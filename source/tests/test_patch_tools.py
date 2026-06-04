"""Tests for app.tools.patch_tools.find_patch_diff."""

from app.tools.patch_tools import find_patch_diff


def _make_patch(root, cve_id):
    target = root / cve_id / "vuln_data" / "vuln_diffs" / "patch.diff"
    target.parent.mkdir(parents=True)
    target.write_text("--- a\n+++ b\n", encoding="utf-8")
    return target


def test_find_patch_diff_respects_custom_search_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    custom_root = tmp_path / "custom_dataset"
    target = _make_patch(custom_root, "CVE-FAKE")

    result = find_patch_diff("CVE-FAKE", search_roots=[str(custom_root)])
    assert result == target


def test_find_patch_diff_default_roots_still_work_when_search_roots_none(tmp_path, monkeypatch):
    """search_roots=None 时回归到双前缀默认（Dataset/, source/Dataset/）。"""

    monkeypatch.chdir(tmp_path)
    _make_patch(tmp_path / "Dataset", "CVE-FAKE")
    result = find_patch_diff("CVE-FAKE")
    assert result is not None
    # Function returns a relative path; resolve for comparison.
    assert result.resolve() == (tmp_path / "Dataset" / "CVE-FAKE" / "vuln_data" / "vuln_diffs" / "patch.diff").resolve()


def test_find_patch_diff_custom_root_falls_back_to_default(tmp_path, monkeypatch):
    """自定义 root 不存在 patch 时，应该兜底到默认前缀。"""

    monkeypatch.chdir(tmp_path)
    _make_patch(tmp_path / "source" / "Dataset", "CVE-FAKE")

    result = find_patch_diff("CVE-FAKE", search_roots=["nonexistent"])
    assert result is not None
    assert result.resolve() == (tmp_path / "source" / "Dataset" / "CVE-FAKE" / "vuln_data" / "vuln_diffs" / "patch.diff").resolve()


def test_find_patch_diff_returns_none_when_nothing_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert find_patch_diff("CVE-FAKE") is None
    assert find_patch_diff("CVE-FAKE", search_roots=["also-nonexistent"]) is None
