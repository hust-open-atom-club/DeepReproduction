"""Tests for app.tools.patch_tools.find_patch_diff."""

from app.tools.patch_tools import (
    find_patch_diff,
    is_unapplyable_binary_stub_section,
    score_patch_candidate,
    should_replace_patch_diff,
    strip_unapplyable_binary_stub_hunks,
)


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


def test_score_prefers_fix_commit_source_diff_over_ecm_bump():
    ecm_bump = """diff --git a/CMakeLists.txt b/CMakeLists.txt
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -5,7 +5,7 @@
-find_package(ECM 5.81.0  NO_MODULE)
+find_package(ECM 5.82.0  NO_MODULE)
"""
    fix_diff = """diff --git a/src/imageformats/xcf.cpp b/src/imageformats/xcf.cpp
--- a/src/imageformats/xcf.cpp
+++ b/src/imageformats/xcf.cpp
@@ -1361,6 +1361,11 @@
+    if (bpp > 4) {
+        return false;
+    }
"""
    assert score_patch_candidate(
        fix_diff,
        url="https://invent.kde.org/.../commit/297ed9a2.diff",
        fixed_ref="297ed9a2",
        preferred_files=["src/imageformats/xcf.cpp"],
    ) > score_patch_candidate(ecm_bump)
    assert should_replace_patch_diff(
        ecm_bump,
        fix_diff,
        candidate_url="https://invent.kde.org/.../commit/297ed9a2.diff",
        fixed_ref="297ed9a2fe339bfe36916b9fce628c3242e5be0f",
        preferred_files=["src/imageformats/xcf.cpp"],
    )


def test_strip_unapplyable_binary_stub_keeps_source_hunks():
    stub_patch = """diff --git a/ChangeLog b/ChangeLog
index 2db53cb80..1e31efb48 100644
--- a/ChangeLog
+++ b/ChangeLog
@@ -1,3 +1,4 @@
+note
 2021-01-03  Jay Berkenbilt  <ejb@ql.org>
diff --git a/fuzz/qpdf_extra/28262.fuzz b/fuzz/qpdf_extra/28262.fuzz
new file mode 100644
index 000000000..4e872ba41
Binary files /dev/null and b/fuzz/qpdf_extra/28262.fuzz differ
diff --git a/libqpdf/Pl_AES_PDF.cc b/libqpdf/Pl_AES_PDF.cc
index 18cf3a4d2..2865f8049 100644
--- a/libqpdf/Pl_AES_PDF.cc
+++ b/libqpdf/Pl_AES_PDF.cc
@@ -238,6 +238,6 @@ Pl_AES_PDF::flush(bool strip_padding)
-    getNext()->write(this->outbuf, bytes);
     this->offset = 0;
+    getNext()->write(this->outbuf, bytes);
"""
    filtered, dropped = strip_unapplyable_binary_stub_hunks(stub_patch)
    assert dropped == ["fuzz/qpdf_extra/28262.fuzz"]
    assert "Binary files" not in filtered
    assert "diff --git a/ChangeLog b/ChangeLog" in filtered
    assert "diff --git a/libqpdf/Pl_AES_PDF.cc b/libqpdf/Pl_AES_PDF.cc" in filtered
    assert "fuzz/qpdf_extra/28262.fuzz" not in filtered


def test_strip_unapplyable_binary_stub_preserves_full_git_binary_patch():
    full_binary = """diff --git a/data/blob.bin b/data/blob.bin
new file mode 100644
index 0000000000000000000000000000000000000000..aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
GIT binary patch
literal 4
zcmV+`000031RcLW#

"""
    filtered, dropped = strip_unapplyable_binary_stub_hunks(full_binary)
    assert dropped == []
    assert filtered == full_binary
    assert is_unapplyable_binary_stub_section(full_binary) is False


def test_strip_unapplyable_binary_stub_noop_without_binary_marker():
    text_only = """diff --git a/a.c b/a.c
--- a/a.c
+++ b/a.c
@@ -1 +1 @@
-old
+new
"""
    filtered, dropped = strip_unapplyable_binary_stub_hunks(text_only)
    assert dropped == []
    assert filtered == text_only
