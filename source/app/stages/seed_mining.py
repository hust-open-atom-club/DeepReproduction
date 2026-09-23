"""文件说明：仓库测试语料挖掘器。

当 Dataset 权威种子池为空时，扫描仓库的 test/ autotests/ tests/
examples/ 目录，按漏洞的输入格式收集真实样本（CVE-2021-36083 的载荷
正是这样来自仓库自带的 simple-rgb.xcf 测试图——LLM 无法凭空写出带
校验的二进制格式，但仓库里常有现成的）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# extension -> keywords that appear in summary/vulnerability_type/repo_url/
# affected_files when that extension is the trigger input format.
_FORMAT_HINTS: dict[str, tuple[str, ...]] = {
    ".cil": ("cil", "selinux", "secilc"),
    ".xcf": ("xcf", "gimp", "kimageformats"),
    ".abc": ("abcm2ps", "abc music", " sheet music"),
    ".mdb": ("mdbtools", "access database", ".mdb"),
    ".accdb": ("mdbtools",),
    ".svg": ("svg", "qtsvg"),
    ".ttf": ("harfbuzz", "font"),
    ".otf": ("harfbuzz", "font"),
    ".ttc": ("harfbuzz", "font"),
    ".ps": ("ghostscript", "postscript"),
    ".pcap": ("pcap", "packet"),
    ".wav": ("audio",),
    ".bmp": ("bmp",),
    ".pcx": ("pcx",),
    ".pix": ("gdal", "pcidsk"),
    ".exr": ("openexr",),
    ".lrz": ("lrzip",),
    ".pdf": ("pdf", "qpdf"),
    ".lua": ("lua",),
    ".mat": ("matio", "matlab", "mat file", "hdf5", "hdf"),
    ".h5": ("hdf5", "hdf", "matio"),
    ".hdf5": ("hdf5", "hdf", "matio"),
}

_TEST_DIRS = ("test", "tests", "autotests", "examples", "testdata", "data")

_TEST_DIR_SET = frozenset(_TEST_DIRS)
_WALK_DIR_BUDGET = 20000

_SEED_BYTE_LIMIT = 2 * 1024 * 1024
_SEED_COUNT_LIMIT = 8


def infer_seed_extensions(*text_sources: Optional[str]) -> list[str]:
    """Infer likely trigger-input file extensions from knowledge text."""

    blob = " ".join((s or "") for s in text_sources).lower()
    hits = [
        ext
        for ext, keywords in _FORMAT_HINTS.items()
        if any(kw in blob for kw in keywords)
    ]
    return hits


def _scan_repo(repo_path: Path, wanted: set[str]) -> tuple[list[Path], list[Path]]:
    """全仓库受限遍历，返回（测试目录命中，非测试目录命中）。

    旧实现只扫仓库顶层的 test*/ 目录，漏掉了 libsemanage/tests/ 这类
    子模块内的测试语料（CVE-2021-36086 的 .cil 样本全在二级目录）。
    """

    test_hits: list[Path] = []
    any_hits: list[Path] = []
    visited = 0
    for root, _dirs, files in os.walk(repo_path):
        visited += 1
        if visited > _WALK_DIR_BUDGET:
            break
        rel_parts = {part.lower() for part in Path(root).relative_to(repo_path).parts}
        in_test_dir = bool(rel_parts & _TEST_DIR_SET)
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in wanted:
                continue
            path = Path(root) / name
            (test_hits if in_test_dir else any_hits).append(path)
    return test_hits, any_hits


def mine_repo_seeds(
    repo_path: Optional[Path],
    extensions: list[str],
    limit: int = _SEED_COUNT_LIMIT,
) -> list[Path]:
    """Collect real input samples from the repo's test directories.

    Smallest files first (small seeds parse faster and minimize faster).
    Test-tree hits win; non-test-tree hits are a fallback only when the
    repo ships no test fixtures for the format. Best-effort: any filesystem
    error returns what was found so far.
    """
    if not repo_path or not extensions or not repo_path.is_dir():
        return []
    wanted = {ext.lower() for ext in extensions}
    try:
        test_hits, any_hits = _scan_repo(repo_path, wanted)
    except OSError:
        return []

    def _usable(paths: list[Path]) -> list[Path]:
        usable: list[Path] = []
        for path in paths:
            try:
                if path.is_file() and path.stat().st_size <= _SEED_BYTE_LIMIT:
                    usable.append(path)
            except OSError:
                continue
        usable.sort(key=lambda p: p.stat().st_size if p.exists() else 1 << 30)
        return usable

    found = _usable(test_hits) or _usable(any_hits)
    # Keep at most `limit` per extension so one format cannot starve another.
    per_ext: dict[str, int] = {}
    selected: list[Path] = []
    for path in found:
        ext = path.suffix.lower()
        if per_ext.get(ext, 0) >= limit:
            continue
        per_ext[ext] = per_ext.get(ext, 0) + 1
        selected.append(path)
        if len(selected) >= limit * 2:
            break
    return selected
