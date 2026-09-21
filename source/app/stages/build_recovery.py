"""文件说明：构建失败的确定性恢复配方。

本模块把历史上由人工执行的构建修复沉淀为「失败签名 → 幂等脚本补丁」
配方表。`plan_and_execute_build` 在 LLM 重规划之前先查配方：命中即改写
build.sh（或 Dockerfile）重试，不消耗 LLM 重规划次数。

配方来源（全部经过真实 CVE 验证）：
- CVE-2021-45926/45927 (mdbtools):  libtoolize 宏目录冲突、configure 带
  ASan 无法运行 sanity 测试、automake 时间戳回触导致 config.status 带
  ASan 重跑。
- CVE-2021-45932/45933 (wolfMQTT):  build-aux/config.rpath 缺失（autogen.sh
  会 touch 该文件）。
- CVE-2021-32434/36083/36084/36085: Windows autocrlf 检出导致 autotools/
  cmake 全线中毒；cmake find_package 版本钉高于发行版。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class RecipeResult:
    """一次配方应用的输出。"""

    matched_kinds: list[str] = field(default_factory=list)
    new_script: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.matched_kinds)


# ---------------------------------------------------------------------------
# 配方实现：每条 = (签名正则, build.sh 变换)。变换必须幂等。
# ---------------------------------------------------------------------------

def _inject_after_line(script: str, anchor_re: str, insert: str) -> str:
    """Insert `insert` after the first line matching `anchor_re` (idempotent)."""
    if insert.strip() and insert.strip() in script:
        return script
    lines = script.splitlines(keepends=True)
    pat = re.compile(anchor_re)
    for i, line in enumerate(lines):
        if pat.search(line):
            block = insert if insert.endswith("\n") else insert + "\n"
            return "".join(lines[: i + 1]) + block + "".join(lines[i + 1 :])
    return script


def _inject_before_configure(script: str, insert: str) -> str:
    """在构建脚本的 configure 步骤前注入 `insert`。

    LLM 写的 build.sh 日志措辞不统一（`log "running configure step"` 只是
    模板写法），锚点必须多级兜底；全部落空时插到 set 行之后。
    """

    if insert.strip() and insert.strip() in script:
        return script
    lines = script.splitlines(keepends=True)
    block = insert if insert.endswith("\n") else insert + "\n"
    anchors = [
        re.compile(r"log \"running configure step\""),
        re.compile(r"log \"[^\"]*configure"),
        re.compile(r"^\s*(?:\./)?autogen\.sh"),
        re.compile(r"^\s*(?:\./)?configure\b"),
        re.compile(r"^\s*autoreconf\b"),
        re.compile(r"^\s*cmake\b"),
    ]
    for pat in anchors:
        for i, line in enumerate(lines):
            if pat.search(line):
                return "".join(lines[:i]) + block + "".join(lines[i:])
    last = -1
    for i, line in enumerate(lines):
        if line.startswith(("set ", "set -", "#!/")):
            last = i
    if last >= 0:
        return "".join(lines[: last + 1]) + block + "".join(lines[last + 1 :])
    return block + script


def _fix_libtoolize_conflict(script: str, log: str = "") -> str:
    """libtoolize: AC_CONFIG_MACRO_DIRS([m4]) conflicts with ACLOCAL_AMFLAGS=-I m4."""
    if "s/^ACLOCAL_AMFLAGS/#ACLOCAL_AMFLAGS/" in script:
        return script
    return _inject_after_line(
        script,
        r"log \"running configure step\"",
        "sed -i 's/^ACLOCAL_AMFLAGS/#ACLOCAL_AMFLAGS/' Makefile.am || true",
    )


def _fix_config_rpath(script: str, log: str = "") -> str:
    """automake: required file 'build-aux/config.rpath' not found — the repo's
    own autogen.sh touches it; prefer autogen.sh over a bare autoreconf."""
    if "autogen.sh" in script:
        return script
    return re.sub(
        r"(?m)^\s*autoreconf\s+(-[a-zA-Z]+\s*)*$",
        "bash ./autogen.sh || autoreconf -i -f",
        script,
        count=1,
    )


def _fix_configure_asan(script: str, log: str = "") -> str:
    """configure: error: cannot run C compiled programs — the configure step
    must not carry -shared-libasan (autoconf sanity tests cannot link it).
    Rewrite any `./configure` invocation to run bare (env-clean)."""
    def _clean(match: re.Match) -> str:
        line = match.group(0)
        if "./configure" not in line or "shared-libasan" not in line:
            return line
        # Drop sanitizer tokens from the configure line.
        cleaned = re.sub(
            r"\b(?:CC|CXX|CFLAGS|CXXFLAGS|LDFLAGS|SANITIZER_FLAGS)=\"[^\"]*\"\s*",
            "",
            line,
        )
        cleaned = re.sub(r"\s+-fsanitize=\S+", "", cleaned)
        cleaned = re.sub(r"\s+-shared-libasan\b", "", cleaned)
        return cleaned
    return re.sub(
        r"(?m)^.*\./configure.*$", _clean, script
    )


def _fix_crlf_contamination(script: str, log: str = "") -> str:
    """CRLF checkout poisons autotools/cmake inputs; normalize before use."""
    if "s/\\r$//" in script or "s/\r\$//" in script:
        return script
    return _inject_after_line(
        script,
        r"^cd \"",
        (
            "git config core.autocrlf false || true\n"
            "find . -path ./.git -prune -o -type f \\( -name '*.ac' -o -name '*.am' "
            "-o -name '*.in' -o -name '*.m4' \\) -exec sed -i 's/\\r$//' {} + || true"
        ),
    )


def _fix_automake_timestamps(script: str, log: str = "") -> str:
    """`git reset/apply` (verify) restores autotools mtimes older than build
    outputs, so make re-runs config.status with ASan CFLAGS and configure
    blows up again. Pin timestamps right before make."""
    if "touch configure config.status" in script:
        return script
    pat = re.compile(r"(?m)^(\s*)make\s")
    m = pat.search(script)
    if not m:
        return script
    lines = script.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if pat.search(line):
            block = (
                "# Pin autotools timestamps: make must not re-run configure\n"
                "# with ASan CFLAGS (autoconf sanity tests cannot link it).\n"
                "touch configure config.status Makefile 2>/dev/null || true\n"
            )
            if "Pin autotools timestamps" in script:
                break
            return "".join(lines[:i]) + block + "".join(lines[i:])
    return script


def _fix_qmake_examples(script: str, log: str = "") -> str:
    """qmake -r: 'You cannot build examples inside the Qt source tree' — use
    top-level qmake plus the sub-src-qmake_all rule instead of -r."""
    if "qmake -r" not in script:
        return script
    return script.replace("qmake -r", "qmake")


@dataclass(frozen=True)
class BuildRecipe:
    kind: str
    signature: re.Pattern
    fix: object  # Callable[[str, str], str]
    note: str


def _fix_missing_autotools_tools(script: str, log: str = "") -> str:
    """libtoolize/autoreconf/automake: command not found — 工具链没装。"""
    install = (
        "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
        "--no-install-recommends libtool automake autoconf pkg-config >/dev/null 2>&1 || true"
    )
    if "libtool automake autoconf pkg-config" in script:
        return script
    return _inject_before_configure(script, install)


def _fix_pie_link_error(script: str, log: str = "") -> str:
    """R_X86_64_32 / recompile with -fPIE — 链接缺 -no-pie。

    cmake 的 CMAKE_EXE_LINKER_FLAGS 已存在但缺 -no-pie 时就地补；
    完全缺失时向 cmake 调用行注入。
    """
    changed = False

    def _upgrade(match: "re.Match[str]") -> str:
        nonlocal changed
        flags = match.group(1)
        if "-no-pie" in flags:
            return match.group(0)
        upgraded = f'-DCMAKE_EXE_LINKER_FLAGS="{flags} -no-pie"'
        changed = True
        return upgraded

    script = re.sub(r'-DCMAKE_EXE_LINKER_FLAGS="([^"]*)"', _upgrade, script)

    if "-DCMAKE_EXE_LINKER_FLAGS=" not in script:

        def _add_flag(match: "re.Match[str]") -> str:
            nonlocal changed
            line = match.group(0)
            inject = ' -DCMAKE_EXE_LINKER_FLAGS="-no-pie" -DCMAKE_SHARED_LINKER_FLAGS="-no-pie"'
            changed = True
            return line.rstrip() + inject

        script = re.sub(r"(?m)^.*\bcmake\b.*$", _add_flag, script)

    return script


def _fix_missing_required_lib(script: str, log: str = "") -> str:
    """configure: error: <lib> is required — 从日志提取库名并安装对应 -dev 包。"""

    names: list[str] = []
    for match in re.finditer(r"configure: error: ([A-Za-z0-9_+-]+) is required", log or ""):
        names.append(match.group(1).lower())
    # 另一种措辞：Could not find lz4 library - please install liblz4-dev
    for match in re.finditer(r"please install ([A-Za-z0-9_.+-]+)", log or ""):
        names.append(match.group(1).lower())
    packages: list[str] = []
    known = {
        "wolfssl": "libwolfssl-dev",
        "ssl": "libssl-dev",
        "openssl": "libssl-dev",
        "zlib": "zlib1g-dev",
        "ncurses": "libncurses-dev",
        "cunit": "libcunit1-dev",
        "fuse": "libfuse-dev",
        "curl": "libcurl4-openssl-dev",
    }
    for name in names:
        # configure 报的名字常带 lib 前缀（libwolfssl）；剥掉再映射，避免
        # 拼出 liblibwolfssl-dev 这种不存在的包。报错里直接给出的包名
        # （liblz4-dev）原样采纳。
        if name.endswith("-dev"):
            pkg = name
        else:
            stem = name[3:] if name.startswith("lib") else name
            pkg = known.get(name) or known.get(stem) or f"lib{stem}-dev"
        if pkg not in packages:
            packages.append(pkg)
    if not packages:
        return script
    marker = " ".join(packages)
    if marker in script:
        return script
    install = (
        "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
        f"--no-install-recommends {marker} >/dev/null 2>&1 || true"
    )
    return _inject_before_configure(script, install)


def _inject_before_first_make(script: str, insert: str) -> str:
    lines = script.splitlines(keepends=True)
    block = insert if insert.endswith(chr(10)) else insert + chr(10)
    for i, line in enumerate(lines):
        if re.match(r"^\s*make\b", line):
            return "".join(lines[:i]) + block + "".join(lines[i:])
    return script


def _fix_missing_makefile(script: str, log: str = "") -> str:
    """make: No targets specified and no makefile found — configure 没跑或没生成 Makefile。"""

    if "[ -f Makefile ]" in script:
        return script
    bootstrap = (
        "[ -f Makefile ] || { ./autogen.sh >/dev/null 2>&1 || autoreconf -i -f >/dev/null 2>&1; "
        # configure 报错必须留在日志里：缺库时 missing_required_lib 配方
        # 靠 `configure: error: X is required` 签名接力安装（wolfmqtt 实证）。
        "./configure || true; }"
    )
    return _inject_before_first_make(script, bootstrap)


def _fix_unbound_ld_library_path(script: str, log: str = "") -> str:
    """set -u 下引用未定义的 LD_LIBRARY_PATH（wolfmqtt/45937 实证）。"""

    guard = 'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"'
    if guard in script:
        return script
    return _inject_before_configure(script, guard)


RECIPES: tuple[BuildRecipe, ...] = (
    BuildRecipe(
        kind="libtoolize_macro_conflict",
        signature=re.compile(
            r"AC_CONFIG_MACRO_DIRS.*conflicts with ACLOCAL_AMFLAGS"
        ),
        fix=_fix_libtoolize_conflict,
        note="commented out ACLOCAL_AMFLAGS in Makefile.am before autoreconf",
    ),
    BuildRecipe(
        kind="missing_config_rpath",
        signature=re.compile(
            r"required file ['\"].*config\.rpath['\"] not found"
        ),
        fix=_fix_config_rpath,
        note="prefer repo autogen.sh (touches build-aux/config.rpath)",
    ),
    BuildRecipe(
        kind="configure_cannot_run",
        signature=re.compile(
            r"configure: error: cannot run C compiled programs|"
            r"cannot run C compiled programs"
        ),
        fix=_fix_configure_asan,
        note="stripped sanitizer flags from the ./configure invocation",
    ),
    BuildRecipe(
        kind="crlf_contamination",
        signature=re.compile(
            r"cannot find input file: `[^']*\.in'|XFile\.pm|"
            r"possibly undefined macro"
        ),
        fix=_fix_crlf_contamination,
        note="normalized CRLF on autotools inputs (Windows autocrlf checkout)",
    ),
    BuildRecipe(
        kind="automake_configure_rerun",
        signature=re.compile(
            r"config\.status: error|make\[\d\]: \*\*\* \[config\.status\]"
        ),
        fix=_fix_automake_timestamps,
        note="pinned autotools timestamps so make does not re-run configure",
    ),
    BuildRecipe(
        kind="qmake_examples_in_tree",
        signature=re.compile(
            r"You cannot build examples inside the Qt source tree"
        ),
        fix=_fix_qmake_examples,
        note="dropped qmake -r (top-level qmake + sub-src-qmake_all)",
    ),
    BuildRecipe(
        kind="missing_autotools_tools",
        signature=re.compile(
            r"libtoolize: command not found|autoreconf: command not found|"
            r"automake: command not found|aclocal: command not found|"
            r"autoreconf: not found|libtoolize: not found"
        ),
        fix=_fix_missing_autotools_tools,
        note="apt-installed libtool/automake/autoconf/pkg-config before autogen",
    ),
    BuildRecipe(
        kind="missing_required_lib",
        signature=re.compile(
            r"configure: error: [A-Za-z0-9_+-]+ is required|"
            r"please install [A-Za-z0-9_.+-]+"
        ),
        fix=_fix_missing_required_lib,
        note="apt-installed the library configure reported as required",
    ),
    BuildRecipe(
        kind="missing_makefile",
        signature=re.compile(
            r"make: \*\*\* No targets specified and no makefile found"
        ),
        fix=_fix_missing_makefile,
        note="bootstrapped autotools (autogen/autoreconf+configure) when make had no Makefile",
    ),
    BuildRecipe(
        kind="unbound_ld_library_path",
        signature=re.compile(
            r"LD_LIBRARY_PATH: unbound variable"
        ),
        fix=_fix_unbound_ld_library_path,
        note="defaulted LD_LIBRARY_PATH under set -u",
    ),
    BuildRecipe(
        kind="pie_link_error",
        signature=re.compile(
            r"recompile with -fPIE|R_X86_64_32 against|"
            r"relocation R_X86_64_32"
        ),
        fix=_fix_pie_link_error,
        note="added -no-pie linker defines to cmake invocations",
    ),
)


def apply_build_recipes(build_log: str, build_script: str) -> RecipeResult:
    """Apply every matching recipe to `build_script` (idempotent).

    Returns the patched script plus per-recipe notes for the artifact log.
    """
    result = RecipeResult(new_script=build_script)
    script = build_script
    text = build_log or ""
    for recipe in RECIPES:
        if not recipe.signature.search(text):
            continue
        patched = recipe.fix(script, text)
        if patched != script:
            result.matched_kinds.append(recipe.kind)
            result.notes.append(f"{recipe.kind}: {recipe.note}")
            script = patched
    result.new_script = script
    return result
