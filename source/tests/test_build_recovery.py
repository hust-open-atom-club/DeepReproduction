"""文件说明：build_recovery 配方表的单测。

夹具日志全部取自真实 CVE 失败（mdbtools/45926、wolfMQTT/45932、
kimageformats/36083 等），确保签名与修复对本仓库的历史失败形态保持
锁定。所有变换断言幂等（二次应用不再变化）。
"""

from __future__ import annotations

import pytest

from app.stages.build_recovery import apply_build_recipes


LIBTOOLIZE_LOG = (
    "[build] running configure step\n"
    "libtoolize:   error: AC_CONFIG_MACRO_DIRS([m4]) conflicts with "
    "ACLOCAL_AMFLAGS=-I m4\n"
    "autoreconf: libtoolize failed with exit status: 1\n"
)

CONFIG_RPATH_LOG = (
    "configure.ac:116: error: required file 'build-aux/config.rpath' not found\n"
    "autoreconf: automake failed with exit status: 1\n"
)

CONFIGURE_ASAN_LOG = (
    "checking whether the C compiler works... no\n"
    "configure: error: cannot run C compiled programs.\n"
    "If you meant to cross compile, use `--host'.\n"
)

CRLF_LOG = (
    "config.status: error: cannot find input file: `Makefile.in'\n"
)

TIMESTAMP_LOG = (
    "config.status: error: cannot find input file: `po/Makefile.in'\n"
    "make[1]: *** [config.status] Error 1\n"
)

QMAKE_LOG = (
    "Project ERROR: You cannot build examples inside the Qt source tree, "
    "except as part of a proper Qt build.\n"
)


SCRIPT_WITH_CONFIGURE = """#!/bin/bash
set -euo pipefail
log "running configure step"
CC=clang CFLAGS="-fsanitize=address -shared-libasan -fno-pie" LDFLAGS="-fsanitize=address -shared-libasan -no-pie" ./configure --disable-glib
make -j$(nproc)
"""

SCRIPT_PLAIN = """#!/bin/bash
set -euo pipefail
cd "/src/proj"
log "running configure step"
autoreconf -i -f
./configure
make
"""


class TestRecipeSignatures:
    def test_libtoolize_conflict_patched(self):
        result = apply_build_recipes(LIBTOOLIZE_LOG, SCRIPT_PLAIN)
        assert "libtoolize_macro_conflict" in result.matched_kinds
        assert "s/^ACLOCAL_AMFLAGS/#ACLOCAL_AMFLAGS/" in result.new_script

    def test_config_rpath_prefers_autogen(self):
        result = apply_build_recipes(CONFIG_RPATH_LOG, SCRIPT_PLAIN)
        assert "missing_config_rpath" in result.matched_kinds
        assert "autogen.sh" in result.new_script

    def test_configure_asan_stripped(self):
        result = apply_build_recipes(CONFIGURE_ASAN_LOG, SCRIPT_WITH_CONFIGURE)
        assert "configure_cannot_run" in result.matched_kinds
        configure_line = [
            l for l in result.new_script.splitlines() if "./configure" in l
        ][0]
        assert "-shared-libasan" not in configure_line
        assert "-fsanitize" not in configure_line
        assert "./configure" in configure_line  # the invocation survives

    def test_crlf_normalizer_injected(self):
        result = apply_build_recipes(CRLF_LOG, SCRIPT_PLAIN)
        assert "crlf_contamination" in result.matched_kinds
        assert "core.autocrlf false" in result.new_script

    def test_timestamp_pin_injected(self):
        result = apply_build_recipes(TIMESTAMP_LOG, SCRIPT_PLAIN)
        assert "automake_configure_rerun" in result.matched_kinds
        assert "touch configure config.status" in result.new_script

    def test_qmake_examples_fixed(self):
        script = "cd /src/qtsvg && qmake -r && make -j4\n"
        result = apply_build_recipes(QMAKE_LOG, script)
        assert "qmake_examples_in_tree" in result.matched_kinds
        assert "qmake -r" not in result.new_script
        assert "qmake" in result.new_script


class TestRecipeProperties:
    def test_idempotent(self):
        once = apply_build_recipes(LIBTOOLIZE_LOG + CONFIGURE_ASAN_LOG, SCRIPT_WITH_CONFIGURE)
        twice = apply_build_recipes(
            LIBTOOLIZE_LOG + CONFIGURE_ASAN_LOG, once.new_script
        )
        assert not twice.changed
        assert once.new_script == twice.new_script

    def test_no_match_is_noop(self):
        result = apply_build_recipes("make: Leaving directory\n", SCRIPT_PLAIN)
        assert not result.changed
        assert result.new_script == SCRIPT_PLAIN

    def test_empty_log_safe(self):
        result = apply_build_recipes("", SCRIPT_PLAIN)
        assert not result.changed

    def test_real_45926_multi_failure_recovers(self):
        """The exact mdbtools failure (libtoolize + CRLF seen together)."""
        log = (
            "autoreconf: libtoolize failed with exit status: 1\n"
            "libtoolize:   error: AC_CONFIG_MACRO_DIRS([m4]) conflicts with "
            "ACLOCAL_AMFLAGS=-I m4\n"
            "config.status: error: cannot find input file: `.in'\n"
        )
        result = apply_build_recipes(log, SCRIPT_PLAIN)
        assert {"libtoolize_macro_conflict", "crlf_contamination"} <= set(
            result.matched_kinds
        )

MISSING_LIB_LOG = (
    "[build] running configure step\n"
    "checking for libwolfssl... no\n"
    "configure: error: libwolfssl is required for wolfmqtt, "
    "It can be obtained from http://www.wolfssl.com/download.html/\n"
)


def test_missing_required_lib_maps_to_dev_package():
    result = apply_build_recipes(MISSING_LIB_LOG, SCRIPT_PLAIN)
    assert "missing_required_lib" in result.matched_kinds
    assert "libwolfssl-dev" in result.new_script
    assert "liblibwolfssl-dev" not in result.new_script


def test_missing_required_lib_is_idempotent():
    once = apply_build_recipes(MISSING_LIB_LOG, SCRIPT_PLAIN)
    twice = apply_build_recipes(MISSING_LIB_LOG, once.new_script)
    assert twice.new_script.count("libwolfssl-dev") == 1
