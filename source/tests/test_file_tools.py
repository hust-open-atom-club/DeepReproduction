"""Regression tests for FileTool newline handling on Windows."""

from pathlib import Path

from app.tools.file_tools import FileTool


def test_write_text_forces_lf_newlines(tmp_path: Path) -> None:
    tool = FileTool()
    target = tmp_path / "build.sh"

    tool.write_text(str(target), "#!/bin/bash\r\nset -euo pipefail\r\necho ok\r")

    raw = target.read_bytes()
    assert b"\r" not in raw
    assert raw == b"#!/bin/bash\nset -euo pipefail\necho ok\n"


def test_write_latin1_preserves_control_bytes(tmp_path: Path) -> None:
    tool = FileTool()
    target = tmp_path / "poc.lua"

    tool.write_latin1(str(target), 'ee=""return\x0cc""\n')

    assert target.read_bytes() == b'ee=""return\x0cc""\n'


def test_write_latin1_preserves_cr_in_binary_payload(tmp_path: Path) -> None:
    """HDF5 / MAT7.3 seeds must keep ``\\r``; LF-normalize would break the signature."""

    tool = FileTool()
    target = tmp_path / "seed.mat"
    # 512-byte MAT userblock prefix + HDF5 signature
    content = ("\x00" * 512) + "\x89HDF\r\n\x1a\n" + "payload"

    tool.write_latin1(str(target), content)

    raw = target.read_bytes()
    assert raw[512:520] == b"\x89HDF\r\n\x1a\n"
    assert b"\r" in raw
    assert raw == content.encode("latin-1")
