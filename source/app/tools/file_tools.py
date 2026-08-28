"""文件说明：文件工具。

这个模块负责工作区文件和阶段产物的统一读写。
它的目标不是提供所有文件系统能力，而是为框架约定一套稳定的落盘接口。

适用场景：
- 写入 Dockerfile、build.sh、run.sh
- 保存知识阶段中间产物
- 读取日志和报告文件
- 创建和准备工作区目录
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _normalize_newlines(content: str) -> str:
    """Normalize all newlines to LF so Linux containers can execute scripts."""

    return content.replace("\r\n", "\n").replace("\r", "\n")


class FileTool:
    """文件系统操作实现。"""

    def ensure_dir(self, path: str) -> None:
        """确保目录存在。"""

        Path(path).mkdir(parents=True, exist_ok=True)

    def write_text(self, path: str, content: str) -> None:
        """写入文本文件（强制 LF，避免 Windows CRLF 破坏容器内 shell 脚本）。"""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(_normalize_newlines(content))

    def write_bytes(self, path: str, content: bytes) -> None:
        """写入原始字节（用于含控制字符/非 UTF-8 的 PoC payload）。"""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    def write_latin1(self, path: str, content: str) -> None:
        """按 latin-1 落盘，保留 0x00-0xFF 单字节语义（含 ``\\r``）。

        PoC/fuzz 种子可能依赖精确字节（例如 HDF5 魔数 ``\\x89HDF\\r\\n\\x1a\\n``）。
        换行归一只用于 ``write_text`` 的脚本/配置，不要在这里做。
        """

        self.write_bytes(path, content.encode("latin-1"))

    def read_text(self, path: str) -> str:
        """读取文本文件。"""

        return Path(path).read_text(encoding="utf-8")

    def write_json(self, path: str, payload: Any) -> None:
        """写入 JSON 文件。"""

        self.write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))

    def exists(self, path: str) -> bool:
        """判断路径是否存在。"""

        return Path(path).exists()

    def safe_persist(self, path: str, content: str, description: str = "") -> bool:
        """Best-effort persist. Returns True on success, False on failure (with stderr warning).

        与 write_text 不同的是：失败时不抛异常，而是打 stderr 警告并返回 False。
        用于"非致命落盘"场景——即便落盘失败也不应让主流程崩溃。
        """

        try:
            self.write_text(path, content)
            return True
        except Exception as error:
            sys.stderr.write(
                f"[WARN] failed to persist {description or path}: {error}\n"
            )
            return False
