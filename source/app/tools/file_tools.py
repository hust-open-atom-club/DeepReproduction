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


class FileTool:
    """文件系统操作实现。"""

    def ensure_dir(self, path: str) -> None:
        """确保目录存在。"""

        Path(path).mkdir(parents=True, exist_ok=True)

    def write_text(self, path: str, content: str) -> None:
        """写入文本文件。"""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def read_text(self, path: str) -> str:
        """读取文本文件。"""

        return Path(path).read_text(encoding="utf-8")

    def write_json(self, path: str, payload: Any) -> None:
        """写入 JSON 文件。"""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

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
