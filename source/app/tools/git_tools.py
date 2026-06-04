"""文件说明：Git 工具。

这个模块负责仓库获取、版本切换和补丁差异导出，
主要服务于 knowledge 阶段和 build 阶段。

设计上只保留框架真正需要的 Git 能力，避免把所有 Git 操作都堆进来。
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from app.tools.process_tools import ProcessRequest, ProcessTool


class RepositorySnapshot(BaseModel):
    """仓库快照信息。"""

    repo_url: str = Field(..., description="仓库地址")
    local_path: str = Field(..., description="本地仓库路径")
    current_ref: str = Field(default="", description="当前版本引用")


class GitTool:
    """Git 操作实现。"""

    def __init__(self, process_tool: ProcessTool | None = None) -> None:
        self.process_tool = process_tool or ProcessTool()

    def clone_repo(self, repo_url: str, target_dir: str) -> RepositorySnapshot:
        """克隆仓库到指定目录。"""

        target_path = Path(target_dir).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if not target_path.exists():
            result = self.process_tool.run(
                ProcessRequest(command=["git", "clone", repo_url, str(target_path)], cwd=str(target_path.parent))
            )
            if not result.success:
                raise RuntimeError(f"git clone failed: {result.stderr or result.stdout}".strip())

        current_ref = self._resolve_head(str(target_path))
        return RepositorySnapshot(repo_url=repo_url, local_path=str(target_path), current_ref=current_ref)

    def checkout_ref(self, repo_path: str, ref: str) -> RepositorySnapshot:
        """切换到指定 commit、tag 或分支。"""

        fetch_result = self.process_tool.run(ProcessRequest(command=["git", "fetch", "--all", "--tags"], cwd=repo_path))
        if not fetch_result.success:
            # Keep going for local-only repositories; checkout may still succeed.
            pass

        checkout_result = self.process_tool.run(ProcessRequest(command=["git", "checkout", ref], cwd=repo_path))
        if not checkout_result.success:
            raise RuntimeError(f"git checkout failed for ref {ref}: {checkout_result.stderr or checkout_result.stdout}".strip())

        current_ref = self._resolve_head(repo_path)
        return RepositorySnapshot(repo_url="", local_path=repo_path, current_ref=current_ref)

    def export_diff(self, repo_path: str, old_ref: str, new_ref: str) -> str:
        """导出两个版本之间的补丁差异。"""

        result = self.process_tool.run(
            ProcessRequest(command=["git", "diff", f"{old_ref}..{new_ref}"], cwd=repo_path, timeout_seconds=600)
        )
        if not result.success:
            raise RuntimeError(f"git diff failed: {result.stderr or result.stdout}".strip())
        return result.stdout

    def _resolve_head(self, repo_path: str) -> str:
        result = self.process_tool.run(ProcessRequest(command=["git", "rev-parse", "HEAD"], cwd=repo_path))
        if not result.success:
            raise RuntimeError(f"git rev-parse failed: {result.stderr or result.stdout}".strip())
        return result.stdout.strip()
