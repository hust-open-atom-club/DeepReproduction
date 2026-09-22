"""文件说明：Git 工具。

这个模块负责仓库获取、版本切换和补丁差异导出，
主要服务于 knowledge 阶段和 build 阶段。

设计上只保留框架真正需要的 Git 能力，避免把所有 Git 操作都堆进来。
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from app.tools.process_tools import ProcessRequest, ProcessResult, ProcessTool

_PROXY_FAILURE_TOKENS = (
    "could not connect to server",
    "failed to connect to",
    "connection refused",
    "could not resolve proxy",
    "proxy connect aborted",
    "timed out",
)

_NO_PROXY_ENV_NAMES = ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")

_GIT_MIRRORS_ENV = "DEEPREPRO_GIT_MIRRORS"
_DEFAULT_GIT_MIRRORS = ("https://gh-proxy.com/", "https://ghfast.top/")


def looks_like_proxy_failure(result: ProcessResult) -> bool:
    """识别「配置了代理但代理不可达」类网络失败。"""

    text = f"{result.stderr}\n{result.stdout}".lower()
    return any(token in text for token in _PROXY_FAILURE_TOKENS)


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
        """克隆仓库到指定目录。

        四层降级：直连 → 去代理重试 → 镜像前缀克隆 → Docker 代克隆
        （守护进程出口独立于宿主机）。
        """

        target_path = Path(target_dir).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        self._purge_invalid_repo_dir(target_path)

        if not target_path.exists():
            result = self.process_tool.run(
                ProcessRequest(
                    # LF working tree: the host repo is COPYed into build images,
                    # and CRLF scripts break bash/autoreconf (e.g. ovs boot.sh).
                    command=["git", "-c", "core.autocrlf=false", "clone", repo_url, str(target_path)],
                    cwd=str(target_path.parent),
                    timeout_seconds=1800,
                )
            )
            if not result.success and looks_like_proxy_failure(result):
                self._purge_invalid_repo_dir(target_path)
                result = self._merge_failure(
                    result,
                    self.process_tool.run(
                        ProcessRequest(
                            command=self._no_proxy_git_command(["clone", repo_url, str(target_path)]),
                            cwd=str(target_path.parent),
                            timeout_seconds=1800,
                            environment=self._no_proxy_environment(),
                        )
                    ),
                )
            if not result.success:
                # 镜像克隆：先全量；超大仓库（gdal 等）镜像代理常拒绝/超时，
                # 回退 blobless（--filter=blob:none --no-checkout，blob 在
                # checkout 时经镜像 origin 按需拉取）。
                for mirror in self._git_mirrors():
                    for extra in ([], ["--filter=blob:none", "--no-checkout"]):
                        self._purge_invalid_repo_dir(target_path)
                        mirrored = self.process_tool.run(
                            ProcessRequest(
                                command=self._no_proxy_git_command(
                                    ["clone", *extra, f"{mirror}{repo_url}", str(target_path)]
                                ),
                                cwd=str(target_path.parent),
                                timeout_seconds=1800,
                                environment=self._no_proxy_environment(),
                            )
                        )
                        if mirrored.success:
                            result = mirrored
                            break
                        result = self._merge_failure(result, mirrored)
                    if result.success:
                        break
            if not result.success:
                self._purge_invalid_repo_dir(target_path)
                result = self._merge_failure(result, self._docker_clone(repo_url, target_path))
            if not result.success:
                raise RuntimeError(f"git clone failed: {result.stderr or result.stdout}".strip())
            self._pin_local_autocrlf_false(str(target_path))

        current_ref = self._resolve_head(str(target_path))
        return RepositorySnapshot(repo_url=repo_url, local_path=str(target_path), current_ref=current_ref)

    def _pin_local_autocrlf_false(self, repo_path: str) -> None:
        """钉死 repo 级 autocrlf=false。

        Git for Windows 默认全局 core.autocrlf=true：克隆时用 `-c` 覆盖只是
        一次性生效，后续 build 阶段不带参数的 `git checkout vulnerable_ref`
        会把整棵工作树重新涂抹成 CRLF（configure 报 `/bin/sh^M: bad
        interpreter`，qpdf/36978 实证）。repo 级配置对所有后续 git 操作生效。
        """

        self.process_tool.run(ProcessRequest(command=["git", "-C", repo_path, "config", "core.autocrlf", "false"]))

    def _no_proxy_git_command(self, git_args: list[str]) -> list[str]:
        """构造绕过全局代理配置的 git 命令。"""

        return ["git", "-c", "http.proxy=", "-c", "https.proxy=", "-c", "core.autocrlf=false"] + git_args

    def _git_mirrors(self) -> list[str]:
        """镜像前缀列表；DEEPREPRO_GIT_MIRRORS（逗号/空白分隔）可覆盖默认。"""

        raw = os.environ.get(_GIT_MIRRORS_ENV, "").strip()
        if not raw:
            return list(_DEFAULT_GIT_MIRRORS)
        mirrors = [item.strip().rstrip("/") + "/" for item in re.split(r"[,\s]+", raw) if item.strip()]
        return mirrors or list(_DEFAULT_GIT_MIRRORS)

    def _docker_clone(self, repo_url: str, target_path: Path) -> ProcessResult:
        """宿主机网络不可用时，用 Docker 守护进程的网络代为克隆。"""

        mount_src = str(target_path.parent).replace("\\", "/")
        return self.process_tool.run(
            ProcessRequest(
                command=[
                    "docker",
                    "run",
                    "--rm",
                    "-v",
                    f"{mount_src}:/w",
                    "alpine/git",
                    "-c",
                    "core.autocrlf=false",
                    "clone",
                    repo_url,
                    f"/w/{target_path.name}",
                ],
                timeout_seconds=1800,
            )
        )

    def _purge_invalid_repo_dir(self, target_path: Path) -> None:
        """失败克隆可能留下空目录/半成品；不是有效 git 仓库就清掉，避免后续跳过克隆。"""

        if not target_path.exists():
            return
        if (target_path / ".git").exists():
            return
        shutil.rmtree(target_path, ignore_errors=False)

    def _merge_failure(self, primary: ProcessResult, fallback: ProcessResult) -> ProcessResult:
        """保留首次失败的报错上下文，仅在兜底成功时整体替换。"""

        return fallback if fallback.success else primary

    def checkout_ref(self, repo_path: str, ref: str) -> RepositorySnapshot:
        """切换到指定 commit、tag 或分支。"""

        fetch_result = self.process_tool.run(ProcessRequest(command=["git", "fetch", "--all", "--tags"], cwd=repo_path))
        if not fetch_result.success and looks_like_proxy_failure(fetch_result):
            fetch_result = self.process_tool.run(
                ProcessRequest(
                    command=["git", "-c", "http.proxy=", "-c", "https.proxy=", "fetch", "--all", "--tags"],
                    cwd=repo_path,
                    timeout_seconds=1800,
                    environment=self._no_proxy_environment(),
                )
            )
        if not fetch_result.success:
            # Keep going for local-only repositories; checkout may still succeed.
            pass

        # 全局 autocrlf=true 会在 checkout 时把 LF 树涂抹成 CRLF（见
        # _pin_local_autocrlf_false）；已有仓库未钉配置时此处兜底。
        self._pin_local_autocrlf_false(repo_path)

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

    def _no_proxy_environment(self) -> dict[str, str]:
        """用空值压掉代理环境变量（libcurl 视空串为未设置）。"""

        return {name: "" for name in _NO_PROXY_ENV_NAMES}
