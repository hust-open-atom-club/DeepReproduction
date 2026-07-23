"""文件说明：Docker 工具。

这个模块负责封装所有与容器构建和容器执行有关的动作，
供 build 阶段和 poc 阶段复用。

它只表达“需要哪些 Docker 能力”，不在这里绑定具体命令实现。
这样后续既可以接 Docker CLI，也可以接 SDK。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.tools.process_tools import ProcessRequest, ProcessTool


class DockerBuildRequest(BaseModel):
    """镜像构建请求。"""

    workspace: str = Field(..., description="构建上下文目录")
    dockerfile_path: str = Field(..., description="Dockerfile 路径")
    image_tag: str = Field(..., description="目标镜像标签")
    build_args: Dict[str, str] = Field(default_factory=dict, description="构建参数")
    network_mode: Optional[str] = Field(default=None, description="可选 docker build 网络模式")


class DockerRunRequest(BaseModel):
    """容器运行请求。"""

    image_tag: str = Field(..., description="待运行镜像标签")
    command: List[str] = Field(default_factory=list, description="容器内执行命令")
    workspace: Optional[str] = Field(default=None, description="宿主机工作区路径，将挂载到容器内的 /workspace")
    environment: Dict[str, str] = Field(default_factory=dict, description="环境变量")
    container_name: Optional[str] = Field(default=None, description="可选容器名称")
    remove: bool = Field(default=True, description="运行结束后是否自动删除容器")


class DockerCommandResult(BaseModel):
    """Docker 命令执行结果。"""

    success: bool = Field(default=False, description="命令是否成功")
    exit_code: int = Field(default=1, description="进程退出码")
    stdout: str = Field(default="", description="标准输出")
    stderr: str = Field(default="", description="标准错误")


class DockerTool:
    """Docker CLI 实现。"""

    def __init__(self, process_tool: ProcessTool | None = None) -> None:
        self.process_tool = process_tool or ProcessTool()

    def build_image(self, request: DockerBuildRequest) -> DockerCommandResult:
        """根据请求构建镜像。"""

        command = [
            "docker",
            "build",
            "-f",
            request.dockerfile_path,
            "-t",
            request.image_tag,
        ]
        if request.network_mode:
            command.extend(["--network", request.network_mode])
        for key, value in request.build_args.items():
            command.extend(["--build-arg", f"{key}={value}"])
        command.append(request.workspace)

        result = self.process_tool.run(ProcessRequest(command=command, cwd=request.workspace, timeout_seconds=1800))
        return DockerCommandResult(
            success=result.success,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def run_container(self, request: DockerRunRequest) -> DockerCommandResult:
        """运行容器并返回执行结果。"""

        command = ["docker", "run"]
        if request.remove:
            command.append("--rm")
        if request.container_name:
            command.extend(["--name", request.container_name])
        if request.workspace:
            command.extend(["-v", f"{request.workspace}:/workspace", "-w", "/workspace"])
        for key, value in request.environment.items():
            command.extend(["-e", f"{key}={value}"])
        command.append(request.image_tag)
        command.extend(request.command)

        result = self.process_tool.run(ProcessRequest(command=command, timeout_seconds=1800))
        return DockerCommandResult(
            success=result.success,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def remove_image(self, image_tag: str) -> None:
        """删除临时镜像。"""

        self.process_tool.run(ProcessRequest(command=["docker", "rmi", "-f", image_tag], timeout_seconds=300))

    def commit_container(self, container_name: str, image_tag: str) -> DockerCommandResult:
        result = self.process_tool.run(
            ProcessRequest(command=["docker", "commit", container_name, image_tag], timeout_seconds=300)
        )
        return DockerCommandResult(
            success=result.success,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def remove_container(self, container_name: str) -> None:
        self.process_tool.run(ProcessRequest(command=["docker", "rm", "-f", container_name], timeout_seconds=120))
