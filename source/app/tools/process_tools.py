"""文件说明：进程执行工具。

这个模块统一封装外部命令执行行为，
供 Git、Docker、构建脚本和运行脚本等调用方复用。

这样可以把超时控制、输出捕获、退出码判断等横切逻辑收敛在一处。
"""

from __future__ import annotations

import os
import subprocess
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ProcessRequest(BaseModel):
    """外部命令执行请求。"""

    command: List[str] = Field(default_factory=list, description="命令及参数")
    cwd: Optional[str] = Field(default=None, description="工作目录")
    timeout_seconds: int = Field(default=300, description="超时时间")
    environment: Dict[str, str] = Field(default_factory=dict, description="环境变量")


class ProcessResult(BaseModel):
    """外部命令执行结果。"""

    success: bool = Field(default=False, description="命令是否成功")
    exit_code: int = Field(default=1, description="退出码")
    stdout: str = Field(default="", description="标准输出")
    stderr: str = Field(default="", description="标准错误")


class ProcessTool:
    """命令执行实现。"""

    def run(self, request: ProcessRequest) -> ProcessResult:
        """执行一个外部命令。"""

        environment = os.environ.copy()
        environment.update(request.environment)

        try:
            completed = subprocess.run(
                request.command,
                cwd=request.cwd,
                env=environment,
                text=True,
                capture_output=True,
                timeout=request.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout or ""
            stderr = error.stderr or ""
            return ProcessResult(
                success=False,
                exit_code=124,
                stdout=stdout,
                stderr=f"{stderr}\nprocess timed out after {request.timeout_seconds} seconds".strip(),
            )
        except FileNotFoundError as error:
            return ProcessResult(success=False, exit_code=127, stdout="", stderr=str(error))

        return ProcessResult(
            success=completed.returncode == 0,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
