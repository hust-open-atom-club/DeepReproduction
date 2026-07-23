"""文件说明：构建阶段产物模型。

该模型用于表达“环境构建阶段最终产生了什么”。
它既包含静态产物，例如 Dockerfile 和构建脚本，
也包含动态结果，例如构建是否成功、日志输出和预期二进制位置。
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class BuildArtifact(BaseModel):
    """构建阶段标准输出。"""

    dockerfile_content: str = Field(..., description="生成的 Dockerfile 内容")
    build_script_content: str = Field(..., description="生成的构建脚本内容")
    install_packages: List[str] = Field(default_factory=list, description="需要安装的依赖")
    build_commands: List[str] = Field(default_factory=list, description="核心构建命令")
    expected_binary_path: Optional[str] = Field(default=None, description="预期产物路径")
    repo_local_path: str = Field(default="", description="本地仓库路径")
    resolved_ref: str = Field(default="", description="实际 checkout 后确认的提交 SHA")
    chosen_vulnerable_ref: str = Field(default="", description="规划阶段选定的漏洞态 ref")
    chosen_fixed_ref: Optional[str] = Field(default=None, description="规划阶段选定的修复态 ref")
    build_system: str = Field(default="", description="确认后的构建系统")
    detected_build_files: List[str] = Field(default_factory=list, description="仓库中扫描到的构建文件")
    dependency_sources: List[str] = Field(default_factory=list, description="依赖与构建信息来源")
    source_of_truth: str = Field(default="", description="构建结论主要来源")
    binary_or_entrypoint: Optional[str] = Field(default=None, description="确认后的二进制或执行入口")
    docker_image_tag: Optional[str] = Field(default=None, description="构建镜像标签")
    compiled_image_tag: Optional[str] = Field(default=None, description="编译成功后提交得到的镜像标签")
    sanitizer_enabled: bool = Field(default=False, description="是否启用 sanitizer")
    build_success: bool = Field(default=False, description="构建是否成功")
    build_logs: str = Field(default="", description="构建日志")
