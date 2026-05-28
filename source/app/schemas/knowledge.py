"""文件说明：漏洞知识模型。

该模型是知识阶段的标准输出，也是后续 build / poc / verify 三个阶段的公共输入。
它的职责不是保存所有原始资料，而是保存“经过提炼后真正对复现有价值的信息”。
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ReproductionRecipe(BaseModel):
    """Structured reproduction recipe extracted from evidence."""

    source_url: str = Field(default="", description="Source URL that contained the reproduction evidence.")
    source_title: str = Field(default="", description="Page title for the evidence source.")
    recipe_type: str = Field(default="unknown", description="Recipe style such as shell_sequence or mixed.")
    steps: List[str] = Field(default_factory=list, description="Ordered reproduction steps preserved from the evidence.")
    repo_setup_commands: List[str] = Field(default_factory=list, description="Repository setup commands such as clone or checkout.")
    build_commands: List[str] = Field(default_factory=list, description="Commands used to compile or build the target.")
    artifact_generation_commands: List[str] = Field(default_factory=list, description="Commands that generate payloads or helper files.")
    run_commands: List[str] = Field(default_factory=list, description="Commands that execute the reproducer against the target.")
    expected_behavior: List[str] = Field(default_factory=list, description="Expected crash signatures or observable behavior.")
    source_excerpt: str = Field(default="", description="Original evidence excerpt used to derive this recipe.")
    confidence: str = Field(default="medium", description="Confidence level assigned to the extraction.")


class KnowledgeModel(BaseModel):
    """漏洞知识的结构化表达。"""

    cve_id: str = Field(..., description="漏洞编号")
    summary: str = Field(..., description="漏洞摘要")
    vulnerability_type: str = Field(..., description="漏洞类型")
    repo_url: Optional[str] = Field(default=None, description="源码仓库地址")
    vulnerable_ref: Optional[str] = Field(default=None, description="漏洞版本引用")
    fixed_ref: Optional[str] = Field(default=None, description="修复版本引用")
    affected_files: List[str] = Field(default_factory=list, description="可能受影响的文件")
    build_systems: List[str] = Field(default_factory=list, description="推断得到的构建系统，例如 make、cmake、cargo。")
    build_files: List[str] = Field(default_factory=list, description="构建相关文件，例如 Makefile、Dockerfile、go.mod。")
    install_commands: List[str] = Field(default_factory=list, description="从证据中提取的安装或依赖准备命令。")
    build_commands: List[str] = Field(default_factory=list, description="从证据中提取的编译或构建命令。")
    build_hints: List[str] = Field(default_factory=list, description="与构建过程相关的提炼提示。")
    reproduction_hints: List[str] = Field(default_factory=list, description="复现提示信息")
    reproduction_recipes: List[ReproductionRecipe] = Field(default_factory=list, description="从网页原文中提取并保留的结构化复现流程。")
    expected_error_patterns: List[str] = Field(default_factory=list, description="预期报错模式")
    expected_stack_keywords: List[str] = Field(default_factory=list, description="预期栈关键词")
    references: List[str] = Field(default_factory=list, description="知识阶段保留的参考资料")
