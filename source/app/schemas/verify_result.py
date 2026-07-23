"""文件说明：验证阶段结果模型。

该模型是整个主流程的最终判定输出。
它描述的不是"PoC 是否运行了"，而是"漏洞是否按预期被触发和修复"。
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class VerifyResult(BaseModel):
    """验证阶段标准输出。"""

    # ===== 现有字段（保留语义）=====
    pre_patch_triggered: bool = Field(..., description="补丁前是否成功触发漏洞")
    post_patch_clean: bool = Field(..., description="补丁后是否不再触发漏洞")
    matched_error_patterns: List[str] = Field(default_factory=list, description="pre 阶段命中的错误模式（向后兼容）")
    matched_stack_keywords: List[str] = Field(default_factory=list, description="pre 阶段命中的栈关键词（向后兼容）")
    verdict: str = Field(..., description="最终结论：success | failed | inconclusive")
    reason: str = Field(..., description="最终结论说明")

    # ===== pre / post 详细观测 =====
    pre_patch_exit_code: Optional[int] = Field(default=None, description="pre 模式下的 execution_exit_code")
    post_patch_exit_code: Optional[int] = Field(default=None, description="post 模式下的 execution_exit_code")
    pre_patch_observed_stdout: str = Field(default="", description="pre 模式下的 stdout 块内容")
    pre_patch_observed_stderr: str = Field(default="", description="pre 模式下的 stderr 块内容")
    post_patch_observed_stdout: str = Field(default="", description="post 模式下的 stdout 块内容")
    post_patch_observed_stderr: str = Field(default="", description="post 模式下的 stderr 块内容")
    pre_patch_observed_crash_type: str = Field(default="", description="pre 模式下识别出的崩溃类型")
    post_patch_observed_crash_type: str = Field(default="", description="post 模式下识别出的崩溃类型")
    pre_patch_log_path: str = Field(default="", description="pre 模式完整日志的相对路径")
    post_patch_log_path: str = Field(default="", description="post 模式完整日志的相对路径")
    pre_patch_matched_error_patterns: List[str] = Field(default_factory=list, description="pre 模式命中的错误模式（向后兼容，等价于 pre_patch_matched_stderr_patterns）")
    pre_patch_matched_stack_keywords: List[str] = Field(default_factory=list, description="pre 模式命中的栈关键词")
    post_patch_matched_error_patterns: List[str] = Field(default_factory=list, description="post 模式命中的错误模式（向后兼容，等价于 post_patch_matched_stderr_patterns）")
    post_patch_matched_stack_keywords: List[str] = Field(default_factory=list, description="post 模式命中的栈关键词")
    pre_patch_matched_stdout_patterns: List[str] = Field(
        default_factory=list,
        description="pre 模式下命中的 stdout 模式",
    )
    pre_patch_matched_stderr_patterns: List[str] = Field(
        default_factory=list,
        description="pre 模式下命中的 stderr 模式（与 matched_error_patterns 同步）",
    )
    post_patch_matched_stdout_patterns: List[str] = Field(
        default_factory=list,
        description="post 模式下命中的 stdout 模式",
    )
    post_patch_matched_stderr_patterns: List[str] = Field(
        default_factory=list,
        description="post 模式下命中的 stderr 模式",
    )

    # ===== patch 应用情况 =====
    patch_apply_log: str = Field(default="", description="post 模式 git apply 输出摘录")
    patch_apply_success: bool = Field(default=False, description="post 模式下 git apply 是否成功")

    # ===== 综合判定辅助 =====
    confidence: str = Field(default="low", description="置信度：high | medium | low")
    evidence_summary: str = Field(default="", description="人类可读的简短证据总结")
