"""文件说明：PoC 阶段产物模型。

该模型用于表达"PoC 阶段产出了什么，以及执行结果如何"。
它连接了生成过程和执行过程，是验证阶段最直接的输入之一。
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PoCArtifact(BaseModel):
    """PoC 阶段标准输出。"""

    root_cause_analysis: str = Field(default="", description="漏洞根因分析")
    payload_generation_strategy: str = Field(default="", description="载荷构造策略")
    trigger_mode: str = Field(default="unknown", description="触发模式，例如 cli-file 或 cli-argv")
    trigger_command: str = Field(default="", description="实际触发命令")
    target_binary: str = Field(default="", description="目标二进制或执行入口")
    poc_filename: str = Field(..., description="PoC 主文件名")
    poc_content: str = Field(..., description="PoC 主文件内容")
    run_script_content: str = Field(..., description="运行脚本内容")
    input_files: List[str] = Field(default_factory=list, description="附属输入文件列表")
    input_file_paths: List[str] = Field(default_factory=list, description="输入文件完整路径列表")
    auxiliary_file_paths: List[str] = Field(default_factory=list, description="附属文件完整路径列表")
    expected_error_patterns: List[str] = Field(default_factory=list, description="预期错误模式")
    expected_stdout_patterns: List[str] = Field(default_factory=list, description="预期标准输出模式")
    expected_stderr_patterns: List[str] = Field(default_factory=list, description="预期标准错误模式")
    expected_exit_code: Optional[int] = Field(default=None, description="预期退出码")
    expected_stack_keywords: List[str] = Field(
        default_factory=list,
        description="预期栈关键词（来自 PoC 阶段 plan，可能比 knowledge 增量）",
    )
    expected_crash_type: str = Field(
        default="",
        description="预期崩溃类型（来自 PoC 阶段 plan）",
    )
    environment_variables: Dict[str, str] = Field(
        default_factory=dict,
        description="PoC 执行时使用的环境变量（verify 必须复用这一组）",
    )
    crash_report_content: str = Field(default="", description="崩溃报告内容")
    observed_exit_code: Optional[int] = Field(default=None, description="观测到的退出码")
    observed_stdout: str = Field(default="", description="观测到的标准输出")
    observed_stderr: str = Field(default="", description="观测到的标准错误")
    observed_crash_type: str = Field(default="", description="观测到的崩溃类型")
    matched_error_patterns: List[str] = Field(default_factory=list, description="实际命中的错误模式（向后兼容字段，等价于 matched_stderr_patterns）")
    matched_stdout_patterns: List[str] = Field(
        default_factory=list,
        description="实际命中的 stdout 模式",
    )
    matched_stderr_patterns: List[str] = Field(
        default_factory=list,
        description="实际命中的 stderr 模式（与 matched_error_patterns 同步，命名清晰版）",
    )
    matched_stack_keywords: List[str] = Field(default_factory=list, description="实际命中的栈关键词")
    reproducer_verified: bool = Field(default=False, description="是否已至少一次命中目标行为")
    execution_success: bool = Field(default=False, description="PoC 执行是否成功")
    execution_logs: str = Field(default="", description="PoC 执行日志")
