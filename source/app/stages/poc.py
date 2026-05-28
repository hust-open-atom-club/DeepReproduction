"""文件说明：PoC 生成与执行阶段实现。

这个模块负责“把漏洞知识和构建结果转成最小复现载荷，并执行它”。
它遵循与 build 阶段一致的结构：
- 收集本地证据
- 生成结构化 PoC 计划
- 写入 PoC 文件和运行脚本
- 在 Docker 中执行并提取观察结果
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Optional, TypedDict

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except ModuleNotFoundError:  # pragma: no cover
    Environment = None
    FileSystemLoader = None
    StrictUndefined = None

from app.config import build_chat_model
from app.schemas.build_artifact import BuildArtifact
from app.schemas.knowledge import KnowledgeModel, ReproductionRecipe
from app.schemas.poc_artifact import PoCArtifact
from app.stages.build import parse_llm_json_payload
from app.tools.docker_tools import DockerBuildRequest, DockerRunRequest, DockerTool
from app.tools.file_tools import FileTool
from app.tools.log_parsing import (
    extract_block as _extract_block_module,
    extract_execution_observation as _extract_execution_observation_module,
    match_patterns as _match_patterns_module,
)
from app.tools.patch_tools import find_patch_diff


class PocStagePaths:
    """Filesystem layout owned by the PoC stage."""

    def __init__(self, workspace: str) -> None:
        self.workspace_root = Path(workspace)
        self.repo_dir = self.workspace_root / "repo"
        self.artifacts_dir = self.workspace_root / "artifacts"
        self.build_dir = self.artifacts_dir / "build"
        self.poc_dir = self.artifacts_dir / "poc"
        self.verify_dir = self.artifacts_dir / "verify"
        self.llm_dir = self.poc_dir / "llm"
        self.payloads_dir = self.poc_dir / "payloads"
        self.inputs_dir = self.poc_dir / "inputs"
        self.poc_context_yaml = self.poc_dir / "poc_context.yaml"
        self.poc_plan_yaml = self.poc_dir / "poc_plan.yaml"
        self.dockerfile = self.poc_dir / "Dockerfile"
        self.run_script = self.poc_dir / "run.sh"
        self.poc_log = self.poc_dir / "poc.log"
        self.crash_report = self.poc_dir / "crash_report.txt"
        self.poc_artifact_yaml = self.poc_dir / "poc_artifact.yaml"
        self.run_verify_yaml = self.poc_dir / "run_verify.yaml"


class PocContext(BaseModel):
    """Collected local evidence consumed by the PoC planner."""

    cve_id: str = Field(..., description="Target CVE identifier.")
    repo_url: str = Field(default="", description="Repository URL.")
    resolved_ref: str = Field(default="", description="Resolved vulnerable ref.")
    repo_local_path: str = Field(default="", description="Local repository path.")
    build_system: str = Field(default="", description="Build system selected by build stage.")
    build_success: bool = Field(default=False, description="Whether build stage succeeded.")
    target_binary: str = Field(default="", description="Binary or entrypoint identified by build stage.")
    patch_diff_excerpt: str = Field(default="", description="Short patch excerpt.")
    patch_affected_files: list[str] = Field(default_factory=list, description="Files touched by patch.")
    patch_changed_functions: list[str] = Field(default_factory=list, description="Functions mentioned in patch hunks.")
    patch_added_checks: list[str] = Field(default_factory=list, description="Guard checks added by the patch.")
    patch_error_strings: list[str] = Field(default_factory=list, description="Interesting error strings or sanitizer markers.")
    inferred_input_modes: list[str] = Field(default_factory=list, description="Likely trigger input modes inferred from evidence.")
    knowledge_summary: str = Field(default="", description="Knowledge summary.")
    reproduction_hints: list[str] = Field(default_factory=list, description="Reproduction hints from knowledge.")
    reproduction_recipe_summaries: list[str] = Field(default_factory=list, description="Structured reproduction recipe summaries.")
    expected_error_patterns: list[str] = Field(default_factory=list, description="Expected error patterns.")
    expected_stack_keywords: list[str] = Field(default_factory=list, description="Expected stack keywords.")
    candidate_entrypoints: list[str] = Field(default_factory=list, description="Candidate entrypoints.")
    candidate_trigger_files: list[str] = Field(default_factory=list, description="Files likely related to triggering.")
    candidate_cli_flags: list[str] = Field(default_factory=list, description="Command-line flags discovered from hints.")
    reference_poc_summaries: list[str] = Field(default_factory=list, description="Reference PoC summaries.")
    repo_evidence_blocks: list[str] = Field(default_factory=list, description="README/tests/examples excerpts.")
    previous_failure_kind: str = Field(default="", description="Previous PoC failure kind.")
    previous_execution_log: str = Field(default="", description="Previous PoC execution log excerpt.")
    previous_run_script_content: str = Field(default="", description="Previous rendered run script content.")
    previous_payload_content: str = Field(default="", description="Previous primary payload content.")
    previous_run_verify_report: str = Field(default="", description="Previous run_verify.yaml content.")
    planner_attempt: int = Field(default=1, description="Planner attempt number.")


class PocPlan(BaseModel):
    """Structured PoC plan produced by rules or the PoC LLM."""

    trigger_mode: str = Field(default="cli-file", description="Trigger mode.")
    target_binary: str = Field(default="", description="Binary or script used to trigger.")
    target_args: list[str] = Field(default_factory=list, description="Arguments passed to the target.")
    environment_variables: dict[str, str] = Field(default_factory=dict, description="Environment variables for execution.")
    payload_filename: str = Field(default="poc.txt", description="Primary payload filename.")
    payload_content: str = Field(default="", description="Primary payload content.")
    auxiliary_files: dict[str, str] = Field(default_factory=dict, description="Auxiliary files to write.")
    run_command: str = Field(default="", description="Command executed inside the container.")
    expected_exit_code: Optional[int] = Field(default=None, description="Expected exit code.")
    expected_stdout_patterns: list[str] = Field(default_factory=list, description="Expected stdout patterns.")
    expected_stderr_patterns: list[str] = Field(default_factory=list, description="Expected stderr patterns.")
    expected_stack_keywords: list[str] = Field(default_factory=list, description="Expected stack keywords.")
    expected_crash_type: str = Field(default="", description="Expected crash type.")
    source_of_truth: str = Field(default="heuristic", description="Primary evidence source.")
    confidence: str = Field(default="medium", description="Planner confidence.")
    rationale: str = Field(default="", description="Short rationale.")
    dockerfile_override: Optional[str] = Field(default=None, description="Optional full Dockerfile override.")
    run_script_override: Optional[str] = Field(default=None, description="Optional full run script override.")


class RunVerifyReport(BaseModel):
    """Minimum-eligibility report for one PoC execution.

    本报告回答一个问题：这一次 PoC 执行是否构成进入 verify agent 的资格。
    它不裁决漏洞是否被复现，只裁决 PoC 这一次"打到目标行为"的可信度。
    """

    script_finished: bool = Field(
        default=False,
        description="run.sh 是否完整跑完。从日志中观察到 execution_exit_code= 行即为 True。",
    )
    log_well_formed: bool = Field(
        default=False,
        description="日志契约是否完整：stdout_begin/end 与 stderr_begin/end 两对标记块是否都出现。",
    )
    target_binary_invoked: bool = Field(
        default=False,
        description="日志中是否出现 target_binary= 行，用于确认 run.sh 跑到了目标二进制调用前。",
    )
    exit_code_observed: Optional[int] = Field(
        default=None,
        description="观测到的 execution_exit_code 值；未观测到时为 None。",
    )
    error_pattern_hits: list[str] = Field(
        default_factory=list,
        description="实际命中的 expected_stderr_patterns 列表（仅指 stderr 流命中）。",
    )
    stdout_pattern_hits: list[str] = Field(
        default_factory=list,
        description="实际命中的 expected_stdout_patterns 列表。",
    )
    stack_keyword_hits: list[str] = Field(
        default_factory=list,
        description="实际命中的 expected_stack_keywords 列表。",
    )
    crash_type_hit: str = Field(
        default="",
        description="日志里识别出的崩溃类型字符串；未识别为空字符串。",
    )
    crash_type_compatible: Optional[bool] = Field(
        default=None,
        description="crash_type_hit 是否与 plan.expected_crash_type 兼容（包含或被包含，大小写不敏感）。"
                    "expected_crash_type 为空时为 None。",
    )
    exit_code_match_expected: Optional[bool] = Field(
        default=None,
        description="exit_code_observed 是否等于 plan.expected_exit_code；"
                    "plan.expected_exit_code 为 None 时本字段为 None。",
    )
    eligible_for_verify: bool = Field(
        default=False,
        description="综合判定结果：这一次 PoC 是否构成进入 verify 的资格。",
    )
    eligibility_reason: str = Field(
        default="",
        description="eligible_for_verify 取值的简短原因，便于人工诊断。",
    )
    evidence_log_excerpt: str = Field(
        default="",
        description="关键日志摘录，最多 2048 字节。",
    )


class PocPreparedRun(BaseModel):
    """Deterministic inputs assembled before PoC planning starts."""

    plan_meta: dict[str, Any]
    context: PocContext


class PocExecutionOutcome(BaseModel):
    """One concrete PoC execution attempt."""

    plan: PocPlan
    artifact: PoCArtifact


class PocGraphState(TypedDict, total=False):
    """Internal LangGraph state for the PoC stage."""

    knowledge: KnowledgeModel
    build: BuildArtifact
    paths: PocStagePaths
    prepared: PocPreparedRun
    current_context: PocContext
    current_plan: PocPlan
    outcome: PocExecutionOutcome
    attempt: int


class PocFallbackSpec(BaseModel):
    """Deterministic fallback planning spec for the PoC stage."""

    trigger_mode: str = "cli-file"
    target_binary: str = ""
    target_args: list[str] = Field(default_factory=list)
    payload_filename: str = "poc.txt"
    payload_content: str = ""
    run_command: str = ""
    expected_stdout_patterns: list[str] = Field(default_factory=list)
    expected_stderr_patterns: list[str] = Field(default_factory=list)
    expected_stack_keywords: list[str] = Field(default_factory=list)
    expected_crash_type: str = ""
    source_of_truth: str = "heuristic"
    confidence: str = "medium"
    rationale: str = ""


class PocPlanner:
    """Encapsulates PoC-stage planning decisions."""

    def __init__(self, stage: "PocStage") -> None:
        self.stage = stage

    def plan(self, knowledge: KnowledgeModel, build: BuildArtifact, context: PocContext) -> PocPlan:
        llm_plan = self.try_llm_plan(knowledge=knowledge, build=build, context=context)
        if llm_plan is not None:
            return self.stage._normalize_poc_plan(llm_plan, repo_url=context.repo_url)
        return self.stage._normalize_poc_plan(
            self.heuristic_plan(knowledge=knowledge, build=build, context=context),
            repo_url=context.repo_url,
        )

    def replan_after_failure(
        self,
        knowledge: KnowledgeModel,
        build: BuildArtifact,
        context: PocContext,
        previous_plan: PocPlan,
        previous_artifact: PoCArtifact,
    ) -> Optional[PocPlan]:
        retry_context = context.model_copy(
            update={
                "planner_attempt": context.planner_attempt + 1,
                "previous_failure_kind": self.stage._classify_failure_kind(previous_artifact.execution_logs),
                "previous_execution_log": previous_artifact.execution_logs[:6000],
            }
        )
        return self.try_llm_plan(
            knowledge=knowledge,
            build=build,
            context=retry_context,
            previous_plan=previous_plan,
            previous_artifact=previous_artifact,
        )

    def heuristic_plan(self, knowledge: KnowledgeModel, build: BuildArtifact, context: PocContext) -> PocPlan:
        spec = self.stage._build_fallback_spec(knowledge=knowledge, build=build, context=context)
        return PocPlan(
            trigger_mode=spec.trigger_mode,
            target_binary=spec.target_binary,
            target_args=spec.target_args,
            payload_filename=spec.payload_filename,
            payload_content=spec.payload_content,
            run_command=spec.run_command,
            expected_exit_code=None,
            expected_stdout_patterns=spec.expected_stdout_patterns,
            expected_stderr_patterns=spec.expected_stderr_patterns,
            expected_stack_keywords=spec.expected_stack_keywords,
            expected_crash_type=spec.expected_crash_type,
            source_of_truth=spec.source_of_truth,
            confidence=spec.confidence,
            rationale=spec.rationale,
        )

    def try_llm_plan(
        self,
        knowledge: KnowledgeModel,
        build: BuildArtifact,
        context: PocContext,
        previous_plan: Optional[PocPlan] = None,
        previous_artifact: Optional[PoCArtifact] = None,
    ) -> Optional[PocPlan]:
        try:
            model = build_chat_model("poc_agent", temperature=0)
        except Exception:
            return None

        prompt = self.stage._build_llm_prompt(
            knowledge=knowledge,
            build=build,
            context=context,
            previous_plan=previous_plan,
            previous_artifact=previous_artifact,
        )
        self.stage._persist_poc_llm_trace(context.planner_attempt, "prompt.txt", prompt)
        retry_errors: list[str] = []
        max_attempts = self.stage.MAX_LLM_NO_RESPONSE_RETRIES + 1
        for invoke_attempt in range(1, max_attempts + 1):
            try:
                response = model.invoke(
                    [
                        SystemMessage(content="You return strict JSON only."),
                        HumanMessage(content=prompt),
                    ]
                )
                raw_response = getattr(response, "content", response)
                raw_response_text = str(raw_response)
                if self.stage._is_empty_llm_response(raw_response_text):
                    retry_errors.append(f"Attempt {invoke_attempt}: empty response")
                    if invoke_attempt < max_attempts:
                        continue
                    self.stage._persist_poc_llm_trace(
                        context.planner_attempt,
                        "error.txt",
                        "LLM returned no content after 3 attempts.",
                    )
                    return None
                self.stage._persist_poc_llm_trace(context.planner_attempt, "response.txt", raw_response_text)
                parsed = parse_llm_json_payload(raw_response)
                if parsed is None:
                    self.stage._persist_poc_llm_trace(
                        context.planner_attempt,
                        "error.txt",
                        "LLM response could not be parsed into JSON.",
                    )
                    return None
                self.stage._persist_poc_llm_trace(
                    context.planner_attempt,
                    "parsed.json",
                    json.dumps(parsed, ensure_ascii=False, indent=2),
                )
                plan = PocPlan(**parsed)
                if not plan.target_binary and not plan.run_command:
                    self.stage._persist_poc_llm_trace(
                        context.planner_attempt,
                        "error.txt",
                        "LLM plan was missing both target_binary and run_command.",
                    )
                    return None
                if previous_plan is not None:
                    failure_kind = context.previous_failure_kind or self.stage._classify_failure_kind(previous_artifact.execution_logs if previous_artifact else "")
                    normalized_plan = self.stage._normalize_poc_plan(plan, repo_url=context.repo_url)
                    if not self.stage._is_valid_replan_candidate(previous_plan, normalized_plan, failure_kind=failure_kind):
                        self.stage._persist_poc_llm_trace(
                            context.planner_attempt,
                            "error.txt",
                            f"Rejected replan candidate for failure kind: {failure_kind or 'unknown'}",
                        )
                        return None
                return plan
            except Exception as error:
                error_text = str(error)
                retry_errors.append(f"Attempt {invoke_attempt}: {error_text}")
                if self.stage._should_retry_llm_request(error_text) and invoke_attempt < max_attempts:
                    continue
                self.stage._persist_poc_llm_trace(
                    context.planner_attempt,
                    "error.txt",
                    "\n".join(retry_errors),
                )
                return None
        self.stage._persist_poc_llm_trace(
            context.planner_attempt,
            "error.txt",
            "\n".join(retry_errors) or "LLM request failed without a response.",
        )
        return None


class PocStage:
    """PoC 阶段协调器。"""

    MAX_REPLAN_ATTEMPTS = 3
    MAX_LLM_NO_RESPONSE_RETRIES = 2
    PATCH_EXCERPT_CHAR_LIMIT = 2200
    REPO_EVIDENCE_BLOCK_LIMIT = 4
    REPO_EVIDENCE_CHAR_LIMIT = 900
    REFERENCE_POC_BLOCK_LIMIT = 2
    REFERENCE_POC_CHAR_LIMIT = 1200
    REFERENCE_POC_SUMMARY_CHAR_LIMIT = 220
    PREVIOUS_EXECUTION_LOG_CHAR_LIMIT = 3500
    PREVIOUS_RUN_SCRIPT_CHAR_LIMIT = 2000
    PREVIOUS_PAYLOAD_CHAR_LIMIT = 2000
    PREVIOUS_RUN_VERIFY_CHAR_LIMIT = 1600

    def __init__(self, file_tool: FileTool | None = None, docker_tool: DockerTool | None = None) -> None:
        self.file_tool = file_tool or FileTool()
        self.docker_tool = docker_tool or DockerTool()
        self.planner = PocPlanner(self)
        self._active_poc_dir = ""

    def build_plan(self, knowledge: KnowledgeModel, build: BuildArtifact, workspace: str) -> dict:
        """生成 PoC 阶段静态元数据。"""

        if not build.build_success:
            raise RuntimeError("build artifact must be successful before running poc stage")

        paths = PocStagePaths(workspace)
        return {
            "workspace": workspace,
            "repo_dir": str(paths.repo_dir),
            "poc_artifacts_dir": str(paths.poc_dir),
            "docker_image_tag": f"deeprepro-{knowledge.cve_id.lower()}-poc",
            "base_image_tag": build.compiled_image_tag or build.docker_image_tag or "",
            "target_binary": build.binary_or_entrypoint or build.expected_binary_path or "",
        }

    def render_prompt(self, knowledge: KnowledgeModel, build: BuildArtifact, plan: dict) -> str:
        """为后续 LLM 规划保留接口。"""

        prompt = {
            "cve_id": knowledge.cve_id,
            "resolved_ref": build.resolved_ref,
            "workspace": plan["workspace"],
            "target_binary": plan["target_binary"],
        }
        return json.dumps(prompt, ensure_ascii=False)

    def collect_poc_context(
        self,
        knowledge: KnowledgeModel,
        build: BuildArtifact,
        workspace: str,
        planner_attempt: int = 1,
        previous_failure_kind: str = "",
        previous_execution_log: str = "",
    ) -> PocContext:
        """Collect local PoC evidence from dataset hints, patch, and build outputs."""

        paths = PocStagePaths(workspace)
        patch_diff_text = self._read_patch_diff(knowledge.cve_id)
        patch_affected_files = sorted(set(re.findall(r"^\+\+\+ b/(.+)$", patch_diff_text, re.MULTILINE)))
        patch_metadata = self._extract_patch_metadata(patch_diff_text)
        candidate_entrypoints = [item for item in [build.binary_or_entrypoint, build.expected_binary_path] if item]
        candidate_entrypoints.extend(self._discover_candidate_binaries(paths.repo_dir))
        trigger_files = patch_affected_files or list(knowledge.affected_files)
        reproduction_recipe_summaries = self._summarize_reproduction_recipes(knowledge.reproduction_recipes)
        cli_flags = self._extract_candidate_cli_flags(knowledge.reproduction_hints + reproduction_recipe_summaries)
        reference_poc_summaries = self._collect_reference_poc_summaries(knowledge.cve_id)
        repo_evidence_blocks = self._collect_repo_evidence(paths.repo_dir, trigger_files)
        inferred_input_modes = self._infer_input_modes(
            hints=knowledge.reproduction_hints + reproduction_recipe_summaries,
            patch_diff_text=patch_diff_text,
            reference_poc_summaries=reference_poc_summaries,
        )
        return PocContext(
            cve_id=knowledge.cve_id,
            repo_url=knowledge.repo_url or "",
            resolved_ref=build.resolved_ref,
            repo_local_path=build.repo_local_path,
            build_system=build.build_system,
            build_success=build.build_success,
            target_binary=build.binary_or_entrypoint or build.expected_binary_path or "",
            patch_diff_excerpt=self._truncate_text(patch_diff_text, self.PATCH_EXCERPT_CHAR_LIMIT),
            patch_affected_files=patch_affected_files or list(knowledge.affected_files),
            patch_changed_functions=patch_metadata["changed_functions"],
            patch_added_checks=patch_metadata["added_checks"],
            patch_error_strings=patch_metadata["error_strings"],
            inferred_input_modes=inferred_input_modes,
            knowledge_summary=knowledge.summary,
            reproduction_hints=list(knowledge.reproduction_hints),
            reproduction_recipe_summaries=reproduction_recipe_summaries,
            expected_error_patterns=list(knowledge.expected_error_patterns),
            expected_stack_keywords=list(knowledge.expected_stack_keywords),
            candidate_entrypoints=sorted(set(candidate_entrypoints)),
            candidate_trigger_files=trigger_files[:12],
            candidate_cli_flags=cli_flags,
            reference_poc_summaries=reference_poc_summaries,
            repo_evidence_blocks=repo_evidence_blocks,
            previous_failure_kind=previous_failure_kind,
            previous_execution_log=self._truncate_text(previous_execution_log, self.PREVIOUS_EXECUTION_LOG_CHAR_LIMIT),
            planner_attempt=planner_attempt,
        )

    def plan_poc(self, knowledge: KnowledgeModel, build: BuildArtifact, context: PocContext) -> PocPlan:
        """Generate the structured PoC plan using the dedicated planner."""

        return self.planner.plan(knowledge=knowledge, build=build, context=context)

    def replan_after_failure(
        self,
        knowledge: KnowledgeModel,
        build: BuildArtifact,
        context: PocContext,
        previous_plan: PocPlan,
        previous_artifact: PoCArtifact,
    ) -> Optional[PocPlan]:
        """Ask the planner to adjust the PoC plan after one failed execution."""

        return self.planner.replan_after_failure(
            knowledge=knowledge,
            build=build,
            context=context,
            previous_plan=previous_plan,
            previous_artifact=previous_artifact,
        )

    def _prepare_workspace(self, paths: PocStagePaths) -> None:
        self.file_tool.ensure_dir(str(paths.workspace_root))
        self.file_tool.ensure_dir(str(paths.poc_dir))
        self.file_tool.ensure_dir(str(paths.llm_dir))
        self.file_tool.ensure_dir(str(paths.payloads_dir))
        self.file_tool.ensure_dir(str(paths.inputs_dir))

    def _persist_poc_llm_trace(self, planner_attempt: int, filename: str, content: str) -> None:
        poc_dir = getattr(self, "_active_poc_dir", None)
        if not poc_dir:
            return
        attempt_dir = Path(poc_dir) / "llm" / f"attempt-{planner_attempt}"
        self.file_tool.ensure_dir(str(attempt_dir))
        self.file_tool.write_text(str(attempt_dir / filename), (content or "").rstrip() + "\n")

    def _write_yaml_file(self, path: Path, payload: Any) -> None:
        """Persist YAML using one consistent formatting policy."""

        self.file_tool.write_text(
            str(path),
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        )

    def _heuristic_poc_plan(self, knowledge: KnowledgeModel, build: BuildArtifact, context: PocContext) -> PocPlan:
        return self.planner.heuristic_plan(knowledge=knowledge, build=build, context=context)

    def _build_fallback_spec(self, knowledge: KnowledgeModel, build: BuildArtifact, context: PocContext) -> PocFallbackSpec:
        """Assemble all heuristic PoC decisions in one place."""

        reference_poc = self._load_reference_poc(knowledge.cve_id)
        payload_filename = "poc.txt"
        payload_content = "trigger\n"
        source_of_truth = "heuristic"
        rationale = "Fallback PoC generated from knowledge hints and build artifact."
        confidence = "medium"
        recipe_plan = self._materialize_recipe_plan(knowledge.reproduction_recipes)
        if recipe_plan is not None:
            payload_filename = recipe_plan["payload_filename"]
            payload_content = recipe_plan["payload_content"]
            source_of_truth = "reproduction_recipe"
            rationale = "Materialized the payload and trigger from a reproduction recipe preserved in knowledge."
            confidence = recipe_plan["confidence"]
        elif reference_poc is not None:
            payload_filename = reference_poc[0]
            payload_content = reference_poc[1]
            source_of_truth = "dataset_poc"
            rationale = "Adopted the dataset-provided PoC as the primary payload."
            confidence = "high"

        target_binary = self._select_target_binary(build, context, payload_filename)
        target_args = self._select_target_args(
            knowledge,
            payload_filename,
            context,
            target_binary,
            recipe_run_command=recipe_plan["run_command"] if recipe_plan is not None else "",
        )
        run_command = self._build_run_command(target_binary, target_args)
        if recipe_plan is not None and recipe_plan["run_command"]:
            run_command = self._normalize_recipe_run_command(recipe_plan["run_command"], payload_filename)

        return PocFallbackSpec(
            trigger_mode=self._infer_trigger_mode(payload_filename, context),
            target_binary=target_binary,
            target_args=target_args,
            payload_filename=payload_filename,
            payload_content=payload_content,
            run_command=run_command,
            expected_stdout_patterns=[],
            expected_stderr_patterns=list(knowledge.expected_error_patterns),
            expected_stack_keywords=list(knowledge.expected_stack_keywords),
            expected_crash_type=self._infer_expected_crash_type(knowledge),
            source_of_truth=source_of_truth,
            confidence=confidence,
            rationale=rationale,
        )

    def _try_llm_poc_plan(
        self,
        knowledge: KnowledgeModel,
        build: BuildArtifact,
        context: PocContext,
        previous_plan: Optional[PocPlan] = None,
        previous_artifact: Optional[PoCArtifact] = None,
    ) -> Optional[PocPlan]:
        return self.planner.try_llm_plan(
            knowledge=knowledge,
            build=build,
            context=context,
            previous_plan=previous_plan,
            previous_artifact=previous_artifact,
        )

    def _build_llm_prompt(
        self,
        knowledge: KnowledgeModel,
        build: BuildArtifact,
        context: PocContext,
        previous_plan: Optional[PocPlan],
        previous_artifact: Optional[PoCArtifact],
    ) -> str:
        reference_poc_blocks = self._reference_poc_prompt_blocks(
            context.reference_poc_summaries,
            detailed=previous_plan is not None,
        )
        sections = [
            "You are the PoC Agent for a vulnerability reproduction framework.",
            "Infer the most plausible minimal reproducer from patch context, repository evidence, build outputs, and existing hints.",
            "Give substantial weight to semantic understanding of the vulnerability and the likely trigger path.",
            "Prefer a minimal reproducible trigger over a large script.",
            "Adapt any existing PoC or hint to the current workspace layout inside Docker.",
            f"The build image keeps the checked-out project under {self._container_project_dir(context.repo_url)}.",
            "The repository is mounted at /workspace/repo.",
            "Prefer compiled binaries from the build-image project directory. /workspace/repo is a mounted source tree and may not contain built executables.",
            "Payload files should normally be written under /workspace/artifacts/poc/payloads/ and auxiliary files under /workspace/artifacts/poc/inputs/.",
            "You may freely change payload filename, suffix, on-disk format, auxiliary files, and wrapper/decoding steps when that improves the trigger.",
            "Return exactly one JSON object and no markdown fences.",
            "Schema:",
            json.dumps(
                {
                    "trigger_mode": "cli-file|cli-stdin|cli-argv|script-driver|library-harness",
                    "target_binary": "string",
                    "target_args": ["string"],
                    "environment_variables": {"KEY": "VALUE"},
                    "payload_filename": "string",
                    "payload_content": "string",
                    "auxiliary_files": {"relative/path": "content"},
                    "run_command": "string",
                    "expected_exit_code": "integer or null",
                    "expected_stdout_patterns": ["string"],
                    "expected_stderr_patterns": ["string"],
                    "expected_stack_keywords": ["string"],
                    "expected_crash_type": "string",
                    "source_of_truth": "string",
                    "confidence": "low|medium|high",
                    "rationale": "string",
                    "dockerfile_override": "string or null",
                    "run_script_override": "string or null",
                },
                ensure_ascii=True,
            ),
            f"CVE: {knowledge.cve_id}",
            f"Repository: {knowledge.repo_url or ''}",
            f"Summary: {knowledge.summary}",
            f"Vulnerability type: {knowledge.vulnerability_type}",
            f"Resolved vulnerable ref: {build.resolved_ref}",
            f"Build system: {build.build_system}",
            f"Build target binary: {build.binary_or_entrypoint or build.expected_binary_path or ''}",
            f"Expected error patterns: {json.dumps(knowledge.expected_error_patterns, ensure_ascii=False)}",
            f"Expected stack keywords: {json.dumps(knowledge.expected_stack_keywords, ensure_ascii=False)}",
            f"Reproduction hints: {json.dumps(knowledge.reproduction_hints, ensure_ascii=False)}",
            "Structured reproduction recipes:",
            "\n\n---\n\n".join(context.reproduction_recipe_summaries[:4]) or "<empty>",
            f"Candidate entrypoints: {json.dumps(context.candidate_entrypoints, ensure_ascii=False)}",
            f"Candidate CLI flags: {json.dumps(context.candidate_cli_flags, ensure_ascii=False)}",
            f"Candidate trigger files: {json.dumps(context.candidate_trigger_files, ensure_ascii=False)}",
            f"Patch changed functions: {json.dumps(context.patch_changed_functions, ensure_ascii=False)}",
            f"Patch added checks: {json.dumps(context.patch_added_checks, ensure_ascii=False)}",
            f"Patch error strings: {json.dumps(context.patch_error_strings, ensure_ascii=False)}",
            f"Inferred input modes: {json.dumps(context.inferred_input_modes, ensure_ascii=False)}",
            "Patch excerpt:",
            context.patch_diff_excerpt or "<empty>",
            "Repository evidence excerpts:",
            "\n\n---\n\n".join(context.repo_evidence_blocks[:8]) or "<empty>",
            "Reference PoC excerpts:",
            "\n\n---\n\n".join(reference_poc_blocks) or "<empty>",
        ]

        if previous_plan is not None and previous_artifact is not None:
            failure_kind = context.previous_failure_kind or self._classify_failure_kind(previous_artifact.execution_logs)
            sections.extend(
                [
                    "",
                    f"Previous failure kind: {failure_kind or '<empty>'}",
                    "Previous plan:",
                    yaml.safe_dump(previous_plan.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
                    f"Observed exit code: {previous_artifact.observed_exit_code}",
                    f"Observed crash type: {previous_artifact.observed_crash_type}",
                    "Previous execution logs:",
                    context.previous_execution_log or "<empty>",
                    "Previous run.sh:",
                    context.previous_run_script_content or "<empty>",
                    "Previous payload content:",
                    context.previous_payload_content or "<empty>",
                    "Previous run_verify.yaml:",
                    context.previous_run_verify_report or "<empty>",
                    "Adjust the plan to improve the trigger while staying minimal.",
                    "Replan contract:",
                    "- If docker image build failed, you must return a new dockerfile_override.",
                    "- If the target ran but did not trigger the expected behavior, you must modify the payload, auxiliary files, run command, environment, or run_script_override.",
                    "- If container execution failed, you must modify how the target is invoked, preferably via run_script_override or by changing target_binary/target_args/run_command/environment.",
                    "- Do not only change rationale, confidence, or source_of_truth.",
                ]
            )
        return "\n".join(sections)

    def _normalize_poc_plan(self, plan: PocPlan, repo_url: str = "") -> PocPlan:
        if not plan.payload_filename:
            plan.payload_filename = "poc.txt"
        if not plan.payload_content:
            plan.payload_content = "trigger\n"
        plan.payload_filename = Path(plan.payload_filename).name
        plan.auxiliary_files = self._normalize_auxiliary_files(plan.auxiliary_files)
        plan.target_binary = self._normalize_target_binary(plan.target_binary, repo_url)
        plan.target_args = [self._normalize_workspace_arg(arg, plan.payload_filename) for arg in plan.target_args]
        if not plan.run_command:
            args = " ".join(self._shell_quote(item) for item in plan.target_args)
            plan.run_command = f"{self._shell_quote(plan.target_binary)} {args}".strip()
        else:
            plan.run_command = self._normalize_run_command(plan.run_command, plan.payload_filename, repo_url)
            plan.run_command = self._align_run_command_with_target_binary(plan.run_command, plan.target_binary)
        if not plan.expected_stderr_patterns and plan.expected_crash_type:
            plan.expected_stderr_patterns = [plan.expected_crash_type]
        plan.expected_stack_keywords = sorted(set(plan.expected_stack_keywords))
        return plan

    def _execute_poc_plan(self, paths: PocStagePaths, plan_meta: dict, plan: PocPlan) -> PoCArtifact:
        payload_path = paths.payloads_dir / plan.payload_filename
        self.file_tool.write_text(str(payload_path), plan.payload_content)

        auxiliary_paths: list[str] = []
        for name, content in plan.auxiliary_files.items():
            target_dir = paths.inputs_dir if "/" not in name else paths.poc_dir
            target_path = target_dir / name
            self.file_tool.write_text(str(target_path), content)
            auxiliary_paths.append(str(target_path))

        docker_context = {
            "base_image_tag": plan_meta["base_image_tag"],
            "workspace_root": "/workspace",
            "artifacts_root": "/workspace/artifacts",
            "poc_artifacts_dir": "/workspace/artifacts/poc",
        }
        script_context = {
            "workspace_root": "/workspace",
            "execution_dir": self._default_execution_dir(plan.target_binary),
            "poc_artifacts_dir": "/workspace/artifacts/poc",
            "target_binary": plan.target_binary,
            "run_command": plan.run_command,
        }

        dockerfile_content = (
            plan.dockerfile_override.rstrip() + "\n"
            if plan.dockerfile_override
            else self._render_template("poc.Dockerfile.j2", docker_context)
        )
        run_script_content = (
            plan.run_script_override.rstrip() + "\n"
            if plan.run_script_override
            else self._render_template("poc_run.sh.j2", script_context)
        )
        self.file_tool.write_text(str(paths.dockerfile), dockerfile_content)
        self.file_tool.write_text(str(paths.run_script), run_script_content)

        workspace_root = str(paths.workspace_root.resolve())
        docker_build_result = self.docker_tool.build_image(
            DockerBuildRequest(
                workspace=workspace_root,
                dockerfile_path=str(paths.dockerfile.resolve()),
                image_tag=plan_meta["docker_image_tag"],
            )
        )
        if docker_build_result.success:
            run_result = self.docker_tool.run_container(
                DockerRunRequest(
                    image_tag=plan_meta["docker_image_tag"],
                    workspace=workspace_root,
                    command=["bash", "/workspace/artifacts/poc/run.sh"],
                    environment=plan.environment_variables,
                )
            )
        else:
            run_result = docker_build_result

        execution_logs = self._compose_poc_logs(docker_build_result, run_result if docker_build_result.success else None)
        self.file_tool.write_text(str(paths.poc_log), execution_logs)

        observation = self._extract_execution_observation(execution_logs)
        crash_report = observation["observed_stderr"] or observation["observed_stdout"]
        self.file_tool.write_text(str(paths.crash_report), crash_report)
        # stdout 模式只在 stdout 找；stderr 模式只在 stderr 找；
        # stack keywords 在合并文本里找（栈帧可能落在任一流）。
        matched_stdout_patterns = self._match_patterns(
            observation["observed_stdout"],
            plan.expected_stdout_patterns,
        )
        matched_stderr_patterns = self._match_patterns(
            observation["observed_stderr"],
            plan.expected_stderr_patterns,
        )
        matched_stack_keywords = self._match_patterns(
            observation["observed_stdout"] + "\n" + observation["observed_stderr"],
            plan.expected_stack_keywords,
        )
        # matched_error_patterns 与 matched_stderr_patterns 同步，向后兼容
        matched_error_patterns = list(matched_stderr_patterns)

        execution_success = docker_build_result.success and bool(run_result.success)
        run_verify_report = self._build_run_verify_report(
            plan=plan,
            observation=observation,
            execution_logs=execution_logs,
            matched_error_patterns=matched_error_patterns,
            matched_stdout_patterns=matched_stdout_patterns,
            matched_stack_keywords=matched_stack_keywords,
        )
        self.file_tool.safe_persist(
            str(paths.run_verify_yaml),
            yaml.safe_dump(run_verify_report.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
            description="run_verify.yaml",
        )
        reproducer_verified = run_verify_report.eligible_for_verify
        return PoCArtifact(
            root_cause_analysis="",
            payload_generation_strategy=plan.rationale,
            trigger_mode=plan.trigger_mode,
            trigger_command=plan.run_command,
            target_binary=plan.target_binary,
            poc_filename=plan.payload_filename,
            poc_content=plan.payload_content,
            run_script_content=run_script_content,
            input_files=sorted(plan.auxiliary_files.keys()),
            input_file_paths=[str(payload_path)],
            auxiliary_file_paths=sorted(auxiliary_paths),
            expected_error_patterns=list(plan.expected_stderr_patterns),
            expected_stdout_patterns=list(plan.expected_stdout_patterns),
            expected_stderr_patterns=list(plan.expected_stderr_patterns),
            expected_exit_code=plan.expected_exit_code,
            expected_stack_keywords=list(plan.expected_stack_keywords),
            expected_crash_type=plan.expected_crash_type,
            environment_variables=dict(plan.environment_variables),
            crash_report_content=crash_report,
            observed_exit_code=observation["observed_exit_code"],
            observed_stdout=observation["observed_stdout"],
            observed_stderr=observation["observed_stderr"],
            observed_crash_type=observation["observed_crash_type"],
            matched_error_patterns=matched_error_patterns,
            matched_stdout_patterns=matched_stdout_patterns,
            matched_stderr_patterns=matched_stderr_patterns,
            matched_stack_keywords=matched_stack_keywords,
            reproducer_verified=reproducer_verified,
            execution_success=execution_success,
            execution_logs=execution_logs,
        )

    def run(self, knowledge: KnowledgeModel, build: BuildArtifact, workspace: str) -> PoCArtifact:
        """执行 PoC 阶段并返回 PoC 产物。"""

        paths = PocStagePaths(workspace)
        subgraph = self.build_internal_graph()
        result = subgraph.invoke(
            {
                "knowledge": knowledge,
                "build": build,
                "paths": paths,
                "attempt": 0,
            }
        )
        outcome = result["outcome"]
        self.persist_poc_outputs(outcome.artifact, paths)
        return outcome.artifact

    def build_internal_graph(self):
        """Build the internal LangGraph subgraph for the PoC stage."""

        builder = StateGraph(PocGraphState)
        builder.add_node("prepare", self._poc_graph_prepare_node)
        builder.add_node("plan", self._poc_graph_plan_node)
        builder.add_node("execute", self._poc_graph_execute_node)
        builder.add_edge(START, "prepare")
        builder.add_edge("prepare", "plan")
        builder.add_edge("plan", "execute")
        builder.add_conditional_edges(
            "execute",
            self._route_after_poc_execute,
            {
                "plan": "plan",
                "done": END,
            },
        )
        return builder.compile()

    def _poc_graph_prepare_node(self, state: PocGraphState) -> PocGraphState:
        prepared = self.prepare_poc_run(
            knowledge=state["knowledge"],
            build=state["build"],
            paths=state["paths"],
        )
        return {
            "prepared": prepared,
            "current_context": prepared.context,
            "attempt": 0,
        }

    def _poc_graph_plan_node(self, state: PocGraphState) -> PocGraphState:
        plan = self.plan_poc(
            knowledge=state["knowledge"],
            build=state["build"],
            context=state["current_context"],
        )
        return {"current_plan": plan}

    def _poc_graph_execute_node(self, state: PocGraphState) -> PocGraphState:
        prepared = state["prepared"]
        paths = state["paths"]
        plan = state["current_plan"]
        self._write_yaml_file(paths.poc_plan_yaml, plan.model_dump(mode="json"))
        outcome = self.execute_poc_attempt(paths, prepared.plan_meta, plan)
        updates: PocGraphState = {
            "outcome": outcome,
            "attempt": state.get("attempt", 0) + 1,
            "current_plan": plan,
        }
        if not (outcome.artifact.execution_success and outcome.artifact.reproducer_verified):
            current_context = self._build_retry_context(
                state["current_context"],
                paths,
                outcome.artifact,
            )
            updates["current_context"] = current_context
            replanned = self.replan_after_failure(
                knowledge=state["knowledge"],
                build=state["build"],
                context=current_context,
                previous_plan=plan,
                previous_artifact=outcome.artifact,
            )
            if replanned is not None:
                updates["current_plan"] = self._normalize_poc_plan(replanned, repo_url=current_context.repo_url)
        return updates

    def _route_after_poc_execute(self, state: PocGraphState) -> str:
        outcome = state.get("outcome")
        attempt = state.get("attempt", 0)
        if outcome is None:
            return "done"
        if outcome.artifact.execution_success and outcome.artifact.reproducer_verified:
            return "done"
        if attempt >= self.MAX_REPLAN_ATTEMPTS:
            return "done"
        return "plan"

    def prepare_poc_run(
        self,
        knowledge: KnowledgeModel,
        build: BuildArtifact,
        paths: PocStagePaths,
    ) -> PocPreparedRun:
        """Collect deterministic PoC inputs before any planning starts."""

        plan_meta = self.build_plan(knowledge=knowledge, build=build, workspace=str(paths.workspace_root))
        self._prepare_workspace(paths)
        self._active_poc_dir = str(paths.poc_dir)
        context = self.collect_poc_context(
            knowledge=knowledge,
            build=build,
            workspace=str(paths.workspace_root),
            planner_attempt=1,
        )
        self._write_yaml_file(paths.poc_context_yaml, context.model_dump(mode="json"))
        return PocPreparedRun(plan_meta=plan_meta, context=context)

    def plan_and_execute_poc(
        self,
        knowledge: KnowledgeModel,
        build: BuildArtifact,
        prepared: PocPreparedRun,
        paths: PocStagePaths,
    ) -> PocExecutionOutcome:
        """Generate a PoC plan, execute it, and optionally replan after failures."""

        current_context = prepared.context
        last_outcome: PocExecutionOutcome | None = None

        for attempt in range(self.MAX_REPLAN_ATTEMPTS):
            plan = self.plan_poc(knowledge=knowledge, build=build, context=current_context)
            self._write_yaml_file(paths.poc_plan_yaml, plan.model_dump(mode="json"))
            last_outcome = self.execute_poc_attempt(paths, prepared.plan_meta, plan)
            if (last_outcome.artifact.execution_success and last_outcome.artifact.reproducer_verified) or attempt + 1 >= self.MAX_REPLAN_ATTEMPTS:
                break

            replanned = self.replan_after_failure(
                knowledge=knowledge,
                build=build,
                context=current_context,
                previous_plan=plan,
                previous_artifact=last_outcome.artifact,
            )
            if replanned is not None:
                replanned = self._normalize_poc_plan(replanned, repo_url=current_context.repo_url)
                self._write_yaml_file(paths.poc_plan_yaml, replanned.model_dump(mode="json"))
                last_outcome = self.execute_poc_attempt(paths, prepared.plan_meta, replanned)
                if (last_outcome.artifact.execution_success and last_outcome.artifact.reproducer_verified) or attempt + 1 >= self.MAX_REPLAN_ATTEMPTS:
                    break

            current_context = self._build_retry_context(current_context, paths, last_outcome.artifact)

        if last_outcome is None:
            raise RuntimeError("poc stage did not produce an artifact")
        return last_outcome

    def execute_poc_attempt(
        self,
        paths: PocStagePaths,
        plan_meta: dict[str, Any],
        plan: PocPlan,
    ) -> PocExecutionOutcome:
        """Execute one concrete PoC attempt from a single plan."""

        artifact = self._execute_poc_plan(paths=paths, plan_meta=plan_meta, plan=plan)
        return PocExecutionOutcome(plan=plan, artifact=artifact)

    def persist_poc_outputs(self, artifact: PoCArtifact, paths: PocStagePaths) -> None:
        """Persist the final PoC artifact."""

        self._write_yaml_file(paths.poc_artifact_yaml, artifact.model_dump(mode="json"))

    def _build_retry_context(self, context: PocContext, paths: PocStagePaths, artifact: PoCArtifact) -> PocContext:
        run_verify_report = ""
        if paths.run_verify_yaml.exists():
            run_verify_report = self._truncate_text(
                paths.run_verify_yaml.read_text(encoding="utf-8", errors="replace"),
                self.PREVIOUS_RUN_VERIFY_CHAR_LIMIT,
            )
        return context.model_copy(
            update={
                "planner_attempt": context.planner_attempt + 1,
                "previous_failure_kind": self._classify_failure_kind(artifact.execution_logs),
                "previous_execution_log": self._truncate_text(
                    artifact.execution_logs,
                    self.PREVIOUS_EXECUTION_LOG_CHAR_LIMIT,
                ),
                "previous_run_script_content": self._truncate_text(
                    artifact.run_script_content,
                    self.PREVIOUS_RUN_SCRIPT_CHAR_LIMIT,
                ),
                "previous_payload_content": self._truncate_text(
                    artifact.poc_content,
                    self.PREVIOUS_PAYLOAD_CHAR_LIMIT,
                ),
                "previous_run_verify_report": run_verify_report,
            }
        )

    def _is_valid_replan_candidate(
        self,
        previous_plan: PocPlan,
        candidate_plan: PocPlan,
        failure_kind: str = "",
    ) -> bool:
        if candidate_plan.model_dump(mode="json") == previous_plan.model_dump(mode="json"):
            return False
        normalized_failure_kind = (failure_kind or "").strip().lower()
        if normalized_failure_kind == "docker_build":
            return bool(candidate_plan.dockerfile_override)
        if normalized_failure_kind == "container_run":
            return self._changes_poc_execution_surface(previous_plan, candidate_plan)
        return self._changes_trigger_strategy(previous_plan, candidate_plan)

    def _changes_poc_execution_surface(self, previous_plan: PocPlan, candidate_plan: PocPlan) -> bool:
        return any(
            [
                previous_plan.target_binary != candidate_plan.target_binary,
                previous_plan.target_args != candidate_plan.target_args,
                previous_plan.environment_variables != candidate_plan.environment_variables,
                previous_plan.run_command != candidate_plan.run_command,
                previous_plan.payload_filename != candidate_plan.payload_filename,
                previous_plan.payload_content != candidate_plan.payload_content,
                previous_plan.auxiliary_files != candidate_plan.auxiliary_files,
                previous_plan.run_script_override != candidate_plan.run_script_override,
                previous_plan.dockerfile_override != candidate_plan.dockerfile_override,
            ]
        )

    def _changes_trigger_strategy(self, previous_plan: PocPlan, candidate_plan: PocPlan) -> bool:
        return any(
            [
                previous_plan.payload_filename != candidate_plan.payload_filename,
                previous_plan.payload_content != candidate_plan.payload_content,
                previous_plan.auxiliary_files != candidate_plan.auxiliary_files,
                previous_plan.target_args != candidate_plan.target_args,
                previous_plan.environment_variables != candidate_plan.environment_variables,
                previous_plan.run_command != candidate_plan.run_command,
                previous_plan.run_script_override != candidate_plan.run_script_override,
                previous_plan.target_binary != candidate_plan.target_binary,
            ]
        )

    def _read_patch_diff(self, cve_id: str) -> str:
        path = find_patch_diff(cve_id)
        if path is None:
            return ""
        return path.read_text(encoding="utf-8", errors="replace")

    def _discover_candidate_binaries(self, repo_dir: Path) -> list[str]:
        candidates: list[str] = []
        for relative in ("src", "bin", "build", "target/debug"):
            root = repo_dir / relative
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file() and path.suffix in {"", ".sh", ".py", ".pl", ".lua"}:
                    candidates.append(str(path.relative_to(repo_dir)))
                if len(candidates) >= 6:
                    return candidates
        return candidates

    def _extract_candidate_cli_flags(self, hints: list[str]) -> list[str]:
        flags: list[str] = []
        for hint in hints:
            flags.extend(re.findall(r"(--?[A-Za-z0-9][A-Za-z0-9_-]*)", hint))
        return sorted(set(flags))

    def _summarize_reproduction_recipes(self, recipes: list[ReproductionRecipe]) -> list[str]:
        summaries: list[str] = []
        for recipe in recipes[:4]:
            payload = {
                "source_url": recipe.source_url,
                "source_title": recipe.source_title,
                "recipe_type": recipe.recipe_type,
                "steps": recipe.steps,
                "repo_setup_commands": recipe.repo_setup_commands,
                "build_commands": recipe.build_commands,
                "artifact_generation_commands": recipe.artifact_generation_commands,
                "run_commands": recipe.run_commands,
                "expected_behavior": recipe.expected_behavior,
                "source_excerpt": self._truncate_text(recipe.source_excerpt, self.REFERENCE_POC_SUMMARY_CHAR_LIMIT),
                "confidence": recipe.confidence,
            }
            summaries.append(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).strip())
        return summaries

    def _extract_patch_metadata(self, patch_diff_text: str) -> dict[str, list[str]]:
        changed_functions = sorted(set(re.findall(r"^@@ .*? ([A-Za-z_][A-Za-z0-9_]*)\s*\(", patch_diff_text, re.MULTILINE)))
        added_checks = sorted(
            set(
                match.strip()
                for match in re.findall(r"^\+.*\b(if|assert|lua[LK]_[A-Za-z0-9_]+)\b.*$", patch_diff_text, re.MULTILINE)
                if match.strip()
            )
        )
        error_strings = sorted(
            set(
                token
                for token in re.findall(r"(AddressSanitizer:[^\n]+|heap-buffer-overflow|stack-overflow|segmentation fault|assert)", patch_diff_text, re.IGNORECASE)
            )
        )
        return {
            "changed_functions": changed_functions[:12],
            "added_checks": added_checks[:12],
            "error_strings": error_strings[:12],
        }

    def _infer_input_modes(self, hints: list[str], patch_diff_text: str, reference_poc_summaries: list[str]) -> list[str]:
        text = "\n".join(hints + [patch_diff_text] + reference_poc_summaries).lower()
        modes: list[str] = []
        if any(token in text for token in ("stdin", "pipe", "readline(")):
            modes.append("stdin")
        if any(token in text for token in ("argv", "option", "--", "command line")):
            modes.append("argv")
        if any(token in text for token in ("file", "dofile", "fopen", "loadfile", ".lua", ".txt", ".json", "payload")):
            modes.append("file")
        if any(token in text for token in ("socket", "http", "request")):
            modes.append("network")
        return modes or ["file"]

    def _collect_reference_poc_summaries(self, cve_id: str) -> list[str]:
        summaries: list[str] = []
        for prefix in ("Dataset", "source/Dataset"):
            poc_dir = Path(prefix) / cve_id / "vuln_data" / "vuln_pocs"
            if not poc_dir.exists():
                continue
            for path in sorted(poc_dir.iterdir()):
                if not path.is_file():
                    continue
                content = path.read_text(encoding="utf-8", errors="replace")
                summaries.append(
                    f"FILE: {path.name}\nCONTENT:\n{self._truncate_text(content, self.REFERENCE_POC_CHAR_LIMIT)}"
                )
        return summaries[: self.REFERENCE_POC_BLOCK_LIMIT]

    def _reference_poc_prompt_blocks(self, blocks: list[str], detailed: bool) -> list[str]:
        if detailed:
            return blocks[: self.REFERENCE_POC_BLOCK_LIMIT]
        compact: list[str] = []
        for block in blocks[: self.REFERENCE_POC_BLOCK_LIMIT]:
            lines = block.splitlines()
            label = lines[0] if lines else "FILE: <unknown>"
            content = "\n".join(lines[2:]) if len(lines) > 2 else "\n".join(lines[1:])
            compact.append(
                f"{label}\nSUMMARY:\n{self._truncate_text(content, self.REFERENCE_POC_SUMMARY_CHAR_LIMIT)}"
            )
        return compact

    def _collect_repo_evidence(self, repo_dir: Path, trigger_files: list[str]) -> list[str]:
        evidence_paths: list[Path] = []
        for rel_path in trigger_files[:6]:
            candidate = repo_dir / rel_path
            if candidate.exists() and candidate.is_file():
                evidence_paths.append(candidate)

        for pattern in ("README*", "readme*", "docs/**/*", "examples/**/*", "tests/**/*", "test/**/*", "fuzz/**/*"):
            for path in repo_dir.glob(pattern):
                if path.is_file():
                    evidence_paths.append(path)
                if len(evidence_paths) >= 10:
                    break
            if len(evidence_paths) >= 10:
                break

        blocks: list[str] = []
        seen: set[str] = set()
        for path in evidence_paths:
            rel = str(path.relative_to(repo_dir))
            if rel in seen:
                continue
            seen.add(rel)
            content = path.read_text(encoding="utf-8", errors="replace")
            blocks.append(f"FILE: {rel}\nCONTENT:\n{self._truncate_text(content, self.REPO_EVIDENCE_CHAR_LIMIT)}")
            if len(blocks) >= self.REPO_EVIDENCE_BLOCK_LIMIT:
                break
        return blocks

    def _truncate_text(self, text: str, limit: int) -> str:
        if limit <= 0:
            return ""
        value = text or ""
        if len(value) <= limit:
            return value
        if limit <= 20:
            return value[:limit]
        omitted = len(value) - limit
        return f"{value[: limit - 20]}\n...[truncated {omitted} chars]"

    def _should_retry_llm_request(self, error_text: str) -> bool:
        normalized = (error_text or "").strip().lower()
        return "timed out" in normalized or "timeout" in normalized

    def _is_empty_llm_response(self, raw_response: str) -> bool:
        return not (raw_response or "").strip()

    def _load_reference_poc(self, cve_id: str) -> Optional[tuple[str, str]]:
        for prefix in ("Dataset", "source/Dataset"):
            poc_dir = Path(prefix) / cve_id / "vuln_data" / "vuln_pocs"
            if not poc_dir.exists():
                continue
            for path in sorted(poc_dir.iterdir()):
                if path.is_file():
                    return path.name, path.read_text(encoding="utf-8", errors="replace")
        return None

    def _materialize_recipe_plan(self, recipes: list[ReproductionRecipe]) -> Optional[dict[str, str]]:
        for recipe in recipes:
            if not recipe.steps:
                continue
            payload = self._payload_from_recipe(recipe)
            if payload is None:
                continue
            return {
                "payload_filename": payload["payload_filename"],
                "payload_content": payload["payload_content"],
                "run_command": self._select_recipe_run_command(recipe, payload["payload_filename"]),
                "confidence": recipe.confidence or "medium",
            }
        return None

    def _payload_from_recipe(self, recipe: ReproductionRecipe) -> Optional[dict[str, str]]:
        commands = list(recipe.artifact_generation_commands) or list(recipe.steps)
        for command in commands:
            parsed = self._decode_base64_payload_command(command)
            if parsed is not None:
                return parsed
        return None

    def _decode_base64_payload_command(self, command: str) -> Optional[dict[str, str]]:
        match = re.search(
            r'echo\s+-n\s+"(?P<data>[A-Za-z0-9+/=]+)"\s*\|\s*base64\s+-d\s*>\s*(?P<name>[A-Za-z0-9._/-]+)',
            command,
        )
        if not match:
            return None
        try:
            decoded = base64.b64decode(match.group("data"), validate=True).decode("utf-8")
        except Exception:
            return None
        filename = Path(match.group("name")).name or "poc.txt"
        if not decoded.endswith("\n"):
            decoded += "\n"
        return {
            "payload_filename": filename,
            "payload_content": decoded,
        }

    def _select_recipe_run_command(self, recipe: ReproductionRecipe, payload_filename: str) -> str:
        payload_tokens = {payload_filename, f"./{payload_filename}", "{payload}"}
        preferred: list[str] = []
        fallback: list[str] = []
        for command in list(recipe.run_commands) + list(recipe.steps):
            normalized = (command or "").strip()
            if not self._looks_like_recipe_command(normalized):
                continue
            if any(token in normalized for token in payload_tokens):
                preferred.append(normalized)
            else:
                fallback.append(normalized)
        if preferred:
            return preferred[-1]
        if fallback:
            return fallback[-1]
        return ""

    def _looks_like_recipe_command(self, command: str) -> bool:
        lowered = (command or "").strip().lower()
        if not lowered or lowered.endswith(":"):
            return False
        command_prefixes = (
            "./",
            "/",
            "lua ",
            "python ",
            "python3 ",
            "perl ",
            "ruby ",
            "node ",
            "bash ",
            "sh ",
        )
        return lowered.startswith(command_prefixes)

    def _select_target_binary(self, build: BuildArtifact, context: PocContext, payload_filename: str = "") -> str:
        if build.binary_or_entrypoint:
            return self._normalize_target_binary(build.binary_or_entrypoint, context.repo_url)
        if build.expected_binary_path:
            return self._normalize_target_binary(build.expected_binary_path, context.repo_url)
        interpreter = self._interpreter_for_payload(payload_filename)
        if interpreter:
            return interpreter
        if context.candidate_entrypoints:
            return self._normalize_target_binary(context.candidate_entrypoints[0], context.repo_url)
        return "./target"

    def _select_target_args(
        self,
        knowledge: KnowledgeModel,
        payload_filename: str,
        context: PocContext,
        target_binary: str,
        recipe_run_command: str = "",
    ) -> list[str]:
        payload_path = f"/workspace/artifacts/poc/payloads/{payload_filename}"
        if recipe_run_command:
            normalized_command = self._normalize_run_command(recipe_run_command, payload_filename, context.repo_url)
            parts = normalized_command.split()
            if parts and self._looks_like_binary(parts[0].strip("'\""), target_binary):
                return parts[1:]
        for hint in knowledge.reproduction_hints:
            if "{payload}" in hint:
                hint = hint.replace("{payload}", payload_path)
                parts = hint.split()
                if parts and self._looks_like_binary(parts[0], target_binary):
                    return parts[1:]
                return parts
        if "stdin" in context.inferred_input_modes:
            return [f"< {payload_path}"]
        return [payload_path]

    def _infer_trigger_mode(self, payload_filename: str, context: PocContext) -> str:
        suffix = Path(payload_filename).suffix.lower()
        if suffix in {".sh", ".py", ".pl"}:
            return "script-driver"
        if "stdin" in context.inferred_input_modes:
            return "cli-stdin"
        if "argv" in context.inferred_input_modes:
            return "cli-argv"
        return "cli-file"

    def _build_run_command(self, target_binary: str, target_args: list[str]) -> str:
        segments = [self._shell_quote(target_binary)]
        segments.extend(self._shell_quote(item) for item in target_args)
        return " ".join(item for item in segments if item).strip()

    def _normalize_recipe_run_command(self, run_command: str, payload_filename: str) -> str:
        payload_path = f"/workspace/artifacts/poc/payloads/{payload_filename}"
        normalized = run_command.replace("{payload}", payload_path)
        for candidate in (f"./{payload_filename}", payload_filename):
            normalized = re.sub(rf"(?<!\S){re.escape(candidate)}(?!\S)", payload_path, normalized)
        return normalized

    def _infer_expected_crash_type(self, knowledge: KnowledgeModel) -> str:
        joined = " ".join(knowledge.expected_error_patterns + knowledge.reproduction_hints).lower()
        for marker in ("segmentation fault", "assert", "abort", "stack-overflow", "heap-buffer-overflow"):
            if marker in joined:
                return marker
        return knowledge.vulnerability_type

    def _interpreter_for_payload(self, payload_filename: str) -> str:
        mapping = {
            ".py": "python3",
            ".sh": "bash",
            ".pl": "perl",
        }
        return mapping.get(Path(payload_filename).suffix.lower(), "")

    def _looks_like_binary(self, token: str, target_binary: str) -> bool:
        if not token or not target_binary:
            return False
        token_name = Path(token).name
        target_name = Path(target_binary).name
        return token_name == target_name

    def _normalize_auxiliary_files(self, auxiliary_files: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for name, content in auxiliary_files.items():
            safe_name = str(Path(name))
            while safe_name.startswith("../"):
                safe_name = safe_name[3:]
            if safe_name.startswith("/"):
                safe_name = safe_name.lstrip("/")
            if not safe_name:
                continue
            normalized[safe_name] = content
        return normalized

    def _normalize_workspace_arg(self, arg: str, payload_filename: str) -> str:
        payload_path = f"/workspace/artifacts/poc/payloads/{payload_filename}"
        if arg == "{payload}":
            return payload_path
        return arg.replace("./payloads/", "/workspace/artifacts/poc/payloads/")

    def _normalize_run_command(self, run_command: str, payload_filename: str, repo_url: str = "") -> str:
        payload_path = f"/workspace/artifacts/poc/payloads/{payload_filename}"
        if not repo_url:
            return run_command.replace("{payload}", payload_path).replace("./payloads/", "/workspace/artifacts/poc/payloads/")
        project_dir = self._container_project_dir(repo_url)
        return (
            run_command.replace("{payload}", payload_path)
            .replace("./payloads/", "/workspace/artifacts/poc/payloads/")
            .replace("/workspace/repo/", f"{project_dir}/")
        )

    def _align_run_command_with_target_binary(self, run_command: str, target_binary: str) -> str:
        stripped = (run_command or "").strip()
        if not stripped or not target_binary:
            return run_command
        quoted_target = self._shell_quote(target_binary)
        parts = stripped.split(maxsplit=1)
        first = parts[0].strip("'\"")
        if Path(first).name != Path(target_binary).name:
            return run_command
        remainder = parts[1] if len(parts) > 1 else ""
        return f"{quoted_target} {remainder}".strip()

    def _container_project_dir(self, repo_url: str) -> str:
        project_name = self._derive_project_name(repo_url)
        return f"/src/{project_name}"

    def _derive_project_name(self, repo_url: str) -> str:
        name = repo_url.rstrip("/").split("/")[-1]
        if name.endswith(".git"):
            name = name[:-4]
        return re.sub(r"[^A-Za-z0-9._-]+", "-", name) or "target"

    def _normalize_target_binary(self, target_binary: str, repo_url: str) -> str:
        if not target_binary:
            return target_binary
        if not repo_url:
            return target_binary
        if target_binary.startswith("/"):
            if target_binary.startswith("/workspace/repo/"):
                return target_binary.replace("/workspace/repo/", f"{self._container_project_dir(repo_url)}/", 1)
            return target_binary
        project_dir = self._container_project_dir(repo_url)
        cleaned = target_binary[2:] if target_binary.startswith("./") else target_binary
        return f"{project_dir}/{cleaned}".replace("//", "/")

    def _render_template(self, template_name: str, context: dict[str, Any]) -> str:
        if Environment is not None and FileSystemLoader is not None and StrictUndefined is not None:
            template_dir = Path(__file__).resolve().parents[1] / "templates"
            env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                undefined=StrictUndefined,
                trim_blocks=True,
                lstrip_blocks=True,
            )
            return env.get_template(template_name).render(**context).strip() + "\n"
        if template_name == "poc.Dockerfile.j2":
            return self._render_poc_dockerfile_fallback(context)
        if template_name == "poc_run.sh.j2":
            return self._render_poc_run_script_fallback(context)
        raise RuntimeError(f"unsupported template without Jinja2: {template_name}")

    def _render_poc_dockerfile_fallback(self, context: dict[str, Any]) -> str:
        base_image_tag = context.get("base_image_tag") or "ubuntu:20.04"
        lines = [
            f"FROM {base_image_tag}",
            "",
            'SHELL ["/bin/bash", "-o", "pipefail", "-c"]',
            "",
            "WORKDIR /workspace",
            "COPY artifacts/poc /workspace/artifacts/poc",
            "",
        ]
        return "\n".join(lines)

    def _render_poc_run_script_fallback(self, context: dict[str, Any]) -> str:
        return "\n".join(
            [
                "#!/bin/bash",
                "set +e",
                "",
                f'POC_ARTIFACTS_DIR="{context.get("poc_artifacts_dir", "/workspace/artifacts/poc")}"',
                f'EXECUTION_DIR="{context.get("execution_dir", "/workspace")}"',
                'mkdir -p "${POC_ARTIFACTS_DIR}"',
                'cd "${EXECUTION_DIR}"',
                'echo "target_binary=' + self._escape_for_echo(context.get("target_binary", "")) + '"',
                'echo "trigger_command=' + self._escape_for_echo(context.get("run_command", "")) + '"',
                'stdout_file="${POC_ARTIFACTS_DIR}/stdout.txt"',
                'stderr_file="${POC_ARTIFACTS_DIR}/stderr.txt"',
                context["run_command"] + ' >"${stdout_file}" 2>"${stderr_file}"',
                'execution_exit_code=$?',
                'echo "execution_exit_code=${execution_exit_code}"',
                'echo "stdout_begin"',
                'cat "${stdout_file}" 2>/dev/null || true',
                'echo "stdout_end"',
                'echo "stderr_begin"',
                'cat "${stderr_file}" 2>/dev/null || true',
                'echo "stderr_end"',
                'exit 0',
                "",
            ]
        )

    def _default_execution_dir(self, target_binary: str) -> str:
        target = (target_binary or "").strip()
        if target.startswith("/"):
            parent = str(Path(target).parent)
            return parent or "/workspace"
        return "/workspace"

    def _compose_poc_logs(self, docker_build_result: Any, run_result: Any | None) -> str:
        parts = [
            f"image_build_success={docker_build_result.success}",
            f"image_build_exit_code={docker_build_result.exit_code}",
            "",
            "[docker_build_stdout]",
            docker_build_result.stdout.strip(),
            "",
            "[docker_build_stderr]",
            docker_build_result.stderr.strip(),
        ]
        if run_result is not None:
            parts.extend(
                [
                    "",
                    f"container_run_success={run_result.success}",
                    f"container_run_exit_code={run_result.exit_code}",
                    "",
                    "[container_run_stdout]",
                    run_result.stdout.strip(),
                    "",
                    "[container_run_stderr]",
                    run_result.stderr.strip(),
                ]
            )
        return "\n".join(parts).strip() + "\n"

    def _extract_execution_observation(self, execution_logs: str) -> dict[str, Any]:
        return _extract_execution_observation_module(execution_logs)

    def _extract_block(self, text: str, begin: str, end: str) -> str:
        return _extract_block_module(text, begin, end)

    def _match_patterns(self, haystack: str, patterns: list[str]) -> list[str]:
        return _match_patterns_module(haystack, patterns)

    def _build_run_verify_report(
        self,
        plan: PocPlan,
        observation: dict[str, Any],
        execution_logs: str,
        matched_error_patterns: list[str],
        matched_stack_keywords: list[str],
        matched_stdout_patterns: Optional[list[str]] = None,
    ) -> RunVerifyReport:
        """Compute the minimum-eligibility report for one PoC execution."""

        # 3.1 script_finished
        script_finished = "execution_exit_code=" in execution_logs

        # 3.2 log_well_formed
        required_markers = ("stdout_begin", "stdout_end", "stderr_begin", "stderr_end")
        log_well_formed = all(marker in execution_logs for marker in required_markers)

        # 3.3 target_binary_invoked
        target_binary_invoked = "target_binary=" in execution_logs

        # 3.4 exit_code_observed
        exit_code_observed = observation.get("observed_exit_code")

        # 3.5 hits
        error_pattern_hits = list(matched_error_patterns)
        stdout_pattern_hits = list(matched_stdout_patterns or [])
        stack_keyword_hits = list(matched_stack_keywords)

        # 3.6 crash_type_hit
        crash_type_hit = observation.get("observed_crash_type") or ""

        # 3.7 crash_type_compatible
        expected_crash = (plan.expected_crash_type or "").strip().lower()
        observed_crash_lower = crash_type_hit.strip().lower()
        if not expected_crash:
            crash_type_compatible: Optional[bool] = None
        elif not observed_crash_lower:
            crash_type_compatible = False
        else:
            crash_type_compatible = (expected_crash in observed_crash_lower) or (observed_crash_lower in expected_crash)

        # 3.8 exit_code_match_expected
        if plan.expected_exit_code is None:
            exit_code_match_expected: Optional[bool] = None
        elif exit_code_observed is None:
            exit_code_match_expected = False
        else:
            exit_code_match_expected = (exit_code_observed == plan.expected_exit_code)

        # 3.9 eligible_for_verify
        eligible_for_verify = False
        eligibility_reason = ""
        if not script_finished:
            eligibility_reason = "script_did_not_finish: missing execution_exit_code marker"
        elif not log_well_formed:
            eligibility_reason = "log_not_well_formed: stdout/stderr block markers missing"
        else:
            # Priority: stderr > stdout > stack > crash_type > exit_code
            if error_pattern_hits:
                eligible_for_verify = True
                eligibility_reason = f"error_pattern_hit: {error_pattern_hits[0]}"
            elif stdout_pattern_hits:
                eligible_for_verify = True
                eligibility_reason = f"stdout_pattern_hit: {stdout_pattern_hits[0]}"
            elif stack_keyword_hits:
                eligible_for_verify = True
                eligibility_reason = f"stack_keyword_hit: {stack_keyword_hits[0]}"
            elif crash_type_compatible is True:
                eligible_for_verify = True
                eligibility_reason = f"crash_type_compatible: observed={crash_type_hit}"
            elif exit_code_match_expected is True:
                eligible_for_verify = True
                eligibility_reason = f"exit_code_match: {exit_code_observed}"
            else:
                eligible_for_verify = False
                eligibility_reason = "no_target_behavior_observed"

        # 3.10 evidence_log_excerpt
        MAX_EXCERPT_BYTES = 2048
        stderr_block = self._extract_block(execution_logs, "stderr_begin", "stderr_end")
        if stderr_block:
            excerpt = stderr_block
        else:
            excerpt = execution_logs
        excerpt_bytes = excerpt.encode("utf-8", errors="replace")
        if len(excerpt_bytes) > MAX_EXCERPT_BYTES:
            excerpt_bytes = excerpt_bytes[-MAX_EXCERPT_BYTES:]
            excerpt = excerpt_bytes.decode("utf-8", errors="replace")
        evidence_log_excerpt = excerpt

        return RunVerifyReport(
            script_finished=script_finished,
            log_well_formed=log_well_formed,
            target_binary_invoked=target_binary_invoked,
            exit_code_observed=exit_code_observed,
            error_pattern_hits=error_pattern_hits,
            stdout_pattern_hits=stdout_pattern_hits,
            stack_keyword_hits=stack_keyword_hits,
            crash_type_hit=crash_type_hit,
            crash_type_compatible=crash_type_compatible,
            exit_code_match_expected=exit_code_match_expected,
            eligible_for_verify=eligible_for_verify,
            eligibility_reason=eligibility_reason,
            evidence_log_excerpt=evidence_log_excerpt,
        )

    def _classify_failure_kind(self, execution_logs: str) -> str:
        if "image_build_success=False" in execution_logs:
            return "docker_build"
        if "container_run_success=False" in execution_logs:
            return "container_run"
        return "non_triggering"

    def _shell_quote(self, value: str) -> str:
        return "'" + value.replace("'", "'\"'\"'") + "'"

    def _escape_for_echo(self, value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')


def poc_node(state):
    """LangGraph 节点：执行 PoC 生成与执行阶段。"""

    knowledge = state["knowledge"]
    build = state["build"]
    workspace = state["workspace"]
    retry_count = dict(state.get("retry_count", {}))
    history = list(state.get("stage_history", []))
    stage_status = dict(state.get("stage_status", {}))
    artifacts = dict(state.get("artifacts", {}))
    stage = PocStage()
    paths = PocStagePaths(workspace)

    try:
        poc = stage.run(knowledge=knowledge, build=build, workspace=workspace)
        artifacts["poc"] = {
            "poc_context_yaml": str(paths.poc_context_yaml),
            "poc_plan_yaml": str(paths.poc_plan_yaml),
            "dockerfile": str(paths.dockerfile),
            "run_script": str(paths.run_script),
            "poc_log": str(paths.poc_log),
            "crash_report": str(paths.crash_report),
            "poc_artifact_yaml": str(paths.poc_artifact_yaml),
            "run_verify_yaml": str(paths.run_verify_yaml),
        }

        if poc.execution_success and poc.reproducer_verified:
            history.append({"stage": "poc", "status": "success"})
            stage_status["poc"] = "success"
            return {
                "poc": poc,
                "current_stage": "verify",
                "review_stage": "",
                "human_action_required": False,
                "review_reason": "",
                "stage_history": history,
                "stage_status": stage_status,
                "artifacts": artifacts,
                "last_error": None,
            }

        if poc.execution_success and not poc.reproducer_verified:
            # 脚本跑通了但没打到目标行为；仍然推进 verify，让 verify 独立判定（任务 0 H5）
            history.append({
                "stage": "poc",
                "status": "executed_but_unverified",
                "note": "PoC executed but no expected behavior observed; deferring to verify for independent judgment",
            })
            stage_status["poc"] = "executed_but_unverified"
            return {
                "poc": poc,
                "current_stage": "verify",
                "review_stage": "",
                "human_action_required": False,
                "review_reason": "",
                "stage_history": history,
                "stage_status": stage_status,
                "artifacts": artifacts,
                "last_error": None,
            }

        # execution_success=False
        retry_count["poc"] = retry_count.get("poc", 0) + 1
        history.append({"stage": "poc", "status": "failed", "error": poc.execution_logs})
        stage_status["poc"] = "failed"
        return {
            "poc": poc,
            "current_stage": "poc",
            "retry_count": retry_count,
            "review_stage": "poc",
            "review_reason": "poc stage completed without a successful execution",
            "stage_history": history,
            "stage_status": stage_status,
            "artifacts": artifacts,
            "last_error": "poc stage completed without a successful execution",
        }
    except Exception as error:
        retry_count["poc"] = retry_count.get("poc", 0) + 1
        history.append({"stage": "poc", "status": "failed", "error": str(error)})
        stage_status["poc"] = "failed"
        artifacts["poc"] = {
            "poc_dir": str(paths.poc_dir),
            "payloads_dir": str(paths.payloads_dir),
            "inputs_dir": str(paths.inputs_dir),
        }
        return {
            "current_stage": "poc",
            "retry_count": retry_count,
            "review_stage": "poc",
            "review_reason": "poc stage raised an exception",
            "stage_history": history,
            "stage_status": stage_status,
            "artifacts": artifacts,
            "last_error": str(error),
        }
