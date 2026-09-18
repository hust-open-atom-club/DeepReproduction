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
import binascii
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

from app.config import build_chat_model, load_app_config
from app.schemas.build_artifact import BuildArtifact
from app.schemas.knowledge import KnowledgeModel, ReproductionRecipe
from app.schemas.poc_artifact import PoCArtifact
from app.stages.asan_hardening import ensure_no_pie_for_asan
from app.stages.build import BuildStage, parse_llm_json_payload
from app.templates.poc_templates import format_template_for_prompt, select_template
from app.tools.docker_tools import DockerBuildRequest, DockerRunRequest, DockerTool
from app.tools.file_tools import FileTool
from app.tools.log_parsing import (
    drop_generic_sanitizer_labels,
    drop_weak_crash_labels_when_specific,
    ensure_specific_crash_token_in_patterns,
    extract_block as _extract_block_module,
    extract_execution_observation as _extract_execution_observation_module,
    filter_hits_for_specific_sanitizer,
    haystack_has_specific_sanitizer_bug,
    match_patterns as _match_patterns_module,
    should_outer_asan_preload,
    specific_sanitizer_bugs_in,
)
from app.tools.patch_tools import find_patch_diff
from app.tools import ossfuzz as ossfuzz_tools


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
    recipe_base64_blobs: list[str] = Field(
        default_factory=list,
        description="Base64 payload blobs extracted from reproduction recipes (source of truth for binary-safe decode).",
    )
    dataset_poc_base64_blobs: list[str] = Field(
        default_factory=list,
        description="Base64 blobs for binary/text PoCs discovered under Dataset/*/vuln_pocs.",
    )
    dataset_poc_filenames: list[str] = Field(
        default_factory=list,
        description="Filenames corresponding to dataset_poc_base64_blobs.",
    )
    dataset_payload_cursor: int = Field(
        default=0,
        description="Index into dataset_poc_base64_blobs selected for the current attempt.",
    )
    expected_error_patterns: list[str] = Field(default_factory=list, description="Expected error patterns.")
    expected_stack_keywords: list[str] = Field(default_factory=list, description="Expected stack keywords.")
    candidate_entrypoints: list[str] = Field(default_factory=list, description="Candidate entrypoints.")
    candidate_trigger_files: list[str] = Field(default_factory=list, description="Files likely related to triggering.")
    candidate_cli_flags: list[str] = Field(default_factory=list, description="Command-line flags discovered from hints.")
    reference_poc_summaries: list[str] = Field(default_factory=list, description="Reference PoC summaries.")
    repo_evidence_blocks: list[str] = Field(default_factory=list, description="README/tests/examples excerpts.")
    chosen_strategy: str = Field(default="", description="Strategy selected for this PoC run.")
    chosen_strategy_rationale: str = Field(default="", description="Why the current strategy was selected.")
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


class PocStrategyDecision(BaseModel):
    """High-level strategy decision before generating a concrete PoC."""

    chosen_strategy: str = Field(default="llm_synthesized", description="Selected strategy.")
    rationale: str = Field(default="", description="Reason for choosing the strategy.")
    evidence: list[str] = Field(default_factory=list, description="Short evidence bullets supporting the strategy.")


class PocPlanner:
    """Encapsulates PoC-stage planning decisions."""

    def __init__(self, stage: "PocStage") -> None:
        self.stage = stage

    def plan(self, knowledge: KnowledgeModel, build: BuildArtifact, context: PocContext) -> PocPlan:
        if not context.chosen_strategy:
            strategy = self.select_strategy(knowledge=knowledge, build=build, context=context)
            if strategy is None:
                raise RuntimeError("poc_agent did not return a valid strategy.")
            context.chosen_strategy = strategy.chosen_strategy
            context.chosen_strategy_rationale = strategy.rationale

        if context.chosen_strategy in {"dataset_poc", "reproduction_recipe"}:
            authoritative = self.stage._build_authoritative_poc_plan(
                knowledge=knowledge,
                build=build,
                context=context,
            )
            if authoritative is not None:
                self.stage._persist_poc_llm_trace(
                    context.planner_attempt,
                    "decision.txt",
                    (
                        f"Skipped LLM plan generation for strategy={context.chosen_strategy}; "
                        "built a deterministic plan from authoritative payload bytes."
                    ),
                )
                return self.stage._apply_vulnerable_hdf5_cve_match_policy_if_gated(
                    authoritative, build
                )

        llm_plan = self.try_llm_plan(knowledge=knowledge, build=build, context=context)
        if llm_plan is not None:
            normalized = self.stage._normalize_poc_plan(
                llm_plan,
                repo_url=context.repo_url,
                recipe_base64_blobs=context.recipe_base64_blobs,
                dataset_poc_base64_blobs=context.dataset_poc_base64_blobs,
                dataset_poc_filenames=context.dataset_poc_filenames,
            )
            normalized = self.stage._apply_authoritative_payload(normalized, context)
            return self.stage._apply_vulnerable_hdf5_cve_match_policy_if_gated(
                normalized, build
            )
        raise RuntimeError("poc_agent did not return a valid plan.")

    def select_strategy(self, knowledge: KnowledgeModel, build: BuildArtifact, context: PocContext) -> Optional[PocStrategyDecision]:
        preferred = self.stage._preferred_authoritative_strategy(context)
        if preferred is not None:
            decision = PocStrategyDecision(
                chosen_strategy=preferred,
                rationale=f"Authoritative {preferred} evidence is present; prefer it over llm_synthesized.",
                evidence=self.stage._authoritative_strategy_evidence(context, preferred),
            )
            self.stage._persist_poc_strategy_trace(
                context.planner_attempt,
                "decision.json",
                json.dumps(decision.model_dump(mode="json"), ensure_ascii=False, indent=2),
            )
            return decision

        try:
            model = build_chat_model(
                "poc_agent",
                temperature=0,
                timeout_seconds=load_app_config().runtime.poc_agent_timeout_seconds,
            )
        except Exception:
            return None

        prompt = self.stage._build_strategy_prompt(knowledge=knowledge, build=build, context=context)
        self.stage._persist_poc_strategy_trace(context.planner_attempt, "prompt.txt", prompt)
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
                    self.stage._persist_poc_strategy_trace(
                        context.planner_attempt,
                        "error.txt",
                        "LLM returned no content after 3 attempts.",
                    )
                    return None
                self.stage._persist_poc_strategy_trace(context.planner_attempt, "response.txt", raw_response_text)
                parsed = parse_llm_json_payload(raw_response)
                if parsed is None:
                    self.stage._persist_poc_strategy_trace(
                        context.planner_attempt,
                        "error.txt",
                        "LLM response could not be parsed into JSON.",
                    )
                    return None
                self.stage._persist_poc_strategy_trace(
                    context.planner_attempt,
                    "parsed.json",
                    json.dumps(parsed, ensure_ascii=False, indent=2),
                )
                decision = PocStrategyDecision(**parsed)
                if decision.chosen_strategy not in self.stage._available_poc_strategies(context):
                    self.stage._persist_poc_strategy_trace(
                        context.planner_attempt,
                        "error.txt",
                        f"LLM selected unavailable strategy: {decision.chosen_strategy}",
                    )
                    return None
                return decision
            except Exception as error:
                error_text = str(error)
                retry_errors.append(f"Attempt {invoke_attempt}: {error_text}")
                if self.stage._should_retry_llm_request(error_text) and invoke_attempt < max_attempts:
                    continue
                self.stage._persist_poc_strategy_trace(
                    context.planner_attempt,
                    "error.txt",
                    "\n".join(retry_errors),
                )
                return None
        self.stage._persist_poc_strategy_trace(
            context.planner_attempt,
            "error.txt",
            "\n".join(retry_errors) or "LLM request failed without a response.",
        )
        return None

    def replan_after_failure(
        self,
        knowledge: KnowledgeModel,
        build: BuildArtifact,
        context: PocContext,
        previous_plan: PocPlan,
        previous_artifact: PoCArtifact,
    ) -> Optional[PocPlan]:
        failure_kind = self.stage._classify_failure_kind(previous_artifact.execution_logs)
        preferred = self.stage._preferred_authoritative_strategy(context)
        strategy = context.chosen_strategy
        rationale = context.chosen_strategy_rationale
        if failure_kind == "payload_invalid" and preferred and preferred != strategy:
            strategy = preferred
            rationale = f"Switched to {preferred} after payload parse/compile failure."
        retry_context = context.model_copy(
            update={
                "planner_attempt": context.planner_attempt + 1,
                "chosen_strategy": strategy,
                "chosen_strategy_rationale": rationale,
                "previous_failure_kind": failure_kind,
                "previous_execution_log": previous_artifact.execution_logs[:6000],
            }
        )
        context.chosen_strategy = strategy
        context.chosen_strategy_rationale = rationale
        return self.try_llm_plan(
            knowledge=knowledge,
            build=build,
            context=retry_context,
            previous_plan=previous_plan,
            previous_artifact=previous_artifact,
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
            model = build_chat_model(
                "poc_agent",
                temperature=0,
                timeout_seconds=load_app_config().runtime.poc_agent_timeout_seconds,
            )
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
                    normalized_plan = self.stage._normalize_poc_plan(
                        plan,
                        repo_url=context.repo_url,
                        recipe_base64_blobs=context.recipe_base64_blobs,
                        dataset_poc_base64_blobs=context.dataset_poc_base64_blobs,
                        dataset_poc_filenames=context.dataset_poc_filenames,
                    )
                    normalized_plan = self.stage._apply_authoritative_payload(normalized_plan, context)
                    normalized_plan = self.stage._apply_vulnerable_hdf5_cve_match_policy_if_gated(
                        normalized_plan, build
                    )
                    if not self.stage._is_valid_replan_candidate(previous_plan, normalized_plan, failure_kind=failure_kind):
                        self.stage._persist_poc_llm_trace(
                            context.planner_attempt,
                            "error.txt",
                            f"Rejected replan candidate for failure kind: {failure_kind or 'unknown'}",
                        )
                        return None
                    return normalized_plan
                return self.stage._apply_vulnerable_hdf5_cve_match_policy_if_gated(plan, build)
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
    MAX_SYNTHESIZED_PAYLOAD_CHARS = 64 * 1024
    DATASET_POC_BYTE_LIMIT = 256 * 1024
    # Maximum number of Dataset/*/vuln_pocs payloads collected per CVE. Kept
    # separate from REFERENCE_POC_BLOCK_LIMIT: one CVE can harvest multiple
    # ClusterFuzz testcases and only some of them trigger the sanitizer.
    DATASET_POC_COUNT_LIMIT = 8
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
        recipe_base64_blobs = self._extract_recipe_base64_blobs(knowledge.reproduction_recipes)
        dataset_poc_filenames, dataset_poc_base64_blobs = self._collect_dataset_poc_payloads(knowledge.cve_id)
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
            recipe_base64_blobs=recipe_base64_blobs,
            dataset_poc_base64_blobs=dataset_poc_base64_blobs,
            dataset_poc_filenames=dataset_poc_filenames,
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

    def _persist_poc_strategy_trace(self, planner_attempt: int, filename: str, content: str) -> None:
        poc_dir = getattr(self, "_active_poc_dir", None)
        if not poc_dir:
            return
        attempt_dir = Path(poc_dir) / "strategy" / f"attempt-{planner_attempt}"
        self.file_tool.ensure_dir(str(attempt_dir))
        self.file_tool.write_text(str(attempt_dir / filename), (content or "").rstrip() + "\n")

    def _write_yaml_file(self, path: Path, payload: Any) -> None:
        """Persist YAML using one consistent formatting policy."""

        self.file_tool.write_text(
            str(path),
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
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

    def _available_poc_strategies(self, context: PocContext) -> list[str]:
        strategies = ["llm_synthesized"]
        if context.reproduction_recipe_summaries or context.recipe_base64_blobs:
            strategies.insert(0, "reproduction_recipe")
        if context.dataset_poc_base64_blobs or context.dataset_poc_filenames or context.reference_poc_summaries:
            insert_at = 1 if strategies and strategies[0] == "reproduction_recipe" else 0
            strategies.insert(insert_at, "dataset_poc")
        return strategies

    def _preferred_authoritative_strategy(self, context: PocContext) -> Optional[str]:
        """Prefer dataset/recipe strategies whenever authoritative payloads exist."""

        available = self._available_poc_strategies(context)
        if "dataset_poc" in available and (
            context.dataset_poc_base64_blobs or context.dataset_poc_filenames or context.reference_poc_summaries
        ):
            return "dataset_poc"
        if "reproduction_recipe" in available and (
            context.reproduction_recipe_summaries or context.recipe_base64_blobs
        ):
            return "reproduction_recipe"
        return None

    def _authoritative_strategy_evidence(self, context: PocContext, strategy: str) -> list[str]:
        evidence: list[str] = []
        if strategy == "dataset_poc":
            if context.dataset_poc_filenames:
                evidence.append(f"dataset_poc_files={context.dataset_poc_filenames}")
            elif context.reference_poc_summaries:
                evidence.append(f"reference_poc_blocks={len(context.reference_poc_summaries)}")
            if context.dataset_poc_base64_blobs:
                evidence.append(f"dataset_poc_blobs={len(context.dataset_poc_base64_blobs)}")
        elif strategy == "reproduction_recipe":
            if context.reproduction_recipe_summaries:
                evidence.append(f"reproduction_recipes={len(context.reproduction_recipe_summaries)}")
            if context.recipe_base64_blobs:
                evidence.append(f"recipe_base64_blobs={len(context.recipe_base64_blobs)}")
        return evidence or [f"strategy={strategy}"]

    def _dataset_payload_index(self, context: PocContext) -> int:
        """Return the dataset payload index to use for the current attempt.

        Multiple ClusterFuzz testcases can be harvested for one CVE (e.g.
        OpenEXR). The first seed is not guaranteed to trigger the sanitizer;
        cycle through them across replan attempts instead of always retrying
        the same payload.
        """

        blobs = context.dataset_poc_base64_blobs or []
        if not blobs:
            return 0
        cursor = max(int(context.dataset_payload_cursor or 0), 0)
        return cursor % len(blobs)

    def _has_untried_dataset_payload(self, context: PocContext) -> bool:
        """True when the dataset_poc strategy still has an unused seed to try."""

        blobs = context.dataset_poc_base64_blobs or []
        if not blobs:
            return False
        return int(context.dataset_payload_cursor or 0) < len(blobs)

    def _apply_authoritative_payload(self, plan: PocPlan, context: PocContext) -> PocPlan:
        """Force recipe/dataset bytes into the plan when those strategies are active."""

        strategy = (context.chosen_strategy or "").strip()
        if strategy == "dataset_poc" and context.dataset_poc_base64_blobs:
            index = self._dataset_payload_index(context)
            raw = base64.b64decode(context.dataset_poc_base64_blobs[index])
            plan.payload_content = raw.decode("latin-1", errors="replace")
            if context.dataset_poc_filenames and index < len(context.dataset_poc_filenames):
                plan.payload_filename = Path(context.dataset_poc_filenames[index]).name
            if not plan.source_of_truth or plan.source_of_truth == "llm_synthesized":
                plan.source_of_truth = "dataset_poc"
            plan = self._prefer_compact_dataset_xcf_payload(
                plan,
                dataset_poc_filenames=context.dataset_poc_filenames,
                dataset_poc_base64_blobs=context.dataset_poc_base64_blobs,
            )
            plan = self._prefer_ossfuzz_minimized_payload(
                plan,
                dataset_poc_filenames=context.dataset_poc_filenames,
                dataset_poc_base64_blobs=context.dataset_poc_base64_blobs,
            )
            # Dataset files sometimes store the recipe's ``base64 -d`` input
            # (transport blob → still base64 text). Unwrap only script-like
            # nested blobs; binary/ClusterFuzz seeds are left alone.
            plan.payload_content = self._unwrap_nested_base64_payload(plan.payload_content)
            return self._sync_payload_filename_into_command(plan)

        if strategy == "reproduction_recipe":
            if context.recipe_base64_blobs:
                raw = base64.b64decode(context.recipe_base64_blobs[0])
                plan.payload_content = self._unwrap_nested_base64_payload(
                    raw.decode("latin-1", errors="replace")
                )
                if not plan.source_of_truth or plan.source_of_truth == "llm_synthesized":
                    plan.source_of_truth = "reproduction_recipe"
                return plan
            extracted = self._extract_embedded_payload_from_recipe_summaries(context.reproduction_recipe_summaries)
            if extracted is not None:
                content, filename = extracted
                plan.payload_content = self._unwrap_nested_base64_payload(content)
                plan.payload_filename = Path(filename).name
                if not plan.source_of_truth or plan.source_of_truth == "llm_synthesized":
                    plan.source_of_truth = "reproduction_recipe"
        return plan

    def _build_authoritative_poc_plan(
        self,
        knowledge: KnowledgeModel,
        build: BuildArtifact,
        context: PocContext,
    ) -> Optional[PocPlan]:
        """Build a concrete PoC plan without LLM when authoritative payload bytes exist."""

        strategy = (context.chosen_strategy or "").strip()
        has_dataset = bool(context.dataset_poc_base64_blobs)
        has_recipe_blob = bool(context.recipe_base64_blobs)
        if strategy == "dataset_poc" and not has_dataset:
            return None
        if strategy == "reproduction_recipe" and not (has_recipe_blob or context.reproduction_recipe_summaries):
            return None
        if strategy not in {"dataset_poc", "reproduction_recipe"}:
            return None
        if strategy == "reproduction_recipe" and not has_recipe_blob:
            # Only skip LLM when we can recover payload bytes from recipe blobs/summaries.
            probe = PocPlan(payload_filename="poc.txt", payload_content="trigger\n", target_binary="placeholder")
            probe = self._apply_authoritative_payload(probe, context)
            if probe.payload_content.strip() in {"", "trigger"}:
                return None

        if strategy == "dataset_poc" and context.dataset_poc_filenames:
            index = self._dataset_payload_index(context)
            if index < len(context.dataset_poc_filenames):
                payload_filename = Path(context.dataset_poc_filenames[index]).name
            else:
                payload_filename = "poc.bin"
        elif strategy == "reproduction_recipe":
            payload_filename = "poc.bin"
            for recipe in knowledge.reproduction_recipes[:4]:
                title = (recipe.source_title or "").strip()
                if title and "." in Path(title).name:
                    payload_filename = Path(title).name
                    break
        else:
            payload_filename = "poc.bin"

        target_binary = self._select_target_binary(build, context, payload_filename=payload_filename)
        payload_path = f"/workspace/artifacts/poc/payloads/{payload_filename}"
        target_args = self._select_target_args(
            knowledge=knowledge,
            payload_filename=payload_filename,
            context=context,
            target_binary=target_binary,
        )
        if not target_args:
            target_args = [payload_path]

        crash_type = (knowledge.vulnerability_type or "").strip()
        lowered = crash_type.lower()
        if "use-after-free" in lowered or "uaf" in lowered:
            crash_type = "heap-use-after-free"
        elif "overflow" in lowered or "over-read" in lowered or "overread" in lowered:
            crash_type = "heap-buffer-overflow"

        stderr_patterns = list(knowledge.expected_error_patterns) or ["AddressSanitizer"]
        stack_keywords = [item for item in knowledge.expected_stack_keywords if item][:8]

        plan = PocPlan(
            trigger_mode="cli-file",
            target_binary=target_binary,
            target_args=target_args,
            environment_variables={},
            payload_filename=payload_filename,
            payload_content="trigger\n",
            auxiliary_files={},
            run_command="",
            expected_exit_code=None,
            expected_stdout_patterns=[],
            expected_stderr_patterns=stderr_patterns,
            expected_stack_keywords=stack_keywords,
            expected_crash_type=crash_type,
            source_of_truth=strategy,
            confidence="high",
            rationale=(
                f"Deterministic {strategy} plan using authoritative payload "
                f"{payload_filename} against {target_binary}."
            ),
            dockerfile_override=None,
            run_script_override=None,
        )
        normalized = self._normalize_poc_plan(
            plan,
            repo_url=context.repo_url,
            recipe_base64_blobs=context.recipe_base64_blobs,
            dataset_poc_base64_blobs=context.dataset_poc_base64_blobs,
            dataset_poc_filenames=context.dataset_poc_filenames,
        )
        normalized = self._apply_authoritative_payload(normalized, context)
        if not (normalized.payload_content or "").strip() or normalized.payload_content.strip() == "trigger":
            return None
        return self._apply_vulnerable_hdf5_cve_match_policy_if_gated(normalized, build)

    def _apply_vulnerable_hdf5_cve_match_policy_if_gated(
        self,
        plan: PocPlan,
        build: BuildArtifact,
    ) -> PocPlan:
        """Tighten CVE match + ASAN when build used the vulnerable-HDF5 gate."""

        script = build.build_script_content or ""
        if BuildStage.VULN_HDF5_MARKER not in script:
            return plan
        (
            plan.expected_stderr_patterns,
            plan.expected_stack_keywords,
            plan.expected_crash_type,
            plan.environment_variables,
        ) = BuildStage.apply_vulnerable_hdf5_cve_match_policy(
            expected_stderr_patterns=list(plan.expected_stderr_patterns),
            expected_stack_keywords=list(plan.expected_stack_keywords),
            expected_crash_type=plan.expected_crash_type or "",
            environment_variables=dict(plan.environment_variables),
        )
        return plan

    def _extract_embedded_payload_from_recipe_summaries(
        self,
        summaries: list[str],
    ) -> Optional[tuple[str, str]]:
        """Best-effort recovery of CIL/S-expression payload text from recipe YAML dumps."""

        for summary in summaries:
            lines = [line.rstrip() for line in summary.splitlines()]
            block: list[str] = []
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    if block:
                        break
                    continue
                if stripped.startswith("(") or (
                    block and (stripped.startswith((")", ";")) or stripped.endswith(")"))
                ):
                    block.append(stripped)
                    continue
                if block:
                    break
            if len(block) >= 3 and block[0].startswith("("):
                content = "\n".join(block).rstrip() + "\n"
                filename = "poc.cil" if any(
                    token in content.lower()
                    for token in ("classmap", "classmapping", "classpermission", "(class ", "(type ")
                ) else "poc.txt"
                return content, filename
        return None

    def _build_strategy_prompt(self, knowledge: KnowledgeModel, build: BuildArtifact, context: PocContext) -> str:
        available_strategies = self._available_poc_strategies(context)
        sections = [
            "You are the PoC Strategy Selector for a vulnerability reproduction framework.",
            "Choose exactly one strategy for generating the next PoC attempt.",
            "The strategy must stay stable across later retry turns unless the implementation proves it impossible.",
            "Return exactly one JSON object and no markdown fences.",
            "Schema:",
            json.dumps(
                {
                    "chosen_strategy": "reproduction_recipe|dataset_poc|llm_synthesized",
                    "rationale": "string",
                    "evidence": ["string"],
                },
                ensure_ascii=True,
            ),
            f"Available strategies: {json.dumps(available_strategies, ensure_ascii=False)}",
            "Preference order when multiple strategies are available: reproduction_recipe > dataset_poc > llm_synthesized.",
            "Choose reproduction_recipe or dataset_poc whenever an explicit recipe/base64/attachment PoC exists; do not invent a new payload in that case.",
            f"CVE: {knowledge.cve_id}",
            f"Summary: {knowledge.summary}",
            f"Vulnerability type: {knowledge.vulnerability_type}",
            f"Build target binary: {build.binary_or_entrypoint or build.expected_binary_path or ''}",
            f"Expected error patterns: {json.dumps(knowledge.expected_error_patterns, ensure_ascii=False)}",
            f"Expected stack keywords: {json.dumps(knowledge.expected_stack_keywords, ensure_ascii=False)}",
            f"Reproduction hints: {json.dumps(knowledge.reproduction_hints, ensure_ascii=False)}",
            "Structured reproduction recipes:",
            "\n\n---\n\n".join(context.reproduction_recipe_summaries[:4]) or "<empty>",
            "Reference PoC excerpts:",
            "\n\n---\n\n".join(self._reference_poc_prompt_blocks(context.reference_poc_summaries, detailed=False)) or "<empty>",
            f"Candidate entrypoints: {json.dumps(context.candidate_entrypoints, ensure_ascii=False)}",
            f"Candidate CLI flags: {json.dumps(context.candidate_cli_flags, ensure_ascii=False)}",
            f"Inferred input modes: {json.dumps(context.inferred_input_modes, ensure_ascii=False)}",
            f"Patch changed functions: {json.dumps(context.patch_changed_functions, ensure_ascii=False)}",
            f"Patch added checks: {json.dumps(context.patch_added_checks, ensure_ascii=False)}",
            f"Patch error strings: {json.dumps(context.patch_error_strings, ensure_ascii=False)}",
            "Patch excerpt:",
            context.patch_diff_excerpt or "<empty>",
            "Repository evidence excerpts:",
            "\n\n---\n\n".join(context.repo_evidence_blocks[:4]) or "<empty>",
        ]
        return "\n".join(sections)

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
        has_reference_evidence = bool(context.reproduction_recipe_summaries or reference_poc_blocks)
        sections = [
            "You are the PoC Agent for a vulnerability reproduction framework.",
            "Infer the most plausible minimal reproducer from patch context, repository evidence, build outputs, and existing hints.",
            f"The strategy for this attempt is fixed to: {context.chosen_strategy or 'llm_synthesized'}.",
            f"Strategy rationale: {context.chosen_strategy_rationale or '<empty>'}",
            "Use that strategy to build the concrete PoC plan. Do not silently switch strategies inside this turn.",
            "Give substantial weight to semantic understanding of the vulnerability and the likely trigger path.",
            "Prefer a minimal reproducible trigger over a large script.",
            "Never invent multi-kilobyte or megabyte payloads when a recipe base64 blob or dataset PoC already exists.",
            "When recipe/dataset base64 is present, set payload_content to that base64 string (or the decoded bytes semantics) and keep the original filename when possible.",
            "Prefer CLI flags from reproduction recipes/hints over inventing short options; if a flag is rejected as unknown, switch to flags shown in evidence.",
            "If the build target is a shared library or Qt plugin (.so/.dylib/.dll, especially under imageformats/), never execute it directly. Use trigger_mode=library-harness with a tiny loader (for Qt: QImageReader + QT_PLUGIN_PATH).",
            "Adapt any existing PoC or hint to the current workspace layout inside Docker.",
            f"The build image keeps the checked-out project under {self._container_project_dir(context.repo_url)}.",
            "The repository is mounted at /workspace/repo.",
            "Prefer compiled binaries from the build-image project directory. /workspace/repo is a mounted source tree and may not contain built executables.",
            "Do not prepend `make` / `make -C ...` rebuilds in run_command; the ASan binary is already produced by the build stage. Re-running make often rebuilds without sanitizers or fails on docs tools (e.g. xmlto).",
            "Payload files should normally be written under /workspace/artifacts/poc/payloads/ and auxiliary files under /workspace/artifacts/poc/inputs/.",
            "You may freely change payload filename, suffix, on-disk format, auxiliary files, and wrapper/decoding steps when that improves the trigger.",
            "ASAN BUILD RULE: whenever run_command compiles the PoC with a sanitizer flag (e.g. -fsanitize=address or -fsanitize=fuzzer,address), ALWAYS add -no-pie to that compile/link command. On WSL2 (vm.mmap_rnd_bits=32) a PIE binary linked against ASan can crash with a bare SIGSEGV (exit 139) during ASan startup, before any ASan report; -no-pie makes the run deterministic.",
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
        inferred_trigger = context.inferred_input_modes[0] if context.inferred_input_modes else ""
        template = select_template(
            trigger_mode=inferred_trigger,
            vulnerability_type=knowledge.vulnerability_type or "",
            inferred_input_modes=context.inferred_input_modes,
            chosen_strategy=context.chosen_strategy,
            target_binary=context.target_binary,
        )
        if template is not None:
            sections.extend([
                "",
                "PoC template reference (adapt to the current CVE, do not copy verbatim):",
                format_template_for_prompt(template),
            ])
        if not has_reference_evidence:
            sections.extend(
                [
                    "Recovered PoC evidence status:",
                    "No explicit PoC or reproduction recipe was recovered from the knowledge sources.",
                    "You must synthesize the smallest plausible trigger from patch semantics, repository evidence, build outputs, candidate entrypoints, input-mode hints, and expected crash behavior.",
                    "A synthesized PoC is valid even if it is only a minimal argument string, input file, or wrapper script that exercises the vulnerable path.",
                    "When you are synthesizing instead of adapting an existing artifact, set source_of_truth to llm_synthesized.",
                ]
            )
        elif context.chosen_strategy == "reproduction_recipe":
            sections.extend(
                [
                    "Strategy guidance:",
                    "Use the structured reproduction recipes as the primary source of truth for payload construction and triggering.",
                    "If artifact_generation_commands contain base64 decode steps, copy the base64 blob into payload_content instead of inventing new bytes.",
                    "You may simplify, decode, wrap, or relocate the recipe steps, but keep the recipe semantics and binary payload intact.",
                ]
            )
        elif context.chosen_strategy == "dataset_poc":
            sections.extend(
                [
                    "Strategy guidance:",
                    "Use the dataset PoC excerpts as the primary source of truth.",
                    "When a dataset PoC is marked ENCODING: base64, put that exact base64 blob into payload_content and keep the FILE name as payload_filename.",
                    "Do not invent a different payload; only adapt the run command/flags to the current workspace.",
                ]
            )

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
                    f"Keep using strategy: {context.chosen_strategy or 'llm_synthesized'} unless the logs prove it impossible.",
                    "Adjust the plan to improve the trigger while staying minimal.",
                    "Replan contract:",
                    "- If docker image build failed, you must return a new dockerfile_override.",
                    "- If the payload failed to parse/compile (payload_invalid), you must change payload_content and/or payload_filename (and keep DSL/CIL syntax valid).",
                    "- If the target ran but did not trigger the expected behavior, you must modify the payload, auxiliary files, run command, environment, or run_script_override.",
                    "- If container execution failed, you must modify how the target is invoked, preferably via run_script_override or by changing target_binary/target_args/run_command/environment.",
                    "- Do not only change rationale, confidence, or source_of_truth.",
                ]
            )
            if failure_kind == "payload_invalid":
                sections.extend(
                    [
                        "Payload invalid guidance:",
                        "The previous payload was rejected by the target parser/compiler before the vulnerable path ran.",
                        "Do not keep inventing similar invalid syntax. Prefer any dataset/recipe payload verbatim, or emit a minimal syntactically valid policy/input for this DSL.",
                        "Changing only flags or the run command is insufficient for payload_invalid.",
                    ]
                )
        return "\n".join(sections)

    def _normalize_poc_plan(
        self,
        plan: PocPlan,
        repo_url: str = "",
        recipe_base64_blobs: Optional[list[str]] = None,
        dataset_poc_base64_blobs: Optional[list[str]] = None,
        dataset_poc_filenames: Optional[list[str]] = None,
    ) -> PocPlan:
        if not plan.payload_filename:
            plan.payload_filename = "poc.txt"
        if not plan.payload_content:
            plan.payload_content = "trigger\n"
        authoritative_blobs = list(recipe_base64_blobs or []) + list(dataset_poc_base64_blobs or [])
        plan.payload_content = self._resolve_payload_content(plan.payload_content, authoritative_blobs)
        if (
            dataset_poc_filenames
            and authoritative_blobs
            and self._payload_matches_blob(plan.payload_content, authoritative_blobs)
            and (not plan.payload_filename or plan.payload_filename in {"poc.txt", "trigger", "payload"})
        ):
            plan.payload_filename = dataset_poc_filenames[0]
        plan.payload_filename = Path(plan.payload_filename).name
        plan = self._prefer_compact_dataset_xcf_payload(
            plan,
            dataset_poc_filenames=dataset_poc_filenames,
            dataset_poc_base64_blobs=dataset_poc_base64_blobs,
        )
        plan = self._prefer_ossfuzz_minimized_payload(
            plan,
            dataset_poc_filenames=dataset_poc_filenames,
            dataset_poc_base64_blobs=dataset_poc_base64_blobs,
        )
        plan = self._sync_payload_filename_into_command(plan)
        plan.auxiliary_files = self._normalize_auxiliary_files(plan.auxiliary_files)
        plan.target_binary = self._normalize_target_binary(plan.target_binary, repo_url)
        plan.target_binary = self._correct_nested_src_binary_path(plan.target_binary, repo_url)
        plan.target_binary = self._correct_qt_plugin_binary_path(plan.target_binary)
        plan.target_binary = self._rewrite_non_executable_or_qt_plugin_target(
            plan.target_binary, plan.payload_filename, repo_url
        )
        plan.target_binary = self._correct_qt_plugin_binary_path(plan.target_binary)
        plan.target_args = [self._normalize_workspace_arg(arg, plan.payload_filename) for arg in plan.target_args]
        if not plan.run_command:
            plan.run_command = self._build_run_command(plan.target_binary, plan.target_args)
        else:
            plan.run_command = self._normalize_run_command(plan.run_command, plan.payload_filename, repo_url)
            plan.run_command = self._rewrite_nested_src_binary_in_command(plan.run_command, repo_url)
            plan.run_command = self._strip_pre_run_make_rebuild(plan.run_command)
            plan.run_command = self._align_run_command_with_target_binary(plan.run_command, plan.target_binary)
        plan = self._coerce_ossfuzz_harness_file_argv(plan)
        plan = self._ensure_shared_library_harness(plan)
        plan.run_command = ensure_no_pie_for_asan(plan.run_command)
        plan.expected_stderr_patterns = ensure_specific_crash_token_in_patterns(
            plan.expected_stderr_patterns,
            plan.expected_crash_type,
        )
        if not plan.expected_stderr_patterns and plan.expected_crash_type:
            plan.expected_stderr_patterns = [plan.expected_crash_type]
        crash_extra = [plan.expected_crash_type] if plan.expected_crash_type else []
        plan.expected_stderr_patterns = drop_generic_sanitizer_labels(
            plan.expected_stderr_patterns,
            extra_texts=crash_extra,
        )
        plan.expected_stderr_patterns = drop_weak_crash_labels_when_specific(
            plan.expected_stderr_patterns,
            extra_texts=crash_extra,
        )
        extra_asan_context = list(plan.expected_stderr_patterns) + crash_extra
        plan.expected_stack_keywords = drop_generic_sanitizer_labels(
            sorted(set(plan.expected_stack_keywords)),
            extra_texts=extra_asan_context,
        )
        return plan

    def _execute_poc_plan(self, paths: PocStagePaths, plan_meta: dict, plan: PocPlan) -> PoCArtifact:
        payload_path = paths.payloads_dir / plan.payload_filename
        self.file_tool.write_latin1(str(payload_path), plan.payload_content)

        auxiliary_paths: list[str] = []
        for name, content in plan.auxiliary_files.items():
            target_dir = paths.inputs_dir if "/" not in name else paths.poc_dir
            target_path = target_dir / name
            self.file_tool.write_latin1(str(target_path), content)
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
            "target_binary_echo": self._escape_for_echo(plan.target_binary),
            "run_command": plan.run_command,
            "run_command_echo": self._escape_for_echo(plan.run_command),
            "run_command_shell": self._shell_quote(plan.run_command or "true"),
            "outer_asan_preload": should_outer_asan_preload(
                plan.run_command, plan.trigger_mode
            ),
            "trigger_timeout_sec": 120,
        }

        dockerfile_content = (
            plan.dockerfile_override.rstrip() + "\n"
            if plan.dockerfile_override
            else self._render_template("poc.Dockerfile.j2", docker_context)
        )
        run_script_content = (
            ensure_no_pie_for_asan(plan.run_script_override.rstrip()) + "\n"
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
        failure_kind = self._classify_failure_kind(execution_logs)
        if failure_kind == "payload_invalid":
            # Parser/compiler rejected the payload before the vulnerable path; treat as a failed
            # PoC attempt so internal/outer retries replan instead of advancing to verify.
            execution_success = False
        run_verify_report = self._build_run_verify_report(
            plan=plan,
            observation=observation,
            execution_logs=execution_logs,
            matched_error_patterns=matched_error_patterns,
            matched_stdout_patterns=matched_stdout_patterns,
            matched_stack_keywords=matched_stack_keywords,
        )
        if failure_kind == "payload_invalid":
            run_verify_report.eligible_for_verify = False
            run_verify_report.eligibility_reason = "payload_invalid: parser/compiler rejected payload"
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
        # False: do not inherit the parent workflow checkpointer. Internal state
        # carries PocStagePaths, which is not msgpack-serializable.
        return builder.compile(checkpointer=False)

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
            # When there are untried authoritative dataset payloads, skip the LLM
            # replan entirely and let the next "plan" node deterministically advance
            # to the next seed. LLM replans here are slow and, for dataset_poc, are
            # overwritten by _build_authoritative_poc_plan anyway.
            if self._has_untried_dataset_payload(current_context):
                replanned = None
            else:
                replanned = self.replan_after_failure(
                    knowledge=state["knowledge"],
                    build=state["build"],
                    context=current_context,
                    previous_plan=plan,
                    previous_artifact=outcome.artifact,
                )
            if replanned is not None:
                normalized = self._normalize_poc_plan(
                    replanned,
                    repo_url=current_context.repo_url,
                    recipe_base64_blobs=current_context.recipe_base64_blobs,
                    dataset_poc_base64_blobs=current_context.dataset_poc_base64_blobs,
                    dataset_poc_filenames=current_context.dataset_poc_filenames,
                )
                normalized = self._apply_authoritative_payload(normalized, current_context)
                updates["current_plan"] = self._apply_vulnerable_hdf5_cve_match_policy_if_gated(
                    normalized, state["build"]
                )
        return updates

    def _route_after_poc_execute(self, state: PocGraphState) -> str:
        outcome = state.get("outcome")
        attempt = state.get("attempt", 0)
        if outcome is None:
            return "done"
        if outcome.artifact.execution_success and outcome.artifact.reproducer_verified:
            return "done"
        # When multiple authoritative dataset payloads exist, keep cycling through
        # them until one triggers (or the cursor wraps past the last seed).
        context = state.get("current_context")
        if context is not None and self._has_untried_dataset_payload(context):
            return "plan"
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
                replanned = self._normalize_poc_plan(
                    replanned,
                    repo_url=current_context.repo_url,
                    recipe_base64_blobs=current_context.recipe_base64_blobs,
                    dataset_poc_base64_blobs=current_context.dataset_poc_base64_blobs,
                    dataset_poc_filenames=current_context.dataset_poc_filenames,
                )
                replanned = self._apply_authoritative_payload(replanned, current_context)
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
                "dataset_payload_cursor": context.dataset_payload_cursor + 1,
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
        if normalized_failure_kind == "payload_invalid":
            return any(
                [
                    previous_plan.payload_content != candidate_plan.payload_content,
                    previous_plan.payload_filename != candidate_plan.payload_filename,
                    previous_plan.auxiliary_files != candidate_plan.auxiliary_files,
                ]
            )
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
        seen: set[str] = set()
        # Prefer built outputs over source-tree files (AUTHORS, headers, …).
        for relative in ("build", "bin", "target/debug", "src"):
            root = repo_dir / relative
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                rel = str(path.relative_to(repo_dir)).replace("\\", "/")
                if rel in seen or self._looks_like_non_executable_source(rel):
                    continue
                suffix = path.suffix.lower()
                if suffix in {".so", ".dylib", ".dll"}:
                    name = path.name.lower()
                    posix = f"/{rel.lower()}"
                    if not (
                        name.startswith("kimg_")
                        or "/imageformats/" in posix
                        or "/bin/" in posix
                    ):
                        continue
                elif suffix not in {"", ".sh", ".py", ".pl", ".lua"}:
                    continue
                seen.add(rel)
                candidates.append(rel)
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
                "steps": self._compact_recipe_commands(recipe.steps),
                "repo_setup_commands": self._compact_recipe_commands(recipe.repo_setup_commands),
                "build_commands": self._compact_recipe_commands(recipe.build_commands),
                "artifact_generation_commands": self._compact_recipe_commands(recipe.artifact_generation_commands),
                "run_commands": self._compact_recipe_commands(recipe.run_commands),
                "expected_behavior": recipe.expected_behavior,
                "source_excerpt": self._truncate_text(recipe.source_excerpt, self.REFERENCE_POC_SUMMARY_CHAR_LIMIT),
                "confidence": recipe.confidence,
            }
            summaries.append(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).strip())
        return summaries

    def _compact_recipe_commands(self, commands: list[str], *, limit: int = 6, max_chars: int = 180) -> list[str]:
        """Keep recipe command lists short so PoC LLM prompts stay within timeout budgets."""

        compacted: list[str] = []
        for command in commands[:limit]:
            text = (command or "").strip()
            if not text:
                continue
            lowered = text.lower()
            if ("base64" in lowered or "printf '%s'" in lowered) and len(text) > max_chars:
                compacted.append(
                    self._truncate_text(
                        text,
                        max_chars,
                    )
                    + "  # large base64 payload omitted; use dataset/recipe blob bytes"
                )
            else:
                compacted.append(self._truncate_text(text, max_chars * 2))
        return compacted

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
                raw = path.read_bytes()
                if len(raw) > self.DATASET_POC_BYTE_LIMIT:
                    summaries.append(
                        f"FILE: {path.name}\nENCODING: omitted\nSIZE: {len(raw)}\nCONTENT:\n<payload too large; use dataset file bytes verbatim>"
                    )
                    continue
                if self._looks_like_binary_payload(raw) or path.suffix.lower() in {
                    ".bin",
                    ".abc",
                    ".dat",
                    ".raw",
                    ".xcf",
                    ".psd",
                    ".pcx",
                    ".tga",
                }:
                    encoded = base64.b64encode(raw).decode("ascii")
                    summaries.append(
                        f"FILE: {path.name}\nENCODING: base64\nSIZE: {len(raw)}\nCONTENT:\n"
                        f"{self._truncate_text(encoded, self.REFERENCE_POC_CHAR_LIMIT)}"
                    )
                else:
                    content = raw.decode("utf-8", errors="replace")
                    summaries.append(
                        f"FILE: {path.name}\nENCODING: text\nCONTENT:\n{self._truncate_text(content, self.REFERENCE_POC_CHAR_LIMIT)}"
                    )
        return summaries[: self.REFERENCE_POC_BLOCK_LIMIT]

    def _collect_dataset_poc_payloads(self, cve_id: str) -> tuple[list[str], list[str]]:
        """Return (filenames, base64 blobs) for authoritative dataset PoC payloads."""

        filenames: list[str] = []
        blobs: list[str] = []
        seen: set[str] = set()
        for prefix in ("Dataset", "source/Dataset"):
            poc_dir = Path(prefix) / cve_id / "vuln_data" / "vuln_pocs"
            if not poc_dir.exists():
                continue
            for path in sorted(poc_dir.iterdir()):
                if not path.is_file() or path.name in seen:
                    continue
                raw = path.read_bytes()
                if not raw or len(raw) > self.DATASET_POC_BYTE_LIMIT:
                    continue
                if path.suffix.lower() in {".zip", ".tar", ".tgz", ".gz", ".md", ".yaml", ".yml"}:
                    continue
                seen.add(path.name)
                filenames.append(path.name)
                blobs.append(base64.b64encode(raw).decode("ascii"))
                if len(blobs) >= self.DATASET_POC_COUNT_LIMIT:
                    break
            if len(blobs) >= self.DATASET_POC_COUNT_LIMIT:
                break
        return self._order_dataset_poc_payloads(filenames, blobs)

    @staticmethod
    def _order_dataset_poc_payloads(filenames: list[str], blobs: list[str]) -> tuple[list[str], list[str]]:
        """Rank dataset PoCs by evidence, not CVE id.

        crafted_bpp*.xcf beats ClusterFuzz XCF corpora when both exist. Otherwise
        OSS-Fuzz minimized seeds beat tiny harvested poc.* files: the latter often
        only null-deref (SEGV) while the minimized testcase is the sanitizer hit.
        """

        if not filenames or not blobs or len(filenames) != len(blobs):
            return filenames, blobs

        has_crafted_bpp = any(re.search(r"crafted_bpp\d*\.xcf$", name.lower()) for name in filenames)

        def rank(item: tuple[str, str]) -> tuple[int, int, str]:
            name, blob = item
            lowered = name.lower()
            try:
                size = len(base64.b64decode(blob))
            except Exception:
                size = 10**9
            if re.search(r"crafted_bpp\d*\.xcf$", lowered):
                return (0, size, lowered)
            if "clusterfuzz" in lowered or "testcase-minimized" in lowered:
                # Demote ClusterFuzz only when a crafted_bpp XCF seed is present.
                return (2, size, lowered) if has_crafted_bpp else (0, size, lowered)
            return (1, size, lowered)

        ordered = sorted(zip(filenames, blobs), key=rank)
        names, encoded = zip(*ordered)
        return list(names), list(encoded)

    def _looks_like_binary_payload(self, data: bytes) -> bool:
        if not data:
            return False
        sample = data[:8192]
        if b"\x00" in sample:
            return True
        nontext = sum(1 for byte in sample if byte < 9 or (13 < byte < 32) or byte == 127)
        return (nontext / max(len(sample), 1)) > 0.30

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

    def _select_target_binary(self, build: BuildArtifact, context: PocContext, payload_filename: str = "") -> str:
        default = self._default_target_binary(build, context, payload_filename=payload_filename)
        preferred = self._prefer_ossfuzz_harness_binary(
            default_binary=default,
            payload_filename=payload_filename,
            build=build,
            repo_url=context.repo_url,
        )
        return preferred

    def _default_target_binary(self, build: BuildArtifact, context: PocContext, payload_filename: str = "") -> str:
        if build.binary_or_entrypoint:
            binary = self._normalize_target_binary(build.binary_or_entrypoint, context.repo_url)
            binary = self._correct_nested_src_binary_path(binary, context.repo_url)
            return self._rewrite_non_executable_or_qt_plugin_target(
                binary, payload_filename, context.repo_url
            )
        if build.expected_binary_path:
            binary = self._normalize_target_binary(build.expected_binary_path, context.repo_url)
            binary = self._correct_nested_src_binary_path(binary, context.repo_url)
            return self._rewrite_non_executable_or_qt_plugin_target(
                binary, payload_filename, context.repo_url
            )
        interpreter = self._interpreter_for_payload(payload_filename)
        if interpreter:
            return interpreter
        usable = [
            item
            for item in (context.candidate_entrypoints or [])
            if item and not self._looks_like_non_executable_source(item)
        ]
        inferred = self._infer_qt_kimg_plugin_path(
            payload_filename, context.repo_url, " ".join(usable)
        )
        if inferred:
            return inferred
        if usable:
            binary = self._normalize_target_binary(usable[0], context.repo_url)
            return self._correct_nested_src_binary_path(binary, context.repo_url)
        # Gated: never invent ./target when an in-tree OSS-Fuzz harness is evidenced.
        harness = ossfuzz_tools.parse_ossfuzz_harness_name(payload_filename)
        repo_path = Path(build.repo_local_path) if build.repo_local_path else None
        if harness and repo_path is not None and repo_path.is_dir():
            if ossfuzz_tools.harness_source_evidence(repo_path, harness):
                rel = ossfuzz_tools.preferred_harness_relpath(repo_path, harness) or harness
                binary = self._normalize_target_binary(rel, context.repo_url)
                return self._correct_nested_src_binary_path(binary, context.repo_url)
        return "./target"

    def _prefer_ossfuzz_harness_binary(
        self,
        *,
        default_binary: str,
        payload_filename: str,
        build: BuildArtifact,
        repo_url: str,
    ) -> str:
        """Override target only when OSS-Fuzz harness name parses and in-tree evidence exists."""

        harness = ossfuzz_tools.parse_ossfuzz_harness_name(payload_filename)
        if not harness:
            return default_binary
        if Path(default_binary or "").name == harness:
            return default_binary

        repo_path = Path(build.repo_local_path) if build.repo_local_path else None
        if repo_path is None or not repo_path.is_dir():
            return default_binary
        if not ossfuzz_tools.harness_source_evidence(repo_path, harness):
            return default_binary
        rel = ossfuzz_tools.preferred_harness_relpath(repo_path, harness)
        if not rel:
            return default_binary
        binary = self._normalize_target_binary(rel, repo_url)
        return self._correct_nested_src_binary_path(binary, repo_url)

    def _select_target_args(
        self,
        knowledge: KnowledgeModel,
        payload_filename: str,
        context: PocContext,
        target_binary: str,
        recipe_run_command: str = "",
    ) -> list[str]:
        payload_path = f"/workspace/artifacts/poc/payloads/{payload_filename}"
        # OSS-Fuzz / libFuzzer harnesses take the reproducer as argv, not shell stdin.
        if self._looks_like_ossfuzz_harness(target_binary, payload_filename):
            return [payload_path]
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

    def _infer_trigger_mode(self, payload_filename: str, context: PocContext, target_binary: str = "") -> str:
        suffix = Path(payload_filename).suffix.lower()
        if suffix in {".sh", ".py", ".pl"}:
            return "script-driver"
        if self._looks_like_ossfuzz_harness(target_binary, payload_filename):
            return "cli-file"
        if "stdin" in context.inferred_input_modes:
            return "cli-stdin"
        if "argv" in context.inferred_input_modes:
            return "cli-argv"
        return "cli-file"

    def _looks_like_ossfuzz_harness(self, target_binary: str = "", payload_filename: str = "") -> bool:
        """Gate: ClusterFuzz/libFuzzer-style targets must receive the file as argv."""

        binary_name = Path((target_binary or "").replace("\\", "/")).name.lower()
        payload_name = Path((payload_filename or "").replace("\\", "/")).name.lower()
        if "fuzzer" in binary_name or binary_name.endswith("_fuzz"):
            return True
        if payload_name.startswith("clusterfuzz-testcase") or "ossfuzz" in payload_name:
            return True
        return False

    def _coerce_ossfuzz_harness_file_argv(self, plan: PocPlan) -> PocPlan:
        """Rewrite stdin-style ``< path`` args for fuzzer harnesses into file argv.

        ``_shell_quote('< /path')`` turns the redirect into a literal filename and
        the harness fails with ``open < /path: No such file``.
        """

        if not self._looks_like_ossfuzz_harness(plan.target_binary, plan.payload_filename):
            # Still fix quoted stdin redirects in run_command for non-fuzzers via
            # _build_run_command when we rebuild; leave plan unless args look broken.
            return self._rebuild_run_command_if_stdin_redirect_quoted(plan)

        payload_path = f"/workspace/artifacts/poc/payloads/{plan.payload_filename}"
        cleaned: list[str] = []
        for arg in plan.target_args or []:
            text = (arg or "").strip().strip("'\"")
            match = re.match(r"^<\s*(.+)$", text)
            if match:
                path = match.group(1).strip().strip("'\"")
                cleaned.append(path or payload_path)
                continue
            cleaned.append(arg)
        if not cleaned:
            cleaned = [payload_path]
        elif not any(
            payload_path == item
            or item.endswith(f"/{plan.payload_filename}")
            or Path(item).name == plan.payload_filename
            for item in cleaned
        ):
            cleaned = [payload_path] + cleaned

        plan.target_args = cleaned
        if plan.trigger_mode == "cli-stdin":
            plan.trigger_mode = "cli-file"
        plan.run_command = self._build_run_command(plan.target_binary, plan.target_args)
        return plan

    def _rebuild_run_command_if_stdin_redirect_quoted(self, plan: PocPlan) -> PocPlan:
        """Unquote accidental ``'< /path'`` tokens that break real shell redirects."""

        command = plan.run_command or ""
        if "'<" not in command and '"<' not in command:
            # Also rebuild when target_args still encode a redirect token.
            if any(re.match(r"^<\s*\S", (arg or "").strip()) for arg in (plan.target_args or [])):
                plan.run_command = self._build_run_command(plan.target_binary, plan.target_args)
            return plan
        plan.run_command = self._build_run_command(plan.target_binary, plan.target_args)
        return plan

    def _build_run_command(self, target_binary: str, target_args: list[str]) -> str:
        segments = [self._shell_quote(target_binary)]
        for item in target_args or []:
            text = (item or "").strip()
            # Keep shell redirect operators out of quoted argv tokens.
            match = re.match(r"^<\s*(.+)$", text)
            if match:
                path = match.group(1).strip().strip("'\"")
                segments.append("<")
                segments.append(self._shell_quote(path))
                continue
            segments.append(self._shell_quote(item))
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
            normalized[safe_name] = self._maybe_decode_base64_payload(content)
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

    def _strip_pre_run_make_rebuild(self, run_command: str) -> str:
        """Drop `make` segments chained before the actual trigger.

        Recipes often prepend `make -C <subdir>` before running the binary. That
        rebuilds the default `all` target (man pages / xmlto) without the build
        stage's ASan flags and fails before the sanitizer trigger can run.
        Prefer the already-built binary from the build stage.
        """
        stripped = (run_command or "").strip()
        if not stripped or "&&" not in stripped:
            return run_command

        parts = [part.strip() for part in re.split(r"\s*&&\s*", stripped) if part.strip()]
        if len(parts) < 2:
            return run_command

        def _is_make_rebuild(part: str) -> bool:
            remainder = part.strip()
            while True:
                match = re.match(
                    r"""^[A-Za-z_][A-Za-z0-9_]*=(?:\"[^\"]*\"|'[^']*'|[^\s]+)\s+""",
                    remainder,
                )
                if not match:
                    break
                remainder = remainder[match.end() :]
            return bool(re.match(r"^(?:g?make)\b", remainder))

        kept = [part for part in parts if not _is_make_rebuild(part)]
        if not kept or kept == parts:
            return run_command
        return " && ".join(kept)

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

    def _correct_nested_src_binary_path(self, target_binary: str, repo_url: str) -> str:
        """Collapse mistaken `/src/<proj>/src/<proj>` paths to `/src/<proj>/<proj>`."""

        if not target_binary or not repo_url:
            return target_binary
        project_dir = self._container_project_dir(repo_url)
        project_name = self._derive_project_name(repo_url)
        nested = f"{project_dir}/src/{project_name}"
        if target_binary == nested:
            return f"{project_dir}/{project_name}"
        return target_binary

    def _looks_like_shared_library(self, path: str) -> bool:
        lowered = (path or "").lower().replace("\\", "/")
        return bool(re.search(r"\.(so|dylib|dll)(\b|$)", lowered))

    _NON_EXECUTABLE_NAMES: frozenset[str] = frozenset(
        {
            "authors",
            "copying",
            "license",
            "licence",
            "changelog",
            "changes",
            "news",
            "install",
            "readme",
            "todo",
            "credits",
            "maintainers",
            "cmakelists.txt",
            "makefile",
            "makefile.am",
            "makefile.in",
        }
    )
    _NON_EXECUTABLE_SUFFIXES: frozenset[str] = frozenset(
        {".md", ".rst", ".txt", ".html", ".in", ".cmake", ".h", ".hpp", ".c", ".cc", ".cpp", ".cxx"}
    )
    _QT_KIMG_FORMATS: frozenset[str] = frozenset(
        {"xcf", "psd", "tga", "pcx", "ras", "rgb", "pic", "hdr", "ora", "kra", "eps", "exr"}
    )

    def _looks_like_non_executable_source(self, path: str) -> bool:
        """True for docs/source files that must never be executed as a PoC target."""

        name = Path((path or "").replace("\\", "/")).name.lower()
        if not name:
            return False
        stem = name.split(".", 1)[0]
        if name in self._NON_EXECUTABLE_NAMES or stem in self._NON_EXECUTABLE_NAMES:
            return True
        if name.startswith("readme") or name.startswith("copying") or name.startswith("license"):
            return True
        suffix = Path(name).suffix.lower()
        return suffix in self._NON_EXECUTABLE_SUFFIXES

    def _looks_like_imageformats_source_file(self, path: str) -> bool:
        cleaned = (path or "").replace("\\", "/").lower()
        if "/imageformats/" not in cleaned:
            return False
        return not self._looks_like_shared_library(cleaned)

    def _infer_qt_kimg_plugin_path(
        self,
        payload_filename: str,
        repo_url: str = "",
        current_target: str = "",
    ) -> str:
        """Map an image payload suffix to build/bin/imageformats/kimg_<fmt>.so when gated."""

        suffix = Path(payload_filename or "").suffix.lower().lstrip(".")
        if suffix not in self._QT_KIMG_FORMATS:
            return ""
        marker = f"{current_target} {repo_url}".lower().replace("\\", "/")
        if not any(token in marker for token in ("imageformats", "kimageformats", "kimg_")):
            return ""
        project_dir = ""
        if repo_url:
            project_dir = self._container_project_dir(repo_url)
        if not project_dir:
            match = re.search(r"(/src/[^/]+)/", (current_target or "").replace("\\", "/"))
            project_dir = match.group(1) if match else ""
        if not project_dir:
            return ""
        return f"{project_dir}/build/bin/imageformats/kimg_{suffix}.so"

    def _rewrite_non_executable_or_qt_plugin_target(
        self,
        target_binary: str,
        payload_filename: str,
        repo_url: str = "",
    ) -> str:
        """Replace docs/source-tree files with the Qt image plugin inferred from the payload."""

        inferred = self._infer_qt_kimg_plugin_path(payload_filename, repo_url, target_binary)
        if not inferred:
            return target_binary
        if (
            not target_binary
            or self._looks_like_non_executable_source(target_binary)
            or self._looks_like_imageformats_source_file(target_binary)
        ):
            return inferred
        return target_binary

    def _correct_qt_plugin_binary_path(self, target_binary: str) -> str:
        """Rewrite mistaken Qt image-plugin paths to build/bin/imageformats/.

        Common LLM mistakes:
        - build/src/imageformats/kimg_*.so  (source tree, not install/build output)
        - build/bin/kimg_*.so               (missing imageformats/ subdirectory)
        """

        if not target_binary:
            return target_binary
        cleaned = target_binary.replace("\\", "/")
        cleaned = re.sub(
            r"(/build)/src/imageformats/(kimg_[^/]+\.so)\b",
            r"\1/bin/imageformats/\2",
            cleaned,
        )
        # Only rewrite bare build/bin/kimg_*.so — not paths that already include imageformats/.
        cleaned = re.sub(
            r"(/build/bin)/(?!imageformats/)(kimg_[^/]+\.so)\b",
            r"\1/imageformats/\2",
            cleaned,
        )
        return cleaned

    def _prefer_compact_dataset_xcf_payload(
        self,
        plan: PocPlan,
        dataset_poc_filenames: Optional[list[str]] = None,
        dataset_poc_base64_blobs: Optional[list[str]] = None,
    ) -> PocPlan:
        """Prefer small crafted XCF PoCs over huge ClusterFuzz corpora when both exist.

        CVE-2021-36083-style XCF overflows are reproduced by crafted_bpp*.xcf; the
        large minimized ClusterFuzz seed often only yields ASan DEADLYSIGNAL storms
        without a matchable stack-buffer-overflow report.
        """

        filenames = list(dataset_poc_filenames or [])
        blobs = list(dataset_poc_base64_blobs or [])
        if not filenames or not blobs or len(filenames) != len(blobs):
            return plan

        preferred_idx = None
        for idx, name in enumerate(filenames):
            if re.search(r"crafted_bpp\d*\.xcf$", name.lower()):
                preferred_idx = idx
                break
        if preferred_idx is None:
            return plan

        current = (plan.payload_filename or "").lower()
        if re.search(r"crafted_bpp\d*\.xcf$", current):
            return plan

        current_size = len((plan.payload_content or "").encode("latin-1", errors="replace"))
        should_switch = (
            "clusterfuzz" in current
            or "testcase-minimized" in current
            or current_size >= 4096
            or not current.endswith(".xcf")
            or current in {"poc.txt", "poc.bin", "trigger", "payload"}
        )
        if not should_switch:
            return plan

        name = filenames[preferred_idx]
        raw = base64.b64decode(blobs[preferred_idx])
        plan.payload_filename = Path(name).name
        plan.payload_content = raw.decode("latin-1", errors="replace")
        if not plan.source_of_truth or plan.source_of_truth in {"llm_synthesized", "dataset_poc"}:
            plan.source_of_truth = "dataset_poc"
        return plan

    def _prefer_ossfuzz_minimized_payload(
        self,
        plan: PocPlan,
        dataset_poc_filenames: Optional[list[str]] = None,
        dataset_poc_base64_blobs: Optional[list[str]] = None,
    ) -> PocPlan:
        """Prefer OSS-Fuzz minimized seeds over tiny harvested poc.* files.

        GitHub commit pages often embed a short poc.cil that only hits a null
        SEGV. The ClusterFuzz minimized testcase is the actual sanitizer
        reproducer. Do not override crafted_bpp*.xcf.
        """

        filenames = list(dataset_poc_filenames or [])
        blobs = list(dataset_poc_base64_blobs or [])
        if not filenames or not blobs or len(filenames) != len(blobs):
            return plan
        if any(re.search(r"crafted_bpp\d*\.xcf$", name.lower()) for name in filenames):
            return plan

        preferred_idx = None
        for idx, name in enumerate(filenames):
            lowered = name.lower()
            if "clusterfuzz" in lowered or "testcase-minimized" in lowered:
                preferred_idx = idx
                break
        if preferred_idx is None:
            return plan

        current = Path(plan.payload_filename or "").name.lower()
        if "clusterfuzz" in current or "testcase-minimized" in current:
            return plan
        generic_names = {"poc.cil", "poc.txt", "poc.bin", "poc", "trigger", "payload", "poc.c"}
        if current not in generic_names and not re.fullmatch(r"poc\.[a-z0-9]+", current):
            return plan

        name = filenames[preferred_idx]
        raw = base64.b64decode(blobs[preferred_idx])
        plan.payload_filename = Path(name).name
        plan.payload_content = raw.decode("latin-1", errors="replace")
        if not plan.source_of_truth or plan.source_of_truth in {"llm_synthesized", "dataset_poc"}:
            plan.source_of_truth = "dataset_poc"
        return plan

    def _sync_payload_filename_into_command(self, plan: PocPlan) -> PocPlan:
        filename = Path(plan.payload_filename or "").name
        if not filename:
            return plan
        payload_path = f"/workspace/artifacts/poc/payloads/{filename}"
        plan.target_args = [
            re.sub(r"/workspace/artifacts/poc/payloads/[^/\s'\"]+", payload_path, arg)
            if "/workspace/artifacts/poc/payloads/" in arg
            else self._normalize_workspace_arg(arg, filename)
            for arg in (plan.target_args or [])
        ]
        if plan.run_command:
            plan.run_command = re.sub(
                r"/workspace/artifacts/poc/payloads/[^/\s'\"]+",
                payload_path,
                plan.run_command,
            )
            plan.run_command = self._normalize_run_command(plan.run_command, filename)
        return plan

    def _qt_plugin_search_path(self, plugin_so_path: str) -> str:
        cleaned = (plugin_so_path or "").replace("\\", "/")
        if "/imageformats/" in cleaned:
            return cleaned.rsplit("/imageformats/", 1)[0]
        from pathlib import PurePosixPath

        return str(PurePosixPath(cleaned).parent) if cleaned else "/tmp"

    def _image_format_hint(self, payload_filename: str, plugin_so_path: str) -> str:
        suffix = Path(payload_filename or "").suffix.lower().lstrip(".")
        if suffix:
            return suffix
        match = re.search(r"kimg_([a-z0-9]+)\.so\b", (plugin_so_path or "").lower())
        if match:
            return match.group(1)
        return ""

    def _qimage_harness_source(self) -> str:
        return """#include <cstdio>
#include <cstdlib>

#include <QCoreApplication>
#include <QImage>
#include <QImageReader>
#include <QString>

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <image> [format]\\n", argv[0]);
        return 2;
    }

    QCoreApplication app(argc, argv);
    if (const char *plugin_path = std::getenv("QT_PLUGIN_PATH")) {
        if (*plugin_path) {
            QCoreApplication::addLibraryPath(QString::fromLocal8Bit(plugin_path));
        }
    }

    QImageReader reader(QString::fromLocal8Bit(argv[1]));
    if (argc >= 3 && argv[2] && *argv[2]) {
        reader.setFormat(argv[2]);
    }

    const QImage image = reader.read();
    if (image.isNull()) {
        std::fprintf(stderr, "read failed: %s\\n", qPrintable(reader.errorString()));
        return 1;
    }

    std::fprintf(stderr, "read ok: %dx%d\\n", image.width(), image.height());
    return 0;
}
"""

    def _ensure_shared_library_harness(self, plan: PocPlan) -> PocPlan:
        """Rewrite plans that try to execute a shared library/plugin into a loader harness.

        Qt image plugins (kimg_*.so) and other .so targets are not executables. Directly
        invoking them yields exit 127 and never reaches the vulnerable parser path.
        """

        plugin_path = plan.target_binary or ""
        if not self._looks_like_shared_library(plugin_path):
            match = re.search(r"(/src/[^\s'\"]+\.so)\b", plan.run_command or "")
            if not match:
                match = re.search(r"((?:build/)?[^\s'\"]+\.so)\b", plan.run_command or "")
            if not match:
                return plan
            plugin_path = match.group(1)
            if not plugin_path.startswith("/"):
                # Best-effort: leave relative; normalize later only if already absolute.
                pass
            plugin_path = self._correct_qt_plugin_binary_path(plugin_path)
            plan.target_binary = plugin_path

        if not self._looks_like_shared_library(plan.target_binary):
            return plan

        plan.target_binary = self._correct_qt_plugin_binary_path(plan.target_binary)
        plugin_path = plan.target_binary
        payload_path = f"/workspace/artifacts/poc/payloads/{plan.payload_filename}"
        fmt = self._image_format_hint(plan.payload_filename, plugin_path)
        plugin_root = self._qt_plugin_search_path(plugin_path)

        harness_rel = "inputs/qimage_harness.cpp"
        auxiliaries = dict(plan.auxiliary_files or {})
        auxiliaries[harness_rel] = self._qimage_harness_source()
        plan.auxiliary_files = auxiliaries

        harness_src = f"/workspace/artifacts/poc/{harness_rel}"
        harness_bin = "/tmp/qimage_harness"
        # Gate: library-harness loader stays unsanitized. The Qt plugin is already
        # built with -shared-libasan; compiling the tiny QImage loader the same
        # way double-inits ASan across unsanitized system Qt and yields SIGILL /
        # nested DEADLYSIGNAL. Preload the runtime onto the unsanitized loader
        # so dlopen(plugin) still sees ASan first (LLVM's DSO-only model).
        # Do not export LD_PRELOAD around clang++/timeout.
        compile_cmd = (
            "clang++ -g -O0 -fno-omit-frame-pointer "
            f"{self._shell_quote(harness_src)} -o {self._shell_quote(harness_bin)} "
            "$(pkg-config --cflags --libs Qt5Gui Qt5Core)"
        )
        asan_options = (
            "abort_on_error=1:halt_on_error=1:symbolize=1:detect_leaks=0:"
            "alloc_dealloc_mismatch=0:fast_unwind_on_fatal=0"
        )
        asan_rt_setup = (
            "_ASAN_RT=; "
            "for _asan_dir in /usr/lib/llvm-*/lib/clang/*/lib/linux /usr/lib/clang/*/lib/linux; do "
            'if [[ -f "${_asan_dir}/libclang_rt.asan-x86_64.so" ]]; then '
            'export LD_LIBRARY_PATH="${_asan_dir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; '
            '_ASAN_RT="${_asan_dir}/libclang_rt.asan-x86_64.so"; '
            "break; "
            "fi; "
            "done"
        )
        fmt_arg = f" {self._shell_quote(fmt)}" if fmt else ""
        plan.trigger_mode = "library-harness"
        plan.run_command = (
            f"export QT_PLUGIN_PATH={self._shell_quote(plugin_root)} "
            f"ASAN_OPTIONS={self._shell_quote(asan_options)}; "
            f"{asan_rt_setup}; "
            f"{compile_cmd} && "
            f"timeout 90s env LD_PRELOAD="
            '"${_ASAN_RT}" '
            f"{self._shell_quote(harness_bin)} "
            f"{self._shell_quote(payload_path)}{fmt_arg}"
        )
        env = dict(plan.environment_variables or {})
        env["QT_PLUGIN_PATH"] = plugin_root
        env["ASAN_OPTIONS"] = asan_options
        plan.environment_variables = env
        note = "Normalized shared-library/plugin target into QImage harness (do not exec .so)."
        if note not in (plan.rationale or ""):
            plan.rationale = f"{(plan.rationale or '').rstrip()} {note}".strip()
        return plan

    def _rewrite_nested_src_binary_in_command(self, run_command: str, repo_url: str) -> str:
        """Rewrite mistaken nested `src/<proj>` binary references in run commands."""

        if not run_command or not repo_url:
            return run_command
        project_name = self._derive_project_name(repo_url)
        project_dir = self._container_project_dir(repo_url)
        corrected = f"{project_dir}/{project_name}"
        patterns = [
            rf"(?<!\S){re.escape(project_dir)}/src/{re.escape(project_name)}(?!\S)",
            rf"(?<!\S)\./src/{re.escape(project_name)}(?!\S)",
            rf"(?<!\S)src/{re.escape(project_name)}(?!\S)",
        ]
        rewritten = run_command
        for pattern in patterns:
            rewritten = re.sub(pattern, corrected, rewritten)
        return rewritten

    def _maybe_decode_base64_payload(self, content: str) -> str:
        """Decode single-blob base64 payloads commonly copied from reproduction recipes."""

        decoded = self._decode_base64_blob(content)
        if decoded is None:
            return content
        return decoded if decoded.endswith("\n") else f"{decoded}\n"

    def _unwrap_nested_base64_payload(self, content: str) -> str:
        """Unwrap a second base64 layer when the file stores recipe ``base64 -d`` input.

        Transport decoding of ``dataset_poc_base64_blobs`` yields the on-disk
        file bytes. Some harvested PoCs keep those bytes as ASCII base64 of a
        script (``echo -n '...' | base64 -d > poc``). Only unwrap when the
        inner decode looks like source (script markers); binary seeds and
        already-decoded scripts are unchanged.
        """

        decoded = self._decode_base64_blob(content)
        if decoded is None:
            return content
        markers = (
            "function",
            "local ",
            "#!/",
            "import ",
            "#include",
            "print(",
            "return ",
            "<const>",
        )
        if not any(marker in decoded for marker in markers):
            return content
        return decoded if decoded.endswith("\n") else f"{decoded}\n"

    def _resolve_payload_content(self, content: str, recipe_base64_blobs: list[str]) -> str:
        """Prefer recipe/dataset base64 bytes when LLM text mangled or invented content."""

        recipe_decoded: Optional[str] = None
        for blob in recipe_base64_blobs:
            recipe_decoded = self._decode_base64_blob(blob)
            if recipe_decoded is not None:
                break

        content_decoded = self._decode_base64_blob(content)
        if content_decoded is not None:
            # Content itself is still base64 — decode it, but recipe wins on mismatch.
            if recipe_decoded is not None and self._should_prefer_recipe_payload(content_decoded, recipe_decoded):
                chosen = recipe_decoded
            else:
                chosen = content_decoded
        elif recipe_decoded is not None and self._should_prefer_recipe_payload(content, recipe_decoded):
            chosen = recipe_decoded
        else:
            chosen = content

        if recipe_decoded is not None and (
            not (chosen or "").strip()
            or chosen.strip() in {"trigger", "poc", "payload"}
            or len(chosen) > max(self.MAX_SYNTHESIZED_PAYLOAD_CHARS, len(recipe_decoded) * 2)
        ):
            chosen = recipe_decoded

        if recipe_decoded is None and len(chosen or "") > self.MAX_SYNTHESIZED_PAYLOAD_CHARS:
            chosen = chosen[: self.MAX_SYNTHESIZED_PAYLOAD_CHARS]

        chosen = chosen if chosen.endswith("\n") else f"{chosen}\n"
        # Same nested unwrap as authoritative dataset_poc: file/recipe may still
        # be the ``echo | base64 -d`` input rather than the final script bytes.
        return self._unwrap_nested_base64_payload(chosen)

    def _payload_matches_blob(self, content: str, blobs: list[str]) -> bool:
        decoded_content = content[:-1] if content.endswith("\n") else content
        for blob in blobs:
            recipe_decoded = self._decode_base64_blob(blob)
            if recipe_decoded is None:
                continue
            if decoded_content == recipe_decoded or decoded_content == recipe_decoded.rstrip("\n"):
                return True
        return False

    def _decode_base64_blob(self, content: str) -> Optional[str]:
        """Return latin-1 text for a pure base64 blob, else None."""

        raw = content or ""
        candidate = re.sub(r"\s+", "", raw.strip())
        if len(candidate) < 16 or len(candidate) % 4 != 0:
            return None
        if not re.fullmatch(r"[A-Za-z0-9+/]+=*", candidate):
            return None
        try:
            decoded = base64.b64decode(candidate, validate=True)
        except binascii.Error:
            return None
        text = decoded.decode("latin-1")
        markers = ("function", "local ", "#!/", "import ", "#include", "print(", "return ", "<const>")
        if not any(marker in text for marker in markers) and "\n" not in text:
            # Avoid treating short random tokens as payloads.
            # Binary PoCs often lack newlines/markers — allow larger blobs.
            if len(candidate) < 48:
                return None
        return text

    def _should_prefer_recipe_payload(self, content: str, recipe_decoded: str) -> bool:
        """True when content is missing control bytes present in the recipe decode."""

        if not content.strip() or content.strip() in {"trigger", "poc", "payload"}:
            return True
        if len(content) > max(self.MAX_SYNTHESIZED_PAYLOAD_CHARS, len(recipe_decoded) * 2):
            return True
        if self._decode_base64_blob(content) is not None:
            # Caller already compared decoded forms; prefer recipe if they diverge on controls.
            pass

        def control_bytes(text: str) -> set[str]:
            return {ch for ch in text if ord(ch) < 32 and ch not in "\n\t\r"}

        recipe_controls = control_bytes(recipe_decoded)
        content_controls = control_bytes(content)
        if recipe_controls - content_controls:
            printable_content = "".join(ch for ch in content if ch.isprintable() or ch in "\n\t")
            printable_recipe = "".join(ch for ch in recipe_decoded if ch.isprintable() or ch in "\n\t")
            # Same skeleton, or LLM introduced backslash-letter stand-ins for controls.
            if (
                printable_content.replace(" ", "") == printable_recipe.replace(" ", "")
                or printable_recipe[:40] in printable_content
                or printable_content[:40] in printable_recipe
                or re.search(r"(?<!\\)\\[a-zA-Z]", content) is not None
            ):
                return True
        recipe_is_binary = any(ord(ch) < 32 and ch not in "\n\t\r" for ch in recipe_decoded) or "\x00" in recipe_decoded
        if recipe_is_binary and abs(len(content) - len(recipe_decoded)) > max(64, len(recipe_decoded) // 4):
            return True
        return False

    def _extract_recipe_base64_blobs(self, recipes: list[ReproductionRecipe]) -> list[str]:
        """Pull base64 blobs out of recipe steps / artifact commands / excerpts."""

        blob_re = re.compile(r"(?<![A-Za-z0-9+/])([A-Za-z0-9+/]{24,}={0,2})(?![A-Za-z0-9+/])")
        blobs: list[str] = []
        for recipe in recipes:
            texts: list[str] = []
            texts.extend(recipe.steps or [])
            texts.extend(recipe.artifact_generation_commands or [])
            texts.append(recipe.source_excerpt or "")
            for text in texts:
                for match in blob_re.finditer(text or ""):
                    candidate = match.group(1)
                    if self._decode_base64_blob(candidate) is None:
                        continue
                    if candidate not in blobs:
                        blobs.append(candidate)
        return blobs

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
        # Container paths are always POSIX; Path() on Windows would rewrite
        # "/src/lua/lua" into "\src\lua" and break Docker execution.
        from pathlib import PurePosixPath

        target = (target_binary or "").strip()
        if target.startswith("/"):
            parent = str(PurePosixPath(target).parent)
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

        signal_exit_observed = exit_code_observed in {134, 139}

        expected_signal_texts = list(plan.expected_stderr_patterns or [])
        if plan.expected_crash_type:
            expected_signal_texts.append(plan.expected_crash_type)
        specific_error_hits = filter_hits_for_specific_sanitizer(
            error_pattern_hits, expected_signal_texts
        )

        # 3.9 eligible_for_verify
        eligible_for_verify = False
        eligibility_reason = ""
        if not script_finished:
            eligibility_reason = "script_did_not_finish: missing execution_exit_code marker"
        elif not log_well_formed:
            eligibility_reason = "log_not_well_formed: stdout/stderr block markers missing"
        else:
            # Priority: stderr > stdout > stack > crash_type > exit_code
            haystack = "\n".join(
                [
                    str(observation.get("observed_stderr") or ""),
                    str(observation.get("observed_stdout") or ""),
                    execution_logs or "",
                ]
            )
            if specific_error_hits:
                eligible_for_verify = True
                eligibility_reason = f"error_pattern_hit: {specific_error_hits[0]}"
            elif haystack_has_specific_sanitizer_bug(haystack, expected_signal_texts):
                eligible_for_verify = True
                eligibility_reason = "specific_sanitizer_bug_in_haystack"
            elif stdout_pattern_hits:
                eligible_for_verify = True
                eligibility_reason = f"stdout_pattern_hit: {stdout_pattern_hits[0]}"
            elif stack_keyword_hits and not specific_sanitizer_bugs_in(expected_signal_texts):
                # Function names in library DIAG traces must not stand in for a
                # concrete sanitizer report when the plan asked for one.
                eligible_for_verify = True
                eligibility_reason = f"stack_keyword_hit: {stack_keyword_hits[0]}"
            elif crash_type_compatible is True:
                eligible_for_verify = True
                eligibility_reason = f"crash_type_compatible: observed={crash_type_hit}"
            elif exit_code_match_expected is True:
                eligible_for_verify = True
                eligibility_reason = f"exit_code_match: {exit_code_observed}"
            elif signal_exit_observed:
                eligible_for_verify = True
                eligibility_reason = f"signal_exit_observed: {exit_code_observed}"
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
        lowered = (execution_logs or "").lower()
        payload_invalid_markers = (
            "invalid syntax",
            "failed to compile",
            "syntax error",
            "parse error",
            "parsing error",
            "bad classpermission",
            "unexpected token",
            "unexpected end",
            "unknown keyword",
            "not a valid",
            "malformed",
            "yaml: ",
            "json.decode",
            "json.decoder",
            "expaterror",
            "xml.parsers",
        )
        if any(marker in lowered for marker in payload_invalid_markers):
            return "payload_invalid"
        return "non_triggering"

    def _shell_quote(self, value: str) -> str:
        return "'" + value.replace("'", "'\"'\"'") + "'"

    def _escape_for_echo(self, value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")


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
