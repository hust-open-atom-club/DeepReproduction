"""文件说明：环境构建阶段实现。

这个模块负责把知识阶段的候选线索收敛为“确认后的构建事实”。
它会先 clone 仓库并读取真实源码中的构建文件、README、CI 配置和 patch，
再把这些本地证据交给模型规划构建方案；若模型不可用，则回退到规则规划。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TypedDict

import yaml
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal envs.
    Environment = None
    FileSystemLoader = None
    StrictUndefined = None

from app.config import build_chat_model, load_app_config
from app.schemas.build_artifact import BuildArtifact
from app.schemas.knowledge import KnowledgeModel
from app.tools.docker_tools import DockerBuildRequest, DockerCommandResult, DockerRunRequest, DockerTool
from app.tools.file_tools import FileTool
from app.tools.git_tools import GitTool
from app.tools import ossfuzz as ossfuzz_tools
from app.tools.patch_tools import find_patch_diff, strip_unapplyable_binary_stub_hunks
from app.tools.process_tools import ProcessRequest, ProcessTool


class BuildStagePaths:
    """Filesystem layout owned by the build stage."""

    def __init__(self, workspace: str) -> None:
        self.workspace_root = Path(workspace)
        self.repo_dir = self.workspace_root / "repo"
        self.artifacts_dir = self.workspace_root / "artifacts"
        self.build_dir = self.artifacts_dir / "build"
        self.poc_dir = self.artifacts_dir / "poc"
        self.verify_dir = self.artifacts_dir / "verify"
        self.dockerfile = self.build_dir / "Dockerfile"
        self.build_script = self.build_dir / "build.sh"
        self.build_log = self.build_dir / "build.log"
        self.build_plan_yaml = self.build_dir / "build_plan.yaml"
        self.build_context_yaml = self.build_dir / "build_context.yaml"
        self.build_artifact_yaml = self.build_dir / "build_artifact.yaml"
        self.build_verify_yaml = self.build_dir / "build_verify.yaml"
        self.llm_dir = self.build_dir / "llm"


class RefSnapshot(BaseModel):
    """Repository snapshot collected from one candidate ref."""

    label: str = Field(..., description="Human-readable label such as vulnerable_ref or fixed_parent.")
    requested_ref: str = Field(..., description="Requested git ref string.")
    resolved_ref: str = Field(default="", description="Resolved commit SHA.")
    build_files: list[str] = Field(default_factory=list, description="Detected build files.")
    evidence_files: list[str] = Field(default_factory=list, description="Detected README or INSTALL files.")
    ci_files: list[str] = Field(default_factory=list, description="Detected CI configuration files.")
    file_excerpts: list[str] = Field(default_factory=list, description="Short excerpts from key files at this ref.")


class BuildPlan(BaseModel):
    """Structured build plan produced by rules or the build LLM."""

    chosen_vulnerable_ref: str = Field(..., description="Chosen vulnerable ref to build.")
    chosen_fixed_ref: Optional[str] = Field(default=None, description="Chosen fixed ref used for comparison.")
    build_system: str = Field(default="unknown", description="Chosen build system.")
    install_packages: list[str] = Field(default_factory=list, description="System packages to install in Docker.")
    configure_commands: list[str] = Field(default_factory=list, description="Configure commands to run before building.")
    clean_commands: list[str] = Field(default_factory=list, description="Cleanup commands to run before building.")
    build_commands: list[str] = Field(default_factory=list, description="Main build commands.")
    expected_binary_path: Optional[str] = Field(default=None, description="Expected output binary or entrypoint path.")
    base_image: Optional[str] = Field(
        default=None,
        description="Docker base image selected for dependency compatibility (for example ubuntu:22.04).",
    )
    dockerfile_override: Optional[str] = Field(default=None, description="Optional full Dockerfile override.")
    build_script_override: Optional[str] = Field(default=None, description="Optional full build script override.")
    source_of_truth: str = Field(default="manual_fallback", description="Primary evidence source behind the plan.")
    confidence: str = Field(default="medium", description="Planner confidence level.")
    rationale: str = Field(default="", description="Short explanation for the decision.")


class BuildContext(BaseModel):
    """Collected local evidence consumed by the planner."""

    cve_id: str = Field(..., description="Target CVE identifier.")
    repo_url: str = Field(default="", description="Repository URL.")
    task_vulnerable_ref: Optional[str] = Field(default=None, description="Task or knowledge vulnerable ref.")
    task_fixed_ref: Optional[str] = Field(default=None, description="Task or knowledge fixed ref.")
    patch_diff_excerpt: str = Field(default="", description="Short excerpt from patch.diff.")
    patch_affected_files: list[str] = Field(default_factory=list, description="Files touched by patch.diff.")
    knowledge_summary: str = Field(default="", description="Knowledge-stage summary.")
    knowledge_build_hints: list[str] = Field(default_factory=list, description="Build-related hints from knowledge.")
    knowledge_reproduction_hints: list[str] = Field(default_factory=list, description="Reproduction hints from knowledge.")
    snapshots: list[RefSnapshot] = Field(default_factory=list, description="Candidate ref snapshots.")
    planner_attempt: int = Field(default=1, description="Current planning attempt number.")
    previous_failure_kind: str = Field(default="", description="Failure kind such as docker_build or container_run.")
    previous_build_failure: str = Field(default="", description="Previous build failure logs for replanning.")
    previous_dockerfile_content: str = Field(default="", description="Previously rendered Dockerfile for replanning.")
    previous_build_script_content: str = Field(default="", description="Previously rendered build script for replanning.")


class BuildPreparedRun(BaseModel):
    """Deterministic inputs assembled before planning starts."""

    plan_meta: dict[str, Any]
    repo_path: str
    context: BuildContext


class BuildExecutionOutcome(BaseModel):
    """One concrete build attempt outcome."""

    plan: BuildPlan
    artifact: BuildArtifact


class BuildGraphState(TypedDict, total=False):
    """Internal LangGraph state for the build stage."""

    knowledge: KnowledgeModel
    paths: BuildStagePaths
    prepared: BuildPreparedRun
    current_context: BuildContext
    current_plan: BuildPlan
    outcome: BuildExecutionOutcome
    attempt: int
    should_retry: bool


class BuildFallbackSpec(BaseModel):
    """Deterministic fallback planning spec used when LLM planning is unavailable."""

    chosen_vulnerable_ref: str
    chosen_fixed_ref: Optional[str] = None
    build_system: str = "unknown"
    install_packages: list[str] = Field(default_factory=list)
    configure_commands: list[str] = Field(default_factory=list)
    clean_commands: list[str] = Field(default_factory=list)
    build_commands: list[str] = Field(default_factory=list)
    expected_binary_path: Optional[str] = None
    source_of_truth: str = "heuristic"
    confidence: str = "medium"
    rationale: str = ""


class BuildPlanner:
    """Encapsulates build-stage planning decisions."""

    def __init__(self, stage: "BuildStage") -> None:
        self.stage = stage

    def plan(self, knowledge: KnowledgeModel, context: BuildContext, project_name: str) -> BuildPlan:
        llm_plan = self.try_llm_plan(knowledge=knowledge, context=context, project_name=project_name)
        if llm_plan is not None:
            return llm_plan
        return self.heuristic_plan(knowledge=knowledge, context=context, project_name=project_name)

    def replan_after_failure(
        self,
        knowledge: KnowledgeModel,
        context: BuildContext,
        project_name: str,
        previous_plan: BuildPlan,
        previous_artifact: BuildArtifact,
    ) -> Optional[BuildPlan]:
        retry_context = context.model_copy(
            update={
                "planner_attempt": context.planner_attempt + 1,
                "previous_failure_kind": self.stage._classify_failure_kind(previous_artifact.build_logs),
                "previous_build_failure": previous_artifact.build_logs[:6000],
                "previous_dockerfile_content": previous_artifact.dockerfile_content[:8000],
                "previous_build_script_content": previous_artifact.build_script_content[:8000],
            }
        )
        return self.try_llm_plan(
            knowledge=knowledge,
            context=retry_context,
            project_name=project_name,
            previous_plan=previous_plan,
        )

    def try_llm_plan(
        self,
        knowledge: KnowledgeModel,
        context: BuildContext,
        project_name: str,
        previous_plan: Optional[BuildPlan] = None,
    ) -> Optional[BuildPlan]:
        try:
            model = build_chat_model(
                "build_agent",
                temperature=0,
                timeout_seconds=load_app_config().runtime.build_agent_timeout_seconds,
            )
        except Exception:
            return None

        prompt = self.build_llm_prompt(
            knowledge=knowledge,
            context=context,
            project_name=project_name,
            previous_plan=previous_plan,
        )
        self.stage._persist_build_llm_trace(context.planner_attempt, "prompt.txt", prompt)
        messages = self.build_llm_messages(
            knowledge=knowledge,
            context=context,
            project_name=project_name,
            previous_plan=previous_plan,
        )
        retry_errors: list[str] = []
        max_attempts = self.stage.MAX_LLM_NO_RESPONSE_RETRIES + 1
        for invoke_attempt in range(1, max_attempts + 1):
            try:
                response = model.invoke(messages)
                raw_response = getattr(response, "content", response)
                raw_text = raw_response if isinstance(raw_response, str) else str(raw_response)
                if self.stage._is_empty_llm_response(raw_text):
                    retry_errors.append(f"Attempt {invoke_attempt}: empty response")
                    if invoke_attempt < max_attempts:
                        continue
                    self.stage._persist_build_llm_trace(
                        context.planner_attempt,
                        "error.txt",
                        f"LLM returned no content after {max_attempts} attempts.",
                    )
                    return None
                self.stage._persist_build_llm_trace(context.planner_attempt, "response.txt", raw_text)
                parsed = parse_llm_json_payload(raw_response)
                if parsed is None:
                    self.stage._persist_build_llm_trace(
                        context.planner_attempt,
                        "error.txt",
                        "Failed to parse build-agent response as JSON.",
                    )
                    return None
                self.stage._persist_build_llm_trace(
                    context.planner_attempt,
                    "parsed.json",
                    json.dumps(parsed, ensure_ascii=False, indent=2),
                )
                plan = BuildPlan(**parsed)
                if not plan.build_commands:
                    self.stage._persist_build_llm_trace(
                        context.planner_attempt,
                        "error.txt",
                        "Parsed build-agent response did not include build_commands.",
                    )
                    return None
                if previous_plan is not None and not self.stage._is_valid_replan_candidate(
                    previous_plan,
                    plan,
                    failure_kind=context.previous_failure_kind,
                    failure_logs=context.previous_build_failure,
                ):
                    self.stage._persist_build_llm_trace(
                        context.planner_attempt,
                        "error.txt",
                        "Rejected build-agent replan because it did not produce a valid execution-surface override.",
                    )
                    return None
                return plan
            except Exception as error:
                error_text = str(error)
                retry_errors.append(f"Attempt {invoke_attempt}: {error_text}")
                if self.stage._should_retry_llm_request(error_text) and invoke_attempt < max_attempts:
                    continue
                self.stage._persist_build_llm_trace(
                    context.planner_attempt,
                    "error.txt",
                    "\n".join(retry_errors),
                )
                return None

        self.stage._persist_build_llm_trace(
            context.planner_attempt,
            "error.txt",
            "\n".join(retry_errors) or "LLM request failed without a response.",
        )
        return None

    def build_llm_prompt(
        self,
        knowledge: KnowledgeModel,
        context: BuildContext,
        project_name: str,
        previous_plan: Optional[BuildPlan],
    ) -> str:
        return self.stage._build_llm_prompt(
            knowledge=knowledge,
            context=context,
            project_name=project_name,
            previous_plan=previous_plan,
        )

    def build_llm_messages(
        self,
        knowledge: KnowledgeModel,
        context: BuildContext,
        project_name: str,
        previous_plan: Optional[BuildPlan],
    ) -> list[SystemMessage | HumanMessage | AIMessage]:
        system_message = SystemMessage(content="You return strict JSON only.")
        initial_prompt = self.stage._build_llm_prompt(
            knowledge=knowledge,
            context=context.model_copy(
                update={
                    "previous_failure_kind": "",
                    "previous_build_failure": "",
                    "previous_dockerfile_content": "",
                    "previous_build_script_content": "",
                }
            ),
            project_name=project_name,
            previous_plan=None,
        )
        if previous_plan is None:
            return [system_message, HumanMessage(content=initial_prompt)]

        retry_prompt = self.stage._build_llm_retry_prompt(
            context=context,
            previous_plan=previous_plan,
        )
        return [
            system_message,
            HumanMessage(content=initial_prompt),
            AIMessage(
                content=yaml.safe_dump(
                    previous_plan.model_dump(mode="json"),
                    sort_keys=False,
                    allow_unicode=True,
                )
            ),
            HumanMessage(content=retry_prompt),
        ]

    def heuristic_plan(self, knowledge: KnowledgeModel, context: BuildContext, project_name: str) -> BuildPlan:
        fallback_spec = self.stage._build_fallback_spec(
            knowledge=knowledge,
            context=context,
            project_name=project_name,
        )
        return BuildPlan(
            chosen_vulnerable_ref=fallback_spec.chosen_vulnerable_ref,
            chosen_fixed_ref=fallback_spec.chosen_fixed_ref,
            build_system=fallback_spec.build_system,
            install_packages=fallback_spec.install_packages,
            configure_commands=fallback_spec.configure_commands,
            clean_commands=fallback_spec.clean_commands,
            build_commands=fallback_spec.build_commands,
            expected_binary_path=fallback_spec.expected_binary_path,
            dockerfile_override=None,
            build_script_override=None,
            source_of_truth=fallback_spec.source_of_truth,
            confidence=fallback_spec.confidence,
            rationale=fallback_spec.rationale,
        )


class BuildStage:
    """构建阶段协调器。"""

    BUILD_FILE_PATTERNS = (
        "Makefile",
        "makefile",
        "CMakeLists.txt",
        "configure",
        "configure.ac",
        "meson.build",
        "build.ninja",
        "Cargo.toml",
        "go.mod",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "package.json",
    )
    README_PATTERNS = ("README", "README.md", "README.txt", "INSTALL", "INSTALL.md")
    MAX_REPLAN_ATTEMPTS = 3
    MAX_LLM_NO_RESPONSE_RETRIES = 2
    DEFAULT_BASE_IMAGE = "ubuntu:20.04"
    MODERN_UBUNTU_BASE_IMAGE = "ubuntu:22.04"
    # Packages that are absent from Ubuntu 20.04 (focal) default apt archives.
    UBUNTU_FOCAL_UNAVAILABLE_PACKAGES = frozenset(
        {
            "libavif-dev",
            "libavif14",
            "libavif-bin",
        }
    )
    # Focal ships ECM/Qt around these ceilings; newer CMake requirements need jammy+.
    UBUNTU_FOCAL_MAX_ECM = (5, 68)
    UBUNTU_FOCAL_MAX_QT = (5, 12)
    UBUNTU_FOCAL_MAX_KF = (5, 68)
    REQUIRED_DOCKER_PACKAGES = (
        "build-essential",
        "ca-certificates",
        "clang",
        "g++",
        "gcc",
        "git",
        "make",
        "pkg-config",
    )

    def __init__(
        self,
        file_tool: FileTool | None = None,
        process_tool: ProcessTool | None = None,
        git_tool: GitTool | None = None,
        docker_tool: DockerTool | None = None,
    ) -> None:
        self.file_tool = file_tool or FileTool()
        self.process_tool = process_tool or ProcessTool()
        self.git_tool = git_tool or GitTool(process_tool=self.process_tool)
        self.docker_tool = docker_tool or DockerTool(process_tool=self.process_tool)
        self.planner = BuildPlanner(self)
        self._active_build_dir: str = ""

    def build_plan(self, knowledge: KnowledgeModel, workspace: str) -> dict:
        """生成构建阶段最初的静态计划。"""

        if not knowledge.repo_url:
            raise RuntimeError("knowledge.repo_url is required for build stage")
        if not knowledge.vulnerable_ref and not knowledge.fixed_ref:
            raise RuntimeError("knowledge.vulnerable_ref or knowledge.fixed_ref is required for build stage")

        paths = BuildStagePaths(workspace)
        project_name = self._derive_project_name(knowledge.repo_url)
        return {
            "repo_url": knowledge.repo_url,
            "workspace": workspace,
            "project_name": project_name,
            "project_dir_name": project_name,
            "repo_dir": str(paths.repo_dir),
            "artifacts_dir": str(paths.artifacts_dir),
            "build_artifacts_dir": str(paths.build_dir),
            "poc_artifacts_dir": str(paths.poc_dir),
            "verify_artifacts_dir": str(paths.verify_dir),
            "docker_image_tag": f"deeprepro-{knowledge.cve_id.lower()}-build",
            "compiled_image_tag": f"deeprepro-{knowledge.cve_id.lower()}-build-compiled",
            "build_container_name": f"deeprepro-{knowledge.cve_id.lower()}-build-run",
        }

    def render_prompt(self, knowledge: KnowledgeModel, plan: dict) -> str:
        """生成 build planner 提示词。"""

        prompt = {
            "cve_id": knowledge.cve_id,
            "repo_url": plan["repo_url"],
            "workspace": plan["workspace"],
        }
        return json.dumps(prompt, ensure_ascii=False)

    def run(self, knowledge: KnowledgeModel, workspace: str) -> BuildArtifact:
        """执行构建阶段并返回构建产物。

        结构上拆成三层：
        1. collect_context：准备 workspace、clone repo、收集本地证据
        2. plan_and_execute：生成 BuildPlan，并在失败时做有限次再规划
        3. verify_and_persist：固化最终产物并做事实型自检
        """

        paths = BuildStagePaths(workspace)
        subgraph = self.build_internal_graph()
        result = subgraph.invoke(
            {
                "knowledge": knowledge,
                "paths": paths,
                "attempt": 0,
            }
        )
        prepared = result["prepared"]
        outcome = result["outcome"]
        self.persist_build_outputs(
            artifact=outcome.artifact,
            paths=paths,
            plan_meta=prepared.plan_meta,
            cve_id=knowledge.cve_id,
        )
        return outcome.artifact

    def build_internal_graph(self):
        """Build the internal LangGraph subgraph for the build stage."""

        builder = StateGraph(BuildGraphState)
        builder.add_node("prepare", self._build_graph_prepare_node)
        builder.add_node("plan", self._build_graph_plan_node)
        builder.add_node("execute", self._build_graph_execute_node)

        builder.add_edge(START, "prepare")
        builder.add_edge("prepare", "plan")
        builder.add_edge("plan", "execute")
        builder.add_conditional_edges(
            "execute",
            self._route_after_build_execute,
            {
                "plan": "plan",
                "done": END,
            },
        )
        # False: do not inherit the parent workflow checkpointer. Internal state
        # carries BuildStagePaths, which is not msgpack-serializable.
        return builder.compile(checkpointer=False)

    def _build_graph_prepare_node(self, state: BuildGraphState) -> BuildGraphState:
        knowledge = state["knowledge"]
        paths = state["paths"]
        prepared = self.prepare_build_run(knowledge=knowledge, paths=paths)
        return {
            "prepared": prepared,
            "current_context": prepared.context,
            "attempt": 0,
        }

    def _build_graph_plan_node(self, state: BuildGraphState) -> BuildGraphState:
        knowledge = state["knowledge"]
        prepared = state["prepared"]
        current_context = state["current_context"]
        current_plan = self.plan_build(
            knowledge=knowledge,
            context=current_context,
            project_name=prepared.plan_meta["project_name"],
        )
        return {"current_plan": current_plan}

    def _build_graph_execute_node(self, state: BuildGraphState) -> BuildGraphState:
        prepared = state["prepared"]
        paths = state["paths"]
        repo_path = Path(prepared.repo_path)
        current_plan = self._normalize_build_plan(
            repo_path,
            state["current_plan"],
            knowledge=state["knowledge"],
        )
        self._write_yaml_file(paths.build_plan_yaml, current_plan.model_dump(mode="json"))
        outcome = self.execute_build_attempt(
            repo_path=repo_path,
            paths=paths,
            plan_meta=prepared.plan_meta,
            build_plan=current_plan,
        )

        if outcome.artifact.build_success:
            return {
                "current_plan": current_plan,
                "outcome": outcome,
                "attempt": state.get("attempt", 0) + 1,
                "should_retry": False,
            }

        replanned, next_context = self._replan_from_failed_attempt(
            knowledge=state["knowledge"],
            context=state["current_context"],
            project_name=prepared.plan_meta["project_name"],
            previous_plan=current_plan,
            artifact=outcome.artifact,
            repo_path=repo_path,
        )
        updates: BuildGraphState = {
            "current_plan": current_plan,
            "outcome": outcome,
            "attempt": state.get("attempt", 0) + 1,
            "should_retry": False,
        }
        if replanned is not None and next_context is not None:
            updates["current_plan"] = replanned
            updates["current_context"] = next_context
            updates["should_retry"] = True
        return updates

    def _route_after_build_execute(self, state: BuildGraphState) -> str:
        outcome = state.get("outcome")
        attempt = state.get("attempt", 0)
        current_plan = state.get("current_plan")
        if outcome is None:
            return "done"
        if outcome.artifact.build_success:
            return "done"
        if attempt >= self.MAX_REPLAN_ATTEMPTS:
            return "done"
        if current_plan is None or not state.get("should_retry"):
            return "done"
        return "plan"

    def prepare_build_run(self, knowledge: KnowledgeModel, paths: BuildStagePaths) -> BuildPreparedRun:
        """Collect deterministic build inputs before any planning starts."""

        plan_meta = self.build_plan(knowledge=knowledge, workspace=str(paths.workspace_root))
        self._prepare_workspace(paths)
        self._active_build_dir = str(paths.build_dir)
        repo = self.git_tool.clone_repo(plan_meta["repo_url"], plan_meta["repo_dir"])
        repo_path = Path(repo.local_path)
        context = self.collect_build_context(
            knowledge=knowledge,
            repo_path=repo_path,
            planner_attempt=1,
        )
        self._write_yaml_file(paths.build_context_yaml, context.model_dump(mode="json"))
        return BuildPreparedRun(
            plan_meta=plan_meta,
            repo_path=str(repo_path),
            context=context,
        )

    def plan_and_execute_build(
        self,
        knowledge: KnowledgeModel,
        prepared: BuildPreparedRun,
        paths: BuildStagePaths,
    ) -> BuildExecutionOutcome:
        """Generate a plan, execute it, and optionally replan after failures."""

        repo_path = Path(prepared.repo_path)
        current_context = prepared.context
        current_plan = self.plan_build(
            knowledge=knowledge,
            context=current_context,
            project_name=prepared.plan_meta["project_name"],
        )
        last_outcome: BuildExecutionOutcome | None = None

        for attempt in range(self.MAX_REPLAN_ATTEMPTS):
            current_plan = self._normalize_build_plan(repo_path, current_plan, knowledge=knowledge)
            self._write_yaml_file(paths.build_plan_yaml, current_plan.model_dump(mode="json"))
            last_outcome = self.execute_build_attempt(
                repo_path=repo_path,
                paths=paths,
                plan_meta=prepared.plan_meta,
                build_plan=current_plan,
            )
            if last_outcome.artifact.build_success or attempt + 1 >= self.MAX_REPLAN_ATTEMPTS:
                break

            replanned, next_context = self._replan_from_failed_attempt(
                knowledge=knowledge,
                context=current_context,
                project_name=prepared.plan_meta["project_name"],
                previous_plan=current_plan,
                artifact=last_outcome.artifact,
                repo_path=repo_path,
            )
            if replanned is None or next_context is None:
                break
            current_plan = replanned
            current_context = next_context

        if last_outcome is None:
            raise RuntimeError("build stage did not produce an artifact")
        return last_outcome

    def execute_build_attempt(
        self,
        repo_path: Path,
        paths: BuildStagePaths,
        plan_meta: dict[str, Any],
        build_plan: BuildPlan,
    ) -> BuildExecutionOutcome:
        """Execute one concrete build attempt from a single plan."""

        checkout = self.git_tool.checkout_ref(str(repo_path), build_plan.chosen_vulnerable_ref)
        artifact = self._execute_build_plan(
            repo_path=repo_path,
            paths=paths,
            plan_meta=plan_meta,
            build_plan=build_plan,
            resolved_ref=checkout.current_ref,
        )
        return BuildExecutionOutcome(plan=build_plan, artifact=artifact)

    def persist_build_outputs(
        self,
        artifact: BuildArtifact,
        paths: BuildStagePaths,
        plan_meta: dict[str, Any],
        cve_id: str,
    ) -> None:
        """Persist the final build artifact and self-verification payload."""

        self._write_yaml_file(paths.build_artifact_yaml, artifact.model_dump(mode="json"))

        try:
            verify_payload = self._verify_build_artifact(
                artifact=artifact,
                paths=paths,
                plan_meta=plan_meta,
                cve_id=cve_id,
            )
        except Exception as error:
            verify_payload = {
                "verify_status": "verify_self_failed",
                "verify_error": str(error),
            }
        self._write_yaml_file(paths.build_verify_yaml, verify_payload)

    def collect_build_context(self, knowledge: KnowledgeModel, repo_path: Path, planner_attempt: int = 1) -> BuildContext:
        """Collect local build evidence from repo snapshots, patch diff, and knowledge outputs."""

        patch_diff_path = find_patch_diff(knowledge.cve_id)
        patch_diff_text = ""
        if patch_diff_path is not None:
            patch_diff_text = patch_diff_path.read_text(encoding="utf-8", errors="replace")
        patch_affected_files = sorted(set(re.findall(r"^\+\+\+ b/(.+)$", patch_diff_text, re.MULTILINE)))

        snapshots: list[RefSnapshot] = []
        for label, requested_ref in self._candidate_refs(repo_path, knowledge).items():
            snapshot = self._collect_ref_snapshot(repo_path, label=label, requested_ref=requested_ref, affected_files=patch_affected_files)
            if snapshot is not None:
                snapshots.append(snapshot)

        return BuildContext(
            cve_id=knowledge.cve_id,
            repo_url=knowledge.repo_url or "",
            task_vulnerable_ref=knowledge.vulnerable_ref,
            task_fixed_ref=knowledge.fixed_ref,
            patch_diff_excerpt=patch_diff_text[:4000],
            patch_affected_files=patch_affected_files or list(knowledge.affected_files),
            knowledge_summary=knowledge.summary,
            knowledge_build_hints=list(knowledge.build_hints),
            knowledge_reproduction_hints=list(knowledge.reproduction_hints),
            snapshots=snapshots,
            planner_attempt=planner_attempt,
        )

    def plan_build(self, knowledge: KnowledgeModel, context: BuildContext, project_name: str) -> BuildPlan:
        """Plan the build using the dedicated planner."""

        return self.planner.plan(knowledge=knowledge, context=context, project_name=project_name)

    def replan_after_failure(
        self,
        knowledge: KnowledgeModel,
        context: BuildContext,
        project_name: str,
        previous_plan: BuildPlan,
        previous_artifact: BuildArtifact,
    ) -> Optional[BuildPlan]:
        """Give the planner one chance to adjust the plan after a build failure."""

        return self.planner.replan_after_failure(
            knowledge=knowledge,
            context=context,
            project_name=project_name,
            previous_plan=previous_plan,
            previous_artifact=previous_artifact,
        )

    def _replan_from_failed_attempt(
        self,
        knowledge: KnowledgeModel,
        context: BuildContext,
        project_name: str,
        previous_plan: BuildPlan,
        artifact: BuildArtifact,
        repo_path: Path,
    ) -> tuple[Optional[BuildPlan], Optional[BuildContext]]:
        """Convert a failed attempt into replanning inputs."""

        failure_kind = self._classify_failure_kind(artifact.build_logs)
        replanned = self.replan_after_failure(
            knowledge=knowledge,
            context=context,
            project_name=project_name,
            previous_plan=previous_plan,
            previous_artifact=artifact,
        )
        if replanned is None:
            return None, None

        replanned = self._normalize_build_plan(repo_path, replanned, knowledge=knowledge)
        if replanned.model_dump(mode="json") == previous_plan.model_dump(mode="json"):
            return None, None

        next_context = context.model_copy(
            update={
                "planner_attempt": context.planner_attempt + 1,
                "previous_failure_kind": failure_kind,
                "previous_build_failure": artifact.build_logs[:6000],
                "previous_dockerfile_content": artifact.dockerfile_content[:8000],
                "previous_build_script_content": artifact.build_script_content[:8000],
            }
        )
        return replanned, next_context

    def _prepare_workspace(self, paths: BuildStagePaths) -> None:
        self.file_tool.ensure_dir(str(paths.workspace_root))
        self.file_tool.ensure_dir(str(paths.build_dir))
        self.file_tool.ensure_dir(str(paths.poc_dir))
        self.file_tool.ensure_dir(str(paths.verify_dir))
        self.file_tool.ensure_dir(str(paths.llm_dir))

    def _persist_build_llm_trace(self, planner_attempt: int, filename: str, content: str) -> None:
        build_dir = getattr(self, "_active_build_dir", None)
        if not build_dir:
            return
        attempt_dir = Path(build_dir) / "llm" / f"attempt-{planner_attempt}"
        self.file_tool.ensure_dir(str(attempt_dir))
        self.file_tool.write_text(str(attempt_dir / filename), (content or "").rstrip() + "\n")

    def _should_retry_llm_request(self, error_text: str) -> bool:
        normalized = (error_text or "").strip().lower()
        return "timed out" in normalized or "timeout" in normalized

    def _is_empty_llm_response(self, raw_response: str) -> bool:
        return not (raw_response or "").strip()

    def _write_yaml_file(self, path: Path, payload: Any) -> None:
        """Persist YAML using one consistent formatting policy."""

        self.file_tool.write_text(
            str(path),
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        )

    def _normalize_build_plan(
        self,
        repo_path: Path,
        build_plan: BuildPlan,
        knowledge: KnowledgeModel | None = None,
    ) -> BuildPlan:
        build_plan.chosen_vulnerable_ref = self._resolve_existing_ref(repo_path, build_plan.chosen_vulnerable_ref)
        if build_plan.chosen_fixed_ref:
            build_plan.chosen_fixed_ref = self._resolve_existing_ref(repo_path, build_plan.chosen_fixed_ref)
        build_plan = self._align_vulnerable_ref_with_applyable_patch(
            repo_path,
            build_plan,
            knowledge=knowledge,
        )
        build_plan.install_packages = self._ensure_required_docker_packages(build_plan.install_packages)
        build_plan.install_packages = self._augment_install_packages_from_repo(
            build_plan.install_packages,
            repo_path,
        )
        build_plan.base_image = self._select_base_image(repo_path, build_plan)
        base_image = build_plan.base_image or self.DEFAULT_BASE_IMAGE
        build_plan.install_packages = self._filter_unavailable_base_image_packages(
            build_plan.install_packages,
            base_image=base_image,
        )
        incompatible_optional = self._incompatible_optional_packages(repo_path)
        if incompatible_optional:
            build_plan.install_packages = [
                package
                for package in build_plan.install_packages
                if package.strip().lower() not in incompatible_optional
            ]
        build_plan.build_commands = self._sanitize_make_build_commands(
            build_plan.build_commands,
            repo_path,
            build_plan.build_system,
        )
        build_plan = self._narrow_selinux_cil_build(repo_path, build_plan)
        build_plan.configure_commands = self._ensure_configure_commands(
            build_plan.configure_commands,
            repo_path,
        )
        build_plan.clean_commands = self._sanitize_clean_commands(
            build_plan.clean_commands,
            has_configure=bool(build_plan.configure_commands),
        )
        build_plan = self._normalize_cmake_layout(repo_path, build_plan)
        build_plan = self._prefer_ossfuzz_harness_target(repo_path, build_plan, knowledge)
        if self._should_inject_make_asan_overrides(repo_path, build_plan):
            # Makefiles that hardcode CFLAGS=/CC= ignore env exports (Lua, many
            # autotools). Force ASan (and clang) on the make command line.
            build_plan.build_commands = self._inject_make_asan_flag_overrides(
                build_plan.build_commands,
                repo_path,
            )
        if build_plan.build_script_override:
            build_plan.build_script_override = self._sanitize_make_script_override(
                build_plan.build_script_override,
                repo_path,
            )
            build_plan.build_script_override = self._soften_make_clean_in_script(
                build_plan.build_script_override,
            )
            build_plan.build_script_override = self._demote_distclean_in_script(
                build_plan.build_script_override,
            )
            build_plan.build_script_override = self._rewrite_broken_cmake_in_script(
                build_plan.build_script_override,
            )
            if self._is_cmake_build(repo_path, build_plan):
                build_plan.build_script_override = self._ensure_cmake_asan_flags_in_script(
                    build_plan.build_script_override,
                )
            build_plan.build_script_override = self._ensure_shared_libasan_in_build_script(
                build_plan.build_script_override,
                repo_path,
            )
            if self._is_cmake_build(repo_path, build_plan):
                build_plan.build_script_override = self._strip_global_asan_env_exports(
                    build_plan.build_script_override,
                )
            else:
                build_plan.build_script_override = self._defer_asan_env_until_after_configure(
                    build_plan.build_script_override,
                )
                if self._should_inject_make_asan_overrides(repo_path, build_plan):
                    build_plan.build_script_override = self._inject_make_asan_flag_overrides_in_script(
                        build_plan.build_script_override,
                        repo_path,
                    )
        if build_plan.dockerfile_override:
            build_plan.dockerfile_override = self._ensure_dockerfile_base_image(
                build_plan.dockerfile_override,
                base_image,
            )
            build_plan.dockerfile_override = self._strip_unavailable_packages_from_dockerfile(
                build_plan.dockerfile_override,
                base_image=base_image,
            )
            if incompatible_optional:
                build_plan.dockerfile_override = self._strip_named_packages_from_dockerfile(
                    build_plan.dockerfile_override,
                    incompatible_optional,
                )
            build_plan.dockerfile_override = self._ensure_dockerfile_override_has_required_tools(
                build_plan.dockerfile_override,
                build_plan.install_packages,
            )
            build_plan.dockerfile_override = self._ensure_dockerfile_override_includes_packages(
                build_plan.dockerfile_override,
                build_plan.install_packages,
            )
        build_plan.expected_binary_path = self._sanitize_expected_binary_path(
            build_plan.expected_binary_path,
            repo_path,
        )
        # Canonical -shared-libasan is clang-only; rewrite LLM --CC=gcc (etc.).
        build_plan = self._align_compiler_with_shared_libasan(build_plan)
        return build_plan

    def _sanitize_expected_binary_path(self, expected_binary_path: Optional[str], repo_path: Path) -> Optional[str]:
        """Prefer a root make target over a mistaken `src/<name>` path."""

        if not expected_binary_path:
            return expected_binary_path
        cleaned = expected_binary_path.replace("\\", "/")
        # KDE/Qt image plugins are emitted under build/bin/imageformats, not build/src/...
        cleaned = re.sub(
            r"(/build)/src/imageformats/(kimg_[^/]+\.so)$",
            r"\1/bin/imageformats/\2",
            cleaned,
        )
        cleaned = re.sub(
            r"^build/src/imageformats/(kimg_[^/]+\.so)$",
            r"build/bin/imageformats/\1",
            cleaned,
        )
        cleaned = re.sub(
            r"(/build/bin)/(?!imageformats/)(kimg_[^/]+\.so)$",
            r"\1/imageformats/\2",
            cleaned,
        )
        cleaned = re.sub(
            r"^build/bin/(?!imageformats/)(kimg_[^/]+\.so)$",
            r"build/bin/imageformats/\1",
            cleaned,
        )
        rel = cleaned.lstrip("/")
        match = re.fullmatch(r"src/([^/]+)", rel)
        if not match:
            return cleaned
        candidate = match.group(1)
        if (repo_path / "src" / candidate).exists():
            return cleaned
        targets = self._extract_makefile_targets(repo_path)
        if candidate in targets:
            return candidate
        return cleaned

    def _sanitize_clean_commands(
        self,
        clean_commands: list[str],
        *,
        has_configure: bool = False,
    ) -> list[str]:
        """Make cleanup best-effort; demote distclean when configure will run.

        ``make distclean`` after ``./configure`` wipes the generated Makefile.
        Template order is clean → configure → build, so prefer ``make clean``.
        """

        sanitized: list[str] = []
        for command in clean_commands or []:
            stripped = (command or "").strip()
            if not stripped:
                continue
            if has_configure and re.search(r"\bdistclean\b", stripped):
                sanitized.append("make clean || true")
                continue
            if "||" in stripped:
                sanitized.append(stripped)
                continue
            if re.search(r"\bmake\b.*\b(clean|distclean)\b", stripped):
                sanitized.append(f"{stripped} || true")
            else:
                sanitized.append(stripped)
        return sanitized

    def _ensure_configure_commands(self, configure_commands: list[str], repo_path: Path) -> list[str]:
        """Ensure autotools repos run configure before make when evidence exists."""

        autogen = repo_path / "autogen.sh"
        if autogen.is_file():
            # A repo's own bootstrap script is authoritative: it handles
            # project-specific quirks (e.g. wolfMQTT's build-aux/config.rpath)
            # that a bare ``autoreconf`` misses. Prefer it whenever present.
            commands = list(configure_commands or [])
            if not commands:
                return ["bash ./autogen.sh", "./configure"]
            rewritten = [
                "bash ./autogen.sh" if re.search(r"\bautoreconf\b", command or "") else command
                for command in commands
            ]
            return rewritten
        if configure_commands:
            return list(configure_commands)
        if (repo_path / "configure").is_file():
            return ["./configure"]
        if (repo_path / "configure.ac").is_file() or (repo_path / "configure.in").is_file():
            return ["autoreconf -fi || true", "./configure"]
        return list(configure_commands or [])

    def _detect_base_image(self, dockerfile_override: Optional[str]) -> str:
        """Return Dockerfile FROM image, defaulting to the stage base image."""

        if dockerfile_override:
            match = re.search(r"(?im)^FROM\s+([^\s]+)", dockerfile_override)
            if match:
                return match.group(1).strip()
        return self.DEFAULT_BASE_IMAGE

    def _parse_version_tuple(self, value: str) -> Optional[tuple[int, ...]]:
        parts = re.findall(r"\d+", value or "")
        if not parts:
            return None
        return tuple(int(part) for part in parts[:3])

    def _scan_cmake_version_requirements(self, repo_path: Path) -> dict[str, tuple[int, ...]]:
        """Extract minimum ECM/Qt/KF versions declared in CMake files."""

        requirements: dict[str, tuple[int, ...]] = {}
        cmake_files: list[Path] = []
        root = repo_path / "CMakeLists.txt"
        if root.is_file():
            cmake_files.append(root)
        for pattern in ("cmake/*.cmake", "CMakeLists.txt"):
            cmake_files.extend(path for path in repo_path.glob(pattern) if path.is_file())

        patterns = {
            "ecm": [
                re.compile(r"find_package\s*\(\s*ECM\s+([0-9]+(?:\.[0-9]+){0,2})", re.IGNORECASE),
                re.compile(r"set\s*\(\s*(?:KF5_MIN_VERSION|KF_MIN_VERSION|ECM_VERSION)\s+[\"']?([0-9]+(?:\.[0-9]+){0,2})", re.IGNORECASE),
            ],
            "qt": [
                re.compile(r"set\s*\(\s*REQUIRED_QT_VERSION\s+([0-9]+(?:\.[0-9]+){0,2})", re.IGNORECASE),
                re.compile(r"find_package\s*\(\s*Qt5(?:Gui|Core|Widgets|PrintSupport)?\s+([0-9]+(?:\.[0-9]+){0,2})", re.IGNORECASE),
            ],
            "kf": [
                re.compile(r"find_package\s*\(\s*KF5\w*\s+([0-9]+(?:\.[0-9]+){0,2})", re.IGNORECASE),
                re.compile(r"set\s*\(\s*(?:KF5_DEP_VERSION|KF_DEP_VERSION)\s+[\"']?([0-9]+(?:\.[0-9]+){0,2})", re.IGNORECASE),
            ],
        }

        for path in cmake_files[:20]:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for key, regexes in patterns.items():
                for regex in regexes:
                    for match in regex.finditer(text):
                        parsed = self._parse_version_tuple(match.group(1))
                        if parsed is None:
                            continue
                        current = requirements.get(key)
                        if current is None or parsed > current:
                            requirements[key] = parsed
        return requirements

    def _required_base_image_from_repo(self, repo_path: Path) -> Optional[str]:
        """Pick a newer Ubuntu image when CMake deps exceed focal package ceilings."""

        requirements = self._scan_cmake_version_requirements(repo_path)
        ecm = requirements.get("ecm")
        qt = requirements.get("qt")
        kf = requirements.get("kf")
        if ecm and ecm > self.UBUNTU_FOCAL_MAX_ECM:
            return self.MODERN_UBUNTU_BASE_IMAGE
        if qt and qt > self.UBUNTU_FOCAL_MAX_QT:
            return self.MODERN_UBUNTU_BASE_IMAGE
        if kf and kf > self.UBUNTU_FOCAL_MAX_KF:
            return self.MODERN_UBUNTU_BASE_IMAGE
        return None

    def _ubuntu_image_rank(self, image: str) -> tuple[int, int]:
        lowered = (image or "").lower()
        match = re.search(r"ubuntu:(\d+)\.(\d+)", lowered)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        if "noble" in lowered or "24.04" in lowered:
            return (24, 4)
        if "jammy" in lowered or "22.04" in lowered:
            return (22, 4)
        if "focal" in lowered or "20.04" in lowered:
            return (20, 4)
        if "bionic" in lowered or "18.04" in lowered:
            return (18, 4)
        return (0, 0)

    def _newest_ubuntu_image(self, images: list[str]) -> str:
        ranked = sorted(
            ((self._ubuntu_image_rank(image), image) for image in images if image),
            key=lambda item: item[0],
            reverse=True,
        )
        if not ranked or ranked[0][0] == (0, 0):
            return self.DEFAULT_BASE_IMAGE
        return ranked[0][1]

    def _select_base_image(self, repo_path: Path, build_plan: BuildPlan) -> str:
        """Choose base image from plan/override/repo dependency evidence."""

        candidates = [self.DEFAULT_BASE_IMAGE]
        if build_plan.base_image:
            candidates.append(build_plan.base_image)
        if build_plan.dockerfile_override:
            candidates.append(self._detect_base_image(build_plan.dockerfile_override))
        required = self._required_base_image_from_repo(repo_path)
        if required:
            candidates.append(required)
        return self._newest_ubuntu_image(candidates)

    def _ensure_dockerfile_base_image(self, dockerfile_content: str, base_image: str) -> str:
        """Rewrite Dockerfile FROM line when dependency evidence requires a newer image."""

        if not dockerfile_content or not base_image:
            return dockerfile_content
        current = self._detect_base_image(dockerfile_content)
        if self._ubuntu_image_rank(current) >= self._ubuntu_image_rank(base_image):
            return dockerfile_content if dockerfile_content.endswith("\n") else dockerfile_content + "\n"
        rewritten = re.sub(
            r"(?im)^FROM\s+[^\s]+",
            f"FROM {base_image}",
            dockerfile_content,
            count=1,
        )
        return rewritten if rewritten.endswith("\n") else rewritten + "\n"

    def _unavailable_packages_for_base_image(self, base_image: str) -> set[str]:
        """Return package names that should not be apt-installed for this base image."""

        lowered = (base_image or "").lower()
        rank = self._ubuntu_image_rank(lowered)
        if rank and rank <= (20, 4):
            return {name.lower() for name in self.UBUNTU_FOCAL_UNAVAILABLE_PACKAGES}
        if "ubuntu" in lowered and rank == (0, 0):
            # Unrecognized/default Ubuntu tag — treat as focal for safety.
            return {name.lower() for name in self.UBUNTU_FOCAL_UNAVAILABLE_PACKAGES}
        return set()

    def _filter_unavailable_base_image_packages(self, packages: list[str], base_image: str) -> list[str]:
        """Drop apt packages known to be unavailable on the selected base image."""

        unavailable = self._unavailable_packages_for_base_image(base_image)
        if not unavailable:
            return list(packages)
        return [package for package in packages if package.strip().lower() not in unavailable]

    def _strip_unavailable_packages_from_dockerfile(self, dockerfile_content: str, base_image: str) -> str:
        """Remove unavailable package tokens from apt-get install lines."""

        unavailable = self._unavailable_packages_for_base_image(base_image)
        if not unavailable or not dockerfile_content:
            return dockerfile_content

        updated_lines: list[str] = []
        for line in dockerfile_content.splitlines():
            if "apt-get install" not in line.lower():
                updated_lines.append(line)
                continue
            tokens = line.split()
            filtered: list[str] = []
            for token in tokens:
                bare = token.strip().rstrip("\\")
                if bare.lower() in unavailable:
                    continue
                filtered.append(token)
            # Keep line continuations tidy when the final package was removed.
            rewritten = " ".join(filtered)
            if rewritten.rstrip().endswith("\\") is False and line.rstrip().endswith("\\"):
                rewritten = rewritten.rstrip() + " \\"
            rewritten = re.sub(r"[ \t]{2,}", " ", rewritten)
            updated_lines.append(rewritten)
        return "\n".join(updated_lines).rstrip() + "\n"

    def _is_cmake_command(self, command: str) -> bool:
        stripped = (command or "").strip()
        if not stripped:
            return False
        if re.match(r"^cmake\b", stripped):
            return True
        if re.match(r"^mkdir\s+(-p\s+)?build\b", stripped):
            return True
        if re.match(r"^cd\s+build\b", stripped):
            return True
        return False

    def _extract_cmake_define_flags(self, commands: list[str]) -> list[str]:
        """Preserve non-default -D flags from existing cmake invocations."""

        skipped_keys = {
            "CMAKE_BUILD_TYPE",
            "CMAKE_C_FLAGS",
            "CMAKE_CXX_FLAGS",
            "CMAKE_EXE_LINKER_FLAGS",
            "CMAKE_SHARED_LINKER_FLAGS",
            "CMAKE_MODULE_LINKER_FLAGS",
            "CMAKE_C_COMPILER",
            "CMAKE_CXX_COMPILER",
        }
        flags: list[str] = []
        seen: set[str] = set()
        for command in commands:
            for match in re.finditer(r"-D([A-Za-z0-9_]+)=([^\s\"']+)", command or ""):
                key = match.group(1)
                if key.upper() in skipped_keys:
                    continue
                flag = f"-D{key}={match.group(2)}"
                if flag not in seen:
                    seen.add(flag)
                    flags.append(flag)
        return flags

    def _repo_uses_legacy_libavif_api(self, repo_path: Path) -> bool:
        """Detect old libavif mirror API that breaks against modern libavif headers."""

        if not repo_path.exists():
            return False
        markers = ("imir.axis", "avifImageMirror")
        candidates: list[Path] = []
        for pattern in ("**/*avif*.cpp", "**/*avif*.c", "**/*avif*.h", "**/*avif*.hpp"):
            candidates.extend(path for path in repo_path.glob(pattern) if path.is_file())
        for path in candidates[:20]:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if "imir.axis" in text:
                return True
            if all(marker in text for marker in markers):
                return True
        return False

    def _incompatible_optional_packages(self, repo_path: Path) -> set[str]:
        """Optional apt packages that would enable plugins known to fail on this checkout."""

        incompatible: set[str] = set()
        if self._repo_uses_legacy_libavif_api(repo_path):
            incompatible.update({"libavif-dev", "libavif14", "libavif-bin"})
        return incompatible

    def _strip_named_packages_from_dockerfile(self, dockerfile_content: str, packages: set[str]) -> str:
        """Remove specific package tokens from apt-get install lines."""

        if not dockerfile_content or not packages:
            return dockerfile_content
        blocked = {name.lower() for name in packages}
        updated_lines: list[str] = []
        for line in dockerfile_content.splitlines():
            if "apt-get install" not in line.lower():
                updated_lines.append(line)
                continue
            tokens = line.split()
            filtered = [token for token in tokens if token.strip().rstrip("\\").lower() not in blocked]
            rewritten = " ".join(filtered)
            if line.rstrip().endswith("\\") and not rewritten.rstrip().endswith("\\"):
                rewritten = rewritten.rstrip() + " \\"
            rewritten = re.sub(r"[ \t]{2,}", " ", rewritten)
            updated_lines.append(rewritten)
        return "\n".join(updated_lines).rstrip() + "\n"

    def _repo_links_shared_objects(self, repo_path: Path) -> bool:
        """True when Makefiles / .mk files link shared libraries (-shared)."""

        names = {"makefile", "gnumakefile"}
        for path in repo_path.rglob("*"):
            if not path.is_file():
                continue
            lower = path.name.lower()
            if lower not in names and not lower.endswith(".mk"):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if re.search(r"(^|[\s\"'])-shared\b", text):
                return True
        return False

    def _default_sanitizer_flags(self, repo_path: Path | None = None) -> str:
        """Canonical ASan compile flags (CFLAGS/CXXFLAGS) for every C/C++ build.

        Always use -shared-libasan. Static ASan archives frequently fail when
        linking shared objects / plugins (undefined __asan_*), and shared mode
        is also fine for plain executables as long as run scripts put the clang
        runtime dir on LD_LIBRARY_PATH (poc_run / verify_run templates do).

        Non-PIE hardening is split across compile/link: this returns the compile
        side (-fno-pie only); the link side adds -no-pie via
        ``_default_sanitizer_link_flags``. -no-pie must NOT go into CFLAGS because
        clang treats it as an unused compile argument (and -Werror builds such as
        fluent-bit's mbedtls fail on it). -fno-pie alone is NOT enough to force a
        non-PIE link on distros with PIE-by-default specs, so -no-pie is required
        at link time.
        """

        del repo_path  # signature kept for call-site compatibility
        return "-fsanitize=address -shared-libasan -fno-omit-frame-pointer -fno-pie"

    def _default_sanitizer_link_flags(self, repo_path: Path | None = None) -> str:
        """Canonical ASan link flags (LDFLAGS): compile flags + ``-no-pie``.

        ``-no-pie`` forces a non-PIE executable even under PIE-by-default
        toolchains; it belongs at link time only (clang errors on it during pure
        compilation, e.g. ``-Werror,-Wunused-command-line-argument``).
        """

        del repo_path
        return "-fsanitize=address -shared-libasan -fno-omit-frame-pointer -no-pie"

    def _is_cmake_build(self, repo_path: Path | None, build_plan: BuildPlan | None = None) -> bool:
        """True when the plan or repo evidence indicates a CMake build."""

        if build_plan is not None and (build_plan.build_system or "").strip().lower() == "cmake":
            return True
        if repo_path is not None and (repo_path / "CMakeLists.txt").is_file():
            return True
        return False

    def _ensure_shared_libasan_in_build_script(self, script: str, repo_path: Path | None = None) -> str:
        """Rewrite any ASan build script to the shared-libasan policy.

        Covers template output, LLM overrides, and retries so make/cmake/meson
        plans cannot regress into the static-ASan + .so link failure class.
        """

        del repo_path
        if not script or "-fsanitize=address" not in script:
            return script
        if "-shared-libasan" in script:
            return script
        return script.replace("-fsanitize=address", "-fsanitize=address -shared-libasan")

    def _align_compiler_with_shared_libasan(self, build_plan: BuildPlan) -> BuildPlan:
        """Force clang when the canonical ``-shared-libasan`` ASan policy applies.

        gcc does not accept ``-shared-libasan``. LLM plans often set ``--CC=gcc``
        from README habits while the framework injects shared-libasan into make
        CFLAGS (autotools deferred injection / cmake CMAKE_* flags).
        """

        if "-shared-libasan" not in self._default_sanitizer_flags():
            return build_plan

        configure = list(build_plan.configure_commands or [])
        rewritten_configure = [self._rewrite_gcc_compiler_to_clang(cmd) for cmd in configure]
        if rewritten_configure != configure:
            build_plan.configure_commands = rewritten_configure

        build_cmds = list(build_plan.build_commands or [])
        rewritten_build = [self._rewrite_gcc_compiler_to_clang(cmd) for cmd in build_cmds]
        if rewritten_build != build_cmds:
            build_plan.build_commands = rewritten_build

        if build_plan.build_script_override:
            rewritten_script = self._rewrite_gcc_compiler_to_clang(build_plan.build_script_override)
            if rewritten_script != build_plan.build_script_override:
                build_plan.build_script_override = rewritten_script
        return build_plan

    @staticmethod
    def _rewrite_gcc_compiler_to_clang(text: str) -> str:
        """Rewrite gcc/g++ compiler selections to clang/clang++ in a command or script."""

        if not text:
            return text

        rewritten = text
        # Custom configure flags (abcm2ps --CC=gcc) and env / make assignments.
        # Use (?![\w.+]) instead of \b so quoted forms like CC="gcc" still match.
        replacements = (
            (r"(--CC=)([\"']?)gcc\2(?![\w.+])", r"\1\2clang\2"),
            (r"(--CXX=)([\"']?)g\+\+\2(?![\w.+])", r"\1\2clang++\2"),
            (r"(?<![\w-])(CC=)([\"']?)gcc\2(?![\w.+])", r"\1\2clang\2"),
            (r"(?<![\w-])(CXX=)([\"']?)g\+\+\2(?![\w.+])", r"\1\2clang++\2"),
            (r"(-DCMAKE_C_COMPILER=)([\"']?)gcc\2(?![\w.+])", r"\1\2clang\2"),
            (r"(-DCMAKE_CXX_COMPILER=)([\"']?)g\+\+\2(?![\w.+])", r"\1\2clang++\2"),
            (r"(?<![\w-])(CMAKE_C_COMPILER=)([\"']?)gcc\2(?![\w.+])", r"\1\2clang\2"),
            (r"(?<![\w-])(CMAKE_CXX_COMPILER=)([\"']?)g\+\+\2(?![\w.+])", r"\1\2clang++\2"),
        )
        for pattern, repl in replacements:
            rewritten = re.sub(pattern, repl, rewritten)
        return rewritten

    def _strip_asan_flag_tokens(self, flags: str) -> str:
        """Remove ASan-related tokens from a flag string while keeping other flags."""

        cleaned = re.sub(
            r"(?:^|\s)(?:-fsanitize=address|-shared-libasan|-fno-omit-frame-pointer)\b",
            " ",
            flags or "",
        )
        return re.sub(r"\s+", " ", cleaned).strip()

    def _strip_global_asan_env_exports(self, script: str) -> str:
        """Remove ASan from exported CFLAGS/CXXFLAGS/LDFLAGS for cmake builds.

        Host compilers (gcc) and ExternalProject configure tests inherit these
        env vars; -shared-libasan is clang-only and breaks those steps.
        """

        if not script:
            return script

        pattern = re.compile(
            r'(?m)^(?P<prefix>\s*)export\s+(?P<var>CFLAGS|CXXFLAGS|LDFLAGS)=(?P<quote>["\'])'
            r"(?P<value>.*?)(?P=quote)(?P<suffix>\s*)$"
        )

        def _replace(match: re.Match[str]) -> str:
            cleaned = self._strip_asan_flag_tokens(match.group("value"))
            return (
                f"{match.group('prefix')}export {match.group('var')}="
                f"{match.group('quote')}{cleaned}{match.group('quote')}{match.group('suffix')}"
            )

        return pattern.sub(_replace, script)

    def _ensure_cmake_asan_flags_in_script(self, script: str) -> str:
        """Ensure cmake configure lines carry CMAKE_* ASan flags."""

        if not script or "cmake" not in script:
            return script

        asan_blob = " " + " ".join(self._cmake_asan_flag_args())
        pattern = re.compile(r"(?m)^(?P<prefix>\s*)cmake(?!\s+--build)(?P<rest>[^\n]*)$")

        def _replace(match: re.Match[str]) -> str:
            rest = match.group("rest") or ""
            if "-fsanitize=address" in rest or "CMAKE_SHARED_LINKER_FLAGS" in rest:
                return match.group(0)
            # Only inject into configure-like lines ( -S/-B / cmake . / -D... ).
            if not re.search(r"(?:-S\b|-B\b|\s\.(?:\s|$)|-D\w)", rest):
                return match.group(0)
            return f"{match.group('prefix')}cmake{rest.rstrip()}{asan_blob}"

        return pattern.sub(_replace, script)

    def _cmake_asan_flag_args(self) -> list[str]:
        """CMake args that force the canonical ASan policy into all link kinds."""

        flags = self._default_sanitizer_flags()
        compile_flags = f"-g -O0 {flags}"
        link_flags = flags
        return [
            f'-DCMAKE_C_FLAGS="{compile_flags}"',
            f'-DCMAKE_CXX_FLAGS="{compile_flags}"',
            f'-DCMAKE_EXE_LINKER_FLAGS="{link_flags}"',
            f'-DCMAKE_SHARED_LINKER_FLAGS="{link_flags}"',
            f'-DCMAKE_MODULE_LINKER_FLAGS="{link_flags}"',
            "-DCMAKE_C_COMPILER=clang",
            "-DCMAKE_CXX_COMPILER=clang++",
        ]

    def _normalize_cmake_layout(self, repo_path: Path, build_plan: BuildPlan) -> BuildPlan:
        """Force out-of-tree CMake builds and keep clean before configure.

        The build.sh template runs configure → clean → build. For CMake, clean is
        typically `rm -rf build`, so cmake configure must live in build_commands.
        """

        is_cmake = build_plan.build_system == "cmake" or (repo_path / "CMakeLists.txt").is_file()
        if not is_cmake:
            return build_plan
        build_plan.build_system = "cmake"

        existing_commands = list(build_plan.configure_commands or []) + list(build_plan.build_commands or [])
        if not any(self._is_cmake_command(command) for command in existing_commands):
            if build_plan.build_script_override:
                # Structured commands are unused; script override is rewritten separately.
                return build_plan
            existing_commands = [
                "cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=OFF",
                "cmake --build build -j$(nproc)",
            ]
        define_flags = self._extract_cmake_define_flags(existing_commands)
        if not any(flag.startswith("-DBUILD_TESTING=") for flag in define_flags):
            define_flags.append("-DBUILD_TESTING=OFF")
        if self._repo_uses_legacy_libavif_api(repo_path):
            if not any(flag.startswith("-DCMAKE_DISABLE_FIND_PACKAGE_libavif=") for flag in define_flags):
                define_flags.append("-DCMAKE_DISABLE_FIND_PACKAGE_libavif=ON")
        flag_blob = (" " + " ".join(define_flags)) if define_flags else ""
        asan_blob = " " + " ".join(self._cmake_asan_flag_args())
        configure_cmd = f"cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug{flag_blob}{asan_blob}"
        build_cmd = "cmake --build build -j$(nproc)"

        build_plan.configure_commands = [
            command for command in (build_plan.configure_commands or []) if not self._is_cmake_command(command)
        ]
        non_cmake_build = [
            command for command in (build_plan.build_commands or []) if not self._is_cmake_command(command)
        ]
        build_plan.build_commands = [configure_cmd, build_cmd, *non_cmake_build]

        clean_commands = list(build_plan.clean_commands or [])
        if not any(re.search(r"\brm\s+-rf\s+(?:\./)?build\b", command or "") for command in clean_commands):
            clean_commands.insert(0, "rm -rf build")
        build_plan.clean_commands = clean_commands
        return build_plan

    def _rewrite_broken_cmake_in_script(self, script: str) -> str:
        """Rewrite common in-source / parent-dir cmake mistakes inside overrides."""

        if not script or "cmake" not in script:
            return script

        rewritten = script
        # `cmake ... ..` or `cmake ..` from source root.
        rewritten = re.sub(
            r"(?m)^(?P<prefix>\s*)cmake(?P<flags>[^\n]*?)\s+\.\.(?P<suffix>\s*)$",
            lambda match: (
                f"{match.group('prefix')}cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug"
                f"{self._cmake_flags_without_generator_path(match.group('flags'))}{match.group('suffix')}"
            ),
            rewritten,
        )
        # Bare in-tree configure: `cmake .` / `cmake -D... .`
        rewritten = re.sub(
            r"(?m)^(?P<prefix>\s*)cmake(?P<flags>[^\n]*?)\s+\.(?P<suffix>\s*)$",
            lambda match: (
                f"{match.group('prefix')}cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug"
                f"{self._cmake_flags_without_generator_path(match.group('flags'))}{match.group('suffix')}"
            ),
            rewritten,
        )
        rewritten = re.sub(
            r"(?m)^(?P<prefix>\s*)cmake\s+--build\s+\.(?P<rest>\s.*)?$",
            lambda match: f"{match.group('prefix')}cmake --build build{match.group('rest') or ''}",
            rewritten,
        )
        return rewritten

    def _cmake_flags_without_generator_path(self, flags: str) -> str:
        """Keep -D flags from a broken cmake line, drop redundant build-type duplicates."""

        kept = self._extract_cmake_define_flags([flags or ""])
        if not kept:
            return ""
        return " " + " ".join(flag for flag in kept if not flag.startswith("-DCMAKE_BUILD_TYPE="))

    def _soften_make_clean_in_script(self, script: str) -> str:
        """Rewrite bare `make clean` lines inside overrides to best-effort form."""

        if not script:
            return script
        pattern = re.compile(
            r"(?m)^(?P<prefix>\s*)(?P<cmd>make(?:\s+\S+)*\s+(?:clean|distclean))(?P<suffix>\s*)$"
        )

        def _replace(match: re.Match[str]) -> str:
            cmd = match.group("cmd").strip()
            if "||" in match.group(0):
                return match.group(0)
            return f"{match.group('prefix')}{cmd} || true{match.group('suffix')}"

        return pattern.sub(_replace, script)

    def _should_defer_asan_until_after_configure(
        self,
        repo_path: Path | None,
        build_plan: BuildPlan,
    ) -> bool:
        """True when configure must run without ASan, then compile with ASan."""

        if self._is_cmake_build(repo_path, build_plan):
            return False
        system = (build_plan.build_system or "").strip().lower()
        if system == "autotools":
            return True
        return bool(build_plan.configure_commands)

    def _makefile_source_text(self, repo_path: Path | None) -> str:
        if repo_path is None:
            return ""
        for name in ("Makefile", "makefile", "GNUmakefile"):
            path = repo_path / name
            if path.is_file():
                try:
                    return path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    return ""
        return ""

    def _makefile_var_assignment(self, repo_path: Path | None, name: str) -> str | None:
        """Return the RHS of a simple ``NAME=...`` Makefile assignment, if any."""

        text = self._makefile_source_text(repo_path)
        if not text:
            return None
        match = re.search(
            rf"(?m)^{re.escape(name)}\s*=\s*(.*)$",
            text,
        )
        if not match:
            return None
        return match.group(1).strip()

    def _makefile_hardcodes_cflags(self, repo_path: Path | None) -> bool:
        """True when Makefile assigns CFLAGS= (env export alone will not apply)."""

        text = self._makefile_source_text(repo_path)
        return bool(re.search(r"(?m)^CFLAGS\s*=", text))

    def _makefile_uses_mycflags(self, repo_path: Path | None) -> bool:
        return self._makefile_var_assignment(repo_path, "MYCFLAGS") is not None

    def _should_inject_make_asan_overrides(
        self,
        repo_path: Path | None,
        build_plan: BuildPlan,
    ) -> bool:
        """True when ASan must be passed on the ``make`` argv, not only via env."""

        if self._is_cmake_build(repo_path, build_plan):
            return False
        if self._should_defer_asan_until_after_configure(repo_path, build_plan):
            return True
        # Plain make projects (e.g. Lua) hardcode CC=/CFLAGS= and ignore exports.
        return self._makefile_uses_mycflags(repo_path) or self._makefile_hardcodes_cflags(
            repo_path
        )

    @staticmethod
    def _shell_single_quote(value: str) -> str:
        """Quote a string for safe inclusion in a POSIX shell command."""

        return "'" + value.replace("'", "'\"'\"'") + "'"

    def _make_asan_override_assignments(self, repo_path: Path | None = None) -> str:
        """Shell assignments that force ASan onto `make` despite Makefile CFLAGS=."""

        asan = self._default_sanitizer_flags()
        asan_ld = self._default_sanitizer_link_flags()
        cflags = f"-g -O0 {asan}"
        # Lua-style: CFLAGS embeds $(MYCFLAGS); override MYCFLAGS/MYLDFLAGS + CC so
        # platform defines survive and -shared-libasan uses clang.
        myc = self._makefile_var_assignment(repo_path, "MYCFLAGS")
        if myc is not None:
            myld = self._makefile_var_assignment(repo_path, "MYLDFLAGS") or ""
            myc_q = self._shell_single_quote(f"{myc} {cflags}".strip())
            myld_q = self._shell_single_quote(f"{myld} {asan_ld}".strip())
            return f"CC=clang MYCFLAGS={myc_q} MYLDFLAGS={myld_q}"
        return (
            f'CC=clang CFLAGS="{cflags}" '
            f'CXXFLAGS="{cflags}" '
            f'LDFLAGS="{asan_ld}"'
        )

    def _inject_make_asan_flag_overrides(
        self,
        build_commands: list[str],
        repo_path: Path | None = None,
    ) -> list[str]:
        """Rewrite bare ``make ...`` build lines to pass ASan via make variables."""

        assignments = self._make_asan_override_assignments(repo_path)
        rewritten: list[str] = []
        for command in build_commands or []:
            stripped = (command or "").strip()
            if not stripped:
                continue
            if re.search(r"\bmake\b", stripped) and not re.search(
                r"\bmake\b.*\b(clean|distclean|mostlyclean)\b",
                stripped,
            ):
                if re.search(r"\b(CFLAGS|CXXFLAGS|LDFLAGS|MYCFLAGS)=", stripped):
                    rewritten.append(stripped)
                    continue
                rewritten.append(re.sub(r"\bmake\b", f"make {assignments}", stripped, count=1))
            else:
                rewritten.append(stripped)
        return rewritten

    def _inject_make_asan_flag_overrides_in_script(
        self,
        script: str,
        repo_path: Path | None = None,
    ) -> str:
        """Apply make ASan overrides to build lines inside a script override."""

        if not script or "make" not in script:
            return script
        assignments = self._make_asan_override_assignments(repo_path)
        pattern = re.compile(
            r"(?m)^(?P<pre>\s*)(?P<cmd>make\b(?![^\n]*\b(?:clean|distclean|mostlyclean)\b)[^\n]*)$"
        )

        def _replace(match: re.Match[str]) -> str:
            cmd = match.group("cmd")
            if re.search(r"\b(CFLAGS|CXXFLAGS|LDFLAGS|MYCFLAGS)=", cmd):
                return match.group(0)
            updated = re.sub(r"^make\b", f"make {assignments}", cmd, count=1)
            return f"{match.group('pre')}{updated}"

        return pattern.sub(_replace, script)

    def _demote_distclean_in_script(self, script: str) -> str:
        """Rewrite ``make ... distclean`` to ``make ... clean || true`` in overrides."""

        if not script or "distclean" not in script:
            return script
        pattern = re.compile(
            r"(?m)^(?P<prefix>\s*)(?P<cmd>make(?:\s+\S+)*)\s+distclean"
            r"(?:\s*\|\|\s*true)?(?P<suffix>\s*)$"
        )
        return pattern.sub(
            lambda match: (
                f"{match.group('prefix')}{match.group('cmd')} clean || true"
                f"{match.group('suffix')}"
            ),
            script,
        )

    def _defer_asan_env_until_after_configure(self, script: str) -> str:
        """Strip ASan from early env exports and re-apply after ./configure.

        Autotools configure tests compile and run a probe binary; with
        ``-shared-libasan`` that often fails as ``cannot run C compiled programs``.
        """

        if not script:
            return script
        if not re.search(r"(?m)^\s*(?:\./configure|autoreconf)\b", script):
            return script
        if not re.search(
            r"(?m)^export\s+(?:CFLAGS|CXXFLAGS|LDFLAGS)=.*-fsanitize=address",
            script,
        ):
            return script
        if "applying sanitizer flags for build" in script:
            return script

        script = self._strip_global_asan_env_exports(script)
        asan = self._default_sanitizer_flags()
        asan_ld = self._default_sanitizer_link_flags()
        block = (
            'log "applying sanitizer flags for build"\n'
            f'export CFLAGS="-g -O0 {asan}"\n'
            f'export CXXFLAGS="-g -O0 {asan}"\n'
            f'export LDFLAGS="{asan_ld}"\n'
        )
        matches = list(
            re.finditer(r"(?m)^(?P<line>\s*(?:\./configure|autoreconf)\b[^\n]*)\n?", script)
        )
        if not matches:
            return script
        insert_at = matches[-1].end()
        script = script[:insert_at] + block + script[insert_at:]
        return self._inject_make_asan_flag_overrides_in_script(script)

    def _is_valid_replan_candidate(
        self,
        previous_plan: BuildPlan,
        candidate_plan: BuildPlan,
        failure_kind: str = "",
        failure_logs: str = "",
    ) -> bool:
        if candidate_plan.model_dump(mode="json") == previous_plan.model_dump(mode="json"):
            return False
        normalized_failure_kind = (failure_kind or "").strip().lower()
        failure_logs_lower = (failure_logs or "").lower()
        if "no rule to make target" in failure_logs_lower:
            changed_commands = previous_plan.build_commands != candidate_plan.build_commands
            changed_configure = previous_plan.configure_commands != candidate_plan.configure_commands
            changed_clean = previous_plan.clean_commands != candidate_plan.clean_commands
            changed_script = bool(candidate_plan.build_script_override) and (
                candidate_plan.build_script_override != previous_plan.build_script_override
            )
            if not (changed_commands or changed_configure or changed_clean or changed_script):
                return False
        if normalized_failure_kind == "docker_build":
            if not candidate_plan.dockerfile_override:
                return False
            return True
        if normalized_failure_kind == "container_run":
            if candidate_plan.build_script_override or candidate_plan.dockerfile_override:
                return True
            return self._changes_build_execution_surface(previous_plan, candidate_plan)
        if candidate_plan.dockerfile_override or candidate_plan.build_script_override:
            return True
        return self._changes_build_execution_surface(previous_plan, candidate_plan)

    def _changes_build_execution_surface(self, previous_plan: BuildPlan, candidate_plan: BuildPlan) -> bool:
        return any(
            [
                previous_plan.install_packages != candidate_plan.install_packages,
                previous_plan.configure_commands != candidate_plan.configure_commands,
                previous_plan.clean_commands != candidate_plan.clean_commands,
                previous_plan.build_commands != candidate_plan.build_commands,
                previous_plan.build_system != candidate_plan.build_system,
                previous_plan.chosen_vulnerable_ref != candidate_plan.chosen_vulnerable_ref,
                previous_plan.chosen_fixed_ref != candidate_plan.chosen_fixed_ref,
            ]
        )

    def _execute_build_plan(
        self,
        repo_path: Path,
        paths: BuildStagePaths,
        plan_meta: dict,
        build_plan: BuildPlan,
        resolved_ref: str,
    ) -> BuildArtifact:
        repo_scan = self._scan_repo(repo_path)
        docker_context = {
            "repo_url": plan_meta["repo_url"],
            "vulnerable_ref": build_plan.chosen_vulnerable_ref,
            "project_name": plan_meta["project_name"],
            "project_dir_name": plan_meta["project_dir_name"],
            "project_dir": f"/src/{plan_meta['project_dir_name']}",
            "base_image": build_plan.base_image or self.DEFAULT_BASE_IMAGE,
            "workspace_root": "/workspace",
            "artifacts_root": "/workspace/artifacts",
            "build_artifacts_dir": "/workspace/artifacts/build",
            "poc_artifacts_dir": "/workspace/artifacts/poc",
            "verify_artifacts_dir": "/workspace/artifacts/verify",
            "apt_packages": build_plan.install_packages,
        }
        script_context = {
            "project_name": plan_meta["project_name"],
            "project_dir_name": plan_meta["project_dir_name"],
            "project_dir": f"/src/{plan_meta['project_dir_name']}",
            "workspace_root": "/workspace",
            "artifacts_root": "/workspace/artifacts",
            "build_artifacts_dir": "/workspace/artifacts/build",
            "build_system": build_plan.build_system or "unknown",
            "cc": self._select_compiler(build_plan),
            "cxx": self._select_cxx(build_plan),
            "sanitizer_flags": self._default_sanitizer_flags(repo_path),
            "configure_commands": build_plan.configure_commands,
            "clean_commands": build_plan.clean_commands,
            "build_commands": build_plan.build_commands,
        }

        dockerfile_content = (
            build_plan.dockerfile_override.rstrip() + "\n"
            if build_plan.dockerfile_override
            else self._render_template("Dockerfile.j2", docker_context)
        )
        # LLM overrides often keep only FROM+apt when bumping the base image and
        # drop git clone; inject the standard checkout scaffolding when missing.
        dockerfile_content = self._ensure_dockerfile_clones_project(
            dockerfile_content,
            repo_url=str(docker_context.get("repo_url") or ""),
            vulnerable_ref=str(
                docker_context.get("vulnerable_ref") or build_plan.chosen_vulnerable_ref or ""
            ),
            project_dir=str(docker_context.get("project_dir") or f"/src/{plan_meta['project_dir_name']}"),
            workspace_root=str(docker_context.get("workspace_root") or "/workspace"),
            artifacts_root=str(docker_context.get("artifacts_root") or "/workspace/artifacts"),
            build_artifacts_dir=str(
                docker_context.get("build_artifacts_dir") or "/workspace/artifacts/build"
            ),
            poc_artifacts_dir=str(
                docker_context.get("poc_artifacts_dir") or "/workspace/artifacts/poc"
            ),
            verify_artifacts_dir=str(
                docker_context.get("verify_artifacts_dir") or "/workspace/artifacts/verify"
            ),
        )
        build_script_content = (
            build_plan.build_script_override.rstrip() + "\n"
            if build_plan.build_script_override
            else self._render_template("build.sh.j2", script_context)
        )
        build_script_content = self._ensure_shared_libasan_in_build_script(
            build_script_content,
            repo_path,
        )
        if self._is_cmake_build(repo_path, build_plan):
            build_script_content = self._ensure_cmake_asan_flags_in_script(build_script_content)
            build_script_content = self._strip_global_asan_env_exports(build_script_content)
        else:
            if build_plan.build_script_override:
                build_script_content = self._demote_distclean_in_script(build_script_content)
                build_script_content = self._defer_asan_env_until_after_configure(build_script_content)
            if self._should_inject_make_asan_overrides(repo_path, build_plan):
                build_script_content = self._inject_make_asan_flag_overrides_in_script(
                    build_script_content,
                    repo_path,
                )
        self.file_tool.write_text(str(paths.dockerfile), dockerfile_content)
        self.file_tool.write_text(str(paths.build_script), build_script_content)

        workspace_root = str(paths.workspace_root.resolve())
        docker_proxy = self._get_docker_build_proxy()
        docker_build_result = self.docker_tool.build_image(
            DockerBuildRequest(
                workspace=workspace_root,
                dockerfile_path=str(paths.dockerfile.resolve()),
                image_tag=plan_meta["docker_image_tag"],
                build_args=self._build_docker_proxy_args(docker_proxy),
                network_mode=self._select_docker_build_network_mode(docker_proxy),
            )
        )
        compiled_image_tag: Optional[str] = None
        if docker_build_result.success:
            build_result = self.docker_tool.run_container(
                DockerRunRequest(
                    image_tag=plan_meta["docker_image_tag"],
                    workspace=workspace_root,
                    command=["bash", "/workspace/artifacts/build/build.sh"],
                    container_name=plan_meta["build_container_name"],
                    remove=False,
                )
            )
            if build_result.success:
                commit_result = self.docker_tool.commit_container(
                    plan_meta["build_container_name"],
                    plan_meta["compiled_image_tag"],
                )
                if commit_result.success:
                    compiled_image_tag = plan_meta["compiled_image_tag"]
                else:
                    build_result = DockerCommandResult(
                        success=False,
                        exit_code=commit_result.exit_code,
                        stdout=build_result.stdout,
                        stderr=(build_result.stderr + "\n" + commit_result.stderr).strip(),
                    )
            self.docker_tool.remove_container(plan_meta["build_container_name"])
        else:
            build_result = docker_build_result

        build_logs = self._compose_build_logs(
            docker_build_result=docker_build_result,
            run_result=build_result if docker_build_result.success else None,
        )
        self.file_tool.write_text(str(paths.build_log), build_logs)

        return BuildArtifact(
            dockerfile_content=dockerfile_content,
            build_script_content=build_script_content,
            install_packages=build_plan.install_packages,
            build_commands=build_plan.build_commands,
            expected_binary_path=build_plan.expected_binary_path,
            repo_local_path=str(repo_path),
            resolved_ref=resolved_ref,
            build_system=build_plan.build_system,
            detected_build_files=repo_scan["build_files"],
            dependency_sources=self._build_dependency_sources(repo_scan),
            source_of_truth=build_plan.source_of_truth,
            binary_or_entrypoint=build_plan.expected_binary_path,
            docker_image_tag=plan_meta["docker_image_tag"],
            compiled_image_tag=compiled_image_tag,
            sanitizer_enabled=self._sanitizer_enabled(build_script_content),
            build_success=docker_build_result.success and build_result.success,
            build_logs=build_logs,
            chosen_vulnerable_ref=build_plan.chosen_vulnerable_ref,
            chosen_fixed_ref=build_plan.chosen_fixed_ref,
        )

    def _candidate_refs(self, repo_path: Path, knowledge: KnowledgeModel) -> dict[str, str]:
        refs: dict[str, str] = {}
        if knowledge.vulnerable_ref:
            refs["knowledge_vulnerable"] = knowledge.vulnerable_ref
        if knowledge.fixed_ref:
            refs["knowledge_fixed"] = knowledge.fixed_ref
            fixed_parent = self._maybe_resolve_ref(repo_path, f"{knowledge.fixed_ref}^")
            if fixed_parent:
                refs["fixed_parent"] = fixed_parent
        return refs

    def _collect_ref_snapshot(
        self,
        repo_path: Path,
        label: str,
        requested_ref: str,
        affected_files: list[str],
    ) -> Optional[RefSnapshot]:
        try:
            checkout = self.git_tool.checkout_ref(str(repo_path), requested_ref)
        except Exception:
            return None

        repo_scan = self._scan_repo(repo_path)
        key_files = self._choose_key_files(repo_scan, affected_files)
        excerpts = [self._read_excerpt(repo_path / rel_path) for rel_path in key_files]
        excerpts = [block for block in excerpts if block]
        return RefSnapshot(
            label=label,
            requested_ref=requested_ref,
            resolved_ref=checkout.current_ref,
            build_files=repo_scan["build_files"],
            evidence_files=repo_scan["evidence_files"],
            ci_files=repo_scan["ci_files"],
            file_excerpts=excerpts,
        )

    def _scan_repo(self, repo_dir: Path) -> dict[str, list[str]]:
        build_files: list[str] = []
        evidence_files: list[str] = []
        ci_files: list[str] = []

        for pattern in self.BUILD_FILE_PATTERNS:
            for path in repo_dir.rglob(pattern):
                if path.is_file():
                    build_files.append(str(path.relative_to(repo_dir)))
        for pattern in self.README_PATTERNS:
            for path in repo_dir.rglob(pattern):
                if path.is_file():
                    evidence_files.append(str(path.relative_to(repo_dir)))

        workflow_dir = repo_dir / ".github" / "workflows"
        if workflow_dir.exists():
            for path in workflow_dir.rglob("*"):
                if path.is_file():
                    ci_files.append(str(path.relative_to(repo_dir)))
        gitlab_ci = repo_dir / ".gitlab-ci.yml"
        if gitlab_ci.exists():
            ci_files.append(str(gitlab_ci.relative_to(repo_dir)))

        return {
            "build_files": sorted(set(build_files)),
            "evidence_files": sorted(set(evidence_files)),
            "ci_files": sorted(set(ci_files)),
        }

    def _choose_key_files(self, repo_scan: dict[str, list[str]], affected_files: list[str]) -> list[str]:
        selected: list[str] = []
        for rel_path in affected_files:
            if rel_path not in selected:
                selected.append(rel_path)
        for group in ("build_files", "evidence_files", "ci_files"):
            for rel_path in repo_scan[group]:
                if rel_path not in selected:
                    selected.append(rel_path)
                if len(selected) >= 8:
                    return selected
        return selected[:8]

    def _read_excerpt(self, path: Path, limit: int = 1600) -> str:
        if not path.exists() or not path.is_file():
            return ""
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""
        header = f"FILE: {path.name}\nPATH: {path}\n"
        return header + content[:limit]

    def _try_llm_build_plan(
        self,
        knowledge: KnowledgeModel,
        context: BuildContext,
        project_name: str,
        previous_plan: Optional[BuildPlan] = None,
    ) -> Optional[BuildPlan]:
        try:
            model = build_chat_model("build_agent", temperature=0)
        except Exception:
            return None

        prompt = self._build_llm_prompt(knowledge=knowledge, context=context, project_name=project_name, previous_plan=previous_plan)
        try:
            response = model.invoke(
                [
                    SystemMessage(content="You return strict JSON only."),
                    HumanMessage(content=prompt),
                ]
            )
            parsed = parse_llm_json_payload(getattr(response, "content", response))
            if parsed is None:
                return None
            plan = BuildPlan(**parsed)
            if not plan.build_commands:
                return None
            return plan
        except Exception:
            return None

    def _build_llm_prompt(
        self,
        knowledge: KnowledgeModel,
        context: BuildContext,
        project_name: str,
        previous_plan: Optional[BuildPlan],
    ) -> str:
        sections = [
            "You are the Build Agent for a vulnerability reproduction framework.",
            "Your job is to choose the best vulnerable build target and produce a concrete build plan from local repository evidence.",
            "Prefer real repo evidence over hints. Use fixed_ref^ when it is a better vulnerable baseline than knowledge.vulnerable_ref.",
            "Read the patch, affected files, Makefile/CMakeLists/README/CI excerpts, and choose a plan that has the best chance to compile.",
            "When replanning after failure, distinguish between docker image build failures and container runtime build failures.",
            "If previous_failure_kind is docker_build, prioritize fixing Dockerfile steps, install_packages, base image choice, and dockerfile_override.",
            "If previous_failure_kind is container_run, prioritize fixing build_commands, configure_commands, clean_commands, compiler choice, and build_script_override.",
            "When a previous attempt is provided, treat the failure log, rendered Dockerfile, and rendered build.sh as the primary debugging evidence.",
            "Do not repeat the same failing plan. Return a materially updated plan that explains how it addresses the observed failure.",
            "When the failure points to missing packages, compiler mismatch, environment setup, or checkout/build script issues, prefer changing install_packages and/or emitting dockerfile_override/build_script_override.",
            "For make-based projects, only use make targets that exist in the repository Makefile. Prefer `make -j$(nproc)` when unsure; never invent platform targets such as `make linux` unless that target is present.",
            "If the failure log contains `No rule to make target`, you must change build_commands (or build_script_override) to a valid target such as plain `make -j$(nproc)`.",
            f"Default base image is {self.DEFAULT_BASE_IMAGE}. Do not install packages unavailable on that release (for example libavif-dev on Ubuntu 20.04) unless you also change the base image in dockerfile_override.",
            f"If CMakeLists requires ECM/Qt/KF newer than Ubuntu 20.04 provides (ECM>{self.UBUNTU_FOCAL_MAX_ECM[0]}.{self.UBUNTU_FOCAL_MAX_ECM[1]}, Qt>=5.15), set base_image/dockerfile_override to {self.MODERN_UBUNTU_BASE_IMAGE}.",
            "Always compile/link C/C++ with AddressSanitizer using -fsanitize=address -shared-libasan (never static-only ASan).",
            "Because -shared-libasan is clang-only, always use clang/clang++ (export CC/CXX and any --CC=/CMAKE_*_COMPILER). Never set --CC=gcc or CC=gcc when ASan shared-libasan is enabled.",
            "Always keep sanitizer builds non-PIE: add -fno-pie to CFLAGS/CXXFLAGS and -no-pie to LDFLAGS (for CMake: -fno-pie to CMAKE_C_FLAGS/CMAKE_CXX_FLAGS, -no-pie to CMAKE_EXE_LINKER_FLAGS). Never put -no-pie into CFLAGS (clang errors 'unused argument during compilation' under -Werror). On WSL2 (vm.mmap_rnd_bits=32) a PIE ASan binary can bare-crash (exit 139) during sanitizer startup.",
            "ASan injection is build-system specific: for make/autotools export sanitizer via CFLAGS/CXXFLAGS/LDFLAGS; for CMake do NOT put sanitizer in global CFLAGS/LDFLAGS (it breaks host tools / ExternalProject). Instead set CMAKE_C_FLAGS/CMAKE_CXX_FLAGS and CMAKE_EXE/SHARED/MODULE_LINKER_FLAGS.",
            "When reproducing an OSS-Fuzz/ClusterFuzz testcase whose filename encodes a harness (e.g. flb-it-fuzz-parser_fuzzer) and that harness exists in-tree, build that harness binary (e.g. -DFLB_TESTS_INTERNAL_FUZZ=On) and set expected_binary_path to it. Build only that target (`cmake --build build --target <harness>`); do not build every fuzzer. Do not force a missing harness name such as secilc-fuzzer over an existing entrypoint like secilc.",
            "If an optional plugin (for example AVIF) is API-incompatible with distro headers, omit that package and disable the find_package instead of failing the whole build.",
            "For CMake projects, use out-of-tree builds: `cmake -S . -B build ...` then `cmake --build build -j$(nproc)`. Never run `cmake ..` from the source root.",
            "Any Dockerfile used for build execution must preserve the ability to clone the target repository inside the image.",
            "If you set dockerfile_override, it must include git clone + git checkout of the vulnerable ref into /src/<project> (or leave dockerfile_override null and only change base_image/install_packages). Apt-only overrides are incomplete.",
            "Do not remove required base packages such as git from install_packages or dockerfile_override.",
            "Return exactly one JSON object and no markdown fences.",
            "Schema:",
            json.dumps(
                {
                    "chosen_vulnerable_ref": "string",
                    "chosen_fixed_ref": "string or null",
                    "build_system": "string",
                    "install_packages": ["string"],
                    "configure_commands": ["string"],
                    "clean_commands": ["string"],
                    "build_commands": ["string"],
                    "expected_binary_path": "string or null",
                    "base_image": "string or null",
                    "dockerfile_override": "string or null",
                    "build_script_override": "string or null",
                    "source_of_truth": "string",
                    "confidence": "low|medium|high",
                    "rationale": "string",
                },
                ensure_ascii=True,
            ),
        ]

        if previous_plan is not None:
            sections.extend(
                [
                    "",
                    "Previous attempt debugging evidence:",
                    f"Previous failure kind: {context.previous_failure_kind or '<empty>'}",
                    "Previous failure logs:",
                    context.previous_build_failure or "<empty>",
                    "Previous plan:",
                    yaml.safe_dump(previous_plan.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
                    "Previously rendered Dockerfile:",
                    context.previous_dockerfile_content or "<empty>",
                    "Previously rendered build.sh:",
                    context.previous_build_script_content or "<empty>",
                    "Rewrite the plan using these concrete artifacts. If a script or Dockerfile edit is needed, use dockerfile_override and/or build_script_override instead of describing the change abstractly.",
                ]
            )

        sections.extend(
            [
                "",
                f"CVE: {knowledge.cve_id}",
                f"Repository: {knowledge.repo_url or ''}",
                f"Knowledge summary: {knowledge.summary}",
                f"Knowledge vulnerable_ref: {knowledge.vulnerable_ref or ''}",
                f"Knowledge fixed_ref: {knowledge.fixed_ref or ''}",
                f"Knowledge build_hints: {json.dumps(knowledge.build_hints, ensure_ascii=False)}",
                f"Knowledge reproduction_hints: {json.dumps(knowledge.reproduction_hints, ensure_ascii=False)}",
                f"Patch affected files: {json.dumps(context.patch_affected_files, ensure_ascii=False)}",
                "Patch excerpt:",
                context.patch_diff_excerpt or "<empty>",
            ]
        )

        for snapshot in context.snapshots:
            sections.extend(
                [
                    "",
                    f"Snapshot label: {snapshot.label}",
                    f"Requested ref: {snapshot.requested_ref}",
                    f"Resolved ref: {snapshot.resolved_ref}",
                    f"Build files: {json.dumps(snapshot.build_files, ensure_ascii=False)}",
                    f"Evidence files: {json.dumps(snapshot.evidence_files, ensure_ascii=False)}",
                    f"CI files: {json.dumps(snapshot.ci_files, ensure_ascii=False)}",
                    "Excerpts:",
                    "\n\n---\n\n".join(snapshot.file_excerpts[:6]) or "<empty>",
                ]
            )
        return "\n".join(sections)

    def _build_llm_retry_prompt(
        self,
        context: BuildContext,
        previous_plan: BuildPlan,
    ) -> str:
        failure_summary = self._summarize_build_failure(context.previous_build_failure)
        return "\n".join(
            [
                "Revise your previous build plan using the new failure evidence below.",
                "Keep the same JSON schema and return strict JSON only.",
                "Do not repeat the same failing plan.",
                "",
                f"Previous failure kind: {context.previous_failure_kind or '<empty>'}",
                "Failure summary:",
                failure_summary,
                "Previous failure logs:",
                self._truncate_tail(context.previous_build_failure or "<empty>", 2400),
                "Previously rendered Dockerfile:",
                self._truncate_text(context.previous_dockerfile_content or "<empty>", 2400),
                "Previously rendered build.sh:",
                self._truncate_text(context.previous_build_script_content or "<empty>", 2600),
                "Previous plan for reference:",
                yaml.safe_dump(previous_plan.model_dump(mode='json'), sort_keys=False, allow_unicode=True),
            ]
        )

    def _summarize_build_failure(self, build_logs: str) -> str:
        value = (build_logs or "").strip()
        if not value:
            return "<empty>"
        lines = [line.rstrip() for line in value.splitlines() if line.strip()]
        error_like = [
            line for line in lines
            if any(token in line.lower() for token in ("error", "fatal", "no rule", "failed", "not found", "undefined reference"))
        ]
        summary_lines = error_like[-5:] if error_like else lines[-5:]
        return "\n".join(summary_lines)

    def _truncate_text(self, text: str, limit: int) -> str:
        value = text or ""
        if len(value) <= limit:
            return value
        if limit <= 32:
            return value[:limit]
        omitted = len(value) - limit
        return f"{value[: limit - 24]}\n...[truncated {omitted} chars]"

    def _truncate_tail(self, text: str, limit: int) -> str:
        value = text or ""
        if len(value) <= limit:
            return value
        if limit <= 32:
            return value[-limit:]
        omitted = len(value) - limit
        return f"...[truncated {omitted} chars]\n{value[-(limit - 24):]}"

    def _heuristic_build_plan(self, knowledge: KnowledgeModel, context: BuildContext, project_name: str) -> BuildPlan:
        fallback_spec = self._build_fallback_spec(
            knowledge=knowledge,
            context=context,
            project_name=project_name,
        )
        return BuildPlan(
            chosen_vulnerable_ref=fallback_spec.chosen_vulnerable_ref,
            chosen_fixed_ref=fallback_spec.chosen_fixed_ref,
            build_system=fallback_spec.build_system,
            install_packages=fallback_spec.install_packages,
            configure_commands=fallback_spec.configure_commands,
            clean_commands=fallback_spec.clean_commands,
            build_commands=fallback_spec.build_commands,
            expected_binary_path=fallback_spec.expected_binary_path,
            dockerfile_override=None,
            build_script_override=None,
            source_of_truth=fallback_spec.source_of_truth,
            confidence=fallback_spec.confidence,
            rationale=fallback_spec.rationale,
        )

    def _build_fallback_spec(
        self,
        knowledge: KnowledgeModel,
        context: BuildContext,
        project_name: str,
    ) -> BuildFallbackSpec:
        """Assemble all heuristic build decisions in one place."""

        chosen_snapshot = self._choose_fallback_snapshot(context)
        detected_files = chosen_snapshot.build_files if chosen_snapshot else []
        build_system = self._select_build_system(knowledge, detected_files)
        return BuildFallbackSpec(
            chosen_vulnerable_ref=self._select_fallback_vulnerable_ref(knowledge, context, chosen_snapshot),
            chosen_fixed_ref=knowledge.fixed_ref or context.task_fixed_ref,
            build_system=build_system,
            install_packages=self._select_install_packages(build_system, knowledge),
            configure_commands=self._select_configure_commands(build_system),
            clean_commands=self._select_clean_commands(build_system),
            build_commands=self._select_build_commands(knowledge, build_system),
            expected_binary_path=self._guess_binary_or_entrypoint(build_system, project_name),
            source_of_truth="repo_scan" if chosen_snapshot and chosen_snapshot.build_files else "knowledge_hint",
            confidence="medium",
            rationale="Heuristic fallback plan based on repo scan, patch-affected files, and knowledge hints.",
        )

    def _choose_fallback_snapshot(self, context: BuildContext) -> Optional[RefSnapshot]:
        """Choose the best local snapshot for fallback planning."""

        snapshots = {item.label: item for item in context.snapshots}
        return snapshots.get("fixed_parent") or snapshots.get("knowledge_vulnerable") or next(iter(snapshots.values()), None)

    def _select_fallback_vulnerable_ref(
        self,
        knowledge: KnowledgeModel,
        context: BuildContext,
        chosen_snapshot: Optional[RefSnapshot],
    ) -> str:
        """Resolve the vulnerable ref used by the deterministic fallback planner."""

        return (
            (chosen_snapshot.resolved_ref if chosen_snapshot else None)
            or knowledge.vulnerable_ref
            or context.task_vulnerable_ref
            or ""
        )

    def _select_build_system(self, knowledge: KnowledgeModel, detected_files: list[str]) -> str:
        lowered = {item.lower() for item in detected_files}
        for filename, system in self._build_system_mapping():
            if filename.lower() in lowered or any(path.endswith(filename) for path in detected_files):
                return system
        if knowledge.build_systems:
            return knowledge.build_systems[0]
        return "unknown"

    def _select_build_commands(self, knowledge: KnowledgeModel, build_system: str) -> list[str]:
        if knowledge.build_commands:
            return list(knowledge.build_commands)
        defaults = self._default_build_commands()
        return defaults.get(build_system, defaults["unknown"])

    def _select_configure_commands(self, build_system: str) -> list[str]:
        if build_system == "autotools":
            return ["./configure"]
        return []

    def _select_clean_commands(self, build_system: str) -> list[str]:
        if build_system in {"make", "autotools"}:
            return ["make clean || true"]
        if build_system == "cargo":
            return ["cargo clean || true"]
        if build_system == "cmake":
            return ["rm -rf build"]
        return []

    def _select_install_packages(self, build_system: str, knowledge: KnowledgeModel) -> list[str]:
        defaults = self._default_install_packages()
        packages = list(defaults.get(build_system, defaults["unknown"]))
        packages = self._augment_install_packages_from_hints(packages, knowledge.install_commands)
        return self._ensure_required_docker_packages(sorted(set(packages)))

    def _build_system_mapping(self) -> list[tuple[str, str]]:
        """Canonical file-to-build-system mapping for fallback planning."""

        return [
            ("Cargo.toml", "cargo"),
            ("go.mod", "go"),
            ("CMakeLists.txt", "cmake"),
            ("meson.build", "meson"),
            ("configure.ac", "autotools"),
            ("configure", "autotools"),
            ("Makefile", "make"),
            ("makefile", "make"),
            ("pom.xml", "maven"),
            ("build.gradle", "gradle"),
            ("build.gradle.kts", "gradle"),
            ("package.json", "npm"),
        ]

    def _default_build_commands(self) -> dict[str, list[str]]:
        """Default build commands used by the deterministic fallback planner."""

        return {
            "cmake": ["cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug", "cmake --build build -j$(nproc)"],
            "make": ["make clean || true", "make -j$(nproc)"],
            "autotools": ["./configure", "make -j$(nproc)"],
            "cargo": ["cargo build"],
            "go": ["go build ./..."],
            "meson": ["meson setup build", "ninja -C build"],
            "maven": ["mvn package -DskipTests"],
            "gradle": ["./gradlew build -x test"],
            "npm": ["npm install", "npm run build"],
            "unknown": ['echo "[build] build_commands unresolved after repo scan; please confirm build entrypoint." >&2', "exit 2"],
        }

    def _makefile_paths(self, repo_path: Path) -> list[Path]:
        paths: list[Path] = []
        for name in ("Makefile", "makefile", "GNUmakefile"):
            candidate = repo_path / name
            if candidate.is_file():
                paths.append(candidate)
        return paths

    def _extract_makefile_targets(self, repo_path: Path) -> set[str]:
        """Parse explicit Makefile targets from the repository root."""

        targets: set[str] = set()
        target_re = re.compile(r"^([A-Za-z0-9_./+-]+)\s*:")
        phony_re = re.compile(r"^\.PHONY\s*:\s*(.+)$")
        for makefile in self._makefile_paths(repo_path):
            try:
                text = makefile.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for raw_line in text.splitlines():
                line = raw_line.split("#", 1)[0].rstrip()
                if not line or line.startswith("\t"):
                    continue
                phony_match = phony_re.match(line)
                if phony_match:
                    targets.update(token for token in phony_match.group(1).split() if token)
                    continue
                target_match = target_re.match(line)
                if not target_match:
                    continue
                name = target_match.group(1)
                if name.startswith("."):
                    continue
                targets.add(name)
        return targets

    def _make_targets_requested_by_command(self, command: str) -> list[str]:
        """Return explicit make targets requested by a shell command."""

        requested: list[str] = []
        option_takes_value = {
            "-C",
            "--directory",
            "-f",
            "--file",
            "--makefile",
            "-I",
            "--include-dir",
            "-o",
            "--old-file",
            "-W",
            "--what-if",
            "--new-file",
            "--assume-new",
        }
        for segment in re.split(r"\s*(?:&&|\|\||;)\s*", command or ""):
            tokens = segment.split()
            if not tokens or tokens[0] != "make":
                continue
            skip_next = False
            for token in tokens[1:]:
                if skip_next:
                    skip_next = False
                    continue
                if token in option_takes_value:
                    skip_next = True
                    continue
                if token.startswith("-"):
                    continue
                if "=" in token:
                    continue
                requested.append(token)
        return requested

    def _sanitize_make_build_commands(
        self,
        build_commands: list[str],
        repo_path: Path,
        build_system: str,
    ) -> list[str]:
        """Replace make commands that reference nonexistent Makefile targets."""

        del build_system  # reserved for future system-specific policies
        if not build_commands:
            return build_commands

        targets = self._extract_makefile_targets(repo_path)
        if not targets:
            return build_commands

        default_make = "make -j$(nproc)"
        sanitized: list[str] = []
        for command in build_commands:
            requested = self._make_targets_requested_by_command(command)
            invalid = [
                target
                for target in requested
                if target not in targets and not self._is_best_effort_make_target(target)
            ]
            if invalid:
                sanitized.append(default_make)
            else:
                sanitized.append(command)
        return sanitized or list(self._default_build_commands().get("make", [default_make]))

    def _sanitize_make_script_override(self, script: str, repo_path: Path) -> str:
        """Rewrite invalid `make <target>` invocations inside a script override."""

        targets = self._extract_makefile_targets(repo_path)
        if not targets or not script:
            return script

        pattern = re.compile(r"(?m)^(?P<prefix>\s*)make(?P<flags>(?:\s+-\S+)*)\s+(?P<target>[A-Za-z0-9_./+-]+)(?P<suffix>\s.*)?$")

        def _replace(match: re.Match[str]) -> str:
            target = match.group("target")
            if target in targets or self._is_best_effort_make_target(target):
                return match.group(0)
            suffix = match.group("suffix") or ""
            return f"{match.group('prefix')}make -j$(nproc){suffix}"

        return pattern.sub(_replace, script)

    def _is_best_effort_make_target(self, target: str) -> bool:
        """Cleanup targets are allowed even when absent; they are softened separately."""

        return target in {"clean", "distclean", "mostlyclean"}

    def _default_install_packages(self) -> dict[str, list[str]]:
        """Default system packages used by the deterministic fallback planner."""

        return {
            "cmake": ["build-essential", "clang", "cmake", "git", "make", "pkg-config"],
            "make": ["build-essential", "clang", "git", "make", "pkg-config"],
            "autotools": ["autoconf", "automake", "build-essential", "clang", "git", "libtool", "make", "pkg-config"],
            "cargo": ["build-essential", "cargo", "clang", "git", "pkg-config", "rustc"],
            "go": ["build-essential", "clang", "git", "golang"],
            "meson": ["build-essential", "clang", "git", "meson", "ninja-build", "pkg-config"],
            "maven": ["git", "maven", "openjdk-17-jdk"],
            "gradle": ["git", "gradle", "openjdk-17-jdk"],
            "npm": ["git", "nodejs", "npm"],
            "unknown": ["build-essential", "clang", "git", "make", "pkg-config"],
        }

    def _augment_install_packages_from_hints(self, packages: list[str], install_commands: list[str]) -> list[str]:
        """Expand fallback package guesses from lightweight knowledge hints."""

        for command in install_commands:
            lower = command.lower()
            if "zlib" in lower and "zlib1g-dev" not in packages:
                packages.append("zlib1g-dev")
            if "openssl" in lower or "libssl" in lower:
                if "libssl-dev" not in packages:
                    packages.append("libssl-dev")
            if "pcre2" in lower and "libpcre2-dev" not in packages:
                packages.append("libpcre2-dev")
            elif re.search(r"\bpcre\b", lower) and "libpcre3-dev" not in packages and "libpcre2-dev" not in packages:
                packages.append("libpcre3-dev")
        return packages

    def _augment_install_packages_from_repo(self, packages: list[str], repo_path: Path) -> list[str]:
        """Expand packages from local repo evidence such as Makefile feature flags."""

        merged = list(packages)
        if self._repo_indicates_readline(repo_path) and "libreadline-dev" not in merged:
            merged.append("libreadline-dev")
        if self._repo_indicates_pcre2(repo_path):
            if "libpcre2-dev" not in merged:
                merged.append("libpcre2-dev")
        elif self._repo_indicates_pcre(repo_path) and "libpcre3-dev" not in merged:
            merged.append("libpcre3-dev")
        if self._repo_indicates_libaudit(repo_path) and "libaudit-dev" not in merged:
            merged.append("libaudit-dev")
        if self._repo_indicates_python3(repo_path) and "python3" not in merged:
            merged.append("python3")
        if self._repo_indicates_python3(repo_path) and "python3-distutils" not in merged:
            # Ubuntu 20.04+ python3 is slim; selinux pywrap clean imports distutils.
            merged.append("python3-distutils")
        return sorted(set(merged))

    def _repo_indicates_pcre(self, repo_path: Path) -> bool:
        """Detect default PCRE1 builds (pcre.h / -lpcre / USE_PCRE2 disabled)."""

        if self._repo_text_contains_any(
            repo_path,
            markers=("use_pcre2 ?= n", "use_pcre2=n", "use_pcre2 := n"),
            names=("Makefile", "makefile", "GNUmakefile"),
        ):
            return True
        return self._repo_text_contains_any(
            repo_path,
            markers=(
                "#include <pcre.h>",
                '#include "pcre.h"',
                "-lpcre ",
                "-lpcre\n",
                "-lpcre\t",
                "pcre_ldlibs ?= -lpcre",
                "pkg-config --libs libpcre",
                "pkg-config --cflags libpcre",
                "pcre_module ?= libpcre",
            ),
            names=("Makefile", "makefile", "GNUmakefile", "regex.h", "regex.c"),
        )

    def _repo_indicates_pcre2(self, repo_path: Path) -> bool:
        """Detect builds that enable PCRE2 by default (USE_PCRE2=y)."""

        # Do not treat optional #ifdef USE_PCRE2 / #include <pcre2.h> as enabled;
        # selinux defaults to USE_PCRE2 ?= n and still ships the PCRE2 code paths.
        return self._repo_text_contains_any(
            repo_path,
            markers=(
                "use_pcre2 ?= y",
                "use_pcre2=y",
                "use_pcre2 := y",
                "pcre_module ?= libpcre2",
                "pcre_ldlibs ?= -lpcre2",
            ),
            names=("Makefile", "makefile", "GNUmakefile"),
        )

    def _repo_indicates_libaudit(self, repo_path: Path) -> bool:
        """Detect libaudit usage (#include <libaudit.h> / -laudit)."""

        return self._repo_text_contains_any(
            repo_path,
            markers=("#include <libaudit.h>", "-laudit", "libaudit.h"),
            names=("Makefile", "makefile", "GNUmakefile", "seusers_local.c", "direct_api.c"),
        )

    def _repo_indicates_python3(self, repo_path: Path) -> bool:
        """Detect build/clean steps that invoke python3 (e.g. selinux pywrap)."""

        return self._repo_text_contains_any(
            repo_path,
            markers=("python3 setup.py", "python ?= python3", "clean-pywrap"),
            names=("Makefile", "makefile", "GNUmakefile"),
        )

    def _narrow_selinux_cil_build(self, repo_path: Path, build_plan: BuildPlan) -> BuildPlan:
        """For secilc/libsepol CVEs, build only those dirs instead of the full tree.

        Top-level `make` also builds libselinux/libsemanage/... which pull in PCRE,
        libaudit, python bindings, etc. CVE-2021-36084 only needs the CIL compiler
        against the in-tree libsepol.

        secilc includes <sepol/cil/cil.h>, which only appears under include/sepol/cil
        after libsepol's header install step. Mirror that with a symlink so we do not
        need DESTDIR install or xmlto (man pages).
        """

        if not (repo_path / "libsepol").is_dir() or not (repo_path / "secilc").is_dir():
            return build_plan
        expected = (build_plan.expected_binary_path or "").replace("\\", "/").lower()
        rationale = (build_plan.rationale or "").lower()
        wants_secilc = (
            "secilc" in expected
            or "secilc" in rationale
            or "cil" in rationale
            or "libsepol" in rationale
        )
        if not wants_secilc:
            return build_plan

        build_plan.build_commands = [
            "make -C libsepol -j$(nproc)",
            # Install layout puts cil/*.h under include/sepol/cil; expose in-tree.
            "mkdir -p libsepol/include/sepol",
            "ln -sfn ../../cil/include/cil libsepol/include/sepol/cil",
            (
                # Build only the secilc binary — skip man pages (xmlto) and extras.
                'make -C secilc secilc -j$(nproc) '
                'CFLAGS="${CFLAGS} -I../libsepol/include" '
                'LDFLAGS="${LDFLAGS} -L../libsepol/src"'
            ),
        ]
        build_plan.clean_commands = [
            "make -C libsepol clean || true",
            "make -C secilc clean || true",
            "rm -f libsepol/include/sepol/cil || true",
        ]
        # Entrypoint must match what we actually build. LLM plans often name
        # checkpolicy (or leave a stale path) even when the CVE is in CIL/secilc.
        build_plan.expected_binary_path = "secilc/secilc"
        # Full-tree-only packages are unnecessary once the build is narrowed.
        drop = {
            "libselinux1-dev",
            "libsemanage1-dev",
            "libsepol-dev",
            "libpcre3-dev",
            "libpcre2-dev",
            "libaudit-dev",
            "python3",
            "python3-distutils",
            "python3-dev",
            "xmlto",
        }
        build_plan.install_packages = [
            package
            for package in (build_plan.install_packages or [])
            if package.strip().lower() not in drop
        ]
        return build_plan

    def _prefer_ossfuzz_harness_target(
        self,
        repo_path: Path,
        build_plan: BuildPlan,
        knowledge: KnowledgeModel | None = None,
    ) -> BuildPlan:
        """When an OSS-Fuzz harness exists in-tree, build/run that instead of the main binary.

        Safety gate: only switch when harness name is parsed AND repo evidence exists.
        ClusterFuzz SELinux payloads name ``secilc-fuzzer`` but lack an in-tree harness
        binary — those keep ``secilc/secilc``.
        """

        candidates = self._collect_ossfuzz_harness_candidates(build_plan, knowledge)
        if not candidates:
            return build_plan

        for harness in candidates:
            if not ossfuzz_tools.harness_source_evidence(repo_path, harness):
                continue
            rel = ossfuzz_tools.preferred_harness_relpath(repo_path, harness)
            if not rel:
                continue
            build_plan = self._enable_ossfuzz_harness_build_options(repo_path, build_plan, harness)
            current = (build_plan.expected_binary_path or "").replace("\\", "/")
            if current != rel:
                build_plan.expected_binary_path = rel
                note = f"OSS-Fuzz harness target selected with in-tree evidence: {rel}"
                if note not in (build_plan.rationale or ""):
                    build_plan.rationale = f"{(build_plan.rationale or '').rstrip()} {note}".strip()
            return build_plan
        return build_plan

    def _collect_ossfuzz_harness_candidates(
        self,
        build_plan: BuildPlan,
        knowledge: KnowledgeModel | None = None,
    ) -> list[str]:
        texts: list[str] = []
        if build_plan.rationale:
            texts.append(build_plan.rationale)
        if build_plan.expected_binary_path:
            texts.append(build_plan.expected_binary_path)
        if knowledge is not None:
            for recipe in knowledge.reproduction_recipes or []:
                if recipe.source_title:
                    texts.append(recipe.source_title)
                if recipe.source_excerpt:
                    texts.append(recipe.source_excerpt)
            for hint in knowledge.reproduction_hints or []:
                texts.append(hint)
        found = ossfuzz_tools.extract_ossfuzz_harness_names(*texts)
        # LLM plans often set expected_binary_path to fuzz/<harness> without a
        # ClusterFuzz filename in the same string; accept that basename when it
        # looks like a fuzzer (still gated by harness_source_evidence later).
        bare = Path((build_plan.expected_binary_path or "").replace("\\", "/")).name
        if bare and ("fuzzer" in bare.lower() or bare.lower().endswith("_fuzz")):
            if bare not in found:
                found.insert(0, bare)
        return found

    def _enable_ossfuzz_harness_build_options(
        self,
        repo_path: Path,
        build_plan: BuildPlan,
        harness: str,
    ) -> BuildPlan:
        """Turn on harness compile options and narrow build to that target only.

        Full ``cmake --build`` / ``make all`` often compiles every fuzzer; unrelated
        harnesses may fail to link and abort the reproduction build.

        Standalone ``ossfuzz/<harness>.cpp`` trees (matio) are *not* cmake targets:
        keep the library build intact and append a gated link step instead of
        ``cmake --build --target <harness>``.

        qpdf-style ``fuzz/build.mk`` trees expose ``fuzz/build/<harness>`` make
        goals (not a bare ``<harness>`` target) and reject ``--enable-fuzzers``.
        """

        standalone = ossfuzz_tools.standalone_ossfuzz_harness_relpath(repo_path, harness)
        if standalone:
            return self._append_standalone_ossfuzz_harness_compile(
                build_plan,
                harness=harness,
                source_rel=standalone,
                repo_path=repo_path,
            )

        fuzz_mk_rel = ossfuzz_tools.in_tree_fuzz_mk_harness_relpath(repo_path, harness)
        if fuzz_mk_rel:
            build_plan = self._strip_unsupported_enable_fuzzers(repo_path, build_plan)
            return self._narrow_build_to_harness_target(build_plan, fuzz_mk_rel)

        fuzzers_cmake = repo_path / "tests" / "internal" / "fuzzers" / "CMakeLists.txt"
        if fuzzers_cmake.is_file():
            build_plan = self._inject_cmake_define(build_plan, "-DFLB_TESTS_INTERNAL_FUZZ=On")
            build_plan = self._inject_cmake_define(build_plan, "-DFLB_TESTS_INTERNAL=On")
        return self._narrow_build_to_harness_target(build_plan, harness)

    def _strip_unsupported_enable_fuzzers(
        self,
        repo_path: Path,
        build_plan: BuildPlan,
    ) -> BuildPlan:
        """Drop ``--enable-fuzzers`` when configure does not define that option.

        qpdf uses ``--enable-oss-fuzz``; LLM plans often invent ``--enable-fuzzers``.
        """

        if self._repo_declares_enable_fuzzers(repo_path):
            return build_plan

        def _strip_flag(command: str) -> str:
            cleaned = re.sub(r"(^|\s)--enable-fuzzers\b", " ", command or "")
            return re.sub(r"\s{2,}", " ", cleaned).strip()

        commands = list(build_plan.configure_commands or [])
        rewritten = [_strip_flag(cmd) for cmd in commands]
        if rewritten != commands:
            build_plan.configure_commands = rewritten

        if build_plan.build_script_override:
            script = build_plan.build_script_override
            stripped_script = re.sub(r"(^|\s)--enable-fuzzers\b", " ", script)
            stripped_script = re.sub(r"[ \t]{2,}", " ", stripped_script)
            if stripped_script != script:
                build_plan.build_script_override = stripped_script
        return build_plan

    @staticmethod
    def _repo_declares_enable_fuzzers(repo_path: Path) -> bool:
        for name in ("configure.ac", "configure.in", "configure"):
            path = repo_path / name
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if "enable-fuzzers" in text or "enable_fuzzers" in text:
                return True
        return False

    # CVE-2021-36977-style matio crashes need HDF5 1.12.0 (system libhdf5 is often
    # 1.10.x and rejects the ClusterFuzz seed before the vulnerable path runs).
    # Verify post-pass swaps to FIXED_* via the same build.sh rewritten in-place.
    VULN_HDF5_PREFIX = "/opt/hdf5-vuln"
    VULN_HDF5_MARKER = "deeprepro:vulnerable-hdf5-1.12.0"
    # Prefer the autotools release tarball (ships ./configure). The GitHub tag
    # archive is cmake-first and needs autogen.sh as a fallback.
    VULN_HDF5_TARBALL_URL = (
        "https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.0/src/hdf5-1.12.0.tar.gz"
    )
    VULN_HDF5_TARBALL_URL_FALLBACK = (
        "https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_12_0.tar.gz"
    )
    FIXED_HDF5_PREFIX = "/opt/hdf5-fixed"
    FIXED_HDF5_MARKER = "deeprepro:fixed-hdf5-1.12.1"
    # Prefer the release tarball (ships ./configure). The GitHub tag archive is
    # cmake-first and ConfigureChecks.pac_Cconftest fails on this image for 1.12.1.
    FIXED_HDF5_TARBALL_URL = (
        "https://github.com/HDFGroup/hdf5/releases/download/hdf5-1_12_1/hdf5-1.12.1.tar.gz"
    )
    FIXED_HDF5_TARBALL_URL_FALLBACK = (
        "https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_12_1.tar.gz"
    )
    # CVE-2021-36977 / OSV-2021-440: require specific crash signature, not bare
    # "AddressSanitizer" (allocation-size-too-big also matches that and survives 1.12.1).
    VULN_HDF5_CVE_STDERR_PATTERNS = ("heap-buffer-overflow", "H5MM_memcpy")
    VULN_HDF5_CVE_STACK_KEYWORDS = ("H5MM_memcpy", "H5MM_malloc")
    # Appear in HDF5-DIAG without an ASan overflow; never treat as the CVE hit.
    VULN_HDF5_WEAK_STACK_KEYWORDS = frozenset({"H5C_load_entry", "H5C__load_entry"})
    VULN_HDF5_CVE_CRASH_TYPE = "heap-buffer-overflow"
    VULN_HDF5_ASAN_OPTIONS = (
        "abort_on_error=1:symbolize=0:detect_leaks=0:allocator_may_return_null=1"
    )

    @classmethod
    def apply_vulnerable_hdf5_cve_match_policy(
        cls,
        *,
        expected_stderr_patterns: list[str] | None = None,
        expected_stack_keywords: list[str] | None = None,
        expected_crash_type: str = "",
        environment_variables: dict[str, str] | None = None,
    ) -> tuple[list[str], list[str], str, dict[str, str]]:
        """Tighten match policy for the vulnerable-HDF5 dependency gate.

        Returns (stderr_patterns, stack_keywords, crash_type, env).
        """

        stderr = list(cls.VULN_HDF5_CVE_STDERR_PATTERNS)
        stack = list(cls.VULN_HDF5_CVE_STACK_KEYWORDS)
        # Keep extra function names, but drop DIAG-only frames and YAML noise
        # (``affected:``, ``+id: OSV-...``) harvested from OSV pages.
        for item in expected_stack_keywords or []:
            token = (item or "").strip()
            if not token or token in stack:
                continue
            if token.lower() in {"addresssanitizer", "asan"}:
                continue
            if token in cls.VULN_HDF5_WEAK_STACK_KEYWORDS:
                continue
            if ":" in token:
                continue
            stack.append(token)
        env = dict(environment_variables or {})
        env["ASAN_OPTIONS"] = cls.VULN_HDF5_ASAN_OPTIONS
        return stderr, stack, cls.VULN_HDF5_CVE_CRASH_TYPE, env

    def _append_standalone_ossfuzz_harness_compile(
        self,
        build_plan: BuildPlan,
        *,
        harness: str,
        source_rel: str,
        repo_path: Path | None = None,
    ) -> BuildPlan:
        """After the library build, compile a standalone ossfuzz harness binary."""

        if not harness or not source_rel:
            return build_plan
        marker = f"-o {harness}"
        existing = "\n".join(build_plan.build_commands or [])
        if build_plan.build_script_override:
            existing = f"{existing}\n{build_plan.build_script_override}"
        if marker in existing and source_rel in existing:
            return build_plan

        use_vuln_hdf5 = self._standalone_ossfuzz_needs_vulnerable_hdf5(repo_path)
        if use_vuln_hdf5:
            build_plan = self._prepare_vulnerable_hdf5_dependency(build_plan)

        cxx = self._select_cxx(build_plan) or "clang++"
        lib_name = self._guess_standalone_ossfuzz_link_lib(harness, repo_path)
        # -fsanitize=fuzzer provides the libFuzzer driver so ClusterFuzz seeds can
        # be passed as argv (same as OSS-Fuzz reproducers). Keep shared-libasan to
        # match the rest of the pipeline's ASan linking.
        # Do not narrow cmake/make to this harness name — it is not a cmake target.
        if use_vuln_hdf5:
            hdf5_inc = f"-I{self.VULN_HDF5_PREFIX}/include"
            hdf5_libs = (
                f"-L{self.VULN_HDF5_PREFIX}/lib "
                f"-Wl,-rpath,{self.VULN_HDF5_PREFIX}/lib -lhdf5"
            )
            compile_cmd = (
                f"{cxx} -g -O0 -fsanitize=address,fuzzer -shared-libasan "
                f"-fno-omit-frame-pointer -std=c++17 "
                f"-Iossfuzz -Isrc -Ibuild -Ibuild/src -I. {hdf5_inc} "
                f"{source_rel} -o {harness} "
                f"-Lbuild -Wl,-rpath,\"$PWD/build\" -l{lib_name} {hdf5_libs} -lz -pthread"
            )
        else:
            compile_cmd = (
                "HDF5_LIBS=\"$(pkg-config --libs hdf5 2>/dev/null "
                "|| pkg-config --libs hdf5_serial 2>/dev/null "
                "|| echo '-lhdf5')\"; "
                f"{cxx} -g -O0 -fsanitize=address,fuzzer -shared-libasan "
                f"-fno-omit-frame-pointer -std=c++17 "
                f"-Iossfuzz -Isrc -Ibuild -Ibuild/src -I. "
                f"{source_rel} -o {harness} "
                f"-Lbuild -Wl,-rpath,\"$PWD/build\" -l{lib_name} $HDF5_LIBS -lz -pthread"
            )
        commands = list(build_plan.build_commands or [])
        commands.append(compile_cmd)
        build_plan.build_commands = commands

        if build_plan.build_script_override:
            script = build_plan.build_script_override.rstrip() + "\n" + compile_cmd + "\n"
            build_plan.build_script_override = script

        note = f"Standalone OSS-Fuzz harness compile appended for {source_rel} -> {harness}."
        if use_vuln_hdf5:
            note = (
                f"{note} Gated vulnerable HDF5 1.12.0 prefix enabled "
                f"({self.VULN_HDF5_PREFIX})."
            )
        if note not in (build_plan.rationale or ""):
            build_plan.rationale = f"{(build_plan.rationale or '').rstrip()} {note}".strip()
        return build_plan

    def _standalone_ossfuzz_needs_vulnerable_hdf5(self, repo_path: Path | None) -> bool:
        """True when standalone ossfuzz tree also depends on HDF5 (e.g. matio)."""

        if repo_path is None or not Path(repo_path).exists():
            return False
        return self._repo_text_contains_any(
            Path(repo_path),
            markers=(
                "MATIO_WITH_HDF5",
                "find_package(HDF5)",
                "MATIO_CHECK_HDF5",
                "HAVE_HDF5",
                "--with-hdf5",
                "HDF5_LIBS",
            ),
            names=(
                "CMakeLists.txt",
                "thirdParties.cmake",
                "options.cmake",
                "configure.ac",
                "configure",
                "Makefile.am",
            ),
        )

    def _prepare_vulnerable_hdf5_dependency(self, build_plan: BuildPlan) -> BuildPlan:
        """Build HDF5 1.12.0 into a private prefix and point cmake/link at it.

        Only invoked for gated standalone-ossfuzz+HDF5 trees. Avoids system
        libhdf5 (often 1.10.x) so ClusterFuzz seeds can reach the vulnerable path.
        """

        blocked = {
            "libhdf5-dev",
            "libhdf5-serial-dev",
            "libhdf5-cpp-103",
            "libhdf5-103",
            "libhdf5-103-1",
        }
        packages = [
            package
            for package in (build_plan.install_packages or [])
            if package.strip().lower() not in blocked
        ]
        for required in ("zlib1g-dev", "curl", "ca-certificates", "autoconf", "automake", "libtool"):
            if required not in packages:
                packages.append(required)
        build_plan.install_packages = sorted(set(packages))

        prefix = self.VULN_HDF5_PREFIX
        hdf5_build_cmd = self._vulnerable_hdf5_build_command()
        commands = list(build_plan.build_commands or [])
        joined = "\n".join(commands)
        if self.VULN_HDF5_MARKER not in joined:
            commands.insert(0, hdf5_build_cmd)
            build_plan.build_commands = commands

        build_plan = self._inject_cmake_define(build_plan, f"-DHDF5_ROOT={prefix}")
        if build_plan.build_script_override and self.VULN_HDF5_MARKER not in build_plan.build_script_override:
            build_plan.build_script_override = (
                hdf5_build_cmd.rstrip() + "\n" + build_plan.build_script_override.lstrip()
            )
        note = f"Vulnerable HDF5 1.12.0 will be built at {prefix} ({self.VULN_HDF5_MARKER})."
        if note not in (build_plan.rationale or ""):
            build_plan.rationale = f"{(build_plan.rationale or '').rstrip()} {note}".strip()
        return build_plan

    def _vulnerable_hdf5_build_command(self) -> str:
        """Install ASan HDF5 1.12.0 under VULN_HDF5_PREFIX via autotools.

        OSS-Fuzz builds this dependency with ``./configure --disable-asserts
        --disable-internal-debug``. cmake RelWithDebInfo diverges and turns the
        ``H5MM_memcpy`` overflow into a DIAG / huge-malloc miss.
        """

        return self._hdf5_autotools_prefix_build_command(
            prefix=self.VULN_HDF5_PREFIX,
            url=self.VULN_HDF5_TARBALL_URL,
            fallback_url=self.VULN_HDF5_TARBALL_URL_FALLBACK,
            marker=self.VULN_HDF5_MARKER,
            work_prefix="hdf5-vuln",
            with_asan=True,
        )

    def _fixed_hdf5_build_command(self) -> str:
        """Install HDF5 1.12.1 under FIXED_HDF5_PREFIX via autotools (no ASan).

        HDF5 1.12.1's cmake ConfigureChecks fail on this image (missing
        ``pac_Cconftest.out`` / decimal-precision=0) with both clang and gcc.
        Autotools avoids that path; post-verify only needs a fixed dependency.
        """

        return self._hdf5_autotools_prefix_build_command(
            prefix=self.FIXED_HDF5_PREFIX,
            url=self.FIXED_HDF5_TARBALL_URL,
            fallback_url=self.FIXED_HDF5_TARBALL_URL_FALLBACK,
            marker=self.FIXED_HDF5_MARKER,
            work_prefix="hdf5-fixed",
        )

    @classmethod
    def _hdf5_autotools_prefix_build_command(
        cls,
        *,
        prefix: str,
        url: str,
        fallback_url: str,
        marker: str,
        work_prefix: str,
        with_asan: bool = False,
    ) -> str:
        """Install HDF5 with ./configure && make (avoids broken 1.12.1 cmake checks).

        ``with_asan=True`` matches OSS-Fuzz: clang + shared-libasan, and
        ``--disable-asserts --disable-internal-debug`` so ClusterFuzz seeds can
        reach ``H5MM_memcpy`` rather than a DIAG / failed huge malloc.
        """

        src = f"/tmp/{work_prefix}-src"
        # HDF5 runs host tools (H5detect/H5make_libsettings) during the build;
        # with -shared-libasan those need the clang runtime on LD_LIBRARY_PATH.
        # Do not pass sanitizer flags via LDFLAGS: HDF5's Makefile does
        # ``LD_LIBRARY_PATH=$LD_LIBRARY_PATH`echo $LDFLAGS | sed -e 's/-L/:/g' -e 's/ //g'```
        # which concatenates ``-fsanitize=address-shared-libasan`` onto the asan
        # dir and makes H5make_libsettings fail to load libclang_rt.asan.
        asan_prelude = (
            "for _asan_dir in /usr/lib/llvm-*/lib/clang/*/lib/linux /usr/lib/clang/*/lib/linux; do "
            'if [[ -f "${_asan_dir}/libclang_rt.asan-x86_64.so" ]]; then '
            'export LD_LIBRARY_PATH="${_asan_dir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; '
            "break; "
            "fi; "
            "done; "
            "export CC=clang CXX=clang++; "
        )
        if with_asan:
            prelude = asan_prelude
            cflags = (
                "-g -O0 -fsanitize=address -shared-libasan "
                "-fno-omit-frame-pointer -fPIC -Wl,-rpath,${_asan_dir}"
            )
            # Keep LDFLAGS empty so the Makefile path-append cannot clobber
            # LD_LIBRARY_PATH. ASan stays in CFLAGS (compile + libtool link).
            ldflags_assign = ' LDFLAGS=""'
            extra_configure = (
                "--disable-deprecated-symbols --disable-parallel --disable-trace "
                "--disable-internal-debug --disable-asserts --with-pic "
            )
        else:
            prelude = ""
            cflags = "-O2 -fPIC"
            ldflags_assign = ""
            extra_configure = "--enable-build-mode=production "
        # Subshell so ``cd {src}`` cannot leak: otherwise the next cmake
        # ``-S .`` runs against ``/`` after a leftover ``cd /``.
        return (
            f"if [[ ! -e {prefix}/lib/libhdf5.so && ! -e {prefix}/lib/libhdf5.a ]]; then "
            f"echo '[build] {marker}'; "
            "("
            f"{prelude}"
            "apt-get update -qq && "
            "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
            "autoconf automake libtool zlib1g-dev >/dev/null; "
            f"rm -rf {src}; mkdir -p {src}; "
            f"if ! curl -fsSL {url} | tar -xz -C {src} --strip-components=1; then "
            f"  rm -rf {src}; mkdir -p {src}; "
            f"  curl -fsSL {fallback_url} | tar -xz -C {src} --strip-components=1; "
            "fi; "
            f"cd {src}; "
            "if [[ ! -f configure ]]; then "
            "  if [[ -f autogen.sh ]]; then ./autogen.sh; "
            "  else autoreconf -if; fi; "
            "fi; "
            f"./configure --prefix={prefix} --enable-shared --disable-static "
            f"{extra_configure}"
            "--disable-hl --disable-tests "
            "--disable-tools --disable-fortran --disable-java --disable-cxx "
            f'--with-zlib CFLAGS="{cflags}" CXXFLAGS="{cflags}"{ldflags_assign}; '
            'make -j"$(nproc)"; make install'
            "); "
            "fi"
        )

    @classmethod
    def rewrite_build_script_for_fixed_hdf5(cls, build_script_path: str) -> str:
        """Install fixed HDF5 via autotools, then rebuild matio/harness against it.

        Pre-installs ``/opt/hdf5-fixed`` so the rewritten ``build.sh`` skips its
        ASan HDF5 block (``if [[ ! -e .../libhdf5.so ]]``) and only recompiles
        matio + the fuzzer with ``-DHDF5_ROOT=/opt/hdf5-fixed``.
        """

        install_fixed = cls._hdf5_autotools_prefix_build_command(
            prefix=cls.FIXED_HDF5_PREFIX,
            url=cls.FIXED_HDF5_TARBALL_URL,
            fallback_url=cls.FIXED_HDF5_TARBALL_URL_FALLBACK,
            marker=cls.FIXED_HDF5_MARKER,
            work_prefix="hdf5-fixed",
        )
        rewrite_and_run = (
            f'sed -e "s|{cls.VULN_HDF5_PREFIX}|{cls.FIXED_HDF5_PREFIX}|g" '
            f'-e "s|{cls.VULN_HDF5_MARKER}|{cls.FIXED_HDF5_MARKER}|g" '
            f'-e "s|hdf5-1.12.0|hdf5-1.12.1|g" '
            f'-e "s|hdf5-1_12_0|hdf5-1_12_1|g" '
            f'-e "s|/tmp/hdf5-vuln-|/tmp/hdf5-fixed-|g" '
            f"{build_script_path} > /tmp/build-hdf5-fixed.sh && "
            "bash /tmp/build-hdf5-fixed.sh"
        )
        return f"{install_fixed} && {rewrite_and_run}"

    @staticmethod
    def _guess_standalone_ossfuzz_link_lib(harness: str, repo_path: Path | None) -> str:
        """Infer ``-l<name>`` for standalone ossfuzz harnesses.

        Workspace checkouts are often named ``repo``; prefer the harness stem
        (``matio_fuzzer`` → ``matio``) over the directory basename.
        """

        name = (harness or "").strip()
        for suffix in ("_fuzzer", "-fuzzer"):
            if name.endswith(suffix) and len(name) > len(suffix):
                stem = name[: -len(suffix)]
                if re.fullmatch(r"[A-Za-z][\w\-]*", stem):
                    return stem
        generic_dirs = {"repo", "src", "source", "project", "code", "workdir", "tree"}
        if repo_path is not None:
            candidate = repo_path.name.strip().lower().removesuffix(".git")
            if (
                candidate
                and candidate not in generic_dirs
                and re.fullmatch(r"[a-zA-Z][\w\-]*", candidate)
            ):
                return candidate
        return "matio"

    def _narrow_build_to_harness_target(self, build_plan: BuildPlan, harness: str) -> BuildPlan:
        """Rewrite build commands to compile only the selected harness target."""

        if not harness:
            return build_plan

        commands = list(build_plan.build_commands or [])
        changed = False
        for index, command in enumerate(commands):
            rewritten = self._rewrite_command_for_harness_target(command, harness)
            if rewritten != command:
                commands[index] = rewritten
                changed = True
        if changed:
            build_plan.build_commands = commands

        if build_plan.build_script_override:
            script = build_plan.build_script_override
            rewritten_script = re.sub(
                r"(?m)^(?P<prefix>\s*)(?P<cmd>cmake\s+--build\b[^\n]*|make\b[^\n]*)$",
                lambda match: (
                    f"{match.group('prefix')}"
                    f"{self._rewrite_command_for_harness_target(match.group('cmd'), harness)}"
                ),
                script,
            )
            build_plan.build_script_override = rewritten_script
        return build_plan

    def _rewrite_command_for_harness_target(self, command: str, harness: str) -> str:
        """Attach a single-target selector to cmake/make build lines when safe."""

        stripped = (command or "").strip()
        if not stripped or not harness:
            return command

        make_target = harness.replace("\\", "/")
        bare = Path(make_target).name

        if re.search(rf"(?:--target\s+|make\s+(?:\S+\s+)*){re.escape(make_target)}\b", stripped):
            return command

        if "cmake --build" in stripped:
            cleaned = re.sub(r"\s*--target\s+\S+", "", stripped)
            match = re.match(r"^(?P<pre>cmake\s+--build\s+\S+)(?P<rest>\s.*)?$", cleaned)
            # CMake target names are the bare harness id, not a path.
            cmake_target = bare
            if match:
                rest = match.group("rest") or ""
                return f"{match.group('pre')} --target {cmake_target}{rest}"
            return f"{cleaned} --target {cmake_target}"

        # Plain recursive make without an explicit goal: build only the harness.
        if re.fullmatch(r"make(?:\s+-j\$\(nproc\))?", stripped):
            return f"make {make_target} -j$(nproc)"
        if re.fullmatch(r"make\s+-j\$\(nproc\)", stripped):
            return f"make {make_target} -j$(nproc)"

        # Replace bare / wrong-path make goals (e.g. `make qpdf_fuzzer` →
        # `make fuzz/build/qpdf_fuzzer`) when the harness is a path target.
        make_match = re.match(
            rf"^make\s+(?P<goal>\S+)(?P<rest>(?:\s+-j\$\(nproc\))?(?:\s.*)?)?$",
            stripped,
        )
        if make_match:
            goal = make_match.group("goal").replace("\\", "/")
            rest = make_match.group("rest") or ""
            if goal == make_target:
                return command
            if goal == bare or Path(goal).name == bare:
                if "-j$(nproc)" not in rest:
                    rest = f"{rest} -j$(nproc)" if rest.strip() else " -j$(nproc)"
                return f"make {make_target}{rest}"
        return command

    def _inject_cmake_define(self, build_plan: BuildPlan, define: str) -> BuildPlan:
        """Append a -D... flag to the cmake configure command when missing."""

        key = define.split("=", 1)[0]
        commands = list(build_plan.build_commands or [])
        changed = False
        for index, command in enumerate(commands):
            if not self._is_cmake_command(command) or "cmake --build" in command:
                continue
            if key + "=" in command or key in command.split():
                return build_plan
            commands[index] = f"{command.rstrip()} {define}"
            changed = True
            break
        if changed:
            build_plan.build_commands = commands
            return build_plan

        if build_plan.build_script_override and "cmake" in build_plan.build_script_override:
            script = build_plan.build_script_override
            if key + "=" in script:
                return build_plan

            def _inject(match: re.Match[str]) -> str:
                line = match.group(0)
                if key + "=" in line or "cmake --build" in line:
                    return line
                return f"{line.rstrip()} {define}"

            build_plan.build_script_override = re.sub(
                r"(?m)^[^\n]*\bcmake(?!\s+--build)[^\n]*$",
                _inject,
                script,
                count=1,
            )
        return build_plan

    def _repo_text_contains_any(
        self,
        repo_path: Path,
        *,
        markers: tuple[str, ...],
        names: tuple[str, ...],
        max_bytes: int = 200_000,
    ) -> bool:
        if not repo_path.exists():
            return False
        lowered_markers = tuple(marker.lower() for marker in markers)
        candidate_files: list[Path] = []
        for name in names:
            path = repo_path / name
            if path.is_file():
                candidate_files.append(path)
        try:
            children = list(repo_path.iterdir())
        except OSError:
            children = []
        for child in children:
            if not child.is_dir() or child.name.startswith("."):
                continue
            for name in names:
                nested = child / name
                if nested.is_file():
                    candidate_files.append(nested)
            # One more nesting level (libselinux/src/regex.h).
            try:
                grandchildren = list(child.iterdir())
            except OSError:
                grandchildren = []
            for grandchild in grandchildren:
                if not grandchild.is_dir() or grandchild.name.startswith("."):
                    continue
                for name in names:
                    deep = grandchild / name
                    if deep.is_file():
                        candidate_files.append(deep)

        for path in candidate_files:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")[:max_bytes].lower()
            except OSError:
                continue
            if any(marker in text for marker in lowered_markers):
                return True
        return False

    def _repo_indicates_readline(self, repo_path: Path) -> bool:
        """Detect whether the checked-out repo expects GNU readline at build time."""

        return self._repo_text_contains_any(
            repo_path,
            markers=(
                "lua_use_readline",
                "readline/readline.h",
                "readline/history.h",
                "-lreadline",
            ),
            names=("Makefile", "makefile", "GNUmakefile", "lua.c"),
        )
    def _ensure_dockerfile_override_includes_packages(
        self,
        dockerfile_content: str,
        packages: list[str],
    ) -> str:
        """Inject any install_packages missing from a Dockerfile override apt-get line."""

        lower = dockerfile_content.lower()
        missing = [
            package
            for package in packages
            if package.strip() and package.lower() not in lower
        ]
        if not missing or "apt-get install" not in lower:
            return dockerfile_content.rstrip() + "\n"

        lines = dockerfile_content.splitlines()
        updated: list[str] = []
        replaced = False
        package_blob = " ".join(missing)
        for line in lines:
            if not replaced and "apt-get install" in line:
                if "--no-install-recommends" in line:
                    line = line.replace(
                        "--no-install-recommends ",
                        f"--no-install-recommends {package_blob} ",
                        1,
                    )
                else:
                    line = line.replace(
                        "apt-get install -y ",
                        f"apt-get install -y {package_blob} ",
                        1,
                    )
                replaced = True
            updated.append(line)
        return "\n".join(updated).rstrip() + "\n"

    def _guess_binary_or_entrypoint(self, build_system: str, project_name: str) -> Optional[str]:
        guesses = {
            "cmake": f"build/{project_name}",
            "make": project_name,
            "autotools": f"src/{project_name}",
            "cargo": f"target/debug/{project_name}",
            "go": project_name,
        }
        return guesses.get(build_system)

    def _select_compiler(self, build_plan: BuildPlan) -> str:
        # -shared-libasan is clang-only; never export CC=gcc under that policy.
        if "-shared-libasan" in self._default_sanitizer_flags():
            return "clang"
        tokens = self._compiler_tokens(build_plan)
        if "gcc" in tokens and "clang" not in tokens:
            return "gcc"
        return "clang"

    def _select_cxx(self, build_plan: BuildPlan) -> str:
        """Prefer clang++ for ASan builds; avoid matching the 'g++' suffix inside 'clang++'."""

        if "-shared-libasan" in self._default_sanitizer_flags():
            return "clang++"
        tokens = self._compiler_tokens(build_plan)
        if "clang++" in tokens or "clang" in tokens:
            return "clang++"
        if "g++" in tokens or "gcc" in tokens:
            return "g++"
        return "clang++"

    def _compiler_tokens(self, build_plan: BuildPlan) -> set[str]:
        joined = " ".join(build_plan.build_commands or []).lower()
        return set(re.findall(r"[a-z0-9+._-]+", joined))

    def _build_dependency_sources(self, repo_scan: dict[str, list[str]]) -> list[str]:
        sources: list[str] = []
        if repo_scan["build_files"]:
            sources.append("repo_scan:build_files")
        if repo_scan["evidence_files"]:
            sources.append("repo_scan:readme_install")
        if repo_scan["ci_files"]:
            sources.append("repo_scan:ci")
        return sources

    def _maybe_resolve_ref(self, repo_path: Path, requested_ref: str) -> Optional[str]:
        result = self.process_tool.run(ProcessRequest(command=["git", "rev-parse", requested_ref], cwd=str(repo_path)))
        if not result.success:
            return None
        return result.stdout.strip()

    def _resolve_existing_ref(self, repo_path: Path, requested_ref: str) -> str:
        resolved = self._maybe_resolve_ref(repo_path, requested_ref)
        if resolved:
            return resolved
        return requested_ref

    def _rewrite_dockerfile_git_checkout_ref(
        self,
        dockerfile_content: str,
        old_ref: str,
        new_ref: str,
    ) -> str:
        """Rewrite ``git checkout <old_ref>`` in a Dockerfile override."""

        if not dockerfile_content or not old_ref or not new_ref or old_ref == new_ref:
            return dockerfile_content
        pattern = re.compile(
            r'(git\s+checkout\s+)(["\']?)(' + re.escape(old_ref) + r')(\2)',
            re.IGNORECASE,
        )
        return pattern.sub(rf"\g<1>\g<2>{new_ref}\g<4>", dockerfile_content)

    def _patch_applies_at_ref(self, repo_path: Path, ref: str, patch_path: Path) -> bool:
        """True when ``git apply --check`` succeeds on ``ref``."""

        if not ref or not patch_path.exists():
            return False
        checkout = self.process_tool.run(
            ProcessRequest(
                command=["git", "checkout", "--force", ref],
                cwd=str(repo_path),
            )
        )
        if not checkout.success:
            return False
        check = self.process_tool.run(
            ProcessRequest(
                command=["git", "apply", "--check", str(patch_path)],
                cwd=str(repo_path),
            )
        )
        return bool(check.success)

    def _align_vulnerable_ref_with_applyable_patch(
        self,
        repo_path: Path,
        build_plan: BuildPlan,
        knowledge: KnowledgeModel | None = None,
        patch_diff_path: Optional[Path] = None,
    ) -> BuildPlan:
        """If patch.diff does not apply to the planned vulnerable ref, use fixed_ref^.

        LLM/knowledge often pick a nearby but unrelated commit (or a later
        rewrite on another branch). Verify then fails at ``git apply``. The
        authoritative tree for an in-tree fix is the first parent of
        ``chosen_fixed_ref`` when that is the commit the patch was taken from.
        """

        current_ref = (build_plan.chosen_vulnerable_ref or "").strip()
        fixed_ref = (build_plan.chosen_fixed_ref or "").strip()
        if knowledge is not None and not fixed_ref:
            fixed_ref = (knowledge.fixed_ref or "").strip()
        if not current_ref or not fixed_ref:
            return build_plan
        current_ref = self._resolve_existing_ref(repo_path, current_ref)
        fixed_ref = self._resolve_existing_ref(repo_path, fixed_ref)
        build_plan.chosen_vulnerable_ref = current_ref
        build_plan.chosen_fixed_ref = fixed_ref

        resolved_patch = patch_diff_path
        if resolved_patch is None and knowledge is not None and knowledge.cve_id:
            found = find_patch_diff(knowledge.cve_id)
            resolved_patch = found
        if resolved_patch is None or not Path(resolved_patch).exists():
            return build_plan

        patch_path = Path(resolved_patch)
        raw_patch = patch_path.read_text(encoding="utf-8", errors="replace")
        filtered_patch, _dropped = strip_unapplyable_binary_stub_hunks(raw_patch)
        if not (filtered_patch or "").strip():
            return build_plan

        parent_ref = self._maybe_resolve_ref(repo_path, f"{fixed_ref}^")
        if not parent_ref:
            return build_plan

        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".diff",
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write(filtered_patch)
            temp_patch = Path(handle.name)
        try:
            if self._patch_applies_at_ref(repo_path, current_ref, temp_patch):
                return build_plan
            if parent_ref == current_ref:
                return build_plan
            if not self._patch_applies_at_ref(repo_path, parent_ref, temp_patch):
                self.process_tool.run(
                    ProcessRequest(
                        command=["git", "checkout", "--force", current_ref],
                        cwd=str(repo_path),
                    )
                )
                return build_plan
        finally:
            try:
                temp_patch.unlink(missing_ok=True)
            except OSError:
                pass

        previous_ref = build_plan.chosen_vulnerable_ref
        build_plan.chosen_vulnerable_ref = parent_ref
        if build_plan.dockerfile_override:
            rewritten = build_plan.dockerfile_override
            for token in {previous_ref, current_ref}:
                rewritten = self._rewrite_dockerfile_git_checkout_ref(
                    rewritten,
                    token,
                    parent_ref,
                )
            build_plan.dockerfile_override = rewritten
        return build_plan


    def _derive_project_name(self, repo_url: str) -> str:
        name = repo_url.rstrip("/").split("/")[-1]
        if name.endswith(".git"):
            name = name[:-4]
        return re.sub(r"[^A-Za-z0-9._-]+", "-", name) or "target"

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

        if template_name == "Dockerfile.j2":
            return self._render_dockerfile_fallback(context)
        if template_name == "build.sh.j2":
            return self._render_build_script_fallback(context)
        raise RuntimeError(f"unsupported template without Jinja2: {template_name}")

    def _render_dockerfile_fallback(self, context: dict[str, Any]) -> str:
        apt_packages = context.get("apt_packages") or []
        lines = [
            f"FROM {context.get('base_image', 'ubuntu:20.04')}",
            "",
            'SHELL ["/bin/bash", "-o", "pipefail", "-c"]',
            "",
            'ARG HTTP_PROXY=""',
            'ARG HTTPS_PROXY=""',
            'ARG ALL_PROXY=""',
            'ARG NO_PROXY=""',
            'ARG http_proxy=""',
            'ARG https_proxy=""',
            'ARG all_proxy=""',
            'ARG no_proxy=""',
            "",
            "ENV DEBIAN_FRONTEND=noninteractive",
            "ENV HTTP_PROXY=${HTTP_PROXY}",
            "ENV HTTPS_PROXY=${HTTPS_PROXY}",
            "ENV ALL_PROXY=${ALL_PROXY}",
            "ENV NO_PROXY=${NO_PROXY}",
            "ENV http_proxy=${http_proxy}",
            "ENV https_proxy=${https_proxy}",
            "ENV all_proxy=${all_proxy}",
            "ENV no_proxy=${no_proxy}",
            f"ENV WORKSPACE_ROOT={context.get('workspace_root', '/workspace')}",
            f"ENV ARTIFACTS_ROOT={context.get('artifacts_root', '/workspace/artifacts')}",
            f"ENV BUILD_ARTIFACTS_DIR={context.get('build_artifacts_dir', '/workspace/artifacts/build')}",
            f"ENV POC_ARTIFACTS_DIR={context.get('poc_artifacts_dir', '/workspace/artifacts/poc')}",
            f"ENV VERIFY_ARTIFACTS_DIR={context.get('verify_artifacts_dir', '/workspace/artifacts/verify')}",
            'ENV SRC_ROOT=/src',
            f"ENV PROJECT_DIR={context.get('project_dir', '/src/target')}",
            "",
            "RUN apt-get update && \\",
            f"    apt-get install -y --no-install-recommends {' '.join(apt_packages)} && \\",
            "    apt-get clean && \\",
            "    rm -rf /var/lib/apt/lists/*",
            "",
            "RUN mkdir -p ${SRC_ROOT} ${WORKSPACE_ROOT} ${ARTIFACTS_ROOT} ${BUILD_ARTIFACTS_DIR} ${POC_ARTIFACTS_DIR} ${VERIFY_ARTIFACTS_DIR}",
            "",
            "RUN set -eux; \\",
            f'    git clone "{context["repo_url"]}" "${{PROJECT_DIR}}" && \\',
            '    cd "${PROJECT_DIR}" && \\',
            f'    git checkout "{context["vulnerable_ref"]}" && \\',
            "    git rev-parse HEAD",
            "",
            "COPY artifacts/build ${BUILD_ARTIFACTS_DIR}",
            "COPY artifacts/poc ${POC_ARTIFACTS_DIR}",
            "COPY artifacts/verify ${VERIFY_ARTIFACTS_DIR}",
            "",
            "WORKDIR ${PROJECT_DIR}",
            "",
        ]
        return "\n".join(lines)

    def _get_docker_build_proxy(self) -> str:
        for key in (
            "DOCKER_BUILD_PROXY",
            "DOCKER_PROXY",
            "HTTPS_PROXY",
            "https_proxy",
            "HTTP_PROXY",
            "http_proxy",
        ):
            value = (os.getenv(key) or "").strip()
            if value:
                return value
        return ""

    def _build_docker_proxy_args(self, proxy_url: str) -> dict[str, str]:
        if not proxy_url:
            return {}
        no_proxy = os.getenv("NO_PROXY") or os.getenv("no_proxy") or ""
        return {
            "HTTP_PROXY": proxy_url,
            "HTTPS_PROXY": proxy_url,
            "ALL_PROXY": proxy_url,
            "NO_PROXY": no_proxy,
            "http_proxy": proxy_url,
            "https_proxy": proxy_url,
            "all_proxy": proxy_url,
            "no_proxy": no_proxy,
        }

    def _select_docker_build_network_mode(self, proxy_url: str) -> Optional[str]:
        if not proxy_url:
            return None
        lowered = proxy_url.lower()
        if "127.0.0.1:" in lowered or "localhost:" in lowered:
            return "host"
        return None

    def _render_build_script_fallback(self, context: dict[str, Any]) -> str:
        build_commands = context.get("build_commands") or []
        configure_commands = context.get("configure_commands") or []
        clean_commands = context.get("clean_commands") or []
        build_system = (context.get("build_system") or "unknown").strip().lower()
        sanitizer_flags = context.get("sanitizer_flags") or self._default_sanitizer_flags()
        sanitizer_link_flags = self._default_sanitizer_link_flags()
        defer_asan = build_system == "autotools" or bool(configure_commands)
        if build_system == "cmake":
            defer_asan = False
        lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            "",
            "log() {",
            "    printf '[build] %s\\n' \"$*\" >&2",
            "}",
            "",
            f'PROJECT_DIR="{context["project_dir"]}"',
            f'BUILD_ARTIFACTS_DIR="{context.get("build_artifacts_dir", "artifacts/build")}"',
            'mkdir -p "${BUILD_ARTIFACTS_DIR}"',
            'cd "${PROJECT_DIR}"',
            'log "project_dir=$(pwd)"',
            'log "build_artifacts_dir=${BUILD_ARTIFACTS_DIR}"',
            'log "running clean step"',
        ]
        lines.extend(clean_commands)
        lines.extend(["", 'log "running configure step"'])
        lines.extend(configure_commands)
        if defer_asan:
            lines.extend(
                [
                    "",
                    'log "applying sanitizer flags for build"',
                    f'export CFLAGS="-g -O0 {sanitizer_flags}"',
                    f'export CXXFLAGS="-g -O0 {sanitizer_flags}"',
                    f'export LDFLAGS="{sanitizer_link_flags}"',
                ]
            )
        lines.extend(["", 'log "running build step"'])
        lines.extend(build_commands)
        lines.append("")
        return "\n".join(lines)

    def _compose_build_logs(self, docker_build_result: Any, run_result: Any | None) -> str:
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

    def _classify_failure_kind(self, build_logs: str) -> str:
        if "image_build_success=False" in build_logs:
            return "docker_build"
        if "container_run_success=False" in build_logs:
            return "container_run"
        return "unknown"

    def _ensure_required_docker_packages(self, packages: list[str]) -> list[str]:
        merged = list(packages)
        for package in self.REQUIRED_DOCKER_PACKAGES:
            if package not in merged:
                merged.append(package)
        return sorted(set(merged))

    def _missing_required_docker_packages(self, install_packages: list[str], dockerfile_content: str) -> list[str]:
        install_package_set = {package.strip() for package in install_packages if package.strip()}
        dockerfile_lower = dockerfile_content.lower()
        missing: list[str] = []
        for package in self.REQUIRED_DOCKER_PACKAGES:
            if package not in install_package_set or package.lower() not in dockerfile_lower:
                missing.append(package)
        return missing

    def _ensure_dockerfile_override_has_required_tools(self, dockerfile_content: str, install_packages: list[str]) -> str:
        lower = dockerfile_content.lower()
        missing_required = self._missing_required_docker_packages(install_packages, dockerfile_content)
        if "apt-get install" in lower and missing_required:
            install_packages = self._ensure_required_docker_packages(install_packages)
        if "apt-get install" in lower and missing_required:
            lines = dockerfile_content.splitlines()
            updated: list[str] = []
            replaced = False
            for line in lines:
                if not replaced and "apt-get install" in line:
                    if "--no-install-recommends" in line:
                        line = line.replace("--no-install-recommends ", f"--no-install-recommends {' '.join(install_packages)} ", 1)
                    else:
                        line = line.replace("apt-get install -y ", f"apt-get install -y {' '.join(install_packages)} ", 1)
                    replaced = True
                updated.append(line)
            dockerfile_content = "\n".join(updated)
        return dockerfile_content.rstrip() + "\n"

    @staticmethod
    def _dockerfile_has_repo_clone(dockerfile_content: str) -> bool:
        """True when the Dockerfile clones or otherwise materializes the project under /src."""

        if not dockerfile_content:
            return False
        lower = dockerfile_content.lower()
        if "git clone" in lower:
            return True
        # Rare but valid: COPY local checkout into the image project dir.
        if re.search(r"(?im)^COPY\s+\S+\s+/src/\S+", dockerfile_content):
            return True
        return False

    def _ensure_dockerfile_clones_project(
        self,
        dockerfile_content: str,
        *,
        repo_url: str,
        vulnerable_ref: str,
        project_dir: str,
        workspace_root: str = "/workspace",
        artifacts_root: str = "/workspace/artifacts",
        build_artifacts_dir: str = "/workspace/artifacts/build",
        poc_artifacts_dir: str = "/workspace/artifacts/poc",
        verify_artifacts_dir: str = "/workspace/artifacts/verify",
    ) -> str:
        """Append clone/checkout scaffolding when a Dockerfile override omitted it.

        LLM plans often emit apt-only ``dockerfile_override`` when bumping the
        base image (e.g. ubuntu:22.04 for ECM/Qt). That drops ``git clone`` and
        yields ``project_dir does not exist: /src/<name>`` at container run.
        """

        content = (dockerfile_content or "").rstrip() + "\n"
        if self._dockerfile_has_repo_clone(content):
            return content
        repo_url = (repo_url or "").strip()
        vulnerable_ref = (vulnerable_ref or "").strip()
        project_dir = (project_dir or "").strip() or "/src/target"
        if not repo_url or not vulnerable_ref:
            return content

        scaffold = "\n".join(
            [
                f"ENV WORKSPACE_ROOT={workspace_root}",
                f"ENV ARTIFACTS_ROOT={artifacts_root}",
                f"ENV BUILD_ARTIFACTS_DIR={build_artifacts_dir}",
                f"ENV POC_ARTIFACTS_DIR={poc_artifacts_dir}",
                f"ENV VERIFY_ARTIFACTS_DIR={verify_artifacts_dir}",
                "ENV SRC_ROOT=/src",
                f"ENV PROJECT_DIR={project_dir}",
                "",
                "RUN mkdir -p \\",
                '    "${SRC_ROOT}" \\',
                '    "${WORKSPACE_ROOT}" \\',
                '    "${ARTIFACTS_ROOT}" \\',
                '    "${BUILD_ARTIFACTS_DIR}" \\',
                '    "${POC_ARTIFACTS_DIR}" \\',
                '    "${VERIFY_ARTIFACTS_DIR}"',
                "",
                "RUN set -eux; \\",
                f'    git clone "{repo_url}" "${{PROJECT_DIR}}" && \\',
                '    cd "${PROJECT_DIR}" && \\',
                f'    git checkout "{vulnerable_ref}" && \\',
                "    git rev-parse HEAD",
                "",
                "COPY artifacts/build ${BUILD_ARTIFACTS_DIR}",
                "COPY artifacts/poc ${POC_ARTIFACTS_DIR}",
                "COPY artifacts/verify ${VERIFY_ARTIFACTS_DIR}",
                "",
                'RUN if [[ -f "${BUILD_ARTIFACTS_DIR}/build.sh" ]]; then chmod +x "${BUILD_ARTIFACTS_DIR}/build.sh"; fi && \\',
                '    if [[ -f "${POC_ARTIFACTS_DIR}/run.sh" ]]; then chmod +x "${POC_ARTIFACTS_DIR}/run.sh"; fi',
                "",
                "WORKDIR ${PROJECT_DIR}",
                "",
            ]
        )
        return content.rstrip() + "\n\n" + scaffold

    def _sanitizer_enabled(self, build_script_content: str) -> bool:
        return "-fsanitize=" in build_script_content

    def _verify_build_artifact(
        self,
        artifact: BuildArtifact,
        paths: BuildStagePaths,
        plan_meta: dict,
        cve_id: str,
    ) -> dict:
        """对 build 阶段产物做事实型自检，不裁决漏洞、不影响 build_success。"""

        result: dict[str, Any] = {}
        verify_notes: list[str] = []

        # 4.14 timestamp
        result["verify_status"] = "ok"  # placeholder, updated at the end
        result["verified_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        runtime_image_tag = artifact.compiled_image_tag or artifact.docker_image_tag

        # 4.2 image_present / image_digest
        try:
            if not runtime_image_tag:
                result["image_present"] = False
                result["image_digest"] = None
                verify_notes.append("runtime image tag missing in BuildArtifact")
            else:
                inspect_result = self.docker_tool.process_tool.run(
                    ProcessRequest(
                        command=["docker", "image", "inspect", runtime_image_tag, "--format", "{{.Id}}"],
                        timeout_seconds=30,
                    )
                )
                if inspect_result.success:
                    result["image_present"] = True
                    result["image_digest"] = inspect_result.stdout.strip() or None
                else:
                    result["image_present"] = False
                    result["image_digest"] = None
        except Exception:
            result["image_present"] = False
            result["image_digest"] = None

        # 4.3 workspace_layout_ok
        try:
            missing_dirs: list[str] = []
            for directory in (paths.workspace_root, paths.repo_dir, paths.build_dir):
                if not directory.is_dir():
                    missing_dirs.append(str(directory))
            result["workspace_layout_ok"] = len(missing_dirs) == 0
            result["workspace_layout_missing"] = missing_dirs
        except Exception:
            result["workspace_layout_ok"] = False
            result["workspace_layout_missing"] = []

        # 4.4 dockerfile_present
        try:
            result["dockerfile_present"] = (
                paths.dockerfile.exists() and paths.dockerfile.is_file() and paths.dockerfile.stat().st_size > 0
            )
        except Exception:
            result["dockerfile_present"] = False

        # 4.5 build_script_present
        try:
            result["build_script_present"] = (
                paths.build_script.exists() and paths.build_script.is_file() and paths.build_script.stat().st_size > 0
            )
        except Exception:
            result["build_script_present"] = False

        # 4.6 build_log_present
        try:
            result["build_log_present"] = paths.build_log.exists() and paths.build_log.is_file()
        except Exception:
            result["build_log_present"] = False

        # 4.7 image_build_success / container_run_success
        try:
            result["image_build_success"] = "image_build_success=True" in artifact.build_logs
            if "container_run_success=True" in artifact.build_logs:
                result["container_run_success"] = True
            elif "container_run_success=False" in artifact.build_logs:
                result["container_run_success"] = False
            else:
                result["container_run_success"] = None
        except Exception:
            result["image_build_success"] = False
            result["container_run_success"] = None

        # 4.8 binary_in_container
        try:
            if not artifact.expected_binary_path:
                result["binary_in_container"] = {
                    "checked": False,
                    "reason": "expected_binary_path is empty",
                }
                verify_notes.append("expected_binary_path is empty")
            elif not result.get("image_present"):
                result["binary_in_container"] = {
                    "checked": False,
                    "reason": "image not present, cannot check binary",
                }
            else:
                check_cmd = (
                    f'test -x "${{PROJECT_DIR}}/{artifact.expected_binary_path}" '
                    f'&& echo BINARY_FOUND || echo BINARY_MISSING'
                )
                bin_result = self.docker_tool.run_container(
                    DockerRunRequest(
                        image_tag=runtime_image_tag,
                        command=["bash", "-lc", check_cmd],
                    )
                )
                log_excerpt = (bin_result.stdout + "\n" + bin_result.stderr).strip()[:800]
                exists: Optional[bool] = None
                if "BINARY_FOUND" in bin_result.stdout:
                    exists = True
                elif "BINARY_MISSING" in bin_result.stdout:
                    exists = False
                result["binary_in_container"] = {
                    "checked": True,
                    "exit_code": bin_result.exit_code,
                    "exists": exists,
                    "expected_path": artifact.expected_binary_path,
                    "log_excerpt": log_excerpt,
                }
        except Exception:
            result["binary_in_container"] = {
                "checked": False,
                "reason": "exception during binary check",
            }

        # 4.9 patch_appliable_in_container
        try:
            patch_diff_path = find_patch_diff(cve_id)
            if patch_diff_path is None:
                result["patch_appliable_in_container"] = {
                    "checked": False,
                    "reason": "patch.diff not found",
                }
                verify_notes.append("patch.diff not found")
            elif not result.get("image_present"):
                result["patch_appliable_in_container"] = {
                    "checked": False,
                    "reason": "image not present, cannot check patch",
                }
            else:
                raw_patch = patch_diff_path.read_text(encoding="utf-8", errors="replace")
                filtered_patch, dropped_stubs = strip_unapplyable_binary_stub_hunks(raw_patch)
                patch_diff_host_path = str(patch_diff_path.resolve())
                if dropped_stubs:
                    filtered_host = paths.build_dir / "patch.apply.check.diff"
                    filtered_host.parent.mkdir(parents=True, exist_ok=True)
                    filtered_host.write_text(filtered_patch, encoding="utf-8")
                    patch_diff_host_path = str(filtered_host.resolve())
                    verify_notes.append(
                        "stripped unapplyable binary stubs for patch --check: "
                        + ", ".join(dropped_stubs)
                    )
                command = [
                    "docker", "run", "--rm",
                    "-v", f"{patch_diff_host_path}:/tmp/patch.diff:ro",
                    runtime_image_tag,
                    "bash", "-lc",
                    'cd "${PROJECT_DIR}" && git apply --check /tmp/patch.diff',
                ]
                patch_result = self.docker_tool.process_tool.run(
                    ProcessRequest(command=command, timeout_seconds=120)
                )
                log_excerpt = (patch_result.stdout + "\n" + patch_result.stderr).strip()[:1200]
                result["patch_appliable_in_container"] = {
                    "checked": True,
                    "applied": patch_result.success,
                    "exit_code": patch_result.exit_code,
                    "patch_diff_path": str(patch_diff_path),
                    "log_excerpt": log_excerpt,
                }
        except Exception:
            result["patch_appliable_in_container"] = {
                "checked": False,
                "reason": "exception during patch check",
            }

        # 4.11 repo_ref_in_container
        try:
            if not result.get("image_present"):
                result["repo_ref_in_container"] = {
                    "checked": False,
                    "reason": "image not present, cannot check ref",
                }
            else:
                ref_result = self.docker_tool.run_container(
                    DockerRunRequest(
                        image_tag=runtime_image_tag,
                        command=["bash", "-lc", 'cd "${PROJECT_DIR}" && git rev-parse HEAD'],
                    )
                )
                observed_head = ref_result.stdout.strip()
                expected_ref = artifact.chosen_vulnerable_ref
                matches: Optional[bool] = None
                if expected_ref and len(expected_ref) >= 7 and re.fullmatch(r"[0-9a-fA-F]+", expected_ref):
                    matches = observed_head.startswith(expected_ref) or expected_ref.startswith(observed_head)
                result["repo_ref_in_container"] = {
                    "checked": True,
                    "expected_ref": expected_ref,
                    "observed_head": observed_head,
                    "matches": matches,
                    "exit_code": ref_result.exit_code,
                }
        except Exception:
            result["repo_ref_in_container"] = {
                "checked": False,
                "reason": "exception during ref check",
            }

        # 4.12 verify_notes
        result["verify_notes"] = verify_notes

        # 4.13 verify_status
        must_pass = [
            result.get("image_present"),
            result.get("workspace_layout_ok"),
            result.get("dockerfile_present"),
            result.get("build_script_present"),
            result.get("build_log_present"),
        ]

        if artifact.build_success:
            must_pass.append(result.get("image_build_success"))
            must_pass.append(result.get("container_run_success"))

        if artifact.expected_binary_path and artifact.build_success:
            bin_info = result.get("binary_in_container", {})
            if bin_info.get("checked"):
                must_pass.append(bin_info.get("exists"))

        patch_info = result.get("patch_appliable_in_container", {})
        if patch_info.get("checked"):
            must_pass.append(patch_info.get("applied"))

        if all(item is True for item in must_pass):
            result["verify_status"] = "ok"
        else:
            result["verify_status"] = "partial"

        return result


def build_node(state):
    """LangGraph 节点：执行环境构建阶段。"""

    knowledge = state["knowledge"]
    workspace = state["workspace"]
    retry_count = dict(state.get("retry_count", {}))
    history = list(state.get("stage_history", []))
    stage_status = dict(state.get("stage_status", {}))
    artifacts = dict(state.get("artifacts", {}))
    stage = BuildStage()
    paths = BuildStagePaths(workspace)

    try:
        build = stage.run(knowledge=knowledge, workspace=workspace)
        artifacts["build"] = {
            "workspace": str(paths.workspace_root),
            "repo_dir": str(paths.repo_dir),
            "build_context_yaml": str(paths.build_context_yaml),
            "build_plan_yaml": str(paths.build_plan_yaml),
            "dockerfile": str(paths.dockerfile),
            "build_script": str(paths.build_script),
            "build_log": str(paths.build_log),
            "build_artifact_yaml": str(paths.build_artifact_yaml),
            "build_verify_yaml": str(paths.build_verify_yaml),
        }
        if build.build_success:
            history.append({"stage": "build", "status": "success"})
            stage_status["build"] = "success"
            return {
                "build": build,
                "current_stage": "poc",
                "review_stage": "",
                "human_action_required": False,
                "review_reason": "",
                "stage_history": history,
                "stage_status": stage_status,
                "artifacts": artifacts,
                "last_error": None,
            }

        retry_count["build"] = retry_count.get("build", 0) + 1
        history.append({"stage": "build", "status": "failed", "error": build.build_logs})
        stage_status["build"] = "failed"
        return {
            "build": build,
            "current_stage": "build",
            "retry_count": retry_count,
            "review_stage": "build",
            "review_reason": "build stage completed without a successful build",
            "stage_history": history,
            "stage_status": stage_status,
            "artifacts": artifacts,
            "last_error": "build stage completed without a successful build",
        }
    except Exception as error:
        retry_count["build"] = retry_count.get("build", 0) + 1
        history.append({"stage": "build", "status": "failed", "error": str(error)})
        stage_status["build"] = "failed"
        artifacts["build"] = {
            "workspace": str(paths.workspace_root),
            "repo_dir": str(paths.repo_dir),
            "build_dir": str(paths.build_dir),
        }
        return {
            "current_stage": "build",
            "retry_count": retry_count,
            "review_stage": "build",
            "review_reason": "build stage raised an exception",
            "stage_history": history,
            "stage_status": stage_status,
            "artifacts": artifacts,
            "last_error": str(error),
        }


def parse_llm_json_payload(content) -> Optional[dict]:
    """Parse a JSON object from an LLM response payload."""

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            elif isinstance(item, str):
                text_parts.append(item)
        content = "\n".join(text_parts)

    if not isinstance(content, str):
        return None

    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        return json.loads(stripped)
    except Exception:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
