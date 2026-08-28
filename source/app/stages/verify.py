"""文件说明：验证阶段实现（差分验证 agent）。

执行模式（已锁定）：
- 不重新 clone 仓库，不重新构建第二个镜像
- 复用 BuildArtifact.docker_image_tag，挂载 workspace
- 两次独立的 docker run --rm，环境变量 PATCH_MODE=pre|post
- pre 模式：git reset --hard && bash build.sh && trigger
- post 模式：git reset --hard && git apply patch.diff && bash build.sh && trigger
  （若 build 注入了脆弱依赖前缀如 HDF5 1.12.0：跳过 in-tree patch，post 重建改链固定依赖）
- 两次执行的日志契约与 PoC 一致（target_binary= / execution_exit_code= / stdout_begin/end 等）
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import yaml
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except ModuleNotFoundError:  # pragma: no cover
    Environment = None
    FileSystemLoader = None
    StrictUndefined = None

from app.schemas.build_artifact import BuildArtifact
from app.schemas.knowledge import KnowledgeModel
from app.schemas.poc_artifact import PoCArtifact
from app.schemas.verify_result import VerifyResult
from app.stages.asan_hardening import ensure_no_pie_for_asan
from app.stages.build import BuildStage
from app.tools.docker_tools import DockerRunRequest, DockerTool
from app.tools.file_tools import FileTool
from app.tools.log_parsing import (
    extract_block,
    extract_execution_observation,
    haystack_has_specific_sanitizer_bug,
    match_patterns,
    should_outer_asan_preload,
    specific_sanitizer_bugs_in,
)
from app.tools.patch_tools import find_patch_diff, strip_unapplyable_binary_stub_hunks
from app.tools.process_tools import ProcessTool


class VerifyStagePaths:
    """Centralized path layout for the verify stage."""

    def __init__(self, workspace: str) -> None:
        self.workspace_root = Path(workspace)
        self.repo_dir = self.workspace_root / "repo"
        self.artifacts_dir = self.workspace_root / "artifacts"
        self.build_dir = self.artifacts_dir / "build"
        self.poc_dir = self.artifacts_dir / "poc"
        self.verify_dir = self.artifacts_dir / "verify"

        # 输入参考
        self.run_verify_yaml = self.poc_dir / "run_verify.yaml"
        self.poc_artifact_yaml = self.poc_dir / "poc_artifact.yaml"
        self.build_artifact_yaml = self.build_dir / "build_artifact.yaml"

        # 输出
        self.verify_dockerfile = self.verify_dir / "Dockerfile"
        self.verify_run_script = self.verify_dir / "verify_run.sh"
        self.patch_diff_copy = self.verify_dir / "patch.diff"
        self.pre_patch_log = self.verify_dir / "pre_patch.log"
        self.post_patch_log = self.verify_dir / "post_patch.log"
        self.verify_context_yaml = self.verify_dir / "verify_context.yaml"
        self.verify_plan_yaml = self.verify_dir / "verify_plan.yaml"
        self.verify_result_yaml = self.verify_dir / "verify_result.yaml"


class VerifyContext(BaseModel):
    """Evidence aggregated from prior stages, used to plan verify."""

    cve_id: str
    docker_image_tag: str
    chosen_vulnerable_ref: str
    chosen_fixed_ref: str
    project_dir_inside_image: str = Field(
        default="",
        description="容器内仓库路径，来源于 BuildArtifact 推导或默认 ${PROJECT_DIR}",
    )
    build_script_in_image: str = Field(
        default="/workspace/artifacts/build/build.sh",
        description="容器内可调用的 build.sh 路径",
    )
    target_binary: str
    trigger_command: str
    expected_stdout_patterns: List[str] = Field(default_factory=list)
    expected_stderr_patterns: List[str] = Field(default_factory=list)
    expected_stack_keywords: List[str] = Field(default_factory=list)
    expected_exit_code: Optional[int] = None
    expected_crash_type: str = ""
    patch_diff_path: str
    poc_run_verify_eligible: bool = True
    poc_run_verify_reason: str = ""
    environment_variables: Dict[str, str] = Field(
        default_factory=dict,
        description="PoC 持久化的环境变量；verify 必须复用",
    )
    trigger_mode: str = Field(
        default="",
        description="PoC trigger mode; library-harness skips outer ASan LD_PRELOAD.",
    )
    dependency_fix_vulnerable_hdf5: bool = Field(
        default=False,
        description=(
            "True when build.sh gated vulnerable HDF5 (/opt/hdf5-vuln). "
            "Post-pass skips in-tree patch.diff and rebuilds against fixed HDF5 1.12.1."
        ),
    )


class VerifyPlan(BaseModel):
    """Deterministic execution plan for pre/post verify runs."""

    image_tag: str
    pre_run_command: str
    post_run_command: str
    environment_variables: Dict[str, str] = Field(default_factory=dict)
    expected_stdout_patterns: List[str] = Field(default_factory=list)
    expected_stderr_patterns: List[str] = Field(default_factory=list)
    expected_stack_keywords: List[str] = Field(default_factory=list)
    expected_exit_code: Optional[int] = None
    expected_crash_type: str = ""
    patch_apply_command: str = "git apply /workspace/artifacts/verify/patch.diff"
    repo_reset_command: str = "git reset --hard && git clean -fd"
    rebuild_command: str = "bash /workspace/artifacts/build/build.sh"
    post_rebuild_command: str = "bash /workspace/artifacts/build/build.sh"
    pre_log_path: str
    post_log_path: str


class VerifyPreparedRun(BaseModel):
    """Deterministic inputs assembled before verify planning starts."""

    context: VerifyContext


class VerifyExecutionOutcome(BaseModel):
    """One concrete verify execution outcome."""

    plan: VerifyPlan
    execute_result: dict[str, Any]


class VerifyGraphState(TypedDict, total=False):
    """Internal LangGraph state for the verify stage."""

    knowledge: KnowledgeModel
    build: BuildArtifact
    poc: PoCArtifact
    paths: VerifyStagePaths
    dataset_root: Optional[str]
    prepared: VerifyPreparedRun
    plan: VerifyPlan
    outcome: VerifyExecutionOutcome


class VerifyPlanner:
    """Encapsulates verify-stage planning decisions."""

    def __init__(self, stage: "VerifyStage") -> None:
        self.stage = stage

    def plan(self, context: VerifyContext, paths: VerifyStagePaths) -> VerifyPlan:
        rebuild = f"bash {context.build_script_in_image}"
        post_rebuild = rebuild
        patch_apply = "git apply /workspace/artifacts/verify/patch.diff"
        if context.dependency_fix_vulnerable_hdf5:
            # Dependency CVE: authoritative fix is swapping HDF5 1.12.0 -> 1.12.1.
            # In-tree advisory YAML / docs patches must not gate post cleanliness.
            patch_apply = (
                "echo '[verify] deeprepro:dependency-fix skip in-tree patch "
                "(vulnerable HDF5 gate)' && true"
            )
            post_rebuild = BuildStage.rewrite_build_script_for_fixed_hdf5(
                context.build_script_in_image
            )
            (
                stderr_patterns,
                stack_keywords,
                crash_type,
                env,
            ) = BuildStage.apply_vulnerable_hdf5_cve_match_policy(
                expected_stderr_patterns=list(context.expected_stderr_patterns),
                expected_stack_keywords=list(context.expected_stack_keywords),
                expected_crash_type=context.expected_crash_type,
                environment_variables=dict(context.environment_variables),
            )
        else:
            stderr_patterns = list(context.expected_stderr_patterns)
            stack_keywords = list(context.expected_stack_keywords)
            crash_type = context.expected_crash_type
            env = dict(context.environment_variables)
        return VerifyPlan(
            image_tag=context.docker_image_tag,
            pre_run_command=context.trigger_command,
            post_run_command=context.trigger_command,
            environment_variables=env,
            expected_stdout_patterns=list(context.expected_stdout_patterns),
            expected_stderr_patterns=stderr_patterns,
            expected_stack_keywords=stack_keywords,
            expected_exit_code=context.expected_exit_code,
            expected_crash_type=crash_type,
            patch_apply_command=patch_apply,
            rebuild_command=rebuild,
            post_rebuild_command=post_rebuild,
            pre_log_path=str(paths.pre_patch_log),
            post_log_path=str(paths.post_patch_log),
        )


class VerifyStage:
    """Vulnerability verification orchestrator."""

    def __init__(
        self,
        docker_tool: DockerTool | None = None,
        process_tool: ProcessTool | None = None,
        file_tool: FileTool | None = None,
    ) -> None:
        self.process_tool = process_tool or ProcessTool()
        self.docker_tool = docker_tool or DockerTool(process_tool=self.process_tool)
        self.file_tool = file_tool or FileTool()
        self.planner = VerifyPlanner(self)

    # ----- 公共入口 -----
    def run(
        self,
        knowledge: KnowledgeModel,
        build: BuildArtifact,
        poc: PoCArtifact,
        workspace: str,
        dataset_root: Optional[str] = None,
    ) -> VerifyResult:
        paths = VerifyStagePaths(workspace)
        self.file_tool.ensure_dir(str(paths.verify_dir))

        try:
            prepared = self.prepare_verify_run(
                knowledge, build, poc, paths, dataset_root=dataset_root
            )
        except Exception as error:
            return self._stage_exception("collect_verify_context", error, paths)

        short_circuit = self._short_circuit_if_ineligible(prepared.context, paths)
        if short_circuit is not None:
            self.persist_verify_result(short_circuit, paths, "verify_result.yaml (short-circuit)")
            return short_circuit

        try:
            subgraph = self.build_internal_graph()
            result = subgraph.invoke(
                {
                    "paths": paths,
                    "prepared": prepared,
                }
            )
            outcome = result["outcome"]
        except Exception as error:
            where = "execute_verify" if "execute_verify" in str(error) else "plan_verify"
            return self._stage_exception(where, error, paths)

        try:
            result = self.decide_verify_result(outcome, prepared.context)
        except Exception as error:
            return self._stage_exception("decide_verdict", error, paths)

        self.persist_verify_result(result, paths, "verify_result.yaml")
        return result

    def build_internal_graph(self):
        """Build the internal LangGraph subgraph for the verify stage."""

        builder = StateGraph(VerifyGraphState)
        builder.add_node("plan", self._verify_graph_plan_node)
        builder.add_node("execute", self._verify_graph_execute_node)
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "execute")
        builder.add_edge("execute", END)
        # False: do not inherit the parent workflow checkpointer. Internal state
        # carries path helper objects that are not msgpack-serializable.
        return builder.compile(checkpointer=False)

    def _verify_graph_plan_node(self, state: VerifyGraphState) -> VerifyGraphState:
        plan = self.plan_verify(state["prepared"].context, state["paths"])
        self._safe_persist_yaml(
            state["paths"].verify_plan_yaml,
            plan.model_dump(mode="json"),
            "verify_plan.yaml",
        )
        return {"plan": plan}

    def _verify_graph_execute_node(self, state: VerifyGraphState) -> VerifyGraphState:
        context = state["prepared"].context
        plan = state["plan"]
        paths = state["paths"]
        self._render_dockerfile(context, paths)
        self._render_verify_run_script(context, plan, paths)
        self._copy_patch_diff(context, paths)
        execute_result = self._execute_verify(context, plan, paths)
        return {"outcome": VerifyExecutionOutcome(plan=plan, execute_result=execute_result)}

    def prepare_verify_run(
        self,
        knowledge: KnowledgeModel,
        build: BuildArtifact,
        poc: PoCArtifact,
        paths: VerifyStagePaths,
        dataset_root: Optional[str] = None,
    ) -> VerifyPreparedRun:
        """Collect deterministic verify inputs before any planning starts."""

        self._ensure_verify_inputs(paths, build)
        context = self.collect_verify_context(
            knowledge, build, poc, paths, dataset_root=dataset_root
        )
        self._safe_persist_yaml(
            paths.verify_context_yaml,
            context.model_dump(mode="json"),
            "verify_context.yaml",
        )
        return VerifyPreparedRun(context=context)

    def _ensure_verify_inputs(self, paths: VerifyStagePaths, build: BuildArtifact) -> None:
        """Materialize minimal build artifacts required by verify when the workspace is incomplete."""

        self.file_tool.ensure_dir(str(paths.build_dir))
        build_script_path = paths.build_dir / "build.sh"
        if not build_script_path.exists() and build.build_script_content:
            content = build.build_script_content.rstrip() + "\n"
            self.file_tool.write_text(str(build_script_path), content)

    def plan_and_execute_verify(
        self,
        context: VerifyContext,
        paths: VerifyStagePaths,
    ) -> VerifyExecutionOutcome:
        """Build the verify plan, render artifacts, and execute both passes."""

        try:
            plan = self.plan_verify(context, paths)
        except Exception as error:
            raise RuntimeError(f"plan_verify: {error}") from error

        self._safe_persist_yaml(
            paths.verify_plan_yaml,
            plan.model_dump(mode="json"),
            "verify_plan.yaml",
        )

        try:
            self._render_dockerfile(context, paths)
            self._render_verify_run_script(context, plan, paths)
            self._copy_patch_diff(context, paths)
        except Exception as error:
            raise RuntimeError(f"render_or_copy: {error}") from error

        try:
            execute_result = self._execute_verify(context, plan, paths)
        except Exception as error:
            raise RuntimeError(f"execute_verify: {error}") from error

        return VerifyExecutionOutcome(plan=plan, execute_result=execute_result)

    def decide_verify_result(
        self,
        outcome: VerifyExecutionOutcome,
        context: VerifyContext,
    ) -> VerifyResult:
        """Decide the final verify result from execution outcomes."""

        return self._decide_verdict(outcome.execute_result, context)

    def persist_verify_result(self, result: VerifyResult, paths: VerifyStagePaths, description: str) -> None:
        """Persist the final verify result."""

        self._safe_persist_yaml(
            paths.verify_result_yaml,
            result.model_dump(mode="json"),
            description,
        )

    # ----- 五层结构 -----
    def collect_verify_context(
        self,
        knowledge: KnowledgeModel,
        build: BuildArtifact,
        poc: PoCArtifact,
        paths: VerifyStagePaths,
        dataset_root: Optional[str] = None,
    ) -> VerifyContext:
        """Collect evidence for verify planning."""

        search_roots = [dataset_root] if dataset_root else None
        patch_diff = find_patch_diff(knowledge.cve_id, search_roots=search_roots)
        patch_diff_path = str(patch_diff) if patch_diff is not None else ""

        eligible, reason = self._read_run_verify_yaml(paths)

        # PoC plan fields with backward-compatible fallbacks
        expected_stack_keywords = list(poc.expected_stack_keywords) or list(knowledge.expected_stack_keywords)
        expected_crash_type = poc.expected_crash_type or self._infer_crash_type_fallback(poc)
        environment_variables = dict(poc.environment_variables)
        expected_stderr_patterns = list(poc.expected_stderr_patterns)
        dependency_fix_vulnerable_hdf5 = self._detect_vulnerable_hdf5_dependency_fix(
            paths, build
        )
        if dependency_fix_vulnerable_hdf5:
            (
                expected_stderr_patterns,
                expected_stack_keywords,
                expected_crash_type,
                environment_variables,
            ) = BuildStage.apply_vulnerable_hdf5_cve_match_policy(
                expected_stderr_patterns=expected_stderr_patterns,
                expected_stack_keywords=expected_stack_keywords,
                expected_crash_type=expected_crash_type,
                environment_variables=environment_variables,
            )

        return VerifyContext(
            cve_id=knowledge.cve_id,
            docker_image_tag=build.compiled_image_tag or build.docker_image_tag or "",
            chosen_vulnerable_ref=build.chosen_vulnerable_ref or "",
            chosen_fixed_ref=build.chosen_fixed_ref or "",
            project_dir_inside_image=self._resolve_project_dir(build),
            build_script_in_image="/workspace/artifacts/build/build.sh",
            target_binary=poc.target_binary or build.binary_or_entrypoint or build.expected_binary_path or "",
            trigger_command=poc.trigger_command or "",
            expected_stdout_patterns=list(poc.expected_stdout_patterns),
            expected_stderr_patterns=expected_stderr_patterns,
            expected_stack_keywords=expected_stack_keywords,
            expected_exit_code=poc.expected_exit_code,
            expected_crash_type=expected_crash_type,
            patch_diff_path=patch_diff_path,
            poc_run_verify_eligible=eligible,
            poc_run_verify_reason=reason,
            environment_variables=environment_variables,
            trigger_mode=poc.trigger_mode or "",
            dependency_fix_vulnerable_hdf5=dependency_fix_vulnerable_hdf5,
        )

    def _detect_vulnerable_hdf5_dependency_fix(
        self,
        paths: VerifyStagePaths,
        build: BuildArtifact,
    ) -> bool:
        """True when build stage gated vulnerable HDF5 into build.sh."""

        marker = BuildStage.VULN_HDF5_MARKER
        texts: list[str] = []
        build_script = paths.build_dir / "build.sh"
        if build_script.is_file():
            try:
                texts.append(build_script.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
        if build.build_script_content:
            texts.append(build.build_script_content)
        return any(marker in text for text in texts)

    def _infer_crash_type_fallback(self, poc: PoCArtifact) -> str:
        """Fallback for legacy poc_artifact.yaml that lacks expected_crash_type."""
        return poc.observed_crash_type or ""

    def _resolve_project_dir(self, build: BuildArtifact) -> str:
        """Determine the in-container project directory.

        优先级：
          1. CMake out-of-tree plugin/binary: ``/src/<proj>/build/...`` → ``/src/<proj>``
             (do not strip two path components; that yields ``.../build/src`` for
             ``.../build/src/imageformats/kimg_*.so`` and verify then ``cd`` into
             a directory that ``git clean`` deletes).
          2. ``.../src/<binary>`` such as ``/opt/lua-5.4.4/src/lua`` → prefix
          3. 默认 ${PROJECT_DIR}（依赖镜像导出此环境变量）
        """

        binary = (
            (build.binary_or_entrypoint or build.expected_binary_path or "")
            .strip()
            .replace("\\", "/")
        )
        if binary.startswith("/"):
            parts = [part for part in binary.rstrip("/").split("/") if part]
            if "build" in parts:
                idx = parts.index("build")
                if idx >= 1:
                    inferred = "/" + "/".join(parts[:idx])
                    if inferred != "/":
                        return inferred
            if len(parts) >= 3 and parts[-2] == "src":
                inferred = "/" + "/".join(parts[:-2])
                if inferred != "/":
                    return inferred
            if len(parts) >= 3:
                inferred = "/" + "/".join(parts[:-2])
                if inferred != "/":
                    return inferred
        dockerfile = build.dockerfile_content or ""
        env_match = re.search(r"(?im)^ENV\s+PROJECT_DIR=(\S+)", dockerfile)
        if env_match:
            inferred = env_match.group(1).strip().strip("\"'")
            if inferred and inferred != "${PROJECT_DIR}":
                return inferred
        workdir_match = re.search(r"(?im)^WORKDIR\s+(/src/\S+)", dockerfile)
        if workdir_match:
            return workdir_match.group(1).rstrip("/")
        return "${PROJECT_DIR}"

    def plan_verify(self, context: VerifyContext, paths: VerifyStagePaths) -> VerifyPlan:
        """Build a deterministic pre/post execution plan."""

        return self.planner.plan(context, paths)

    def _execute_verify(
        self,
        context: VerifyContext,
        plan: VerifyPlan,
        paths: VerifyStagePaths,
    ) -> dict:
        pre_result = self._run_one_pass("pre", context, plan, paths)
        post_result = self._run_one_pass("post", context, plan, paths)
        return {"pre": pre_result, "post": post_result}

    def _decide_verdict(self, execute_result: dict, context: VerifyContext) -> VerifyResult:
        pre = execute_result["pre"]
        post = execute_result["post"]

        # patch apply 失败 → inconclusive
        patch_apply_exit_code = post.get("patch_apply_exit_code")
        patch_apply_success = patch_apply_exit_code == 0
        if patch_apply_exit_code is not None and not patch_apply_success:
            return self._build_inconclusive_result(
                pre, post, context,
                reason=f"patch_apply_failed: exit_code={patch_apply_exit_code}",
                evidence_summary="post mode failed at git apply step; pre/post comparison skipped.",
            )

        # build rebuild 失败 → inconclusive
        # pre 失败：漏洞态没办法执行 trigger
        # post 失败：patch 后无法编译，post_clean 不可信
        pre_rebuild = pre.get("build_rebuild_exit_code")
        if pre_rebuild is not None and pre_rebuild != 0:
            return self._build_inconclusive_result(
                pre, post, context,
                reason=f"pre_rebuild_failed: exit_code={pre_rebuild}",
                evidence_summary="Vulnerable rebuild failed; trigger never executed in pre mode.",
            )
        post_rebuild = post.get("build_rebuild_exit_code")
        if post_rebuild is not None and post_rebuild != 0:
            return self._build_inconclusive_result(
                pre, post, context,
                reason=f"post_rebuild_failed: exit_code={post_rebuild}",
                evidence_summary="Patched rebuild failed; post-patch trigger result is not trustworthy.",
            )

        # 任何一次脚本不 well_formed → inconclusive
        if not pre.get("log_well_formed") or not post.get("log_well_formed"):
            return self._build_inconclusive_result(
                pre, post, context,
                reason="log_not_well_formed: pre={}, post={}".format(
                    pre.get("log_well_formed"), post.get("log_well_formed")
                ),
                evidence_summary="At least one pass produced an incomplete log; cannot compare reliably.",
            )

        # pre / post 触发判定
        pre_triggered = self._is_triggered(pre, context)
        post_triggered = self._is_triggered(post, context)
        post_clean = not post_triggered

        if pre_triggered and post_clean:
            verdict = "success"
            reason = "pre triggered, post clean — vulnerability reproduced and patch verified"
            confidence = self._compute_confidence(pre, post)
            evidence_summary = (
                f"pre crash_type={pre.get('crash_type')}, exit_code={pre.get('exit_code')}; "
                f"post exit_code={post.get('exit_code')}."
            )
        elif not pre_triggered:
            verdict = "failed"
            reason = "pre_not_triggered"
            confidence = "low"
            evidence_summary = "pre mode did not exhibit any expected error/keyword/crash signal."
        elif not post_clean:
            verdict = "failed"
            reason = "post_still_triggered"
            confidence = "medium"
            evidence_summary = "post mode still showed trigger signals after applying patch."
        else:
            return self._build_inconclusive_result(
                pre, post, context,
                reason="unexpected_state",
                evidence_summary="Decision logic reached an unexpected branch.",
            )

        return self._build_result(
            pre=pre,
            post=post,
            verdict=verdict,
            reason=reason,
            confidence=confidence,
            evidence_summary=evidence_summary,
            patch_apply_success=patch_apply_success,
            pre_triggered=pre_triggered,
            post_clean=post_clean,
        )

    # ----- 工具方法 -----
    def _render_dockerfile(self, context: VerifyContext, paths: VerifyStagePaths) -> None:
        rendered = self._render_template(
            "verify.Dockerfile.j2",
            {"base_image_tag": context.docker_image_tag},
        )
        # Dockerfile 是致命落盘，落盘失败要抛
        self.file_tool.write_text(str(paths.verify_dockerfile), rendered)

    def _render_verify_run_script(
        self,
        context: VerifyContext,
        plan: VerifyPlan,
        paths: VerifyStagePaths,
    ) -> None:
        # WSL2 (vm.mmap_rnd_bits=32): an ASan PoC compiled as PIE can bare-crash
        # (exit 139) during sanitizer startup; inject -no-pie into any -fsanitize
        # compile/link command so the trigger is deterministic pre and post.
        run_command = ensure_no_pie_for_asan(plan.pre_run_command)
        rendered = self._render_template(
            "verify_run.sh.j2",
            {
                "target_binary": context.target_binary,
                "run_command": run_command,
                "run_command_shell": self._shell_quote(run_command or "true"),
                "trigger_mode": context.trigger_mode,
                "outer_asan_preload": should_outer_asan_preload(
                    run_command, context.trigger_mode
                ),
                "trigger_timeout_sec": 120,
                "repo_reset_command": plan.repo_reset_command,
                "rebuild_command": plan.rebuild_command
                or f"bash {context.build_script_in_image}",
                "post_rebuild_command": plan.post_rebuild_command
                or plan.rebuild_command
                or f"bash {context.build_script_in_image}",
                "patch_apply_command": plan.patch_apply_command,
                "project_dir_var": context.project_dir_inside_image or "${PROJECT_DIR}",
            },
        )
        self.file_tool.write_text(str(paths.verify_run_script), rendered)

    @staticmethod
    def _shell_quote(value: str) -> str:
        return "'" + value.replace("'", "'\"'\"'") + "'"

    def _copy_patch_diff(self, context: VerifyContext, paths: VerifyStagePaths) -> None:
        src = Path(context.patch_diff_path)
        if not src.exists():
            raise FileNotFoundError(f"patch.diff vanished: {src}")
        raw = src.read_text(encoding="utf-8", errors="replace")
        # Gate: drop ``Binary files ... differ`` stubs that lack GIT binary
        # payloads so source hunks can still ``git apply`` (qpdf corpus extras).
        filtered, dropped = strip_unapplyable_binary_stub_hunks(raw)
        self.file_tool.write_text(str(paths.patch_diff_copy), filtered)
        if dropped:
            note_path = paths.verify_dir / "patch_filter_notes.txt"
            note = (
                "deeprepro:stripped-unapplyable-binary-stubs\n"
                + "\n".join(f"- {path}" for path in dropped)
                + "\n"
            )
            self.file_tool.write_text(str(note_path), note)

    def _run_one_pass(
        self,
        mode: str,
        context: VerifyContext,
        plan: VerifyPlan,
        paths: VerifyStagePaths,
    ) -> dict:
        """Run verify_run.sh once with PATCH_MODE=mode and return parsed observation."""

        env = dict(plan.environment_variables)
        env["PATCH_MODE"] = mode
        request = DockerRunRequest(
            image_tag=plan.image_tag,
            command=["bash", "/workspace/artifacts/verify/verify_run.sh"],
            workspace=str(paths.workspace_root.resolve()),
            environment=env,
        )
        docker_result = self.docker_tool.run_container(request)
        full_log = self._compose_pass_log(docker_result, mode, request)

        log_path = paths.pre_patch_log if mode == "pre" else paths.post_patch_log
        self.file_tool.safe_persist(
            str(log_path),
            full_log,
            description=f"{mode}_patch.log",
        )

        observation = extract_execution_observation(full_log)
        # Stream-aware matching: stdout patterns only in stdout, stderr patterns only in stderr.
        # Stack keywords still searched in the merged text (stack frames may land on either stream).
        matched_stdout_patterns = match_patterns(observation["observed_stdout"], plan.expected_stdout_patterns)
        matched_stderr_patterns = match_patterns(observation["observed_stderr"], plan.expected_stderr_patterns)
        matched_stack = match_patterns(
            observation["observed_stdout"] + "\n" + observation["observed_stderr"],
            plan.expected_stack_keywords,
        )

        required_markers = ("stdout_begin", "stdout_end", "stderr_begin", "stderr_end")
        log_well_formed = all(marker in full_log for marker in required_markers)
        script_finished = "execution_exit_code=" in full_log

        patch_apply_exit_code = (
            self._parse_patch_apply_exit_code(full_log) if mode == "post" else None
        )
        build_rebuild_exit_code = self._parse_build_rebuild_exit_code(full_log)

        return {
            "exit_code": observation["observed_exit_code"],
            "stdout": observation["observed_stdout"],
            "stderr": observation["observed_stderr"],
            "crash_type": observation["observed_crash_type"],
            "matched_error_patterns": list(matched_stderr_patterns),  # 向后兼容别名
            "matched_stdout_patterns": matched_stdout_patterns,
            "matched_stderr_patterns": matched_stderr_patterns,
            "matched_stack_keywords": matched_stack,
            "patch_apply_exit_code": patch_apply_exit_code,
            "build_rebuild_exit_code": build_rebuild_exit_code,
            "log_path": str(log_path),
            "raw_log": full_log,
            "log_well_formed": log_well_formed,
            "script_finished": script_finished,
        }

    def _compose_pass_log(self, docker_result, mode: str, request: DockerRunRequest) -> str:
        parts = [
            f"=== verify pass: {mode} ===",
            f"image_tag={request.image_tag}",
            f"command={request.command}",
            f"workspace={request.workspace}",
            f"environment={request.environment}",
            f"docker_exit_code={docker_result.exit_code}",
            f"docker_success={docker_result.success}",
            "=== stdout ===",
            docker_result.stdout,
            "=== stderr ===",
            docker_result.stderr,
        ]
        return "\n".join(parts)

    def _safe_persist_yaml(self, path: Path, payload: dict[str, Any], description: str) -> None:
        """Persist YAML using one consistent formatting policy."""

        self.file_tool.safe_persist(
            str(path),
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            description=description,
        )

    def _parse_patch_apply_exit_code(self, log: str) -> Optional[int]:
        match = re.search(r"^patch_apply_exit_code=(-?\d+)$", log, re.MULTILINE)
        return int(match.group(1)) if match else None

    def _parse_build_rebuild_exit_code(self, log: str) -> Optional[int]:
        match = re.search(r"^build_rebuild_exit_code=(-?\d+)$", log, re.MULTILINE)
        return int(match.group(1)) if match else None

    def _short_circuit_if_ineligible(
        self,
        context: VerifyContext,
        paths: VerifyStagePaths,
    ) -> Optional[VerifyResult]:
        # 基础前提缺失：始终 inconclusive（与 run_verify 无关）
        if not context.patch_diff_path:
            return self._build_short_circuit_result(
                context=context,
                verdict="inconclusive",
                reason="short_circuit:patch_diff_not_found",
                evidence_summary="Skipped pre/post execution because patch.diff is unavailable.",
            )
        if not context.docker_image_tag:
            return self._build_short_circuit_result(
                context=context,
                verdict="inconclusive",
                reason="short_circuit:docker_image_tag_missing_in_build_artifact",
                evidence_summary="Skipped pre/post execution because BuildArtifact has no docker_image_tag.",
            )

        # PoC run_verify 短路：按原因分流
        if not context.poc_run_verify_eligible:
            run_verify_reason = (context.poc_run_verify_reason or "").strip()
            verdict, reason, evidence_summary = self._classify_short_circuit_from_run_verify(run_verify_reason)
            return self._build_short_circuit_result(
                context=context,
                verdict=verdict,
                reason=reason,
                evidence_summary=evidence_summary,
            )

        return None

    def _classify_short_circuit_from_run_verify(self, run_verify_reason: str) -> tuple[str, str, str]:
        """Map run_verify eligibility_reason to (verdict, reason, evidence_summary).

        分流规则：
        - script_did_not_finish / log_not_well_formed → inconclusive（PoC 自身不可信）
        - no_target_behavior_observed → failed:pre_not_triggered（短路同意 PoC 的"没触发"判断）
        - 其他 → inconclusive（兜底保守）
        """
        if run_verify_reason.startswith("script_did_not_finish"):
            return (
                "inconclusive",
                "short_circuit:script_did_not_finish",
                "PoC script did not complete; verify cannot judge based on incomplete evidence.",
            )
        if run_verify_reason.startswith("log_not_well_formed"):
            return (
                "inconclusive",
                "short_circuit:log_not_well_formed",
                "PoC log markers incomplete; verify cannot judge.",
            )
        if run_verify_reason.startswith("no_target_behavior_observed"):
            return (
                "failed",
                "short_circuit:pre_not_triggered",
                "PoC executed cleanly with no expected behavior observed; verify agrees.",
            )
        if run_verify_reason.startswith("signal_exit_observed"):
            return (
                "inconclusive",
                "short_circuit:signal_exit_observed",
                "PoC observed a crash-like signal exit; verify should not conclude pre_not_triggered from that alone.",
            )
        return (
            "inconclusive",
            f"short_circuit:unknown_eligibility_reason:{run_verify_reason}",
            f"Unrecognized run_verify reason ({run_verify_reason}); defaulting to inconclusive.",
        )

    def _build_short_circuit_result(
        self,
        context: VerifyContext,
        verdict: str,
        reason: str,
        evidence_summary: str,
    ) -> VerifyResult:
        """Construct a short-circuit VerifyResult (pre/post fields all default).

        与 _build_inconclusive_result 不同：本方法不需要 pre/post dict，
        用于尚未跑 docker 的短路场景。verdict 可以是 inconclusive 或 failed。
        pre_patch_triggered 永远是 False——短路时 verify 没真跑 pre，不能声称触发。
        """

        return VerifyResult(
            pre_patch_triggered=False,
            post_patch_clean=False,
            verdict=verdict,
            reason=reason,
            confidence="low",
            evidence_summary=evidence_summary,
        )

    def _is_triggered(self, pass_result: dict, context: VerifyContext) -> bool:
        """A pass is 'triggered' if expected vulnerability signals are observed.

        When knowledge/plan includes strong crash/sanitizer expectations (ASan,
        SEGV, heap-buffer-overflow, …), only those strong matches count.
        Soft diagnostic strings (e.g. ``Wrong duration in voice overlay``,
        function names like ``calculate_beam``) often survive a correct patch
        and must not alone mark post as still triggered.
        """

        if self._expected_requires_strong_trigger(context):
            return self._has_strong_trigger_match(pass_result, context)

        # Legacy / soft-only expectations: any matched pattern counts.
        if pass_result.get("matched_error_patterns") or pass_result.get("matched_stderr_patterns"):
            return True
        if pass_result.get("matched_stdout_patterns"):
            return True
        if pass_result.get("matched_stack_keywords"):
            return True
        crash_type = pass_result.get("crash_type", "") or ""
        expected = (context.expected_crash_type or "").strip().lower()
        if expected and crash_type and (expected in crash_type.lower() or crash_type.lower() in expected):
            return True
        if context.expected_exit_code is not None and pass_result.get("exit_code") == context.expected_exit_code:
            return True
        return False

    # Sanitizer / fatal-crash tokens. Substring match, case-insensitive.
    STRONG_TRIGGER_TOKENS: tuple[str, ...] = (
        "addresssanitizer",
        "asan:",
        "ubsan",
        "undefinedbehaviorsanitizer",
        "memorysanitizer",
        "threadsanitizer",
        "segv",
        "segmentation fault",
        "heap-buffer-overflow",
        "stack-buffer-overflow",
        "global-buffer-overflow",
        "heap-use-after-free",
        "stack-use-after-free",
        "use-after-poison",
        "use-after-free",
        "double-free",
        "negative-size-param",
        "alloc-dealloc-mismatch",
        "runtime error:",
        "aborting",
    )

    def _is_strong_trigger_token(self, text: str) -> bool:
        lowered = (text or "").strip().lower()
        if not lowered:
            return False
        return any(token in lowered for token in self.STRONG_TRIGGER_TOKENS)

    def _expected_requires_strong_trigger(self, context: VerifyContext) -> bool:
        """True when plan expects at least one strong crash/sanitizer signal."""

        candidates = list(context.expected_stderr_patterns or [])
        candidates.extend(context.expected_stdout_patterns or [])
        if context.expected_crash_type:
            candidates.append(context.expected_crash_type)
        return any(self._is_strong_trigger_token(item) for item in candidates)

    def _iter_matched_signal_texts(self, pass_result: dict) -> list[str]:
        texts: list[str] = []
        for key in (
            "matched_error_patterns",
            "matched_stderr_patterns",
            "matched_stdout_patterns",
            "matched_stack_keywords",
        ):
            texts.extend(str(item) for item in (pass_result.get(key) or []) if item)
        crash_type = pass_result.get("crash_type", "") or ""
        if crash_type:
            texts.append(str(crash_type))
        return texts

    def _expected_signal_texts(self, context: VerifyContext) -> list[str]:
        texts = list(context.expected_stderr_patterns or [])
        texts.extend(context.expected_stdout_patterns or [])
        if context.expected_crash_type:
            texts.append(context.expected_crash_type)
        return texts

    def _has_strong_trigger_match(self, pass_result: dict, context: VerifyContext) -> bool:
        """Triggered under strong-signal policy."""

        expected_texts = self._expected_signal_texts(context)
        # When the plan names a concrete ASan bug (stack-buffer-overflow, …),
        # a generic "AddressSanitizer" hit or unrelated kinds such as
        # alloc-dealloc-mismatch must not count as the CVE trigger.
        if specific_sanitizer_bugs_in(expected_texts):
            haystacks = list(self._iter_matched_signal_texts(pass_result))
            haystacks.append(str(pass_result.get("stderr") or ""))
            haystacks.append(str(pass_result.get("stdout") or ""))
            haystacks.append(str(pass_result.get("crash_type") or ""))
            return haystack_has_specific_sanitizer_bug("\n".join(haystacks), expected_texts)

        if any(self._is_strong_trigger_token(text) for text in self._iter_matched_signal_texts(pass_result)):
            return True
        crash_type = pass_result.get("crash_type", "") or ""
        expected = (context.expected_crash_type or "").strip().lower()
        if (
            expected
            and crash_type
            and self._is_strong_trigger_token(expected)
            and (expected in crash_type.lower() or crash_type.lower() in expected)
        ):
            return True
        # Exit-code-only matches stay out of strong policy: patched runs often
        # still exit non-zero on malformed PoC input without crashing.
        return False

    def _compute_confidence(self, pre: dict, post: dict) -> str:
        """Compute confidence based on signal richness."""

        pre_strong = (
            bool(pre.get("matched_error_patterns"))
            or bool(pre.get("matched_stderr_patterns"))
            or bool(pre.get("matched_stdout_patterns"))
            or bool(pre.get("matched_stack_keywords"))
        )
        post_silent = (
            not post.get("matched_error_patterns")
            and not post.get("matched_stderr_patterns")
            and not post.get("matched_stdout_patterns")
            and not post.get("matched_stack_keywords")
            and not post.get("crash_type")
        )
        if pre_strong and post_silent:
            return "high"
        if pre_strong or post_silent:
            return "medium"
        return "low"

    def _build_inconclusive_result(
        self,
        pre: dict,
        post: dict,
        context: VerifyContext,
        reason: str,
        evidence_summary: str,
    ) -> VerifyResult:
        """Construct an inconclusive VerifyResult with all pre/post fields populated.

        pre/post 的 triggered/clean 状态忠实反映各自数据，不因为 inconclusive 就清零——
        这样上游分析可以看到"虽然 verify 拿不准，但 pre 实际上确实命中了"等细节。
        """

        pre_triggered = self._is_triggered(pre, context)
        post_triggered = self._is_triggered(post, context)
        post_clean = not post_triggered

        patch_apply_exit_code = post.get("patch_apply_exit_code")
        if patch_apply_exit_code is None:
            patch_apply_success = False
        else:
            patch_apply_success = (patch_apply_exit_code == 0)

        return self._build_result(
            pre=pre,
            post=post,
            verdict="inconclusive",
            reason=reason,
            confidence="low",
            evidence_summary=evidence_summary,
            patch_apply_success=patch_apply_success,
            pre_triggered=pre_triggered,
            post_clean=post_clean,
        )

    def _extract_lines_around_marker(self, log: str, marker: str, radius: int = 5) -> str:
        """Return ±radius lines around the first line that starts with marker."""

        if marker not in log:
            return ""
        lines = log.splitlines()
        for idx, line in enumerate(lines):
            if line.startswith(marker):
                start = max(0, idx - radius)
                end = min(len(lines), idx + radius)
                return "\n".join(lines[start:end])
        return ""

    def _extract_patch_apply_log(self, full_log: str) -> str:
        """Extract the patch_apply_stderr block; fall back to ±5 lines around the marker."""

        block = extract_block(full_log, "patch_apply_stderr_begin", "patch_apply_stderr_end")
        if block.strip():
            return block
        return self._extract_lines_around_marker(full_log, "patch_apply_exit_code=", radius=5)

    def _build_result(
        self,
        pre: dict,
        post: dict,
        verdict: str,
        reason: str,
        confidence: str,
        evidence_summary: str,
        patch_apply_success: bool,
        pre_triggered: bool,
        post_clean: bool,
    ) -> VerifyResult:
        # 优先取 matched_stderr_patterns（新字段），回退到 matched_error_patterns（向后兼容）
        pre_matched_stderr = list(pre.get("matched_stderr_patterns") or pre.get("matched_error_patterns") or [])
        pre_matched_stdout = list(pre.get("matched_stdout_patterns") or [])
        pre_matched_stack = list(pre.get("matched_stack_keywords") or [])
        post_matched_stderr = list(post.get("matched_stderr_patterns") or post.get("matched_error_patterns") or [])
        post_matched_stdout = list(post.get("matched_stdout_patterns") or [])
        post_matched_stack = list(post.get("matched_stack_keywords") or [])

        patch_apply_log = self._extract_patch_apply_log(post.get("raw_log") or "")

        return VerifyResult(
            pre_patch_triggered=pre_triggered,
            post_patch_clean=post_clean,
            matched_error_patterns=pre_matched_stderr,
            matched_stack_keywords=pre_matched_stack,
            verdict=verdict,
            reason=reason,
            pre_patch_exit_code=pre.get("exit_code"),
            post_patch_exit_code=post.get("exit_code"),
            pre_patch_observed_stdout=pre.get("stdout") or "",
            pre_patch_observed_stderr=pre.get("stderr") or "",
            post_patch_observed_stdout=post.get("stdout") or "",
            post_patch_observed_stderr=post.get("stderr") or "",
            pre_patch_observed_crash_type=pre.get("crash_type") or "",
            post_patch_observed_crash_type=post.get("crash_type") or "",
            pre_patch_log_path=pre.get("log_path") or "",
            post_patch_log_path=post.get("log_path") or "",
            pre_patch_matched_error_patterns=pre_matched_stderr,
            pre_patch_matched_stack_keywords=pre_matched_stack,
            post_patch_matched_error_patterns=post_matched_stderr,
            post_patch_matched_stack_keywords=post_matched_stack,
            pre_patch_matched_stdout_patterns=pre_matched_stdout,
            pre_patch_matched_stderr_patterns=pre_matched_stderr,
            post_patch_matched_stdout_patterns=post_matched_stdout,
            post_patch_matched_stderr_patterns=post_matched_stderr,
            patch_apply_log=patch_apply_log,
            patch_apply_success=patch_apply_success,
            confidence=confidence,
            evidence_summary=evidence_summary,
        )

    def _read_run_verify_yaml(self, paths: VerifyStagePaths) -> tuple[bool, str]:
        if paths.run_verify_yaml.exists():
            try:
                payload = yaml.safe_load(paths.run_verify_yaml.read_text(encoding="utf-8")) or {}
                eligible = bool(payload.get("eligible_for_verify", True))
                reason = str(payload.get("eligibility_reason", ""))
                return eligible, reason
            except Exception:
                return True, "run_verify.yaml unreadable, treating as eligible"
        return True, "run_verify.yaml not found, treating as eligible (backward compatible)"

    def _stage_exception(self, where: str, error: Exception, paths: VerifyStagePaths) -> VerifyResult:
        result = VerifyResult(
            pre_patch_triggered=False,
            post_patch_clean=False,
            verdict="inconclusive",
            reason=f"stage_exception:{where}: {error}",
            confidence="low",
            evidence_summary=f"verify stage raised in {where}: {error}",
        )
        self.file_tool.safe_persist(
            str(paths.verify_result_yaml),
            yaml.safe_dump(result.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
            description="verify_result.yaml (exception)",
        )
        return result

    def _render_template(self, template_name: str, context: dict[str, Any]) -> str:
        if Environment is not None and FileSystemLoader is not None and StrictUndefined is not None:
            template_dir = Path(__file__).resolve().parents[1] / "templates"
            env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                undefined=StrictUndefined,
                trim_blocks=True,
                lstrip_blocks=True,
            )
            render_ctx = dict(context)
            if template_name == "verify_run.sh.j2":
                run_cmd = render_ctx.get("run_command") or "true"
                render_ctx.setdefault("run_command_shell", self._shell_quote(run_cmd))
                render_ctx.setdefault(
                    "outer_asan_preload",
                    should_outer_asan_preload(run_cmd, str(render_ctx.get("trigger_mode") or "")),
                )
                render_ctx.setdefault("trigger_timeout_sec", 120)
            return env.get_template(template_name).render(**render_ctx).strip() + "\n"

        if template_name == "verify.Dockerfile.j2":
            return f"FROM {context['base_image_tag']}\n\nWORKDIR /workspace\n"
        if template_name == "verify_run.sh.j2":
            return self._render_verify_run_fallback(context)
        raise RuntimeError(f"unsupported template without Jinja2: {template_name}")

    def _render_verify_run_fallback(self, context: dict[str, Any]) -> str:
        return "\n".join(
            [
                "#!/bin/bash",
                "set +e",
                f'PATCH_MODE="${{PATCH_MODE:-pre}}"',
                f'PROJECT_DIR_VAR="{context["project_dir_var"]}"',
                'cd "${PROJECT_DIR_VAR}"',
                f'{context["repo_reset_command"]}',
                'if [[ "${PATCH_MODE}" == "post" ]]; then',
                f'    {context["patch_apply_command"]}',
                '    echo "patch_apply_exit_code=$?"',
                'fi',
                'if [[ "${PATCH_MODE}" == "post" ]]; then',
                f'    {context.get("post_rebuild_command", context["rebuild_command"])}',
                'else',
                f'    {context["rebuild_command"]}',
                'fi',
                'echo "build_rebuild_exit_code=$?"',
                f'echo "target_binary={context["target_binary"]}"',
                f'echo "trigger_command={context["run_command"]}"',
                f'{context["run_command"]}',
                'echo "execution_exit_code=$?"',
                'echo "stdout_begin"; echo "stdout_end"',
                'echo "stderr_begin"; echo "stderr_end"',
                'exit 0',
                "",
            ]
        )


def verify_node(state):
    """LangGraph node: execute verify stage."""

    knowledge = state["knowledge"]
    build = state["build"]
    poc = state["poc"]
    workspace = state["workspace"]
    dataset_root = state.get("dataset_root")
    artifacts = dict(state.get("artifacts", {}))
    stage_status = dict(state.get("stage_status", {}))
    stage = VerifyStage()
    paths = VerifyStagePaths(workspace)

    try:
        verify = stage.run(
            knowledge=knowledge,
            build=build,
            poc=poc,
            workspace=workspace,
            dataset_root=dataset_root,
        )
    except Exception as error:
        verify = VerifyResult(
            pre_patch_triggered=False,
            post_patch_clean=False,
            verdict="inconclusive",
            reason=f"verify_node_exception: {error}",
            confidence="low",
        )

    history = list(state.get("stage_history", []))
    history.append({"stage": "verify", "status": verify.verdict})
    stage_status["verify"] = verify.verdict
    artifacts["verify"] = {
        "verify_dir": str(paths.verify_dir),
        "verify_context_yaml": str(paths.verify_context_yaml),
        "verify_plan_yaml": str(paths.verify_plan_yaml),
        "verify_run_script": str(paths.verify_run_script),
        "patch_diff": str(paths.patch_diff_copy),
        "pre_patch_log": str(paths.pre_patch_log),
        "post_patch_log": str(paths.post_patch_log),
        "verify_result_yaml": str(paths.verify_result_yaml),
    }

    if verify.verdict == "success":
        final_status = "success"
    elif verify.verdict == "inconclusive":
        final_status = "inconclusive"
    else:
        final_status = "failed"

    return {
        "verify": verify,
        "current_stage": "verify",
        "review_stage": "verify" if final_status != "success" else "",
        "human_action_required": final_status != "success",
        "review_reason": verify.reason if final_status != "success" else "",
        "final_status": final_status,
        "stage_history": history,
        "stage_status": stage_status,
        "artifacts": artifacts,
        "last_error": None if final_status == "success" else verify.reason,
    }
