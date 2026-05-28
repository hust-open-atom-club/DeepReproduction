"""Verify stage tests. Validates the differential verification agent."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from app.orchestrator import routers
from app.schemas.build_artifact import BuildArtifact
from app.schemas.knowledge import KnowledgeModel
from app.schemas.poc_artifact import PoCArtifact
from app.schemas.verify_result import VerifyResult
from app.stages import verify as verify_module


def make_context(**overrides):
    payload = {
        "cve_id": "CVE-2022-0000",
        "docker_image_tag": "demo:build",
        "chosen_vulnerable_ref": "abc1234",
        "chosen_fixed_ref": "fff5678",
        "target_binary": "src/lua",
        "trigger_command": "src/lua /workspace/artifacts/poc/payloads/poc.lua",
        "expected_stdout_patterns": [],
        "expected_stderr_patterns": ["heap-buffer-overflow"],
        "expected_stack_keywords": ["singlevar"],
        "expected_exit_code": None,
        "expected_crash_type": "heap-buffer-overflow",
        "patch_diff_path": "/tmp/patch.diff",
        "poc_run_verify_eligible": True,
        "poc_run_verify_reason": "",
    }
    payload.update(overrides)
    return verify_module.VerifyContext(**payload)


def test_prepare_verify_run_prefers_compiled_image_tag(tmp_path):
    stage = verify_module.VerifyStage()
    workspace = tmp_path / "ws"
    paths = verify_module.VerifyStagePaths(str(workspace))
    paths.verify_dir.mkdir(parents=True, exist_ok=True)
    paths.poc_dir.mkdir(parents=True, exist_ok=True)
    paths.build_dir.mkdir(parents=True, exist_ok=True)
    (paths.poc_dir / "run_verify.yaml").write_text("eligible_for_verify: true\n", encoding="utf-8")

    knowledge = KnowledgeModel(cve_id="CVE-2022-0000", summary="demo", vulnerability_type="heap", repo_url="https://example.com/demo.git")
    build = BuildArtifact(
        dockerfile_content="FROM ubuntu:20.04\n",
        build_script_content="#!/bin/bash\n",
        build_success=True,
        build_logs="ok",
        docker_image_tag="demo:build",
        compiled_image_tag="demo:compiled",
    )
    poc = PoCArtifact(poc_filename="poc.txt", poc_content="x", run_script_content="#!/bin/bash\n")

    context = stage.prepare_verify_run(knowledge, build, poc, paths).context

    assert context.docker_image_tag == "demo:compiled"


def test_prepare_verify_run_materializes_missing_build_script(tmp_path):
    stage = verify_module.VerifyStage()
    workspace = tmp_path / "ws"
    paths = verify_module.VerifyStagePaths(str(workspace))
    paths.verify_dir.mkdir(parents=True, exist_ok=True)
    paths.poc_dir.mkdir(parents=True, exist_ok=True)
    (paths.poc_dir / "run_verify.yaml").write_text("eligible_for_verify: true\n", encoding="utf-8")

    knowledge = KnowledgeModel(cve_id="CVE-2022-0000", summary="demo", vulnerability_type="heap")
    build = BuildArtifact(
        dockerfile_content="FROM ubuntu:20.04\n",
        build_script_content="#!/bin/bash\necho rebuild\n",
        build_success=True,
        build_logs="ok",
        docker_image_tag="demo:build",
    )
    poc = PoCArtifact(poc_filename="poc.txt", poc_content="x", run_script_content="#!/bin/bash\n")

    stage.prepare_verify_run(knowledge, build, poc, paths)

    assert (paths.build_dir / "build.sh").exists()
    assert "echo rebuild" in (paths.build_dir / "build.sh").read_text(encoding="utf-8")


def make_pass(
    exit_code=139,
    stdout="",
    stderr="",
    crash_type="",
    matched_error_patterns=None,
    matched_stack_keywords=None,
    patch_apply_exit_code=None,
    build_rebuild_exit_code=0,
    log_well_formed=True,
    script_finished=True,
    raw_log="",
    log_path="",
):
    return {
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "crash_type": crash_type,
        "matched_error_patterns": matched_error_patterns or [],
        "matched_stack_keywords": matched_stack_keywords or [],
        "patch_apply_exit_code": patch_apply_exit_code,
        "build_rebuild_exit_code": build_rebuild_exit_code,
        "log_path": log_path,
        "raw_log": raw_log,
        "log_well_formed": log_well_formed,
        "script_finished": script_finished,
    }


# ===== Case 1 =====
def test_decide_verdict_success_when_pre_triggered_post_clean():
    stage = verify_module.VerifyStage()
    pre = make_pass(
        exit_code=139,
        crash_type="heap-buffer-overflow",
        matched_error_patterns=["heap-buffer-overflow"],
        matched_stack_keywords=["singlevar"],
    )
    post = make_pass(
        exit_code=0,
        crash_type="",
        matched_error_patterns=[],
        matched_stack_keywords=[],
        patch_apply_exit_code=0,
    )
    context = make_context()

    result = stage._decide_verdict({"pre": pre, "post": post}, context)

    assert result.verdict == "success"
    assert result.confidence == "high"
    assert result.pre_patch_triggered is True
    assert result.post_patch_clean is True


# ===== Case 2 =====
def test_decide_verdict_failed_when_pre_not_triggered():
    stage = verify_module.VerifyStage()
    pre = make_pass(exit_code=0, matched_error_patterns=[], matched_stack_keywords=[])
    post = make_pass(exit_code=0, patch_apply_exit_code=0)
    context = make_context()

    result = stage._decide_verdict({"pre": pre, "post": post}, context)

    assert result.verdict == "failed"
    assert result.reason == "pre_not_triggered"
    assert result.confidence == "low"


# ===== Case 3 =====
def test_decide_verdict_failed_when_post_still_triggered():
    stage = verify_module.VerifyStage()
    pre = make_pass(
        exit_code=139,
        matched_error_patterns=["heap-buffer-overflow"],
    )
    post = make_pass(
        exit_code=139,
        matched_error_patterns=["heap-buffer-overflow"],
        patch_apply_exit_code=0,
    )
    context = make_context()

    result = stage._decide_verdict({"pre": pre, "post": post}, context)

    assert result.verdict == "failed"
    assert result.reason == "post_still_triggered"
    assert result.confidence == "medium"


# ===== Case 4 =====
def test_decide_verdict_inconclusive_when_patch_apply_failed():
    stage = verify_module.VerifyStage()
    pre = make_pass(matched_error_patterns=["heap-buffer-overflow"])
    post = make_pass(patch_apply_exit_code=1)
    context = make_context()

    result = stage._decide_verdict({"pre": pre, "post": post}, context)

    assert result.verdict == "inconclusive"
    assert result.reason.startswith("patch_apply_failed")
    assert result.patch_apply_success is False
    assert result.confidence == "low"


# ===== Case 5 =====
def test_decide_verdict_inconclusive_when_log_not_well_formed():
    stage = verify_module.VerifyStage()
    pre = make_pass(
        log_well_formed=False,
        matched_error_patterns=["heap-buffer-overflow"],
    )
    post = make_pass(patch_apply_exit_code=0)
    context = make_context()

    result = stage._decide_verdict({"pre": pre, "post": post}, context)

    assert result.verdict == "inconclusive"
    assert result.reason.startswith("log_not_well_formed")


# ===== Case 6: short-circuit shared harness + 5 split tests =====
def _setup_short_circuit_workspace(tmp_path, eligibility_reason: str, eligible: bool = False, with_patch: bool = True):
    """Helper: build a workspace + run_verify.yaml + (optional) patch.diff for short-circuit tests.

    Returns (workspace, knowledge, build, poc, fake_docker).
    """

    workspace = tmp_path / "ws"
    paths = verify_module.VerifyStagePaths(str(workspace))
    paths.poc_dir.mkdir(parents=True, exist_ok=True)
    payload = {"eligible_for_verify": eligible}
    if eligibility_reason is not None:
        payload["eligibility_reason"] = eligibility_reason
    paths.run_verify_yaml.write_text(yaml.safe_dump(payload), encoding="utf-8")

    if with_patch:
        # Place a patch.diff under the default Dataset/ root in the cwd so find_patch_diff finds it.
        patch_target = tmp_path / "Dataset" / "CVE-2022-0000" / "vuln_data" / "vuln_diffs" / "patch.diff"
        patch_target.parent.mkdir(parents=True, exist_ok=True)
        patch_target.write_text("--- a\n+++ b\n", encoding="utf-8")

    knowledge = KnowledgeModel(
        cve_id="CVE-2022-0000",
        summary="demo",
        vulnerability_type="heap-overflow",
    )
    build = BuildArtifact(
        dockerfile_content="x",
        build_script_content="y",
        build_success=True,
        docker_image_tag="demo:build",
    )
    poc = PoCArtifact(
        poc_filename="poc.lua",
        poc_content="boom",
        run_script_content="#!/bin/bash\n",
    )
    fake_docker = MagicMock()
    fake_docker.run_container = MagicMock()
    return workspace, knowledge, build, poc, fake_docker


def test_short_circuit_script_did_not_finish_to_inconclusive(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    workspace, knowledge, build, poc, fake_docker = _setup_short_circuit_workspace(
        tmp_path, eligibility_reason="script_did_not_finish: missing execution_exit_code marker"
    )
    stage = verify_module.VerifyStage(docker_tool=fake_docker)
    result = stage.run(knowledge=knowledge, build=build, poc=poc, workspace=str(workspace))

    assert result.verdict == "inconclusive"
    assert result.reason == "short_circuit:script_did_not_finish"
    assert fake_docker.run_container.call_count == 0


def test_short_circuit_no_target_behavior_to_failed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    workspace, knowledge, build, poc, fake_docker = _setup_short_circuit_workspace(
        tmp_path, eligibility_reason="no_target_behavior_observed"
    )
    stage = verify_module.VerifyStage(docker_tool=fake_docker)
    result = stage.run(knowledge=knowledge, build=build, poc=poc, workspace=str(workspace))

    assert result.verdict == "failed"
    assert result.reason == "short_circuit:pre_not_triggered"
    assert fake_docker.run_container.call_count == 0


def test_short_circuit_log_not_well_formed_to_inconclusive(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    workspace, knowledge, build, poc, fake_docker = _setup_short_circuit_workspace(
        tmp_path, eligibility_reason="log_not_well_formed: stdout/stderr block markers missing"
    )
    stage = verify_module.VerifyStage(docker_tool=fake_docker)
    result = stage.run(knowledge=knowledge, build=build, poc=poc, workspace=str(workspace))

    assert result.verdict == "inconclusive"
    assert result.reason == "short_circuit:log_not_well_formed"


def test_short_circuit_unknown_reason_falls_back_to_inconclusive(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    workspace, knowledge, build, poc, fake_docker = _setup_short_circuit_workspace(
        tmp_path, eligibility_reason="some_future_reason_we_dont_know"
    )
    stage = verify_module.VerifyStage(docker_tool=fake_docker)
    result = stage.run(knowledge=knowledge, build=build, poc=poc, workspace=str(workspace))

    assert result.verdict == "inconclusive"
    assert "unknown_eligibility_reason" in result.reason


def test_short_circuit_signal_exit_observed_to_inconclusive(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    workspace, knowledge, build, poc, fake_docker = _setup_short_circuit_workspace(
        tmp_path, eligibility_reason="signal_exit_observed: 139"
    )
    stage = verify_module.VerifyStage(docker_tool=fake_docker)
    result = stage.run(knowledge=knowledge, build=build, poc=poc, workspace=str(workspace))

    assert result.verdict == "inconclusive"
    assert result.reason == "short_circuit:signal_exit_observed"
    assert fake_docker.run_container.call_count == 0


def test_short_circuit_patch_diff_not_found_unchanged(tmp_path, monkeypatch):
    """patch.diff 缺失时短路保持 inconclusive，不依赖 run_verify。"""

    monkeypatch.chdir(tmp_path)
    workspace, knowledge, build, poc, fake_docker = _setup_short_circuit_workspace(
        tmp_path, eligibility_reason="error_pattern_hit: foo", eligible=True, with_patch=False,
    )
    stage = verify_module.VerifyStage(docker_tool=fake_docker)
    result = stage.run(knowledge=knowledge, build=build, poc=poc, workspace=str(workspace))

    assert result.verdict == "inconclusive"
    assert result.reason == "short_circuit:patch_diff_not_found"


def test_short_circuit_pre_patch_triggered_always_false(tmp_path, monkeypatch):
    """短路时 verify 没真跑 pre，pre_patch_triggered 必须始终 False，无论 verdict。"""

    cases = [
        "script_did_not_finish: ...",
        "no_target_behavior_observed",
        "log_not_well_formed: ...",
        "some_unknown_reason",
    ]
    for fixture_reason in cases:
        # Each case needs a fresh workspace so cwd-relative patch.diff lookup is clean.
        monkeypatch.chdir(tmp_path)
        sub = tmp_path / fixture_reason.split(":")[0].replace(" ", "_")
        sub.mkdir(exist_ok=True)
        monkeypatch.chdir(sub)
        workspace, knowledge, build, poc, fake_docker = _setup_short_circuit_workspace(
            sub, eligibility_reason=fixture_reason
        )
        stage = verify_module.VerifyStage(docker_tool=fake_docker)
        result = stage.run(knowledge=knowledge, build=build, poc=poc, workspace=str(workspace))
        assert result.pre_patch_triggered is False, f"failed for: {fixture_reason}"


# ===== Case 7 =====
def test_run_one_pass_parses_pre_log_correctly(tmp_path):
    workspace = tmp_path / "ws"
    paths = verify_module.VerifyStagePaths(str(workspace))
    paths.verify_dir.mkdir(parents=True, exist_ok=True)

    pre_stdout = (
        "build_rebuild_exit_code=0\n"
        "target_binary=src/lua\n"
        "trigger_command=src/lua poc.lua\n"
        "execution_exit_code=139\n"
        "stdout_begin\n\nstdout_end\n"
        "stderr_begin\nAddressSanitizer: heap-buffer-overflow at singlevar\nstderr_end\n"
    )

    fake_docker_result = MagicMock()
    fake_docker_result.stdout = pre_stdout
    fake_docker_result.stderr = ""
    fake_docker_result.exit_code = 0
    fake_docker_result.success = True

    fake_docker = MagicMock()
    fake_docker.run_container = MagicMock(return_value=fake_docker_result)
    stage = verify_module.VerifyStage(docker_tool=fake_docker)

    context = make_context()
    plan = verify_module.VerifyPlan(
        image_tag="demo:build",
        pre_run_command="src/lua poc.lua",
        post_run_command="src/lua poc.lua",
        expected_stderr_patterns=["heap-buffer-overflow"],
        expected_stack_keywords=["singlevar"],
        pre_log_path=str(paths.pre_patch_log),
        post_log_path=str(paths.post_patch_log),
    )

    result = stage._run_one_pass("pre", context, plan, paths)

    assert result["exit_code"] == 139
    assert result["log_well_formed"] is True
    assert result["script_finished"] is True
    assert result["patch_apply_exit_code"] is None
    assert result["build_rebuild_exit_code"] == 0
    assert "heap-buffer-overflow" in result["matched_error_patterns"]
    assert "singlevar" in result["matched_stack_keywords"]


def test_run_one_pass_uses_outer_stderr_for_crash_detection(tmp_path):
    workspace = tmp_path / "ws"
    paths = verify_module.VerifyStagePaths(str(workspace))
    paths.verify_dir.mkdir(parents=True, exist_ok=True)

    fake_docker_result = MagicMock()
    fake_docker_result.stdout = (
        "build_rebuild_exit_code=0\n"
        "target_binary=src/lua\n"
        "trigger_command=src/lua poc.lua\n"
        "execution_exit_code=139\n"
        "stdout_begin\n\nstdout_end\n"
        "stderr_begin\n\nstderr_end\n"
    )
    fake_docker_result.stderr = "Segmentation fault (core dumped)\n"
    fake_docker_result.exit_code = 0
    fake_docker_result.success = True

    fake_docker = MagicMock()
    fake_docker.run_container = MagicMock(return_value=fake_docker_result)
    stage = verify_module.VerifyStage(docker_tool=fake_docker)

    context = make_context(expected_stderr_patterns=["Segmentation fault"], expected_crash_type="segmentation fault")
    plan = verify_module.VerifyPlan(
        image_tag="demo:build",
        pre_run_command="src/lua poc.lua",
        post_run_command="src/lua poc.lua",
        expected_stderr_patterns=["Segmentation fault"],
        expected_crash_type="segmentation fault",
        pre_log_path=str(paths.pre_patch_log),
        post_log_path=str(paths.post_patch_log),
    )

    result = stage._run_one_pass("pre", context, plan, paths)

    assert result["exit_code"] == 139
    assert result["crash_type"] == "segmentation fault"
    assert "Segmentation fault" in result["matched_error_patterns"]


# ===== Case 8 =====
def test_verify_node_returns_inconclusive_on_stage_exception(monkeypatch):
    class FakeStage:
        def run(self, knowledge, build, poc, workspace):
            raise RuntimeError("boom")

    monkeypatch.setattr(verify_module, "VerifyStage", FakeStage)

    state = {
        "knowledge": KnowledgeModel(
            cve_id="CVE-2022-0000",
            summary="demo",
            vulnerability_type="heap-overflow",
        ),
        "build": BuildArtifact(
            dockerfile_content="FROM ubuntu\n",
            build_script_content="#!/bin/bash\n",
            build_success=True,
        ),
        "poc": PoCArtifact(
            poc_filename="poc.lua",
            poc_content="boom",
            run_script_content="#!/bin/bash\n",
        ),
        "workspace": "workspaces/CVE-2022-0000",
        "stage_history": [],
    }

    result = verify_module.verify_node(state)

    assert result["verify"].verdict == "inconclusive"
    assert result["verify"].reason.startswith("verify_node_exception")
    assert result["final_status"] == "inconclusive"


# ===== Case 9 =====
def test_verify_run_template_renders_post_with_git_apply():
    stage = verify_module.VerifyStage()
    rendered = stage._render_template(
        "verify_run.sh.j2",
        {
            "target_binary": "src/lua",
            "run_command": "src/lua poc.lua",
            "repo_reset_command": "git reset --hard && git clean -fd",
            "rebuild_command": "bash /workspace/artifacts/build/build.sh",
            "patch_apply_command": "git apply /workspace/artifacts/verify/patch.diff",
            "project_dir_var": "${PROJECT_DIR}",
        },
    )

    assert "git apply" in rendered
    assert "target_binary=" in rendered
    assert "execution_exit_code=" in rendered
    assert "PATCH_MODE" in rendered


# ===== Case 10 =====
def test_route_after_verify_three_way():
    success_state = {"verify": VerifyResult(
        pre_patch_triggered=True, post_patch_clean=True, verdict="success", reason="ok"
    )}
    failed_state = {"verify": VerifyResult(
        pre_patch_triggered=False, post_patch_clean=True, verdict="failed", reason="x"
    )}
    inconclusive_state = {"verify": VerifyResult(
        pre_patch_triggered=False, post_patch_clean=False, verdict="inconclusive", reason="y"
    )}
    none_state = {}

    assert routers.route_after_verify(success_state) == "success"
    assert routers.route_after_verify(failed_state) == "failed"
    assert routers.route_after_verify(inconclusive_state) == "inconclusive"
    assert routers.route_after_verify(none_state) == "failed"


# ===== Fix 1: build_rebuild failure detection =====
def test_decide_verdict_inconclusive_when_post_rebuild_failed():
    stage = verify_module.VerifyStage()
    pre = make_pass(
        exit_code=139,
        crash_type="heap-buffer-overflow",
        matched_error_patterns=["heap-buffer-overflow"],
        matched_stack_keywords=["singlevar"],
        build_rebuild_exit_code=0,
    )
    post = make_pass(
        exit_code=255,
        crash_type="",
        matched_error_patterns=[],
        matched_stack_keywords=[],
        patch_apply_exit_code=0,
        build_rebuild_exit_code=2,
    )
    context = make_context()

    result = stage._decide_verdict({"pre": pre, "post": post}, context)

    assert result.verdict == "inconclusive"
    assert result.reason.startswith("post_rebuild_failed")
    # 关键：pre 真实命中，必须如实回填，不能因为 post 失败就清零
    assert result.pre_patch_triggered is True


def test_decide_verdict_inconclusive_when_pre_rebuild_failed():
    stage = verify_module.VerifyStage()
    pre = make_pass(
        exit_code=255,
        matched_error_patterns=[],
        build_rebuild_exit_code=1,
    )
    post = make_pass(
        exit_code=0,
        matched_error_patterns=[],
        patch_apply_exit_code=0,
        build_rebuild_exit_code=0,
    )
    context = make_context()

    result = stage._decide_verdict({"pre": pre, "post": post}, context)

    assert result.verdict == "inconclusive"
    assert result.reason.startswith("pre_rebuild_failed")


# ===== Fix 2-3.B: collect_verify_context uses PoC plan fields =====
def test_collect_verify_context_uses_poc_plan_fields(tmp_path):
    workspace = tmp_path / "ws"
    paths = verify_module.VerifyStagePaths(str(workspace))
    paths.poc_dir.mkdir(parents=True, exist_ok=True)

    knowledge = KnowledgeModel(
        cve_id="CVE-2022-0000",
        summary="demo",
        vulnerability_type="heap-overflow",
        repo_url="https://example.com/demo.git",
        vulnerable_ref="abc1234",
        expected_stack_keywords=["knowledge_keyword"],
    )
    build = BuildArtifact(
        dockerfile_content="FROM ubuntu\n",
        build_script_content="#!/bin/bash\n",
        build_success=True,
        docker_image_tag="demo:build",
        chosen_vulnerable_ref="abc1234",
        chosen_fixed_ref="fff5678",
        binary_or_entrypoint="src/lua",
    )
    poc = PoCArtifact(
        poc_filename="poc.lua",
        poc_content="boom",
        run_script_content="#!/bin/bash\n",
        target_binary="src/lua",
        trigger_command="src/lua poc.lua",
        execution_success=True,
        expected_stack_keywords=["singlevar"],
        expected_crash_type="heap-buffer-overflow",
        environment_variables={"ASAN_OPTIONS": "detect_leaks=0"},
    )
    stage = verify_module.VerifyStage()

    context = stage.collect_verify_context(knowledge, build, poc, paths)

    assert context.expected_stack_keywords == ["singlevar"]
    assert context.expected_crash_type == "heap-buffer-overflow"
    assert context.environment_variables == {"ASAN_OPTIONS": "detect_leaks=0"}


def test_collect_verify_context_falls_back_to_knowledge_when_poc_keywords_empty(tmp_path):
    workspace = tmp_path / "ws"
    paths = verify_module.VerifyStagePaths(str(workspace))
    paths.poc_dir.mkdir(parents=True, exist_ok=True)

    knowledge = KnowledgeModel(
        cve_id="CVE-2022-0000",
        summary="demo",
        vulnerability_type="heap-overflow",
        expected_stack_keywords=["foo"],
    )
    build = BuildArtifact(
        dockerfile_content="FROM ubuntu\n",
        build_script_content="#!/bin/bash\n",
        build_success=True,
        docker_image_tag="demo:build",
    )
    poc = PoCArtifact(
        poc_filename="poc.lua",
        poc_content="boom",
        run_script_content="#!/bin/bash\n",
    )
    stage = verify_module.VerifyStage()

    context = stage.collect_verify_context(knowledge, build, poc, paths)

    assert context.expected_stack_keywords == ["foo"]


# ===== Fix 2-3.C: env vars propagated to docker call =====
def test_run_one_pass_propagates_environment_variables(tmp_path):
    workspace = tmp_path / "ws"
    paths = verify_module.VerifyStagePaths(str(workspace))
    paths.verify_dir.mkdir(parents=True, exist_ok=True)

    captured = {}

    def fake_run_container(request):
        captured["request"] = request
        result = MagicMock()
        result.stdout = (
            "build_rebuild_exit_code=0\n"
            "target_binary=src/lua\n"
            "execution_exit_code=0\n"
            "stdout_begin\n\nstdout_end\n"
            "stderr_begin\n\nstderr_end\n"
        )
        result.stderr = ""
        result.exit_code = 0
        result.success = True
        return result

    fake_docker = MagicMock()
    fake_docker.run_container = fake_run_container
    stage = verify_module.VerifyStage(docker_tool=fake_docker)

    context = make_context()
    plan = verify_module.VerifyPlan(
        image_tag="demo:build",
        pre_run_command="src/lua poc.lua",
        post_run_command="src/lua poc.lua",
        environment_variables={"ASAN_OPTIONS": "detect_leaks=0"},
        pre_log_path=str(paths.pre_patch_log),
        post_log_path=str(paths.post_patch_log),
    )

    stage._run_one_pass("pre", context, plan, paths)
    env = captured["request"].environment
    assert env.get("ASAN_OPTIONS") == "detect_leaks=0"
    assert env.get("PATCH_MODE") == "pre"


# ===== Fix 4.B: dataset_root flows through =====
def test_collect_verify_context_uses_dataset_root(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    paths = verify_module.VerifyStagePaths(str(workspace))
    paths.poc_dir.mkdir(parents=True, exist_ok=True)

    custom_root = tmp_path / "custom_dataset"
    target = custom_root / "CVE-2022-0000" / "vuln_data" / "vuln_diffs" / "patch.diff"
    target.parent.mkdir(parents=True)
    target.write_text("--- a\n+++ b\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    knowledge = KnowledgeModel(
        cve_id="CVE-2022-0000",
        summary="demo",
        vulnerability_type="heap-overflow",
    )
    build = BuildArtifact(
        dockerfile_content="FROM ubuntu\n",
        build_script_content="#!/bin/bash\n",
        build_success=True,
        docker_image_tag="demo:build",
    )
    poc = PoCArtifact(
        poc_filename="poc.lua",
        poc_content="boom",
        run_script_content="#!/bin/bash\n",
    )
    stage = verify_module.VerifyStage()

    context = stage.collect_verify_context(
        knowledge, build, poc, paths, dataset_root=str(custom_root)
    )

    assert context.patch_diff_path == str(target)


# ===== Fix 5.A/B: _resolve_project_dir =====
def test_resolve_project_dir_from_absolute_binary_path():
    stage = verify_module.VerifyStage()
    build = BuildArtifact(
        dockerfile_content="x",
        build_script_content="y",
        binary_or_entrypoint="/opt/lua-5.4.4/src/lua",
    )
    assert stage._resolve_project_dir(build) == "/opt/lua-5.4.4"


def test_resolve_project_dir_falls_back_to_env_var():
    stage = verify_module.VerifyStage()
    build_relative = BuildArtifact(
        dockerfile_content="x",
        build_script_content="y",
        binary_or_entrypoint="src/lua",
    )
    build_empty = BuildArtifact(
        dockerfile_content="x",
        build_script_content="y",
        binary_or_entrypoint="",
    )
    assert stage._resolve_project_dir(build_relative) == "${PROJECT_DIR}"
    assert stage._resolve_project_dir(build_empty) == "${PROJECT_DIR}"


# ===== Fix 5.C: template uses resolved project dir =====
def test_verify_run_template_uses_resolved_project_dir():
    stage = verify_module.VerifyStage()
    rendered = stage._render_template(
        "verify_run.sh.j2",
        {
            "target_binary": "src/lua",
            "run_command": "src/lua poc.lua",
            "repo_reset_command": "git reset --hard && git clean -fd",
            "rebuild_command": "bash /workspace/artifacts/build/build.sh",
            "patch_apply_command": "git apply /workspace/artifacts/verify/patch.diff",
            "project_dir_var": "/opt/lua-5.4.4",
        },
    )
    assert 'PROJECT_DIR_VAR="/opt/lua-5.4.4"' in rendered


# ===== Fix 6.A/B: patch_apply_log extraction =====
def test_extract_patch_apply_log_uses_dedicated_block():
    stage = verify_module.VerifyStage()
    log = (
        "noise above\n"
        "patch_apply_exit_code=1\n"
        "patch_apply_stderr_begin\n"
        "error: patch failed: src/lua.c:123\n"
        "error: src/lua.c: patch does not apply\n"
        "patch_apply_stderr_end\n"
        "more noise\n"
    )
    result = stage._extract_patch_apply_log(log)
    assert "patch failed: src/lua.c:123" in result
    assert "patch does not apply" in result


def test_extract_patch_apply_log_falls_back_to_legacy_radius():
    stage = verify_module.VerifyStage()
    log = (
        "some preamble\n"
        "patch_apply_exit_code=1\n"
        "some_other_marker=value\n"
    )
    result = stage._extract_patch_apply_log(log)
    assert "patch_apply_exit_code=1" in result


# ===== Fix 7: short_circuit prefix vs real-run no-prefix =====
def test_short_circuit_reason_carries_prefix(tmp_path, monkeypatch):
    """Sanity check: short-circuit reasons always carry the short_circuit: prefix.

    We use the script_did_not_finish fixture (still maps to inconclusive) so this stays
    independent of the verdict-routing tests above.
    """

    monkeypatch.chdir(tmp_path)
    workspace, knowledge, build, poc, fake_docker = _setup_short_circuit_workspace(
        tmp_path, eligibility_reason="script_did_not_finish: missing execution_exit_code marker"
    )
    stage = verify_module.VerifyStage(docker_tool=fake_docker)
    result = stage.run(knowledge=knowledge, build=build, poc=poc, workspace=str(workspace))

    assert result.reason.startswith("short_circuit:")


def test_real_run_failed_reason_has_no_prefix():
    stage = verify_module.VerifyStage()
    pre = make_pass(exit_code=0, matched_error_patterns=[], matched_stack_keywords=[])
    post = make_pass(exit_code=0, patch_apply_exit_code=0)
    context = make_context()
    result = stage._decide_verdict({"pre": pre, "post": post}, context)

    assert result.verdict == "failed"
    assert result.reason == "pre_not_triggered"
    assert not result.reason.startswith("short_circuit:")


# ===== Fix 1.C: _is_triggered recognizes stdout-only match =====
def test_is_triggered_recognizes_stdout_match():
    stage = verify_module.VerifyStage()
    pass_result = {
        "matched_stdout_patterns": ["assertion failed"],
        "matched_stderr_patterns": [],
        "matched_error_patterns": [],
        "matched_stack_keywords": [],
        "crash_type": "",
        "exit_code": 0,
    }
    context = make_context()
    assert stage._is_triggered(pass_result, context) is True
