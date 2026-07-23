"""Build verify tests. Validates build artifact self-check logic."""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.schemas.build_artifact import BuildArtifact
from app.schemas.knowledge import KnowledgeModel
from app.stages import build as build_module
from app.tools.process_tools import ProcessResult


@pytest.fixture(autouse=True)
def _restore_find_patch_diff():
    """Restore the original find_patch_diff after each test so module state doesn't leak."""
    original = build_module.find_patch_diff
    yield
    build_module.find_patch_diff = original


def make_knowledge(**overrides):
    payload = {
        "cve_id": "CVE-2022-0000",
        "summary": "demo",
        "vulnerability_type": "heap-overflow",
        "repo_url": "https://example.com/demo.git",
        "vulnerable_ref": "deadbeef",
    }
    payload.update(overrides)
    return KnowledgeModel(**payload)


def make_artifact(**overrides):
    payload = {
        "dockerfile_content": "FROM ubuntu:20.04\n",
        "build_script_content": "#!/bin/bash\necho build\n",
        "build_success": True,
        "build_logs": "image_build_success=True\nimage_build_exit_code=0\n\ncontainer_run_success=True\ncontainer_run_exit_code=0\n",
        "repo_local_path": "/tmp/repo",
        "resolved_ref": "deadbeef",
        "build_system": "make",
        "binary_or_entrypoint": "src/lua",
        "expected_binary_path": "src/lua",
        "docker_image_tag": "demo:build",
        "chosen_vulnerable_ref": "abc1234",
    }
    payload.update(overrides)
    return BuildArtifact(**payload)


def _prepare_workspace(tmp_path):
    """Create a minimal workspace with all required files."""
    workspace = tmp_path / "ws"
    paths = build_module.BuildStagePaths(str(workspace))

    paths.workspace_root.mkdir(parents=True, exist_ok=True)
    paths.repo_dir.mkdir(parents=True, exist_ok=True)
    paths.build_dir.mkdir(parents=True, exist_ok=True)
    paths.dockerfile.write_text("FROM ubuntu:20.04\n", encoding="utf-8")
    paths.build_script.write_text("#!/bin/bash\n", encoding="utf-8")
    paths.build_log.write_text(
        "image_build_success=True\nimage_build_exit_code=0\n\n"
        "container_run_success=True\ncontainer_run_exit_code=0\n",
        encoding="utf-8",
    )

    return workspace, paths


def _make_stage_with_mocks(
    process_run_side_effect=None,
    container_run_return=None,
    locate_patch_return=None,
):
    """Create a BuildStage with mocked docker/process tools."""
    stage = build_module.BuildStage()

    # Default: all process calls succeed with "sha256:abc123"
    if process_run_side_effect is None:
        process_run_side_effect = lambda req: ProcessResult(
            success=True,
            exit_code=0,
            stdout="sha256:abc123\n",
            stderr="",
        )
    stage.docker_tool.process_tool.run = process_run_side_effect

    # Default: container calls return BINARY_FOUND / HEAD sha
    if container_run_return is None:
        call_count = {"n": 0}

        def default_container_run(req):
            call_count["n"] += 1
            cmd = " ".join(req.command)
            if "BINARY_FOUND" in cmd or "BINARY_MISSING" in cmd:
                return build_module.DockerTool.__init__.__class__.__mro__[0]  # won't reach
            result = MagicMock()
            result.success = True
            result.exit_code = 0
            if "test -x" in cmd:
                result.stdout = "BINARY_FOUND\n"
                result.stderr = ""
            elif "git rev-parse HEAD" in cmd:
                result.stdout = "abc1234deadbeef\n"
                result.stderr = ""
            else:
                result.stdout = ""
                result.stderr = ""
            return result

        container_run_return = default_container_run
    stage.docker_tool.run_container = container_run_return

    # Always patch find_patch_diff at the module level so the build stage sees our value.
    # Use setattr so the override survives across method calls.
    build_module.find_patch_diff = lambda cve_id: locate_patch_return

    return stage


def test_verify_build_artifact_all_green(tmp_path):
    workspace, paths = _prepare_workspace(tmp_path)
    artifact = make_artifact()

    def process_run(req):
        return ProcessResult(success=True, exit_code=0, stdout="sha256:abc123\n", stderr="")

    def container_run(req):
        cmd = " ".join(req.command)
        result = MagicMock()
        result.success = True
        result.exit_code = 0
        result.stderr = ""
        if "test -x" in cmd:
            result.stdout = "BINARY_FOUND\n"
        elif "git rev-parse HEAD" in cmd:
            result.stdout = "abc1234deadbeef\n"
        else:
            result.stdout = ""
        return result

    stage = _make_stage_with_mocks(
        process_run_side_effect=process_run,
        container_run_return=container_run,
        locate_patch_return=None,
    )

    result = stage._verify_build_artifact(
        artifact=artifact, paths=paths, plan_meta={}, cve_id="CVE-2022-0000"
    )

    assert result["verify_status"] == "ok"
    assert result["image_present"] is True
    assert result["workspace_layout_ok"] is True
    assert result["binary_in_container"]["exists"] is True
    assert result["patch_appliable_in_container"]["checked"] is False


def test_verify_build_artifact_dockerfile_missing(tmp_path):
    workspace, paths = _prepare_workspace(tmp_path)
    # Remove the dockerfile
    paths.dockerfile.unlink()

    artifact = make_artifact()

    def process_run(req):
        return ProcessResult(success=True, exit_code=0, stdout="sha256:abc123\n", stderr="")

    def container_run(req):
        cmd = " ".join(req.command)
        result = MagicMock()
        result.success = True
        result.exit_code = 0
        result.stderr = ""
        if "test -x" in cmd:
            result.stdout = "BINARY_FOUND\n"
        elif "git rev-parse HEAD" in cmd:
            result.stdout = "abc1234deadbeef\n"
        else:
            result.stdout = ""
        return result

    stage = _make_stage_with_mocks(
        process_run_side_effect=process_run,
        container_run_return=container_run,
        locate_patch_return=None,
    )

    result = stage._verify_build_artifact(
        artifact=artifact, paths=paths, plan_meta={}, cve_id="CVE-2022-0000"
    )

    assert result["dockerfile_present"] is False
    assert result["verify_status"] == "partial"


def test_verify_build_artifact_parses_container_run_failure(tmp_path):
    workspace, paths = _prepare_workspace(tmp_path)
    # Overwrite the build log with container_run failure
    paths.build_log.write_text(
        "image_build_success=True\nimage_build_exit_code=0\n\n"
        "container_run_success=False\ncontainer_run_exit_code=2\n",
        encoding="utf-8",
    )

    artifact = make_artifact(
        build_success=False,
        build_logs=(
            "image_build_success=True\nimage_build_exit_code=0\n\n"
            "container_run_success=False\ncontainer_run_exit_code=2\n"
        ),
    )

    def process_run(req):
        return ProcessResult(success=True, exit_code=0, stdout="sha256:abc123\n", stderr="")

    def container_run(req):
        cmd = " ".join(req.command)
        result = MagicMock()
        result.success = True
        result.exit_code = 0
        result.stderr = ""
        if "test -x" in cmd:
            result.stdout = "BINARY_FOUND\n"
        elif "git rev-parse HEAD" in cmd:
            result.stdout = "abc1234deadbeef\n"
        else:
            result.stdout = ""
        return result

    stage = _make_stage_with_mocks(
        process_run_side_effect=process_run,
        container_run_return=container_run,
        locate_patch_return=None,
    )

    result = stage._verify_build_artifact(
        artifact=artifact, paths=paths, plan_meta={}, cve_id="CVE-2022-0000"
    )

    assert result["image_build_success"] is True
    assert result["container_run_success"] is False


def test_verify_build_artifact_handles_missing_patch_gracefully(tmp_path):
    workspace, paths = _prepare_workspace(tmp_path)
    artifact = make_artifact()

    def process_run(req):
        return ProcessResult(success=True, exit_code=0, stdout="sha256:abc123\n", stderr="")

    def container_run(req):
        cmd = " ".join(req.command)
        result = MagicMock()
        result.success = True
        result.exit_code = 0
        result.stderr = ""
        if "test -x" in cmd:
            result.stdout = "BINARY_FOUND\n"
        elif "git rev-parse HEAD" in cmd:
            result.stdout = "abc1234deadbeef\n"
        else:
            result.stdout = ""
        return result

    stage = _make_stage_with_mocks(
        process_run_side_effect=process_run,
        container_run_return=container_run,
        locate_patch_return=None,
    )

    result = stage._verify_build_artifact(
        artifact=artifact, paths=paths, plan_meta={}, cve_id="CVE-2022-0000"
    )

    assert result["patch_appliable_in_container"]["checked"] is False
    assert result["patch_appliable_in_container"].get("applied") is None


def test_verify_build_artifact_records_patch_apply_failure(tmp_path):
    workspace, paths = _prepare_workspace(tmp_path)

    # Create a real patch file
    patch_file = tmp_path / "patch.diff"
    patch_file.write_text("--- a/foo.c\n+++ b/foo.c\n@@ -1 +1 @@\n-old\n+new\n", encoding="utf-8")

    artifact = make_artifact()

    def process_run(req):
        cmd = " ".join(req.command)
        if "git apply --check" in cmd:
            return ProcessResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr="error: patch does not apply\n",
            )
        # docker image inspect
        return ProcessResult(success=True, exit_code=0, stdout="sha256:abc123\n", stderr="")

    def container_run(req):
        cmd = " ".join(req.command)
        result = MagicMock()
        result.success = True
        result.exit_code = 0
        result.stderr = ""
        if "test -x" in cmd:
            result.stdout = "BINARY_FOUND\n"
        elif "git rev-parse HEAD" in cmd:
            result.stdout = "abc1234deadbeef\n"
        else:
            result.stdout = ""
        return result

    stage = _make_stage_with_mocks(
        process_run_side_effect=process_run,
        container_run_return=container_run,
        locate_patch_return=patch_file,
    )

    result = stage._verify_build_artifact(
        artifact=artifact, paths=paths, plan_meta={}, cve_id="CVE-2022-0000"
    )

    assert result["patch_appliable_in_container"]["checked"] is True
    assert result["patch_appliable_in_container"]["applied"] is False
    assert result["patch_appliable_in_container"]["exit_code"] == 1
    assert result["verify_status"] == "partial"


def test_find_patch_diff_dual_prefix(tmp_path, monkeypatch):
    from app.tools.patch_tools import find_patch_diff

    original_cwd = os.getcwd()
    monkeypatch.chdir(tmp_path)

    try:
        # Neither exists
        assert find_patch_diff("CVE-FAKE") is None

        # Create in Dataset/
        ds_path = tmp_path / "Dataset" / "CVE-FAKE" / "vuln_data" / "vuln_diffs"
        ds_path.mkdir(parents=True)
        (ds_path / "patch.diff").write_text("diff content\n", encoding="utf-8")
        found = find_patch_diff("CVE-FAKE")
        assert found is not None
        assert "Dataset" in str(found)

        # Remove Dataset/, create source/Dataset/
        (ds_path / "patch.diff").unlink()
        ds_path.rmdir()

        src_path = tmp_path / "source" / "Dataset" / "CVE-FAKE" / "vuln_data" / "vuln_diffs"
        src_path.mkdir(parents=True)
        (src_path / "patch.diff").write_text("diff content 2\n", encoding="utf-8")
        found = find_patch_diff("CVE-FAKE")
        assert found is not None
        assert "source/Dataset" in str(found)
    finally:
        os.chdir(original_cwd)


def test_collect_build_context_finds_patch_via_dual_prefix(tmp_path, monkeypatch):
    stage = build_module.BuildStage()
    original_cwd = os.getcwd()
    monkeypatch.chdir(tmp_path)

    try:
        # Set up source/Dataset/ with a patch
        patch_dir = tmp_path / "source" / "Dataset" / "CVE-2022-0000" / "vuln_data" / "vuln_diffs"
        patch_dir.mkdir(parents=True)
        (patch_dir / "patch.diff").write_text(
            "--- a/lparser.c\n+++ b/lparser.c\n@@ -1,3 +1,5 @@ static void singlevar\n+check\n",
            encoding="utf-8",
        )

        # Create a minimal repo dir for collect_build_context
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        knowledge = make_knowledge()

        # Mock _candidate_refs to return empty (skip git operations)
        monkeypatch.setattr(stage, "_candidate_refs", lambda repo_path, knowledge: {})

        context = stage.collect_build_context(
            knowledge=knowledge, repo_path=repo_dir, planner_attempt=1
        )

        assert context.patch_diff_excerpt != ""
        assert "lparser.c" in context.patch_affected_files
    finally:
        os.chdir(original_cwd)
