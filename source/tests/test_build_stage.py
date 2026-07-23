"""文件说明：Build 阶段测试。用于校验构建系统识别和节点重试语义。"""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import AppConfig, AgentModelConfig, RuntimeConfig
from app.schemas.build_artifact import BuildArtifact
from app.schemas.knowledge import KnowledgeModel
from app.stages import build as build_module


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


def test_select_build_system_prefers_repo_scan():
    stage = build_module.BuildStage()
    knowledge = make_knowledge(build_systems=["make"])
    detected_files = ["subdir/CMakeLists.txt", "README.md"]

    assert stage._select_build_system(knowledge, detected_files) == "cmake"


def test_select_build_commands_falls_back_for_unknown():
    stage = build_module.BuildStage()
    knowledge = make_knowledge(build_commands=[])

    commands = stage._select_build_commands(knowledge, "unknown")

    assert commands[-1] == "exit 2"


def test_heuristic_build_plan_prefers_fixed_parent():
    stage = build_module.BuildStage()
    knowledge = make_knowledge(fixed_ref="feedface")
    context = build_module.BuildContext(
        cve_id=knowledge.cve_id,
        repo_url=knowledge.repo_url or "",
        snapshots=[
            build_module.RefSnapshot(label="knowledge_vulnerable", requested_ref="deadbeef", resolved_ref="deadbeef"),
            build_module.RefSnapshot(
                label="fixed_parent",
                requested_ref="beadfeed",
                resolved_ref="beadfeed",
                build_files=["Makefile"],
            ),
        ],
    )

    plan = stage._heuristic_build_plan(knowledge, context, project_name="demo")

    assert plan.chosen_vulnerable_ref == "beadfeed"
    assert plan.build_system == "make"


def test_build_fallback_spec_centralizes_heuristic_defaults():
    stage = build_module.BuildStage()
    knowledge = make_knowledge(
        install_commands=["apt-get install zlib openssl"],
        fixed_ref="feedface",
    )
    context = build_module.BuildContext(
        cve_id=knowledge.cve_id,
        repo_url=knowledge.repo_url or "",
        snapshots=[
            build_module.RefSnapshot(
                label="fixed_parent",
                requested_ref="beadfeed",
                resolved_ref="beadfeed",
                build_files=["CMakeLists.txt"],
            ),
        ],
    )

    spec = stage._build_fallback_spec(knowledge=knowledge, context=context, project_name="demo")

    assert spec.chosen_vulnerable_ref == "beadfeed"
    assert spec.build_system == "cmake"
    assert "cmake --build build -j$(nproc)" in spec.build_commands
    assert "zlib1g-dev" in spec.install_packages
    assert "libssl-dev" in spec.install_packages


def test_build_node_records_retry_on_unsuccessful_build(monkeypatch):
    artifact = BuildArtifact(
        dockerfile_content="FROM ubuntu:20.04\n",
        build_script_content="#!/bin/bash\nexit 2\n",
        build_success=False,
        build_logs="build failed",
    )

    class FakeStage:
        def run(self, knowledge, workspace):
            return artifact

    monkeypatch.setattr(build_module, "BuildStage", FakeStage)

    state = {
        "knowledge": make_knowledge(),
        "workspace": "workspaces/CVE-2022-0000",
        "retry_count": {},
        "stage_history": [],
    }

    result = build_module.build_node(state)

    assert result["build"].build_success is False
    assert result["retry_count"]["build"] == 1
    assert result["stage_history"][-1]["status"] == "failed"


def test_execute_build_plan_uses_script_overrides(tmp_path):
    class FakeDockerTool:
        def build_image(self, request):
            class Result:
                success = True
                exit_code = 0
                stdout = "built"
                stderr = ""

            return Result()

        def run_container(self, request):
            class Result:
                success = True
                exit_code = 0
                stdout = "ran"
                stderr = ""

            return Result()

        def commit_container(self, container_name, image_tag):
            class Result:
                success = True
                exit_code = 0
                stdout = "committed"
                stderr = ""

            return Result()

        def remove_container(self, container_name):
            return None

    stage = build_module.BuildStage(docker_tool=FakeDockerTool())
    paths = build_module.BuildStagePaths(str(tmp_path / "ws"))
    stage._prepare_workspace(paths)
    repo = paths.repo_dir
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "Makefile").write_text("all:\n\techo ok\n", encoding="utf-8")

    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["gcc", "make"],
        build_commands=["make"],
        dockerfile_override="FROM ubuntu:22.04\n",
        build_script_override="#!/bin/bash\necho custom\n",
    )
    artifact = stage._execute_build_plan(
        repo_path=repo,
        paths=paths,
        plan_meta={
            "repo_url": "https://example.com/demo.git",
            "project_name": "demo",
            "project_dir_name": "demo",
            "docker_image_tag": "demo:latest",
            "compiled_image_tag": "demo:compiled",
            "build_container_name": "demo-build-run",
        },
        build_plan=plan,
        resolved_ref="deadbeef",
    )

    assert artifact.dockerfile_content == "FROM ubuntu:22.04\n"
    assert artifact.build_script_content == "#!/bin/bash\necho custom\n"
    assert artifact.compiled_image_tag == "demo:compiled"


def test_classify_failure_kind_distinguishes_docker_and_container_failures():
    stage = build_module.BuildStage()

    docker_build_log = "image_build_success=False\nimage_build_exit_code=1\n"
    container_run_log = (
        "image_build_success=True\nimage_build_exit_code=0\n\n"
        "container_run_success=False\ncontainer_run_exit_code=2\n"
    )

    assert stage._classify_failure_kind(docker_build_log) == "docker_build"
    assert stage._classify_failure_kind(container_run_log) == "container_run"


def test_valid_replan_candidate_accepts_build_command_changes_for_container_run():
    stage = build_module.BuildStage()
    previous_plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["gcc", "make"],
        build_commands=["make -j$(nproc)"],
    )
    candidate_plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["gcc", "make"],
        build_commands=["make linux"],
    )

    assert stage._is_valid_replan_candidate(
        previous_plan,
        candidate_plan,
        failure_kind="container_run",
    )


def test_normalize_build_plan_preserves_required_docker_packages(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "repo"
    repo.mkdir()

    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["gcc", "make"],
        build_commands=["make"],
    )

    normalized = stage._normalize_build_plan(repo, plan)

    assert "build-essential" in normalized.install_packages
    assert "clang" in normalized.install_packages
    assert "gcc" in normalized.install_packages
    assert "g++" in normalized.install_packages
    assert "git" in normalized.install_packages
    assert "make" in normalized.install_packages
    assert "pkg-config" in normalized.install_packages
    assert "ca-certificates" in normalized.install_packages


def test_normalize_build_plan_repairs_dockerfile_override_without_required_packages(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "repo"
    repo.mkdir()

    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["gcc", "make"],
        build_commands=["make"],
        dockerfile_override=(
            "FROM ubuntu:20.04\n"
            "RUN apt-get update && apt-get install -y --no-install-recommends gcc make && apt-get clean\n"
        ),
    )

    normalized = stage._normalize_build_plan(repo, plan)

    assert "clang" in normalized.install_packages
    assert "gcc" in normalized.install_packages
    assert "g++" in normalized.install_packages
    assert "git" in normalized.install_packages
    assert "ca-certificates" in normalized.install_packages
    assert "make" in normalized.install_packages
    assert "pkg-config" in normalized.install_packages
    assert "clang" in normalized.dockerfile_override
    assert "gcc" in normalized.dockerfile_override
    assert "g++" in normalized.dockerfile_override
    assert "git" in normalized.dockerfile_override
    assert "ca-certificates" in normalized.dockerfile_override


def test_default_make_install_packages_include_ca_certificates():
    stage = build_module.BuildStage()
    knowledge = make_knowledge()

    packages = stage._select_install_packages("make", knowledge)

    assert "build-essential" in packages
    assert "clang" in packages
    assert "gcc" in packages
    assert "g++" in packages
    assert "git" in packages
    assert "make" in packages
    assert "pkg-config" in packages
    assert "ca-certificates" in packages


def test_build_stage_uses_localhost_proxy_with_host_network(monkeypatch):
    stage = build_module.BuildStage()

    monkeypatch.setenv("DOCKER_BUILD_PROXY", "http://127.0.0.1:7897")

    assert stage._get_docker_build_proxy() == "http://127.0.0.1:7897"
    assert stage._build_docker_proxy_args("http://127.0.0.1:7897")["HTTP_PROXY"] == "http://127.0.0.1:7897"
    assert stage._select_docker_build_network_mode("http://127.0.0.1:7897") == "host"


def test_build_replan_prompt_includes_rendered_artifacts():
    stage = build_module.BuildStage()
    knowledge = make_knowledge()
    context = build_module.BuildContext(
        cve_id=knowledge.cve_id,
        repo_url=knowledge.repo_url or "",
        previous_failure_kind="container_run",
        previous_build_failure="compiler not found: clang",
        previous_dockerfile_content="FROM ubuntu:20.04\nRUN apt-get install -y git make\n",
        previous_build_script_content="#!/bin/bash\nexport CC=clang\nmake -j$(nproc)\n",
    )
    previous_plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["build-essential", "git", "make"],
        build_commands=["make -j$(nproc)"],
    )

    prompt = stage._build_llm_prompt(
        knowledge=knowledge,
        context=context,
        project_name="demo",
        previous_plan=previous_plan,
    )

    assert "Previously rendered Dockerfile:" in prompt
    assert "Previously rendered build.sh:" in prompt
    assert "compiler not found: clang" in prompt
    assert "dockerfile_override" in prompt
    assert "build_script_override" in prompt
    assert prompt.index("Previous failure logs:") < prompt.index("Patch excerpt:")


def test_build_replan_messages_reuse_single_conversation_history():
    stage = build_module.BuildStage()
    planner = build_module.BuildPlanner(stage)
    knowledge = make_knowledge()
    context = build_module.BuildContext(
        cve_id=knowledge.cve_id,
        repo_url=knowledge.repo_url or "",
        previous_failure_kind="container_run",
        previous_build_failure="compiler not found: clang",
        previous_dockerfile_content="FROM ubuntu:20.04\nRUN apt-get install -y git make\n",
        previous_build_script_content="#!/bin/bash\nexport CC=clang\nmake -j$(nproc)\n",
    )
    previous_plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["build-essential", "git", "make"],
        build_commands=["make -j$(nproc)"],
    )

    messages = planner.build_llm_messages(
        knowledge=knowledge,
        context=context,
        project_name="demo",
        previous_plan=previous_plan,
    )

    assert len(messages) == 4
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert isinstance(messages[2], AIMessage)
    assert isinstance(messages[3], HumanMessage)
    assert "Patch excerpt:" in messages[1].content
    assert "compiler not found: clang" not in messages[1].content
    assert "chosen_vulnerable_ref: deadbeef" in messages[2].content
    assert "Previous failure logs:" in messages[3].content
    assert "compiler not found: clang" in messages[3].content
    assert "Patch excerpt:" not in messages[3].content


def test_build_retry_prompt_truncates_large_failure_artifacts():
    stage = build_module.BuildStage()
    previous_plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["build-essential", "git", "make"],
        build_commands=["make -j$(nproc)"],
    )
    context = build_module.BuildContext(
        cve_id="CVE-2022-0000",
        previous_failure_kind="container_run",
        previous_build_failure=("line\n" * 2000) + "fatal error: missing header\n",
        previous_dockerfile_content="FROM ubuntu:20.04\n" + ("RUN echo hi\n" * 500),
        previous_build_script_content="#!/bin/bash\n" + ("echo hi\n" * 600),
    )

    prompt = stage._build_llm_retry_prompt(context, previous_plan)

    assert "Failure summary:" in prompt
    assert "fatal error: missing header" in prompt
    assert "[truncated " in prompt
    assert len(prompt) < 9000


def test_build_try_llm_plan_retries_timeout_twice_before_success(tmp_path, monkeypatch):
    class FakeResponse:
        def __init__(self, content):
            self.content = content

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("Request timed out.")
            return FakeResponse(
                '{"chosen_vulnerable_ref":"deadbeef","chosen_fixed_ref":null,"build_system":"make",'
                '"install_packages":["build-essential","git","make"],"configure_commands":[],"clean_commands":["make clean || true"],'
                '"build_commands":["make -j$(nproc)"],"expected_binary_path":"demo","dockerfile_override":null,'
                '"build_script_override":null,"source_of_truth":"llm","confidence":"high","rationale":"retry success"}'
            )

    fake_model = FakeModel()
    monkeypatch.setattr(build_module, "build_chat_model", lambda *args, **kwargs: fake_model)

    stage = build_module.BuildStage()
    planner = build_module.BuildPlanner(stage)
    paths = build_module.BuildStagePaths(str(tmp_path / "ws"))
    stage._prepare_workspace(paths)
    stage._active_build_dir = str(paths.build_dir)

    plan = planner.try_llm_plan(
        knowledge=make_knowledge(),
        context=build_module.BuildContext(cve_id="CVE-2022-28805", planner_attempt=4),
        project_name="demo",
    )

    assert plan is not None
    assert fake_model.calls == 3
    assert (paths.llm_dir / "attempt-4" / "response.txt").exists()


def test_build_try_llm_plan_uses_build_agent_timeout_override(tmp_path, monkeypatch):
    class FakeResponse:
        def __init__(self, content):
            self.content = content

    class FakeModel:
        def invoke(self, messages):
            return FakeResponse(
                '{"chosen_vulnerable_ref":"deadbeef","chosen_fixed_ref":null,"build_system":"make",'
                '"install_packages":["build-essential","git","make"],"configure_commands":[],"clean_commands":["make clean || true"],'
                '"build_commands":["make -j$(nproc)"],"expected_binary_path":"demo","dockerfile_override":null,'
                '"build_script_override":null,"source_of_truth":"llm","confidence":"high","rationale":"ok"}'
            )

    captured = {}

    def fake_build_chat_model(agent_name, model_name=None, temperature=0, timeout_seconds=None):
        captured["agent_name"] = agent_name
        captured["timeout_seconds"] = timeout_seconds
        return FakeModel()

    monkeypatch.setattr(build_module, "build_chat_model", fake_build_chat_model)
    monkeypatch.setattr(
        build_module,
        "load_app_config",
        lambda: AppConfig(
            build_agent=AgentModelConfig(model_name="demo", api_key="key"),
            runtime=RuntimeConfig(build_agent_timeout_seconds=77),
        ),
    )

    stage = build_module.BuildStage()
    planner = build_module.BuildPlanner(stage)
    paths = build_module.BuildStagePaths(str(tmp_path / "ws"))
    stage._prepare_workspace(paths)
    stage._active_build_dir = str(paths.build_dir)

    plan = planner.try_llm_plan(
        knowledge=make_knowledge(),
        context=build_module.BuildContext(cve_id="CVE-2022-28805", planner_attempt=6),
        project_name="demo",
    )

    assert plan is not None
    assert captured["agent_name"] == "build_agent"
    assert captured["timeout_seconds"] == 77


def test_build_try_llm_plan_records_final_error_after_three_empty_responses(tmp_path, monkeypatch):
    class FakeResponse:
        def __init__(self, content):
            self.content = content

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            return FakeResponse("   ")

    fake_model = FakeModel()
    monkeypatch.setattr(build_module, "build_chat_model", lambda *args, **kwargs: fake_model)

    stage = build_module.BuildStage()
    planner = build_module.BuildPlanner(stage)
    paths = build_module.BuildStagePaths(str(tmp_path / "ws"))
    stage._prepare_workspace(paths)
    stage._active_build_dir = str(paths.build_dir)

    plan = planner.try_llm_plan(
        knowledge=make_knowledge(),
        context=build_module.BuildContext(cve_id="CVE-2022-28805", planner_attempt=5),
        project_name="demo",
    )

    assert plan is None
    assert fake_model.calls == 3
    error_text = (paths.llm_dir / "attempt-5" / "error.txt").read_text(encoding="utf-8")
    assert "no content after 3 attempts" in error_text


def test_replan_candidate_requires_meaningful_execution_surface_change():
    stage = build_module.BuildStage()
    previous_plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["build-essential", "git", "make"],
        build_commands=["make -j$(nproc)"],
    )

    identical_shape = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["build-essential", "git", "make"],
        build_commands=["make -j$(nproc)"],
        rationale="different words only",
    )
    changed_packages = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["build-essential", "clang", "git", "make"],
        build_commands=["make -j$(nproc)"],
    )
    override_only = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["build-essential", "git", "make"],
        build_commands=["make -j$(nproc)"],
        dockerfile_override="FROM ubuntu:20.04\nRUN apt-get update && apt-get install -y clang git make\n",
    )
    script_override_only = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["build-essential", "git", "make"],
        build_commands=["make -j$(nproc)"],
        build_script_override="#!/bin/bash\nexport CC=gcc\nmake -j$(nproc)\n",
    )

    assert stage._is_valid_replan_candidate(previous_plan, identical_shape) is False
    assert stage._is_valid_replan_candidate(previous_plan, changed_packages) is True
    assert stage._is_valid_replan_candidate(previous_plan, override_only) is True
    assert stage._is_valid_replan_candidate(previous_plan, changed_packages, failure_kind="docker_build") is False
    assert stage._is_valid_replan_candidate(previous_plan, override_only, failure_kind="docker_build") is True
    assert stage._is_valid_replan_candidate(previous_plan, changed_packages, failure_kind="container_run") is True
    assert stage._is_valid_replan_candidate(previous_plan, script_override_only, failure_kind="container_run") is True


def test_build_llm_trace_is_persisted(tmp_path):
    stage = build_module.BuildStage()
    paths = build_module.BuildStagePaths(str(tmp_path / "ws"))
    stage._prepare_workspace(paths)
    stage._active_build_dir = str(paths.build_dir)

    stage._persist_build_llm_trace(2, "prompt.txt", "hello prompt")
    stage._persist_build_llm_trace(2, "parsed.json", json.dumps({"ok": True}))

    prompt_path = paths.llm_dir / "attempt-2" / "prompt.txt"
    parsed_path = paths.llm_dir / "attempt-2" / "parsed.json"

    assert prompt_path.read_text(encoding="utf-8") == "hello prompt\n"
    assert parsed_path.read_text(encoding="utf-8") == '{"ok": true}\n'
