"""文件说明：PoC 阶段测试。用于校验上下文收集、执行解析和节点重试。"""

from pathlib import Path
from types import SimpleNamespace

from app.schemas.build_artifact import BuildArtifact
from app.schemas.knowledge import KnowledgeModel, ReproductionRecipe
from app.schemas.poc_artifact import PoCArtifact
from app.stages import poc as poc_module


def make_knowledge(**overrides):
    payload = {
        "cve_id": "CVE-2022-28805",
        "summary": "demo",
        "vulnerability_type": "heap-overflow",
        "repo_url": "https://example.com/demo.git",
        "vulnerable_ref": "deadbeef",
    }
    payload.update(overrides)
    return KnowledgeModel(**payload)


def make_build(**overrides):
    payload = {
        "dockerfile_content": "FROM ubuntu:20.04\n",
        "build_script_content": "#!/bin/bash\necho build\n",
        "build_success": True,
        "build_logs": "ok",
        "repo_local_path": "/tmp/repo",
        "resolved_ref": "deadbeef",
        "build_system": "make",
        "binary_or_entrypoint": "demo-bin",
        "docker_image_tag": "demo:build",
    }
    payload.update(overrides)
    return BuildArtifact(**payload)


def test_collect_poc_context_uses_build_artifact_and_hints(tmp_path, monkeypatch):
    stage = poc_module.PocStage()
    workspace = tmp_path / "ws"
    repo_dir = workspace / "repo" / "bin"
    repo_dir.mkdir(parents=True)
    (repo_dir / "helper.sh").write_text("#!/bin/bash\necho ok\n", encoding="utf-8")
    (workspace / "repo" / "README.md").write_text("usage: demo-bin --input {payload}\n", encoding="utf-8")
    monkeypatch.setattr(
        stage,
        "_read_patch_diff",
        lambda cve_id: "@@ -1,3 +1,5 @@ static void singlevar(lua_State *L)\n+ if (check_condition) {\n+   luaK_exp2anyregup(fs, &var);\n+ }\n",
    )

    knowledge = make_knowledge(reproduction_hints=["run with --input {payload} --mode crash"], expected_error_patterns=["segmentation fault"])
    build = make_build(repo_local_path=str(workspace / "repo"))

    context = stage.collect_poc_context(knowledge=knowledge, build=build, workspace=str(workspace))

    assert context.target_binary == "demo-bin"
    assert "--input" in context.candidate_cli_flags
    assert "demo-bin" in context.candidate_entrypoints
    assert context.repo_evidence_blocks
    assert "singlevar" in context.patch_changed_functions
    assert "file" in context.inferred_input_modes


def test_collect_poc_context_truncates_large_evidence(tmp_path, monkeypatch):
    stage = poc_module.PocStage()
    workspace = tmp_path / "ws"
    repo_dir = workspace / "repo"
    repo_dir.mkdir(parents=True)
    large_readme = "A" * 3000
    (repo_dir / "README.md").write_text(large_readme, encoding="utf-8")
    monkeypatch.setattr(stage, "_read_patch_diff", lambda cve_id: "B" * 5000)
    monkeypatch.setattr(stage, "_collect_reference_poc_summaries", lambda cve_id: ["FILE: poc\nCONTENT:\n" + ("C" * 3000)])

    context = stage.collect_poc_context(
        knowledge=make_knowledge(),
        build=make_build(repo_local_path=str(repo_dir)),
        workspace=str(workspace),
    )

    assert len(context.patch_diff_excerpt) <= stage.PATCH_EXCERPT_CHAR_LIMIT + 32
    assert len(context.repo_evidence_blocks) <= stage.REPO_EVIDENCE_BLOCK_LIMIT
    assert "[truncated" in context.repo_evidence_blocks[0]
    assert len(context.reference_poc_summaries) <= stage.REFERENCE_POC_BLOCK_LIMIT


def test_plan_poc_prefers_llm_plan_when_available(monkeypatch):
    class FakeResponse:
        def __init__(self, content):
            self.content = content

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            prompt = messages[-1].content
            if "PoC Strategy Selector" in prompt:
                return FakeResponse(
                    '{"chosen_strategy":"llm_synthesized","rationale":"no reusable artifact","evidence":["patch-only"]}'
                )
            return FakeResponse(
                '{"trigger_mode":"cli-file","target_binary":"demo-bin","target_args":["--input","/workspace/artifacts/poc/payloads/llm.txt"],'
                '"environment_variables":{},"payload_filename":"llm.txt","payload_content":"llm\\n","auxiliary_files":{},'
                '"run_command":"demo-bin --input /workspace/artifacts/poc/payloads/llm.txt","expected_exit_code":null,'
                '"expected_stdout_patterns":[],"expected_stderr_patterns":["segmentation fault"],"expected_crash_type":"segmentation fault",'
                '"source_of_truth":"llm_synthesized","confidence":"high","rationale":"llm plan","dockerfile_override":null,"run_script_override":null}'
            )

    fake_model = FakeModel()
    monkeypatch.setattr(poc_module, "build_chat_model", lambda *args, **kwargs: fake_model)

    stage = poc_module.PocStage()
    knowledge = make_knowledge(expected_error_patterns=["segmentation fault"])
    build = make_build()
    context = poc_module.PocContext(
        cve_id=knowledge.cve_id,
        reference_poc_summaries=["SUMMARY: dataset poc"],
    )

    plan = stage.plan_poc(knowledge=knowledge, build=build, context=context)

    assert plan.source_of_truth == "llm_synthesized"
    assert plan.payload_filename == "llm.txt"
    assert context.chosen_strategy == "llm_synthesized"
    assert fake_model.calls == 2


def test_plan_poc_reuses_selected_strategy_without_reselecting(monkeypatch):
    class FakeResponse:
        def __init__(self, content):
            self.content = content

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            prompt = messages[-1].content
            if "PoC Strategy Selector" in prompt:
                return FakeResponse(
                    '{"chosen_strategy":"dataset_poc","rationale":"existing payload","evidence":["dataset poc"]}'
                )
            return FakeResponse(
                '{"trigger_mode":"cli-file","target_binary":"demo-bin","target_args":["/workspace/artifacts/poc/payloads/llm.txt"],'
                '"environment_variables":{},"payload_filename":"llm.txt","payload_content":"llm\\n","auxiliary_files":{},'
                '"run_command":"demo-bin /workspace/artifacts/poc/payloads/llm.txt","expected_exit_code":null,'
                '"expected_stdout_patterns":[],"expected_stderr_patterns":["segmentation fault"],"expected_stack_keywords":[],"expected_crash_type":"segmentation fault",'
                '"source_of_truth":"dataset_poc","confidence":"high","rationale":"plan","dockerfile_override":null,"run_script_override":null}'
            )

    fake_model = FakeModel()
    monkeypatch.setattr(poc_module, "build_chat_model", lambda *args, **kwargs: fake_model)

    stage = poc_module.PocStage()
    knowledge = make_knowledge(expected_error_patterns=["segmentation fault"])
    build = make_build()
    context = poc_module.PocContext(
        cve_id=knowledge.cve_id,
        reference_poc_summaries=["SUMMARY: dataset poc"],
    )

    stage.plan_poc(knowledge=knowledge, build=build, context=context)
    stage.plan_poc(knowledge=knowledge, build=build, context=context)

    assert context.chosen_strategy == "dataset_poc"
    assert fake_model.calls == 3


def test_llm_prompt_includes_previous_run_artifacts():
    stage = poc_module.PocStage()
    knowledge = make_knowledge(expected_error_patterns=["segmentation fault"])
    build = make_build(binary_or_entrypoint="./demo-bin")
    context = poc_module.PocContext(
        cve_id=knowledge.cve_id,
        repo_url="https://github.com/example/demo.git",
        previous_failure_kind="non_triggering",
        previous_execution_log="execution_exit_code=0",
        previous_run_script_content="#!/bin/bash\ndemo payload\n",
        previous_payload_content="payload\n",
        previous_run_verify_report="eligible_for_verify: false\neligibility_reason: no_target_behavior_observed\n",
    )
    previous_plan = poc_module.PocPlan(
        target_binary="demo-bin",
        payload_filename="poc.txt",
        payload_content="payload\n",
        run_command="demo-bin /workspace/artifacts/poc/payloads/poc.txt",
    )
    previous_artifact = PoCArtifact(
        poc_filename="poc.txt",
        poc_content="payload\n",
        run_script_content="#!/bin/bash\ndemo payload\n",
        execution_success=True,
        reproducer_verified=False,
        execution_logs="execution_exit_code=0\n",
        observed_exit_code=0,
    )

    prompt = stage._build_llm_prompt(
        knowledge=knowledge,
        build=build,
        context=context,
        previous_plan=previous_plan,
        previous_artifact=previous_artifact,
    )

    assert "Previous run.sh:" in prompt
    assert "Previous payload content:" in prompt
    assert "Previous run_verify.yaml:" in prompt
    assert "Replan contract:" in prompt


def test_initial_prompt_uses_compact_reference_poc_summary():
    stage = poc_module.PocStage()
    knowledge = make_knowledge()
    build = make_build()
    context = poc_module.PocContext(
        cve_id=knowledge.cve_id,
        reference_poc_summaries=["FILE: poc.lua\nCONTENT:\n" + ("X" * 800)],
    )

    prompt = stage._build_llm_prompt(
        knowledge=knowledge,
        build=build,
        context=context,
        previous_plan=None,
        previous_artifact=None,
    )

    assert "SUMMARY:" in prompt
    assert "CONTENT:" not in prompt


def test_prompt_includes_structured_reproduction_recipes():
    stage = poc_module.PocStage()
    knowledge = make_knowledge(
        reproduction_recipes=[
            ReproductionRecipe(
                source_url="https://example.com/advisory",
                source_title="Advisory",
                steps=["git clone https://example.com/repo", "./demo {payload}"],
                run_commands=["./demo {payload}"],
                source_excerpt="How to reproduce:\n1. git clone ...\n2. ./demo payload",
            )
        ]
    )
    build = make_build()
    context = poc_module.PocContext(
        cve_id=knowledge.cve_id,
        reproduction_recipe_summaries=["steps:\n- git clone https://example.com/repo\n- ./demo {payload}"],
    )

    prompt = stage._build_llm_prompt(
        knowledge=knowledge,
        build=build,
        context=context,
        previous_plan=None,
        previous_artifact=None,
    )

    assert "Structured reproduction recipes:" in prompt
    assert "./demo {payload}" in prompt


def test_strategy_prompt_lists_available_choices():
    stage = poc_module.PocStage()
    knowledge = make_knowledge()
    build = make_build()
    context = poc_module.PocContext(
        cve_id=knowledge.cve_id,
        reproduction_recipe_summaries=["steps:\n- ./demo {payload}"],
        reference_poc_summaries=["SUMMARY: dataset poc"],
    )

    prompt = stage._build_strategy_prompt(
        knowledge=knowledge,
        build=build,
        context=context,
    )

    assert "Available strategies:" in prompt
    assert "reproduction_recipe" in prompt
    assert "dataset_poc" in prompt
    assert "llm_synthesized" in prompt


def test_prompt_requires_llm_synthesis_when_no_poc_evidence():
    stage = poc_module.PocStage()
    knowledge = make_knowledge(
        reproduction_hints=[],
        reproduction_recipes=[],
        expected_error_patterns=["segmentation fault"],
    )
    build = make_build(binary_or_entrypoint="demo-bin")
    context = poc_module.PocContext(
        cve_id=knowledge.cve_id,
        repo_evidence_blocks=["README: demo-bin accepts --input FILE"],
        candidate_entrypoints=["demo-bin"],
        inferred_input_modes=["file"],
    )

    prompt = stage._build_llm_prompt(
        knowledge=knowledge,
        build=build,
        context=context,
        previous_plan=None,
        previous_artifact=None,
    )

    assert "No explicit PoC or reproduction recipe was recovered" in prompt
    assert "You must synthesize the smallest plausible trigger" in prompt
    assert "source_of_truth to llm_synthesized" in prompt


def test_llm_prompt_pins_selected_strategy():
    stage = poc_module.PocStage()
    knowledge = make_knowledge()
    build = make_build(binary_or_entrypoint="demo-bin")
    context = poc_module.PocContext(
        cve_id=knowledge.cve_id,
        chosen_strategy="dataset_poc",
        chosen_strategy_rationale="existing poc needs adaptation",
    )

    prompt = stage._build_llm_prompt(
        knowledge=knowledge,
        build=build,
        context=context,
        previous_plan=None,
        previous_artifact=None,
    )

    assert "The strategy for this attempt is fixed to: dataset_poc." in prompt
    assert "Strategy rationale: existing poc needs adaptation" in prompt


def test_plan_poc_raises_when_llm_unavailable(monkeypatch):
    monkeypatch.setattr(poc_module, "build_chat_model", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("missing key")))

    stage = poc_module.PocStage()
    knowledge = make_knowledge()
    build = make_build(binary_or_entrypoint="demo-bin")
    context = poc_module.PocContext(cve_id=knowledge.cve_id, target_binary="demo-bin")

    try:
        stage.plan_poc(knowledge=knowledge, build=build, context=context)
    except RuntimeError as error:
        assert "poc_agent did not return a valid strategy" in str(error)
    else:
        raise AssertionError("expected plan_poc to fail without an llm plan")


def test_try_llm_plan_rejects_non_triggering_replan_without_substantive_changes(tmp_path, monkeypatch):
    class FakeResponse:
        def __init__(self, content):
            self.content = content

    class FakeModel:
        def invoke(self, messages):
            return FakeResponse(
                '{"trigger_mode":"cli-file","target_binary":"demo-bin","target_args":["--input","/workspace/artifacts/poc/payloads/old.txt"],'
                '"environment_variables":{},"payload_filename":"old.txt","payload_content":"old\\n","auxiliary_files":{},'
                '"run_command":"demo-bin --input /workspace/artifacts/poc/payloads/old.txt","expected_exit_code":null,'
                '"expected_stdout_patterns":[],"expected_stderr_patterns":["segmentation fault"],"expected_stack_keywords":[],"expected_crash_type":"segmentation fault",'
                '"source_of_truth":"llm","confidence":"high","rationale":"same plan","dockerfile_override":null,"run_script_override":null}'
            )

    monkeypatch.setattr(poc_module, "build_chat_model", lambda *args, **kwargs: FakeModel())

    stage = poc_module.PocStage()
    paths = poc_module.PocStagePaths(str(tmp_path / "ws"))
    stage._prepare_workspace(paths)
    stage._active_poc_dir = str(paths.poc_dir)
    knowledge = make_knowledge(expected_error_patterns=["segmentation fault"])
    build = make_build(binary_or_entrypoint="demo-bin")
    context = poc_module.PocContext(
        cve_id=knowledge.cve_id,
        planner_attempt=2,
        previous_failure_kind="non_triggering",
    )
    previous_plan = stage._normalize_poc_plan(
        poc_module.PocPlan(
            target_binary="demo-bin",
            target_args=["--input", "/workspace/artifacts/poc/payloads/old.txt"],
            payload_filename="old.txt",
            payload_content="old\n",
            run_command="demo-bin --input /workspace/artifacts/poc/payloads/old.txt",
            expected_stderr_patterns=["segmentation fault"],
            expected_crash_type="segmentation fault",
        )
    )
    previous_artifact = PoCArtifact(
        poc_filename="old.txt",
        poc_content="old\n",
        run_script_content="#!/bin/bash\ndemo-bin --input /workspace/artifacts/poc/payloads/old.txt\n",
        execution_success=True,
        reproducer_verified=False,
        execution_logs="execution_exit_code=0\n",
    )

    plan = stage._try_llm_poc_plan(
        knowledge=knowledge,
        build=build,
        context=context,
        previous_plan=previous_plan,
        previous_artifact=previous_artifact,
    )

    assert plan is None
    error_text = (paths.llm_dir / "attempt-2" / "error.txt").read_text(encoding="utf-8")
    assert "Rejected replan candidate" in error_text


def test_try_llm_plan_persists_llm_trace_files(tmp_path, monkeypatch):
    class FakeResponse:
        def __init__(self, content):
            self.content = content

    class FakeModel:
        def invoke(self, messages):
            return FakeResponse(
                '{"trigger_mode":"cli-file","target_binary":"demo-bin","target_args":["/workspace/artifacts/poc/payloads/llm.txt"],'
                '"environment_variables":{},"payload_filename":"llm.txt","payload_content":"llm\\n","auxiliary_files":{},'
                '"run_command":"demo-bin /workspace/artifacts/poc/payloads/llm.txt","expected_exit_code":null,'
                '"expected_stdout_patterns":[],"expected_stderr_patterns":["segmentation fault"],"expected_stack_keywords":[],"expected_crash_type":"segmentation fault",'
                '"source_of_truth":"llm","confidence":"high","rationale":"trace test","dockerfile_override":null,"run_script_override":"#!/bin/bash\\ndemo-bin /workspace/artifacts/poc/payloads/llm.txt\\n"}'
            )

    monkeypatch.setattr(poc_module, "build_chat_model", lambda *args, **kwargs: FakeModel())

    stage = poc_module.PocStage()
    paths = poc_module.PocStagePaths(str(tmp_path / "ws"))
    stage._prepare_workspace(paths)
    stage._active_poc_dir = str(paths.poc_dir)

    plan = stage._try_llm_poc_plan(
        knowledge=make_knowledge(expected_error_patterns=["segmentation fault"]),
        build=make_build(binary_or_entrypoint="demo-bin"),
        context=poc_module.PocContext(cve_id="CVE-2022-28805", planner_attempt=3),
    )

    assert plan is not None
    attempt_dir = paths.llm_dir / "attempt-3"
    assert (attempt_dir / "prompt.txt").exists()
    assert (attempt_dir / "response.txt").exists()
    assert (attempt_dir / "parsed.json").exists()


def test_try_llm_plan_retries_timeout_twice_before_success(tmp_path, monkeypatch):
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
                '{"trigger_mode":"cli-file","target_binary":"demo-bin","target_args":["/workspace/artifacts/poc/payloads/llm.txt"],'
                '"environment_variables":{},"payload_filename":"llm.txt","payload_content":"llm\\n","auxiliary_files":{},'
                '"run_command":"demo-bin /workspace/artifacts/poc/payloads/llm.txt","expected_exit_code":null,'
                '"expected_stdout_patterns":[],"expected_stderr_patterns":["segmentation fault"],"expected_stack_keywords":[],"expected_crash_type":"segmentation fault",'
                '"source_of_truth":"llm","confidence":"high","rationale":"retry success","dockerfile_override":null,"run_script_override":null}'
            )

    fake_model = FakeModel()
    monkeypatch.setattr(poc_module, "build_chat_model", lambda *args, **kwargs: fake_model)

    stage = poc_module.PocStage()
    paths = poc_module.PocStagePaths(str(tmp_path / "ws"))
    stage._prepare_workspace(paths)
    stage._active_poc_dir = str(paths.poc_dir)

    plan = stage._try_llm_poc_plan(
        knowledge=make_knowledge(expected_error_patterns=["segmentation fault"]),
        build=make_build(binary_or_entrypoint="demo-bin"),
        context=poc_module.PocContext(cve_id="CVE-2022-28805", planner_attempt=4),
    )

    assert plan is not None
    assert fake_model.calls == 3
    assert (paths.llm_dir / "attempt-4" / "response.txt").exists()


def test_try_llm_plan_uses_dedicated_poc_timeout(tmp_path, monkeypatch):
    class FakeResponse:
        def __init__(self, content):
            self.content = content

    captured = {}

    class FakeModel:
        def invoke(self, messages):
            return FakeResponse(
                '{"trigger_mode":"cli-file","target_binary":"demo-bin","target_args":["/workspace/artifacts/poc/payloads/llm.txt"],'
                '"environment_variables":{},"payload_filename":"llm.txt","payload_content":"llm\\n","auxiliary_files":{},'
                '"run_command":"demo-bin /workspace/artifacts/poc/payloads/llm.txt","expected_exit_code":null,'
                '"expected_stdout_patterns":[],"expected_stderr_patterns":["segmentation fault"],"expected_stack_keywords":[],"expected_crash_type":"segmentation fault",'
                '"source_of_truth":"llm","confidence":"high","rationale":"timeout capture","dockerfile_override":null,"run_script_override":null}'
            )

    def fake_build_chat_model(agent_name, model_name=None, temperature=0, timeout_seconds=None):
        captured["agent_name"] = agent_name
        captured["timeout_seconds"] = timeout_seconds
        return FakeModel()

    monkeypatch.setattr(poc_module, "build_chat_model", fake_build_chat_model)
    monkeypatch.setattr(
        poc_module,
        "load_app_config",
        lambda: SimpleNamespace(runtime=SimpleNamespace(poc_agent_timeout_seconds=95)),
    )

    stage = poc_module.PocStage()
    paths = poc_module.PocStagePaths(str(tmp_path / "ws"))
    stage._prepare_workspace(paths)
    stage._active_poc_dir = str(paths.poc_dir)

    plan = stage._try_llm_poc_plan(
        knowledge=make_knowledge(expected_error_patterns=["segmentation fault"]),
        build=make_build(binary_or_entrypoint="demo-bin"),
        context=poc_module.PocContext(cve_id="CVE-2022-28805", planner_attempt=6),
    )

    assert plan is not None
    assert captured["agent_name"] == "poc_agent"
    assert captured["timeout_seconds"] == 95


def test_try_llm_plan_records_final_error_after_three_empty_responses(tmp_path, monkeypatch):
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
    monkeypatch.setattr(poc_module, "build_chat_model", lambda *args, **kwargs: fake_model)

    stage = poc_module.PocStage()
    paths = poc_module.PocStagePaths(str(tmp_path / "ws"))
    stage._prepare_workspace(paths)
    stage._active_poc_dir = str(paths.poc_dir)

    plan = stage._try_llm_poc_plan(
        knowledge=make_knowledge(),
        build=make_build(binary_or_entrypoint="demo-bin"),
        context=poc_module.PocContext(cve_id="CVE-2022-28805", planner_attempt=5),
    )

    assert plan is None
    assert fake_model.calls == 3
    error_text = (paths.llm_dir / "attempt-5" / "error.txt").read_text(encoding="utf-8")
    assert "no content after 3 attempts" in error_text


def test_build_plan_prefers_compiled_image_tag_for_poc_base():
    stage = poc_module.PocStage()
    knowledge = make_knowledge()
    build = make_build(compiled_image_tag="demo:compiled", docker_image_tag="demo:build")

    plan_meta = stage.build_plan(knowledge=knowledge, build=build, workspace="/tmp/ws")

    assert plan_meta["base_image_tag"] == "demo:compiled"


def test_heuristic_plan_uses_interpreter_for_script_payload():
    stage = poc_module.PocStage()
    build = make_build(binary_or_entrypoint="")
    context = poc_module.PocContext(cve_id="CVE-2022-9999", inferred_input_modes=["file"])

    target_binary = stage._select_target_binary(build, context, "driver.py")
    trigger_mode = stage._infer_trigger_mode("driver.py", context)

    assert target_binary == "python3"
    assert trigger_mode == "script-driver"


def test_select_target_binary_prefers_build_image_project_dir_when_repo_url_known():
    stage = poc_module.PocStage()
    build = make_build(binary_or_entrypoint="lua")
    context = poc_module.PocContext(cve_id="CVE-2022-28805", repo_url="https://github.com/lua/lua.git")

    target_binary = stage._select_target_binary(build, context, "poc.lua")

    assert target_binary == "/src/lua/lua"


def test_normalize_run_command_rewrites_workspace_repo_binary_path():
    stage = poc_module.PocStage()

    command = stage._normalize_run_command(
        "/workspace/repo/lua {payload}",
        "poc.lua",
        repo_url="https://github.com/lua/lua.git",
    )

    assert command == "/src/lua/lua /workspace/artifacts/poc/payloads/poc.lua"


def test_normalize_poc_plan_aligns_bare_binary_command_with_container_path():
    stage = poc_module.PocStage()
    plan = poc_module.PocPlan(
        target_binary="lua",
        payload_filename="poc.lua",
        payload_content="print('x')\n",
        run_command="'lua' {payload}",
    )

    normalized = stage._normalize_poc_plan(plan, repo_url="https://github.com/lua/lua.git")

    assert normalized.target_binary == "/src/lua/lua"
    assert normalized.run_command == "'/src/lua/lua' /workspace/artifacts/poc/payloads/poc.lua"


def test_default_execution_dir_prefers_parent_of_absolute_target_binary():
    stage = poc_module.PocStage()

    assert stage._default_execution_dir("/src/lua/lua") == "/src/lua"
    assert stage._default_execution_dir("python3") == "/workspace"


def test_build_retry_context_truncates_large_previous_artifacts(tmp_path):
    stage = poc_module.PocStage()
    paths = poc_module.PocStagePaths(str(tmp_path / "ws"))
    stage._prepare_workspace(paths)
    paths.run_verify_yaml.write_text("Z" * 5000, encoding="utf-8")
    context = poc_module.PocContext(cve_id="CVE-2022-0000")
    artifact = PoCArtifact(
        poc_filename="poc.txt",
        poc_content="P" * 5000,
        run_script_content="R" * 5000,
        execution_logs="L" * 7000,
    )

    updated = stage._build_retry_context(context, paths, artifact)

    assert len(updated.previous_execution_log) <= stage.PREVIOUS_EXECUTION_LOG_CHAR_LIMIT + 32
    assert len(updated.previous_run_script_content) <= stage.PREVIOUS_RUN_SCRIPT_CHAR_LIMIT + 32
    assert len(updated.previous_payload_content) <= stage.PREVIOUS_PAYLOAD_CHAR_LIMIT + 32
    assert len(updated.previous_run_verify_report) <= stage.PREVIOUS_RUN_VERIFY_CHAR_LIMIT + 32


def test_poc_replan_gate_requires_dockerfile_override_for_docker_build_failure():
    stage = poc_module.PocStage()
    previous_plan = poc_module.PocPlan(target_binary="demo-bin", payload_filename="poc.txt", payload_content="old\n")
    candidate_plan = poc_module.PocPlan(
        target_binary="demo-bin",
        payload_filename="new.txt",
        payload_content="new\n",
    )

    assert stage._is_valid_replan_candidate(previous_plan, candidate_plan, failure_kind="docker_build") is False


def test_poc_replan_gate_requires_trigger_changes_for_non_triggering_failure():
    stage = poc_module.PocStage()
    previous_plan = poc_module.PocPlan(
        target_binary="demo-bin",
        payload_filename="old.txt",
        payload_content="old\n",
        run_command="demo-bin /workspace/artifacts/poc/payloads/old.txt",
    )
    candidate_plan = poc_module.PocPlan(
        target_binary="demo-bin",
        payload_filename="new.txt",
        payload_content="new\n",
        run_command="demo-bin /workspace/artifacts/poc/payloads/new.txt",
    )

    assert stage._is_valid_replan_candidate(previous_plan, candidate_plan, failure_kind="non_triggering") is True


def test_extract_execution_observation_parses_log_blocks():
    stage = poc_module.PocStage()
    logs = (
        "execution_exit_code=139\n"
        "stdout_begin\n"
        "hello\n"
        "stdout_end\n"
        "stderr_begin\n"
        "Segmentation fault\n"
        "stderr_end\n"
    )

    parsed = stage._extract_execution_observation(logs)

    assert parsed["observed_exit_code"] == 139
    assert parsed["observed_stdout"] == "hello"
    assert parsed["observed_crash_type"] == "segmentation fault"


def test_extract_execution_observation_falls_back_to_outer_stderr_section():
    stage = poc_module.PocStage()
    logs = (
        "container_run_success=True\n"
        "container_run_exit_code=0\n"
        "[container_run_stdout]\n"
        "execution_exit_code=139\n"
        "stdout_begin\n"
        "stdout_end\n"
        "stderr_begin\n"
        "stderr_end\n"
        "[container_run_stderr]\n"
        "Segmentation fault (core dumped)\n"
    )

    parsed = stage._extract_execution_observation(logs)

    assert parsed["observed_exit_code"] == 139
    assert "Segmentation fault" in parsed["observed_stderr"]
    assert parsed["observed_crash_type"] == "segmentation fault"


def test_build_run_verify_report_accepts_signal_exit_as_verify_eligible():
    stage = poc_module.PocStage()
    plan = poc_module.PocPlan(expected_crash_type="", expected_exit_code=None)
    observation = {
        "observed_exit_code": 139,
        "observed_stdout": "",
        "observed_stderr": "",
        "observed_crash_type": "",
    }

    report = stage._build_run_verify_report(
        plan=plan,
        observation=observation,
        execution_logs="execution_exit_code=139\nstdout_begin\nstdout_end\nstderr_begin\nstderr_end\n",
        matched_error_patterns=[],
        matched_stdout_patterns=[],
        matched_stack_keywords=[],
    )

    assert report.eligible_for_verify is True
    assert report.eligibility_reason == "signal_exit_observed: 139"


def test_execute_poc_plan_writes_files_and_returns_artifact(tmp_path):
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
                stdout = "target_binary=demo\ntrigger_command=demo poc\nexecution_exit_code=139\nstdout_begin\nok\nstdout_end\nstderr_begin\nsegmentation fault\nstderr_end\n"
                stderr = ""

            return Result()

    stage = poc_module.PocStage(docker_tool=FakeDockerTool())
    paths = poc_module.PocStagePaths(str(tmp_path / "ws"))
    stage._prepare_workspace(paths)
    (paths.repo_dir).mkdir(parents=True, exist_ok=True)

    plan = poc_module.PocPlan(
        target_binary="demo",
        target_args=["/workspace/artifacts/poc/payloads/poc.txt"],
        payload_filename="poc.txt",
        payload_content="boom\n",
        run_command="demo /workspace/artifacts/poc/payloads/poc.txt",
        expected_stderr_patterns=["segmentation fault"],
        expected_crash_type="segmentation fault",
    )

    artifact = stage._execute_poc_plan(
        paths=paths,
        plan_meta={"docker_image_tag": "demo:poc", "base_image_tag": "demo:build"},
        plan=plan,
    )

    assert artifact.execution_success is True
    assert artifact.reproducer_verified is True
    assert Path(paths.run_script).exists()
    assert Path(paths.payloads_dir / "poc.txt").exists()


def test_execute_poc_plan_marks_verified_on_stack_keyword_match(tmp_path):
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
                stdout = "execution_exit_code=1\nstdout_begin\nsinglevar reached\nstdout_end\nstderr_begin\n\nstderr_end\n"
                stderr = ""

            return Result()

    stage = poc_module.PocStage(docker_tool=FakeDockerTool())
    paths = poc_module.PocStagePaths(str(tmp_path / "ws"))
    stage._prepare_workspace(paths)
    paths.repo_dir.mkdir(parents=True, exist_ok=True)

    plan = poc_module.PocPlan(
        target_binary="demo",
        payload_filename="poc.txt",
        payload_content="boom\n",
        run_command="demo /workspace/artifacts/poc/payloads/poc.txt",
        expected_stack_keywords=["singlevar"],
    )

    artifact = stage._execute_poc_plan(
        paths=paths,
        plan_meta={"docker_image_tag": "demo:poc", "base_image_tag": "demo:build"},
        plan=plan,
    )

    assert artifact.matched_stack_keywords == ["singlevar"]
    assert artifact.reproducer_verified is True


def test_poc_node_records_retry_on_unsuccessful_execution(monkeypatch):
    artifact = PoCArtifact(
        poc_filename="poc.txt",
        poc_content="boom\n",
        run_script_content="#!/bin/bash\nexit 0\n",
        execution_success=False,
        execution_logs="not triggered",
    )

    class FakeStage:
        def run(self, knowledge, build, workspace):
            return artifact

    monkeypatch.setattr(poc_module, "PocStage", FakeStage)

    state = {
        "knowledge": make_knowledge(),
        "build": make_build(),
        "workspace": "workspaces/CVE-2022-28805",
        "retry_count": {},
        "stage_history": [],
    }

    result = poc_module.poc_node(state)

    assert result["poc"].execution_success is False
    assert result["retry_count"]["poc"] == 1
    assert result["stage_history"][-1]["status"] == "failed"


def test_poc_artifact_persists_plan_fields_for_verify(tmp_path):
    """Fix 2-3.A: env vars / expected_stack_keywords / expected_crash_type
    must be copied from PocPlan into PoCArtifact so verify can consume them."""

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
                stdout = (
                    "target_binary=demo\ntrigger_command=demo poc\n"
                    "execution_exit_code=0\n"
                    "stdout_begin\nok\nstdout_end\n"
                    "stderr_begin\n\nstderr_end\n"
                )
                stderr = ""

            return Result()

    stage = poc_module.PocStage(docker_tool=FakeDockerTool())
    paths = poc_module.PocStagePaths(str(tmp_path / "ws"))
    stage._prepare_workspace(paths)
    paths.repo_dir.mkdir(parents=True, exist_ok=True)

    plan = poc_module.PocPlan(
        target_binary="demo",
        payload_filename="poc.txt",
        payload_content="boom\n",
        run_command="demo /workspace/artifacts/poc/payloads/poc.txt",
        expected_stack_keywords=["singlevar"],
        expected_crash_type="heap-buffer-overflow",
        environment_variables={"ASAN_OPTIONS": "detect_leaks=0"},
    )

    artifact = stage._execute_poc_plan(
        paths=paths,
        plan_meta={"docker_image_tag": "demo:poc", "base_image_tag": "demo:build"},
        plan=plan,
    )

    assert artifact.environment_variables == {"ASAN_OPTIONS": "detect_leaks=0"}
    assert artifact.expected_stack_keywords == ["singlevar"]
    assert artifact.expected_crash_type == "heap-buffer-overflow"


# ===== Fix 1.B: stdout/stderr matching is stream-aware =====
def test_match_patterns_separates_stdout_and_stderr(tmp_path):
    """模式放在错误的流里时不应被命中——证明 stream-aware matching 真在工作。"""

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
                stdout = (
                    "target_binary=demo\ntrigger_command=demo poc\n"
                    "execution_exit_code=0\n"
                    "stdout_begin\n"
                    "some output with needle_err but not the other\n"
                    "stdout_end\n"
                    "stderr_begin\n"
                    "error log with needle_out but not the other\n"
                    "stderr_end\n"
                )
                stderr = ""

            return Result()

    stage = poc_module.PocStage(docker_tool=FakeDockerTool())
    paths = poc_module.PocStagePaths(str(tmp_path / "ws"))
    stage._prepare_workspace(paths)
    paths.repo_dir.mkdir(parents=True, exist_ok=True)

    plan = poc_module.PocPlan(
        target_binary="demo",
        payload_filename="poc.txt",
        payload_content="boom\n",
        run_command="demo /workspace/artifacts/poc/payloads/poc.txt",
        expected_stdout_patterns=["needle_out"],   # 实际出现在 stderr
        expected_stderr_patterns=["needle_err"],   # 实际出现在 stdout
    )

    artifact = stage._execute_poc_plan(
        paths=paths,
        plan_meta={"docker_image_tag": "demo:poc", "base_image_tag": "demo:build"},
        plan=plan,
    )

    # 关键：流分离正确，错误流的模式不会跨流命中
    assert artifact.matched_stdout_patterns == []
    assert artifact.matched_stderr_patterns == []
    assert artifact.matched_error_patterns == []  # 与 stderr 同步


# ===== Fix 2.A: replan continues when executed_but_not_verified =====
def test_replan_continues_when_executed_but_not_verified(tmp_path, monkeypatch):
    """第一次跑 reproducer_verified=False，replan 应被触发；第二次成功后停止。"""

    stage = poc_module.PocStage()

    # Mock context-collection / planning / persistence to focus on the replan loop
    fake_context = poc_module.PocContext(cve_id="CVE-2022-0000")
    fake_plan = poc_module.PocPlan(target_binary="demo", payload_filename="poc.txt")

    monkeypatch.setattr(stage, "collect_poc_context", lambda **kw: fake_context)
    monkeypatch.setattr(stage, "plan_poc", lambda **kw: fake_plan)
    monkeypatch.setattr(stage, "_prepare_workspace", lambda paths: None)
    monkeypatch.setattr(stage.file_tool, "write_text", lambda path, content: None)

    call_count = {"execute": 0, "replan": 0}

    def fake_execute(paths, plan_meta, plan):
        call_count["execute"] += 1
        if call_count["execute"] == 1:
            return PoCArtifact(
                poc_filename="poc.txt", poc_content="x", run_script_content="y",
                execution_success=True, reproducer_verified=False, execution_logs="...",
            )
        return PoCArtifact(
            poc_filename="poc.txt", poc_content="x", run_script_content="y",
            execution_success=True, reproducer_verified=True, execution_logs="...",
        )

    def fake_replan(**kwargs):
        call_count["replan"] += 1
        return fake_plan

    monkeypatch.setattr(stage, "_execute_poc_plan", fake_execute)
    monkeypatch.setattr(stage, "replan_after_failure", fake_replan)

    knowledge = make_knowledge()
    build = make_build()

    artifact = stage.run(knowledge=knowledge, build=build, workspace=str(tmp_path / "ws"))

    assert artifact.reproducer_verified is True
    assert call_count["replan"] >= 1


def test_replan_stops_when_max_attempts_reached(tmp_path, monkeypatch):
    """execution_success=True 但 reproducer_verified=False 时，replan 不会无限循环。"""

    stage = poc_module.PocStage()
    monkeypatch.setattr(poc_module.PocStage, "MAX_REPLAN_ATTEMPTS", 2)

    fake_context = poc_module.PocContext(cve_id="CVE-2022-0000")
    fake_plan = poc_module.PocPlan(target_binary="demo", payload_filename="poc.txt")

    monkeypatch.setattr(stage, "collect_poc_context", lambda **kw: fake_context)
    monkeypatch.setattr(stage, "plan_poc", lambda **kw: fake_plan)
    monkeypatch.setattr(stage, "_prepare_workspace", lambda paths: None)
    monkeypatch.setattr(stage.file_tool, "write_text", lambda path, content: None)
    monkeypatch.setattr(stage, "replan_after_failure", lambda **kw: fake_plan)

    execute_count = {"n": 0}

    def fake_execute(paths, plan_meta, plan):
        execute_count["n"] += 1
        return PoCArtifact(
            poc_filename="poc.txt", poc_content="x", run_script_content="y",
            execution_success=True, reproducer_verified=False, execution_logs="...",
        )

    monkeypatch.setattr(stage, "_execute_poc_plan", fake_execute)

    stage.run(knowledge=make_knowledge(), build=make_build(), workspace=str(tmp_path / "ws"))

    # 终止性断言：MAX_REPLAN_ATTEMPTS=2 时最多 2*2=4 次执行（每 attempt 最多 1 initial + 1 replan）
    assert 0 < execute_count["n"] <= 4
    # 最关键的：跑完了——没有无限循环


# ===== Fix 2.C: poc_node history reflects three-way state =====
def test_poc_node_history_executed_but_unverified(monkeypatch):
    artifact = PoCArtifact(
        poc_filename="poc.txt", poc_content="x", run_script_content="y",
        execution_success=True,
        reproducer_verified=False,
        execution_logs="ran but no signal",
    )

    class FakeStage:
        def run(self, knowledge, build, workspace):
            return artifact

    monkeypatch.setattr(poc_module, "PocStage", FakeStage)

    state = {
        "knowledge": make_knowledge(),
        "build": make_build(),
        "workspace": "workspaces/CVE-2022-28805",
        "retry_count": {},
        "stage_history": [],
    }

    result = poc_module.poc_node(state)

    assert result["current_stage"] == "verify"
    assert result["stage_history"][-1]["stage"] == "poc"
    assert result["stage_history"][-1]["status"] == "executed_but_unverified"
    assert "deferring to verify" in result["stage_history"][-1]["note"]


def test_poc_node_history_full_success(monkeypatch):
    """回归：execution_success=True && reproducer_verified=True → status=success."""

    artifact = PoCArtifact(
        poc_filename="poc.txt", poc_content="x", run_script_content="y",
        execution_success=True,
        reproducer_verified=True,
        execution_logs="...",
    )

    class FakeStage:
        def run(self, knowledge, build, workspace):
            return artifact

    monkeypatch.setattr(poc_module, "PocStage", FakeStage)

    state = {
        "knowledge": make_knowledge(),
        "build": make_build(),
        "workspace": "workspaces/CVE-2022-28805",
        "retry_count": {},
        "stage_history": [],
    }

    result = poc_module.poc_node(state)

    assert result["current_stage"] == "verify"
    assert result["stage_history"][-1]["status"] == "success"


# ===== Fix 2.E: route_after_poc still advances on executed_but_unverified =====
def test_route_after_poc_advances_when_executed_but_unverified():
    """H5 设计文档：execution_success=True 即推进 verify，无视 reproducer_verified。"""

    from app.orchestrator.routers import route_after_poc

    state = {
        "poc": PoCArtifact(
            poc_filename="x", poc_content="", run_script_content="",
            execution_success=True,
            reproducer_verified=False,  # 关键：未触发
        ),
        "retry_count": {},
    }
    assert route_after_poc(state) == "verify"
