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


def test_apply_vulnerable_hdf5_policy_when_build_script_has_marker():
    stage = poc_module.PocStage()
    plan = poc_module.PocPlan(
        target_binary="matio_fuzzer",
        expected_stderr_patterns=["AddressSanitizer"],
        expected_stack_keywords=["AddressSanitizer"],
        expected_crash_type="",
        environment_variables={},
    )
    build = make_build(
        build_script_content="#!/bin/bash\necho deeprepro:vulnerable-hdf5-1.12.0\n"
    )
    updated = stage._apply_vulnerable_hdf5_cve_match_policy_if_gated(plan, build)
    assert updated.expected_stderr_patterns == ["heap-buffer-overflow", "H5MM_memcpy"]
    assert "AddressSanitizer" not in updated.expected_stderr_patterns
    assert "H5MM_memcpy" in updated.expected_stack_keywords
    assert "H5C_load_entry" not in updated.expected_stack_keywords
    assert updated.expected_crash_type == "heap-buffer-overflow"
    assert "allocator_may_return_null=1" in updated.environment_variables["ASAN_OPTIONS"]


def test_apply_vulnerable_hdf5_policy_noop_without_marker():
    stage = poc_module.PocStage()
    plan = poc_module.PocPlan(
        target_binary="demo",
        expected_stderr_patterns=["AddressSanitizer"],
        environment_variables={},
    )
    build = make_build()
    updated = stage._apply_vulnerable_hdf5_cve_match_policy_if_gated(plan, build)
    assert updated.expected_stderr_patterns == ["AddressSanitizer"]
    assert "ASAN_OPTIONS" not in updated.environment_variables


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
    )

    plan = stage.plan_poc(knowledge=knowledge, build=build, context=context)

    assert plan.source_of_truth == "llm_synthesized"
    assert plan.payload_filename == "llm.txt"
    assert context.chosen_strategy == "llm_synthesized"
    assert fake_model.calls == 2


def test_plan_poc_forces_dataset_strategy_when_dataset_payload_exists(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            raise AssertionError("LLM must not be called when authoritative dataset bytes exist")

    import base64

    fake_model = FakeModel()
    monkeypatch.setattr(poc_module, "build_chat_model", lambda *args, **kwargs: fake_model)
    stage = poc_module.PocStage()
    knowledge = make_knowledge(
        expected_error_patterns=["AddressSanitizer"],
        vulnerability_type="heap-based buffer over-read",
    )
    build = make_build(binary_or_entrypoint="secilc/secilc")
    payload = b"(class file)\n(classmap file file)\n"
    context = poc_module.PocContext(
        cve_id=knowledge.cve_id,
        repo_url=knowledge.repo_url or "",
        dataset_poc_filenames=["poc.cil"],
        dataset_poc_base64_blobs=[base64.b64encode(payload).decode("ascii")],
        reference_poc_summaries=["FILE: poc.cil\nENCODING: text\nCONTENT:\n(class file)"],
    )

    plan = stage.plan_poc(knowledge=knowledge, build=build, context=context)

    assert context.chosen_strategy == "dataset_poc"
    assert plan.payload_filename == "poc.cil"
    assert plan.payload_content == payload.decode("latin-1")
    assert plan.source_of_truth == "dataset_poc"
    assert "secilc" in plan.target_binary
    assert plan.payload_filename in " ".join(plan.target_args + [plan.run_command])
    assert fake_model.calls == 0


def test_summarize_reproduction_recipes_omits_large_base64_blobs():
    stage = poc_module.PocStage()
    huge = "A" * 5000
    recipes = [
        ReproductionRecipe(
            source_url="https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=1",
            source_title="clusterfuzz-testcase.cil",
            recipe_type="ossfuzz_testcase",
            steps=[f"printf '%s' '{huge}' | base64 -d > poc.cil"],
            artifact_generation_commands=[f"printf '%s' '{huge}' | base64 -d > poc.cil"],
            source_excerpt="Harvested OSS-Fuzz testcase",
            confidence="high",
        )
    ]

    summaries = stage._summarize_reproduction_recipes(recipes)

    assert len(summaries) == 1
    assert huge not in summaries[0]
    assert "large base64 payload omitted" in summaries[0]


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
    # Authoritative dataset evidence skips the strategy selector on the first turn.
    assert fake_model.calls == 2


def test_classify_failure_kind_detects_payload_invalid():
    stage = poc_module.PocStage()
    logs = (
        "container_run_success=True\n"
        "stderr_begin\n"
        "Invalid syntax\n"
        "Bad classpermission declaration at /workspace/artifacts/poc/payloads/poc.cil:5\n"
        "Failed to compile cildb: -1\n"
        "stderr_end\n"
        "execution_exit_code=255\n"
    )
    assert stage._classify_failure_kind(logs) == "payload_invalid"


def test_poc_replan_gate_requires_payload_change_for_payload_invalid():
    stage = poc_module.PocStage()
    previous_plan = poc_module.PocPlan(
        target_binary="secilc",
        payload_filename="poc.cil",
        payload_content="(bad)\n",
        run_command="secilc poc.cil",
    )
    same_payload = poc_module.PocPlan(
        target_binary="secilc",
        payload_filename="poc.cil",
        payload_content="(bad)\n",
        run_command="secilc -v poc.cil",
    )
    changed_payload = poc_module.PocPlan(
        target_binary="secilc",
        payload_filename="poc.cil",
        payload_content="(class file)\n",
        run_command="secilc poc.cil",
    )
    assert stage._is_valid_replan_candidate(previous_plan, same_payload, failure_kind="payload_invalid") is False
    assert stage._is_valid_replan_candidate(previous_plan, changed_payload, failure_kind="payload_invalid") is True


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


def test_select_target_binary_prefers_ossfuzz_harness_with_evidence(tmp_path):
    stage = poc_module.PocStage()
    repo = tmp_path / "fluent-bit"
    fuzz_dir = repo / "tests" / "internal" / "fuzzers"
    fuzz_dir.mkdir(parents=True)
    (repo / "CMakeLists.txt").write_text("project(fluent-bit)\n", encoding="utf-8")
    (fuzz_dir / "parser_fuzzer.c").write_text("int LLVMFuzzerTestOneInput(){return 0;}\n", encoding="utf-8")
    (fuzz_dir / "CMakeLists.txt").write_text("parser_fuzzer.c\n", encoding="utf-8")

    build = make_build(
        binary_or_entrypoint="build/bin/fluent-bit",
        expected_binary_path="build/bin/fluent-bit",
        repo_local_path=str(repo),
    )
    context = poc_module.PocContext(
        cve_id="CVE-2021-36088",
        repo_url="https://github.com/fluent/fluent-bit.git",
    )
    payload = "clusterfuzz-testcase-minimized-flb-it-fuzz-parser_fuzzer_OSSFUZZ-5216297967288320.fuzz"

    target_binary = stage._select_target_binary(build, context, payload)

    assert target_binary == "/src/fluent-bit/build/bin/flb-it-fuzz-parser_fuzzer"


def test_select_target_binary_keeps_secilc_without_harness_evidence(tmp_path):
    stage = poc_module.PocStage()
    repo = tmp_path / "selinux"
    (repo / "secilc").mkdir(parents=True)
    (repo / "secilc" / "secilc").write_text("", encoding="utf-8")

    build = make_build(
        binary_or_entrypoint="secilc/secilc",
        expected_binary_path="secilc/secilc",
        repo_local_path=str(repo),
    )
    context = poc_module.PocContext(
        cve_id="CVE-2021-36086",
        repo_url="https://github.com/SELinuxProject/selinux.git",
    )
    payload = "clusterfuzz-testcase-minimized-secilc-fuzzer-5563841674084352.cil"

    target_binary = stage._select_target_binary(build, context, payload)

    assert target_binary == "/src/selinux/secilc/secilc"


def test_select_target_binary_prefers_standalone_ossfuzz_cpp(tmp_path):
    stage = poc_module.PocStage()
    repo = tmp_path / "matio"
    ossfuzz_dir = repo / "ossfuzz"
    ossfuzz_dir.mkdir(parents=True)
    (ossfuzz_dir / "matio_fuzzer.cpp").write_text(
        'extern "C" int LLVMFuzzerTestOneInput(){return 0;}\n',
        encoding="utf-8",
    )

    build = make_build(
        binary_or_entrypoint="",
        expected_binary_path=None,
        repo_local_path=str(repo),
    )
    context = poc_module.PocContext(
        cve_id="CVE-2021-36977",
        repo_url="https://github.com/tbeu/matio.git",
    )
    payload = "clusterfuzz-testcase-minimized-matio_fuzzer-4806922097262592"

    target_binary = stage._select_target_binary(build, context, payload)

    assert target_binary == "/src/matio/matio_fuzzer"
    assert not target_binary.endswith("/target")


def test_normalize_run_command_rewrites_workspace_repo_binary_path():
    stage = poc_module.PocStage()

    command = stage._normalize_run_command(
        "/workspace/repo/lua {payload}",
        "poc.lua",
        repo_url="https://github.com/lua/lua.git",
    )

    assert command == "/src/lua/lua /workspace/artifacts/poc/payloads/poc.lua"


def test_normalize_poc_plan_strips_pre_run_make_rebuild():
    stage = poc_module.PocStage()
    plan = poc_module.PocPlan(
        target_binary="/src/selinux/secilc/secilc",
        payload_filename="poc.cil",
        payload_content="(class CLASS (PERM))\n",
        run_command=(
            "cd /src/selinux && make -C secilc && "
            "ASAN_OPTIONS=detect_leaks=0:abort_on_error=1 "
            "./secilc/secilc /workspace/artifacts/poc/payloads/poc.cil"
        ),
    )

    normalized = stage._normalize_poc_plan(
        plan,
        repo_url="https://github.com/SELinuxProject/selinux.git",
    )

    assert "make" not in normalized.run_command
    assert normalized.run_command == (
        "cd /src/selinux && "
        "ASAN_OPTIONS=detect_leaks=0:abort_on_error=1 "
        "./secilc/secilc /workspace/artifacts/poc/payloads/poc.cil"
    )


def test_normalize_poc_plan_rewrites_qt_plugin_so_into_qimage_harness():
    stage = poc_module.PocStage()
    plan = poc_module.PocPlan(
        target_binary="/src/kimageformats/build/src/imageformats/kimg_xcf.so",
        payload_filename="crafted.xcf",
        payload_content="xcf-bytes\n",
        run_command=(
            "cd /src/kimageformats && QT_DEBUG_PLUGINS=1 "
            "build/src/imageformats/kimg_xcf.so /workspace/artifacts/poc/payloads/crafted.xcf"
        ),
        expected_stderr_patterns=["Stack-buffer-overflow WRITE"],
    )

    normalized = stage._normalize_poc_plan(
        plan,
        repo_url="https://invent.kde.org/frameworks/kimageformats.git",
    )

    assert normalized.target_binary == "/src/kimageformats/build/bin/imageformats/kimg_xcf.so"
    assert normalized.trigger_mode == "library-harness"
    assert "inputs/qimage_harness.cpp" in normalized.auxiliary_files
    assert "QImageReader" in normalized.auxiliary_files["inputs/qimage_harness.cpp"]
    assert "qimage_harness" in normalized.run_command
    assert "QT_PLUGIN_PATH='/src/kimageformats/build/bin'" in normalized.run_command
    assert "libclang_rt.asan-x86_64.so" in normalized.run_command
    assert "LD_LIBRARY_PATH" in normalized.run_command
    assert "ASAN_OPTIONS=" in normalized.run_command
    # Loader is unsanitized; ASan comes from LD_PRELOAD + the instrumented plugin.
    assert "-fsanitize=address" not in normalized.run_command
    assert "-shared-libasan" not in normalized.run_command
    assert "timeout 90s env LD_PRELOAD=" in normalized.run_command
    assert "export LD_PRELOAD" not in normalized.run_command
    assert normalized.run_command.index("clang++") < normalized.run_command.index("timeout 90s env LD_PRELOAD=")
    assert "abort_on_error=1" in normalized.run_command
    assert "halt_on_error=1" in normalized.run_command
    assert "alloc_dealloc_mismatch=0" in normalized.run_command
    assert "symbolize=1" in normalized.run_command
    assert "fast_unwind_on_fatal=0" in normalized.run_command
    assert "handle_segv" not in normalized.run_command
    assert "timeout 90s" in normalized.run_command
    assert "kimg_xcf.so /workspace" not in normalized.run_command
    assert normalized.environment_variables.get("QT_PLUGIN_PATH") == "/src/kimageformats/build/bin"
    assert "abort_on_error=1" in normalized.environment_variables.get("ASAN_OPTIONS", "")
    assert "alloc_dealloc_mismatch=0" in normalized.environment_variables.get("ASAN_OPTIONS", "")
    assert "symbolize=1" in normalized.environment_variables.get("ASAN_OPTIONS", "")
    assert "handle_segv" not in normalized.environment_variables.get("ASAN_OPTIONS", "")


def test_discover_candidate_binaries_skips_docs_and_finds_kimg_plugin(tmp_path):
    stage = poc_module.PocStage()
    repo = tmp_path / "kimageformats"
    src = repo / "src" / "imageformats"
    plugin_dir = repo / "build" / "bin" / "imageformats"
    src.mkdir(parents=True)
    plugin_dir.mkdir(parents=True)
    (src / "AUTHORS").write_text("names\n", encoding="utf-8")
    (src / "xcf.cpp").write_text("int x;\n", encoding="utf-8")
    (plugin_dir / "kimg_xcf.so").write_bytes(b"\x7fELF")

    found = stage._discover_candidate_binaries(repo)
    assert "src/imageformats/AUTHORS" not in found
    assert any(Path(item).name == "kimg_xcf.so" for item in found)


def test_normalize_poc_plan_rewrites_authors_doc_to_kimg_plugin():
    stage = poc_module.PocStage()
    plan = poc_module.PocPlan(
        target_binary="/src/kimageformats/src/imageformats/AUTHORS",
        payload_filename="crafted_bpp5.xcf",
        payload_content="xcf-bytes\n",
        run_command=(
            "'/src/kimageformats/src/imageformats/AUTHORS' "
            "'/workspace/artifacts/poc/payloads/crafted_bpp5.xcf'"
        ),
        expected_stderr_patterns=["stack-buffer-overflow"],
    )

    normalized = stage._normalize_poc_plan(
        plan,
        repo_url="https://invent.kde.org/frameworks/kimageformats.git",
    )

    assert normalized.target_binary == "/src/kimageformats/build/bin/imageformats/kimg_xcf.so"
    assert normalized.trigger_mode == "library-harness"
    assert "qimage_harness" in normalized.run_command
    assert "AUTHORS" not in normalized.run_command


def test_select_target_binary_skips_authors_for_xcf_payload():
    stage = poc_module.PocStage()
    build = make_build(binary_or_entrypoint="", expected_binary_path=None)
    context = poc_module.PocContext(
        cve_id="CVE-2021-36083",
        repo_url="https://invent.kde.org/frameworks/kimageformats.git",
        candidate_entrypoints=["src/imageformats/AUTHORS", "src/imageformats/xcf.cpp"],
    )
    target = stage._select_target_binary(build, context, "crafted_bpp5.xcf")
    assert target == "/src/kimageformats/build/bin/imageformats/kimg_xcf.so"
    stage = poc_module.PocStage()
    assert (
        stage._correct_qt_plugin_binary_path("/src/kimageformats/build/bin/kimg_xcf.so")
        == "/src/kimageformats/build/bin/imageformats/kimg_xcf.so"
    )
    already = "/src/kimageformats/build/bin/imageformats/kimg_xcf.so"
    assert stage._correct_qt_plugin_binary_path(already) == already


def test_normalize_poc_plan_rewrites_bare_build_bin_kimg_into_harness():
    stage = poc_module.PocStage()
    plan = poc_module.PocPlan(
        target_binary="/src/kimageformats/build/bin/kimg_xcf.so",
        payload_filename="crafted_bpp5.xcf",
        payload_content="xcf-bytes\n",
        run_command=(
            "cd /src/kimageformats && "
            "/src/kimageformats/build/bin/kimg_xcf.so "
            "/workspace/artifacts/poc/payloads/crafted_bpp5.xcf"
        ),
        expected_stderr_patterns=["Stack-buffer-overflow WRITE"],
    )

    normalized = stage._normalize_poc_plan(
        plan,
        repo_url="https://invent.kde.org/frameworks/kimageformats.git",
    )

    assert normalized.target_binary == "/src/kimageformats/build/bin/imageformats/kimg_xcf.so"
    assert "QT_PLUGIN_PATH='/src/kimageformats/build/bin'" in normalized.run_command
    assert "qimage_harness" in normalized.run_command


def test_order_dataset_poc_payloads_prefers_crafted_bpp_xcf():
    import base64

    stage = poc_module.PocStage()
    names = [
        "clusterfuzz-testcase-minimized-xcf_fuzzer-1234567890.xcf",
        "crafted_bpp5.xcf",
        "notes.xcf",
    ]
    blobs = [
        base64.b64encode(b"A" * 200).decode("ascii"),
        base64.b64encode(b"bpp5").decode("ascii"),
        base64.b64encode(b"note").decode("ascii"),
    ]
    ordered_names, _ = stage._order_dataset_poc_payloads(names, blobs)
    assert ordered_names[0] == "crafted_bpp5.xcf"
    assert ordered_names[-1].startswith("clusterfuzz")


def test_order_dataset_poc_payloads_prefers_ossfuzz_over_tiny_poc_cil():
    import base64

    stage = poc_module.PocStage()
    names = [
        "poc.cil",
        "clusterfuzz-testcase-minimized-secilc-fuzzer-5563841674084352.cil",
    ]
    blobs = [
        base64.b64encode(b"(blockinherit b3)\n").decode("ascii"),
        base64.b64encode(b"ossfuzz-seed" * 40).decode("ascii"),
    ]
    ordered_names, _ = stage._order_dataset_poc_payloads(names, blobs)
    assert ordered_names[0].startswith("clusterfuzz")
    assert ordered_names[-1] == "poc.cil"


def test_prefer_ossfuzz_minimized_payload_switches_from_generic_poc_cil():
    import base64

    stage = poc_module.PocStage()
    ossfuzz = b"clusterfuzz-minimized-seed"
    plan = poc_module.PocPlan(
        target_binary="/src/selinux/secilc/secilc",
        target_args=["/workspace/artifacts/poc/payloads/poc.cil"],
        payload_filename="poc.cil",
        payload_content="(blockinherit b3)\n",
        run_command="'/src/selinux/secilc/secilc' '/workspace/artifacts/poc/payloads/poc.cil'",
        source_of_truth="dataset_poc",
    )
    updated = stage._prefer_ossfuzz_minimized_payload(
        plan,
        dataset_poc_filenames=[
            "poc.cil",
            "clusterfuzz-testcase-minimized-secilc-fuzzer-5563841674084352.cil",
        ],
        dataset_poc_base64_blobs=[
            base64.b64encode(b"(blockinherit b3)\n").decode("ascii"),
            base64.b64encode(ossfuzz).decode("ascii"),
        ],
    )
    updated = stage._sync_payload_filename_into_command(updated)
    assert updated.payload_filename.startswith("clusterfuzz")
    assert updated.payload_content == ossfuzz.decode("latin-1")
    assert updated.target_args == [
        "/workspace/artifacts/poc/payloads/clusterfuzz-testcase-minimized-secilc-fuzzer-5563841674084352.cil"
    ]
    assert "clusterfuzz-testcase-minimized-secilc-fuzzer-5563841674084352.cil" in updated.run_command
    assert "poc.cil" not in updated.run_command


def test_prefer_ossfuzz_minimized_payload_does_not_override_crafted_bpp():
    import base64

    stage = poc_module.PocStage()
    plan = poc_module.PocPlan(
        target_binary="/src/kimageformats/build/bin/imageformats/kimg_xcf.so",
        payload_filename="crafted_bpp5.xcf",
        payload_content="small-xcf",
        source_of_truth="dataset_poc",
    )
    updated = stage._prefer_ossfuzz_minimized_payload(
        plan,
        dataset_poc_filenames=[
            "crafted_bpp5.xcf",
            "clusterfuzz-testcase-minimized-xcf_fuzzer-999.xcf",
        ],
        dataset_poc_base64_blobs=[
            base64.b64encode(b"small-xcf").decode("ascii"),
            base64.b64encode(b"Y" * 5000).decode("ascii"),
        ],
    )
    assert updated.payload_filename == "crafted_bpp5.xcf"
    assert updated.payload_content == "small-xcf"


def test_prefer_compact_dataset_xcf_payload_switches_from_clusterfuzz():
    import base64

    stage = poc_module.PocStage()
    crafted = b"small-xcf-payload"
    plan = poc_module.PocPlan(
        target_binary="/src/kimageformats/build/bin/imageformats/kimg_xcf.so",
        payload_filename="clusterfuzz-testcase-minimized-xcf_fuzzer-999.xcf",
        payload_content="X" * 5000,
        source_of_truth="llm_synthesized",
    )
    updated = stage._prefer_compact_dataset_xcf_payload(
        plan,
        dataset_poc_filenames=[
            "clusterfuzz-testcase-minimized-xcf_fuzzer-999.xcf",
            "crafted_bpp5.xcf",
        ],
        dataset_poc_base64_blobs=[
            base64.b64encode(b"Y" * 5000).decode("ascii"),
            base64.b64encode(crafted).decode("ascii"),
        ],
    )
    assert updated.payload_filename == "crafted_bpp5.xcf"
    assert updated.payload_content == crafted.decode("latin-1")
    assert updated.source_of_truth == "dataset_poc"


def test_normalize_poc_plan_drops_generic_asan_when_specific_bug_present():
    stage = poc_module.PocStage()
    plan = poc_module.PocPlan(
        target_binary="/src/kimageformats/build/bin/imageformats/kimg_xcf.so",
        payload_filename="crafted_bpp5.xcf",
        payload_content="xcf\n",
        expected_stderr_patterns=[
            "AddressSanitizer",
            "stack-buffer-overflow",
            "AddressSanitizer: stack-buffer-overflow",
        ],
        expected_stack_keywords=["AddressSanitizer", "loadHierarchy"],
        expected_crash_type="heap-buffer-overflow",
    )

    normalized = stage._normalize_poc_plan(
        plan,
        repo_url="https://invent.kde.org/frameworks/kimageformats.git",
    )

    assert "AddressSanitizer" not in normalized.expected_stderr_patterns
    assert "stack-buffer-overflow" in normalized.expected_stderr_patterns
    assert "AddressSanitizer: stack-buffer-overflow" in normalized.expected_stderr_patterns
    assert "AddressSanitizer" not in normalized.expected_stack_keywords
    assert "loadHierarchy" in normalized.expected_stack_keywords
    assert "alloc_dealloc_mismatch=0" in normalized.environment_variables.get("ASAN_OPTIONS", "")


def test_normalize_poc_plan_keeps_overflow_token_and_drops_segv():
    stage = poc_module.PocStage()
    plan = poc_module.PocPlan(
        target_binary="/src/selinux/secilc/secilc",
        payload_filename="clusterfuzz-testcase-minimized-secilc-fuzzer-1.cil",
        payload_content="seed\n",
        expected_stderr_patterns=["SEGV", "null dereference", "heap-based buffer over-read"],
        expected_crash_type="heap-buffer-overflow",
    )

    normalized = stage._normalize_poc_plan(
        plan,
        repo_url="https://github.com/SELinuxProject/selinux.git",
    )

    assert "heap-buffer-overflow" in normalized.expected_stderr_patterns
    assert "SEGV" not in normalized.expected_stderr_patterns
    assert "null dereference" not in normalized.expected_stderr_patterns


def test_extract_execution_observation_prefers_asan_overflow_over_aborting():
    stage = poc_module.PocStage()
    logs = (
        "execution_exit_code=1\n"
        "stdout_begin\n"
        "stdout_end\n"
        "stderr_begin\n"
        "ERROR: AddressSanitizer: heap-buffer-overflow on address 0x1\n"
        "==41==ABORTING\n"
        "stderr_end\n"
    )

    parsed = stage._extract_execution_observation(logs)

    assert parsed["observed_crash_type"] == "heap-buffer-overflow"


def test_build_run_verify_report_eligible_when_overflow_in_haystack_not_patterns():
    stage = poc_module.PocStage()
    stderr = (
        "ERROR: AddressSanitizer: heap-buffer-overflow on address 0x1\n"
        "==41==ABORTING\n"
    )
    plan = poc_module.PocPlan(
        expected_stderr_patterns=["SEGV", "null dereference"],
        expected_crash_type="heap-buffer-overflow",
        expected_exit_code=None,
    )
    observation = {
        "observed_exit_code": 1,
        "observed_stdout": "",
        "observed_stderr": stderr,
        "observed_crash_type": "heap-buffer-overflow",
    }
    logs = (
        "execution_exit_code=1\nstdout_begin\nstdout_end\nstderr_begin\n"
        f"{stderr}stderr_end\n"
    )
    report = stage._build_run_verify_report(
        plan=plan,
        observation=observation,
        execution_logs=logs,
        matched_error_patterns=[],
        matched_stack_keywords=[],
    )
    assert report.eligible_for_verify is True
    assert report.crash_type_compatible is True
    assert report.eligibility_reason in {
        "specific_sanitizer_bug_in_haystack",
        "crash_type_compatible: observed=heap-buffer-overflow",
    }


def test_build_run_verify_report_ignores_generic_asan_when_specific_expected():
    stage = poc_module.PocStage()
    plan = poc_module.PocPlan(
        expected_stderr_patterns=["AddressSanitizer", "stack-buffer-overflow"],
        expected_crash_type="stack-buffer-overflow",
        expected_exit_code=None,
    )
    observation = {
        "observed_exit_code": 1,
        "observed_stdout": "",
        "observed_stderr": "AddressSanitizer: alloc-dealloc-mismatch (operator new [] vs operator delete)",
        "observed_crash_type": "abort",
    }
    logs = (
        "execution_exit_code=1\nstdout_begin\nstdout_end\nstderr_begin\n"
        "AddressSanitizer: alloc-dealloc-mismatch\nstderr_end\n"
    )
    report = stage._build_run_verify_report(
        plan=plan,
        observation=observation,
        execution_logs=logs,
        matched_error_patterns=["AddressSanitizer"],
        matched_stack_keywords=[],
    )
    assert report.error_pattern_hits == ["AddressSanitizer"]
    assert report.eligible_for_verify is False
    assert report.eligibility_reason == "no_target_behavior_observed"


def test_poc_run_template_skips_outer_preload_when_run_command_scopes_asan():
    stage = poc_module.PocStage()
    with_inner = stage._render_template(
        "poc_run.sh.j2",
        {
            "poc_artifacts_dir": "/workspace/artifacts/poc",
            "execution_dir": "/src/proj",
            "target_binary": "/src/proj/plugin.so",
            "target_binary_echo": "/src/proj/plugin.so",
            "run_command": "export LD_PRELOAD=/usr/lib/libasan.so; /tmp/harness",
            "run_command_echo": "export LD_PRELOAD=...; /tmp/harness",
            "run_command_shell": "'export LD_PRELOAD=/usr/lib/libasan.so; /tmp/harness'",
            "outer_asan_preload": False,
            "trigger_timeout_sec": 120,
        },
    )
    assert "timeout 120s bash -c" in with_inner
    # Outer subshell must not export LD_PRELOAD again before bash -c.
    trigger_block = with_inner.split("stderr_file=")[-1]
    assert 'export LD_PRELOAD="${_ASAN_RT}' not in trigger_block
    assert 'export LD_PRELOAD="$1' not in trigger_block

    without_inner = stage._render_template(
        "poc_run.sh.j2",
        {
            "poc_artifacts_dir": "/workspace/artifacts/poc",
            "execution_dir": "/src/proj",
            "target_binary": "/src/proj/bin",
            "target_binary_echo": "/src/proj/bin",
            "run_command": "/src/proj/bin payload",
            "run_command_echo": "/src/proj/bin payload",
            "run_command_shell": "'/src/proj/bin payload'",
            "outer_asan_preload": True,
            "trigger_timeout_sec": 120,
        },
    )
    assert "timeout 120s bash -c" in without_inner
    assert 'export LD_PRELOAD="$1' in without_inner
    assert without_inner.index("timeout 120s bash -c") < without_inner.index(
        'export LD_PRELOAD="$1'
    )
    assert '"${_ASAN_RT}"' in without_inner.split("stderr_file=")[-1]
    trigger = without_inner.split("stderr_file=")[-1]
    assert "export LD_PRELOAD=" not in trigger.split("timeout", 1)[0]


def test_poc_run_template_clears_asan_rt_when_target_already_links_asan():
    """Gate: DT_NEEDED libclang_rt.asan → skip redundant outer LD_PRELOAD."""
    stage = poc_module.PocStage()
    rendered = stage._render_template(
        "poc_run.sh.j2",
        {
            "poc_artifacts_dir": "/workspace/artifacts/poc",
            "execution_dir": "/src/lua",
            "target_binary": "/src/lua/src/lua",
            "target_binary_echo": "/src/lua/src/lua",
            "run_command": "/src/lua/src/lua /workspace/artifacts/poc/payloads/poc.lua",
            "run_command_echo": "/src/lua/src/lua poc.lua",
            "run_command_shell": "'/src/lua/src/lua /workspace/artifacts/poc/payloads/poc.lua'",
            "outer_asan_preload": True,
            "trigger_timeout_sec": 120,
        },
    )
    assert "libclang_rt\\.asan" in rendered
    assert 'ldd "/src/lua/src/lua"' in rendered
    assert rendered.index('ldd "/src/lua/src/lua"') < rendered.index("timeout 120s bash -c")
    # Still pass _ASAN_RT into bash -c; empty value makes preload a no-op.
    assert '"${_ASAN_RT}"' in rendered.split("stderr_file=")[-1]


def test_should_outer_asan_preload_skips_library_harness_without_inner_preload():
    from app.tools.log_parsing import should_outer_asan_preload

    assert should_outer_asan_preload("/src/bin payload", "cli-file") is True
    assert should_outer_asan_preload("export LD_PRELOAD=/lib/asan.so; /tmp/h", "cli-file") is False
    assert should_outer_asan_preload("clang++ -fsanitize=address ... && timeout 90s /tmp/qimage_harness", "") is False
    assert should_outer_asan_preload("clang++ -fsanitize=address -shared-libasan -o /tmp/qimage_harness x.cpp && timeout 90s /tmp/qimage_harness p", "library-harness") is False


def test_poc_run_template_skips_outer_preload_for_sanitized_qimage_harness():
    stage = poc_module.PocStage()
    plan = poc_module.PocPlan(
        target_binary="/src/kimageformats/build/bin/imageformats/kimg_xcf.so",
        payload_filename="crafted_bpp5.xcf",
        payload_content="xcf\n",
    )
    normalized = stage._normalize_poc_plan(
        plan,
        repo_url="https://invent.kde.org/frameworks/kimageformats.git",
    )
    from app.tools.log_parsing import should_outer_asan_preload

    assert normalized.trigger_mode == "library-harness"
    assert "timeout 90s env LD_PRELOAD=" in normalized.run_command
    assert "export LD_PRELOAD" not in normalized.run_command
    assert should_outer_asan_preload(normalized.run_command, normalized.trigger_mode) is False

    rendered = stage._render_template(
        "poc_run.sh.j2",
        {
            "poc_artifacts_dir": "/workspace/artifacts/poc",
            "execution_dir": "/src/kimageformats",
            "target_binary": normalized.target_binary,
            "target_binary_echo": normalized.target_binary,
            "run_command": normalized.run_command,
            "run_command_echo": "qimage_harness",
            "run_command_shell": "'qimage_harness'",
            "outer_asan_preload": should_outer_asan_preload(
                normalized.run_command, normalized.trigger_mode
            ),
            "trigger_timeout_sec": 120,
        },
    )
    trigger_block = rendered.split("stderr_file=")[-1]
    assert 'export LD_PRELOAD="${_ASAN_RT}' not in trigger_block
    assert "-fsanitize=address" not in normalized.run_command
    assert "-shared-libasan" not in normalized.run_command
    assert "timeout 90s env LD_PRELOAD=" in normalized.run_command


def test_normalize_poc_plan_leaves_executable_targets_unchanged():
    stage = poc_module.PocStage()
    plan = poc_module.PocPlan(
        target_binary="/src/lua/lua",
        payload_filename="poc.lua",
        payload_content="print(1)\n",
        run_command="/src/lua/lua /workspace/artifacts/poc/payloads/poc.lua",
    )

    normalized = stage._normalize_poc_plan(plan, repo_url="https://github.com/lua/lua.git")

    assert normalized.target_binary == "/src/lua/lua"
    assert normalized.trigger_mode != "library-harness"
    assert "qimage_harness" not in normalized.run_command


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


def test_normalize_poc_plan_fixes_nested_src_binary_and_decodes_base64_payload():
    stage = poc_module.PocStage()
    # base64 of: local x=1\nfunction f()\nreturn x\nend\n
    encoded = "bG9jYWwgeD0xCmZ1bmN0aW9uIGYoKQpyZXR1cm4geAplbmQK"
    plan = poc_module.PocPlan(
        target_binary="/src/lua/src/lua",
        payload_filename="poc.lua",
        payload_content=encoded,
        run_command="cd /src/lua && ./src/lua {payload}",
    )

    normalized = stage._normalize_poc_plan(plan, repo_url="https://github.com/lua/lua.git")

    assert normalized.target_binary == "/src/lua/lua"
    assert "function" in normalized.payload_content
    assert "bG9jYWw" not in normalized.payload_content
    assert normalized.run_command == (
        "cd /src/lua && /src/lua/lua /workspace/artifacts/poc/payloads/poc.lua"
    )


def test_normalize_poc_plan_restores_form_feed_from_recipe_base64():
    stage = poc_module.PocStage()
    # Contains a form-feed (0x0c) before the final `c""`.
    encoded = (
        "bG9jYWwgdSxfLE4sXyx3LE4sZCxXCmZ1bmN0aW9uIGMoRSxMLGwsUyxULHUsTSxULGwsaCxoLHUsdSxsLGgsaCx1"
        "LHUsTSx1LHUsdSxsLGgsaCxsKXM9cyBsb2NhbAllLGUsXyxfLE4sZSxzMCxOLFYsXyBmdW5jdGlvbiBjKGIsbClp"
        "LHM9TiBsb2NhbCBjIGxvY2FsIF9FTlY8Y29uc3Q+ID0wIG89MCBmdW5jdGlvbiBlKCllbmQ7ZSIicmV0dXJuIGVu"
        "ZDtlMCxhLHcscyxzLHM9IiJyZXR1cm4jYyIiZW5kO2VlPSIicmV0dXJuDGMiIg=="
    )
    mangled = (
        'local u,_,N,_,w,N,d,W\nfunction c(E,L,l,S,T,u,M,T,l,h,h,u,u,l,h,h,u,u,M,u,u,u,l,h,h,l)'
        's=s local e,e,_,_,N,e,s0,N,V,_ function c(b,l)i,s=N local c local _ENV<const> =0 o=0 '
        'function e()end;e""return end;e0,a,w,s,s,s=""return#c""end;ee=""return\\c""'
    )
    plan = poc_module.PocPlan(
        target_binary="/src/lua/lua",
        payload_filename="poc.lua",
        payload_content=mangled,
        run_command="/src/lua/lua {payload}",
    )

    normalized = stage._normalize_poc_plan(
        plan,
        repo_url="https://github.com/lua/lua.git",
        recipe_base64_blobs=[encoded],
    )

    assert "\x0c" in normalized.payload_content
    assert r"\c" not in normalized.payload_content.replace("\x0c", "")


def test_extract_recipe_base64_blobs_from_artifact_commands():
    stage = poc_module.PocStage()
    encoded = "bG9jYWwgeD0xCmZ1bmN0aW9uIGYoKQpyZXR1cm4geAplbmQK"
    recipes = [
        ReproductionRecipe(
            artifact_generation_commands=[f'echo -n "{encoded}" | base64 -d > poc'],
            steps=[f'echo -n "{encoded}" | base64 -d > poc'],
        )
    ]

    blobs = stage._extract_recipe_base64_blobs(recipes)

    assert blobs == [encoded]


def test_unwrap_nested_base64_payload_decodes_script_stored_as_base64():
    import base64

    stage = poc_module.PocStage()
    # Same shape as CVE-2022-28805 vuln_pocs/poc.lua: on-disk file is base64 text.
    lua = (
        "local u,_,N,_,w,N,d,W\n"
        "function c() local _ENV<const> =0 end\n"
        'return\x0cc""\n'
    )
    nested = base64.b64encode(lua.encode("latin-1")).decode("ascii")
    unwrapped = stage._unwrap_nested_base64_payload(nested)
    assert "local " in unwrapped
    assert "_ENV<const>" in unwrapped
    assert "\x0c" in unwrapped
    assert not unwrapped.strip().startswith("bG9j")


def test_unwrap_nested_base64_payload_leaves_binary_seed_alone():
    stage = poc_module.PocStage()
    binary = "\x00\x01clusterfuzz-seed\xff\xfe"
    assert stage._unwrap_nested_base64_payload(binary) == binary


def test_apply_authoritative_dataset_poc_unwraps_nested_base64_script():
    import base64

    stage = poc_module.PocStage()
    lua = (
        "local u,_,N,_,w,N,d,W\n"
        "function c() local _ENV<const> =0 end\n"
        'return\x0cc""\n'
    )
    # Transport blob encodes the on-disk file (itself still base64 of lua).
    on_disk = base64.b64encode(lua.encode("latin-1")).decode("ascii")
    transport = base64.b64encode(on_disk.encode("ascii")).decode("ascii")
    plan = poc_module.PocPlan(
        target_binary="/src/lua/lua",
        payload_filename="poc.lua",
        payload_content="trigger\n",
        run_command="/src/lua/lua /workspace/artifacts/poc/payloads/poc.lua",
    )
    context = poc_module.PocContext(
        cve_id="CVE-2022-28805",
        chosen_strategy="dataset_poc",
        dataset_poc_filenames=["poc.lua"],
        dataset_poc_base64_blobs=[transport],
    )

    updated = stage._apply_authoritative_payload(plan, context)

    assert "local " in updated.payload_content
    assert "_ENV<const>" in updated.payload_content
    assert "\x0c" in updated.payload_content
    assert not updated.payload_content.lstrip().startswith("bG9j")


def test_apply_authoritative_dataset_poc_keeps_binary_clusterfuzz_bytes():
    import base64

    stage = poc_module.PocStage()
    seed = b"\x00\x01\x02minimized-seed\xff"
    transport = base64.b64encode(seed).decode("ascii")
    plan = poc_module.PocPlan(
        target_binary="/src/matio/matio_fuzzer",
        payload_filename="clusterfuzz-testcase-minimized-matio_fuzzer-1",
        payload_content="trigger\n",
        run_command="/src/matio/matio_fuzzer /workspace/artifacts/poc/payloads/x",
    )
    context = poc_module.PocContext(
        cve_id="CVE-2021-36977",
        chosen_strategy="dataset_poc",
        dataset_poc_filenames=["clusterfuzz-testcase-minimized-matio_fuzzer-1"],
        dataset_poc_base64_blobs=[transport],
    )

    updated = stage._apply_authoritative_payload(plan, context)

    assert updated.payload_content.encode("latin-1", errors="replace") == seed


def test_execute_poc_plan_writes_payload_as_latin1(tmp_path):
    stage = poc_module.PocStage()
    paths = poc_module.PocStagePaths(str(tmp_path / "ws"))
    stage._prepare_workspace(paths)
    plan = poc_module.PocPlan(
        target_binary="/src/lua/lua",
        payload_filename="poc.lua",
        payload_content='return\x0cc""\n',
        run_command="/src/lua/lua {payload}",
    )

    # Avoid docker: only exercise the payload write portion.
    payload_path = paths.payloads_dir / plan.payload_filename
    stage.file_tool.write_latin1(str(payload_path), plan.payload_content)

    raw = payload_path.read_bytes()
    assert b"\x0c" in raw
    assert raw == b'return\x0cc""\n'


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


def test_select_target_args_forces_file_argv_for_ossfuzz_harness():
    stage = poc_module.PocStage()
    knowledge = make_knowledge()
    context = poc_module.PocContext(
        cve_id="CVE-2021-36978",
        inferred_input_modes=["stdin", "file"],
    )
    payload = "clusterfuzz-testcase-minimized-qpdf_fuzzer-5162370603286528.cil"
    args = stage._select_target_args(
        knowledge,
        payload,
        context,
        target_binary="/src/qpdf/fuzz/build/qpdf_fuzzer",
    )
    assert args == [f"/workspace/artifacts/poc/payloads/{payload}"]
    assert not any(a.strip().startswith("<") for a in args)


def test_normalize_poc_plan_coerces_quoted_stdin_redirect_for_fuzzer():
    stage = poc_module.PocStage()
    payload = "clusterfuzz-testcase-minimized-qpdf_fuzzer-1.cil"
    plan = poc_module.PocPlan(
        trigger_mode="cli-stdin",
        target_binary="/src/qpdf/fuzz/build/qpdf_fuzzer",
        target_args=[f"< /workspace/artifacts/poc/payloads/{payload}"],
        payload_filename=payload,
        payload_content="fuzz",
        run_command=(
            f"'/src/qpdf/fuzz/build/qpdf_fuzzer' "
            f"'< /workspace/artifacts/poc/payloads/{payload}'"
        ),
    )
    normalized = stage._normalize_poc_plan(plan, repo_url="https://github.com/qpdf/qpdf.git")
    assert normalized.trigger_mode == "cli-file"
    assert normalized.target_args == [f"/workspace/artifacts/poc/payloads/{payload}"]
    assert "'<" not in normalized.run_command
    assert f"/workspace/artifacts/poc/payloads/{payload}" in normalized.run_command
    assert normalized.run_command.startswith("'/src/qpdf/fuzz/build/qpdf_fuzzer'")


def test_build_run_command_keeps_stdin_redirect_unquoted():
    stage = poc_module.PocStage()
    command = stage._build_run_command(
        "/src/tool/bin",
        ["< /workspace/artifacts/poc/payloads/poc.bin"],
    )
    assert command == "'/src/tool/bin' < '/workspace/artifacts/poc/payloads/poc.bin'"
    assert "'<" not in command
    assert " < " in command
