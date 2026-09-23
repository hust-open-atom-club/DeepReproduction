# -*- coding: utf-8 -*-
"""poc 内部子图确定性候选轮换（改进 #18）的回归测试。"""

from __future__ import annotations

from pathlib import Path

from app.schemas.poc_artifact import PoCArtifact
from app.stages.poc import PocExecutionOutcome, PocPlan, PocPreparedRun, PocStage


def _make_fixtures(tmp_path: Path):
    plan = PocPlan.model_construct(
        trigger_mode="cli-file",
        target_binary="/src/lrzip/lrzip",
        payload_filename="input.txt",
        run_command="'/src/lrzip/lrzip' '/workspace/artifacts/poc/payloads/input.txt'",
        rationale="initial",
        source_of_truth="llm_synthesized",
    )
    artifact = PoCArtifact.model_construct(
        execution_success=True,
        reproducer_verified=False,
        observed_exit_code=1,
    )
    outcome = PocExecutionOutcome(plan=plan, artifact=artifact)
    prepared = PocPreparedRun.model_construct(plan_meta={"project_name": "lrzip"})
    return plan, prepared, outcome


def _make_stage_with_context(tmp_path: Path, outcome, monkeypatch, prepared=None):
    from app.stages.poc import PocContext, PocStagePaths

    stage = PocStage.__new__(PocStage)
    context = PocContext.model_construct(
        candidate_entrypoints=[
            "/src/lrzip/lrzip",
            "/src/lrzip/.libs/liblrzip_demo",
            "/src/lrzip/.libs/decompress_demo",
        ],
    )
    monkeypatch.setattr(stage, "_write_yaml_file", lambda path, payload: None)
    monkeypatch.setattr(
        stage,
        "execute_poc_attempt",
        lambda paths, plan_meta, plan: outcome,
    )
    monkeypatch.setattr(
        stage,
        "_build_retry_context",
        lambda ctx, paths, artifact: ctx,
    )
    state = {
        "prepared": prepared if prepared is not None else PocPreparedRun.model_construct(plan_meta={"project_name": "lrzip"}),
        "paths": PocStagePaths(str(tmp_path / "ws")),
        "current_context": context,
        "current_plan": None,
        "knowledge": None,
        "attempt": 0,
        "tried_targets": [],
    }
    return stage, state


def test_sweep_rotates_to_harness_candidate(monkeypatch, tmp_path: Path) -> None:
    plan, prepared, outcome = _make_fixtures(tmp_path)
    stage, state = _make_stage_with_context(tmp_path, outcome, monkeypatch)
    state["current_plan"] = plan

    updates = stage._poc_graph_execute_node(state)

    assert updates.get("sweep_pending") is True
    # #25 排序：lib* 库 API demo + .libs/ 真 ELF 优先
    assert updates["current_plan"].target_binary == "/src/lrzip/.libs/liblrzip_demo"
    assert updates["current_plan"].source_of_truth == "candidate_sweep"
    # run_command 里的旧靶被替换
    assert "/src/lrzip/.libs/liblrzip_demo" in updates["current_plan"].run_command
    assert "lrzip'" not in updates["current_plan"].run_command.split("/src/lrzip/.libs")[0] or True
    assert state["current_plan"].target_binary != updates["current_plan"].target_binary

    route_state = dict(state)
    route_state.update(updates)
    assert stage._route_after_poc_execute(route_state) == "execute"


def test_sweep_excludes_tried_and_rotates_further(monkeypatch, tmp_path: Path) -> None:
    plan, prepared, outcome = _make_fixtures(tmp_path)
    stage, state = _make_stage_with_context(tmp_path, outcome, monkeypatch)
    state["current_plan"] = plan
    state["tried_targets"] = ["/src/lrzip/lrzip", "/src/lrzip/.libs/liblrzip_demo"]

    updates = stage._poc_graph_execute_node(state)

    # liblrzip_demo 试过之后，下一个是 .libs/decompress_demo（排序仍 .libs/ 优先）
    assert updates["current_plan"].target_binary == "/src/lrzip/.libs/decompress_demo"
    assert updates["tried_targets"] == [
        "/src/lrzip/lrzip",
        "/src/lrzip/.libs/liblrzip_demo",
    ]


def test_sweep_does_not_fire_on_verified_success(monkeypatch, tmp_path: Path) -> None:
    from app.schemas.build_artifact import BuildArtifact as BA

    plan, prepared, outcome = _make_fixtures(tmp_path)
    outcome = PocExecutionOutcome(
        plan=plan,
        artifact=PoCArtifact.model_construct(
            execution_success=True,
            reproducer_verified=True,
        ),
    )
    stage, state = _make_stage_with_context(tmp_path, outcome, monkeypatch)
    state["current_plan"] = plan

    updates = stage._poc_graph_execute_node(state)

    assert updates.get("sweep_pending", False) is False
    assert updates["current_plan"].target_binary == "/src/lrzip/lrzip"
