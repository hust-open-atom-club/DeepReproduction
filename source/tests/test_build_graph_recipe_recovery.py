"""文件说明：build 内部子图 tier-1 配方恢复接线的回归测试。

背景：recipe 恢复原本只接在 plan_and_execute_build（standalone 路径），
而 run() 真实走的是内部 LangGraph 子图 —— 配方在真实执行链上是死代码
（CVE-2022-28044 lrzip libtoolize 冲突 4 连败实证）。本测试锁定：
失败后先查配方表，命中即携补丁脚本重执行，不触发 LLM replan。
"""

from __future__ import annotations

from pathlib import Path

from app.schemas.build_artifact import BuildArtifact
from app.stages.build import BuildExecutionOutcome, BuildPlan, BuildPreparedRun, BuildStage, BuildStagePaths


LIBTOOLIZE_LOG = (
    "[build] running configure step\n"
    "libtoolize:   error: AC_CONFIG_MACRO_DIRS([m4]) conflicts with "
    "ACLOCAL_AMFLAGS=-I m4\n"
    "autoreconf: libtoolize failed with exit status: 1\n"
)

SCRIPT_PLAIN = """#!/bin/bash
set -euo pipefail
log "running configure step"
autoreconf -i -f
./configure
make -j$(nproc)
"""


def _make_fixtures(tmp_path: Path):
    plan = BuildPlan.model_construct(
        build_system="autotools",
        chosen_vulnerable_ref="abc",
        chosen_fixed_ref="def",
        rationale="initial plan",
        build_script_override=None,
    )
    artifact = BuildArtifact.model_construct(
        build_success=False,
        build_logs=LIBTOOLIZE_LOG,
        build_script_content=SCRIPT_PLAIN,
    )
    outcome = BuildExecutionOutcome(plan=plan, artifact=artifact)
    prepared = BuildPreparedRun.model_construct(
        plan_meta={"project_name": "lrzip"},
        repo_path=str(tmp_path),
        context=None,
    )
    return plan, prepared, outcome


def test_recipe_retry_preempts_llm_replan(monkeypatch, tmp_path: Path) -> None:
    plan, prepared, outcome = _make_fixtures(tmp_path)
    stage = object.__new__(BuildStage)

    monkeypatch.setattr(stage, "_normalize_build_plan", lambda repo_path, plan, knowledge: plan)
    monkeypatch.setattr(stage, "_write_yaml_file", lambda path, payload: None)
    monkeypatch.setattr(
        stage,
        "execute_build_attempt",
        lambda repo_path, paths, plan_meta, build_plan: outcome,
    )

    def _must_not_replan(**kwargs):
        raise AssertionError("recipe hit must not consume LLM replan")

    monkeypatch.setattr(stage, "_replan_from_failed_attempt", _must_not_replan)

    state: dict = {
        "prepared": prepared,
        "current_plan": plan,
        "current_context": None,
        "knowledge": None,
        "paths": BuildStagePaths(str(tmp_path)),
        "attempt": 0,
        "recipe_attempts": 0,
    }

    updates = stage._build_graph_execute_node(state)

    assert updates["recipe_retry_pending"] is True
    assert updates["recipe_attempts"] == 1
    assert updates["should_retry"] is True
    override = updates["current_plan"].build_script_override or ""
    assert "s/^ACLOCAL_AMFLAGS/#ACLOCAL_AMFLAGS/" in override
    assert "recipe-recovery: libtoolize_macro_conflict" in updates["current_plan"].rationale

    route_state = dict(state)
    route_state.update(updates)
    assert stage._route_after_build_execute(route_state) == "execute"


def test_recipe_budget_exhausted_falls_back_to_replan(monkeypatch, tmp_path: Path) -> None:
    plan, prepared, outcome = _make_fixtures(tmp_path)
    stage = object.__new__(BuildStage)

    monkeypatch.setattr(stage, "_normalize_build_plan", lambda repo_path, plan, knowledge: plan)
    monkeypatch.setattr(stage, "_write_yaml_file", lambda path, payload: None)
    monkeypatch.setattr(
        stage,
        "execute_build_attempt",
        lambda repo_path, paths, plan_meta, build_plan: outcome,
    )
    monkeypatch.setattr(stage, "_replan_from_failed_attempt", lambda **kwargs: (None, None))

    state: dict = {
        "prepared": prepared,
        "current_plan": plan,
        "current_context": None,
        "knowledge": None,
        "paths": BuildStagePaths(str(tmp_path)),
        "attempt": 0,
        "recipe_attempts": stage.MAX_RECIPE_ATTEMPTS,  # 预算耗尽
    }

    updates = stage._build_graph_execute_node(state)

    assert not updates.get("recipe_retry_pending", False)
    assert updates["should_retry"] is False  # replan 返回 None → 不再重试
