"""End-to-end orchestration tests for the LangGraph v1 workflow."""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.main import build_graph_config, build_initial_state
from app.orchestrator.graph import build_app_graph
from app.schemas.build_artifact import BuildArtifact
from app.schemas.knowledge import KnowledgeModel
from app.schemas.poc_artifact import PoCArtifact
from app.schemas.task import TaskModel
from app.schemas.verify_result import VerifyResult


def _make_task():
    return TaskModel(
        task_id="CVE-2022-0000",
        cve_id="CVE-2022-0000",
        repo_url="https://example.com/demo.git",
    )


def test_build_initial_state_populates_langgraph_runtime_fields():
    task = _make_task()
    state = build_initial_state(
        task,
        dataset_root="Dataset",
        workspace_root="workspaces",
        thread_id="thread-demo",
    )

    assert state["thread_id"] == "thread-demo"
    assert state["dataset_root"] == "Dataset"
    assert state["workspace"] == "workspaces/CVE-2022-0000"
    assert state["current_stage"] == "knowledge"
    assert state["final_status"] == "running"


def test_graph_interrupts_for_review_and_resumes(monkeypatch):
    def fake_knowledge_node(state):
        history = list(state.get("stage_history", []))
        history.append({"stage": "knowledge", "status": "success"})
        return {
            "knowledge": KnowledgeModel(
                cve_id="CVE-2022-0000",
                summary="demo",
                vulnerability_type="heap-overflow",
                repo_url="https://example.com/demo.git",
                vulnerable_ref="deadbeef",
            ),
            "current_stage": "build",
            "stage_history": history,
            "stage_status": {"knowledge": "success"},
            "artifacts": {"knowledge": {"knowledge_yaml": "Dataset/demo/knowledge.yaml"}},
            "last_error": None,
        }

    def fake_build_node(state):
        history = list(state.get("stage_history", []))
        history.append({"stage": "build", "status": "success"})
        return {
            "build": BuildArtifact(
                dockerfile_content="FROM ubuntu:20.04\n",
                build_script_content="#!/bin/bash\nexit 0\n",
                build_success=True,
                build_logs="ok",
            ),
            "current_stage": "poc",
            "stage_history": history,
            "stage_status": {"build": "success"},
            "artifacts": {"build": {"build_artifact_yaml": "workspaces/demo/build_artifact.yaml"}},
            "last_error": None,
        }

    def fake_poc_node(state):
        history = list(state.get("stage_history", []))
        history.append({"stage": "poc", "status": "success"})
        return {
            "poc": PoCArtifact(
                poc_filename="poc.txt",
                poc_content="demo",
                run_script_content="#!/bin/bash\nexit 0\n",
                execution_success=True,
            ),
            "current_stage": "verify",
            "stage_history": history,
            "stage_status": {"poc": "success"},
            "artifacts": {"poc": {"poc_artifact_yaml": "workspaces/demo/poc_artifact.yaml"}},
            "last_error": None,
        }

    def fake_verify_node(state):
        history = list(state.get("stage_history", []))
        history.append({"stage": "verify", "status": "inconclusive"})
        return {
            "verify": VerifyResult(
                pre_patch_triggered=False,
                post_patch_clean=False,
                verdict="inconclusive",
                reason="need human review",
            ),
            "current_stage": "verify",
            "review_stage": "verify",
            "human_action_required": True,
            "review_reason": "need human review",
            "stage_history": history,
            "stage_status": {"verify": "inconclusive"},
            "artifacts": {"verify": {"verify_result_yaml": "workspaces/demo/verify_result.yaml"}},
            "final_status": "needs_review",
            "last_error": "need human review",
        }

    monkeypatch.setattr("app.orchestrator.graph.knowledge_node", fake_knowledge_node)
    monkeypatch.setattr("app.orchestrator.graph.build_node", fake_build_node)
    monkeypatch.setattr("app.orchestrator.graph.poc_node", fake_poc_node)
    monkeypatch.setattr("app.orchestrator.graph.verify_node", fake_verify_node)

    task = _make_task()
    graph = build_app_graph(checkpointer=InMemorySaver())
    config = build_graph_config("thread-review-demo")
    initial_state = build_initial_state(task, thread_id="thread-review-demo")

    interrupted = graph.invoke(initial_state, config=config)

    assert "__interrupt__" in interrupted
    assert interrupted["__interrupt__"][0].value["stage"] == "verify"
    assert interrupted["__interrupt__"][0].value["reason"] == "need human review"

    resumed = graph.invoke(Command(resume={"action": "abort"}), config=config)

    assert resumed["final_status"] == "aborted"
    assert resumed["review_decision"]["action"] == "abort"
