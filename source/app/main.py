"""Project entrypoint for the LangGraph v1 multi-agent workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from langgraph.types import Command


def bootstrap_import_path() -> Path:
    """Ensure the source directory is importable when running this file directly."""

    source_root = Path(__file__).resolve().parents[1]
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    return source_root


bootstrap_import_path()

from app.orchestrator.graph import build_app_graph
from app.schemas.task import TaskModel


def load_task(task_path: str) -> TaskModel:
    """Load a task YAML into the shared task schema."""

    with open(task_path, "r", encoding="utf-8") as file:
        task_data = yaml.safe_load(file)

    return TaskModel(**task_data)


def build_initial_state(
    task: TaskModel,
    dataset_root: str = "Dataset",
    workspace_root: str = "workspaces",
    thread_id: str | None = None,
) -> dict:
    """Construct the initial LangGraph state."""

    resolved_thread_id = thread_id or task.task_id
    workspace = str(Path(workspace_root) / task.task_id)
    return {
        "task": task,
        "run_id": task.task_id,
        "thread_id": resolved_thread_id,
        "dataset_root": dataset_root,
        "workspace_root": workspace_root,
        "workspace": workspace,
        "retry_count": {},
        "stage_history": [],
        "stage_status": {},
        "artifacts": {},
        "current_stage": "knowledge",
        "review_stage": "",
        "human_action_required": False,
        "review_reason": "",
        "review_payload": {},
        "review_decision": {},
        "final_status": "running",
        "last_error": None,
    }


def build_graph_config(thread_id: str) -> dict:
    """Build LangGraph runtime config using a stable thread id."""

    return {"configurable": {"thread_id": thread_id}}


def invoke_workflow(
    task_path: str,
    dataset_root: str = "Dataset",
    workspace_root: str = "workspaces",
    thread_id: str | None = None,
    checkpointer=None,
):
    """Start a new workflow execution from a task file."""

    task = load_task(task_path)
    resolved_thread_id = thread_id or task.task_id
    graph = build_app_graph(checkpointer=checkpointer)
    initial_state = build_initial_state(
        task=task,
        dataset_root=dataset_root,
        workspace_root=workspace_root,
        thread_id=resolved_thread_id,
    )
    result = graph.invoke(initial_state, config=build_graph_config(resolved_thread_id))
    return graph, result


def resume_workflow(thread_id: str, resume_value: Any, checkpointer=None):
    """Resume an interrupted workflow with a human review decision."""

    graph = build_app_graph(checkpointer=checkpointer)
    result = graph.invoke(Command(resume=resume_value), config=build_graph_config(thread_id))
    return graph, result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or resume the DeepReproduction LangGraph workflow.")
    parser.add_argument("--task", help="Task YAML path used to start a new workflow.")
    parser.add_argument("--dataset-root", default="Dataset", help="Dataset root directory.")
    parser.add_argument("--workspace-root", default="workspaces", help="Workspace root directory.")
    parser.add_argument("--thread-id", help="Stable LangGraph thread identifier.")
    parser.add_argument(
        "--resume-json",
        help="Resume an interrupted workflow with a JSON payload, for example '{\"action\": \"retry\"}'.",
    )
    return parser


def main():
    """CLI entrypoint for local workflow execution."""

    parser = _build_parser()
    args = parser.parse_args()

    if args.resume_json:
        if not args.thread_id:
            parser.error("--thread-id is required when using --resume-json")
        _, result = resume_workflow(args.thread_id, json.loads(args.resume_json))
        print(result)
        return

    if not args.task:
        parser.error("--task is required unless --resume-json is used")

    _, result = invoke_workflow(
        task_path=args.task,
        dataset_root=args.dataset_root,
        workspace_root=args.workspace_root,
        thread_id=args.thread_id,
    )
    print(result)


if __name__ == "__main__":
    main()
