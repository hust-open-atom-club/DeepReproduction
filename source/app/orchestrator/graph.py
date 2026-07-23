"""LangGraph v1 workflow assembly for the DeepReproduction multi-agent system."""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.orchestrator.nodes import finalize_node, review_node
from app.orchestrator.routers import (
    route_after_build,
    route_after_knowledge,
    route_after_poc,
    route_after_review,
    route_after_verify,
)
from app.orchestrator.state import AppState
from app.stages.build import build_node
from app.stages.knowledge import knowledge_node
from app.stages.poc import poc_node
from app.stages.verify import verify_node


def build_app_graph(checkpointer=None):
    """Construct and compile the LangGraph v1 workflow."""

    builder = StateGraph(AppState)

    builder.add_node("knowledge", knowledge_node)
    builder.add_node("build", build_node)
    builder.add_node("poc", poc_node)
    builder.add_node("verify", verify_node)
    builder.add_node("review", review_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "knowledge")

    builder.add_conditional_edges(
        "knowledge",
        route_after_knowledge,
        {
            "build": "build",
            "review": "review",
            "finalize": "finalize",
        },
    )
    builder.add_conditional_edges(
        "build",
        route_after_build,
        {
            "poc": "poc",
            "build": "build",
            "review": "review",
        },
    )
    builder.add_conditional_edges(
        "poc",
        route_after_poc,
        {
            "verify": "verify",
            "poc": "poc",
            "review": "review",
        },
    )
    builder.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "success": "finalize",
            "failed": "review",
            "inconclusive": "review",
        },
    )
    builder.add_conditional_edges(
        "review",
        route_after_review,
        {
            "knowledge": "knowledge",
            "build": "build",
            "poc": "poc",
            "verify": "verify",
            "finalize": "finalize",
        },
    )

    builder.add_edge("finalize", END)

    resolved_checkpointer = checkpointer or InMemorySaver()
    return builder.compile(checkpointer=resolved_checkpointer)
