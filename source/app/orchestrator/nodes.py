"""LangGraph v1 orchestration helpers for review and finalize nodes."""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt


def _normalize_review_action(decision: Any, default_action: str = "abort") -> dict:
    """Normalize human review input into a stable action payload."""

    if isinstance(decision, bool):
        return {"action": "retry" if decision else "abort"}

    if isinstance(decision, str):
        action = decision.strip().lower()
        if action in {"retry", "continue", "abort"}:
            return {"action": action}
        return {"action": default_action, "raw": decision}

    if isinstance(decision, dict):
        action = str(decision.get("action", default_action)).strip().lower()
        if action not in {"retry", "continue", "abort"}:
            action = default_action
        normalized = dict(decision)
        normalized["action"] = action
        return normalized

    return {"action": default_action, "raw": decision}


def review_node(state):
    """Pause the workflow and request a human decision."""

    review_stage = state.get("review_stage") or state.get("current_stage") or "unknown"
    payload = {
        "kind": "human_review",
        "stage": review_stage,
        "reason": state.get("review_reason") or state.get("last_error") or "manual review requested",
        "error": state.get("last_error"),
        "final_status": state.get("final_status", "running"),
        "artifacts": state.get("artifacts", {}),
        "stage_history": state.get("stage_history", []),
        "suggested_actions": ["retry", "continue", "abort"],
    }
    decision = interrupt(payload)
    normalized = _normalize_review_action(decision, default_action="abort")

    history = list(state.get("stage_history", []))
    history.append(
        {
            "stage": "review",
            "status": "resolved",
            "review_stage": review_stage,
            "decision": normalized.get("action", "abort"),
        }
    )

    return {
        "review_payload": payload,
        "review_decision": normalized,
        "human_action_required": False,
        "stage_history": history,
        "last_error": None if normalized.get("action") != "abort" else payload["reason"],
    }


def finalize_node(state):
    """Unify final status at the end of the graph."""

    verify = state.get("verify")
    review_decision = state.get("review_decision", {})
    final_status = state.get("final_status", "running")

    if review_decision.get("action") == "abort":
        final_status = "aborted"
    elif verify is not None:
        if verify.verdict == "success":
            final_status = "success"
        elif verify.verdict == "inconclusive":
            final_status = "inconclusive"
        else:
            final_status = "failed"
    elif state.get("knowledge") is None:
        final_status = "failed"
    elif state.get("build") is not None and not state["build"].build_success:
        final_status = "failed"
    elif state.get("poc") is not None and not state["poc"].execution_success:
        final_status = "failed"

    return {
        "final_status": final_status,
        "current_stage": "finalize",
    }
