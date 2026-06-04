"""Router tests for the LangGraph v1 workflow."""

from app.orchestrator import routers


def test_route_after_build_goes_to_poc_on_success():
    state = {
        "build": type("Build", (), {"build_success": True})(),
        "retry_count": {"build": 1},
    }
    assert routers.route_after_build(state) == "poc"


def test_route_after_build_retries_before_review():
    state = {
        "build": type("Build", (), {"build_success": False})(),
        "retry_count": {"build": 1},
    }
    assert routers.route_after_build(state) == "build"


def test_route_after_build_requests_review_after_max_retries():
    state = {
        "build": type("Build", (), {"build_success": False})(),
        "retry_count": {"build": routers.MAX_BUILD_RETRY},
        "review_stage": "build",
    }
    assert routers.route_after_build(state) == "review"


def test_route_after_poc_requests_review_after_max_retries():
    state = {
        "poc": type("PoC", (), {"execution_success": False})(),
        "retry_count": {"poc": routers.MAX_POC_RETRY},
        "review_stage": "poc",
    }
    assert routers.route_after_poc(state) == "review"


def test_route_after_verify_success_finalizes():
    state = {
        "verify": type("Verify", (), {"verdict": "success"})(),
    }
    assert routers.route_after_verify(state) == "success"


def test_route_after_verify_failed_requests_review():
    state = {
        "verify": type("Verify", (), {"verdict": "failed"})(),
        "review_stage": "verify",
    }
    assert routers.route_after_verify(state) == "failed"


def test_route_after_review_maps_retry_to_original_stage():
    state = {
        "review_stage": "poc",
        "review_decision": {"action": "retry"},
    }
    assert routers.route_after_review(state) == "poc"


def test_route_after_review_maps_continue_to_next_stage():
    state = {
        "review_stage": "build",
        "review_decision": {"action": "continue"},
    }
    assert routers.route_after_review(state) == "poc"


def test_route_after_review_maps_abort_to_finalize():
    state = {
        "review_stage": "verify",
        "review_decision": {"action": "abort"},
    }
    assert routers.route_after_review(state) == "finalize"
