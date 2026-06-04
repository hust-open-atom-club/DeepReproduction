"""文件说明：阶段路由规则。

这里定义 LangGraph v1 主流程的控制逻辑：
- 正常推进：knowledge -> build -> poc -> verify -> finalize
- 自动重试：build / poc 在限定次数内自动重跑
- 人工介入：knowledge 失败、build/poc 达到重试上限、verify 非 success
- 最终收口：统一进入 finalize 节点产出 final_status
"""

MAX_BUILD_RETRY = 2
MAX_POC_RETRY = 2


def route_after_knowledge(state):
    """知识阶段后的路由。"""

    if state.get("knowledge") is not None:
        return "build"
    if state.get("human_action_required"):
        return "review"
    return "finalize"


def route_after_build(state):
    """构建阶段后的路由。"""

    if state.get("human_action_required"):
        return "review"

    build = state.get("build")
    retry_count = state.get("retry_count", {})

    if build and build.build_success:
        return "poc"

    if retry_count.get("build", 0) < MAX_BUILD_RETRY:
        return "build"

    return "review"


def route_after_poc(state):
    """PoC 阶段后的路由。"""

    if state.get("human_action_required"):
        return "review"

    poc = state.get("poc")
    retry_count = state.get("retry_count", {})

    if poc and poc.execution_success:
        return "verify"

    if retry_count.get("poc", 0) < MAX_POC_RETRY:
        return "poc"

    return "review"


def route_after_verify(state):
    """验证阶段后的路由。"""

    verify = state.get("verify")
    if verify is None:
        return "failed"
    if verify.verdict == "success":
        return "success"
    if verify.verdict == "inconclusive":
        return "inconclusive"
    return "failed"


def route_after_review(state):
    """人工 review 阶段后的路由。"""

    decision = state.get("review_decision", {})
    action = decision.get("action", "abort")
    review_stage = state.get("review_stage") or state.get("current_stage") or ""

    if action == "retry":
        if review_stage in {"knowledge", "build", "poc", "verify"}:
            return review_stage
        return "finalize"

    if action == "continue":
        if review_stage == "knowledge":
            return "build"
        if review_stage == "build":
            return "poc"
        if review_stage == "poc":
            return "verify"
        return "finalize"

    return "finalize"
