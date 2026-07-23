"""文件说明：流程共享状态定义。

`AppState` 是 LangGraph v1 在各阶段之间传递的唯一共享容器。
它既承载阶段输入输出，也承载 durable execution 所需的运行态信息，
确保流程可以 checkpoint、resume、人工介入与最终收口。
"""

from typing import Any, Dict, List, Optional, TypedDict

from app.schemas.build_artifact import BuildArtifact
from app.schemas.knowledge import KnowledgeModel
from app.schemas.poc_artifact import PoCArtifact
from app.schemas.task import TaskModel
from app.schemas.verify_result import VerifyResult


class AppState(TypedDict, total=False):
    """LangGraph 主流程共享状态。"""

    task: TaskModel
    run_id: str
    thread_id: str
    dataset_root: str
    workspace_root: str
    workspace: str

    knowledge: KnowledgeModel
    build: BuildArtifact
    poc: PoCArtifact
    verify: VerifyResult

    current_stage: str
    review_stage: str
    retry_count: Dict[str, int]
    stage_history: List[dict]
    stage_status: Dict[str, str]
    artifacts: Dict[str, Dict[str, str]]
    last_error: Optional[str]
    human_action_required: bool
    review_reason: str
    review_payload: Dict[str, Any]
    review_decision: Dict[str, Any]
    final_status: str
