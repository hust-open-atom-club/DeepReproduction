"""文件说明：最后成功构建快照的读写（改进 #24）。

build 成功时固化 last_good_build.yaml；poc/verify 在当前 build_artifact
被后续失败重跑覆盖（build_success=false）时回读快照，恢复真实的
编译镜像与脚本元数据。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from app.schemas.build_artifact import BuildArtifact


def load_last_good_build(build_dir: Path) -> Optional[BuildArtifact]:
    """读取 last_good_build.yaml；不存在或损坏时返回 None。"""

    path = Path(build_dir) / "last_good_build.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace")) or {}
    except OSError:
        return None
    artifact_payload = payload.get("artifact") or {}
    if not isinstance(artifact_payload, dict):
        return None
    try:
        artifact = BuildArtifact(**artifact_payload)
    except Exception:
        return None
    if not artifact.build_success:
        return None
    return artifact


def restore_from_compiled_image(
    build_dir: Path,
    build: BuildArtifact,
    docker_tool=None,
) -> Optional[BuildArtifact]:
    """无快照时的兜底（#24 第二梯队）：确定性命名的编译镜像仍在本地时，
    从镜像里提取当年验证过的 build.sh 落回 workspace，并把 build_success
    翻转回 true（镜像可运行即证明构建产物存在）。

    背景：build 失败的重跑会用未验证的 LLM 新脚本覆盖 workspace 的
    build.sh，下游 verify rebuild 拿到垃圾脚本报 127（28044 实证）。
    """

    if docker_tool is None:
        from app.tools.docker_tools import DockerTool

        docker_tool = DockerTool()
    tag = (build.compiled_image_tag or build.docker_image_tag or "").strip()
    if not tag:
        return None
    probe = docker_tool.run_container(
        __import__("app.tools.docker_tools", fromlist=["DockerRunRequest"]).DockerRunRequest(
            image_tag=tag, command=["true"]
        )
    )
    if not probe.success:
        return None
    dump = docker_tool.run_container(
        __import__("app.tools.docker_tools", fromlist=["DockerRunRequest"]).DockerRunRequest(
            image_tag=tag,
            command=["bash", "-c", "cat /workspace/artifacts/build/build.sh 2>/dev/null"],
        )
    )
    script = (dump.stdout or "").strip()
    if not dump.success or not script:
        return None
    try:
        (Path(build_dir) / "build.sh").write_text(script + "\n", encoding="utf-8", newline="\n")
    except OSError:
        return None
    return build.model_copy(
        update={
            "build_success": True,
            "build_script_content": script + "\n",
            "compiled_image_tag": tag,
            "docker_image_tag": tag,
        }
    )
