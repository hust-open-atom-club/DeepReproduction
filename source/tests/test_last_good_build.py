# -*- coding: utf-8 -*-
"""last_good_build 快照读写（load / 镜像恢复兜底）的单测。"""

from __future__ import annotations

import yaml

from app.schemas.build_artifact import BuildArtifact
from app.stages.last_good_build import load_last_good_build, restore_from_compiled_image


def _write_snapshot(build_dir, artifact_payload):
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "last_good_build.yaml").write_text(
        yaml.safe_dump({"cve_id": "CVE-0-0", "artifact": artifact_payload}),
        encoding="utf-8",
    )


def test_load_returns_artifact_when_success(tmp_path):
    _write_snapshot(
        tmp_path,
        BuildArtifact(
            dockerfile_content="FROM x",
            build_script_content="set -e",
            build_success=True,
            compiled_image_tag="img:1",
        ).model_dump(mode="json"),
    )
    got = load_last_good_build(tmp_path)
    assert got is not None and got.build_success and got.compiled_image_tag == "img:1"


def test_load_missing_or_failed_returns_none(tmp_path):
    assert load_last_good_build(tmp_path) is None
    _write_snapshot(tmp_path, {"build_success": False})
    assert load_last_good_build(tmp_path) is None


def test_restore_from_compiled_image_writes_script(tmp_path):
    class FakeDockerTool:
        def run_container(self, request):
            class R:
                success = True
                stdout = "#!/bin/bash\nset -e\n"
                stderr = ""

            return R()

    build = BuildArtifact(
        dockerfile_content="",
        build_script_content="garbage",
        build_success=False,
        compiled_image_tag="img:2",
    )
    got = restore_from_compiled_image(tmp_path, build, docker_tool=FakeDockerTool())
    assert got is not None and got.build_success
    assert got.build_script_content.startswith("#!/bin/bash")
    assert (tmp_path / "build.sh").read_text(encoding="utf-8").startswith("#!/bin/bash")
