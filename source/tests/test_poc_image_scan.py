"""文件说明：poc 阶段编译镜像 harness 二进制扫描的单测。

背景（CVE-2022-28044 实证）：库级漏洞的触发二进制（liblrzip_demo 等）
只存在于编译镜像内，宿主 repo 目录扫描永远找不到，LLM 只能拿到 CLI
候选导致 poc 不触发。镜像扫描必须进入候选列表。
"""

from __future__ import annotations

from app.stages.poc import PocStage
from app.tools.docker_tools import DockerCommandResult, DockerRunRequest


class FakeDockerTool:
    def __init__(self, stdout: str, success: bool = True, fail: bool = False) -> None:
        self._stdout = stdout
        self._success = success
        self._fail = fail
        self.requests: list[DockerRunRequest] = []

    def run_container(self, request: DockerRunRequest) -> DockerCommandResult:
        if self._fail:
            raise RuntimeError("docker daemon down")
        self.requests.append(request)
        return DockerCommandResult(success=self._success, exit_code=0, stdout=self._stdout, stderr="")


def test_scan_parses_harness_binaries() -> None:
    stdout = (
        "/src/lrzip/.libs/decompress_demo\n"
        "/src/lrzip/.libs/liblrzip_demo\n"
        "/src/lrzip/decompress_demo\n"
        "/src/lrzip/liblrzip_demo\n"
    )
    fake = FakeDockerTool(stdout)
    stage = PocStage(docker_tool=fake)

    class FakeBuild:
        compiled_image_tag = "deeprepro-cve-2022-28044-build-compiled:latest"

    candidates = stage._scan_image_candidate_binaries(FakeBuild())

    assert "/src/lrzip/.libs/liblrzip_demo" in candidates
    assert all(c.startswith("/") for c in candidates)
    assert fake.requests[0].image_tag.endswith(":latest")
    assert "*_demo" in fake.requests[0].command[-1]


def test_scan_tolerates_missing_image_and_failures() -> None:
    stage = PocStage(docker_tool=FakeDockerTool("", fail=True))

    class NoTag:
        compiled_image_tag = ""

    class Tag:
        compiled_image_tag = "some-image:latest"

    assert stage._scan_image_candidate_binaries(NoTag()) == []
    assert stage._scan_image_candidate_binaries(Tag()) == []
