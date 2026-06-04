"""文件说明：Docker 工具测试。用于校验镜像构建、容器执行和日志采集相关能力。"""

from app.tools import docker_tools as docker_module
from app.tools.process_tools import ProcessResult


def test_build_image_uses_explicit_build_args_only():
    captured = {}

    class FakeProcessTool:
        def run(self, request):
            captured["command"] = request.command
            return ProcessResult(success=True, exit_code=0, stdout="ok", stderr="")

    tool = docker_module.DockerTool(process_tool=FakeProcessTool())
    tool.build_image(
        docker_module.DockerBuildRequest(
            workspace="/tmp/ws",
            dockerfile_path="/tmp/ws/Dockerfile",
            image_tag="demo:latest",
            build_args={"DEMO_FLAG": "enabled"},
        )
    )

    command = captured["command"]
    assert "--build-arg" in command
    assert "DEMO_FLAG=enabled" in command


def test_build_image_supports_explicit_network_mode():
    captured = {}

    class FakeProcessTool:
        def run(self, request):
            captured["command"] = request.command
            return ProcessResult(success=True, exit_code=0, stdout="ok", stderr="")

    tool = docker_module.DockerTool(process_tool=FakeProcessTool())
    tool.build_image(
        docker_module.DockerBuildRequest(
            workspace="/tmp/ws",
            dockerfile_path="/tmp/ws/Dockerfile",
            image_tag="demo:latest",
            network_mode="host",
        )
    )

    command = captured["command"]
    assert "--network" in command
    assert "host" in command


def test_run_container_uses_explicit_environment_only():
    captured = {}

    class FakeProcessTool:
        def run(self, request):
            captured["command"] = request.command
            return ProcessResult(success=True, exit_code=0, stdout="ok", stderr="")

    tool = docker_module.DockerTool(process_tool=FakeProcessTool())
    tool.run_container(
        docker_module.DockerRunRequest(
            image_tag="demo:latest",
            workspace="/tmp/ws",
            command=["bash", "-lc", "env"],
            environment={"DEMO_ENV": "enabled"},
        )
    )

    command = captured["command"]
    assert "-e" in command
    assert "DEMO_ENV=enabled" in command


def test_run_container_supports_named_non_rm_execution():
    captured = {}

    class FakeProcessTool:
        def run(self, request):
            captured["command"] = request.command
            return ProcessResult(success=True, exit_code=0, stdout="ok", stderr="")

    tool = docker_module.DockerTool(process_tool=FakeProcessTool())
    tool.run_container(
        docker_module.DockerRunRequest(
            image_tag="demo:latest",
            command=["bash", "-lc", "env"],
            container_name="demo-container",
            remove=False,
        )
    )

    command = captured["command"]
    assert "--rm" not in command
    assert "--name" in command
    assert "demo-container" in command


def test_commit_container_uses_expected_command():
    captured = {}

    class FakeProcessTool:
        def run(self, request):
            captured["command"] = request.command
            return ProcessResult(success=True, exit_code=0, stdout="sha256:demo", stderr="")

    tool = docker_module.DockerTool(process_tool=FakeProcessTool())
    result = tool.commit_container("demo-container", "demo:compiled")

    assert result.success is True
    assert captured["command"] == ["docker", "commit", "demo-container", "demo:compiled"]
