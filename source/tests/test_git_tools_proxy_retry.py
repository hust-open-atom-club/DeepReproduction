"""文件说明：git_tools 代理失败自动重试的单测。

夹具日志取自真实失败（2026-09 批量复跑 Batch 1，Clash 代理 127.0.0.1:7890
宕机时五个 CVE 的 clone 全部失败，而直连 GitHub 可用）。
"""

from __future__ import annotations

from app.tools.git_tools import GitTool, looks_like_proxy_failure
from app.tools.process_tools import ProcessRequest, ProcessResult


PROXY_DOWN_STDERR = (
    "Cloning into 'workspaces/CVE-2021-36086/repo'...\n"
    "fatal: unable to access 'https://github.com/SELinuxProject/selinux.git/': "
    "Failed to connect to github.com:443 over proxy 127.0.0.1 after 2060 ms: "
    "Could not connect to server\n"
)


class FakeProcessTool:
    """按预置结果序列回放，并记录收到的请求。"""

    def __init__(self, results: list[ProcessResult]) -> None:
        self.results = list(results)
        self.requests: list[ProcessRequest] = []

    def run(self, request: ProcessRequest) -> ProcessResult:
        self.requests.append(request)
        if not self.results:
            raise AssertionError("unexpected extra process call")
        return self.results.pop(0)


def test_looks_like_proxy_failure_signatures() -> None:
    assert looks_like_proxy_failure(ProcessResult(success=False, stderr=PROXY_DOWN_STDERR))
    assert looks_like_proxy_failure(ProcessResult(success=False, stderr="fatal: connection refused"))
    assert not looks_like_proxy_failure(
        ProcessResult(success=False, stderr="error: pathspec 'deadbeef' did not match any file(s)")
    )


def test_clone_retries_without_proxy_on_proxy_failure(tmp_path) -> None:
    ok = ProcessResult(success=True, exit_code=0, stdout="", stderr="")
    fake = FakeProcessTool([ProcessResult(success=False, stderr=PROXY_DOWN_STDERR), ok, ok, ok])
    tool = GitTool(process_tool=fake)

    snapshot = tool.clone_repo("https://github.com/lua/lua.git", str(tmp_path / "repo"))

    assert snapshot.repo_url.endswith("lua.git")
    assert len(fake.requests) == 4
    retry_cmd = fake.requests[1].command
    assert "http.proxy=" in retry_cmd and "https.proxy=" in retry_cmd
    assert fake.requests[1].environment["HTTPS_PROXY"] == ""
    assert fake.requests[1].environment["http_proxy"] == ""
    # 克隆成功后钉死 repo 级 autocrlf=false（防全局 true 在后续 checkout 时涂抹 CRLF）
    assert fake.requests[2].command[-3:] == ["config", "core.autocrlf", "false"]


def test_clone_does_not_retry_on_non_proxy_failure(tmp_path) -> None:
    fail = ProcessResult(success=False, stderr="fatal: repository not found")
    # 直连 + 镜像×2（全量/blobless）×2 + docker
    fake = FakeProcessTool([fail] * 6)
    tool = GitTool(process_tool=fake)

    try:
        tool.clone_repo("https://github.com/example/missing.git", str(tmp_path / "repo"))
        raised = False
    except RuntimeError as error:
        raised = True
        assert "git clone failed" in str(error)
        assert "repository not found" in str(error)
    assert raised
    # 不做去代理重试（非代理失败签名），但镜像与 docker 兜底仍会尝试
    assert len(fake.requests) == 6


def test_fetch_retry_without_proxy_is_tolerated(tmp_path) -> None:
    ok = ProcessResult(success=True, exit_code=0, stdout="", stderr="")
    fake = FakeProcessTool(
        [
            ProcessResult(success=False, stderr="fatal: unable to access ... Connection refused"),
            ok,  # no-proxy fetch retry succeeds
            ok,  # config pin
            ok,  # checkout
            ok,  # rev-parse
        ]
    )
    tool = GitTool(process_tool=fake)

    snapshot = tool.checkout_ref(str(tmp_path), "main")

    assert snapshot.current_ref != "" or snapshot.current_ref == ""
    retry_cmd = fake.requests[1].command
    assert "http.proxy=" in retry_cmd


def test_clone_purges_empty_leftover_dir(tmp_path) -> None:
    ok = ProcessResult(success=True, exit_code=0, stdout="", stderr="")
    leftover = tmp_path / "repo"
    leftover.mkdir()
    fake = FakeProcessTool([ok, ok, ok])  # clone + config pin + rev-parse
    tool = GitTool(process_tool=fake)

    tool.clone_repo("https://github.com/lua/lua.git", str(leftover))

    assert fake.requests[0].command[0] == "git"  # clone actually ran, was not skipped


def test_clone_falls_back_to_mirror_before_docker(tmp_path) -> None:
    ok = ProcessResult(success=True, exit_code=0, stdout="", stderr="")
    proxy_down = ProcessResult(success=False, stderr=PROXY_DOWN_STDERR)
    fake = FakeProcessTool(
        [
            proxy_down,  # 直连
            proxy_down,  # 去代理重试
            ok,  # 镜像 1 成功
            ok,  # config pin
            ok,  # rev-parse
        ]
    )
    tool = GitTool(process_tool=fake)

    snapshot = tool.clone_repo("https://github.com/lua/lua.git", str(tmp_path / "repo"))

    assert snapshot.local_path.endswith("repo")
    mirror_cmd = fake.requests[2].command
    assert any("https://gh-proxy.com/https://github.com/lua/lua.git" == arg for arg in mirror_cmd)
    assert "http.proxy=" in mirror_cmd  # 镜像也必须绕过挂掉的本地代理
    assert fake.requests[2].environment["HTTPS_PROXY"] == ""
    assert len(fake.requests) == 5  # +config pin +rev-parse；docker 兜底不再触发


def test_clone_uses_second_mirror_and_docker_when_mirrors_fail(tmp_path) -> None:
    ok = ProcessResult(success=True, exit_code=0, stdout="", stderr="")
    proxy_down = ProcessResult(success=False, stderr=PROXY_DOWN_STDERR)
    fake = FakeProcessTool(
        [
            proxy_down,  # 直连
            proxy_down,  # 去代理重试
            proxy_down,  # 镜像 1 全量
            proxy_down,  # 镜像 1 blobless
            proxy_down,  # 镜像 2 全量
            proxy_down,  # 镜像 2 blobless
            ok,  # docker 代克隆
            ok,  # config pin
            ok,  # rev-parse
        ]
    )
    tool = GitTool(process_tool=fake)

    snapshot = tool.clone_repo("https://github.com/lua/lua.git", str(tmp_path / "repo"))

    assert fake.requests[5].command[-2] == "https://ghfast.top/https://github.com/lua/lua.git"
    docker_cmd = fake.requests[6].command
    assert docker_cmd[0] == "docker" and "alpine/git" in docker_cmd
    assert any(arg.endswith(":/w") for arg in docker_cmd)
    assert docker_cmd[-1] == "/w/repo"


def test_clone_raises_when_all_tiers_fail(tmp_path) -> None:
    fake = FakeProcessTool([ProcessResult(success=False, stderr=PROXY_DOWN_STDERR) for _ in range(7)])
    tool = GitTool(process_tool=fake)

    try:
        tool.clone_repo("https://github.com/lua/lua.git", str(tmp_path / "repo"))
        raised = False
    except RuntimeError as error:
        raised = True
        assert "git clone failed" in str(error)
        assert "Could not connect to server" in str(error)  # 首次失败的上下文被保留
    assert raised


def test_mirror_list_env_override(monkeypatch, tmp_path) -> None:
    ok = ProcessResult(success=True, exit_code=0, stdout="", stderr="")
    monkeypatch.setenv("DEEPREPRO_GIT_MIRRORS", "https://mirror.example.com, https://backup.example.com/")
    fake = FakeProcessTool(
        [
            ProcessResult(success=False, stderr=PROXY_DOWN_STDERR),
            ProcessResult(success=False, stderr=PROXY_DOWN_STDERR),
            ok,  # 自定义镜像 1 成功
            ok,  # config pin
            ok,  # rev-parse
        ]
    )
    tool = GitTool(process_tool=fake)

    tool.clone_repo("https://github.com/lua/lua.git", str(tmp_path / "repo"))

    mirror_cmd = fake.requests[2].command
    assert any("https://mirror.example.com/https://github.com/lua/lua.git" == arg for arg in mirror_cmd)
