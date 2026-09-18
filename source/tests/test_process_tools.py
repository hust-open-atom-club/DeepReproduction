"""Process tool tests. Coerce None/bytes subprocess captures into strings."""

from types import SimpleNamespace
from unittest.mock import patch

from app.tools.docker_tools import DockerCommandResult
from app.tools.process_tools import ProcessRequest, ProcessResult, ProcessTool, coerce_process_text


def test_coerce_process_text_normalizes_none_and_bytes():
    assert coerce_process_text(None) == ""
    assert coerce_process_text("ok") == "ok"
    assert coerce_process_text(b"abc") == "abc"


def test_process_result_accepts_none_stdout_stderr():
    result = ProcessResult(success=True, exit_code=0, stdout=None, stderr=None)
    assert result.stdout == ""
    assert result.stderr == ""


def test_docker_command_result_accepts_none_stdout_stderr():
    result = DockerCommandResult(success=False, exit_code=1, stdout=None, stderr=None)
    assert result.stdout == ""
    assert result.stderr == ""


def test_process_tool_run_coerces_none_captures():
    completed = SimpleNamespace(returncode=0, stdout=None, stderr=None)

    with patch("app.tools.process_tools.subprocess.run", return_value=completed):
        result = ProcessTool().run(ProcessRequest(command=["true"], timeout_seconds=5))

    assert result.success is True
    assert result.exit_code == 0
    assert result.stdout == ""
    assert result.stderr == ""
