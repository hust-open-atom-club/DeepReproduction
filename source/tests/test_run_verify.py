"""Run verify tests. Validates the minimum-eligibility report for one PoC execution."""

from app.stages import poc as poc_module


def make_well_formed_logs(exit_code: int = 139, stdout: str = "", stderr: str = "") -> str:
    return (
        "target_binary=demo\n"
        "trigger_command=demo poc\n"
        f"execution_exit_code={exit_code}\n"
        "stdout_begin\n"
        f"{stdout}\n"
        "stdout_end\n"
        "stderr_begin\n"
        f"{stderr}\n"
        "stderr_end\n"
    )


def make_observation(
    exit_code=139,
    stdout: str = "",
    stderr: str = "",
    crash_type: str = "",
):
    return {
        "observed_exit_code": exit_code,
        "observed_stdout": stdout,
        "observed_stderr": stderr,
        "observed_crash_type": crash_type,
    }


def test_run_verify_full_pattern_hit_eligible():
    stage = poc_module.PocStage()
    plan = poc_module.PocPlan(
        target_binary="demo",
        payload_filename="poc.txt",
        run_command="demo poc",
        expected_stderr_patterns=["segmentation fault"],
    )
    logs = make_well_formed_logs(exit_code=139, stderr="Segmentation fault")
    observation = make_observation(
        exit_code=139, stderr="Segmentation fault", crash_type="segmentation fault"
    )

    report = stage._build_run_verify_report(
        plan=plan,
        observation=observation,
        execution_logs=logs,
        matched_error_patterns=["segmentation fault"],
        matched_stack_keywords=[],
    )

    assert report.script_finished is True
    assert report.log_well_formed is True
    assert report.target_binary_invoked is True
    assert report.eligible_for_verify is True
    assert report.eligibility_reason.startswith("error_pattern_hit")


def test_run_verify_only_stack_keyword_hit_eligible():
    stage = poc_module.PocStage()
    plan = poc_module.PocPlan(
        target_binary="demo",
        payload_filename="poc.txt",
        run_command="demo poc",
        expected_stack_keywords=["singlevar"],
    )
    logs = make_well_formed_logs(exit_code=1, stdout="singlevar reached")
    observation = make_observation(exit_code=1, stdout="singlevar reached")

    report = stage._build_run_verify_report(
        plan=plan,
        observation=observation,
        execution_logs=logs,
        matched_error_patterns=[],
        matched_stack_keywords=["singlevar"],
    )

    assert report.eligible_for_verify is True
    assert report.eligibility_reason.startswith("stack_keyword_hit")


def test_run_verify_exit_code_match_only_eligible():
    stage = poc_module.PocStage()
    plan = poc_module.PocPlan(
        target_binary="demo",
        payload_filename="poc.txt",
        run_command="demo poc",
        expected_exit_code=139,
    )
    logs = make_well_formed_logs(exit_code=139)
    observation = make_observation(exit_code=139)

    report = stage._build_run_verify_report(
        plan=plan,
        observation=observation,
        execution_logs=logs,
        matched_error_patterns=[],
        matched_stack_keywords=[],
    )

    assert report.error_pattern_hits == []
    assert report.stack_keyword_hits == []
    assert report.exit_code_match_expected is True
    assert report.eligible_for_verify is True
    assert report.eligibility_reason.startswith("exit_code_match")


def test_run_verify_log_not_well_formed_blocks_eligibility():
    stage = poc_module.PocStage()
    plan = poc_module.PocPlan(
        target_binary="demo",
        payload_filename="poc.txt",
        run_command="demo poc",
        expected_stderr_patterns=["segmentation fault"],
    )
    # Missing stderr_end marker
    logs = (
        "target_binary=demo\n"
        "execution_exit_code=139\n"
        "stdout_begin\n\nstdout_end\n"
        "stderr_begin\nSegmentation fault\n"
    )
    observation = make_observation(
        exit_code=139, stderr="Segmentation fault", crash_type="segmentation fault"
    )

    report = stage._build_run_verify_report(
        plan=plan,
        observation=observation,
        execution_logs=logs,
        matched_error_patterns=["segmentation fault"],
        matched_stack_keywords=[],
    )

    assert report.log_well_formed is False
    assert report.eligible_for_verify is False
    assert report.eligibility_reason.startswith("log_not_well_formed")


def test_run_verify_script_did_not_finish_blocks_eligibility():
    stage = poc_module.PocStage()
    plan = poc_module.PocPlan(
        target_binary="demo",
        payload_filename="poc.txt",
        run_command="demo poc",
        expected_stderr_patterns=["segmentation fault"],
    )
    # No execution_exit_code= marker
    logs = (
        "target_binary=demo\n"
        "stdout_begin\n\nstdout_end\n"
        "stderr_begin\nSegmentation fault\nstderr_end\n"
    )
    observation = make_observation(
        exit_code=None, stderr="Segmentation fault", crash_type="segmentation fault"
    )

    report = stage._build_run_verify_report(
        plan=plan,
        observation=observation,
        execution_logs=logs,
        matched_error_patterns=["segmentation fault"],
        matched_stack_keywords=[],
    )

    assert report.script_finished is False
    assert report.eligible_for_verify is False
    assert report.eligibility_reason.startswith("script_did_not_finish")


def test_run_verify_no_target_behavior_when_everything_clean():
    stage = poc_module.PocStage()
    plan = poc_module.PocPlan(
        target_binary="demo",
        payload_filename="poc.txt",
        run_command="demo poc",
    )
    logs = make_well_formed_logs(exit_code=0, stdout="ok", stderr="")
    observation = make_observation(exit_code=0, stdout="ok", stderr="", crash_type="")

    report = stage._build_run_verify_report(
        plan=plan,
        observation=observation,
        execution_logs=logs,
        matched_error_patterns=[],
        matched_stack_keywords=[],
    )

    assert report.eligible_for_verify is False
    assert report.eligibility_reason == "no_target_behavior_observed"


def test_run_verify_crash_type_partial_match_eligible():
    stage = poc_module.PocStage()
    plan = poc_module.PocPlan(
        target_binary="demo",
        payload_filename="poc.txt",
        run_command="demo poc",
        expected_crash_type="heap-buffer-overflow",
    )
    logs = make_well_formed_logs(
        exit_code=1, stderr="AddressSanitizer: heap-buffer-overflow on address ..."
    )
    observation = make_observation(
        exit_code=1,
        stderr="AddressSanitizer: heap-buffer-overflow on address ...",
        crash_type="addresssanitizer: heap-buffer-overflow",
    )

    report = stage._build_run_verify_report(
        plan=plan,
        observation=observation,
        execution_logs=logs,
        matched_error_patterns=[],
        matched_stack_keywords=[],
    )

    assert report.crash_type_compatible is True
    assert report.eligible_for_verify is True
    assert report.eligibility_reason.startswith("crash_type_compatible")


# ===== Fix 1.A: stdout pattern hit produces eligible_for_verify =====
def test_run_verify_stdout_pattern_hit_eligible():
    """脚本在 stdout 输出错误信息（stderr 干净），expected_stdout_patterns 命中应判 eligible。"""

    stage = poc_module.PocStage()
    plan = poc_module.PocPlan(
        target_binary="demo",
        payload_filename="poc.txt",
        run_command="demo poc",
        expected_stdout_patterns=["stack overflow"],
        expected_stderr_patterns=[],
    )
    logs = make_well_formed_logs(
        exit_code=1,
        stdout="Error: stack overflow at line 42",
        stderr="",
    )
    observation = make_observation(
        exit_code=1,
        stdout="Error: stack overflow at line 42",
        stderr="",
        crash_type="",
    )

    report = stage._build_run_verify_report(
        plan=plan,
        observation=observation,
        execution_logs=logs,
        matched_error_patterns=[],
        matched_stack_keywords=[],
        matched_stdout_patterns=["stack overflow"],
    )

    assert report.stdout_pattern_hits == ["stack overflow"]
    assert report.error_pattern_hits == []
    assert report.eligible_for_verify is True
    assert report.eligibility_reason.startswith("stdout_pattern_hit")
