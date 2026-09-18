"""文件说明：Build 阶段测试。用于校验构建系统识别和节点重试语义。"""

import json
import re
import subprocess
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import AppConfig, AgentModelConfig, RuntimeConfig
from app.schemas.build_artifact import BuildArtifact
from app.schemas.knowledge import KnowledgeModel
from app.stages import build as build_module


def make_knowledge(**overrides):
    payload = {
        "cve_id": "CVE-2022-0000",
        "summary": "demo",
        "vulnerability_type": "heap-overflow",
        "repo_url": "https://example.com/demo.git",
        "vulnerable_ref": "deadbeef",
    }
    payload.update(overrides)
    return KnowledgeModel(**payload)


def test_select_build_system_prefers_repo_scan():
    stage = build_module.BuildStage()
    knowledge = make_knowledge(build_systems=["make"])
    detected_files = ["subdir/CMakeLists.txt", "README.md"]

    assert stage._select_build_system(knowledge, detected_files) == "cmake"


def test_select_build_commands_falls_back_for_unknown():
    stage = build_module.BuildStage()
    knowledge = make_knowledge(build_commands=[])

    commands = stage._select_build_commands(knowledge, "unknown")

    assert commands[-1] == "exit 2"


def test_heuristic_build_plan_prefers_fixed_parent():
    stage = build_module.BuildStage()
    knowledge = make_knowledge(fixed_ref="feedface")
    context = build_module.BuildContext(
        cve_id=knowledge.cve_id,
        repo_url=knowledge.repo_url or "",
        snapshots=[
            build_module.RefSnapshot(label="knowledge_vulnerable", requested_ref="deadbeef", resolved_ref="deadbeef"),
            build_module.RefSnapshot(
                label="fixed_parent",
                requested_ref="beadfeed",
                resolved_ref="beadfeed",
                build_files=["Makefile"],
            ),
        ],
    )

    plan = stage._heuristic_build_plan(knowledge, context, project_name="demo")

    assert plan.chosen_vulnerable_ref == "beadfeed"
    assert plan.build_system == "make"


def test_build_fallback_spec_centralizes_heuristic_defaults():
    stage = build_module.BuildStage()
    knowledge = make_knowledge(
        install_commands=["apt-get install zlib openssl"],
        fixed_ref="feedface",
    )
    context = build_module.BuildContext(
        cve_id=knowledge.cve_id,
        repo_url=knowledge.repo_url or "",
        snapshots=[
            build_module.RefSnapshot(
                label="fixed_parent",
                requested_ref="beadfeed",
                resolved_ref="beadfeed",
                build_files=["CMakeLists.txt"],
            ),
        ],
    )

    spec = stage._build_fallback_spec(knowledge=knowledge, context=context, project_name="demo")

    assert spec.chosen_vulnerable_ref == "beadfeed"
    assert spec.build_system == "cmake"
    assert "cmake --build build -j$(nproc)" in spec.build_commands
    assert "zlib1g-dev" in spec.install_packages
    assert "libssl-dev" in spec.install_packages


def test_build_node_records_retry_on_unsuccessful_build(monkeypatch):
    artifact = BuildArtifact(
        dockerfile_content="FROM ubuntu:20.04\n",
        build_script_content="#!/bin/bash\nexit 2\n",
        build_success=False,
        build_logs="build failed",
    )

    class FakeStage:
        def run(self, knowledge, workspace):
            return artifact

    monkeypatch.setattr(build_module, "BuildStage", FakeStage)

    state = {
        "knowledge": make_knowledge(),
        "workspace": "workspaces/CVE-2022-0000",
        "retry_count": {},
        "stage_history": [],
    }

    result = build_module.build_node(state)

    assert result["build"].build_success is False
    assert result["retry_count"]["build"] == 1
    assert result["stage_history"][-1]["status"] == "failed"


def test_execute_build_plan_uses_script_overrides(tmp_path):
    class FakeDockerTool:
        def build_image(self, request):
            class Result:
                success = True
                exit_code = 0
                stdout = "built"
                stderr = ""

            return Result()

        def run_container(self, request):
            class Result:
                success = True
                exit_code = 0
                stdout = "ran"
                stderr = ""

            return Result()

        def commit_container(self, container_name, image_tag):
            class Result:
                success = True
                exit_code = 0
                stdout = "committed"
                stderr = ""

            return Result()

        def remove_container(self, container_name):
            return None

    stage = build_module.BuildStage(docker_tool=FakeDockerTool())
    paths = build_module.BuildStagePaths(str(tmp_path / "ws"))
    stage._prepare_workspace(paths)
    repo = paths.repo_dir
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "Makefile").write_text("all:\n\techo ok\n", encoding="utf-8")

    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["gcc", "make"],
        build_commands=["make"],
        dockerfile_override="FROM ubuntu:22.04\n",
        build_script_override="#!/bin/bash\necho custom\n",
    )
    artifact = stage._execute_build_plan(
        repo_path=repo,
        paths=paths,
        plan_meta={
            "repo_url": "https://example.com/demo.git",
            "project_name": "demo",
            "project_dir_name": "demo",
            "docker_image_tag": "demo:latest",
            "compiled_image_tag": "demo:compiled",
            "build_container_name": "demo-build-run",
        },
        build_plan=plan,
        resolved_ref="deadbeef",
    )

    assert artifact.dockerfile_content.startswith("FROM ubuntu:22.04\n")
    assert 'git clone "https://example.com/demo.git"' in artifact.dockerfile_content
    assert 'git checkout "deadbeef"' in artifact.dockerfile_content
    assert "ENV PROJECT_DIR=/src/demo" in artifact.dockerfile_content
    assert artifact.build_script_content == "#!/bin/bash\necho custom\n"
    assert artifact.compiled_image_tag == "demo:compiled"


def test_classify_failure_kind_distinguishes_docker_and_container_failures():
    stage = build_module.BuildStage()

    docker_build_log = "image_build_success=False\nimage_build_exit_code=1\n"
    container_run_log = (
        "image_build_success=True\nimage_build_exit_code=0\n\n"
        "container_run_success=False\ncontainer_run_exit_code=2\n"
    )

    assert stage._classify_failure_kind(docker_build_log) == "docker_build"
    assert stage._classify_failure_kind(container_run_log) == "container_run"


def test_valid_replan_candidate_accepts_build_command_changes_for_container_run():
    stage = build_module.BuildStage()
    previous_plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["gcc", "make"],
        build_commands=["make -j$(nproc)"],
    )
    candidate_plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["gcc", "make"],
        build_commands=["make linux"],
    )

    assert stage._is_valid_replan_candidate(
        previous_plan,
        candidate_plan,
        failure_kind="container_run",
    )


def test_normalize_build_plan_strips_legacy_libavif_and_injects_asan_link_flags(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "repo"
    src = repo / "src" / "imageformats"
    src.mkdir(parents=True)
    (repo / "CMakeLists.txt").write_text(
        "\n".join(
            [
                "find_package(ECM 5.82.0 NO_MODULE)",
                "find_package(libavif 0.8.2 CONFIG)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (src / "avif.cpp").write_text(
        "switch (m_decoder->image->imir.axis) { default: break; }\n",
        encoding="utf-8",
    )

    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="cmake",
        install_packages=["cmake", "extra-cmake-modules", "libavif-dev", "qtbase5-dev"],
        build_commands=["cmake -S . -B build", "cmake --build build -j$(nproc)"],
        base_image="ubuntu:22.04",
    )

    normalized = stage._normalize_build_plan(repo, plan)

    assert "libavif-dev" not in normalized.install_packages
    assert "CMAKE_DISABLE_FIND_PACKAGE_libavif=ON" in normalized.build_commands[0]
    assert "CMAKE_MODULE_LINKER_FLAGS=" in normalized.build_commands[0]
    assert "-shared-libasan" in normalized.build_commands[0]
    assert "-fsanitize=address" in normalized.build_commands[0]
    assert "CMAKE_CXX_COMPILER=clang++" in normalized.build_commands[0]


def test_cmake_build_script_keeps_asan_out_of_global_cflags():
    stage = build_module.BuildStage()
    sanitizer = stage._default_sanitizer_flags()
    cmake_script = stage._render_template(
        "build.sh.j2",
        {
            "project_name": "fluent-bit",
            "project_dir_name": "fluent-bit",
            "project_dir": "/src/fluent-bit",
            "build_system": "cmake",
            "sanitizer_flags": sanitizer,
            "configure_commands": [],
            "clean_commands": ["rm -rf build"],
            "build_commands": [
                'cmake -S . -B build -DCMAKE_C_FLAGS="-g -O0 ' + sanitizer + '"',
                "cmake --build build -j$(nproc)",
            ],
        },
    )
    cflags_line = next(line for line in cmake_script.splitlines() if line.startswith("export CFLAGS="))
    ldflags_line = next(line for line in cmake_script.splitlines() if line.startswith("export LDFLAGS="))
    assert "-fsanitize=address" not in cflags_line
    assert "-shared-libasan" not in cflags_line
    assert "-fsanitize=address" not in ldflags_line
    assert "asan_injection=cmake_flags" in cmake_script
    assert "-fsanitize=address" in cmake_script  # still present via CMAKE / SANITIZER_FLAGS

    make_script = stage._render_template(
        "build.sh.j2",
        {
            "project_name": "libsepol",
            "project_dir_name": "libsepol",
            "project_dir": "/src/libsepol",
            "build_system": "make",
            "sanitizer_flags": sanitizer,
            "configure_commands": [],
            "clean_commands": ["make clean || true"],
            "build_commands": ["make -j$(nproc)"],
        },
    )
    make_cflags = next(line for line in make_script.splitlines() if line.startswith("export CFLAGS="))
    assert "-fsanitize=address" in make_cflags
    assert "-shared-libasan" in make_cflags
    assert "asan_injection=env_cflags" in make_script


def test_autotools_build_script_defers_asan_until_after_configure():
    stage = build_module.BuildStage()
    sanitizer = stage._default_sanitizer_flags()
    script = stage._render_template(
        "build.sh.j2",
        {
            "project_name": "qpdf",
            "project_dir_name": "qpdf",
            "project_dir": "/src/qpdf",
            "build_system": "autotools",
            "sanitizer_flags": sanitizer,
            "configure_commands": ["./configure --disable-shared"],
            "clean_commands": ["make clean || true"],
            "build_commands": ["make qpdf_fuzzer -j$(nproc)"],
        },
    )
    lines = script.splitlines()
    first_cflags = next(line for line in lines if line.startswith("export CFLAGS="))
    assert first_cflags == 'export CFLAGS="-g -O0"'
    assert "asan_injection=deferred_env_cflags" in script
    assert 'log "applying sanitizer flags for build"' in script

    clean_idx = script.index('log "running clean step"')
    configure_idx = script.index('log "running configure step"')
    asan_idx = script.index('log "applying sanitizer flags for build"')
    build_idx = script.index('log "running build step"')
    assert clean_idx < configure_idx < asan_idx < build_idx

    asan_block = script[asan_idx:build_idx]
    assert "-shared-libasan" in asan_block
    assert "-fsanitize=address" in asan_block
    # Configure region must not already carry ASan exports.
    configure_region = script[configure_idx:asan_idx]
    assert "-fsanitize=address" not in configure_region


def test_inject_make_asan_overrides_for_autotools_makefile_cflags(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "configure").write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="autotools",
        install_packages=["gcc", "make"],
        configure_commands=["./configure"],
        clean_commands=["make clean || true"],
        build_commands=["make -j$(nproc)"],
    )
    normalized = stage._normalize_build_plan(repo, plan)
    assert len(normalized.build_commands) == 1
    assert "CFLAGS=" in normalized.build_commands[0]
    assert normalized.build_commands[0].startswith("make CC=clang")
    assert "-shared-libasan" in normalized.build_commands[0]
    assert normalized.clean_commands == ["make clean || true"]
    assert all("CFLAGS=" not in cmd for cmd in normalized.clean_commands)


def test_inject_make_asan_overrides_for_lua_style_mycflags(tmp_path):
    """Gate: Makefile CFLAGS=$(MYCFLAGS) ignores env; inject MYCFLAGS + CC=clang."""
    stage = build_module.BuildStage()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "Makefile").write_text(
        "MYCFLAGS= $(LOCAL) -std=c99 -DLUA_USE_LINUX\n"
        "MYLDFLAGS= $(LOCAL) -Wl,-E\n"
        "CC= gcc\n"
        "CFLAGS= -Wall -O2 $(MYCFLAGS)\n",
        encoding="utf-8",
    )
    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["gcc", "make", "clang"],
        build_commands=["make -j$(nproc)"],
        build_script_override=(
            "#!/bin/bash\nset -euxo pipefail\ncd /src/lua\n"
            'export CC=clang\nexport CFLAGS="-fsanitize=address -shared-libasan -g -O1"\n'
            'export LDFLAGS="-fsanitize=address -shared-libasan"\n'
            "make clean || true\n"
            "make -j$(nproc)\n"
        ),
    )
    normalized = stage._normalize_build_plan(repo, plan)
    assert "MYCFLAGS=" in normalized.build_commands[0]
    assert "CC=clang" in normalized.build_commands[0]
    assert "-shared-libasan" in normalized.build_commands[0]
    assert "DLUA_USE_LINUX" in normalized.build_commands[0]
    assert "MYCFLAGS=" in normalized.build_script_override
    assert re.search(
        r"(?m)^make CC=clang MYCFLAGS=.*-shared-libasan",
        normalized.build_script_override,
    )


def test_sanitize_clean_demotes_distclean_when_configure_present():
    stage = build_module.BuildStage()
    assert stage._sanitize_clean_commands(
        ["make distclean", "rm -f *.o"],
        has_configure=True,
    ) == ["make clean || true", "rm -f *.o"]
    assert stage._sanitize_clean_commands(
        ["make distclean"],
        has_configure=False,
    ) == ["make distclean || true"]


def test_sanitize_expected_binary_path_rewrites_bare_kimg_plugin(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "repo"
    repo.mkdir()
    assert (
        stage._sanitize_expected_binary_path("build/bin/kimg_xcf.so", repo)
        == "build/bin/imageformats/kimg_xcf.so"
    )
    assert (
        stage._sanitize_expected_binary_path("build/src/imageformats/kimg_xcf.so", repo)
        == "build/bin/imageformats/kimg_xcf.so"
    )
    assert (
        stage._sanitize_expected_binary_path("build/bin/imageformats/kimg_xcf.so", repo)
        == "build/bin/imageformats/kimg_xcf.so"
    )
    assert (
        stage._sanitize_expected_binary_path(
            "/src/kimageformats/build/src/imageformats/kimg_xcf.so", repo
        )
        == "/src/kimageformats/build/bin/imageformats/kimg_xcf.so"
    )


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _init_git_repo(repo: Path) -> None:
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")


def test_align_vulnerable_ref_switches_to_fixed_parent_when_patch_misses(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    src = repo / "src"
    src.mkdir()
    target = src / "x.c"
    target.write_text("void f(void) {\n    char buf[4];\n}\n", encoding="utf-8")
    _git(repo, "add", "src/x.c")
    _git(repo, "commit", "-m", "vulnerable")
    parent = _git(repo, "rev-parse", "HEAD").stdout.strip()

    target.write_text(
        "void f(void) {\n    if (bpp > 4) return;\n    char buf[4];\n}\n",
        encoding="utf-8",
    )
    _git(repo, "add", "src/x.c")
    _git(repo, "commit", "-m", "fix overflow")
    fixed = _git(repo, "rev-parse", "HEAD").stdout.strip()
    patch = _git(repo, "show", "--format=", fixed).stdout
    patch_path = tmp_path / "patch.diff"
    patch_path.write_text(patch, encoding="utf-8")

    target.write_text("void f(void) {\n    char other[8];\n}\n", encoding="utf-8")
    _git(repo, "add", "src/x.c")
    _git(repo, "commit", "-m", "unrelated rewrite")
    later = _git(repo, "rev-parse", "HEAD").stdout.strip()

    plan = build_module.BuildPlan(
        chosen_vulnerable_ref=later,
        chosen_fixed_ref=fixed,
        build_system="cmake",
        dockerfile_override=(
            "FROM ubuntu:22.04\n"
            f'RUN git clone https://example.com/demo.git /src/demo && git checkout "{later}"\n'
        ),
    )
    aligned = stage._align_vulnerable_ref_with_applyable_patch(
        repo,
        plan,
        patch_diff_path=patch_path,
    )
    assert aligned.chosen_vulnerable_ref == parent
    assert f'git checkout "{parent}"' in aligned.dockerfile_override
    assert later not in aligned.dockerfile_override


def test_align_vulnerable_ref_keeps_ref_when_patch_already_applies(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    src = repo / "src"
    src.mkdir()
    target = src / "x.c"
    target.write_text("void f(void) {\n    char buf[4];\n}\n", encoding="utf-8")
    _git(repo, "add", "src/x.c")
    _git(repo, "commit", "-m", "vulnerable")
    parent = _git(repo, "rev-parse", "HEAD").stdout.strip()

    target.write_text(
        "void f(void) {\n    if (bpp > 4) return;\n    char buf[4];\n}\n",
        encoding="utf-8",
    )
    _git(repo, "add", "src/x.c")
    _git(repo, "commit", "-m", "fix overflow")
    fixed = _git(repo, "rev-parse", "HEAD").stdout.strip()
    patch = _git(repo, "show", "--format=", fixed).stdout
    patch_path = tmp_path / "patch.diff"
    patch_path.write_text(patch, encoding="utf-8")

    plan = build_module.BuildPlan(
        chosen_vulnerable_ref=parent,
        chosen_fixed_ref=fixed,
        build_system="make",
    )
    aligned = stage._align_vulnerable_ref_with_applyable_patch(
        repo,
        plan,
        patch_diff_path=patch_path,
    )
    assert aligned.chosen_vulnerable_ref == parent


def test_defer_asan_and_demote_distclean_in_script_override():
    stage = build_module.BuildStage()
    script = "\n".join(
        [
            'export CFLAGS="-g -O0 -fsanitize=address -shared-libasan -fno-omit-frame-pointer"',
            'export CXXFLAGS="-g -O0 -fsanitize=address -shared-libasan -fno-omit-frame-pointer"',
            'export LDFLAGS="-fsanitize=address -shared-libasan -fno-omit-frame-pointer"',
            "./configure --disable-shared",
            "make distclean || true",
            "make qpdf_fuzzer -j$(nproc)",
            "",
        ]
    )
    demoted = stage._demote_distclean_in_script(script)
    assert "distclean" not in demoted
    assert "make clean || true" in demoted
    deferred = stage._defer_asan_env_until_after_configure(demoted)
    cflags_line = next(line for line in deferred.splitlines() if line.startswith("export CFLAGS="))
    assert cflags_line == 'export CFLAGS="-g -O0"'
    assert 'log "applying sanitizer flags for build"' in deferred
    apply_idx = deferred.index('log "applying sanitizer flags for build"')
    configure_idx = deferred.index("./configure --disable-shared")
    assert configure_idx < apply_idx
    assert "-shared-libasan" in deferred[apply_idx:]


def test_cmake_override_strips_global_asan_and_injects_cmake_flags():
    stage = build_module.BuildStage()
    script = "\n".join(
        [
            'export CFLAGS="-g -O0 -fsanitize=address -shared-libasan -fno-omit-frame-pointer"',
            'export CXXFLAGS="-g -O0 -fsanitize=address -shared-libasan -fno-omit-frame-pointer"',
            'export LDFLAGS="-fsanitize=address -shared-libasan -fno-omit-frame-pointer"',
            "cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug",
            "cmake --build build -j$(nproc)",
            "",
        ]
    )
    with_flags = stage._ensure_cmake_asan_flags_in_script(script)
    assert "CMAKE_SHARED_LINKER_FLAGS=" in with_flags
    assert "-shared-libasan" in with_flags
    stripped = stage._strip_global_asan_env_exports(with_flags)
    cflags_line = next(line for line in stripped.splitlines() if line.startswith("export CFLAGS="))
    ldflags_line = next(line for line in stripped.splitlines() if line.startswith("export LDFLAGS="))
    assert cflags_line == 'export CFLAGS="-g -O0"'
    assert ldflags_line == 'export LDFLAGS=""'
    assert "CMAKE_C_FLAGS=" in stripped
    # cmake --build must stay untouched
    assert "cmake --build build -j$(nproc)" in stripped


def test_prefer_ossfuzz_harness_when_in_tree_evidence_exists(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "fluent-bit"
    fuzz_dir = repo / "tests" / "internal" / "fuzzers"
    fuzz_dir.mkdir(parents=True)
    (repo / "CMakeLists.txt").write_text("project(fluent-bit)\n", encoding="utf-8")
    (fuzz_dir / "parser_fuzzer.c").write_text("int LLVMFuzzerTestOneInput(){return 0;}\n", encoding="utf-8")
    (fuzz_dir / "CMakeLists.txt").write_text("parser_fuzzer.c\n", encoding="utf-8")

    knowledge = make_knowledge(
        repo_url="https://github.com/fluent/fluent-bit.git",
        reproduction_hints=[
            "Harvested OSS-Fuzz testcase into vuln_pocs/"
            "clusterfuzz-testcase-minimized-flb-it-fuzz-parser_fuzzer_OSSFUZZ-5216297967288320.fuzz"
        ],
        reproduction_recipes=[
            {
                "source_url": "https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=33750",
                "source_title": "clusterfuzz-testcase-minimized-flb-it-fuzz-parser_fuzzer_OSSFUZZ-5216297967288320.fuzz",
                "recipe_type": "ossfuzz_testcase",
                "steps": [],
                "confidence": "high",
            }
        ],
    )
    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="cmake",
        install_packages=["cmake", "clang"],
        build_commands=["cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug", "cmake --build build -j$(nproc)"],
        expected_binary_path="build/bin/fluent-bit",
        base_image="ubuntu:20.04",
    )

    normalized = stage._normalize_build_plan(repo, plan, knowledge=knowledge)

    assert normalized.expected_binary_path == "build/bin/flb-it-fuzz-parser_fuzzer"
    assert "FLB_TESTS_INTERNAL_FUZZ=On" in normalized.build_commands[0]
    assert any(
        "cmake --build" in cmd and "--target flb-it-fuzz-parser_fuzzer" in cmd
        for cmd in normalized.build_commands
    )
    assert all(
        "cmake --build" not in cmd or "--target flb-it-fuzz-parser_fuzzer" in cmd
        for cmd in normalized.build_commands
    )


def test_narrow_build_to_harness_rewrites_cmake_build_command():
    stage = build_module.BuildStage()
    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="cmake",
        build_commands=[
            "cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug",
            "cmake --build build -j$(nproc)",
        ],
    )
    narrowed = stage._narrow_build_to_harness_target(plan, "flb-it-fuzz-parser_fuzzer")
    assert narrowed.build_commands[1] == (
        "cmake --build build --target flb-it-fuzz-parser_fuzzer -j$(nproc)"
    )


def test_prefer_ossfuzz_harness_skips_without_in_tree_evidence(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "selinux"
    (repo / "libsepol").mkdir(parents=True)
    (repo / "secilc").mkdir(parents=True)
    (repo / "Makefile").write_text("all:\n\ttrue\n", encoding="utf-8")

    knowledge = make_knowledge(
        repo_url="https://github.com/SELinuxProject/selinux.git",
        reproduction_hints=[
            "Harvested OSS-Fuzz testcase into vuln_pocs/"
            "clusterfuzz-testcase-minimized-secilc-fuzzer-5563841674084352.cil"
        ],
    )
    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        build_commands=["make -j$(nproc)"],
        expected_binary_path="secilc/secilc",
        rationale="secilc CIL double-free",
        base_image="ubuntu:20.04",
    )

    normalized = stage._normalize_build_plan(repo, plan, knowledge=knowledge)

    assert normalized.expected_binary_path == "secilc/secilc"
    assert all("FLB_TESTS_INTERNAL_FUZZ" not in cmd for cmd in (normalized.build_commands or []))


def test_prefer_qpdf_style_fuzz_mk_harness_rewrites_configure_and_make(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "qpdf"
    fuzz_dir = repo / "fuzz"
    fuzz_dir.mkdir(parents=True)
    (fuzz_dir / "qpdf_fuzzer.cc").write_text(
        "extern \"C\" int LLVMFuzzerTestOneInput(const uint8_t*, size_t){return 0;}\n",
        encoding="utf-8",
    )
    (fuzz_dir / "build.mk").write_text(
        "FUZZERS = qpdf_fuzzer\n"
        "BINS_fuzz = $(foreach B,$(FUZZERS),fuzz/$(OUTPUT_DIR)/$(B))\n"
        "TARGETS_fuzz = $(BINS_fuzz)\n",
        encoding="utf-8",
    )
    (fuzz_dir / "oss-fuzz-build").write_text(
        "#!/bin/bash\n./configure --enable-oss-fuzz\nmake install_fuzz\n",
        encoding="utf-8",
    )
    (repo / "configure.ac").write_text(
        "AC_INIT([qpdf],[10])\n"
        "AC_ARG_ENABLE(oss-fuzz, [AS_HELP_STRING([--enable-oss-fuzz],[oss-fuzz])])\n",
        encoding="utf-8",
    )
    (repo / "Makefile").write_text("OUTPUT_DIR = build\nall:\n\ttrue\n", encoding="utf-8")

    knowledge = make_knowledge(
        repo_url="https://github.com/qpdf/qpdf.git",
        reproduction_hints=[
            "Harvested OSS-Fuzz testcase into vuln_pocs/"
            "clusterfuzz-testcase-minimized-qpdf_fuzzer-5162370603286528.cil"
        ],
    )
    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="autotools",
        configure_commands=["./configure --enable-fuzzers --disable-shared --enable-static"],
        clean_commands=["make clean || true"],
        build_commands=["make qpdf_fuzzer -j$(nproc)"],
        expected_binary_path="fuzz/qpdf_fuzzer",
        base_image="ubuntu:20.04",
    )

    normalized = stage._normalize_build_plan(repo, plan, knowledge=knowledge)

    assert normalized.expected_binary_path == "fuzz/build/qpdf_fuzzer"
    assert all("--enable-fuzzers" not in cmd for cmd in (normalized.configure_commands or []))
    assert any(
        cmd.startswith("./configure") and "--disable-shared" in cmd
        for cmd in (normalized.configure_commands or [])
    )
    assert any(
        re.search(r"\bfuzz/build/qpdf_fuzzer\b", cmd)
        for cmd in (normalized.build_commands or [])
    )
    assert all(
        not re.search(r"(?<!fuzz/build/)make(?:\s+\S+=\S+)*\s+qpdf_fuzzer\b", cmd)
        for cmd in (normalized.build_commands or [])
    )


def test_qpdf_style_fuzz_mk_gate_skips_without_oss_fuzz_wiring(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "other"
    fuzz_dir = repo / "fuzz"
    fuzz_dir.mkdir(parents=True)
    (fuzz_dir / "qpdf_fuzzer.cc").write_text("int x;\n", encoding="utf-8")
    (fuzz_dir / "build.mk").write_text(
        "FUZZERS = qpdf_fuzzer\nfuzz/$(OUTPUT_DIR)/qpdf_fuzzer:\n",
        encoding="utf-8",
    )
    # No oss-fuzz-build and no enable-oss-fuzz in configure.ac → gate closed.
    (repo / "configure.ac").write_text("AC_INIT([other],[1])\n", encoding="utf-8")
    (repo / "Makefile").write_text("OUTPUT_DIR = build\n", encoding="utf-8")

    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="autotools",
        configure_commands=["./configure --enable-fuzzers"],
        build_commands=["make qpdf_fuzzer -j$(nproc)"],
        expected_binary_path="fuzz/qpdf_fuzzer",
    )
    knowledge = make_knowledge(
        reproduction_hints=[
            "clusterfuzz-testcase-minimized-qpdf_fuzzer-5162370603286528.cil"
        ],
    )

    normalized = stage._normalize_build_plan(repo, plan, knowledge=knowledge)

    # Evidence finds the .cc under fuzz/, but without qpdf-style OSS wiring the
    # preferred path must not become fuzz/build/... and configure stays put.
    assert normalized.expected_binary_path != "fuzz/build/qpdf_fuzzer"
    assert "--enable-fuzzers" in (normalized.configure_commands or [""])[0]
    assert all("fuzz/build/qpdf_fuzzer" not in cmd for cmd in (normalized.build_commands or []))
    assert any(
        re.search(r"\bqpdf_fuzzer\b", cmd) for cmd in (normalized.build_commands or [])
    )


def test_prefer_standalone_ossfuzz_cpp_harness_appends_compile(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "matio"
    ossfuzz_dir = repo / "ossfuzz"
    ossfuzz_dir.mkdir(parents=True)
    (repo / "CMakeLists.txt").write_text(
        "project(matio)\noption(MATIO_WITH_HDF5 \"Check for HDF5 library\" ON)\n",
        encoding="utf-8",
    )
    cmake_dir = repo / "cmake"
    cmake_dir.mkdir()
    (cmake_dir / "thirdParties.cmake").write_text(
        "if(MATIO_WITH_HDF5)\n  find_package(HDF5)\nendif()\n",
        encoding="utf-8",
    )
    (ossfuzz_dir / "matio_fuzzer.cpp").write_text(
        'extern "C" int LLVMFuzzerTestOneInput(){return 0;}\n',
        encoding="utf-8",
    )

    knowledge = make_knowledge(
        repo_url="https://github.com/tbeu/matio.git",
        reproduction_hints=[
            "Harvested OSS-Fuzz testcase into vuln_pocs/"
            "clusterfuzz-testcase-minimized-matio_fuzzer-4806922097262592"
        ],
        reproduction_recipes=[
            {
                "source_url": "https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=31265",
                "source_title": "clusterfuzz-testcase-minimized-matio_fuzzer-4806922097262592",
                "recipe_type": "ossfuzz_testcase",
                "steps": [],
                "confidence": "high",
            }
        ],
    )
    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="cmake",
        install_packages=["cmake", "clang", "libhdf5-dev"],
        build_commands=[
            "cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug",
            "cmake --build build -j$(nproc)",
        ],
        expected_binary_path=None,
        base_image="ubuntu:20.04",
    )

    normalized = stage._normalize_build_plan(repo, plan, knowledge=knowledge)

    assert normalized.expected_binary_path == "matio_fuzzer"
    # Must keep full library build; do NOT narrow to --target matio_fuzzer.
    assert any(
        "cmake --build" in cmd and "--target matio_fuzzer" not in cmd
        for cmd in normalized.build_commands
    )
    assert any(
        "ossfuzz/matio_fuzzer.cpp" in cmd
        and "-o matio_fuzzer" in cmd
        and "-lmatio" in cmd
        and "-pthread" in cmd
        and "/opt/hdf5-vuln" in cmd
        for cmd in normalized.build_commands
    )
    assert any("deeprepro:vulnerable-hdf5-1.12.0" in cmd for cmd in normalized.build_commands)
    assert any("-DHDF5_ROOT=/opt/hdf5-vuln" in cmd for cmd in normalized.build_commands)
    hdf5_cmd = next(
        cmd for cmd in normalized.build_commands if "deeprepro:vulnerable-hdf5-1.12.0" in cmd
    )
    assert "./configure" in hdf5_cmd
    assert "--disable-asserts" in hdf5_cmd
    assert "--disable-internal-debug" in hdf5_cmd
    assert "-fsanitize=address" in hdf5_cmd
    assert "RelWithDebInfo" not in hdf5_cmd
    assert "cmake -S" not in hdf5_cmd
    assert any(
        "libclang_rt.asan-x86_64.so" in cmd and "LD_LIBRARY_PATH=" in cmd
        for cmd in normalized.build_commands
    )
    assert "libhdf5-dev" not in [p.lower() for p in (normalized.install_packages or [])]
    assert "zlib1g-dev" in (normalized.install_packages or [])
    assert "curl" in (normalized.install_packages or [])
    assert all("-lrepo" not in cmd for cmd in (normalized.build_commands or []))
    assert all("FLB_TESTS_INTERNAL_FUZZ" not in cmd for cmd in (normalized.build_commands or []))


def test_vulnerable_hdf5_build_command_matches_ossfuzz_autotools_flags():
    cmd = build_module.BuildStage()._vulnerable_hdf5_build_command()
    assert "deeprepro:vulnerable-hdf5-1.12.0" in cmd
    assert "./configure" in cmd
    assert "--disable-asserts" in cmd
    assert "--disable-internal-debug" in cmd
    assert "--disable-deprecated-symbols" in cmd
    assert "--enable-shared" in cmd
    assert "CC=clang" in cmd
    assert "-shared-libasan" in cmd
    assert "libclang_rt.asan-x86_64.so" in cmd
    assert "-Wl,-rpath,${_asan_dir}" in cmd
    assert 'LDFLAGS=""' in cmd
    assert 'LDFLAGS="-fsanitize=address' not in cmd
    assert "RelWithDebInfo" not in cmd
    assert "cmake -S" not in cmd
    assert "hdf5-1.12.0.tar.gz" in cmd
    assert "hdf5-1_12_0.tar.gz" in cmd  # GitHub tag fallback
    assert "cd /;" not in cmd
    assert "); " in cmd


def test_hdf5_autotools_install_does_not_leave_cwd_at_root():
    vuln = build_module.BuildStage()._vulnerable_hdf5_build_command()
    fixed = build_module.BuildStage()._fixed_hdf5_build_command()
    for cmd in (vuln, fixed):
        assert "cd /;" not in cmd
        echo_at = cmd.find("echo '[build]")
        subshell_open = cmd.find("(", echo_at)
        subshell_close = cmd.rfind(")")
        cd_src = cmd.find("cd /tmp/")
        assert echo_at != -1 and subshell_open != -1 and subshell_close != -1
        assert subshell_open < cd_src < subshell_close


def test_fixed_hdf5_autotools_stays_production_without_asan():
    cmd = build_module.BuildStage()._fixed_hdf5_build_command()
    assert "deeprepro:fixed-hdf5-1.12.1" in cmd
    assert "./configure" in cmd
    assert "--enable-build-mode=production" in cmd
    assert "--disable-asserts" not in cmd
    assert "-fsanitize=address" not in cmd
    assert "CC=clang" not in cmd


def test_rewrite_build_script_for_fixed_hdf5_swaps_prefix_and_version():
    cmd = build_module.BuildStage.rewrite_build_script_for_fixed_hdf5(
        "/workspace/artifacts/build/build.sh"
    )
    # Pre-install fixed HDF5 via autotools (avoids 1.12.1 cmake ConfigureChecks).
    assert "deeprepro:fixed-hdf5-1.12.1" in cmd
    assert "./configure" in cmd
    assert "cmake -S" not in cmd.split("&&")[0]
    assert "/opt/hdf5-vuln" in cmd
    assert "/opt/hdf5-fixed" in cmd
    assert "hdf5-1.12.1.tar.gz" in cmd
    assert "hdf5-1.12.0" in cmd  # rewritten from vuln URL
    assert "s|hdf5-1.12.0|hdf5-1.12.1|g" in cmd
    assert "/tmp/build-hdf5-fixed.sh" in cmd
    assert "&&" in cmd


def test_apply_vulnerable_hdf5_cve_match_policy_tightens_patterns_and_asan():
    stderr, stack, crash, env = build_module.BuildStage.apply_vulnerable_hdf5_cve_match_policy(
        expected_stderr_patterns=["AddressSanitizer", "heap-buffer-overflow"],
        expected_stack_keywords=["AddressSanitizer", "Mat_VarRead", "H5C_load_entry", "affected:", "+id: OSV-2021-1166"],
        expected_crash_type="",
        environment_variables={"FOO": "1"},
    )
    assert stderr == ["heap-buffer-overflow", "H5MM_memcpy"]
    assert "AddressSanitizer" not in stderr
    assert "AddressSanitizer" not in stack
    assert "H5MM_memcpy" in stack
    assert "H5MM_malloc" in stack
    assert "H5C_load_entry" not in stack
    assert "affected:" not in stack
    assert "+id: OSV-2021-1166" not in stack
    assert "Mat_VarRead" in stack  # preserved non-generic keyword
    assert crash == "heap-buffer-overflow"
    assert env["FOO"] == "1"
    assert "allocator_may_return_null=1" in env["ASAN_OPTIONS"]


def test_standalone_ossfuzz_without_hdf5_skips_vulnerable_hdf5_prefix(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "other"
    ossfuzz_dir = repo / "ossfuzz"
    ossfuzz_dir.mkdir(parents=True)
    (repo / "CMakeLists.txt").write_text("project(other)\n", encoding="utf-8")
    (ossfuzz_dir / "other_fuzzer.cpp").write_text(
        'extern "C" int LLVMFuzzerTestOneInput(){return 0;}\n',
        encoding="utf-8",
    )
    knowledge = make_knowledge(
        repo_url="https://example.com/other.git",
        reproduction_hints=[
            "Harvested OSS-Fuzz testcase into vuln_pocs/"
            "clusterfuzz-testcase-minimized-other_fuzzer-4806922097262592"
        ],
        reproduction_recipes=[
            {
                "source_url": "https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=1",
                "source_title": "clusterfuzz-testcase-minimized-other_fuzzer-4806922097262592",
                "recipe_type": "ossfuzz_testcase",
                "steps": [],
                "confidence": "high",
            }
        ],
    )
    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="cmake",
        install_packages=["cmake", "clang"],
        build_commands=[
            "cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug",
            "cmake --build build -j$(nproc)",
        ],
        expected_binary_path=None,
        base_image="ubuntu:20.04",
    )

    normalized = stage._normalize_build_plan(repo, plan, knowledge=knowledge)

    assert normalized.expected_binary_path == "other_fuzzer"
    assert all("hdf5-vuln" not in cmd for cmd in (normalized.build_commands or []))
    assert all("vulnerable-hdf5" not in cmd for cmd in (normalized.build_commands or []))
    # Still appends standalone compile, but via system/pkg-config hdf5 fallback path.
    assert any("ossfuzz/other_fuzzer.cpp" in cmd for cmd in (normalized.build_commands or []))


def test_guess_standalone_ossfuzz_link_lib_prefers_harness_stem():
    stage = build_module.BuildStage()
    assert stage._guess_standalone_ossfuzz_link_lib("matio_fuzzer", Path("/tmp/repo")) == "matio"
    assert stage._guess_standalone_ossfuzz_link_lib("matio_fuzzer", Path("/tmp/matio")) == "matio"
    assert stage._guess_standalone_ossfuzz_link_lib("weird", Path("/tmp/customlib")) == "customlib"


def test_select_cxx_does_not_treat_clangxx_as_gxx():
    stage = build_module.BuildStage()
    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="cmake",
        build_commands=[
            'cmake -S . -B build -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_MODULE_LINKER_FLAGS="-fsanitize=address"',
        ],
    )
    assert stage._select_cxx(plan) == "clang++"
    assert stage._select_compiler(plan) == "clang"


def test_rewrite_gcc_compiler_to_clang_handles_abcm2ps_style_cc_flag():
    stage = build_module.BuildStage()
    assert stage._rewrite_gcc_compiler_to_clang("./configure --CC=gcc") == "./configure --CC=clang"
    assert stage._rewrite_gcc_compiler_to_clang('export CC="gcc"\n') == 'export CC="clang"\n'
    assert (
        stage._rewrite_gcc_compiler_to_clang("cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++")
        == "cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++"
    )


def test_normalize_build_plan_forces_clang_when_shared_libasan(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "abcm2ps"
    repo.mkdir()
    (repo / "configure").write_text("#!/bin/sh\n", encoding="utf-8")
    (repo / "Makefile.in").write_text("all:\n\t$(CC) -o abcm2ps *.c\n", encoding="utf-8")
    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="autotools",
        install_packages=["clang", "gcc", "make"],
        configure_commands=["./configure --CC=gcc"],
        clean_commands=["make clean || true"],
        build_commands=["make -j$(nproc)"],
        build_script_override="#!/bin/bash\nexport CC=gcc\n./configure --CC=gcc\nmake -j$(nproc)\n",
        expected_binary_path="abcm2ps",
        rationale="README uses gcc; should still align with shared-libasan.",
    )
    normalized = stage._normalize_build_plan(repo, plan)
    assert normalized.configure_commands == ["./configure --CC=clang"]
    assert "CC=clang" in (normalized.build_script_override or "")
    assert "--CC=clang" in (normalized.build_script_override or "")
    assert "CC=gcc" not in (normalized.build_script_override or "")
    assert "--CC=gcc" not in (normalized.build_script_override or "")
    assert stage._select_compiler(normalized) == "clang"
    assert stage._select_cxx(normalized) == "clang++"


def test_make_shared_lib_gets_shared_libasan(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "selinux"
    src = repo / "libsepol" / "src"
    src.mkdir(parents=True)
    (src / "Makefile").write_text(
        "libsepol.so: $(LOBJS)\n\t$(CC) $(CFLAGS) $(LDFLAGS) -shared -o $@ $(LOBJS)\n",
        encoding="utf-8",
    )
    assert stage._repo_links_shared_objects(repo) is True
    assert "-shared-libasan" in stage._default_sanitizer_flags(repo)

    script = (
        'export SANITIZER_FLAGS="-fsanitize=address -fno-omit-frame-pointer"\n'
        'export CFLAGS="-g -O0 -fsanitize=address -fno-omit-frame-pointer"\n'
        'export LDFLAGS="-fsanitize=address -fno-omit-frame-pointer"\n'
    )
    fixed = stage._ensure_shared_libasan_in_build_script(script, repo)
    assert fixed.count("-shared-libasan") == 3
    assert "-fsanitize=address -shared-libasan" in fixed


def test_augment_packages_adds_pcre_and_python3_for_selinux(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "selinux"
    src = repo / "libselinux" / "src"
    src.mkdir(parents=True)
    (src / "regex.h").write_text("#include <pcre.h>\n", encoding="utf-8")
    (src / "Makefile").write_text(
        "PYTHON ?= python3\nPCRE_LDLIBS ?= -lpcre\nclean-pywrap:\n\tpython3 setup.py clean\n",
        encoding="utf-8",
    )
    packages = stage._augment_install_packages_from_repo(["clang", "make"], repo)
    assert "libpcre3-dev" in packages
    assert "python3" in packages
    assert "python3-distutils" in packages
    assert "libpcre2-dev" not in packages


def test_narrow_selinux_cil_build_skips_full_tree(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "selinux"
    (repo / "libsepol").mkdir(parents=True)
    (repo / "secilc").mkdir(parents=True)
    (repo / "Makefile").write_text("all:\n\t$(MAKE) -C libsepol\n", encoding="utf-8")
    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["clang", "make", "libpcre3-dev", "libaudit-dev", "libsepol-dev"],
        build_commands=["make -j$(nproc)"],
        clean_commands=["make clean || true"],
        expected_binary_path="secilc/secilc",
        rationale="libsepol CIL / secilc vulnerability",
    )
    narrowed = stage._narrow_selinux_cil_build(repo, plan)
    assert any("make -C libsepol" in cmd for cmd in narrowed.build_commands)
    assert any("make -C secilc secilc" in cmd for cmd in narrowed.build_commands)
    assert any("libsepol/include/sepol/cil" in cmd for cmd in narrowed.build_commands)
    assert "libpcre3-dev" not in narrowed.install_packages
    assert "libaudit-dev" not in narrowed.install_packages


def test_narrow_selinux_cil_build_rewrites_checkpolicy_entrypoint(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "selinux"
    (repo / "libsepol").mkdir(parents=True)
    (repo / "secilc").mkdir(parents=True)
    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["clang", "make"],
        build_commands=["make -j$(nproc)"],
        expected_binary_path="checkpolicy/checkpolicy",
        rationale="cil_reset_perm UAF in libsepol; trigger via checkpolicy",
    )
    narrowed = stage._narrow_selinux_cil_build(repo, plan)
    assert narrowed.expected_binary_path == "secilc/secilc"
    assert any("make -C secilc secilc" in cmd for cmd in narrowed.build_commands)


def test_normalize_build_plan_upgrades_base_image_for_modern_ecm(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "CMakeLists.txt").write_text(
        "\n".join(
            [
                "cmake_minimum_required(VERSION 3.16)",
                "find_package(ECM 5.82.0 NO_MODULE)",
                "set(REQUIRED_QT_VERSION 5.15.0)",
                "find_package(Qt5Gui ${REQUIRED_QT_VERSION} REQUIRED NO_MODULE)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="cmake",
        install_packages=["cmake", "extra-cmake-modules", "libavif-dev", "qtbase5-dev"],
        build_commands=["cmake -S . -B build", "cmake --build build -j$(nproc)"],
    )

    normalized = stage._normalize_build_plan(repo, plan)

    assert normalized.base_image == "ubuntu:22.04"
    assert "libavif-dev" in normalized.install_packages


def test_normalize_build_plan_rewrites_override_from_when_upgrading(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "CMakeLists.txt").write_text(
        "find_package(ECM 5.82.0 NO_MODULE)\n",
        encoding="utf-8",
    )

    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="cmake",
        install_packages=["cmake", "extra-cmake-modules"],
        build_commands=["cmake -S . -B build", "cmake --build build -j$(nproc)"],
        dockerfile_override=(
            "FROM ubuntu:20.04\n"
            "RUN apt-get update && apt-get install -y --no-install-recommends cmake extra-cmake-modules\n"
        ),
    )

    normalized = stage._normalize_build_plan(repo, plan)

    assert normalized.base_image == "ubuntu:22.04"
    assert normalized.dockerfile_override.startswith("FROM ubuntu:22.04\n")


def test_normalize_build_plan_strips_focal_unavailable_packages(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.16)\n", encoding="utf-8")

    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="cmake",
        install_packages=["cmake", "libavif-dev", "qtbase5-dev"],
        configure_commands=["cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=OFF .."],
        clean_commands=["rm -rf build"],
        build_commands=["cmake --build . -j$(nproc)"],
        dockerfile_override=(
            "FROM ubuntu:20.04\n"
            "RUN apt-get update && apt-get install -y --no-install-recommends cmake libavif-dev qtbase5-dev\n"
        ),
    )

    normalized = stage._normalize_build_plan(repo, plan)

    assert "libavif-dev" not in normalized.install_packages
    assert "qtbase5-dev" in normalized.install_packages
    assert "libavif-dev" not in (normalized.dockerfile_override or "")
    assert normalized.configure_commands == []
    assert normalized.build_commands[0].startswith("cmake -S . -B build")
    assert "BUILD_TESTING=OFF" in normalized.build_commands[0]
    assert normalized.build_commands[1] == "cmake --build build -j$(nproc)"
    assert any("rm -rf build" in command for command in normalized.clean_commands)


def test_normalize_build_plan_keeps_libavif_on_jammy(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.16)\n", encoding="utf-8")

    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="cmake",
        install_packages=["cmake", "libavif-dev"],
        build_commands=["cmake -S . -B build", "cmake --build build -j$(nproc)"],
        dockerfile_override="FROM ubuntu:22.04\nRUN apt-get install -y libavif-dev\n",
    )

    normalized = stage._normalize_build_plan(repo, plan)

    assert "libavif-dev" in normalized.install_packages
    assert "libavif-dev" in (normalized.dockerfile_override or "")


def test_rewrite_broken_cmake_in_script():
    stage = build_module.BuildStage()
    script = "\n".join(
        [
            "#!/bin/bash",
            "cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=OFF ..",
            "cmake --build . -j$(nproc)",
            "",
        ]
    )
    rewritten = stage._rewrite_broken_cmake_in_script(script)
    assert "cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug" in rewritten
    assert "BUILD_TESTING=OFF" in rewritten
    assert "cmake --build build -j$(nproc)" in rewritten
    assert "cmake --build ." not in rewritten


def test_normalize_build_plan_preserves_required_docker_packages(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "repo"
    repo.mkdir()

    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["gcc", "make"],
        build_commands=["make"],
    )

    normalized = stage._normalize_build_plan(repo, plan)

    assert "build-essential" in normalized.install_packages
    assert "clang" in normalized.install_packages
    assert "gcc" in normalized.install_packages
    assert "g++" in normalized.install_packages
    assert "git" in normalized.install_packages
    assert "make" in normalized.install_packages
    assert "pkg-config" in normalized.install_packages
    assert "ca-certificates" in normalized.install_packages


def test_normalize_build_plan_adds_libreadline_when_makefile_requires_it(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "Makefile").write_text(
        "MYCFLAGS= -DLUA_USE_READLINE\nMYLIBS= -lreadline\n",
        encoding="utf-8",
    )

    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["gcc", "make"],
        build_commands=["make"],
    )

    normalized = stage._normalize_build_plan(repo, plan)

    assert "libreadline-dev" in normalized.install_packages


def test_normalize_build_plan_rewrites_invalid_make_targets(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "Makefile").write_text("all:\n\techo ok\nclean:\n\trm -f *.o\n", encoding="utf-8")

    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["gcc", "make"],
        build_commands=["make clean || true", "make linux"],
    )

    normalized = stage._normalize_build_plan(repo, plan)

    assert normalized.build_commands == ["make clean || true", "make -j$(nproc)"]


def test_normalize_build_plan_softens_make_clean_and_adds_configure(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "configure").write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    (repo / "Makefile.in").write_text("all:\n\techo ok\n", encoding="utf-8")

    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["gcc", "make"],
        configure_commands=[],
        clean_commands=["make clean"],
        build_commands=["make -j$(nproc)"],
    )

    normalized = stage._normalize_build_plan(repo, plan)

    assert normalized.clean_commands == ["make clean || true"]
    assert normalized.configure_commands == ["./configure"]


def test_normalize_build_plan_demotes_distclean_when_configure_exists(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "configure").write_text("#!/bin/sh\necho ok\n", encoding="utf-8")

    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="autotools",
        install_packages=["gcc", "make"],
        configure_commands=["./configure"],
        clean_commands=["make distclean"],
        build_commands=["make -j$(nproc)"],
    )

    normalized = stage._normalize_build_plan(repo, plan)

    assert normalized.clean_commands == ["make clean || true"]
    assert "distclean" not in " ".join(normalized.clean_commands)


def test_soften_make_clean_in_script_override(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "Makefile").write_text("all:\n\techo ok\n", encoding="utf-8")

    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["gcc", "make"],
        build_commands=["make -j$(nproc)"],
        build_script_override="#!/bin/bash\nmake clean\nmake -j$(nproc)\n",
    )

    normalized = stage._normalize_build_plan(repo, plan)

    assert "make clean || true" in normalized.build_script_override
    assert re.search(r"(?m)^make clean$", normalized.build_script_override) is None


def test_replan_accepts_clean_or_configure_change_after_missing_make_target():
    stage = build_module.BuildStage()
    previous_plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        clean_commands=["make clean"],
        configure_commands=[],
        build_commands=["make -j$(nproc)"],
    )
    candidate_plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        clean_commands=["make clean || true"],
        configure_commands=["./configure"],
        build_commands=["make -j$(nproc)"],
    )

    assert stage._is_valid_replan_candidate(
        previous_plan,
        candidate_plan,
        failure_kind="container_run",
        failure_logs="make: *** No rule to make target 'clean'.  Stop.",
    )


def test_normalize_build_plan_rewrites_invalid_make_target_in_script_override(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "Makefile").write_text("all:\n\techo ok\n", encoding="utf-8")

    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["gcc", "make"],
        build_commands=["make -j$(nproc)"],
        build_script_override="#!/bin/bash\nmake linux\n",
    )

    normalized = stage._normalize_build_plan(repo, plan)

    assert "make -j$(nproc)" in normalized.build_script_override
    assert "make linux" not in normalized.build_script_override


def test_replan_rejects_unchanged_commands_after_missing_make_target():
    stage = build_module.BuildStage()
    previous_plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["gcc", "make"],
        build_commands=["make linux"],
    )
    candidate_plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["clang", "make"],
        build_commands=["make linux"],
    )

    assert (
        stage._is_valid_replan_candidate(
            previous_plan,
            candidate_plan,
            failure_kind="container_run",
            failure_logs="make: *** No rule to make target 'linux'.  Stop.",
        )
        is False
    )
    candidate_plan.build_commands = ["make -j$(nproc)"]
    assert (
        stage._is_valid_replan_candidate(
            previous_plan,
            candidate_plan,
            failure_kind="container_run",
            failure_logs="make: *** No rule to make target 'linux'.  Stop.",
        )
        is True
    )


def test_normalize_build_plan_injects_libreadline_into_dockerfile_override(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "lua.c").write_text('#include <readline/readline.h>\n', encoding="utf-8")

    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["gcc", "make"],
        build_commands=["make"],
        dockerfile_override=(
            "FROM ubuntu:20.04\n"
            "RUN apt-get update && apt-get install -y --no-install-recommends gcc make && apt-get clean\n"
        ),
    )

    normalized = stage._normalize_build_plan(repo, plan)

    assert "libreadline-dev" in normalized.install_packages
    assert "libreadline-dev" in normalized.dockerfile_override


def test_normalize_build_plan_repairs_dockerfile_override_without_required_packages(tmp_path):
    stage = build_module.BuildStage()
    repo = tmp_path / "repo"
    repo.mkdir()

    plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["gcc", "make"],
        build_commands=["make"],
        dockerfile_override=(
            "FROM ubuntu:20.04\n"
            "RUN apt-get update && apt-get install -y --no-install-recommends gcc make && apt-get clean\n"
        ),
    )

    normalized = stage._normalize_build_plan(repo, plan)

    assert "clang" in normalized.install_packages
    assert "gcc" in normalized.install_packages
    assert "g++" in normalized.install_packages
    assert "git" in normalized.install_packages
    assert "ca-certificates" in normalized.install_packages
    assert "make" in normalized.install_packages
    assert "pkg-config" in normalized.install_packages
    assert "clang" in normalized.dockerfile_override
    assert "gcc" in normalized.dockerfile_override
    assert "g++" in normalized.dockerfile_override
    assert "git" in normalized.dockerfile_override
    assert "ca-certificates" in normalized.dockerfile_override


def test_ensure_dockerfile_clones_project_injects_clone_when_apt_only_override():
    stage = build_module.BuildStage()
    apt_only = (
        "FROM ubuntu:22.04\n"
        "RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y "
        "--no-install-recommends git clang cmake qtbase5-dev\n"
    )
    fixed = stage._ensure_dockerfile_clones_project(
        apt_only,
        repo_url="https://invent.kde.org/frameworks/kimageformats.git",
        vulnerable_ref="c60e77c048d32ccf743cec695743b77b2b25dc87",
        project_dir="/src/kimageformats",
    )
    assert "git clone" in fixed
    assert "https://invent.kde.org/frameworks/kimageformats.git" in fixed
    assert 'git checkout "c60e77c048d32ccf743cec695743b77b2b25dc87"' in fixed
    assert "ENV PROJECT_DIR=/src/kimageformats" in fixed
    assert "COPY artifacts/build" in fixed
    assert "WORKDIR ${PROJECT_DIR}" in fixed
    assert fixed.startswith("FROM ubuntu:22.04")


def test_ensure_dockerfile_clones_project_keeps_existing_clone():
    stage = build_module.BuildStage()
    with_clone = (
        "FROM ubuntu:20.04\n"
        "RUN apt-get update && apt-get install -y git\n"
        'RUN git clone "https://example.com/demo.git" /src/demo && '
        'cd /src/demo && git checkout deadbeef\n'
        "WORKDIR /src/demo\n"
    )
    fixed = stage._ensure_dockerfile_clones_project(
        with_clone,
        repo_url="https://example.com/demo.git",
        vulnerable_ref="deadbeef",
        project_dir="/src/demo",
    )
    assert fixed.count("git clone") == 1
    assert fixed == with_clone if with_clone.endswith("\n") else with_clone + "\n"


def test_default_make_install_packages_include_ca_certificates():
    stage = build_module.BuildStage()
    knowledge = make_knowledge()

    packages = stage._select_install_packages("make", knowledge)

    assert "build-essential" in packages
    assert "clang" in packages
    assert "gcc" in packages
    assert "g++" in packages
    assert "git" in packages
    assert "make" in packages
    assert "pkg-config" in packages
    assert "ca-certificates" in packages


def test_build_stage_uses_localhost_proxy_with_host_network(monkeypatch):
    stage = build_module.BuildStage()

    monkeypatch.setenv("DOCKER_BUILD_PROXY", "http://127.0.0.1:7897")

    assert stage._get_docker_build_proxy() == "http://127.0.0.1:7897"
    assert stage._build_docker_proxy_args("http://127.0.0.1:7897")["HTTP_PROXY"] == "http://127.0.0.1:7897"
    assert stage._select_docker_build_network_mode("http://127.0.0.1:7897") == "host"


def test_build_replan_prompt_includes_rendered_artifacts():
    stage = build_module.BuildStage()
    knowledge = make_knowledge()
    context = build_module.BuildContext(
        cve_id=knowledge.cve_id,
        repo_url=knowledge.repo_url or "",
        previous_failure_kind="container_run",
        previous_build_failure="compiler not found: clang",
        previous_dockerfile_content="FROM ubuntu:20.04\nRUN apt-get install -y git make\n",
        previous_build_script_content="#!/bin/bash\nexport CC=clang\nmake -j$(nproc)\n",
    )
    previous_plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["build-essential", "git", "make"],
        build_commands=["make -j$(nproc)"],
    )

    prompt = stage._build_llm_prompt(
        knowledge=knowledge,
        context=context,
        project_name="demo",
        previous_plan=previous_plan,
    )

    assert "Previously rendered Dockerfile:" in prompt
    assert "Previously rendered build.sh:" in prompt
    assert "compiler not found: clang" in prompt
    assert "dockerfile_override" in prompt
    assert "build_script_override" in prompt
    assert prompt.index("Previous failure logs:") < prompt.index("Patch excerpt:")


def test_build_replan_messages_reuse_single_conversation_history():
    stage = build_module.BuildStage()
    planner = build_module.BuildPlanner(stage)
    knowledge = make_knowledge()
    context = build_module.BuildContext(
        cve_id=knowledge.cve_id,
        repo_url=knowledge.repo_url or "",
        previous_failure_kind="container_run",
        previous_build_failure="compiler not found: clang",
        previous_dockerfile_content="FROM ubuntu:20.04\nRUN apt-get install -y git make\n",
        previous_build_script_content="#!/bin/bash\nexport CC=clang\nmake -j$(nproc)\n",
    )
    previous_plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["build-essential", "git", "make"],
        build_commands=["make -j$(nproc)"],
    )

    messages = planner.build_llm_messages(
        knowledge=knowledge,
        context=context,
        project_name="demo",
        previous_plan=previous_plan,
    )

    assert len(messages) == 4
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert isinstance(messages[2], AIMessage)
    assert isinstance(messages[3], HumanMessage)
    assert "Patch excerpt:" in messages[1].content
    assert "compiler not found: clang" not in messages[1].content
    assert "chosen_vulnerable_ref: deadbeef" in messages[2].content
    assert "Previous failure logs:" in messages[3].content
    assert "compiler not found: clang" in messages[3].content
    assert "Patch excerpt:" not in messages[3].content


def test_build_retry_prompt_truncates_large_failure_artifacts():
    stage = build_module.BuildStage()
    previous_plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["build-essential", "git", "make"],
        build_commands=["make -j$(nproc)"],
    )
    context = build_module.BuildContext(
        cve_id="CVE-2022-0000",
        previous_failure_kind="container_run",
        previous_build_failure=("line\n" * 2000) + "fatal error: missing header\n",
        previous_dockerfile_content="FROM ubuntu:20.04\n" + ("RUN echo hi\n" * 500),
        previous_build_script_content="#!/bin/bash\n" + ("echo hi\n" * 600),
    )

    prompt = stage._build_llm_retry_prompt(context, previous_plan)

    assert "Failure summary:" in prompt
    assert "fatal error: missing header" in prompt
    assert "[truncated " in prompt
    assert len(prompt) < 9000


def test_build_try_llm_plan_retries_timeout_twice_before_success(tmp_path, monkeypatch):
    class FakeResponse:
        def __init__(self, content):
            self.content = content

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("Request timed out.")
            return FakeResponse(
                '{"chosen_vulnerable_ref":"deadbeef","chosen_fixed_ref":null,"build_system":"make",'
                '"install_packages":["build-essential","git","make"],"configure_commands":[],"clean_commands":["make clean || true"],'
                '"build_commands":["make -j$(nproc)"],"expected_binary_path":"demo","dockerfile_override":null,'
                '"build_script_override":null,"source_of_truth":"llm","confidence":"high","rationale":"retry success"}'
            )

    fake_model = FakeModel()
    monkeypatch.setattr(build_module, "build_chat_model", lambda *args, **kwargs: fake_model)

    stage = build_module.BuildStage()
    planner = build_module.BuildPlanner(stage)
    paths = build_module.BuildStagePaths(str(tmp_path / "ws"))
    stage._prepare_workspace(paths)
    stage._active_build_dir = str(paths.build_dir)

    plan = planner.try_llm_plan(
        knowledge=make_knowledge(),
        context=build_module.BuildContext(cve_id="CVE-2022-28805", planner_attempt=4),
        project_name="demo",
    )

    assert plan is not None
    assert fake_model.calls == 3
    assert (paths.llm_dir / "attempt-4" / "response.txt").exists()


def test_build_try_llm_plan_uses_build_agent_timeout_override(tmp_path, monkeypatch):
    class FakeResponse:
        def __init__(self, content):
            self.content = content

    class FakeModel:
        def invoke(self, messages):
            return FakeResponse(
                '{"chosen_vulnerable_ref":"deadbeef","chosen_fixed_ref":null,"build_system":"make",'
                '"install_packages":["build-essential","git","make"],"configure_commands":[],"clean_commands":["make clean || true"],'
                '"build_commands":["make -j$(nproc)"],"expected_binary_path":"demo","dockerfile_override":null,'
                '"build_script_override":null,"source_of_truth":"llm","confidence":"high","rationale":"ok"}'
            )

    captured = {}

    def fake_build_chat_model(agent_name, model_name=None, temperature=0, timeout_seconds=None):
        captured["agent_name"] = agent_name
        captured["timeout_seconds"] = timeout_seconds
        return FakeModel()

    monkeypatch.setattr(build_module, "build_chat_model", fake_build_chat_model)
    monkeypatch.setattr(
        build_module,
        "load_app_config",
        lambda: AppConfig(
            build_agent=AgentModelConfig(model_name="demo", api_key="key"),
            runtime=RuntimeConfig(build_agent_timeout_seconds=77),
        ),
    )

    stage = build_module.BuildStage()
    planner = build_module.BuildPlanner(stage)
    paths = build_module.BuildStagePaths(str(tmp_path / "ws"))
    stage._prepare_workspace(paths)
    stage._active_build_dir = str(paths.build_dir)

    plan = planner.try_llm_plan(
        knowledge=make_knowledge(),
        context=build_module.BuildContext(cve_id="CVE-2022-28805", planner_attempt=6),
        project_name="demo",
    )

    assert plan is not None
    assert captured["agent_name"] == "build_agent"
    assert captured["timeout_seconds"] == 77


def test_build_try_llm_plan_records_final_error_after_three_empty_responses(tmp_path, monkeypatch):
    class FakeResponse:
        def __init__(self, content):
            self.content = content

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            return FakeResponse("   ")

    fake_model = FakeModel()
    monkeypatch.setattr(build_module, "build_chat_model", lambda *args, **kwargs: fake_model)

    stage = build_module.BuildStage()
    planner = build_module.BuildPlanner(stage)
    paths = build_module.BuildStagePaths(str(tmp_path / "ws"))
    stage._prepare_workspace(paths)
    stage._active_build_dir = str(paths.build_dir)

    plan = planner.try_llm_plan(
        knowledge=make_knowledge(),
        context=build_module.BuildContext(cve_id="CVE-2022-28805", planner_attempt=5),
        project_name="demo",
    )

    assert plan is None
    assert fake_model.calls == 3
    error_text = (paths.llm_dir / "attempt-5" / "error.txt").read_text(encoding="utf-8")
    assert "no content after 3 attempts" in error_text


def test_replan_candidate_requires_meaningful_execution_surface_change():
    stage = build_module.BuildStage()
    previous_plan = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["build-essential", "git", "make"],
        build_commands=["make -j$(nproc)"],
    )

    identical_shape = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["build-essential", "git", "make"],
        build_commands=["make -j$(nproc)"],
        rationale="different words only",
    )
    changed_packages = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["build-essential", "clang", "git", "make"],
        build_commands=["make -j$(nproc)"],
    )
    override_only = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["build-essential", "git", "make"],
        build_commands=["make -j$(nproc)"],
        dockerfile_override="FROM ubuntu:20.04\nRUN apt-get update && apt-get install -y clang git make\n",
    )
    script_override_only = build_module.BuildPlan(
        chosen_vulnerable_ref="deadbeef",
        build_system="make",
        install_packages=["build-essential", "git", "make"],
        build_commands=["make -j$(nproc)"],
        build_script_override="#!/bin/bash\nexport CC=gcc\nmake -j$(nproc)\n",
    )

    assert stage._is_valid_replan_candidate(previous_plan, identical_shape) is False
    assert stage._is_valid_replan_candidate(previous_plan, changed_packages) is True
    assert stage._is_valid_replan_candidate(previous_plan, override_only) is True
    assert stage._is_valid_replan_candidate(previous_plan, changed_packages, failure_kind="docker_build") is False
    assert stage._is_valid_replan_candidate(previous_plan, override_only, failure_kind="docker_build") is True
    assert stage._is_valid_replan_candidate(previous_plan, changed_packages, failure_kind="container_run") is True
    assert stage._is_valid_replan_candidate(previous_plan, script_override_only, failure_kind="container_run") is True


def test_build_llm_trace_is_persisted(tmp_path):
    stage = build_module.BuildStage()
    paths = build_module.BuildStagePaths(str(tmp_path / "ws"))
    stage._prepare_workspace(paths)
    stage._active_build_dir = str(paths.build_dir)

    stage._persist_build_llm_trace(2, "prompt.txt", "hello prompt")
    stage._persist_build_llm_trace(2, "parsed.json", json.dumps({"ok": True}))

    prompt_path = paths.llm_dir / "attempt-2" / "prompt.txt"
    parsed_path = paths.llm_dir / "attempt-2" / "parsed.json"

    assert prompt_path.read_text(encoding="utf-8") == "hello prompt\n"
    assert parsed_path.read_text(encoding="utf-8") == '{"ok": true}\n'
