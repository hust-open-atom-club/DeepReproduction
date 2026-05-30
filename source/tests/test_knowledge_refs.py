"""Regression tests for Git ref inference in the knowledge stage."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from app.schemas.fetched_page import FetchedPage
from app.schemas.knowledge import KnowledgeModel
from app.schemas.task import TaskModel
from app.stages.knowledge import (
    KnowledgeSourcesModel,
    KnowledgeStage,
    build_knowledge_paths,
    ReferenceRecord,
    derive_reference_variants,
    extract_reproduction_recipes,
    extract_summary_candidate,
    heuristic_summary_from_pages,
    infer_git_refs,
    infer_vulnerability_type,
    infer_repo_url,
    osv_has_commit_reference,
    sanitize_filename,
    should_follow_discovered_link,
)
from app.tools.patch_tools import PatchSummary


class InferGitRefsTests(unittest.TestCase):
    def test_vulnerable_ref_uses_fixed_commit_parent(self) -> None:
        osv_payload = {
            "references": [
                {"url": "https://github.com/lua/lua/commit/1f3c6f4534c6411313361697d98d1145a1f030fa"},
            ],
            "affected": [
                {
                    "ranges": [
                        {
                            "type": "GIT",
                            "events": [
                                {"introduced": "c33b1728aeb7dfeec4013562660e07d32697aa6b"},
                                {"fixed": "1f3c6f4534c6411313361697d98d1145a1f030fa"},
                            ],
                        }
                    ]
                }
            ],
        }

        with patch(
            "app.stages.knowledge.fetch_github_parent_ref",
            return_value="25b143dd34fb587d1e35290c4b25bc08954800e2",
        ) as mocked_parent_lookup:
            vulnerable_ref, fixed_ref = infer_git_refs(
                osv_payload,
                fallback_fixed=None,
                fallback_vulnerable=None,
                repo_url="https://github.com/lua/lua.git",
            )

        self.assertEqual(fixed_ref, "1f3c6f4534c6411313361697d98d1145a1f030fa")
        self.assertEqual(vulnerable_ref, "25b143dd34fb587d1e35290c4b25bc08954800e2")
        mocked_parent_lookup.assert_called_once_with(
            "https://github.com/lua/lua.git",
            "1f3c6f4534c6411313361697d98d1145a1f030fa",
        )

    def test_vulnerable_ref_falls_back_when_parent_lookup_fails(self) -> None:
        osv_payload = {
            "references": [],
            "affected": [
                {
                    "ranges": [
                        {
                            "type": "GIT",
                            "events": [
                                {"fixed": "fixed-sha"},
                            ],
                        }
                    ]
                }
            ],
        }

        with patch("app.stages.knowledge.fetch_github_parent_ref", return_value=None):
            vulnerable_ref, fixed_ref = infer_git_refs(
                osv_payload,
                fallback_fixed=None,
                fallback_vulnerable="existing-vulnerable",
                repo_url="https://github.com/example/project.git",
            )

        self.assertEqual(fixed_ref, "fixed-sha")
        self.assertEqual(vulnerable_ref, "existing-vulnerable")

    def test_summary_prefers_advisory_description_over_commit_page_chrome(self) -> None:
        commit_page = FetchedPage(
            url="https://github.com/example/project/commit/abc123",
            title="Fix bug",
            html="",
            cleaned_text="Fix bug\nNavigation Menu\nToggle navigation\nPlatform\nSecurity\nPricing",
            status_code=200,
            content_type="text/html",
            local_path=None,
            links=[],
        )
        advisory_page = FetchedPage(
            url="https://github.com/example/project/security/advisories/GHSA-xxxx-yyyy-zzzz",
            title="Advisory",
            html="",
            cleaned_text=(
                "Advisory\n\nDescription\nNodes can publish ATXs which reference the incorrect previous "
                "ATX of the smesher, breaking the expected chain validation rule.\n\nImpact\n"
                "Attackers can abuse the stale reference to gain rewards."
            ),
            status_code=200,
            content_type="text/html",
            local_path=None,
            links=[],
        )

        summary = heuristic_summary_from_pages([commit_page, advisory_page])
        self.assertIn("reference the incorrect previous ATX", summary)
        self.assertEqual(infer_vulnerability_type(summary), "improper-input-validation")

    def test_summary_candidate_extracts_description_section(self) -> None:
        page = FetchedPage(
            url="https://nvd.nist.gov/vuln/detail/CVE-2024-34360",
            title="NVD",
            html="",
            cleaned_text=(
                "NVD - CVE-2024-34360\n\nDescription\n"
                "go-spacemesh allows an incorrect previous ATX reference, breaking protocol validation.\n\n"
                "Metrics\nCVSS 3.1"
            ),
            status_code=200,
            content_type="text/html",
            local_path=None,
            links=[],
        )

        self.assertEqual(
            extract_summary_candidate(page),
            "go-spacemesh allows an incorrect previous ATX reference, breaking protocol validation.",
        )

    def test_summary_candidate_falls_back_to_clean_title_for_navigation_pages(self) -> None:
        page = FetchedPage(
            url="https://github.com/google/oss-fuzz-vulns/blob/main/vulns/libredwg/OSV-2021-814.yaml",
            title="oss-fuzz-vulns/vulns/libredwg/OSV-2021-814.yaml at main · google/oss-fuzz-vulns · GitHub",
            html="",
            cleaned_text=(
                "oss-fuzz-vulns/vulns/libredwg/OSV-2021-814.yaml at main · google/oss-fuzz-vulns · GitHub\n"
                "Navigation Menu\nToggle navigation\nAppearance settings\nSearch or jump to...\nSaved searches"
            ),
            status_code=200,
            content_type="text/html",
            local_path=None,
            links=[],
        )

        self.assertEqual(
            extract_summary_candidate(page),
            "oss-fuzz-vulns/vulns/libredwg/OSV-2021-814.yaml at main · google/oss-fuzz-vulns",
        )

    def test_run_uses_existing_local_evidence_when_remote_fetch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            paths = build_knowledge_paths("CVE-LOCAL", dataset_root=str(dataset_root))
            paths.cleaned_dir.mkdir(parents=True, exist_ok=True)
            paths.diff_dir.mkdir(parents=True, exist_ok=True)
            paths.cleaned_dir.joinpath("advisory.md").write_text(
                "# Advisory\n\nSource: https://example.com/advisory\n\n"
                "How to reproduce:\n1. ./demo poc\n\n"
                "Heap overflow happens when parsing crafted input.\n",
                encoding="utf-8",
            )
            paths.patch_diff.write_text(
                "diff --git a/demo.c b/demo.c\n"
                "--- a/demo.c\n"
                "+++ b/demo.c\n"
                "@@ -1,1 +1,2 @@\n"
                "+ assert(ptr != NULL);\n",
                encoding="utf-8",
            )

            stage = KnowledgeStage()
            task = TaskModel(
                task_id="CVE-LOCAL",
                cve_id="CVE-LOCAL",
                repo_url="https://example.com/demo.git",
                vulnerable_ref="old",
                fixed_ref="new",
                references=["https://example.com/advisory"],
                reference_details=[],
            )

            with patch.object(KnowledgeStage, "bootstrap_task", return_value=(task, None)), \
                 patch.object(KnowledgeStage, "prioritize_references", return_value=([ReferenceRecord(url="https://example.com/advisory", selected=True)], [])), \
                 patch.object(KnowledgeStage, "collect_evidence", return_value=([], [], [], [ReferenceRecord(url="https://example.com/advisory", selected=True)], [])), \
                 patch.object(KnowledgeStage, "extract_and_write_poc", return_value=None):
                knowledge = stage.run("CVE-LOCAL", dataset_root=str(dataset_root))

            self.assertTrue(knowledge.summary)
            self.assertTrue(knowledge.reproduction_hints)
            self.assertIn("demo.c", knowledge.affected_files)

    def test_run_falls_back_to_bootstrap_metadata_when_no_evidence_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            stage = KnowledgeStage()
            task = TaskModel(
                task_id="CVE-MIN",
                cve_id="CVE-MIN",
                cve_url="https://example.com/cve/CVE-MIN",
                repo_url="https://example.com/demo.git",
                vulnerable_ref="old",
                fixed_ref="new",
                references=["https://example.com/advisory"],
                reference_details=[],
            )

            with patch.object(KnowledgeStage, "bootstrap_task", return_value=(task, None)), \
                 patch.object(KnowledgeStage, "prioritize_references", return_value=([ReferenceRecord(url="https://example.com/advisory", selected=True)], [])), \
                 patch.object(KnowledgeStage, "collect_evidence", return_value=([], [], [], [ReferenceRecord(url="https://example.com/advisory", selected=True)], [])), \
                 patch.object(KnowledgeStage, "extract_and_write_poc", return_value=None):
                knowledge = stage.run("CVE-MIN", dataset_root=str(dataset_root))

            self.assertEqual(knowledge.repo_url, "https://example.com/demo.git")
            self.assertEqual(knowledge.vulnerable_ref, "old")
            self.assertEqual(knowledge.fixed_ref, "new")
            self.assertIn("https://example.com/advisory", knowledge.references)

    def test_recursive_crawl_rejects_github_navigation_pages(self) -> None:
        parent = "https://github.com/example/project/commit/abc123"
        self.assertFalse(
            should_follow_discovered_link(parent, "https://github.com/security/advanced-security")
        )
        self.assertFalse(
            should_follow_discovered_link(parent, "https://github.com/login?return_to=%2Fexample%2Fproject")
        )
        self.assertTrue(
            should_follow_discovered_link(parent, "https://github.com/example/project/pull/42")
        )
        self.assertFalse(
            should_follow_discovered_link(parent, "https://github.com/example/project/commits/main/src")
        )
        self.assertFalse(
            should_follow_discovered_link(parent, "https://invent.kde.org/frameworks/kimageformats/-/commit/abc123?view=parallel")
        )

    def test_llm_result_backfills_empty_heuristic_fields(self) -> None:
        task = TaskModel(
            task_id="CVE-TEST",
            cve_id="CVE-TEST",
            repo_url="https://github.com/example/project.git",
            vulnerable_ref="old-ref",
            fixed_ref="new-ref",
            references=[],
            reference_details=[],
        )
        source_registry = KnowledgeSourcesModel(
            cve_id="CVE-TEST",
            selected_references=[ReferenceRecord(url="https://example.com/advisory")],
        )
        fetched_pages = [
            FetchedPage(
                url="https://nvd.nist.gov/vuln/detail/CVE-TEST",
                title="NVD",
                html="",
                cleaned_text=(
                    "Description\nThe service does not validate the previous record correctly, "
                    "allowing a stale reference to bypass protocol checks."
                ),
                status_code=200,
                content_type="text/html",
                local_path=None,
                links=[],
            )
        ]
        patch_summaries = [
            PatchSummary(
                affected_files=["src/validator.go"],
                changed_functions=["func validatePreviousRecord() error"],
                summary="Patch touches 1 file(s).",
            )
        ]

        stage = KnowledgeStage()
        with patch.object(
            stage,
            "_try_llm_synthesis",
            return_value=KnowledgeModel(
                cve_id="CVE-TEST",
                summary="LLM summary",
                vulnerability_type="improper-input-validation",
                repo_url="",
                vulnerable_ref="",
                fixed_ref="",
                affected_files=[],
                reproduction_hints=[],
                expected_error_patterns=[],
                expected_stack_keywords=[],
                references=[],
            ),
        ):
            knowledge = stage.synthesize_knowledge(task, source_registry, fetched_pages, patch_summaries)

        self.assertEqual(knowledge.summary, "LLM summary")
        self.assertEqual(knowledge.affected_files, ["src/validator.go"])
        self.assertTrue(knowledge.reproduction_hints)
        self.assertEqual(knowledge.reproduction_recipes, [])
        self.assertEqual(knowledge.expected_stack_keywords, ["func validatePreviousRecord() error"])
        self.assertEqual(knowledge.references, ["https://example.com/advisory"])

    def test_synthesize_knowledge_limits_output_references(self) -> None:
        task = TaskModel(
            task_id="CVE-TEST",
            cve_id="CVE-TEST",
            references=[],
            reference_details=[],
        )
        source_registry = KnowledgeSourcesModel(
            cve_id="CVE-TEST",
            selected_references=[
                ReferenceRecord(url="https://example.com/one"),
                ReferenceRecord(url="https://example.com/two"),
                ReferenceRecord(url="https://example.com/three"),
            ],
        )
        fetched_pages = [
            FetchedPage(
                url="https://nvd.nist.gov/vuln/detail/CVE-TEST",
                title="NVD",
                html="",
                cleaned_text="Description\nA vulnerability description that is long enough to be used as summary.",
                status_code=200,
                content_type="text/html",
                local_path=None,
                links=[],
            )
        ]

        stage = KnowledgeStage()
        stage.max_output_references = 2
        with patch.object(stage, "_try_llm_synthesis", return_value=None):
            knowledge = stage.synthesize_knowledge(task, source_registry, fetched_pages, [])

        self.assertEqual(knowledge.references, ["https://example.com/one", "https://example.com/two"])

    def test_prioritize_references_applies_selected_reference_cap(self) -> None:
        stage = KnowledgeStage()
        stage.max_selected_references = 2

        selected, skipped = stage.prioritize_references(
            [
                "https://github.com/example/project/security/advisories/GHSA-0000-0000-0001",
                "https://github.com/example/project/pull/1",
                "https://nvd.nist.gov/vuln/detail/CVE-TEST-0001",
            ],
        )

        self.assertEqual(len(selected), 2)
        self.assertTrue(any("selected reference cap" in record.note for record in skipped))

    def test_build_information_is_extracted_from_pages_and_patch(self) -> None:
        task = TaskModel(
            task_id="CVE-BUILD",
            cve_id="CVE-BUILD",
            language="Go",
            references=[],
            reference_details=[],
        )
        source_registry = KnowledgeSourcesModel(
            cve_id="CVE-BUILD",
            selected_references=[ReferenceRecord(url="https://example.com/build-docs")],
        )
        fetched_pages = [
            FetchedPage(
                url="https://example.com/docs/build",
                title="Build docs",
                html="",
                cleaned_text=(
                    "Description\nBuild instructions\n\n"
                    "Makefile\n"
                    "go.mod\n"
                    "$ apt-get install -y build-essential pkg-config\n"
                    "$ go build ./cmd/server\n"
                    "$ make release\n"
                ),
                status_code=200,
                content_type="text/html",
                local_path=None,
                links=[],
            )
        ]
        patch_summaries = [
            PatchSummary(
                affected_files=["Makefile", "go.mod", "cmd/server/main.go"],
                changed_functions=[],
                summary="Patch touches 3 file(s).",
            )
        ]

        stage = KnowledgeStage()
        with patch.object(stage, "_try_llm_synthesis", return_value=None):
            knowledge = stage.synthesize_knowledge(task, source_registry, fetched_pages, patch_summaries)

        self.assertEqual(knowledge.build_files, ["Makefile", "go.mod"])
        self.assertEqual(knowledge.build_systems, ["make", "go"])
        self.assertIn("apt-get install -y build-essential pkg-config", knowledge.install_commands)
        self.assertIn("go build ./cmd/server", knowledge.build_commands)
        self.assertIn("make release", knowledge.build_commands)
        self.assertTrue(knowledge.build_hints)

    def test_reproduction_recipe_preserves_full_shell_sequence(self) -> None:
        fetched_pages = [
            FetchedPage(
                url="https://osv.dev/reference",
                title="OSV reference",
                html="",
                cleaned_text=(
                    "How to reproduce:\n"
                    "1. git clone https://github.com/lua/lua -q\n"
                    "2. cd lua/ && make -j$(nproc)\n"
                    '3. echo -n "bG9jYWwgdSxfLE4sXyx3LE4sZCxXCmZ1bmN0aW9uIGMoRSxMLGwsUyxULHUsTSxULGwsaCxoLHUsdSxsLGgsaCx1LHUsTSx1LHUsdSxsLGgsaCxsKXM9cyBsb2NhbAllLGUsXyxfLE4sZSxzMCxOLFYsXyBmdW5jdGlvbiBjKGIsbClpLHM9TiBsb2NhbCBjIGxvY2FsIF9FTlY8Y29uc3Q+ID0wIG89MCBmdW5jdGlvbiBlKCllbmQ7ZSIicmV0dXJuIGVuZDtlMCxhLHcscyxzLHM9IiJyZXR1cm4jYyIiZW5kO2VlPSIicmV0dXJuDGMiIg==" | base64 -d > poc\n'
                    "4. ./lua ./poc\n"
                ),
                status_code=200,
                content_type="text/html",
                local_path=None,
                links=[],
            )
        ]

        recipes = extract_reproduction_recipes(fetched_pages)

        self.assertEqual(len(recipes), 1)
        self.assertEqual(
            recipes[0].steps,
            [
                "git clone https://github.com/lua/lua -q",
                "cd lua/ && make -j$(nproc)",
                'echo -n "bG9jYWwgdSxfLE4sXyx3LE4sZCxXCmZ1bmN0aW9uIGMoRSxMLGwsUyxULHUsTSxULGwsaCxoLHUsdSxsLGgsaCx1LHUsTSx1LHUsdSxsLGgsaCxsKXM9cyBsb2NhbAllLGUsXyxfLE4sZSxzMCxOLFYsXyBmdW5jdGlvbiBjKGIsbClpLHM9TiBsb2NhbCBjIGxvY2FsIF9FTlY8Y29uc3Q+ID0wIG89MCBmdW5jdGlvbiBlKCllbmQ7ZSIicmV0dXJuIGVuZDtlMCxhLHcscyxzLHM9IiJyZXR1cm4jYyIiZW5kO2VlPSIicmV0dXJuDGMiIg==" | base64 -d > poc',
                "./lua ./poc",
            ],
        )
        self.assertIn("git clone https://github.com/lua/lua -q", recipes[0].repo_setup_commands)
        self.assertIn("cd lua/ && make -j$(nproc)", recipes[0].build_commands)
        self.assertIn(
            'echo -n "bG9jYWwgdSxfLE4sXyx3LE4sZCxXCmZ1bmN0aW9uIGMoRSxMLGwsUyxULHUsTSxULGwsaCxoLHUsdSxsLGgsaCx1LHUsTSx1LHUsdSxsLGgsaCxsKXM9cyBsb2NhbAllLGUsXyxfLE4sZSxzMCxOLFYsXyBmdW5jdGlvbiBjKGIsbClpLHM9TiBsb2NhbCBjIGxvY2FsIF9FTlY8Y29uc3Q+ID0wIG89MCBmdW5jdGlvbiBlKCllbmQ7ZSIicmV0dXJuIGVuZDtlMCxhLHcscyxzLHM9IiJyZXR1cm4jYyIiZW5kO2VlPSIicmV0dXJuDGMiIg==" | base64 -d > poc',
            recipes[0].artifact_generation_commands,
        )
        self.assertIn("./lua ./poc", recipes[0].run_commands)
        self.assertIn('base64 -d > poc', recipes[0].source_excerpt)

    def test_reproduction_recipe_ignores_preamble_noise_before_heading_commands(self) -> None:
        fetched_pages = [
            FetchedPage(
                url="https://lua-users.org/lists/lua-l/2022-02/msg00001.html",
                title="lua-l report",
                html="",
                cleaned_text=(
                    "Hi,\n"
                    "I found a heap a heap buffer overflow on read in luaH_getshortstr function.\n"
                    "Lua version:\n"
                    "Lua 5.4.4 Copyright (C) 1994-2022 Lua.org, PUC-Rio\n"
                    "How to reprocude:\n"
                    "----------\n"
                    "1. git clone https://github.com/lua/lua -q\n"
                    "2. cd lua/ && make -j$(nproc)\n"
                    '3. echo -n "AAAA" | base64 -d > poc\n'
                    "4. ./lua ./poc\n"
                    "----------\n"
                    "Tested on Ubuntu-20.04 x86_64.\n"
                ),
                status_code=200,
                content_type="text/html",
                local_path=None,
                links=[],
            )
        ]

        recipes = extract_reproduction_recipes(fetched_pages)

        self.assertEqual(len(recipes), 1)
        self.assertEqual(
            recipes[0].steps,
            [
                "git clone https://github.com/lua/lua -q",
                "cd lua/ && make -j$(nproc)",
                'echo -n "AAAA" | base64 -d > poc',
                "./lua ./poc",
            ],
        )
        self.assertEqual(recipes[0].run_commands, ["./lua ./poc"])

    def test_synthesize_knowledge_backfills_reproduction_recipes_from_heuristics(self) -> None:
        task = TaskModel(
            task_id="CVE-LUA",
            cve_id="CVE-LUA",
            repo_url="https://github.com/lua/lua.git",
            references=[],
            reference_details=[],
        )
        source_registry = KnowledgeSourcesModel(
            cve_id="CVE-LUA",
            selected_references=[ReferenceRecord(url="https://osv.dev/reference")],
        )
        fetched_pages = [
            FetchedPage(
                url="https://osv.dev/reference",
                title="OSV reference",
                html="",
                cleaned_text=(
                    "How to reproduce:\n"
                    "1. git clone https://github.com/lua/lua -q\n"
                    "2. cd lua/ && make -j$(nproc)\n"
                    '3. echo -n "AAAA" | base64 -d > poc\n'
                    "4. ./lua ./poc\n"
                ),
                status_code=200,
                content_type="text/html",
                local_path=None,
                links=[],
            )
        ]

        stage = KnowledgeStage()
        with patch.object(
            stage,
            "_try_llm_synthesis",
            return_value=KnowledgeModel(
                cve_id="CVE-LUA",
                summary="Lua issue",
                vulnerability_type="memory-corruption",
                repo_url="",
                vulnerable_ref="",
                fixed_ref="",
                affected_files=[],
                reproduction_hints=[],
                reproduction_recipes=[],
                expected_error_patterns=[],
                expected_stack_keywords=[],
                references=[],
            ),
        ):
            knowledge = stage.synthesize_knowledge(task, source_registry, fetched_pages, [])

        self.assertEqual(len(knowledge.reproduction_recipes), 1)
        self.assertEqual(knowledge.reproduction_recipes[0].steps[-1], "./lua ./poc")
        self.assertTrue(knowledge.reproduction_hints)

    def test_sanitize_filename_stays_short_for_long_urls(self) -> None:
        value = "https://invent.kde.org/frameworks/kimageformats/-/commit/297ed9a2fe339bfe36916b9fce628c3242e5be0f?action=show&controller=projects%2Fcommit&id=297ed9a2fe339bfe36916b9fce628c3242e5be0f"
        sanitized = sanitize_filename(value)
        self.assertLessEqual(len(sanitized), 109)
        self.assertRegex(sanitized, r"_[0-9a-f]{12}$")

    def test_blob_url_derives_raw_variant(self) -> None:
        variants = derive_reference_variants(
            "https://github.com/google/oss-fuzz-vulns/blob/main/vulns/libredwg/OSV-2021-814.yaml"
        )
        self.assertIn(
            "https://raw.githubusercontent.com/google/oss-fuzz-vulns/main/vulns/libredwg/OSV-2021-814.yaml",
            variants,
        )

    def test_gitlab_commit_derives_diff_variant(self) -> None:
        variants = derive_reference_variants(
            "https://invent.kde.org/frameworks/kimageformats/-/commit/297ed9a2fe339bfe36916b9fce628c3242e5be0f"
        )
        self.assertIn(
            "https://invent.kde.org/frameworks/kimageformats/-/commit/297ed9a2fe339bfe36916b9fce628c3242e5be0f.diff",
            variants,
        )

    def test_osv_without_commit_reference_is_out_of_scope(self) -> None:
        osv_payload = {
            "references": [
                {"type": "REPORT", "url": "https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=34766"},
                {"type": "EVIDENCE", "url": "https://github.com/google/oss-fuzz-vulns/blob/main/vulns/libredwg/OSV-2021-814.yaml"},
            ]
        }
        self.assertFalse(osv_has_commit_reference(osv_payload))

    def test_merge_osv_into_task_backfills_language_from_repo_api(self) -> None:
        task = TaskModel(
            task_id="CVE-TEST",
            cve_id="CVE-TEST",
            references=[],
            reference_details=[],
        )
        osv_payload = {
            "references": [
                {"type": "FIX", "url": "https://github.com/lua/lua/commit/1f3c6f4534c6411313361697d98d1145a1f030fa"},
            ],
            "affected": [{}],
        }
        stage = KnowledgeStage()

        with patch("app.stages.knowledge.fetch_repo_primary_language", return_value="C"):
            merged = stage._merge_osv_into_task(task, osv_payload)

        self.assertEqual(infer_repo_url(osv_payload), "https://github.com/lua/lua.git")
        self.assertEqual(merged.language, "C")


if __name__ == "__main__":
    unittest.main()
