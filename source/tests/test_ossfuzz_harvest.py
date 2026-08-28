"""Tests for OSS-Fuzz testcase harvesting."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from app.schemas.fetched_page import FetchedPage
from app.stages.knowledge import KnowledgeStage, build_knowledge_paths, score_reference
from app.tools import ossfuzz


class OssFuzzUrlTests(unittest.TestCase):
    def test_detect_monorail_and_issue_tracker_urls(self) -> None:
        monorail = "https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=32177"
        tracker = "https://issues.oss-fuzz.com/42494623"
        self.assertTrue(ossfuzz.is_ossfuzz_issue_url(monorail))
        self.assertTrue(ossfuzz.is_ossfuzz_issue_url(tracker))
        self.assertEqual(ossfuzz.monorail_id_from_url(monorail), "32177")
        self.assertEqual(ossfuzz.issue_tracker_id_from_url(tracker), "42494623")
        self.assertEqual(score_reference(monorail), "P0")

    def test_extract_testcase_ids_from_escaped_json(self) -> None:
        payload = (
            r'Reproducer Testcase: https://oss-fuzz.com/download?testcase_id\u003d6087260287139840\n'
            r'also testcase_id%3D111\n'
        )
        self.assertEqual(
            ossfuzz.extract_ossfuzz_testcase_ids(payload),
            ["6087260287139840", "111"],
        )

    def test_parse_ossfuzz_harness_name(self) -> None:
        self.assertEqual(
            ossfuzz.parse_ossfuzz_harness_name(
                "clusterfuzz-testcase-minimized-flb-it-fuzz-parser_fuzzer_OSSFUZZ-5216297967288320.fuzz"
            ),
            "flb-it-fuzz-parser_fuzzer",
        )
        self.assertEqual(
            ossfuzz.parse_ossfuzz_harness_name(
                "clusterfuzz-testcase-minimized-secilc-fuzzer-5563841674084352.cil"
            ),
            "secilc-fuzzer",
        )
        self.assertIsNone(ossfuzz.parse_ossfuzz_harness_name("poc.cil"))
        self.assertIsNone(ossfuzz.parse_ossfuzz_harness_name("fluent-bit.conf"))

    def test_harness_source_evidence_requires_in_tree_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # SELinux-like tree: no secilc-fuzzer harness → no evidence.
            (root / "secilc").mkdir()
            (root / "secilc" / "secilc.c").write_text("int main(){return 0;}\n", encoding="utf-8")
            self.assertFalse(ossfuzz.harness_source_evidence(root, "secilc-fuzzer"))

            # fluent-bit-like fuzzer source → evidence + preferred path.
            fuzz_dir = root / "tests" / "internal" / "fuzzers"
            fuzz_dir.mkdir(parents=True)
            (fuzz_dir / "parser_fuzzer.c").write_text("int LLVMFuzzerTestOneInput(){return 0;}\n", encoding="utf-8")
            (fuzz_dir / "CMakeLists.txt").write_text("parser_fuzzer.c\nflb-it-fuzz-parser_fuzzer\n", encoding="utf-8")
            self.assertTrue(ossfuzz.harness_source_evidence(root, "flb-it-fuzz-parser_fuzzer"))
            self.assertEqual(
                ossfuzz.preferred_harness_relpath(root, "flb-it-fuzz-parser_fuzzer"),
                "build/bin/flb-it-fuzz-parser_fuzzer",
            )

    def test_harness_source_evidence_accepts_standalone_ossfuzz_cpp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ossfuzz_dir = root / "ossfuzz"
            ossfuzz_dir.mkdir()
            (ossfuzz_dir / "matio_fuzzer.cpp").write_text(
                "extern \"C\" int LLVMFuzzerTestOneInput(){return 0;}\n",
                encoding="utf-8",
            )
            self.assertTrue(ossfuzz.harness_source_evidence(root, "matio_fuzzer"))
            self.assertEqual(
                ossfuzz.standalone_ossfuzz_harness_relpath(root, "matio_fuzzer"),
                "ossfuzz/matio_fuzzer.cpp",
            )
            self.assertEqual(
                ossfuzz.preferred_harness_relpath(root, "matio_fuzzer"),
                "matio_fuzzer",
            )
            # SELinux-style name still has no evidence in this tree.
            self.assertFalse(ossfuzz.harness_source_evidence(root, "secilc-fuzzer"))


class OssFuzzHarvestTests(unittest.TestCase):
    def test_harvest_writes_vuln_poc_and_recipe(self) -> None:
        payload = (
            b"(optional o1(optional o(classpermission char_w)"
            b"(classmap files(read))(classmapping l a _))"
            b"(classmapping files read char_w))"
        )

        def fake_resolve(issue_url: str, timeout: int = 30) -> str:
            return "42494623"

        def fake_fetch(tracker_id: str, timeout: int = 30) -> str:
            return r'testcase_id\u003d6087260287139840'

        def fake_download(testcase_id: str, output_dir: Path, timeout: int = 60, preferred_name: str = ""):
            output_dir.mkdir(parents=True, exist_ok=True)
            dest = output_dir / f"clusterfuzz-testcase-minimized-secilc-fuzzer-{testcase_id}.cil"
            dest.write_bytes(payload)
            return dest

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "vuln_pocs"
            with patch.object(ossfuzz, "resolve_issue_tracker_id", side_effect=fake_resolve), patch.object(
                ossfuzz, "fetch_issue_action_payload", side_effect=fake_fetch
            ), patch.object(ossfuzz, "download_testcase", side_effect=fake_download):
                recipes = ossfuzz.harvest_ossfuzz_testcases(
                    output_dir=output_dir,
                    issue_urls=["https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=32177"],
                )

            self.assertEqual(len(recipes), 1)
            self.assertEqual(recipes[0].recipe_type, "ossfuzz_testcase")
            written = list(output_dir.glob("*.cil"))
            self.assertEqual(len(written), 1)
            self.assertEqual(written[0].read_bytes(), payload)

    def test_knowledge_stage_harvest_ossfuzz_pocs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = build_knowledge_paths(str(root), "CVE-TEST")
            pages = [
                FetchedPage(
                    url="https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=32177",
                    title="oss-fuzz",
                    html="",
                    cleaned_text="",
                    status_code=200,
                    content_type="text/html",
                    local_path="",
                    links=[],
                )
            ]

            fake_recipe_payload = (
                b"(optional o1(classpermission char_w)"
                b"(classmap files(read))(classmapping files read char_w))"
            )

            def fake_harvest(*, output_dir, issue_urls, timeout=30, limit=4):
                output_dir.mkdir(parents=True, exist_ok=True)
                dest = output_dir / "poc.cil"
                dest.write_bytes(fake_recipe_payload)
                from app.tools.poc_attachments import build_attachment_recipe

                recipe = build_attachment_recipe(
                    filename=dest.name,
                    payload=fake_recipe_payload,
                    source_url=issue_urls[0],
                    source_title=dest.name,
                )
                recipe.recipe_type = "ossfuzz_testcase"
                return [recipe]

            with patch("app.stages.knowledge.harvest_ossfuzz_testcases", side_effect=fake_harvest):
                recipes = KnowledgeStage().harvest_ossfuzz_pocs(paths=paths, fetched_pages=pages)

            self.assertEqual(len(recipes), 1)
            self.assertTrue((paths.pocs_dir / "poc.cil").exists())


if __name__ == "__main__":
    unittest.main()
