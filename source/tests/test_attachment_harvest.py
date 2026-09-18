"""Tests for GitHub attachment harvesting and PoC payload preference."""

from __future__ import annotations

import base64
from pathlib import Path
import sys
import tempfile
import unittest

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from app.schemas.fetched_page import FetchedPage
from app.stages.knowledge import (
    KnowledgeStage,
    build_knowledge_paths,
    score_reference,
    should_follow_discovered_link,
)
from app.stages.poc import PocPlan, PocStage
from app.tools.poc_attachments import (
    extract_embedded_sexpr_payloads,
    extract_github_attachment_urls,
    harvest_embedded_text_pocs,
    harvest_poc_files,
    infer_run_commands_from_text,
    is_github_attachment_url,
)


class AttachmentUrlTests(unittest.TestCase):
    def test_extract_github_repo_and_user_attachment_urls(self) -> None:
        html = """
        <a href="https://github.com/leesavide/abcm2ps/files/6381411/poc_calculate_beam_357.zip">zip</a>
        see https://github.com/user-attachments/files/12345/crash.bin
        """
        urls = extract_github_attachment_urls(html)
        self.assertEqual(
            urls,
            [
                "https://github.com/leesavide/abcm2ps/files/6381411/poc_calculate_beam_357.zip",
                "https://github.com/user-attachments/files/12345/crash.bin",
            ],
        )
        self.assertTrue(is_github_attachment_url(urls[0]))
        self.assertEqual(score_reference(urls[0]), "P0")

    def test_should_follow_github_attachment_from_issue(self) -> None:
        parent = "https://github.com/leesavide/abcm2ps/issues/83"
        child = "https://github.com/leesavide/abcm2ps/files/6381411/poc_calculate_beam_357.zip"
        self.assertTrue(should_follow_discovered_link(parent, child))


class HarvestAttachmentTests(unittest.TestCase):
    def test_harvest_extracts_zip_member_into_vuln_pocs_and_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extracted = root / "extracted" / "poc_zip"
            extracted.mkdir(parents=True)
            payload = b"X" * 64 + b"\x00\x01\x02crash"
            (extracted / "poc_calculate_beam_357.bin").write_bytes(payload)
            pocs_dir = root / "vuln_pocs"
            evidence = "Run with:\n./abcm2ps -E [poc]\n"
            recipes = harvest_poc_files(
                search_roots=[extracted],
                output_dir=pocs_dir,
                evidence_text=evidence,
                source_url="https://github.com/leesavide/abcm2ps/issues/83",
            )
            self.assertEqual(len(recipes), 1)
            self.assertTrue((pocs_dir / "poc_calculate_beam_357.bin").exists())
            self.assertEqual(recipes[0].confidence, "high")
            self.assertTrue(recipes[0].artifact_generation_commands)
            self.assertIn("base64", recipes[0].artifact_generation_commands[0])
            self.assertTrue(any("-E" in cmd for cmd in recipes[0].run_commands))
            encoded = base64.b64encode(payload).decode("ascii")
            self.assertIn(encoded, recipes[0].artifact_generation_commands[0])

    def test_infer_run_commands_rewrites_poc_placeholder(self) -> None:
        commands = infer_run_commands_from_text("./abcm2ps -E [poc]", "poc.bin")
        self.assertEqual(commands, ["./abcm2ps -E ./poc.bin"])

    def test_knowledge_stage_harvest_attachment_pocs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp)
            paths = build_knowledge_paths("CVE-TEST-ATTACH", dataset_root=str(dataset_root))
            paths.extracted_dir.mkdir(parents=True)
            paths.pocs_dir.mkdir(parents=True)
            payload = b"\x00binary-poc-bytes-0123456789abcdef"
            member = paths.extracted_dir / "poc_overflow.bin"
            member.write_bytes(payload)
            pages = [
                FetchedPage(
                    url="https://github.com/example/project/issues/1",
                    title="issue",
                    html="",
                    cleaned_text="Reproduce:\n./target -E [poc]\n",
                    status_code=200,
                    content_type="text/markdown",
                    local_path="",
                    links=[],
                )
            ]
            recipes = KnowledgeStage().harvest_attachment_pocs(paths=paths, fetched_pages=pages)
            self.assertEqual(len(recipes), 1)
            self.assertTrue((paths.pocs_dir / "poc_overflow.bin").exists())


class EmbeddedTextPocHarvestTests(unittest.TestCase):
    def test_extract_minimized_cil_from_commit_message(self) -> None:
        text = """
Fix use-after-free in map perm.

Here is a minimized CIL policy which reproduces the issue:

(class file (open read write))
(classmap file file)
(classmapping file file (open (self process)))
(optional bad ((allow process self (file (open)))))
"""
        blocks = extract_embedded_sexpr_payloads(text)
        self.assertEqual(len(blocks), 1)
        self.assertIn("(classmap file file)", blocks[0])
        self.assertIn("(optional bad", blocks[0])

    def test_harvest_embedded_text_pocs_writes_vuln_pocs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "vuln_pocs"
            text = (
                "Here is a minimized CIL policy which reproduces the issue:\n"
                "\n"
                "(class file)\n"
                "(classmap file file)\n"
                "(classmapping file file (open (self process)))\n"
            )
            recipes = harvest_embedded_text_pocs(
                output_dir=output_dir,
                evidence_pages=[("https://example.com/commit", "fix commit", text)],
            )
            self.assertEqual(len(recipes), 1)
            self.assertEqual(recipes[0].recipe_type, "embedded_text_payload")
            written = list(output_dir.glob("poc*.cil"))
            self.assertEqual(len(written), 1)
            self.assertIn("(classmap file file)", written[0].read_text(encoding="utf-8"))

    def test_knowledge_stage_harvest_embedded_pocs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = build_knowledge_paths(str(root), "CVE-TEST")
            pages = [
                FetchedPage(
                    url="https://example.com/commit/abc",
                    title="commit",
                    html="",
                    cleaned_text=(
                        "minimized CIL policy:\n"
                        "(class file)\n"
                        "(type process)\n"
                        "(allow process process (file (open)))\n"
                    ),
                    status_code=200,
                    content_type="text/plain",
                    local_path="",
                    links=[],
                )
            ]
            recipes = KnowledgeStage().harvest_embedded_pocs(paths=paths, fetched_pages=pages)
            self.assertEqual(len(recipes), 1)
            self.assertTrue(any(paths.pocs_dir.glob("poc*.cil")))


class PocPayloadPreferenceTests(unittest.TestCase):
    def test_normalize_prefers_dataset_blob_over_invented_payload(self) -> None:
        stage = PocStage()
        payload = b"\x00\x01" + b"A" * 80
        blob = base64.b64encode(payload).decode("ascii")
        plan = PocPlan(
            target_binary="./abcm2ps",
            payload_filename="poc.txt",
            payload_content="X" * 5000,
            run_command="./abcm2ps -E ./poc.txt",
        )
        normalized = stage._normalize_poc_plan(
            plan,
            repo_url="https://github.com/leesavide/abcm2ps.git",
            recipe_base64_blobs=[],
            dataset_poc_base64_blobs=[blob],
            dataset_poc_filenames=["poc_calculate_beam_357.bin"],
        )
        self.assertEqual(normalized.payload_content.encode("latin-1"), payload + b"\n")
        self.assertEqual(normalized.payload_filename, "poc_calculate_beam_357.bin")

    def test_normalize_caps_synthesized_payload_without_authoritative_blob(self) -> None:
        stage = PocStage()
        huge = "A" * (stage.MAX_SYNTHESIZED_PAYLOAD_CHARS + 1000)
        plan = PocPlan(
            target_binary="./target",
            payload_filename="poc.txt",
            payload_content=huge,
            run_command="./target ./poc.txt",
        )
        normalized = stage._normalize_poc_plan(plan, repo_url="https://github.com/example/project.git")
        self.assertLessEqual(len(normalized.payload_content), stage.MAX_SYNTHESIZED_PAYLOAD_CHARS + 1)


if __name__ == "__main__":
    unittest.main()
