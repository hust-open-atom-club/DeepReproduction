"""Standalone knowledge agent implementation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import build_chat_model, load_app_config
from app.schemas.fetched_page import FetchedPage
from app.schemas.knowledge import KnowledgeModel, ReproductionRecipe
from app.schemas.task import TaskModel, TaskReference
from app.tools.archive_tools import ArchiveTool
from app.tools.content_cleaner import ContentCleaner
from app.tools.patch_tools import PatchSummary, PatchTool, should_replace_patch_diff
from app.tools.poc_attachments import (
    extract_github_attachment_urls,
    harvest_embedded_text_pocs,
    harvest_poc_files,
    is_github_attachment_url,
)
from app.tools.ossfuzz import (
    collect_ossfuzz_issue_urls,
    harvest_ossfuzz_testcases,
    is_ossfuzz_issue_url,
)
from app.tools.reference_extractor import ReferenceExtractor
from app.tools.web_fetch import WebFetchTool

class ReferenceRecord(BaseModel):
    """Single reference tracked by the knowledge stage."""

    url: str = Field(..., description="Reference URL.")
    source_type: str = Field(default="reference", description="High-level source type.")
    priority: str = Field(default="P2", description="Priority level assigned by heuristics.")
    depth: int = Field(default=0, description="Traversal depth.")
    selected: bool = Field(default=True, description="Whether the reference was kept.")
    note: str = Field(default="", description="Short decision note.")
    reference_kind: str = Field(default="", description="Original reference type such as FIX or EVIDENCE.")


class KnowledgeSourcesModel(BaseModel):
    """Registry of sources considered by the knowledge stage."""

    cve_id: str = Field(..., description="CVE identifier.")
    osv_url: str = Field(default="", description="OSV endpoint used for bootstrap.")
    seed_references: List[ReferenceRecord] = Field(default_factory=list, description="Initial references.")
    selected_references: List[ReferenceRecord] = Field(default_factory=list, description="References kept for evidence collection.")
    skipped_references: List[ReferenceRecord] = Field(default_factory=list, description="References dropped during filtering.")
    local_evidence_files: List[str] = Field(default_factory=list, description="Local evidence files created by this stage.")


class ExtractedPoc(BaseModel):
    """Structured PoC candidate extracted from evidence."""

    has_poc: bool = Field(default=False, description="Whether the evidence contains an explicit PoC or reproducer.")
    filename: str = Field(default="poc.txt", description="Suggested PoC filename.")
    content: str = Field(default="", description="PoC or reproduction script content.")
    rationale: str = Field(default="", description="Short explanation for the extraction decision.")


class ExtractedRecipeBundle(BaseModel):
    """Structured reproduction recipes extracted from evidence."""

    recipes: List[ReproductionRecipe] = Field(default_factory=list, description="Recovered reproduction recipes.")


@dataclass(frozen=True)
class KnowledgePaths:
    """Filesystem layout owned by the knowledge stage."""

    cve_root: Path
    yaml_dir: Path
    evidence_dir: Path
    raw_dir: Path
    cleaned_dir: Path
    extracted_dir: Path
    diff_dir: Path
    pocs_dir: Path
    task_yaml: Path
    knowledge_yaml: Path
    runtime_state_yaml: Path
    knowledge_sources_yaml: Path
    patch_diff: Path


def build_knowledge_paths(cve_id: str, dataset_root: str = "Dataset") -> KnowledgePaths:
    """Resolve the dataset layout for the standalone knowledge stage."""

    cve_root = Path(dataset_root) / cve_id
    yaml_dir = cve_root / "vuln_yaml"
    evidence_dir = cve_root / "vuln_data" / "knowledge_sources"
    diff_dir = cve_root / "vuln_data" / "vuln_diffs"
    pocs_dir = cve_root / "vuln_data" / "vuln_pocs"

    return KnowledgePaths(
        cve_root=cve_root,
        yaml_dir=yaml_dir,
        evidence_dir=evidence_dir,
        raw_dir=evidence_dir / "raw",
        cleaned_dir=evidence_dir / "cleaned",
        extracted_dir=evidence_dir / "extracted",
        diff_dir=diff_dir,
        pocs_dir=pocs_dir,
        task_yaml=yaml_dir / "task.yaml",
        knowledge_yaml=yaml_dir / "knowledge.yaml",
        runtime_state_yaml=yaml_dir / "runtime_state.yaml",
        knowledge_sources_yaml=yaml_dir / "knowledge_sources.yaml",
        patch_diff=diff_dir / "patch.diff",
    )


class KnowledgeStage:
    """Knowledge-stage agent."""

    _USER_AGENT = "DeepReproductionKnowledgeAgent/1.0"

    def __init__(self) -> None:
        self.reference_extractor = ReferenceExtractor()
        self.fetcher = WebFetchTool()
        self.cleaner = ContentCleaner()
        self.archives = ArchiveTool()
        self.patches = PatchTool()
        runtime = load_app_config().runtime
        self.max_reference_depth = runtime.knowledge_max_reference_depth
        self.max_fetch_count = runtime.knowledge_max_fetch_count
        self.max_selected_references = runtime.knowledge_max_selected_references
        self.max_discovered_references_per_page = runtime.knowledge_max_discovered_references_per_page
        self.max_output_references = runtime.knowledge_max_output_references
        self.fetch_timeout_seconds = runtime.knowledge_fetch_timeout_seconds
        self.enable_llm_curation = runtime.knowledge_enable_llm_curation
        self.last_llm_status = "disabled"
        self.last_llm_error: Optional[str] = None

    def run(self, cve_id: str, dataset_root: str = "Dataset") -> KnowledgeModel:
        """Run the full standalone knowledge-stage workflow."""

        paths = build_knowledge_paths(cve_id=cve_id, dataset_root=dataset_root)
        prepare_layout(paths)
        curated_knowledge = self._load_curated_knowledge(paths.knowledge_yaml)

        try:
            task, bootstrap_error = self.bootstrap_task(cve_id=cve_id, paths=paths)
            reset_stage_outputs(paths)
            write_yaml(paths.task_yaml, task.model_dump(mode="json"))

            selected_references, skipped_references = self.prioritize_references(
                task.references,
                task.reference_details,
            )
            (
                fetched_pages,
                local_evidence_files,
                patch_summaries,
                selected_references,
                skipped_references,
            ) = self.collect_evidence(
                selected_references,
                skipped_references,
                paths,
                fixed_ref=task.fixed_ref,
                preferred_files=list(curated_knowledge.affected_files) if curated_knowledge else [],
            )
            existing_pages, existing_evidence_files, existing_patch_summaries = self.load_existing_evidence(paths)
            if existing_evidence_files:
                local_evidence_files = dedupe_preserve_order([*local_evidence_files, *existing_evidence_files])
            if not fetched_pages and existing_pages:
                fetched_pages = existing_pages
            if not patch_summaries and existing_patch_summaries:
                patch_summaries = existing_patch_summaries
            ensure_empty_file(paths.patch_diff)

            source_registry = KnowledgeSourcesModel(
                cve_id=task.cve_id,
                osv_url=task.cve_url or "",
                seed_references=[
                    ReferenceRecord(
                        url=url,
                        source_type=guess_source_type(url),
                        priority=score_reference(url, reference_type=reference_type_for_url(task.reference_details, url)),
                        note="Seed reference from task input or OSV bootstrap.",
                        reference_kind=reference_type_for_url(task.reference_details, url) or "",
                    )
                    for url in dedupe_preserve_order(task.references)
                ],
                selected_references=selected_references,
                skipped_references=skipped_references,
                local_evidence_files=local_evidence_files,
            )
            write_yaml(paths.knowledge_sources_yaml, source_registry.model_dump(mode="json"))

            if bootstrap_error and not task.references:
                raise RuntimeError(f"Failed to bootstrap references from OSV: {bootstrap_error}")

            if not fetched_pages and not patch_summaries and not local_evidence_files:
                bootstrap_page = self._build_bootstrap_fallback_page(task, source_registry.selected_references)
                if bootstrap_page is not None:
                    fetched_pages = [bootstrap_page]
                elif selected_references:
                    failed_urls = ", ".join(record.url for record in selected_references[:5])
                    raise RuntimeError(f"Failed to fetch any evidence from selected references. First targets: {failed_urls}")
                else:
                    raise RuntimeError("Knowledge stage produced no references and no evidence.")

            knowledge = self.synthesize_knowledge(
                task=task,
                source_registry=source_registry,
                fetched_pages=fetched_pages,
                patch_summaries=patch_summaries,
            )
            attachment_recipes = self.harvest_attachment_pocs(paths=paths, fetched_pages=fetched_pages)
            embedded_recipes = self.harvest_embedded_pocs(paths=paths, fetched_pages=fetched_pages)
            ossfuzz_recipes = self.harvest_ossfuzz_pocs(
                paths=paths,
                fetched_pages=fetched_pages,
                selected_references=source_registry.selected_references,
            )
            harvested_recipes = [*ossfuzz_recipes, *attachment_recipes, *embedded_recipes]
            if harvested_recipes:
                knowledge.reproduction_recipes = [
                    *harvested_recipes,
                    *knowledge.reproduction_recipes,
                ]
                knowledge.reproduction_hints = dedupe_preserve_order(
                    [
                        *[_harvest_hint_for_recipe(recipe) for recipe in harvested_recipes],
                        *knowledge.reproduction_hints,
                    ]
                )
                if not knowledge.expected_error_patterns:
                    knowledge.expected_error_patterns = ["AddressSanitizer", "SEGV"]
            knowledge = self._merge_curated_knowledge(knowledge, curated_knowledge)
            write_yaml(paths.knowledge_yaml, knowledge.model_dump(mode="json"))
            self.extract_and_write_poc(
                task=task,
                fetched_pages=fetched_pages,
                patch_summaries=patch_summaries,
                paths=paths,
            )
            write_yaml(
                paths.runtime_state_yaml,
                build_runtime_state_payload(
                    task_id=task.cve_id,
                    success=True,
                    message="Knowledge stage completed successfully.",
                    llm_status=self.last_llm_status,
                    llm_error=self.last_llm_error,
                ),
            )
            return knowledge
        except Exception as error:
            write_yaml(
                paths.runtime_state_yaml,
                build_runtime_state_payload(
                    task_id=cve_id,
                    success=False,
                    message=str(error),
                    llm_status=self.last_llm_status,
                    llm_error=self.last_llm_error,
                ),
            )
            raise

    def _load_curated_knowledge(self, knowledge_yaml: Path) -> Optional[KnowledgeModel]:
        """Load a previously curated knowledge.yaml before the stage clears outputs."""

        if not knowledge_yaml.exists():
            return None
        try:
            payload = yaml.safe_load(knowledge_yaml.read_text(encoding="utf-8")) or {}
            return KnowledgeModel(**payload)
        except Exception:
            return None

    def _merge_curated_knowledge(
        self,
        knowledge: KnowledgeModel,
        curated: Optional[KnowledgeModel],
    ) -> KnowledgeModel:
        """Prefer curated reproduction recipes/hints when they contain concrete artifacts."""

        if curated is None:
            return knowledge

        curated_recipes = [
            recipe
            for recipe in curated.reproduction_recipes
            if recipe.artifact_generation_commands or recipe.run_commands or recipe.steps
        ]
        if curated_recipes:
            knowledge.reproduction_recipes = curated_recipes
        if curated.reproduction_hints:
            knowledge.reproduction_hints = dedupe_preserve_order(
                [*curated.reproduction_hints, *knowledge.reproduction_hints]
            )
        if curated.expected_error_patterns:
            knowledge.expected_error_patterns = dedupe_preserve_order(
                [*curated.expected_error_patterns, *knowledge.expected_error_patterns]
            )
        if curated.expected_stack_keywords:
            knowledge.expected_stack_keywords = dedupe_preserve_order(
                [*curated.expected_stack_keywords, *knowledge.expected_stack_keywords]
            )
        if curated.build_commands and not knowledge.build_commands:
            knowledge.build_commands = list(curated.build_commands)
        if curated.build_systems and not knowledge.build_systems:
            knowledge.build_systems = list(curated.build_systems)
        if curated.build_files and not knowledge.build_files:
            knowledge.build_files = list(curated.build_files)
        if curated.build_hints:
            knowledge.build_hints = dedupe_preserve_order([*curated.build_hints, *knowledge.build_hints])
        return knowledge

    def bootstrap_task(self, cve_id: str, paths: KnowledgePaths) -> tuple[TaskModel, Optional[str]]:
        """Create or refresh a task model from local YAML and OSV metadata."""

        base_task = self._load_existing_task(paths.task_yaml, cve_id)
        try:
            osv_payload = self._fetch_osv(cve_id)
        except Exception as error:
            return base_task, str(error)
        if not osv_payload:
            return base_task, "OSV returned an empty payload."
        if not osv_has_commit_reference(osv_payload):
            raise RuntimeError("OSV references did not include a commit URL; stopping because this CVE is outside the supported scope.")
        return self._merge_osv_into_task(base_task, osv_payload), None

    def prioritize_references(
        self,
        references: Iterable[str],
        reference_details: Optional[Iterable[TaskReference]] = None,
    ) -> tuple[List[ReferenceRecord], List[ReferenceRecord]]:
        """Prioritize references and produce a bounded crawl queue."""

        selected: List[ReferenceRecord] = []
        skipped: List[ReferenceRecord] = []
        reference_type_map = build_reference_type_map(reference_details or [])

        normalized = self.reference_extractor.normalize(references)
        filtered = set(self.reference_extractor.filter_relevant(normalized))

        for url in normalized:
            reference_type = reference_type_map.get(url)
            if url not in filtered:
                skipped.append(
                    ReferenceRecord(
                        url=url,
                        source_type=guess_source_type(url),
                        priority="P3",
                        selected=False,
                        note="Dropped by domain filter.",
                        reference_kind=reference_type or "",
                    )
                )
                continue

            priority = score_reference(url, reference_type=reference_type)
            record = ReferenceRecord(
                url=url,
                source_type=guess_source_type(url),
                priority=priority,
                depth=0,
                selected=priority != "P3",
                note="Selected by deterministic URL heuristics." if priority != "P3" else "Skipped by deterministic URL heuristics.",
                reference_kind=reference_type or "",
            )
            if record.selected:
                selected.append(record)
                for derived in derive_reference_variants(url):
                    selected.append(
                        ReferenceRecord(
                            url=derived,
                            source_type=guess_source_type(derived),
                            priority="P0",
                            depth=0,
                            selected=True,
                            note="Derived patch-style variant from selected reference.",
                            reference_kind=reference_type or "",
                        )
                    )
            else:
                skipped.append(record)

        dedup_selected = dedupe_reference_records(selected)
        selected_urls = {item.url for item in dedup_selected}
        dedup_skipped = [record for record in dedupe_reference_records(skipped) if record.url not in selected_urls]
        capped_selected, overflow_skipped = truncate_reference_records(
            dedup_selected,
            self.max_selected_references,
            "Dropped because it exceeded the selected reference cap.",
        )
        capped_urls = {item.url for item in capped_selected}
        dedup_skipped = [record for record in dedup_skipped if record.url not in capped_urls]
        return capped_selected, dedupe_reference_records([*dedup_skipped, *overflow_skipped])

    def collect_evidence(
        self,
        selected_references: List[ReferenceRecord],
        skipped_references: List[ReferenceRecord],
        paths: KnowledgePaths,
        fixed_ref: Optional[str] = None,
        preferred_files: Optional[List[str]] = None,
    ) -> tuple[List[FetchedPage], List[str], List[PatchSummary], List[ReferenceRecord], List[ReferenceRecord]]:
        """Fetch selected references, recursively expand links, and persist evidence files."""

        fetched_pages: list[FetchedPage] = []
        evidence_files: list[str] = []
        patch_summaries: list[PatchSummary] = []
        preferred_files = list(preferred_files or [])

        selected_by_url = {record.url: record for record in selected_references}
        skipped_by_url = {record.url: record for record in skipped_references}
        queue: list[ReferenceRecord] = list(selected_references)
        fetched_urls: set[str] = set()

        while queue and len(fetched_urls) < self.max_fetch_count:
            record = queue.pop(0)
            if record.url in fetched_urls:
                continue
            fetched_urls.add(record.url)

            try:
                fetched = self.fetcher.fetch_one(
                    record.url,
                    download_dir=str(paths.raw_dir),
                    timeout=self.fetch_timeout_seconds,
                )
            except Exception as error:
                record.note = f"{record.note} Fetch failed: {error}"
                continue

            fetched_pages.append(fetched)

            if fetched.local_path:
                evidence_files.append(fetched.local_path)
                if self.archives.is_supported_archive(fetched.local_path):
                    extracted_dir = paths.extracted_dir / sanitize_filename(Path(fetched.local_path).stem)
                    extraction = self.archives.extract(fetched.local_path, str(extracted_dir))
                    evidence_files.extend(extraction.extracted_files)
                    continue

            if fetched.html:
                if looks_like_patch(record.url, fetched.content_type, fetched.html):
                    patch_summary = self.patches.parse_diff(fetched.html)
                    patch_summaries.append(patch_summary)
                    if record.url.lower().endswith(".diff") or record.url.lower().endswith(".patch"):
                        existing = (
                            paths.patch_diff.read_text(encoding="utf-8")
                            if paths.patch_diff.exists()
                            else ""
                        )
                        if should_replace_patch_diff(
                            existing,
                            fetched.html,
                            candidate_url=record.url,
                            fixed_ref=fixed_ref,
                            preferred_files=preferred_files or patch_summary.affected_files,
                        ):
                            paths.patch_diff.write_text(fetched.html, encoding="utf-8")
                            evidence_files.append(str(paths.patch_diff))
                        else:
                            patch_file = paths.raw_dir / f"{sanitize_filename(record.url)}.patch"
                            patch_file.write_text(fetched.html, encoding="utf-8")
                            evidence_files.append(str(patch_file))
                    else:
                        patch_file = paths.raw_dir / f"{sanitize_filename(record.url)}.patch"
                        patch_file.write_text(fetched.html, encoding="utf-8")
                        evidence_files.append(str(patch_file))
                else:
                    cleaned = self.cleaner.clean_html(fetched.html, source_url=fetched.url) if fetched.content_type == "text/html" else self.cleaner.clean_markdown(fetched.html, source_url=fetched.url)
                    cleaned = self.cleaner.trim_for_prompt(cleaned, max_chars=12000)
                    fetched.cleaned_text = cleaned.cleaned_text
                    if cleaned.title and not fetched.title:
                        fetched.title = cleaned.title
                    cleaned_path = paths.cleaned_dir / f"{sanitize_filename(record.url)}.md"
                    cleaned_path.write_text(render_cleaned_markdown(fetched.url, cleaned.title, cleaned.cleaned_text), encoding="utf-8")
                    evidence_files.append(str(cleaned_path))

                    attachment_urls = extract_github_attachment_urls(
                        fetched.html or "",
                        cleaned.cleaned_text,
                        "\n".join(fetched.links or []),
                    )
                    for attachment_url in attachment_urls:
                        if attachment_url in fetched_urls or attachment_url in selected_by_url:
                            continue
                        child_record = ReferenceRecord(
                            url=attachment_url,
                            source_type="attachment",
                            priority="P0",
                            depth=record.depth + 1,
                            selected=True,
                            note=f"GitHub issue/PR file attachment discovered from {record.url}.",
                            reference_kind="EVIDENCE",
                        )
                        selected_by_url[attachment_url] = child_record
                        queue.insert(0, child_record)

                    discovered_selected, discovered_skipped = self.discover_child_references(
                        parent=record,
                        child_links=fetched.links,
                        selected_by_url=selected_by_url,
                        skipped_by_url=skipped_by_url,
                    )
                    for child_record in discovered_selected:
                        selected_by_url[child_record.url] = child_record
                        if child_record.url not in fetched_urls:
                            queue.append(child_record)
                    for child_record in discovered_skipped:
                        if child_record.url not in selected_by_url:
                            skipped_by_url[child_record.url] = child_record

        return (
            fetched_pages,
            dedupe_preserve_order(evidence_files),
            patch_summaries,
            list(selected_by_url.values()),
            list(skipped_by_url.values()),
        )

    def load_existing_evidence(self, paths: KnowledgePaths) -> tuple[List[FetchedPage], List[str], List[PatchSummary]]:
        """Load curated local evidence from the dataset when remote fetching is unavailable."""

        fetched_pages: list[FetchedPage] = []
        evidence_files: list[str] = []
        patch_summaries: list[PatchSummary] = []

        for cleaned_path in sorted(paths.cleaned_dir.glob("*.md")):
            content = cleaned_path.read_text(encoding="utf-8", errors="replace")
            title = ""
            url = cleaned_path.resolve().as_uri()
            body = content
            lines = content.splitlines()
            if lines and lines[0].startswith("# "):
                title = lines[0][2:].strip()
                body = "\n".join(lines[1:]).lstrip()
            source_match = re.search(r"^Source:\s*(\S+)\s*$", content, re.MULTILINE)
            if source_match:
                url = source_match.group(1).strip()
                body = content[source_match.end():].lstrip()
            fetched_pages.append(
                FetchedPage(
                    url=url,
                    title=title,
                    html="",
                    cleaned_text=body.strip(),
                    status_code=200,
                    content_type="text/markdown",
                    local_path=str(cleaned_path),
                    links=[],
                )
            )
            evidence_files.append(str(cleaned_path))

        if paths.patch_diff.exists():
            diff_text = paths.patch_diff.read_text(encoding="utf-8", errors="replace")
            if diff_text.strip():
                patch_summaries.append(self.patches.parse_diff(diff_text))
                evidence_files.append(str(paths.patch_diff))

        for directory in (paths.raw_dir, paths.extracted_dir, paths.pocs_dir):
            if not directory.exists():
                continue
            for file_path in sorted(path for path in directory.rglob("*") if path.is_file()):
                evidence_files.append(str(file_path))

        return fetched_pages, dedupe_preserve_order(evidence_files), patch_summaries

    def _build_bootstrap_fallback_page(
        self,
        task: TaskModel,
        selected_references: List[ReferenceRecord],
    ) -> Optional[FetchedPage]:
        """Build a minimal synthetic evidence page from bootstrap task metadata."""

        reference_urls = [record.url for record in selected_references[:8] if record.url]
        lines = [
            f"CVE: {task.cve_id}",
            f"Repository: {task.repo_url or ''}",
            f"Vulnerable ref: {task.vulnerable_ref or ''}",
            f"Fixed ref: {task.fixed_ref or ''}",
        ]
        if reference_urls:
            lines.append("References:")
            lines.extend(reference_urls)
        cleaned_text = "\n".join(line for line in lines if line.strip()).strip()
        if not cleaned_text:
            return None
        return FetchedPage(
            url=task.cve_url or (reference_urls[0] if reference_urls else f"task:{task.cve_id}"),
            title=f"Bootstrap metadata for {task.cve_id}",
            html="",
            cleaned_text=cleaned_text,
            status_code=200,
            content_type="text/plain",
            local_path=None,
            links=[],
        )

    def discover_child_references(
        self,
        parent: ReferenceRecord,
        child_links: Iterable[str],
        selected_by_url: dict[str, ReferenceRecord],
        skipped_by_url: dict[str, ReferenceRecord],
    ) -> tuple[List[ReferenceRecord], List[ReferenceRecord]]:
        """Discover and classify child references from a fetched page."""

        if parent.depth >= self.max_reference_depth:
            return [], []

        discovered_selected: list[ReferenceRecord] = []
        discovered_skipped: list[ReferenceRecord] = []

        normalized_links = self.reference_extractor.normalize(child_links)
        filtered_links = self.reference_extractor.filter_relevant(normalized_links)

        for url in filtered_links:
            if url in selected_by_url or url in skipped_by_url:
                continue
            if not should_follow_discovered_link(parent.url, url):
                discovered_skipped.append(
                    ReferenceRecord(
                        url=url,
                        source_type=guess_source_type(url),
                        priority="P3",
                        depth=parent.depth + 1,
                        selected=False,
                        note=f"Discovered from {parent.url} but filtered by recursive crawl rules.",
                    )
                )
                continue

            priority = score_reference(url)
            if priority == "P3":
                discovered_skipped.append(
                    ReferenceRecord(
                        url=url,
                        source_type=guess_source_type(url),
                        priority=priority,
                        depth=parent.depth + 1,
                        selected=False,
                        note=f"Discovered from {parent.url} but scored too low for recursive crawl.",
                    )
                )
                continue

            record = ReferenceRecord(
                url=url,
                source_type=guess_source_type(url),
                priority=priority,
                depth=parent.depth + 1,
                selected=True,
                note=f"Discovered recursively from {parent.url}.",
            )
            discovered_selected.append(record)

            for derived in derive_reference_variants(url):
                if derived in selected_by_url or derived in skipped_by_url:
                    continue
                discovered_selected.append(
                    ReferenceRecord(
                        url=derived,
                        source_type=guess_source_type(derived),
                        priority="P0",
                        depth=parent.depth + 1,
                        selected=True,
                        note=f"Derived patch-style variant from recursively discovered URL {url}.",
                    )
                )

        dedup_selected = dedupe_reference_records(discovered_selected)
        dedup_skipped = dedupe_reference_records(discovered_skipped)
        capped_selected, overflow_skipped = truncate_reference_records(
            dedup_selected,
            self.max_discovered_references_per_page,
            "Dropped because it exceeded the per-page discovered reference cap.",
        )
        capped_urls = {item.url for item in capped_selected}
        dedup_skipped = [record for record in dedup_skipped if record.url not in capped_urls]
        return capped_selected, dedupe_reference_records([*dedup_skipped, *overflow_skipped])

    def synthesize_knowledge(
        self,
        task: TaskModel,
        source_registry: KnowledgeSourcesModel,
        fetched_pages: List[FetchedPage],
        patch_summaries: List[PatchSummary],
    ) -> KnowledgeModel:
        """Synthesize the final `knowledge.yaml` record."""

        summary_source = heuristic_summary_from_pages(fetched_pages)
        heuristic_vuln_type = infer_vulnerability_type(summary_source)
        heuristic_affected_files = dedupe_preserve_order(item for patch in patch_summaries for item in patch.affected_files)
        heuristic_build_files = extract_build_files(fetched_pages, patch_summaries)
        heuristic_build_systems = infer_build_systems(heuristic_build_files, task.language)
        heuristic_install_commands = extract_install_commands(fetched_pages)
        heuristic_build_commands = extract_build_commands(fetched_pages)
        heuristic_build_hints = build_build_hints(
            build_files=heuristic_build_files,
            build_systems=heuristic_build_systems,
            install_commands=heuristic_install_commands,
            build_commands=heuristic_build_commands,
            patch_summaries=patch_summaries,
        )
        heuristic_reproduction_recipes = extract_reproduction_recipes(fetched_pages)
        heuristic_reproduction_hints = build_reproduction_hints(task, fetched_pages, patch_summaries)
        heuristic_expected_stack_keywords = extract_stack_keywords(patch_summaries)
        selected_reference_urls = limit_output_urls(
            [record.url for record in source_registry.selected_references],
            self.max_output_references,
        )
        llm_result = self._try_llm_synthesis(task, fetched_pages, patch_summaries)
        if llm_result is not None:
            llm_result.summary = llm_result.summary or summary_source
            llm_result.vulnerability_type = llm_result.vulnerability_type or heuristic_vuln_type
            llm_result.references = selected_reference_urls
            llm_result.repo_url = llm_result.repo_url or task.repo_url
            llm_result.vulnerable_ref = llm_result.vulnerable_ref or task.vulnerable_ref
            llm_result.fixed_ref = llm_result.fixed_ref or task.fixed_ref
            llm_result.affected_files = llm_result.affected_files or heuristic_affected_files
            llm_result.build_files = llm_result.build_files or heuristic_build_files
            llm_result.build_systems = llm_result.build_systems or heuristic_build_systems
            llm_result.install_commands = llm_result.install_commands or heuristic_install_commands
            llm_result.build_commands = llm_result.build_commands or heuristic_build_commands
            llm_result.build_hints = llm_result.build_hints or heuristic_build_hints
            llm_result.reproduction_recipes = llm_result.reproduction_recipes or heuristic_reproduction_recipes
            llm_result.reproduction_hints = llm_result.reproduction_hints or heuristic_reproduction_hints
            llm_result.expected_stack_keywords = llm_result.expected_stack_keywords or heuristic_expected_stack_keywords
            llm_result.expected_error_patterns = llm_result.expected_error_patterns or default_error_patterns(
                llm_result.vulnerability_type or heuristic_vuln_type
            )
            return llm_result

        return KnowledgeModel(
            cve_id=task.cve_id,
            summary=summary_source,
            vulnerability_type=heuristic_vuln_type,
            repo_url=task.repo_url,
            vulnerable_ref=task.vulnerable_ref,
            fixed_ref=task.fixed_ref,
            affected_files=heuristic_affected_files,
            build_files=heuristic_build_files,
            build_systems=heuristic_build_systems,
            install_commands=heuristic_install_commands,
            build_commands=heuristic_build_commands,
            build_hints=heuristic_build_hints,
            reproduction_hints=heuristic_reproduction_hints,
            reproduction_recipes=heuristic_reproduction_recipes,
            expected_error_patterns=default_error_patterns(heuristic_vuln_type),
            expected_stack_keywords=heuristic_expected_stack_keywords,
            references=selected_reference_urls,
        )

    def _try_llm_synthesis(
        self,
        task: TaskModel,
        fetched_pages: List[FetchedPage],
        patch_summaries: List[PatchSummary],
    ) -> Optional[KnowledgeModel]:
        """Try JSON-based LLM synthesis using plain text output."""

        if not self.enable_llm_curation:
            self.last_llm_status = "disabled"
            self.last_llm_error = None
            return None

        evidence_blocks: list[str] = []
        for page in fetched_pages[:8]:
            if not page.cleaned_text:
                continue
            evidence_blocks.append("\n".join([f"URL: {page.url}", f"Title: {page.title}", "Content:", page.cleaned_text[:4000]]))

        for patch in patch_summaries[:5]:
            evidence_blocks.append(
                "\n".join(
                    [
                        "Patch summary:",
                        patch.summary,
                        f"Affected files: {', '.join(patch.affected_files)}",
                        f"Changed functions: {', '.join(patch.changed_functions)}",
                    ]
                )
            )

        reproduction_blocks = build_reproduction_evidence_blocks(fetched_pages)
        if reproduction_blocks:
            evidence_blocks.append("Reproduction evidence:\n" + "\n\n---\n\n".join(reproduction_blocks[:4]))

        if not evidence_blocks:
            self.last_llm_status = "skipped_no_evidence"
            self.last_llm_error = None
            return None

        prompt = "\n\n".join(
            [
                "You are a security research assistant.",
                "Extract a compact structured knowledge record for the CVE.",
                "Only include facts supported by the evidence.",
                "Return exactly one JSON object and no markdown fences.",
                "Use this schema:",
                json.dumps(
                    {
                        "cve_id": "string",
                        "summary": "string",
                        "vulnerability_type": "string",
                        "repo_url": "string or null",
                        "vulnerable_ref": "string or null",
                        "fixed_ref": "string or null",
                        "affected_files": ["string"],
                        "build_systems": ["string"],
                        "build_files": ["string"],
                        "install_commands": ["string"],
                        "build_commands": ["string"],
                        "build_hints": ["string"],
                        "reproduction_hints": ["string"],
                        "reproduction_recipes": [
                            {
                                "source_url": "string",
                                "source_title": "string",
                                "recipe_type": "string",
                                "steps": ["string"],
                                "repo_setup_commands": ["string"],
                                "build_commands": ["string"],
                                "artifact_generation_commands": ["string"],
                                "run_commands": ["string"],
                                "expected_behavior": ["string"],
                                "source_excerpt": "string",
                                "confidence": "string",
                            }
                        ],
                        "expected_error_patterns": ["string"],
                        "expected_stack_keywords": ["string"],
                        "references": ["string"],
                    },
                    ensure_ascii=True,
                ),
                "If the evidence shows a reproducer or numbered shell steps, preserve the sequence in reproduction_recipes.steps.",
                "Keep source_excerpt grounded in the original evidence instead of paraphrasing away the command details.",
                f"CVE: {task.cve_id}",
                f"Repository: {task.repo_url or ''}",
                f"Vulnerable ref: {task.vulnerable_ref or ''}",
                f"Fixed ref: {task.fixed_ref or ''}",
                "Evidence:",
                "\n\n---\n\n".join(evidence_blocks),
            ]
        )

        try:
            model = build_chat_model("knowledge_agent", temperature=0)
            response = model.invoke(
                [
                    SystemMessage(content="You return strict JSON only."),
                    HumanMessage(content=prompt),
                ]
            )
            content = getattr(response, "content", response)
            parsed = parse_llm_json_payload(content)
            if parsed is None:
                self.last_llm_status = "unexpected_response"
                self.last_llm_error = "LLM response was not valid JSON."
                return None
            self.last_llm_status = "success"
            self.last_llm_error = None
            return KnowledgeModel(**parsed)
        except Exception as error:
            self.last_llm_status = "failed"
            self.last_llm_error = str(error)
            return None
        return None

    def _load_existing_task(self, task_yaml: Path, cve_id: str) -> TaskModel:
        if task_yaml.exists():
            payload = read_yaml(task_yaml)
            details = payload.get("reference_details") or []
            if not details and payload.get("references"):
                payload["reference_details"] = [{"url": url, "type": None} for url in payload.get("references", [])]
            return TaskModel(**payload)

        return TaskModel(
            task_id=cve_id,
            cve_id=cve_id,
            cve_url=f"https://api.osv.dev/v1/vulns/{cve_id}",
            references=[],
            reference_details=[],
        )

    def _fetch_osv(self, cve_id: str) -> dict:
        url = f"https://api.osv.dev/v1/vulns/{cve_id}"
        request = Request(url, headers={"User-Agent": self._USER_AGENT})
        with urlopen(request, timeout=20) as response:
            raw = response.read().decode("utf-8", errors="replace")
        return json.loads(raw)

    def _enrich_osv_with_related_advisories(self, osv_payload: dict) -> dict:
        """Merge GIT ranges from linked OSS-Fuzz OSV advisories (e.g. OSV-2021-440)."""

        related_ids = _related_osv_ids_from_payload(osv_payload)
        if not related_ids:
            return osv_payload
        enriched = dict(osv_payload)
        affected = list(enriched.get("affected") or [])
        for related_id in related_ids:
            try:
                related = self._fetch_osv(related_id)
            except Exception:
                continue
            for item in related.get("affected") or []:
                if item:
                    affected.append(item)
        enriched["affected"] = affected
        return enriched

    def _merge_osv_into_task(self, task: TaskModel, osv_payload: dict) -> TaskModel:
        references = list(task.references)
        reference_details = [TaskReference(**item.model_dump(mode="json")) for item in task.reference_details]
        for item in osv_payload.get("references", []):
            url = item.get("url")
            if url:
                references.append(url)
                reference_details.append(TaskReference(url=url, type=item.get("type")))

        task_data = task.model_dump(mode="json")
        task_data["cve_url"] = task.cve_url or f"https://api.osv.dev/v1/vulns/{task.cve_id}"
        task_data["references"] = dedupe_preserve_order(references)
        task_data["reference_details"] = dedupe_task_references(reference_details)

        osv_payload = self._enrich_osv_with_related_advisories(osv_payload)

        repo_url = task.repo_url or ""
        recovering_from_metadata = bool(repo_url) and _is_metadata_repo_url(repo_url)
        if not repo_url or recovering_from_metadata:
            inferred_repo = infer_repo_url(osv_payload)
            if inferred_repo:
                repo_url = inferred_repo
        language_seed = None if recovering_from_metadata else task.language
        language = language_seed or infer_language(osv_payload) or fetch_repo_primary_language(repo_url)
        # Drop stale refs that only belong to metadata mirrors (even when repo_url
        # was already corrected to the product repository in a previous run).
        fallback_fixed = task.fixed_ref
        fallback_vulnerable = task.vulnerable_ref
        stale_fixed = recovering_from_metadata or _ref_incompatible_with_repo(
            osv_payload, fallback_fixed, repo_url
        )
        stale_vulnerable = recovering_from_metadata or _ref_incompatible_with_repo(
            osv_payload, fallback_vulnerable, repo_url
        )
        if stale_fixed:
            fallback_fixed = None
            # Parent-of-metadata SHAs often are absent from OSV; drop with fixed_ref.
            stale_vulnerable = True
        if stale_vulnerable:
            fallback_vulnerable = None
        vulnerable_ref, fixed_ref = infer_git_refs(
            osv_payload,
            fallback_fixed=fallback_fixed,
            fallback_vulnerable=fallback_vulnerable,
            repo_url=repo_url,
        )

        if repo_url:
            task_data["repo_url"] = repo_url
        if language:
            task_data["language"] = language
        if vulnerable_ref:
            task_data["vulnerable_ref"] = vulnerable_ref
        elif stale_vulnerable:
            task_data["vulnerable_ref"] = None
        if fixed_ref:
            task_data["fixed_ref"] = fixed_ref
        elif stale_fixed:
            task_data["fixed_ref"] = None

        return TaskModel(**task_data)

    def harvest_attachment_pocs(
        self,
        paths: KnowledgePaths,
        fetched_pages: List[FetchedPage],
    ) -> List[ReproductionRecipe]:
        """Copy downloaded/extracted attachment payloads into vuln_pocs and build recipes."""

        evidence_text = "\n".join(page.cleaned_text or "" for page in fetched_pages if page.cleaned_text)
        source_url = next((page.url for page in fetched_pages if is_github_attachment_url(page.url)), "")
        if not source_url:
            source_url = next((page.url for page in fetched_pages if page.url), "")
        return harvest_poc_files(
            search_roots=[paths.extracted_dir, paths.raw_dir, paths.pocs_dir],
            output_dir=paths.pocs_dir,
            evidence_text=evidence_text,
            source_url=source_url,
            limit=4,
        )

    def harvest_embedded_pocs(
        self,
        paths: KnowledgePaths,
        fetched_pages: List[FetchedPage],
    ) -> List[ReproductionRecipe]:
        """Harvest minimized CIL/policy PoCs embedded in commit messages or advisories."""

        evidence_pages = [
            (page.url or "", page.title or "", page.cleaned_text or "")
            for page in fetched_pages
            if page.cleaned_text
        ]
        return harvest_embedded_text_pocs(
            output_dir=paths.pocs_dir,
            evidence_pages=evidence_pages,
            limit=4,
        )

    def harvest_ossfuzz_pocs(
        self,
        paths: KnowledgePaths,
        fetched_pages: List[FetchedPage],
        selected_references: Optional[List[ReferenceRecord]] = None,
    ) -> List[ReproductionRecipe]:
        """Download public OSS-Fuzz reproducer testcases into vuln_pocs."""

        reference_urls = [record.url for record in (selected_references or []) if record.url]
        page_texts = [
            "\n".join(filter(None, [page.url, page.html, page.cleaned_text, "\n".join(page.links or [])]))
            for page in fetched_pages
        ]
        issue_urls = collect_ossfuzz_issue_urls(reference_urls=reference_urls, page_texts=page_texts)
        if not issue_urls:
            return []
        return harvest_ossfuzz_testcases(
            output_dir=paths.pocs_dir,
            issue_urls=issue_urls,
            timeout=self.fetch_timeout_seconds,
            limit=4,
        )

    def extract_and_write_poc(
        self,
        task: TaskModel,
        fetched_pages: List[FetchedPage],
        patch_summaries: List[PatchSummary],
        paths: KnowledgePaths,
    ) -> Optional[Path]:
        """Use the model to detect and persist an explicit PoC when present."""

        if not self.enable_llm_curation:
            return None

        poc_candidate = self._try_llm_poc_extraction(task, fetched_pages, patch_summaries)
        if poc_candidate is None or not poc_candidate.has_poc or not poc_candidate.content.strip():
            return None

        filename = poc_candidate.filename.strip() or "poc.txt"
        filename = sanitize_filename(filename)
        if "." not in filename:
            filename = f"{filename}.txt"
        poc_path = paths.pocs_dir / filename
        poc_path.write_text(poc_candidate.content.rstrip() + "\n", encoding="utf-8")
        return poc_path

    def _try_llm_poc_extraction(
        self,
        task: TaskModel,
        fetched_pages: List[FetchedPage],
        patch_summaries: List[PatchSummary],
    ) -> Optional[ExtractedPoc]:
        """Ask the model whether the evidence contains an explicit PoC."""

        evidence_blocks: list[str] = []
        for page in fetched_pages[:10]:
            if not page.cleaned_text:
                continue
            text = page.cleaned_text
            lowered = text.lower()
            if any(token in lowered for token in ("poc", "proof of concept", "reproduce", "reproducer", "payload", "pcall(", "_env", "```", "assert(", "base64")):
                evidence_blocks.append("\n".join([f"URL: {page.url}", f"Title: {page.title}", "Content:", text[:5000]]))

        if not evidence_blocks and patch_summaries:
            for patch in patch_summaries[:3]:
                evidence_blocks.append(
                    "\n".join(
                        [
                            "Patch summary:",
                            patch.summary,
                            f"Affected files: {', '.join(patch.affected_files)}",
                            f"Changed functions: {', '.join(patch.changed_functions)}",
                        ]
                    )
                )

        if not evidence_blocks:
            return None

        prompt = "\n\n".join(
            [
                "You are a security research assistant.",
                "Determine whether the evidence contains an explicit proof-of-concept, reproducer, runnable script, shell command sequence, or directly reconstructable payload.",
                "Only return has_poc=true when the evidence is explicit enough to save as a file.",
                "If no explicit PoC is present, return has_poc=false and empty content.",
                "Return exactly one JSON object and no markdown fences.",
                "Use this schema:",
                json.dumps(
                    {
                        "has_poc": "boolean",
                        "filename": "string",
                        "content": "string",
                        "rationale": "string",
                    },
                    ensure_ascii=True,
                ),
                f"CVE: {task.cve_id}",
                "Evidence:",
                "\n\n---\n\n".join(evidence_blocks),
            ]
        )

        try:
            model = build_chat_model("knowledge_agent", temperature=0)
            response = model.invoke(
                [
                    SystemMessage(content="You return strict JSON only."),
                    HumanMessage(content=prompt),
                ]
            )
            content = getattr(response, "content", response)
            parsed = parse_llm_json_payload(content)
            if parsed is None:
                return None
            return ExtractedPoc(**parsed)
        except Exception:
            return None


def run_knowledge_agent(cve_id: str, dataset_root: str = "Dataset") -> KnowledgeModel:
    """Public function called directly by the top-level controller."""

    return KnowledgeStage().run(cve_id=cve_id, dataset_root=dataset_root)


def knowledge_node(state):
    """LangGraph v1 节点：执行 knowledge 阶段。"""

    task = state["task"]
    dataset_root = state.get("dataset_root", "Dataset")
    history = list(state.get("stage_history", []))
    stage_status = dict(state.get("stage_status", {}))
    artifacts = dict(state.get("artifacts", {}))
    paths = build_knowledge_paths(task.cve_id, dataset_root=dataset_root)

    try:
        knowledge = run_knowledge_agent(cve_id=task.cve_id, dataset_root=dataset_root)
        history.append({"stage": "knowledge", "status": "success"})
        stage_status["knowledge"] = "success"
        artifacts["knowledge"] = {
            "task_yaml": str(paths.task_yaml),
            "knowledge_yaml": str(paths.knowledge_yaml),
            "runtime_state_yaml": str(paths.runtime_state_yaml),
            "knowledge_sources_yaml": str(paths.knowledge_sources_yaml),
            "patch_diff": str(paths.patch_diff),
        }
        return {
            "knowledge": knowledge,
            "current_stage": "build",
            "review_stage": "",
            "human_action_required": False,
            "review_reason": "",
            "stage_history": history,
            "stage_status": stage_status,
            "artifacts": artifacts,
            "last_error": None,
        }
    except Exception as error:
        history.append({"stage": "knowledge", "status": "failed", "error": str(error)})
        stage_status["knowledge"] = "failed"
        artifacts["knowledge"] = {
            "task_yaml": str(paths.task_yaml),
            "runtime_state_yaml": str(paths.runtime_state_yaml),
            "knowledge_sources_yaml": str(paths.knowledge_sources_yaml),
            "patch_diff": str(paths.patch_diff),
        }
        return {
            "current_stage": "knowledge",
            "review_stage": "knowledge",
            "human_action_required": True,
            "review_reason": "knowledge stage failed",
            "stage_history": history,
            "stage_status": stage_status,
            "artifacts": artifacts,
            "last_error": str(error),
            "final_status": "needs_review",
        }


def prepare_layout(paths: KnowledgePaths) -> None:
    """Create directories owned by the knowledge stage."""

    paths.yaml_dir.mkdir(parents=True, exist_ok=True)
    paths.evidence_dir.mkdir(parents=True, exist_ok=True)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.cleaned_dir.mkdir(parents=True, exist_ok=True)
    paths.extracted_dir.mkdir(parents=True, exist_ok=True)
    paths.diff_dir.mkdir(parents=True, exist_ok=True)
    paths.pocs_dir.mkdir(parents=True, exist_ok=True)


def reset_stage_outputs(paths: KnowledgePaths) -> None:
    """Clear stage-owned outputs before writing a fresh run."""

    for path in (paths.knowledge_yaml, paths.runtime_state_yaml, paths.knowledge_sources_yaml):
        if path.exists():
            path.unlink()


def build_runtime_state_payload(
    task_id: str,
    success: bool,
    message: str,
    llm_status: str = "disabled",
    llm_error: Optional[str] = None,
) -> dict:
    """Build a runtime-state payload for the knowledge stage."""

    return {
        "task_id": task_id,
        "current_stage": "knowledge",
        "retry_count": {},
        "stage_history": [{"stage": "knowledge", "status": "success" if success else "failed"}],
        "last_error": None if success else message,
        "llm_status": llm_status,
        "llm_error": llm_error,
        "workspace": f"workspaces/{task_id}",
        "final_status": "success" if success else "failed",
    }


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def write_yaml(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=True)


def render_cleaned_markdown(url: str, title: str, cleaned_text: str) -> str:
    heading = title or url
    return f"# {heading}\n\nSource: {url}\n\n{cleaned_text}\n"


def _harvest_hint_for_recipe(recipe: ReproductionRecipe) -> str:
    name = recipe.source_title or "poc"
    if recipe.recipe_type == "ossfuzz_testcase":
        return f"Harvested OSS-Fuzz testcase into vuln_pocs/{name}"
    if recipe.recipe_type == "embedded_text_payload":
        return f"Harvested embedded PoC into vuln_pocs/{name}"
    return f"Harvested attachment PoC into vuln_pocs/{name}"


def sanitize_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    sanitized = sanitized.strip("._") or "artifact"
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    if len(sanitized) > 96:
        sanitized = sanitized[:96].rstrip("._")
    return f"{sanitized}_{digest}"


def ensure_empty_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")


_REFERENCE_PRIORITY_ORDER = {
    "P0": 0,
    "P1": 1,
    "P2": 2,
    "P3": 3,
}

_BUILD_FILE_TO_SYSTEM = {
    "makefile": "make",
    "gnumakefile": "make",
    "dockerfile": "docker",
    "docker-compose.yml": "docker",
    "docker-compose.yaml": "docker",
    "cmakelists.txt": "cmake",
    "meson.build": "meson",
    "build.ninja": "ninja",
    "configure": "autotools",
    "configure.ac": "autotools",
    "go.mod": "go",
    "cargo.toml": "cargo",
    "package.json": "npm",
    "pom.xml": "maven",
    "build.gradle": "gradle",
    "build.gradle.kts": "gradle",
    "gradlew": "gradle",
    "requirements.txt": "python",
    "setup.py": "python",
    "pyproject.toml": "python",
}

_BUILD_FILE_HINTS = sorted(_BUILD_FILE_TO_SYSTEM.keys(), key=len, reverse=True)
_BUILD_FILE_RE = re.compile(
    r"(?i)(?:^|[\s`'\"(])(?P<path>[A-Za-z0-9_./-]*(?:"
    + "|".join(re.escape(item) for item in _BUILD_FILE_HINTS)
    + r"))(?=$|[\s`'\":),])"
)

_INSTALL_COMMAND_PREFIXES = (
    "apt-get install",
    "apt install",
    "yum install",
    "dnf install",
    "apk add",
    "pacman -s",
    "brew install",
    "pip install",
    "python -m pip install",
    "npm install",
    "npm ci",
    "yarn install",
    "pnpm install",
    "go mod download",
    "cargo fetch",
    "bundle install",
    "composer install",
)

_BUILD_COMMAND_PREFIXES = (
    "./configure",
    "configure",
    "make",
    "cmake",
    "cmake --build",
    "ninja",
    "meson setup",
    "meson compile",
    "go build",
    "go test",
    "cargo build",
    "cargo test",
    "python -m build",
    "python setup.py build",
    "python setup.py install",
    "mvn package",
    "mvn install",
    "mvn test",
    "gradle build",
    "./gradlew build",
    "./gradlew assemble",
    "npm run build",
    "npm run compile",
    "yarn build",
    "pnpm build",
    "docker build",
    "bazel build",
)


def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def dedupe_reference_records(records: Iterable[ReferenceRecord]) -> List[ReferenceRecord]:
    seen = set()
    ordered: list[ReferenceRecord] = []
    for record in records:
        if record.url in seen:
            continue
        seen.add(record.url)
        ordered.append(record)
    return ordered


def dedupe_task_references(records: Iterable[TaskReference]) -> List[dict]:
    extractor = ReferenceExtractor()
    seen = set()
    ordered: list[dict] = []
    for record in records:
        normalized = extractor.normalize([record.url])
        normalized_url = normalized[0] if normalized else record.url
        if normalized_url in seen:
            continue
        seen.add(normalized_url)
        ordered.append({"url": normalized_url, "type": record.type})
    return ordered


def truncate_reference_records(
    records: List[ReferenceRecord],
    max_count: int,
    drop_note: str,
) -> tuple[List[ReferenceRecord], List[ReferenceRecord]]:
    if max_count <= 0 or len(records) <= max_count:
        return records, []

    ordered = sorted(
        enumerate(records),
        key=lambda item: (
            _REFERENCE_PRIORITY_ORDER.get(item[1].priority, 99),
            item[1].depth,
            item[0],
        ),
    )
    kept_indexes = {index for index, _ in ordered[:max_count]}

    kept: list[ReferenceRecord] = []
    dropped: list[ReferenceRecord] = []
    for index, record in enumerate(records):
        if index in kept_indexes:
            kept.append(record)
            continue
        dropped.append(record.model_copy(update={"selected": False, "note": f"{record.note} {drop_note}".strip()}))
    return kept, dropped


def limit_output_urls(urls: List[str], max_count: int) -> List[str]:
    if max_count <= 0:
        return []
    return dedupe_preserve_order(urls)[:max_count]


def score_reference(url: str, reference_type: Optional[str] = None) -> str:
    normalized_type = (reference_type or "").upper()
    if normalized_type in {"FIX", "EVIDENCE"}:
        return "P0"

    lowered = url.lower()
    if is_github_attachment_url(url) or is_ossfuzz_issue_url(url):
        return "P0"
    if lowered.endswith((".zip", ".tar", ".tar.gz", ".tgz")) and any(
        token in lowered for token in ("poc", "proof", "crash", "repro", "fuzz", "overflow")
    ):
        return "P0"
    if ".diff" in lowered or ".patch" in lowered or "/-/commit/" in lowered or "/commit/" in lowered:
        return "P0"
    if "/-/commits/" in lowered or "/commits/" in lowered:
        return "P3"
    if "/pull/" in lowered or "/issues/" in lowered or "advisory" in lowered or "security" in lowered:
        return "P1"
    if "osv.dev" in lowered or "nvd.nist.gov" in lowered or "lists." in lowered:
        return "P2"
    return "P3"


def build_reference_type_map(reference_details: Iterable[TaskReference]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    extractor = ReferenceExtractor()
    for item in reference_details:
        normalized = extractor.normalize([item.url])
        if normalized:
            mapping[normalized[0]] = (item.type or "").upper()
    return mapping


def reference_type_for_url(reference_details: Iterable[TaskReference], url: str) -> Optional[str]:
    return build_reference_type_map(reference_details).get(url)


def guess_source_type(url: str) -> str:
    lowered = url.lower()
    if is_github_attachment_url(url):
        return "attachment"
    if is_ossfuzz_issue_url(url):
        return "ossfuzz"
    if ".diff" in lowered:
        return "diff"
    if ".patch" in lowered:
        return "patch"
    if "/commit/" in lowered or "/commits/" in lowered:
        return "commit"
    if "/pull/" in lowered:
        return "pull_request"
    if "/issues/" in lowered:
        return "issue"
    if "osv.dev" in lowered:
        return "osv"
    if "nvd.nist.gov" in lowered:
        return "nvd"
    return "reference"


def derive_reference_variants(url: str) -> List[str]:
    lowered = url.lower()
    variants: list[str] = []
    path = urlsplit(url).path
    # GitHub and GitLab(KDE invent) commit pages → raw unified diff.
    # Invent often uses both `/commit/<sha>` and `/-/commit/<sha>`.
    is_commit_page = bool(re.search(r"/(?:-/)?commit/[0-9a-fA-F]{7,40}/?$", path))
    if is_commit_page and not lowered.endswith(".diff") and "?" not in url:
        variants.append(f"{url.rstrip('/')}.diff")
    elif "github.com" in lowered and "/commit/" in lowered and not lowered.endswith(".diff"):
        variants.append(f"{url.rstrip('/')}.diff")
    if "github.com" in lowered and "/blob/" in lowered:
        parts = urlsplit(url)
        path_parts = [segment for segment in parts.path.split("/") if segment]
        if len(path_parts) >= 5:
            owner, repo, _, ref = path_parts[:4]
            raw_path = "/".join(path_parts[4:])
            variants.append(f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{raw_path}")
    if "/-/blob/" in lowered:
        variants.append(url.replace("/-/blob/", "/-/raw/", 1))
    return variants


def should_follow_discovered_link(parent_url: str, child_url: str) -> bool:
    parent_parts = urlsplit(parent_url)
    child_parts = urlsplit(child_url)

    if child_parts.scheme not in {"http", "https"}:
        return False

    lowered = child_url.lower()
    if any(token in lowered for token in ("/login", "/signup", "/search", "/features", "/marketplace")):
        return False
    if "/-/commits/" in lowered or "/commits/" in lowered:
        return False
    if ("/-/commit/" in lowered or "/commit/" in lowered) and child_parts.query:
        return False

    blocked_suffixes = (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".ico",
        ".css",
        ".js",
        ".woff",
        ".woff2",
        ".mp4",
        ".mp3",
    )
    lowered_path = child_parts.path.lower()
    if lowered_path.endswith(blocked_suffixes):
        return False

    if "nvd.nist.gov" in parent_parts.netloc.lower():
        return any(
            signal in lowered
            for signal in (
                "/vuln/detail/cve-",
                "/security/advisories/",
                "/commit/",
                ".diff",
                ".patch",
                "github.com",
                "gitlab.com",
            )
        )

    if "lists.fedoraproject.org" in child_parts.netloc.lower():
        return "/message/" in lowered or "cve-" in lowered

    if "github.com" in parent_parts.netloc.lower() and "github.com" in child_parts.netloc.lower():
        if is_github_attachment_url(child_url):
            return True
        parent_repo = [segment for segment in parent_parts.path.split("/") if segment][:2]
        child_repo = [segment for segment in child_parts.path.split("/") if segment][:2]
        if len(parent_repo) == 2 and parent_repo == child_repo:
            return score_reference(child_url) in {"P0", "P1"}
        return "/security/advisories/" in lowered and score_reference(child_url) in {"P0", "P1"}

    if child_parts.netloc == parent_parts.netloc:
        return True

    if "github.com" in lowered or "gitlab.com" in lowered:
        return score_reference(child_url) in {"P0", "P1"}
    if "advisory" in lowered or "security" in lowered or "cve-" in lowered:
        return True
    if score_reference(child_url) in {"P0", "P1", "P2"}:
        return True

    return False


def looks_like_patch(url: str, content_type: str, body: str) -> bool:
    lowered_url = url.lower()
    if lowered_url.endswith(".patch") or lowered_url.endswith(".diff"):
        return True
    if content_type in {"text/plain", "text/x-diff"} and ("diff --git" in body or body.startswith("From ")):
        return True
    return False


def infer_language(osv_payload: dict) -> Optional[str]:
    mapping = {
        "PyPI": "Python",
        "Go": "Go",
        "Maven": "Java",
        "npm": "JavaScript",
        "crates.io": "Rust",
    }
    for affected in osv_payload.get("affected", []):
        ecosystem = (affected.get("package") or {}).get("ecosystem", "")
        language = mapping.get(ecosystem)
        if language:
            return language
    return None


def infer_repo_url(osv_payload: dict) -> Optional[str]:
    """Infer the vulnerable product repository from OSV metadata.

    Prefer explicit GIT range repos, then commit URLs that are not metadata
    mirrors (e.g. google/oss-fuzz-vulns). Package-name matches rank higher.
    """

    package_names = _osv_package_names(osv_payload)
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(candidate: Optional[str]) -> None:
        normalized = _normalize_repo_git_url(candidate)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        candidates.append(normalized)

    for affected in osv_payload.get("affected", []) or []:
        for item in affected.get("ranges", []) or []:
            if (item.get("type") or "").upper() != "GIT":
                continue
            _add(item.get("repo"))

    for item in osv_payload.get("references", []) or []:
        _add(_repo_url_from_commit_reference(item.get("url", "")))

    if not candidates:
        return None

    preferred = [url for url in candidates if not _is_metadata_repo_url(url)]
    pool = preferred or candidates

    if package_names:
        matched = [url for url in pool if _repo_matches_package_names(url, package_names)]
        if matched:
            return matched[0]
    return pool[0]


_METADATA_REPO_SLUGS = frozenset(
    {
        "google/oss-fuzz-vulns",
        "google/oss-fuzz",
        "github/advisory-database",
        "pypa/advisory-database",
        "rustsec/advisory-db",
        "rubysec/ruby-advisory-db",
    }
)


def _normalize_repo_git_url(repo_url: Optional[str]) -> Optional[str]:
    if not repo_url:
        return None
    raw = str(repo_url).strip()
    if not raw:
        return None
    parts = urlsplit(raw)
    if not parts.scheme or not parts.netloc:
        return None
    path = parts.path.rstrip("/")
    if not path:
        return None
    if not path.endswith(".git"):
        path = f"{path}.git"
    return f"{parts.scheme}://{parts.netloc}{path}"


def _related_osv_ids_from_payload(osv_payload: dict) -> list[str]:
    """Extract linked OSV-* IDs from oss-fuzz-vulns advisory URLs."""

    found: list[str] = []
    seen: set[str] = set()
    pattern = re.compile(r"(OSV-\d{4}-\d+)", re.IGNORECASE)
    for item in osv_payload.get("references", []) or []:
        url = str(item.get("url") or "")
        if "oss-fuzz-vulns" not in url.lower():
            continue
        match = pattern.search(url)
        if not match:
            continue
        osv_id = match.group(1).upper()
        if osv_id in seen:
            continue
        seen.add(osv_id)
        found.append(osv_id)
    return found


def _repo_basename(repo_url: Optional[str]) -> str:
    if not repo_url:
        return ""
    path = urlsplit(repo_url).path.lower().rstrip("/").removesuffix(".git")
    if not path:
        return ""
    return path.rsplit("/", 1)[-1]


def _git_range_matches_target(
    range_repo: Optional[str],
    preferred_slug: str,
    preferred_path: str,
    package_names: list[str],
) -> bool:
    """True when a GIT range belongs to the chosen product repo (incl. SF mirrors)."""

    range_repo = _normalize_repo_git_url(range_repo) or (range_repo or "")
    if not range_repo or _is_metadata_repo_url(range_repo):
        return False
    range_slug = (extract_github_repo_slug(range_repo) or "").lower()
    range_path = urlsplit(range_repo).path.lower().rstrip("/").removesuffix(".git")
    if preferred_slug and range_slug and range_slug == preferred_slug:
        return True
    if preferred_path and range_path and range_path == preferred_path:
        return True
    preferred_base = preferred_path.rsplit("/", 1)[-1] if preferred_path else ""
    range_base = _repo_basename(range_repo)
    if not preferred_base or preferred_base != range_base:
        return False
    lowered_packages = {name.lower() for name in package_names}
    return preferred_base in lowered_packages or range_base in lowered_packages


def _is_metadata_repo_url(repo_url: str) -> bool:
    slug = extract_github_repo_slug(repo_url)
    if slug and slug.lower() in _METADATA_REPO_SLUGS:
        return True
    lowered = (repo_url or "").lower()
    return "oss-fuzz-vulns" in lowered or "/advisory-database" in lowered


def _repos_mentioning_commit_sha(osv_payload: dict, sha: str) -> list[str]:
    """Return repository URLs associated with a commit SHA in OSV evidence."""

    needle = (sha or "").strip().lower()
    if len(needle) < 7:
        return []
    found: list[str] = []
    seen: set[str] = set()

    def _add(repo: Optional[str]) -> None:
        normalized = _normalize_repo_git_url(repo)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        found.append(normalized)

    for item in osv_payload.get("references", []) or []:
        url = item.get("url", "") or ""
        if needle not in url.lower():
            continue
        if "/commit/" in url or "/-/commit/" in url:
            _add(_repo_url_from_commit_reference(url))

    for affected in osv_payload.get("affected", []) or []:
        for item in affected.get("ranges", []) or []:
            if (item.get("type") or "").upper() != "GIT":
                continue
            for event in item.get("events", []) or []:
                for key in ("fixed", "introduced", "last_affected", "limit"):
                    value = str(event.get(key) or "").strip().lower()
                    if value and (value == needle or value.startswith(needle) or needle.startswith(value)):
                        _add(item.get("repo"))
                        break
    return found


def _ref_incompatible_with_repo(
    osv_payload: dict,
    sha: Optional[str],
    repo_url: Optional[str],
) -> bool:
    """True when SHA is only evidenced on metadata/other repos, not the target."""

    if not sha or not repo_url or _is_metadata_repo_url(repo_url):
        return False
    mentions = _repos_mentioning_commit_sha(osv_payload, sha)
    if not mentions:
        return False
    target = _normalize_repo_git_url(repo_url)
    target_slug = (extract_github_repo_slug(target) or "").lower()
    for mention in mentions:
        if target and mention == target:
            return False
        mention_slug = (extract_github_repo_slug(mention) or "").lower()
        if target_slug and mention_slug == target_slug:
            return False
        if not _is_metadata_repo_url(mention) and target_slug and mention_slug == target_slug:
            return False
    # Known associations exist, but none are the product repo.
    return True


def _osv_package_names(osv_payload: dict) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for affected in osv_payload.get("affected", []) or []:
        package = affected.get("package") or {}
        for key in ("name", "purl"):
            value = str(package.get(key) or "").strip()
            if not value:
                continue
            tokens = [value.lower()]
            if "/" in value:
                tokens.append(value.lower().rstrip("/").rsplit("/", 1)[-1])
            for token in tokens:
                token = token.removeprefix("pkg:").split("@", 1)[0]
                if token and token not in seen:
                    seen.add(token)
                    names.append(token)
    return names


def _repo_matches_package_names(repo_url: str, package_names: list[str]) -> bool:
    slug = (extract_github_repo_slug(repo_url) or "").lower()
    path = urlsplit(repo_url).path.lower().rstrip("/").removesuffix(".git")
    repo_name = path.rsplit("/", 1)[-1] if path else ""
    haystacks = {slug, path.lstrip("/"), repo_name}
    for name in package_names:
        if name in haystacks:
            return True
        if name.startswith("github.com/") and name.removeprefix("github.com/") in haystacks:
            return True
        if repo_name and (repo_name == name or name.endswith(f"/{repo_name}") or repo_name.endswith(name)):
            return True
    return False


def _repo_url_from_commit_reference(url: str) -> Optional[str]:
    if not url:
        return None
    if "github.com" in url and "/commit/" in url:
        parts = urlsplit(url)
        path_parts = [segment for segment in parts.path.split("/") if segment]
        if len(path_parts) >= 2:
            return f"{parts.scheme}://{parts.netloc}/{path_parts[0]}/{path_parts[1]}.git"
    if "/commit/" in url:
        parts = urlsplit(url)
        path_parts = [segment for segment in parts.path.split("/") if segment]
        if "commit" in path_parts:
            marker_index = path_parts.index("commit")
            if marker_index >= 2:
                project_path = "/".join(path_parts[:marker_index])
                return f"{parts.scheme}://{parts.netloc}/{project_path}.git"
    if "/-/commit/" in url:
        parts = urlsplit(url)
        path_parts = [segment for segment in parts.path.split("/") if segment]
        if "-" in path_parts:
            marker_index = path_parts.index("-")
            if marker_index >= 2:
                project_path = "/".join(path_parts[:marker_index])
                return f"{parts.scheme}://{parts.netloc}/{project_path}.git"
    return None


def osv_has_commit_reference(osv_payload: dict) -> bool:
    for item in osv_payload.get("references", []):
        url = item.get("url", "")
        lowered = url.lower()
        if "/commit/" in lowered or "/-/commit/" in lowered:
            return True
    for affected in osv_payload.get("affected", []) or []:
        for item in affected.get("ranges", []) or []:
            if (item.get("type") or "").upper() == "GIT" and (item.get("repo") or item.get("events")):
                return True
    return False


def extract_github_repo_slug(repo_url: Optional[str]) -> Optional[str]:
    if not repo_url:
        return None

    parts = urlsplit(repo_url)
    if "github.com" not in parts.netloc.lower():
        return None

    path_parts = [segment for segment in parts.path.split("/") if segment]
    if len(path_parts) < 2:
        return None

    owner = path_parts[0]
    repo = path_parts[1].removesuffix(".git")
    if not owner or not repo:
        return None
    return f"{owner}/{repo}"


def extract_gitlab_project_path(repo_url: Optional[str]) -> Optional[str]:
    if not repo_url:
        return None

    parts = urlsplit(repo_url)
    path_parts = [segment for segment in parts.path.split("/") if segment]
    if len(path_parts) < 2:
        return None

    normalized_parts = path_parts[:-1] + [path_parts[-1].removesuffix(".git")]
    if not normalized_parts[-1]:
        return None
    return "/".join(normalized_parts)


def fetch_repo_primary_language(repo_url: Optional[str]) -> Optional[str]:
    if not repo_url:
        return None

    parts = urlsplit(repo_url)
    headers = {
        "User-Agent": KnowledgeStage._USER_AGENT,
        "Accept": "application/json",
    }

    try:
        if "github.com" in parts.netloc.lower():
            repo_slug = extract_github_repo_slug(repo_url)
            if not repo_slug:
                return None
            request = Request(
                f"https://api.github.com/repos/{repo_slug}/languages",
                headers=headers,
            )
        else:
            project_path = extract_gitlab_project_path(repo_url)
            if not project_path:
                return None
            request = Request(
                f"{parts.scheme}://{parts.netloc}/api/v4/projects/{quote(project_path, safe='')}/languages",
                headers=headers,
            )

        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception:
        return None

    if not isinstance(payload, dict) or not payload:
        return None

    top_language = max(
        ((name, weight) for name, weight in payload.items() if isinstance(name, str)),
        key=lambda item: item[1] if isinstance(item[1], (int, float)) else 0,
        default=(None, None),
    )[0]
    return top_language if isinstance(top_language, str) and top_language else None


def fetch_github_parent_ref(repo_url: Optional[str], commit_ref: Optional[str]) -> Optional[str]:
    repo_slug = extract_github_repo_slug(repo_url)
    if not repo_slug or not commit_ref:
        return None

    request = Request(
        f"https://api.github.com/repos/{repo_slug}/commits/{commit_ref}",
        headers={
            "User-Agent": KnowledgeStage._USER_AGENT,
            "Accept": "application/vnd.github+json",
        },
    )
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception:
        return None

    parents = payload.get("parents") or []
    if not parents:
        return None

    first_parent = parents[0]
    if not isinstance(first_parent, dict):
        return None
    parent_sha = first_parent.get("sha")
    return parent_sha if isinstance(parent_sha, str) and parent_sha else None


def infer_git_refs(
    osv_payload: dict,
    fallback_fixed: Optional[str],
    fallback_vulnerable: Optional[str],
    repo_url: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    fixed_ref = fallback_fixed
    if fixed_ref and _ref_incompatible_with_repo(osv_payload, fixed_ref, repo_url):
        fixed_ref = None
    preferred_slug = (extract_github_repo_slug(repo_url) or "").lower()
    preferred_path = ""
    if repo_url:
        preferred_path = urlsplit(repo_url).path.lower().rstrip("/").removesuffix(".git")
    package_names = _osv_package_names(osv_payload)

    commit_candidates: list[str] = []
    for item in osv_payload.get("references", []) or []:
        url = item.get("url", "")
        if "/-/commit/" not in url and "/commit/" not in url:
            continue
        sha = url.rstrip("/").split("/")[-1].replace(".patch", "").replace(".diff", "")
        if not sha:
            continue
        commit_repo = _repo_url_from_commit_reference(url)
        commit_slug = (extract_github_repo_slug(commit_repo) or "").lower() if commit_repo else ""
        commit_path = ""
        if commit_repo:
            commit_path = urlsplit(commit_repo).path.lower().rstrip("/").removesuffix(".git")
        belongs_to_target = bool(
            preferred_slug
            and commit_slug
            and commit_slug == preferred_slug
        ) or bool(
            preferred_path
            and commit_path
            and commit_path == preferred_path
        )
        # Skip metadata-repo commits when we already know the product repo.
        if preferred_slug and _is_metadata_repo_url(commit_repo or ""):
            continue
        if belongs_to_target:
            commit_candidates.insert(0, sha)
        else:
            commit_candidates.append(sha)

    if not fixed_ref and commit_candidates:
        fixed_ref = commit_candidates[0]

    introduced_ref: Optional[str] = None
    last_affected_ref: Optional[str] = None
    for affected in osv_payload.get("affected", []):
        for item in affected.get("ranges", []):
            if item.get("type") != "GIT":
                continue
            range_repo = _normalize_repo_git_url(item.get("repo")) or item.get("repo")
            if range_repo and (preferred_slug or preferred_path):
                if not _git_range_matches_target(
                    range_repo, preferred_slug, preferred_path, package_names
                ):
                    # Keep scanning other ranges; skip metadata / unrelated repos.
                    continue
            for event in item.get("events", []):
                fixed = event.get("fixed")
                if fixed:
                    fixed_ref = fixed_ref or fixed
                introduced = event.get("introduced")
                if introduced and introduced != "0":
                    introduced_ref = introduced_ref or introduced
                last_affected = event.get("last_affected")
                if last_affected:
                    last_affected_ref = last_affected_ref or last_affected

    fallback_vulnerable_ok = fallback_vulnerable
    if fallback_vulnerable_ok and _ref_incompatible_with_repo(
        osv_payload, fallback_vulnerable_ok, repo_url
    ):
        fallback_vulnerable_ok = None
    vulnerable_ref = (
        fetch_github_parent_ref(repo_url, fixed_ref)
        or last_affected_ref
        or introduced_ref
        or fallback_vulnerable_ok
    )
    return vulnerable_ref, fixed_ref


def extract_summary_candidate(page: FetchedPage) -> str:
    text = page.cleaned_text.strip()
    if not text:
        return ""

    title_candidate = normalize_summary_title(page.title or text.splitlines()[0])
    lowered_text = text.lower()
    noise_markers = (
        "navigation menu",
        "toggle navigation",
        "search or jump to",
        "appearance settings",
        "loading",
        "try again",
        "saved searches",
        "provide feedback",
    )
    if sum(marker in lowered_text for marker in noise_markers) >= 2:
        return title_candidate[:200] if title_candidate else ""

    section_markers = ("Description", "Impact", "Summary", "Details")
    lines = [line.strip() for line in text.splitlines()]
    for index, line in enumerate(lines):
        if line not in section_markers:
            continue
        collected: list[str] = []
        for candidate in lines[index + 1 :]:
            if not candidate:
                if collected:
                    break
                continue
            if candidate in section_markers or len(candidate) <= 2:
                if collected:
                    break
                continue
            collected.append(candidate)
            if sum(len(item) for item in collected) >= 700:
                break
        if collected:
            return " ".join(collected)[:1000]

    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    for paragraph in paragraphs:
        lowered = paragraph.lower()
        if any(noise in lowered for noise in ("navigation menu", "toggle navigation", "search or jump to")):
            continue
        if len(paragraph) >= 80:
            return paragraph[:1000]

    first_paragraph = paragraphs[0] if paragraphs else ""
    return first_paragraph[:1000] if first_paragraph else ""


def normalize_summary_title(title: str) -> str:
    normalized = re.sub(r"\s+", " ", title).strip()
    normalized = re.sub(r"\s*[·|-]\s*(GitHub|GitLab)$", "", normalized)
    normalized = re.sub(r"\s*[·|-]\s*(Commits?|Issues?)$", "", normalized)
    return normalized[:200]


def page_summary_score(page: FetchedPage) -> int:
    lowered_url = page.url.lower()
    lowered_text = page.cleaned_text.lower()
    score = 0

    if "nvd.nist.gov/vuln/detail/" in lowered_url:
        score += 50
    if "/security/advisories/" in lowered_url or "ghsa-" in lowered_url:
        score += 45
    if "cveproject" in lowered_url or "cvelist" in lowered_url or "osv.dev" in lowered_url:
        score += 35
    if "/commit/" in lowered_url:
        score -= 20
    if ".diff" in lowered_url or ".patch" in lowered_url:
        score -= 10
    if "description" in lowered_text:
        score += 20
    if "impact" in lowered_text:
        score += 10
    if "navigation menu" in lowered_text:
        score -= 15

    candidate = extract_summary_candidate(page)
    if len(candidate) >= 120:
        score += 10
    return score


def heuristic_summary_from_pages(fetched_pages: List[FetchedPage]) -> str:
    ranked_pages = sorted(
        (page for page in fetched_pages if page.cleaned_text.strip()),
        key=page_summary_score,
        reverse=True,
    )
    for page in ranked_pages:
        candidate = extract_summary_candidate(page)
        if candidate:
            return candidate
    return ""


def infer_vulnerability_type(text: str) -> str:
    lowered = text.lower()
    mapping = [
        ("heap-buffer-overflow", "heap-buffer-overflow"),
        ("stack-buffer-overflow", "stack-buffer-overflow"),
        ("use-after-free", "use-after-free"),
        ("null pointer", "null-pointer-dereference"),
        ("out-of-bounds", "out-of-bounds-access"),
        ("denial of service", "denial-of-service"),
        ("privilege escalation", "privilege-escalation"),
        ("authentication bypass", "authentication-bypass"),
        ("authorization bypass", "authorization-bypass"),
        ("business logic", "business-logic-error"),
        ("protocol rule", "business-logic-error"),
        ("improper validation", "improper-input-validation"),
        ("validation", "improper-input-validation"),
        ("not checked", "improper-input-validation"),
    ]
    for needle, label in mapping:
        if needle in lowered:
            return label
    return ""


def default_error_patterns(vulnerability_type: str) -> List[str]:
    if vulnerability_type == "heap-buffer-overflow":
        return ["AddressSanitizer: heap-buffer-overflow"]
    if vulnerability_type == "stack-buffer-overflow":
        return ["AddressSanitizer: stack-buffer-overflow"]
    if vulnerability_type == "use-after-free":
        return ["AddressSanitizer: heap-use-after-free"]
    return []


def extract_stack_keywords(patch_summaries: List[PatchSummary]) -> List[str]:
    keywords = []
    for summary in patch_summaries:
        keywords.extend(summary.changed_functions)
    return dedupe_preserve_order(keyword for keyword in keywords if keyword)[:10]


def normalize_build_path(path: str) -> str:
    normalized = path.strip().strip("`'\"()[]{}<>.,:;")
    normalized = normalized.replace("\\", "/")
    return normalized.lstrip("./")


def build_file_basename(path: str) -> str:
    normalized = normalize_build_path(path)
    return normalized.split("/")[-1].lower()


def is_build_related_file(path: str) -> bool:
    return build_file_basename(path) in _BUILD_FILE_TO_SYSTEM


def extract_build_files(fetched_pages: List[FetchedPage], patch_summaries: List[PatchSummary]) -> List[str]:
    build_files: list[str] = []

    for patch in patch_summaries:
        for path in patch.affected_files:
            normalized = normalize_build_path(path)
            if normalized and is_build_related_file(normalized):
                build_files.append(normalized)

    for page in fetched_pages:
        for match in _BUILD_FILE_RE.finditer(page.cleaned_text):
            normalized = normalize_build_path(match.group("path"))
            if normalized and is_build_related_file(normalized):
                build_files.append(normalized)

        page_path = normalize_build_path(urlsplit(page.url).path)
        if page_path and is_build_related_file(page_path):
            build_files.append(page_path)

    return dedupe_preserve_order(build_files)[:20]


def infer_build_systems(build_files: List[str], language: Optional[str]) -> List[str]:
    systems: list[str] = []
    for path in build_files:
        system = _BUILD_FILE_TO_SYSTEM.get(build_file_basename(path))
        if system:
            systems.append(system)

    language_mapping = {
        "Go": "go",
        "Rust": "cargo",
        "Python": "python",
        "Java": "maven",
        "JavaScript": "npm",
    }
    inferred_from_language = language_mapping.get(language or "")
    if inferred_from_language and inferred_from_language not in systems:
        systems.append(inferred_from_language)

    return dedupe_preserve_order(systems)


def normalize_command_candidate(candidate: str) -> str:
    line = candidate.strip()
    line = re.sub(r"^(?:[-*]\s+|\d+\.\s+)", "", line)
    line = re.sub(r"^(?:\$|#|>)\s*", "", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip("` ")


def extract_commands_from_pages(fetched_pages: List[FetchedPage], prefixes: tuple[str, ...], limit: int = 12) -> List[str]:
    commands: list[str] = []
    prefix_set = tuple(prefix.lower() for prefix in prefixes)

    for page in fetched_pages:
        candidates = list(page.cleaned_text.splitlines())
        candidates.extend(re.findall(r"`([^`\n]{3,200})`", page.cleaned_text))
        for candidate in candidates:
            normalized = normalize_command_candidate(candidate)
            lowered = normalized.lower()
            if not normalized or len(normalized) < 3:
                continue
            if any(lowered == prefix or lowered.startswith(prefix + " ") for prefix in prefix_set):
                commands.append(normalized)
            if len(dedupe_preserve_order(commands)) >= limit:
                return dedupe_preserve_order(commands)[:limit]

    return dedupe_preserve_order(commands)[:limit]


def extract_install_commands(fetched_pages: List[FetchedPage]) -> List[str]:
    return extract_commands_from_pages(fetched_pages, _INSTALL_COMMAND_PREFIXES, limit=10)


def extract_build_commands(fetched_pages: List[FetchedPage]) -> List[str]:
    return extract_commands_from_pages(fetched_pages, _BUILD_COMMAND_PREFIXES, limit=10)


def build_build_hints(
    build_files: List[str],
    build_systems: List[str],
    install_commands: List[str],
    build_commands: List[str],
    patch_summaries: List[PatchSummary],
) -> List[str]:
    hints: list[str] = []

    if build_files:
        hints.append(f"Inspect build-related files such as: {', '.join(build_files[:5])}.")
    if build_systems:
        hints.append(f"Detected build systems: {', '.join(build_systems[:4])}.")
    if install_commands:
        hints.append("The collected references include explicit dependency or environment preparation commands.")
    if build_commands:
        hints.append("The collected references include explicit compilation or build commands.")

    changed_build_files = [
        path
        for patch in patch_summaries
        for path in patch.affected_files
        if is_build_related_file(path)
    ]
    if changed_build_files:
        hints.append(f"The patch touches build-related files: {', '.join(dedupe_preserve_order(changed_build_files)[:5])}.")

    return dedupe_preserve_order(hints)


def build_reproduction_hints(task: TaskModel, fetched_pages: List[FetchedPage], patch_summaries: List[PatchSummary]) -> List[str]:
    hints: list[str] = []
    if task.repo_url:
        hints.append("Clone the target repository and inspect the vulnerable and fixed revisions.")
    if task.vulnerable_ref:
        hints.append(f"Start from the vulnerable revision: {task.vulnerable_ref}.")
    if task.fixed_ref:
        hints.append(f"Compare with the fixed revision: {task.fixed_ref}.")
    if patch_summaries:
        hints.append("Review the patch to identify the modified files and triggering path.")
    recipes = extract_reproduction_recipes(fetched_pages)
    if recipes:
        hints.append("Recovered explicit reproduction steps from the collected evidence.")
    if any(page.cleaned_text for page in fetched_pages):
        hints.append("Use the cleaned advisory or discussion pages to recover trigger conditions and expected error signatures.")
    return dedupe_preserve_order(hints)


_REPRO_SECTION_KEYWORDS = (
    "how to reproduce",
    "how to reprocude",
    "reproduce",
    "reprocude",
    "reproducer",
    "proof of concept",
    "poc",
    "steps",
    "trigger",
    "base64",
    "minimized",
    "cil policy",
    "minimized cil",
    "classmap",
    "classmapping",
)

_COMMAND_START_RE = re.compile(
    r"^(?:\$|#|>)?\s*(?:"
    r"git clone\b|git checkout\b|cd\b|make\b|cmake\b|ninja\b|cargo\b|go build\b|python\b|python3\b|"
    r"perl\b|ruby\b|node\b|npm\b|yarn\b|./|echo\b|base64\b|cat\b|cp\b|mv\b|ln\b|lua\b|docker\b"
    r")",
    re.IGNORECASE,
)

_RUN_COMMAND_PREFIXES = ("./", "lua ", "python ", "python3 ", "node ", "perl ", "ruby ", "bash ", "sh ")
_BUILD_COMMAND_HINTS = ("make", "cmake", "ninja", "cargo", "go build", "meson", "configure")


def build_reproduction_evidence_blocks(fetched_pages: List[FetchedPage], limit: int = 6) -> List[str]:
    blocks: list[str] = []

    for page in fetched_pages:
        if not page.cleaned_text:
            continue
        excerpt = extract_reproduction_excerpt(page.cleaned_text)
        if not excerpt:
            continue
        blocks.append(
            "\n".join(
                [
                    f"URL: {page.url}",
                    f"Title: {page.title}",
                    "Excerpt:",
                    excerpt,
                ]
            )
        )
        if len(blocks) >= limit:
            break

    return blocks


def extract_reproduction_recipes(fetched_pages: List[FetchedPage], limit: int = 4) -> List[ReproductionRecipe]:
    recipes: list[ReproductionRecipe] = []

    for page in fetched_pages:
        if not page.cleaned_text:
            continue
        excerpt = extract_reproduction_excerpt(page.cleaned_text)
        if not excerpt:
            continue
        steps = extract_recipe_steps(excerpt)
        if len(steps) < 2:
            continue
        recipes.append(
            ReproductionRecipe(
                source_url=page.url,
                source_title=page.title,
                recipe_type="shell_sequence",
                steps=steps,
                repo_setup_commands=classify_recipe_steps(steps, "repo_setup"),
                build_commands=classify_recipe_steps(steps, "build"),
                artifact_generation_commands=classify_recipe_steps(steps, "artifact_generation"),
                run_commands=classify_recipe_steps(steps, "run"),
                source_excerpt=excerpt,
                confidence="medium",
            )
        )
        if len(recipes) >= limit:
            break

    return recipes


def extract_reproduction_excerpt(cleaned_text: str, max_chars: int = 2400) -> str:
    heading_excerpt = extract_reproduction_excerpt_from_lines(cleaned_text, max_chars=max_chars)
    if heading_excerpt:
        return heading_excerpt

    paragraphs = [segment.strip() for segment in re.split(r"\n\s*\n", cleaned_text) if segment.strip()]
    for index, paragraph in enumerate(paragraphs):
        lowered = paragraph.lower()
        if any(keyword in lowered for keyword in _REPRO_SECTION_KEYWORDS):
            candidate_parts = [paragraph]
            for neighbor in paragraphs[index + 1 : index + 4]:
                if looks_like_command_block(neighbor) or contains_reproduction_signal(neighbor):
                    candidate_parts.append(neighbor)
                else:
                    break
            return truncate_preserving_lines("\n\n".join(candidate_parts), max_chars)

    command_blocks = [paragraph for paragraph in paragraphs if looks_like_command_block(paragraph)]
    if command_blocks:
        return truncate_preserving_lines(command_blocks[0], max_chars)

    return ""


def extract_reproduction_excerpt_from_lines(cleaned_text: str, max_chars: int = 2400) -> str:
    lines = [line.rstrip() for line in cleaned_text.splitlines()]
    for index, line in enumerate(lines):
        if not contains_reproduction_signal(line):
            continue
        command_lines: list[str] = []
        collecting = False
        for candidate in lines[index + 1 :]:
            stripped = candidate.strip()
            if not stripped:
                if collecting:
                    break
                continue
            if stripped.startswith(("---", "===", "***")):
                if collecting:
                    continue
                continue
            if is_numbered_or_command_line(stripped):
                command_lines.append(stripped)
                collecting = True
                continue
            if collecting:
                break
        if len(command_lines) >= 2:
            excerpt = "\n".join([line.strip()] + command_lines)
            return truncate_preserving_lines(excerpt, max_chars)
    return ""


def contains_reproduction_signal(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in _REPRO_SECTION_KEYWORDS)


def looks_like_command_block(text: str) -> bool:
    command_lines = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if is_numbered_or_command_line(stripped):
            command_lines += 1
    return command_lines >= 2


def extract_recipe_steps(excerpt: str) -> List[str]:
    steps: list[str] = []
    for line in excerpt.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        normalized = normalize_command_candidate(stripped)
        if _COMMAND_START_RE.match(normalized):
            steps.append(normalized)
            continue
    return dedupe_preserve_order(steps)


def is_numbered_or_command_line(line: str) -> bool:
    normalized = normalize_command_candidate(line)
    return bool(normalized and _COMMAND_START_RE.match(normalized))


def classify_recipe_steps(steps: List[str], category: str) -> List[str]:
    selected: list[str] = []
    for step in steps:
        lowered = step.lower()
        if category == "repo_setup" and ("git clone" in lowered or lowered.startswith("git checkout") or lowered.startswith("cd ")):
            selected.append(step)
        elif category == "build" and any(token in lowered for token in _BUILD_COMMAND_HINTS):
            selected.append(step)
        elif category == "artifact_generation" and any(token in lowered for token in ("base64", "cat ", "echo ", ">", "printf ")):
            selected.append(step)
        elif category == "run" and lowered.startswith(_RUN_COMMAND_PREFIXES):
            selected.append(step)
    return dedupe_preserve_order(selected)


def truncate_preserving_lines(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    lines: list[str] = []
    total = 0
    for line in text.splitlines():
        extra = len(line) + (1 if lines else 0)
        if lines and total + extra > max_chars:
            break
        if not lines and len(line) > max_chars:
            return line[: max_chars - 3].rstrip() + "..."
        lines.append(line)
        total += extra
    truncated = "\n".join(lines).rstrip()
    if truncated == text:
        return truncated
    return truncated + "\n..."


def parse_llm_json_payload(content) -> Optional[dict]:
    """Parse a JSON object from an LLM response payload."""

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            elif isinstance(item, str):
                text_parts.append(item)
        content = "\n".join(text_parts)

    if not isinstance(content, str):
        return None

    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        return json.loads(stripped)
    except Exception:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
