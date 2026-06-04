"""Run the standalone build stage for a single CVE identifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def configure_console_streams() -> None:
    """Avoid console encoding crashes when logs contain Unicode."""

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        encoding = getattr(stream, "encoding", None) or "utf-8"
        stream.reconfigure(encoding=encoding, errors="backslashreplace")


def bootstrap_import_path() -> Path:
    """Ensure the source directory is importable when running the script directly."""

    source_root = Path(__file__).resolve().parents[1]
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    return source_root


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for manual build-stage testing."""

    parser = argparse.ArgumentParser(
        description="Run the standalone build stage for one CVE identifier."
    )
    parser.add_argument(
        "cve_id",
        help="Target CVE identifier, for example CVE-2022-28805.",
    )
    parser.add_argument(
        "--dataset-root",
        default="Dataset",
        help="Dataset root directory relative to the current working directory.",
    )
    parser.add_argument(
        "--workspace-root",
        default="workspaces",
        help="Workspace root directory relative to the current working directory.",
    )
    return parser


def load_knowledge_model(cve_id: str, dataset_root: str):
    """Load the knowledge stage output as the build-stage input."""

    from app.schemas.knowledge import KnowledgeModel

    knowledge_path = Path(dataset_root) / cve_id / "vuln_yaml" / "knowledge.yaml"
    if not knowledge_path.exists():
        raise FileNotFoundError(f"knowledge.yaml not found: {knowledge_path}")

    payload = yaml.safe_load(knowledge_path.read_text(encoding="utf-8")) or {}
    return KnowledgeModel(**payload), knowledge_path


def main() -> int:
    """Execute the build stage and print the output file locations."""

    configure_console_streams()
    bootstrap_import_path()

    from app.stages.build import BuildStage, BuildStagePaths

    parser = build_parser()
    args = parser.parse_args()

    knowledge, knowledge_path = load_knowledge_model(args.cve_id, args.dataset_root)
    workspace = str(Path(args.workspace_root) / args.cve_id)
    stage = BuildStage()
    result = stage.run(knowledge=knowledge, workspace=workspace)
    paths = BuildStagePaths(workspace)

    print(f"Build stage completed for {args.cve_id}.")
    print(f"Knowledge YAML: {knowledge_path}")
    print(f"Workspace: {paths.workspace_root}")
    print(f"Repo path: {paths.repo_dir}")
    print(f"Build context YAML: {paths.build_context_yaml}")
    print(f"Build plan YAML: {paths.build_plan_yaml}")
    print(f"Dockerfile: {paths.dockerfile}")
    print(f"Build script: {paths.build_script}")
    print(f"Build log: {paths.build_log}")
    print(f"Build artifact YAML: {paths.build_artifact_yaml}")
    print(f"Build success: {result.build_success}")
    print(f"Build system: {result.build_system}")
    print(f"Resolved ref: {result.resolved_ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
