"""Run the standalone poc stage for a single CVE identifier."""

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
    """Build the command-line interface for manual poc-stage testing."""

    parser = argparse.ArgumentParser(
        description="Run the standalone poc stage for one CVE identifier."
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


def load_inputs(cve_id: str, dataset_root: str, workspace_root: str):
    """Load knowledge / build artifacts from prior stages."""

    from app.schemas.knowledge import KnowledgeModel
    from app.schemas.build_artifact import BuildArtifact

    workspace = Path(workspace_root) / cve_id

    knowledge_path = Path(dataset_root) / cve_id / "vuln_yaml" / "knowledge.yaml"
    build_artifact_path = workspace / "artifacts" / "build" / "build_artifact.yaml"

    for path in (knowledge_path, build_artifact_path):
        if not path.exists():
            raise FileNotFoundError(f"required input missing: {path}")

    knowledge = KnowledgeModel(**(yaml.safe_load(knowledge_path.read_text(encoding="utf-8")) or {}))
    build = BuildArtifact(**(yaml.safe_load(build_artifact_path.read_text(encoding="utf-8")) or {}))
    return knowledge, build, str(workspace)


def main() -> int:
    """Execute the poc stage and print the output file locations."""

    configure_console_streams()
    bootstrap_import_path()

    from app.stages.poc import PocStage, PocStagePaths

    parser = build_parser()
    args = parser.parse_args()

    knowledge, build, workspace = load_inputs(args.cve_id, args.dataset_root, args.workspace_root)
    stage = PocStage()
    result = stage.run(knowledge=knowledge, build=build, workspace=workspace)
    paths = PocStagePaths(workspace)

    print(f"PoC stage completed for {args.cve_id}.")
    print(f"Workspace: {paths.workspace_root}")
    print(f"PoC context YAML: {paths.poc_context_yaml}")
    print(f"PoC plan YAML: {paths.poc_plan_yaml}")
    print(f"Dockerfile: {paths.dockerfile}")
    print(f"Run script: {paths.run_script}")
    print(f"PoC log: {paths.poc_log}")
    print(f"Crash report: {paths.crash_report}")
    print(f"PoC artifact YAML: {paths.poc_artifact_yaml}")
    print(f"Run verify YAML: {paths.run_verify_yaml}")
    print(f"Execution success: {result.execution_success}")
    print(f"Reproducer verified: {result.reproducer_verified}")
    print(f"Trigger mode: {result.trigger_mode}")
    print(f"Target binary: {result.target_binary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
