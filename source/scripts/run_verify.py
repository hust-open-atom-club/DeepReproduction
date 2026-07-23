"""Run the standalone verify stage for a single CVE identifier."""

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
    """Build the command-line interface for manual verify-stage testing."""

    parser = argparse.ArgumentParser(
        description="Run the standalone verify stage for one CVE identifier."
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
    """Load knowledge / build / poc artifacts from prior stages."""

    from app.schemas.knowledge import KnowledgeModel
    from app.schemas.build_artifact import BuildArtifact
    from app.schemas.poc_artifact import PoCArtifact

    workspace = Path(workspace_root) / cve_id

    knowledge_path = Path(dataset_root) / cve_id / "vuln_yaml" / "knowledge.yaml"
    build_artifact_path = workspace / "artifacts" / "build" / "build_artifact.yaml"
    poc_artifact_path = workspace / "artifacts" / "poc" / "poc_artifact.yaml"

    for path in (knowledge_path, build_artifact_path, poc_artifact_path):
        if not path.exists():
            raise FileNotFoundError(f"required input missing: {path}")

    knowledge = KnowledgeModel(**(yaml.safe_load(knowledge_path.read_text(encoding="utf-8")) or {}))
    build = BuildArtifact(**(yaml.safe_load(build_artifact_path.read_text(encoding="utf-8")) or {}))
    poc = PoCArtifact(**(yaml.safe_load(poc_artifact_path.read_text(encoding="utf-8")) or {}))
    return knowledge, build, poc, str(workspace)


def main() -> int:
    """Execute the verify stage and print the output file locations."""

    configure_console_streams()
    bootstrap_import_path()

    from app.stages.verify import VerifyStage, VerifyStagePaths

    parser = build_parser()
    args = parser.parse_args()

    knowledge, build, poc, workspace = load_inputs(args.cve_id, args.dataset_root, args.workspace_root)
    stage = VerifyStage()
    result = stage.run(
        knowledge=knowledge,
        build=build,
        poc=poc,
        workspace=workspace,
        dataset_root=args.dataset_root,
    )
    paths = VerifyStagePaths(workspace)

    print(f"Verify stage completed for {args.cve_id}.")
    print(f"Workspace: {paths.workspace_root}")
    print(f"Verify dir: {paths.verify_dir}")
    print(f"Pre-patch log: {paths.pre_patch_log}")
    print(f"Post-patch log: {paths.post_patch_log}")
    print(f"Verify result YAML: {paths.verify_result_yaml}")
    print(f"Verdict: {result.verdict}")
    print(f"Reason: {result.reason}")
    print(f"Confidence: {result.confidence}")
    return 0 if result.verdict == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
