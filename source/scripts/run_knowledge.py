"""Run the standalone knowledge stage for a single CVE identifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def configure_console_streams() -> None:
    """Avoid Windows console encoding crashes when summaries contain Unicode."""

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
    """Build the command-line interface for manual knowledge-stage testing."""

    parser = argparse.ArgumentParser(
        description="Run the standalone knowledge stage for one CVE identifier."
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
    return parser


def main() -> int:
    """Execute the knowledge stage and print the output file locations."""

    configure_console_streams()
    bootstrap_import_path()

    from app.stages.knowledge import build_knowledge_paths, run_knowledge_agent

    parser = build_parser()
    args = parser.parse_args()

    result = run_knowledge_agent(args.cve_id, dataset_root=args.dataset_root)
    paths = build_knowledge_paths(args.cve_id, dataset_root=args.dataset_root)

    print(f"Knowledge stage completed for {args.cve_id}.")
    print(f"Task YAML: {paths.task_yaml}")
    print(f"Knowledge YAML: {paths.knowledge_yaml}")
    print(f"Runtime state YAML: {paths.runtime_state_yaml}")
    print(f"Patch diff: {paths.patch_diff}")
    print(f"Knowledge summary: {result.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
