"""Harvest GitHub issue file attachments into dataset PoCs and recipes."""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
from urllib.parse import unquote, urlsplit

from app.schemas.knowledge import ReproductionRecipe

GITHUB_REPO_FILES_RE = re.compile(
    r"https://github\.com/[^/\s\"'<>]+/[^/\s\"'<>]+/files/\d+/[^\s\"'<>]+",
    re.IGNORECASE,
)
GITHUB_USER_ATTACHMENTS_RE = re.compile(
    r"https://github\.com/user-attachments/files/\d+/[^\s\"'<>]+",
    re.IGNORECASE,
)

ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")
TEXTISH_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".py",
    ".sh",
    ".js",
    ".ts",
    ".java",
    ".go",
    ".rs",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".css",
    ".lua",
    ".diff",
    ".patch",
    ".cil",
}
POC_NAME_HINTS = (
    "poc",
    "proof",
    "crash",
    "repro",
    "overflow",
    "asan",
    "fuzz",
    "trigger",
    "exploit",
    "sample",
    "testcase",
    "clusterfuzz",
)

# Keep recipe base64 payloads bounded so knowledge.yaml stays usable.
MAX_POC_BYTES_FOR_BASE64 = 256 * 1024

_EMBEDDED_PAYLOAD_HINTS = (
    "minimized cil",
    "minimized policy",
    "minimized",
    "here is a minimized",
    "reproduces the issue",
    "reproduces the bug",
    "proof of concept",
    "poc follows",
    "poc:",
    "reproducer:",
    "trigger policy",
    "cil policy",
)

_CIL_TOKEN_HINTS = (
    "classmap",
    "classmapping",
    "classpermission",
    "typeattribute",
    "(class ",
    "(type ",
    "(optional",
    "(block ",
    "(macro ",
    "(allow ",
    "(typealias",
)


def is_github_attachment_url(url: str) -> bool:
    """Return whether URL points at a GitHub issue/PR file attachment."""

    lowered = (url or "").lower()
    if "github.com" not in lowered:
        return False
    return "/files/" in lowered and (
        "/user-attachments/files/" in lowered or re.search(r"/[^/]+/[^/]+/files/\d+/", lowered) is not None
    )


def extract_github_attachment_urls(*texts: str) -> List[str]:
    """Extract GitHub attachment download URLs from HTML/markdown/plain text."""

    found: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if not text:
            continue
        for match in GITHUB_REPO_FILES_RE.finditer(text):
            url = _strip_trailing_punctuation(match.group(0))
            if url not in seen:
                seen.add(url)
                found.append(url)
        for match in GITHUB_USER_ATTACHMENTS_RE.finditer(text):
            url = _strip_trailing_punctuation(match.group(0))
            if url not in seen:
                seen.add(url)
                found.append(url)
    return found


def attachment_filename_from_url(url: str) -> str:
    """Best-effort filename from an attachment URL path."""

    path = unquote(urlsplit(url).path)
    name = Path(path).name
    return name or "attachment.bin"


def is_supported_archive_name(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith(ARCHIVE_SUFFIXES)


def looks_like_binary_bytes(data: bytes) -> bool:
    if not data:
        return False
    sample = data[:8192]
    if b"\x00" in sample:
        return True
    # High ratio of non-text bytes suggests a binary PoC.
    nontext = sum(1 for byte in sample if byte < 9 or (13 < byte < 32) or byte == 127)
    return (nontext / max(len(sample), 1)) > 0.30


def is_likely_poc_file(path: Path, data: Optional[bytes] = None) -> bool:
    """Heuristic: file looks like a reproducer payload rather than source/docs."""

    name = path.name.lower()
    if is_supported_archive_name(name):
        return False
    if name.startswith(".") or name in {"makefile", "cmakelists.txt", "readme", "license"}:
        return False

    payload = data if data is not None else path.read_bytes()
    if not payload or len(payload) > MAX_POC_BYTES_FOR_BASE64:
        return False

    if any(hint in name for hint in POC_NAME_HINTS):
        return True
    if looks_like_binary_bytes(payload):
        return True

    suffix = path.suffix.lower()
    if suffix in {".bin", ".abc", ".dat", ".raw", ".crash", ".input"}:
        return True
    if not suffix and looks_like_binary_bytes(payload):
        return True
    if suffix in TEXTISH_SUFFIXES and any(hint in name for hint in POC_NAME_HINTS):
        return True
    return False


def infer_run_commands_from_text(text: str, payload_name: str, limit: int = 3) -> List[str]:
    """Infer target invocation lines that mention a PoC/payload placeholder."""

    if not text:
        return []

    commands: list[str] = []
    seen: set[str] = set()
    payload_token = payload_name or "poc"
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        stripped = re.sub(r"^\d+[\).\:\-]\s*", "", stripped).strip()
        stripped = stripped.strip("`")
        lowered = stripped.lower()
        if not any(token in lowered for token in ("poc", "repro", "proof", payload_token.lower(), "[poc]", "{poc}")):
            continue
        if not _looks_like_run_command(stripped):
            continue
        command = re.sub(
            r"(\[poc\]|\{poc\}|\bpoc\.(?:bin|zip|abc|txt|dat)\b|\bpoc\b)",
            f"./{payload_token}",
            stripped,
            count=1,
            flags=re.IGNORECASE,
        )
        command = re.sub(r"\s+", " ", command).strip()
        if command in seen:
            continue
        seen.add(command)
        commands.append(command)
        if len(commands) >= limit:
            break
    return commands


def build_attachment_recipe(
    *,
    filename: str,
    payload: bytes,
    source_url: str = "",
    source_title: str = "",
    run_commands: Optional[Sequence[str]] = None,
    expected_behavior: Optional[Sequence[str]] = None,
) -> ReproductionRecipe:
    """Build a high-confidence recipe that materializes a harvested attachment."""

    encoded = base64.b64encode(payload).decode("ascii")
    artifact_cmd = f"printf '%s' '{encoded}' | base64 -d > {filename}"
    runs = list(run_commands or [])
    steps = [artifact_cmd, *runs] if runs else [artifact_cmd]
    return ReproductionRecipe(
        source_url=source_url,
        source_title=source_title or filename,
        recipe_type="attachment_binary",
        steps=steps,
        artifact_generation_commands=[artifact_cmd],
        run_commands=runs,
        expected_behavior=list(expected_behavior or ["AddressSanitizer", "SEGV", "heap-buffer-overflow"]),
        source_excerpt=f"Harvested attachment {filename} ({len(payload)} bytes).",
        confidence="high",
    )


def harvest_poc_files(
    search_roots: Iterable[Path],
    output_dir: Path,
    *,
    evidence_text: str = "",
    source_url: str = "",
    limit: int = 4,
) -> List[ReproductionRecipe]:
    """Copy likely PoC files into output_dir and return base64 recipes."""

    output_dir.mkdir(parents=True, exist_ok=True)
    recipes: list[ReproductionRecipe] = []
    seen_hashes: set[str] = set()

    candidates: list[Path] = []
    for root in search_roots:
        if root is None or not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                candidates.append(path)

    for path in candidates:
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if not is_likely_poc_file(path, data):
            continue

        content_key = hashlib.sha256(data).hexdigest()
        if content_key in seen_hashes:
            continue
        seen_hashes.add(content_key)

        dest_name = _destination_filename(output_dir, path.name, data)
        dest = output_dir / dest_name
        if path.resolve() != dest.resolve():
            dest.write_bytes(data)

        run_commands = infer_run_commands_from_text(evidence_text, dest_name)
        recipes.append(
            build_attachment_recipe(
                filename=dest_name,
                payload=data,
                source_url=source_url,
                source_title=dest_name,
                run_commands=run_commands,
            )
        )
        if len(recipes) >= limit:
            break

    return recipes


def looks_like_cil_or_sexpr_payload(content: str) -> bool:
    """Return whether text looks like a CIL / S-expression policy PoC."""

    if not content or len(content) > MAX_POC_BYTES_FOR_BASE64:
        return False
    lowered = content.lower()
    if any(token in lowered for token in _CIL_TOKEN_HINTS):
        return True
    return content.count("(") >= 4 and content.count(")") >= 4


def extract_embedded_sexpr_payloads(text: str, *, min_lines: int = 3) -> List[str]:
    """Extract embedded S-expression / CIL policy blocks from advisory or commit text."""

    if not text:
        return []

    lines = text.splitlines()
    blocks: list[str] = []
    seen: set[str] = set()
    index = 0
    while index < len(lines):
        line = lines[index]
        lowered = line.lower()
        hinted = any(hint in lowered for hint in _EMBEDDED_PAYLOAD_HINTS)
        stripped = line.strip()
        if not hinted and not stripped.startswith("("):
            index += 1
            continue

        cursor = index + 1 if hinted else index
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1

        block_lines: list[str] = []
        while cursor < len(lines):
            candidate_stripped = lines[cursor].strip()
            if not candidate_stripped:
                if block_lines:
                    break
                cursor += 1
                continue
            if candidate_stripped.startswith(("+", "-", "@@", "diff ", "index ", "commit ")):
                break
            if candidate_stripped.startswith("(") or (
                block_lines and (candidate_stripped.startswith((")", ";")) or candidate_stripped.endswith(")"))
            ):
                block_lines.append(candidate_stripped)
                cursor += 1
                continue
            break

        if len(block_lines) >= min_lines:
            content = "\n".join(block_lines).rstrip() + "\n"
            if looks_like_cil_or_sexpr_payload(content) and content not in seen:
                seen.add(content)
                blocks.append(content)
            index = max(cursor, index + 1)
            continue
        index += 1
    return blocks


def infer_embedded_payload_filename(content: str, index: int = 0) -> str:
    """Choose a stable filename for an embedded text PoC."""

    suffix = ".cil" if looks_like_cil_or_sexpr_payload(content) else ".txt"
    if index <= 0:
        return f"poc{suffix}"
    return f"poc_{index}{suffix}"


def harvest_embedded_text_pocs(
    *,
    output_dir: Path,
    evidence_pages: Sequence[tuple[str, str, str]],
    limit: int = 4,
) -> List[ReproductionRecipe]:
    """Persist embedded text PoCs from page/commit bodies into vuln_pocs and build recipes.

    evidence_pages entries are (url, title, cleaned_text).
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    recipes: list[ReproductionRecipe] = []
    seen_hashes: set[str] = set()
    payload_index = 0

    for source_url, source_title, cleaned_text in evidence_pages:
        if not cleaned_text:
            continue
        for block in extract_embedded_sexpr_payloads(cleaned_text):
            data = block.encode("utf-8")
            content_key = hashlib.sha256(data).hexdigest()
            if content_key in seen_hashes:
                continue
            seen_hashes.add(content_key)
            filename = infer_embedded_payload_filename(block, payload_index)
            payload_index += 1
            dest_name = _destination_filename(output_dir, filename, data)
            dest = output_dir / dest_name
            dest.write_bytes(data)
            run_commands = infer_run_commands_from_text(cleaned_text, dest_name)
            recipes.append(
                build_attachment_recipe(
                    filename=dest_name,
                    payload=data,
                    source_url=source_url,
                    source_title=source_title or dest_name,
                    run_commands=run_commands,
                    expected_behavior=["AddressSanitizer", "SEGV", "heap-use-after-free", "use-after-free"],
                )
            )
            recipes[-1].recipe_type = "embedded_text_payload"
            recipes[-1].source_excerpt = (
                f"Harvested embedded text PoC from {source_url or source_title or 'evidence'} "
                f"({len(data)} bytes)."
            )
            if len(recipes) >= limit:
                return recipes
    return recipes


def _destination_filename(directory: Path, preferred_name: str, data: bytes) -> str:
    safe = re.sub(r"[^\w.\-]+", "_", preferred_name).strip("._") or "poc.bin"
    candidate = safe
    index = 1
    while True:
        target = directory / candidate
        if not target.exists():
            return candidate
        try:
            if target.read_bytes() == data:
                return candidate
        except OSError:
            pass
        stem = Path(safe).stem
        suffix = Path(safe).suffix
        candidate = f"{stem}_{index}{suffix}"
        index += 1


def _strip_trailing_punctuation(url: str) -> str:
    return url.rstrip(").,;\"'")


def _looks_like_run_command(line: str) -> bool:
    lowered = line.lower()
    if lowered.startswith(("http://", "https://", "#", "//", "/*")):
        return False
    if any(token in lowered for token in ("git clone", "wget ", "curl ", "unzip ", "tar ")):
        return False
    return bool(
        re.match(r"^(\./|/|[A-Za-z0-9_.\-]+)", line)
        and (
            " -" in line
            or line.startswith("./")
            or any(token in lowered for token in ("lua ", "python", "node ", "perl ", "ruby "))
        )
    )
