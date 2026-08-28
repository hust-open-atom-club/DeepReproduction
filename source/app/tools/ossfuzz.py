"""Harvest OSS-Fuzz / ClusterFuzz reproducer testcases into dataset PoCs."""

from __future__ import annotations

import hashlib
import re
from email.message import Message
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
from urllib.parse import parse_qs, unquote, urlsplit
from urllib.request import Request, urlopen

from app.schemas.knowledge import ReproductionRecipe
from app.tools.poc_attachments import (
    MAX_POC_BYTES_FOR_BASE64,
    build_attachment_recipe,
    looks_like_cil_or_sexpr_payload,
)

OSSFUZZ_MONORAIL_RE = re.compile(
    r"https?://bugs\.chromium\.org/p/oss-fuzz/issues/detail\?[^\s\"'<>]*\bid=(\d+)",
    re.IGNORECASE,
)
OSSFUZZ_ISSUE_TRACKER_RE = re.compile(
    r"https?://issues\.oss-fuzz\.com/(?:issues/)?(\d+)\b",
    re.IGNORECASE,
)
OSSFUZZ_DOWNLOAD_RE = re.compile(
    r"https?://oss-fuzz\.com/download\?[^\s\"'<>]*testcase_id(?:=|%3D|\\u003d)(\d+)",
    re.IGNORECASE,
)
TESTCASE_ID_RE = re.compile(
    r"testcase_id(?:=|%3D|\\u003d)(\d+)",
    re.IGNORECASE,
)
CONTENT_DISPOSITION_FILENAME_RE = re.compile(
    r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?',
    re.IGNORECASE,
)

_USER_AGENT = "DeepReproductionKnowledgeAgent/1.0"


def is_ossfuzz_issue_url(url: str) -> bool:
    """Return whether URL points at an OSS-Fuzz issue (Monorail or Issue Tracker)."""

    lowered = (url or "").lower()
    if "bugs.chromium.org/p/oss-fuzz/issues/detail" in lowered:
        return True
    if "issues.oss-fuzz.com/" in lowered:
        return True
    return False


def is_ossfuzz_download_url(url: str) -> bool:
    lowered = (url or "").lower()
    return "oss-fuzz.com/download" in lowered and "testcase_id" in lowered


def extract_ossfuzz_issue_urls(*texts: str) -> List[str]:
    """Extract OSS-Fuzz issue URLs from HTML/markdown/plain text."""

    found: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if not text:
            continue
        for match in OSSFUZZ_MONORAIL_RE.finditer(text):
            url = _strip_trailing_punctuation(match.group(0))
            if url not in seen:
                seen.add(url)
                found.append(url)
        for match in OSSFUZZ_ISSUE_TRACKER_RE.finditer(text):
            url = _strip_trailing_punctuation(match.group(0))
            if url not in seen:
                seen.add(url)
                found.append(url)
    return found


def extract_ossfuzz_testcase_ids(*texts: str) -> List[str]:
    """Extract ClusterFuzz/OSS-Fuzz testcase ids from text or escaped JSON."""

    found: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if not text:
            continue
        for match in TESTCASE_ID_RE.finditer(text):
            testcase_id = match.group(1)
            if testcase_id not in seen:
                seen.add(testcase_id)
                found.append(testcase_id)
        for match in OSSFUZZ_DOWNLOAD_RE.finditer(text):
            testcase_id = match.group(1)
            if testcase_id not in seen:
                seen.add(testcase_id)
                found.append(testcase_id)
    return found


def monorail_id_from_url(url: str) -> Optional[str]:
    match = OSSFUZZ_MONORAIL_RE.search(url or "")
    if match:
        return match.group(1)
    parts = urlsplit(url or "")
    if "bugs.chromium.org" in parts.netloc.lower() and "/p/oss-fuzz/issues/detail" in parts.path.lower():
        values = parse_qs(parts.query).get("id") or []
        return values[0] if values else None
    return None


def issue_tracker_id_from_url(url: str) -> Optional[str]:
    match = OSSFUZZ_ISSUE_TRACKER_RE.search(url or "")
    return match.group(1) if match else None


def resolve_issue_tracker_id(issue_url: str, *, timeout: int = 30) -> Optional[str]:
    """Resolve Monorail/Issue Tracker issue id usable with issues.oss-fuzz.com/action."""

    tracker_id = issue_tracker_id_from_url(issue_url)
    if tracker_id:
        return tracker_id

    monorail_id = monorail_id_from_url(issue_url)
    if not monorail_id:
        return None

    redirect_html = _http_get_text(
        f"https://bugs.chromium.org/p/oss-fuzz/issues/detail?id={monorail_id}",
        timeout=timeout,
    )
    if not redirect_html:
        return None
    match = OSSFUZZ_ISSUE_TRACKER_RE.search(redirect_html)
    return match.group(1) if match else None


def fetch_issue_action_payload(tracker_id: str, *, timeout: int = 30) -> str:
    """Fetch the public Issue Tracker action JSON for an OSS-Fuzz issue."""

    url = f"https://issues.oss-fuzz.com/action/issues/{tracker_id}?pageSize=25"
    text = _http_get_text(url, timeout=timeout, accept="application/json,text/plain,*/*")
    if text.startswith(")]}'"):
        text = text[4:].lstrip()
    return text


def download_testcase(
    testcase_id: str,
    output_dir: Path,
    *,
    timeout: int = 60,
    preferred_name: str = "",
) -> Optional[Path]:
    """Download one OSS-Fuzz reproducer into output_dir."""

    url = f"https://oss-fuzz.com/download?testcase_id={testcase_id}"
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        payload = response.read()
        content_disposition = response.headers.get("Content-Disposition") or ""
        final_url = response.geturl()

    if not payload or len(payload) > MAX_POC_BYTES_FOR_BASE64:
        return None

    filename = preferred_name or _filename_from_content_disposition(content_disposition)
    if not filename:
        filename = Path(urlsplit(final_url).path).name or f"ossfuzz-testcase-{testcase_id}"
    filename = _sanitize_filename(filename)
    if "." not in filename and looks_like_cil_or_sexpr_payload(payload.decode("utf-8", errors="replace")):
        filename = f"{filename}.cil"

    output_dir.mkdir(parents=True, exist_ok=True)
    dest_name = _destination_filename(output_dir, filename, payload)
    dest = output_dir / dest_name
    dest.write_bytes(payload)
    return dest


def harvest_ossfuzz_testcases(
    *,
    output_dir: Path,
    issue_urls: Sequence[str],
    timeout: int = 30,
    limit: int = 4,
) -> List[ReproductionRecipe]:
    """Resolve OSS-Fuzz issues, download reproducers, and build high-confidence recipes."""

    recipes: list[ReproductionRecipe] = []
    seen_hashes: set[str] = set()
    seen_issue_keys: set[str] = set()

    for issue_url in issue_urls:
        if len(recipes) >= limit:
            break
        if not issue_url or not is_ossfuzz_issue_url(issue_url):
            continue
        issue_key = monorail_id_from_url(issue_url) or issue_tracker_id_from_url(issue_url) or issue_url
        if issue_key in seen_issue_keys:
            continue
        seen_issue_keys.add(issue_key)

        try:
            tracker_id = resolve_issue_tracker_id(issue_url, timeout=timeout)
            if not tracker_id:
                continue
            payload = fetch_issue_action_payload(tracker_id, timeout=timeout)
            testcase_ids = extract_ossfuzz_testcase_ids(payload)
        except Exception:
            continue

        for testcase_id in testcase_ids:
            if len(recipes) >= limit:
                break
            try:
                dest = download_testcase(testcase_id, output_dir, timeout=max(timeout, 60))
            except Exception:
                continue
            if dest is None or not dest.exists():
                continue
            data = dest.read_bytes()
            content_key = hashlib.sha256(data).hexdigest()
            if content_key in seen_hashes:
                continue
            seen_hashes.add(content_key)
            recipe = build_attachment_recipe(
                filename=dest.name,
                payload=data,
                source_url=issue_url,
                source_title=dest.name,
                run_commands=[],
                expected_behavior=["AddressSanitizer", "SEGV", "heap-use-after-free", "heap-buffer-overflow"],
            )
            recipe.recipe_type = "ossfuzz_testcase"
            recipe.source_excerpt = (
                f"Harvested OSS-Fuzz testcase {testcase_id} from {issue_url} ({len(data)} bytes)."
            )
            recipe.confidence = "high"
            recipes.append(recipe)
    return recipes


def collect_ossfuzz_issue_urls(
    *,
    reference_urls: Iterable[str] = (),
    page_texts: Iterable[str] = (),
) -> List[str]:
    """Collect unique OSS-Fuzz issue URLs from references and page bodies."""

    found: list[str] = []
    seen: set[str] = set()
    for url in reference_urls:
        if is_ossfuzz_issue_url(url) and url not in seen:
            seen.add(url)
            found.append(url)
    for text in page_texts:
        for url in extract_ossfuzz_issue_urls(text):
            if url not in seen:
                seen.add(url)
                found.append(url)
    return found


def _http_get_text(url: str, *, timeout: int = 30, accept: str = "text/html,*/*") -> str:
    request = Request(url, headers={"User-Agent": _USER_AGENT, "Accept": accept})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
    return raw.decode(charset, errors="replace")


def _filename_from_content_disposition(header: str) -> str:
    if not header:
        return ""
    # Prefer stdlib parsing when available.
    message = Message()
    message["Content-Disposition"] = header
    filename = message.get_filename()
    if filename:
        return unquote(filename)
    match = CONTENT_DISPOSITION_FILENAME_RE.search(header)
    return unquote(match.group(1).strip()) if match else ""


def _sanitize_filename(value: str) -> str:
    safe = re.sub(r"[^\w.\-]+", "_", value).strip("._")
    return safe or "ossfuzz_testcase.bin"


def _destination_filename(directory: Path, preferred_name: str, data: bytes) -> str:
    safe = _sanitize_filename(preferred_name)
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


# ClusterFuzz / OSS-Fuzz reproducer filenames encode the harness, e.g.
#   clusterfuzz-testcase-minimized-flb-it-fuzz-parser_fuzzer_OSSFUZZ-123.fuzz
#   clusterfuzz-testcase-minimized-secilc-fuzzer-456.cil
_CLUSTERFUZZ_PREFIXES = (
    "clusterfuzz-testcase-minimized-",
    "clusterfuzz-testcase-",
    "crash-",
)
_OSSFUZZ_ID_TAIL_RE = re.compile(r"(?i)(?:_OSSFUZZ|[-_]OSSFUZZ)?[-_]?\d{4,}$")
_CLUSTERFUZZ_TOKEN_RE = re.compile(
    r"(?i)\b(?:clusterfuzz-testcase(?:-minimized)?-|crash-)[\w.\-]+",
)


def parse_ossfuzz_harness_name(filename: str) -> Optional[str]:
    """Extract the fuzzer/harness name from a ClusterFuzz testcase filename.

    Returns None when the string does not look like a ClusterFuzz reproducer name.
    """

    if not filename:
        return None
    stem = Path(str(filename).replace("\\", "/").split("/")[-1]).name
    # Drop compound suffixes like .fuzz / .cil while keeping harness tokens.
    lowered = stem.lower()
    body = stem
    matched_prefix = False
    for prefix in _CLUSTERFUZZ_PREFIXES:
        if lowered.startswith(prefix):
            body = stem[len(prefix) :]
            matched_prefix = True
            break
    if not matched_prefix:
        return None
    # Strip extension(s).
    while True:
        nxt = Path(body)
        if not nxt.suffix or nxt.stem == body:
            break
        body = nxt.stem
    body = _OSSFUZZ_ID_TAIL_RE.sub("", body).strip("-_")
    if len(body) < 3:
        return None
    return body


def extract_ossfuzz_harness_names(*texts: str) -> List[str]:
    """Collect unique harness names mentioned in free-form hint/recipe text."""

    found: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if not text:
            continue
        candidates = list(_CLUSTERFUZZ_TOKEN_RE.findall(text))
        # Also accept a bare filename line.
        stripped = text.strip().split()[-1] if text.strip() else ""
        if stripped:
            candidates.append(stripped)
        for candidate in candidates:
            harness = parse_ossfuzz_harness_name(candidate)
            if harness and harness not in seen:
                seen.add(harness)
                found.append(harness)
    return found


_HARNESS_SOURCE_EXTS = (".c", ".cc", ".cpp", ".cxx")


def standalone_ossfuzz_harness_relpath(repo_path: Path, harness: str) -> Optional[str]:
    """Return ``ossfuzz/<harness>.{c,cc,cpp,cxx}`` when that standalone source exists.

    Projects like matio ship the ClusterFuzz harness under ``ossfuzz/`` and compile it
    outside CMake (see ``ossfuzz/build.sh``). This is distinct from in-tree cmake
    fuzzer targets (fluent-bit).
    """

    if not harness or not repo_path or not Path(repo_path).is_dir():
        return None
    ossfuzz_dir = Path(repo_path) / "ossfuzz"
    if not ossfuzz_dir.is_dir():
        return None
    for ext in _HARNESS_SOURCE_EXTS:
        candidate = ossfuzz_dir / f"{harness}{ext}"
        if candidate.is_file():
            return f"ossfuzz/{harness}{ext}"
    return None


def in_tree_fuzz_mk_harness_relpath(repo_path: Path, harness: str) -> Optional[str]:
    """Return ``fuzz/<OUTPUT_DIR>/<harness>`` for qpdf-style ``fuzz/build.mk`` trees.

    Gate (all required):
    - ``fuzz/<harness>.{c,cc,cpp,cxx}`` exists
    - ``fuzz/build.mk`` mentions the harness and ``fuzz/$(OUTPUT_DIR)/...`` targets
    - OSS-Fuzz wiring exists via ``fuzz/oss-fuzz-build`` and/or
      ``configure.ac`` ``--enable-oss-fuzz``

    Without ``--enable-oss-fuzz``, qpdf still builds non-static fuzzers under
    ``fuzz/build/`` (linked with ``standalone_fuzz_target_runner``). The bare
    make goal ``qpdf_fuzzer`` / path ``fuzz/qpdf_fuzzer`` are both wrong.
    """

    if not harness or not repo_path or not Path(repo_path).is_dir():
        return None
    root = Path(repo_path)
    fuzz_dir = root / "fuzz"
    build_mk = fuzz_dir / "build.mk"
    if not build_mk.is_file():
        return None

    source_found = False
    for ext in _HARNESS_SOURCE_EXTS:
        if (fuzz_dir / f"{harness}{ext}").is_file():
            source_found = True
            break
    if not source_found:
        return None

    try:
        build_mk_text = build_mk.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    if harness not in build_mk_text:
        return None
    if "fuzz/$(OUTPUT_DIR)" not in build_mk_text:
        return None

    has_oss_fuzz_build = (fuzz_dir / "oss-fuzz-build").is_file()
    has_enable_oss_fuzz = False
    configure_ac = root / "configure.ac"
    if configure_ac.is_file():
        try:
            ac_text = configure_ac.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            ac_text = ""
        if "enable-oss-fuzz" in ac_text or "enable_oss_fuzz" in ac_text:
            has_enable_oss_fuzz = True
    if not has_oss_fuzz_build and not has_enable_oss_fuzz:
        return None

    output_dir = "build"
    makefile = root / "Makefile"
    if makefile.is_file():
        try:
            make_text = makefile.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            make_text = ""
        match = re.search(r"^OUTPUT_DIR\s*=\s*(\S+)", make_text, re.MULTILINE)
        if match:
            output_dir = match.group(1).strip()
    return f"fuzz/{output_dir}/{harness}"


def harness_source_evidence(repo_path: Path, harness: str) -> bool:
    """True when the repo clearly contains this OSS-Fuzz harness (source or binary).

    Used as a safety gate: SELinux clusterfuzz payloads name ``secilc-fuzzer`` but
    the in-tree entrypoint is ``secilc`` — without evidence we must not override.
    """

    if not harness or not repo_path or not Path(repo_path).is_dir():
        return False
    root = Path(repo_path)
    for rel in (f"build/bin/{harness}", f"bin/{harness}", harness):
        if (root / rel).is_file():
            return True

    if standalone_ossfuzz_harness_relpath(root, harness):
        return True

    short = harness
    flb_match = re.match(r"(?i)^flb-it-fuzz-(.+)$", harness)
    if flb_match:
        short = flb_match.group(1)

    fuzz_dirs = (
        root / "tests" / "internal" / "fuzzers",
        root / "fuzz",
        root / "fuzzers",
        root / "test" / "fuzz",
        root / "tests" / "fuzz",
        root / "ossfuzz",
    )
    for fuzz_dir in fuzz_dirs:
        if not fuzz_dir.is_dir():
            continue
        for base in (harness, short):
            if not base:
                continue
            for ext in _HARNESS_SOURCE_EXTS:
                if (fuzz_dir / f"{base}{ext}").is_file():
                    return True
        cmake = fuzz_dir / "CMakeLists.txt"
        if cmake.is_file():
            try:
                text = cmake.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                text = ""
            if harness in text or (short and f"{short}.c" in text):
                return True

    # Limited top-level / tests CMake mention of the exact harness target.
    for cmake in (
        root / "CMakeLists.txt",
        root / "tests" / "CMakeLists.txt",
        root / "tests" / "internal" / "CMakeLists.txt",
    ):
        if not cmake.is_file():
            continue
        try:
            if harness in cmake.read_text(encoding="utf-8", errors="ignore"):
                return True
        except OSError:
            continue
    return False


def preferred_harness_relpath(repo_path: Path, harness: str) -> Optional[str]:
    """Return a repo-relative binary path for a harness when evidence exists."""

    if not harness_source_evidence(repo_path, harness):
        return None
    root = Path(repo_path)
    for rel in (f"build/bin/{harness}", f"bin/{harness}", harness):
        if (root / rel).is_file():
            return rel
    # Standalone ossfuzz/*.cpp harnesses are linked to ./<harness> after the
    # library build (see BuildStage._append_standalone_ossfuzz_harness_compile).
    if standalone_ossfuzz_harness_relpath(root, harness):
        return harness
    # qpdf-style fuzz/build.mk → fuzz/build/<harness> (before cmake default).
    fuzz_mk_rel = in_tree_fuzz_mk_harness_relpath(root, harness)
    if fuzz_mk_rel:
        return fuzz_mk_rel
    # fluent-bit and similar cmake trees emit fuzzers under build/bin/.
    if (root / "tests" / "internal" / "fuzzers").is_dir() or (root / "CMakeLists.txt").is_file():
        return f"build/bin/{harness}"
    return harness
