#!/usr/bin/env python3
"""Check the portable observer core and launcher contracts for platform drift."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
WINDOWS_OBSERVER = ROOT / "setc" / "src" / "observer"
LINUX_OBSERVER = ROOT / "setc_linux" / "src" / "observer"
WINDOWS_REPLAY = ROOT / "setc" / "scripts" / "start_cpp_live_replay.bat"
LINUX_REPLAY = ROOT / "setc_linux" / "scripts" / "start_cpp_live_replay.sh"
WINDOWS_LIVE_INPUT = ROOT / "setc" / "scripts" / "start_cpp_live_input.bat"
LINUX_LIVE_INPUT = ROOT / "setc_linux" / "scripts" / "start_cpp_live_input.sh"

REPLAY_DEFAULTS = {
    "INPUT_GROUP_SIZE": "3",
    "INPUT_GROUP_STRIDE": "3",
    "GROUP_ANCHOR_INDEX": "1",
    "GROUP_MODEL_WIDTH": "406",
    "GROUP_MODEL_HEIGHT": "252",
    "TARGET_WIDTH": "700",
    "TARGET_SIZE": "700",
    "CANVAS_WIDTH": "770",
    "CANVAS_HEIGHT": "630",
    "FIRST_MODEL_WIDTH": "700",
    "FIRST_MODEL_HEIGHT": "434",
    "QUEUE_CAPACITY": "3",
    "PORT": "37651",
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def normalized_bytes(path: Path) -> bytes:
    """Normalize CRLF only; preserve all other source bytes for comparison."""

    return path.read_bytes().replace(b"\r\n", b"\n")


def relative_files(directory: Path) -> Dict[str, Path]:
    return {
        path.relative_to(directory).as_posix(): path
        for path in directory.rglob("*")
        if path.is_file()
    }


def check_observer_core() -> bool:
    problems: List[str] = []
    if not WINDOWS_OBSERVER.is_dir():
        problems.append(f"missing directory: {WINDOWS_OBSERVER}")
    if not LINUX_OBSERVER.is_dir():
        problems.append(f"missing directory: {LINUX_OBSERVER}")
    if problems:
        for problem in problems:
            print(f"[FAIL] observer core: {problem}")
        print("observer core: FAIL")
        return False

    windows_files = relative_files(WINDOWS_OBSERVER)
    linux_files = relative_files(LINUX_OBSERVER)
    missing_from_linux = sorted(set(windows_files) - set(linux_files))
    missing_from_windows = sorted(set(linux_files) - set(windows_files))
    for name in missing_from_linux:
        problems.append(f"only in setc: {name}")
    for name in missing_from_windows:
        problems.append(f"only in setc_linux: {name}")

    for name in sorted(set(windows_files) & set(linux_files)):
        if normalized_bytes(windows_files[name]) != normalized_bytes(linux_files[name]):
            problems.append(f"content differs: {name}")

    if problems:
        for problem in problems:
            print(f"[FAIL] observer core: {problem}")
        print("observer core: FAIL")
        return False

    print("observer core: PASS")
    return True


def parse_batch_assignments(text: str) -> Dict[str, List[Tuple[str, bool]]]:
    """Return SET assignments as (value, is_guarded_by_if-not-defined)."""

    assignments: Dict[str, List[Tuple[str, bool]]] = {}
    guarded_pattern = re.compile(
        r"^\s*if\s+not\s+defined\s+([A-Za-z_][A-Za-z0-9_]*)\s+"
        r"set\s+\"([^\"]*)\"\s*$",
        re.IGNORECASE,
    )
    set_pattern = re.compile(r'^\s*set\s+"([^"]*)"\s*$', re.IGNORECASE)

    for raw_line in text.splitlines():
        line = raw_line.strip()
        guarded_match = guarded_pattern.match(line)
        if guarded_match:
            variable = guarded_match.group(1)
            assignment = guarded_match.group(2)
            name, separator, value = assignment.partition("=")
            if separator and name.upper() == variable.upper():
                assignments.setdefault(variable.upper(), []).append((value, True))
            continue

        set_match = set_pattern.match(line)
        if not set_match:
            continue
        assignment = set_match.group(1)
        name, separator, value = assignment.partition("=")
        if separator and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            assignments.setdefault(name.upper(), []).append((value, False))
    return assignments


def parse_shell_defaults(text: str) -> Dict[str, str]:
    defaults: Dict[str, str] = {}
    pattern = re.compile(
        r'^\s*([A-Za-z_][A-Za-z0-9_]*)="\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}"\s*$'
    )
    for raw_line in text.splitlines():
        match = pattern.match(raw_line)
        if match and match.group(1) == match.group(2):
            defaults[match.group(1).upper()] = match.group(3)
    return defaults


def check_replay_contract() -> bool:
    problems: List[str] = []
    if not WINDOWS_REPLAY.is_file():
        problems.append(f"missing Windows launcher: {WINDOWS_REPLAY}")
    if not LINUX_REPLAY.is_file():
        problems.append(f"missing Linux launcher: {LINUX_REPLAY}")
    if problems:
        for problem in problems:
            print(f"[FAIL] replay contract: {problem}")
        print("replay contract: FAIL")
        return False

    windows_assignments = parse_batch_assignments(read_text(WINDOWS_REPLAY))
    linux_defaults = parse_shell_defaults(read_text(LINUX_REPLAY))

    for variable, expected in REPLAY_DEFAULTS.items():
        assignments = windows_assignments.get(variable, [])
        guarded = [value for value, is_guarded in assignments if is_guarded]
        unguarded = [value for value, is_guarded in assignments if not is_guarded]
        windows_value = guarded[-1] if guarded else (assignments[-1][0] if assignments else None)
        linux_value = linux_defaults.get(variable)

        if windows_value != expected or linux_value != expected:
            problems.append(
                f"{variable}: Windows={windows_value!r}, Linux={linux_value!r}, expected={expected!r}"
            )
        if not guarded:
            problems.append(f"{variable}: Windows default is not environment-overridable")
        if unguarded:
            problems.append(f"{variable}: Windows has unconditional assignment(s): {unguarded!r}")

    windows_replay = read_text(WINDOWS_REPLAY)
    stale_descriptions = (
        "lossless offline replay",
        "retain every frame",
        "server stays independent",
        "QUEUE_CAPACITY=1024",
    )
    for description in stale_descriptions:
        if description.lower() in windows_replay.lower():
            problems.append(f"Windows launcher still contains stale text: {description!r}")
    if "Stop-Process" not in windows_replay and "taskkill /F /IM omnivggt_stream_server.exe" not in windows_replay:
        problems.append("Windows launcher has no viewer-exit server cleanup")

    if problems:
        for problem in problems:
            print(f"[FAIL] replay contract: {problem}")
        print("replay contract: FAIL")
        return False

    print("replay contract: PASS")
    return True


def contains_all(text: str, fragments: Iterable[str], label: str, problems: List[str]) -> None:
    for fragment in fragments:
        if fragment not in text:
            problems.append(f"{label} is missing {fragment!r}")


def check_live_input_contract() -> bool:
    problems: List[str] = []
    if not WINDOWS_LIVE_INPUT.is_file():
        problems.append(f"missing Windows launcher: {WINDOWS_LIVE_INPUT}")
    if not LINUX_LIVE_INPUT.is_file():
        problems.append(f"missing Linux launcher: {LINUX_LIVE_INPUT}")
    if problems:
        for problem in problems:
            print(f"[FAIL] live-input contract: {problem}")
        print("live-input contract: FAIL")
        return False

    windows = read_text(WINDOWS_LIVE_INPUT)
    linux = read_text(LINUX_LIVE_INPUT)
    windows_assignments = parse_batch_assignments(windows)

    max_inflight = windows_assignments.get("MAX_INFLIGHT_GROUPS", [])
    if not any(value == "3" and guarded for value, guarded in max_inflight):
        problems.append("Windows MAX_INFLIGHT_GROUPS default is not guarded at 3")
    if "if not \"%MAX_INFLIGHT_GROUPS%\"==\"3\"" not in windows:
        problems.append("Windows MAX_INFLIGHT_GROUPS != 3 validation is missing")
    if "if not defined HISTORY_KEEP_GROUPS" not in windows:
        problems.append("Windows HISTORY_KEEP_GROUPS required check is missing")
    if "[ERROR] HISTORY_KEEP_GROUPS is required." not in windows:
        problems.append("Windows HISTORY_KEEP_GROUPS error text is missing")
    if "HISTORY_KEEP_GROUPS" in windows_assignments:
        problems.append("Windows HISTORY_KEEP_GROUPS must remain caller-provided without a default")
    if "--model-group3" in windows:
        problems.append("Windows live-input launcher must not use --model-group3")

    windows_fragments = (
        '--image-dir "%IMAGE_DIR%"',
        '--output-dir "%OUTPUT_DIR%"',
        '--model "%MODEL%"',
        '--model-pair "%PAIR_MODEL%"',
        "--pair-letterbox",
        "--input-group-size 3",
        "--input-group-stride 1",
        "--group-anchor-index 1",
        "--history-keep-groups %HISTORY_KEEP_GROUPS%",
        "--device cuda",
        "--dtype bf16",
    )
    linux_fragments = (
        '--image-dir "${IMAGE_DIR}"',
        '--output-dir "${OUTPUT_DIR}"',
        '--model "${MODEL}"',
        '--model-pair "${PAIR_MODEL}"',
        "--pair-letterbox",
        "--input-group-size 3",
        "--input-group-stride 1",
        "--group-anchor-index 1",
        '"${HISTORY_KEEP_GROUPS}"',
        "--device cuda",
        "--dtype bf16",
    )
    contains_all(windows, windows_fragments, "Windows live-input contract", problems)
    contains_all(linux, linux_fragments, "Linux live-input contract", problems)

    if problems:
        for problem in problems:
            print(f"[FAIL] live-input contract: {problem}")
        print("live-input contract: FAIL")
        return False

    print("live-input contract: PASS")
    return True


def main() -> int:
    observer_ok = check_observer_core()
    replay_ok = check_replay_contract()
    live_input_ok = check_live_input_contract()
    return 0 if observer_ok and replay_ok and live_input_ok else 1


if __name__ == "__main__":
    sys.exit(main())
