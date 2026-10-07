"""Build and audit the public source release without third-party data."""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
import re
import sys
import zipfile


INCLUDED = (
    ".gitignore", "CITATION.cff", "DATA_ACCESS.md", "LICENSE", "README.md", "build_release.py",
    "RELEASE_NOTES_v1.3.1.md", "requirements.txt", "analysis", "direction_reset",
    "figures", "manuscript", "results", "tests",
)
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".venv"}
EXCLUDED_SUFFIXES = {".aux", ".bbl", ".blg", ".log", ".out"}
FORBIDDEN_SUFFIXES = {".parquet", ".feather", ".db", ".sqlite", ".xlsx", ".xls",
                      ".pkl", ".joblib", ".zip", ".7z", ".rar"}
FORBIDDEN_COLUMNS = {"participant_id", "pid", "study_id", "device_id", "latitude",
                     "longitude", "timestamp", "date", "time"}
TEXT_SUFFIXES = {".py", ".md", ".tex", ".json", ".txt", ".csv", ".cff"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def release_files(stage: Path) -> list[Path]:
    paths: list[Path] = []
    for name in INCLUDED:
        target = stage / name
        if not target.exists():
            raise FileNotFoundError(target)
        paths.extend([target] if target.is_file() else target.rglob("*"))
    files = sorted(path for path in paths if path.is_file())
    return [path for path in files
            if not any(part in EXCLUDED_PARTS for part in path.relative_to(stage).parts)
            and path.suffix.lower() not in EXCLUDED_SUFFIXES]


def contains_private_absolute_path(value: str) -> bool:
    windows_user = re.compile(r"[A-Za-z]:[\\/]Users[\\/][^\\/\s]+", re.IGNORECASE)
    project_root = re.compile(r"[A-Za-z]:[\\/]psychology[\\/]psychology_b", re.IGNORECASE)
    return bool(windows_user.search(value) or project_root.search(value))


def audit(path: Path, stage: Path) -> bytes:
    relative = path.relative_to(stage).as_posix()
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ValueError(f"Forbidden file type: {relative}")
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        value = data.decode("utf-8-sig", errors="replace")
        if contains_private_absolute_path(value):
            raise ValueError(f"Local private path in {relative}")
        if path.suffix.lower() == ".csv":
            columns = next(csv.reader(io.StringIO(value)), [])
            if any(column.strip().lower() in FORBIDDEN_COLUMNS for column in columns):
                raise ValueError(f"Potential participant-level column in {relative}")
    return data


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python build_release.py OUTPUT_ZIP")
    stage = Path(__file__).resolve().parent
    output = Path(sys.argv[1]).resolve()
    if output.is_relative_to(stage):
        raise ValueError("Output ZIP must be outside the repository")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing ZIP: {output}")
    files = release_files(stage)
    audited = [(path.relative_to(stage).as_posix(), audit(path, stage)) for path in files]
    manifest = "".join(f"{digest(data)}  {relative}\n" for relative, data in audited).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative, data in audited:
            archive.writestr(relative, data)
        archive.writestr("MANIFEST.sha256", manifest)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC test failed")
        for expected, relative in (line.split("  ", 1) for line in manifest.decode().splitlines()):
            if digest(archive.read(relative)) != expected:
                raise RuntimeError(f"Manifest mismatch: {relative}")
    print(f"Created {output.name}: {len(audited)} files, SHA-256 {digest(output.read_bytes())}")


if __name__ == "__main__":
    main()
