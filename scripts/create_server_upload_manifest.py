from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INCLUDE = (
    "configs",
    "data/real/chartqa_medium",
    "docs",
    "scripts",
    "src",
    "tests",
    "PROJECT_GUIDE.md",
    "README.md",
    "requirements-models.txt",
    "requirements-real.txt",
    "requirements-server.txt",
)
EXCLUDED_PARTS = {"__pycache__", ".DS_Store"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a checksum manifest for server upload.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "server_upload_manifest.json",
    )
    args = parser.parse_args()

    files: list[Path] = []
    missing: list[str] = []
    for relative in DEFAULT_INCLUDE:
        path = PROJECT_ROOT / relative
        if not path.exists():
            missing.append(relative)
            continue
        candidates = path.rglob("*") if path.is_dir() else [path]
        files.extend(candidate for candidate in candidates if _include_file(candidate))
    files = sorted(set(files), key=lambda item: item.relative_to(PROJECT_ROOT).as_posix())

    rows = [
        {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ready" if not missing else "incomplete",
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "missing_required_paths": missing,
        "excluded_runtime_directories": [
            ".cache",
            ".venv*",
            "features",
            "indexes",
            "outputs",
        ],
        "files": rows,
    }
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"status: {payload['status']}")
    print(f"files: {payload['file_count']}")
    print(f"bytes: {payload['total_bytes']}")
    print(f"manifest: {output.relative_to(PROJECT_ROOT).as_posix()}")
    if missing:
        raise SystemExit(1)


def _include_file(path: Path) -> bool:
    if not path.is_file():
        return False
    relative_parts = set(path.relative_to(PROJECT_ROOT).parts)
    return not (relative_parts & EXCLUDED_PARTS) and path.suffix not in EXCLUDED_SUFFIXES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
