from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlm_rag.dataset_schema import load_bundle
from vlm_rag.ocr_extractor import PaddleOCRExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PP-OCR text from ChartQA pages.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "real" / "chartqa",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "real" / "chartqa" / "ocr_text.jsonl",
    )
    parser.add_argument("--minimum-score", type=float, default=0.5)
    args = parser.parse_args()

    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
    bundle = load_bundle(args.dataset_dir)
    completed = _load_completed_page_ids(args.output)
    remaining = [page for page in bundle.pages if page.page_id not in completed]
    if not remaining:
        print(f"All {len(bundle.pages)} pages already have OCR text at {_relative(args.output)}")
        return

    print(f"OCR pages: completed={len(completed)}, remaining={len(remaining)}")
    extractor = PaddleOCRExtractor(minimum_score=args.minimum_score)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as output_file:
        for position, page in enumerate(remaining, start=1):
            image_path = _resolve_image_path(page.image_path)
            result = extractor.extract(page.page_id, image_path)
            output_file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            output_file.flush()
            print(
                f"[{position}/{len(remaining)}] {page.page_id}: "
                f"lines={len(result.lines)}, elapsed_ms={result.elapsed_ms:.1f}"
            )
    print(f"output: {_relative(args.output)}")


def _load_completed_page_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        page_id = payload.get("page_id")
        if not isinstance(page_id, str):
            raise ValueError(f"{path}:{line_number}: invalid page_id")
        completed.add(page_id)
    return completed


def _resolve_image_path(image_path: str) -> Path:
    path = Path(image_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
