from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlm_rag.dataset_adapters import download_chartqa_sample
from vlm_rag.dataset_schema import save_bundle, validate_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a small ChartQA sample and convert it to the canonical VLM-RAG schema."
    )
    parser.add_argument("--train-rows", type=int, default=60)
    parser.add_argument("--dev-rows", type=int, default=20)
    parser.add_argument("--test-rows", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "real" / "chartqa",
    )
    args = parser.parse_args()
    if min(args.train_rows, args.dev_rows, args.test_rows) < 0:
        parser.error("row limits must be non-negative")

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    bundle = download_chartqa_sample(
        output_dir=args.output_dir,
        project_root=PROJECT_ROOT,
        split_row_limits={
            "train": args.train_rows,
            "dev": args.dev_rows,
            "test": args.test_rows,
        },
    )
    warnings = validate_bundle(bundle, project_root=PROJECT_ROOT)
    manifest = save_bundle(bundle, args.output_dir)

    print(f"pages: {manifest['page_count']}")
    print(f"samples: {manifest['sample_count']}")
    print(f"splits: {manifest['splits']}")
    print(f"output: {args.output_dir}")
    for warning in warnings:
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()
