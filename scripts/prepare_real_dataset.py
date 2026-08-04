from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlm_rag.dataset_adapters import convert_legacy_demo_dataset
from vlm_rag.dataset_schema import save_bundle, validate_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a source dataset into the canonical page-level VLM-RAG schema."
    )
    parser.add_argument("--source", choices=("legacy-demo",), default="legacy-demo")
    parser.add_argument("--pages", type=Path, default=PROJECT_ROOT / "data" / "pages.json")
    parser.add_argument("--queries", type=Path, default=PROJECT_ROOT / "data" / "queries.json")
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "data" / "splits")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "canonical")
    args = parser.parse_args()

    if args.source == "legacy-demo":
        bundle = convert_legacy_demo_dataset(args.pages, args.queries, args.split_dir)
    else:  # pragma: no cover - argparse restricts this branch
        raise ValueError(f"unsupported source: {args.source}")

    warnings = validate_bundle(bundle, project_root=PROJECT_ROOT)
    manifest = save_bundle(bundle, args.output_dir)
    print(f"pages: {manifest['page_count']}")
    print(f"samples: {manifest['sample_count']}")
    print(f"output: {args.output_dir}")
    for warning in warnings:
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()
