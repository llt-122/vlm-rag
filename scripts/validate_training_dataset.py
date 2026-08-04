from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlm_rag.dataset_schema import load_bundle, validate_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly validate a training dataset bundle.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--allow-missing-images", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    report_path = args.report_path or dataset_dir / "training_validation.json"
    bundle = load_bundle(dataset_dir)
    schema_warnings = validate_bundle(bundle, project_root=PROJECT_ROOT)

    page_by_id = {page.page_id: page for page in bundle.pages}
    docs_by_split: dict[str, set[str]] = defaultdict(set)
    pages_by_split: dict[str, set[str]] = defaultdict(set)
    missing_images: list[str] = []
    empty_files: list[str] = []
    for page in bundle.pages:
        path = Path(page.image_path)
        resolved = path if path.is_absolute() else PROJECT_ROOT / path
        if not resolved.exists():
            missing_images.append(page.page_id)
        elif resolved.stat().st_size == 0:
            empty_files.append(page.page_id)

    query_text_splits: dict[str, set[str]] = defaultdict(set)
    for sample in bundle.samples:
        normalized_query = " ".join(sample.query.casefold().split())
        query_text_splits[normalized_query].add(sample.split)
        for page_id in sample.evidence_page_ids:
            page = page_by_id[page_id]
            docs_by_split[sample.split].add(page.doc_id)
            pages_by_split[sample.split].add(page_id)

    leakage: list[dict[str, object]] = []
    all_docs = set().union(*docs_by_split.values()) if docs_by_split else set()
    for doc_id in sorted(all_docs):
        splits = sorted(split for split, docs in docs_by_split.items() if doc_id in docs)
        if len(splits) > 1:
            leakage.append({"doc_id": doc_id, "splits": splits})
    duplicate_queries = [
        {"query": query, "splits": sorted(splits)}
        for query, splits in query_text_splits.items()
        if len(splits) > 1
    ]

    split_counts = Counter(sample.split for sample in bundle.samples)
    required_splits = {"train", "dev", "test"}
    missing_splits = sorted(required_splits - set(split_counts))
    fatal_errors: list[str] = []
    if missing_splits:
        fatal_errors.append(f"missing required splits: {missing_splits}")
    if leakage:
        fatal_errors.append(f"document leakage across splits: {len(leakage)} document(s)")
    if missing_images and not args.allow_missing_images:
        fatal_errors.append(f"missing page images: {len(missing_images)}")
    if empty_files:
        fatal_errors.append(f"empty page image files: {len(empty_files)}")

    report = {
        "status": "failed" if fatal_errors else "ready",
        "dataset_dir": _relative(dataset_dir),
        "page_count": len(bundle.pages),
        "query_count": len(bundle.samples),
        "split_query_counts": dict(sorted(split_counts.items())),
        "split_page_counts": {key: len(value) for key, value in sorted(pages_by_split.items())},
        "split_document_counts": {key: len(value) for key, value in sorted(docs_by_split.items())},
        "document_types": dict(sorted(Counter(page.doc_type for page in bundle.pages).items())),
        "schema_warnings": schema_warnings,
        "missing_images": missing_images,
        "empty_images": empty_files,
        "document_leakage": leakage,
        "duplicate_queries_across_splits": duplicate_queries,
        "fatal_errors": fatal_errors,
        "source_checksums": {
            name: _sha256(dataset_dir / name) for name in ("pages.jsonl", "samples.jsonl")
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if fatal_errors:
        raise SystemExit(1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
