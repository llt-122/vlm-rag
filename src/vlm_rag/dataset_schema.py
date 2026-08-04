from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "1.0"
VALID_SPLITS = {"train", "dev", "test", "unspecified"}


@dataclass(frozen=True)
class DocumentPage:
    """A page-level retrieval unit shared by all real datasets."""

    page_id: str
    doc_id: str
    doc_type: str
    page_no: int
    image_path: str
    title: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentQASample:
    """A question with one or more valid answers and evidence pages."""

    query_id: str
    query: str
    answers: list[str]
    evidence_page_ids: list[str]
    doc_type: str
    split: str = "unspecified"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetBundle:
    pages: list[DocumentPage]
    samples: list[DocumentQASample]


def save_bundle(bundle: DatasetBundle, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "pages.jsonl", (asdict(page) for page in bundle.pages))
    _write_jsonl(output_dir / "samples.jsonl", (asdict(sample) for sample in bundle.samples))

    warnings = validate_bundle(bundle)
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "page_count": len(bundle.pages),
        "sample_count": len(bundle.samples),
        "splits": _count_values(sample.split for sample in bundle.samples),
        "document_types": _count_values(page.doc_type for page in bundle.pages),
        "warnings": warnings,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def load_bundle(dataset_dir: Path) -> DatasetBundle:
    pages = [DocumentPage(**item) for item in _read_jsonl(dataset_dir / "pages.jsonl")]
    samples = [DocumentQASample(**item) for item in _read_jsonl(dataset_dir / "samples.jsonl")]
    bundle = DatasetBundle(pages=pages, samples=samples)
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle: DatasetBundle, project_root: Path | None = None) -> list[str]:
    errors: list[str] = []
    warnings: list[str] = []

    page_ids = [page.page_id for page in bundle.pages]
    duplicate_pages = _duplicates(page_ids)
    if duplicate_pages:
        errors.append(f"duplicate page_id values: {sorted(duplicate_pages)}")

    query_ids = [sample.query_id for sample in bundle.samples]
    duplicate_queries = _duplicates(query_ids)
    if duplicate_queries:
        errors.append(f"duplicate query_id values: {sorted(duplicate_queries)}")

    page_id_set = set(page_ids)
    for sample in bundle.samples:
        if not sample.query.strip():
            errors.append(f"{sample.query_id}: query is empty")
        if not sample.answers or not any(answer.strip() for answer in sample.answers):
            errors.append(f"{sample.query_id}: answers are empty")
        if not sample.evidence_page_ids:
            errors.append(f"{sample.query_id}: evidence_page_ids are empty")
        missing_pages = sorted(set(sample.evidence_page_ids) - page_id_set)
        if missing_pages:
            errors.append(f"{sample.query_id}: missing evidence pages {missing_pages}")
        if sample.split not in VALID_SPLITS:
            errors.append(f"{sample.query_id}: unsupported split {sample.split!r}")

    if project_root is not None:
        for page in bundle.pages:
            image_path = Path(page.image_path)
            resolved_path = image_path if image_path.is_absolute() else project_root / image_path
            if not resolved_path.exists():
                warnings.append(f"{page.page_id}: image not found at {page.image_path}")

    warnings.extend(_document_split_warnings(bundle))
    if errors:
        raise ValueError("dataset validation failed:\n- " + "\n- ".join(errors))
    return warnings


def _document_split_warnings(bundle: DatasetBundle) -> list[str]:
    page_to_doc = {page.page_id: page.doc_id for page in bundle.pages}
    doc_splits: dict[str, set[str]] = {}
    for sample in bundle.samples:
        if sample.split == "unspecified":
            continue
        for page_id in sample.evidence_page_ids:
            doc_id = page_to_doc.get(page_id)
            if doc_id is not None:
                doc_splits.setdefault(doc_id, set()).add(sample.split)
    return [
        f"document leakage: {doc_id} appears in splits {sorted(splits)}"
        for doc_id, splits in sorted(doc_splits.items())
        if len(splits) > 1
    ]


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def _duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _count_values(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
