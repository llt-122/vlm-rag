from __future__ import annotations

import json
import hashlib
import io
from pathlib import Path
from typing import Iterable

from .dataset_schema import DatasetBundle, DocumentPage, DocumentQASample


def convert_legacy_demo_dataset(
    pages_path: Path,
    queries_path: Path,
    split_dir: Path | None = None,
) -> DatasetBundle:
    """Convert this repository's original JSON files into the canonical schema."""

    raw_pages = _load_json_array(pages_path)
    raw_queries = _load_json_array(queries_path)
    split_by_query = _load_split_mapping(split_dir)

    pages = [
        DocumentPage(
            page_id=str(item["page_id"]),
            doc_id=str(item["doc_id"]),
            doc_type=str(item["doc_type"]),
            page_no=int(item["page_no"]),
            image_path=str(item["image_path"]),
            title=str(item.get("title", "")),
            metadata={
                "visual_text": str(item.get("visual_text", "")),
                "layout": str(item.get("layout", "")),
                "facts": item.get("facts", {}),
                "source_dataset": "legacy_demo",
            },
        )
        for item in raw_pages
    ]

    samples = [
        DocumentQASample(
            query_id=str(item["query_id"]),
            query=str(item["text"]),
            answers=[str(item["answer"])],
            evidence_page_ids=[str(value) for value in item["positive_page_ids"]],
            doc_type=str(item["doc_type"]),
            split=split_by_query.get(str(item["query_id"]), "unspecified"),
            metadata={"source_dataset": "legacy_demo"},
        )
        for item in raw_queries
    ]
    return DatasetBundle(pages=pages, samples=samples)


def download_chartqa_sample(
    output_dir: Path,
    project_root: Path,
    split_row_limits: dict[str, int],
) -> DatasetBundle:
    """Stream a small ChartQA subset and convert it into the canonical schema.

    Images are deduplicated by their PNG bytes. The original ``train``, ``val``
    and ``test`` splits are preserved, with ``val`` renamed to ``dev``. If the
    same image occurs in multiple source splits, only the first split keeps it
    so page-level evidence cannot leak across train/dev/test.
    """

    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "ChartQA import requires the optional 'datasets' and 'Pillow' packages"
        ) from error

    image_dir = output_dir / "pages"
    image_dir.mkdir(parents=True, exist_ok=True)
    pages_by_id: dict[str, DocumentPage] = {}
    page_split: dict[str, str] = {}
    samples: list[DocumentQASample] = []
    query_counter = 0

    source_splits = (("train", "train"), ("val", "dev"), ("test", "test"))
    for source_split, target_split in source_splits:
        row_limit = split_row_limits.get(target_split, 0)
        if row_limit <= 0:
            continue

        dataset = load_dataset(
            "HuggingFaceM4/ChartQA",
            split=source_split,
            streaming=True,
        )
        accepted_rows = 0
        for source_index, row in enumerate(dataset):
            if accepted_rows >= row_limit:
                break
            png_bytes, width, height = _normalise_image(row["image"])
            image_hash = hashlib.sha256(png_bytes).hexdigest()
            page_id = f"chartqa_{image_hash[:16]}"
            existing_split = page_split.get(page_id)
            if existing_split is not None and existing_split != target_split:
                continue
            image_path = image_dir / f"{page_id}.png"
            if not image_path.exists():
                image_path.write_bytes(png_bytes)

            display_path = _relative_path(image_path, project_root)
            existing_page = pages_by_id.get(page_id)
            if existing_page is None:
                page_split[page_id] = target_split
                pages_by_id[page_id] = DocumentPage(
                    page_id=page_id,
                    doc_id=page_id,
                    doc_type="chart",
                    page_no=1,
                    image_path=display_path,
                    title="ChartQA chart",
                    metadata={
                        "source_dataset": "HuggingFaceM4/ChartQA",
                        "width": width,
                        "height": height,
                        "sha256": image_hash,
                    },
                )

            query_counter += 1
            answers = _normalise_answers(row.get("label"))
            samples.append(
                DocumentQASample(
                    query_id=f"chartqa_q_{query_counter:06d}",
                    query=str(row["query"]).strip(),
                    answers=answers,
                    evidence_page_ids=[page_id],
                    doc_type="chart",
                    split=target_split,
                    metadata={
                        "source_dataset": "HuggingFaceM4/ChartQA",
                        "source_split": source_split,
                        "source_row": source_index,
                        "human_or_machine": int(row.get("human_or_machine", -1)),
                    },
                )
            )
            accepted_rows += 1

    return DatasetBundle(pages=list(pages_by_id.values()), samples=samples)


def _load_split_mapping(split_dir: Path | None) -> dict[str, str]:
    if split_dir is None:
        return {}

    mapping: dict[str, str] = {}
    for split in ("train", "dev", "test"):
        split_path = split_dir / f"{split}_queries.json"
        if not split_path.exists():
            continue
        for item in _load_json_array(split_path):
            query_id = str(item["query_id"])
            previous = mapping.get(query_id)
            if previous is not None and previous != split:
                raise ValueError(f"query {query_id} appears in both {previous} and {split}")
            mapping[query_id] = split
    return mapping


def _load_json_array(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{path}: expected a JSON array of objects")
    return value


def _normalise_image(image: object) -> tuple[bytes, int, int]:
    if not hasattr(image, "save") or not hasattr(image, "size"):
        raise ValueError(f"ChartQA row contains unsupported image value: {type(image)!r}")
    rgb_image = image.convert("RGB")  # type: ignore[attr-defined]
    buffer = io.BytesIO()
    rgb_image.save(buffer, format="PNG", optimize=False)
    width, height = rgb_image.size
    return buffer.getvalue(), int(width), int(height)


def _normalise_answers(value: object) -> list[str]:
    if isinstance(value, list):
        answers = [str(item).strip() for item in value if str(item).strip()]
    elif value is None:
        answers = []
    else:
        answer = str(value).strip()
        answers = [answer] if answer else []
    if not answers:
        raise ValueError("ChartQA row does not contain a non-empty answer")
    return answers


def _relative_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _take(rows: Iterable[dict[str, object]], limit: int) -> Iterable[dict[str, object]]:
    for index, row in enumerate(rows):
        if index >= limit:
            break
        yield row
