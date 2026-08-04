from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlm_rag.dataset_schema import DocumentQASample, load_bundle
from vlm_rag.encoders import cosine_similarity
from vlm_rag.text_encoder import SentenceTextEncoder, TextEncoderConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the real PP-OCR + BGE retrieval baseline.")
    parser.add_argument(
        "--dataset-dir", type=Path, default=PROJECT_ROOT / "data" / "real" / "chartqa"
    )
    parser.add_argument(
        "--ocr-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "real" / "chartqa" / "ocr_text.jsonl",
    )
    parser.add_argument(
        "--index-dir", type=Path, default=PROJECT_ROOT / "indexes" / "ocr_bge_chartqa"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "ocr_bge_chartqa"
    )
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    bundle = load_bundle(args.dataset_dir)
    ocr_rows = _load_ocr_rows(args.ocr_file)
    page_ids = [page.page_id for page in bundle.pages]
    missing = sorted(set(page_ids) - set(ocr_rows))
    if missing:
        raise ValueError(f"OCR text is missing for {len(missing)} pages: {missing[:3]}")

    encoder = SentenceTextEncoder(
        TextEncoderConfig(
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size,
        )
    )
    print(f"Encoding OCR text for {len(page_ids)} pages...")
    page_vectors = encoder.encode_documents(ocr_rows[page_id]["text"] for page_id in page_ids)
    vectors_by_page = dict(zip(page_ids, page_vectors, strict=True))
    _save_index(args.index_dir, args.model, page_ids, vectors_by_page)

    print(f"Encoding {len(bundle.samples)} queries...")
    query_vectors = encoder.encode_queries(sample.query for sample in bundle.samples)
    results, overall, by_split = _evaluate(bundle.samples, query_vectors, page_ids, vectors_by_page)
    elapsed_values = [float(ocr_rows[page_id]["elapsed_ms"]) for page_id in page_ids]
    ocr_stats = {
        "empty_page_count": sum(not str(ocr_rows[page_id]["text"]).strip() for page_id in page_ids),
        "mean_page_ms": statistics.fmean(elapsed_values),
        "median_page_ms": statistics.median(elapsed_values),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "retrieval_results.json"
    metrics_path = args.output_dir / "metrics.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_payload = {
        "method": "ppocr_bge",
        "ocr_model": "PP-OCRv6-medium",
        "embedding_model": args.model,
        "page_count": len(page_ids),
        "query_count": len(bundle.samples),
        "overall": overall,
        "by_split": by_split,
        "ocr_timing": ocr_stats,
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison_path = _write_comparison(overall)

    print(f"pages: {len(page_ids)}")
    print(f"queries: {len(bundle.samples)}")
    print(f"embedding device: {encoder.device}")
    for name, value in overall.items():
        print(f"{name}: {value:.4f}")
    print(f"mean OCR page time: {ocr_stats['mean_page_ms']:.1f} ms")
    print(f"index: {_relative(args.index_dir)}")
    print(f"metrics: {_relative(metrics_path)}")
    print(f"comparison: {_relative(comparison_path)}")


def _load_ocr_rows(path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        page_id = row.get("page_id")
        if not isinstance(page_id, str):
            raise ValueError(f"{path}:{line_number}: invalid page_id")
        existing = rows.get(page_id)
        if existing is not None:
            if existing.get("text") != row.get("text"):
                raise ValueError(f"{path}:{line_number}: conflicting OCR text for {page_id}")
            if float(row.get("elapsed_ms", float("inf"))) >= float(
                existing.get("elapsed_ms", float("inf"))
            ):
                continue
        rows[page_id] = row
    return rows


def _save_index(
    index_dir: Path,
    model_name: str,
    page_ids: list[str],
    vectors: dict[str, list[float]],
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "page_vectors.json").write_text(
        json.dumps(vectors, ensure_ascii=False), encoding="utf-8"
    )
    metadata = {
        "format_version": "1.0",
        "method": "ppocr_bge",
        "model_name": model_name,
        "page_count": len(page_ids),
        "embedding_dim": len(next(iter(vectors.values()))),
        "normalized": True,
        "similarity": "cosine",
        "page_ids": page_ids,
    }
    (index_dir / "index_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _evaluate(
    samples: list[DocumentQASample],
    query_vectors: list[list[float]],
    page_ids: list[str],
    page_vectors: dict[str, list[float]],
) -> tuple[list[dict[str, object]], dict[str, float], dict[str, dict[str, float]]]:
    results: list[dict[str, object]] = []
    all_ranks: list[int | None] = []
    ranks_by_split: dict[str, list[int | None]] = defaultdict(list)
    for sample, query_vector in zip(samples, query_vectors, strict=True):
        scored = [
            (page_id, cosine_similarity(query_vector, page_vectors[page_id]))
            for page_id in page_ids
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        top_10 = scored[:10]
        evidence = set(sample.evidence_page_ids)
        first_rank = next(
            (rank for rank, (page_id, _) in enumerate(top_10, start=1) if page_id in evidence),
            None,
        )
        all_ranks.append(first_rank)
        ranks_by_split[sample.split].append(first_rank)
        results.append(
            {
                "query_id": sample.query_id,
                "query": sample.query,
                "split": sample.split,
                "evidence_page_ids": sample.evidence_page_ids,
                "first_evidence_rank": first_rank,
                "top_10": [
                    {"page_id": page_id, "rank": rank, "score": round(score, 8)}
                    for rank, (page_id, score) in enumerate(top_10, start=1)
                ],
            }
        )
    return (
        results,
        _metrics(all_ranks),
        {split: _metrics(ranks) for split, ranks in sorted(ranks_by_split.items())},
    )


def _metrics(ranks: list[int | None]) -> dict[str, float]:
    count = len(ranks)
    return {
        "recall@1": sum(rank is not None and rank <= 1 for rank in ranks) / count,
        "recall@3": sum(rank is not None and rank <= 3 for rank in ranks) / count,
        "recall@10": sum(rank is not None and rank <= 10 for rank in ranks) / count,
        "mrr@10": sum(1.0 / rank for rank in ranks if rank is not None and rank <= 10) / count,
    }


def _write_comparison(ocr_metrics: dict[str, float]) -> Path:
    rows = [{"method": "ppocr_bge", **ocr_metrics}]
    siglip_path = PROJECT_ROOT / "outputs" / "siglip_chartqa" / "metrics.json"
    if siglip_path.exists():
        siglip = json.loads(siglip_path.read_text(encoding="utf-8"))["overall"]
        rows.insert(0, {"method": "siglip_zero_shot", **siglip})
    path = PROJECT_ROOT / "outputs" / "baseline_retrieval_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["method", "recall@1", "recall@3", "recall@10", "mrr@10"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
