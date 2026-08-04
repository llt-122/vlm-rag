from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlm_rag.dataset_schema import DocumentQASample, load_bundle
from vlm_rag.siglip_encoder import SigLIPConfig, SigLIPEncoder
from vlm_rag.siglip_index import (
    SigLIPVectorIndex,
    build_siglip_index,
    load_siglip_index,
    save_siglip_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a real SigLIP page index and evaluate ChartQA Top-K retrieval."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "real" / "chartqa",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=PROJECT_ROOT / "indexes" / "siglip_chartqa",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "siglip_chartqa",
    )
    parser.add_argument("--model", default="google/siglip-base-patch16-224")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    bundle = load_bundle(args.dataset_dir)
    encoder = SigLIPEncoder(
        SigLIPConfig(
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size,
        )
    )
    index = _get_or_build_index(bundle.pages, encoder, args.index_dir, args.rebuild_index)
    if index.metadata["model_name"] != args.model:
        raise ValueError("stored index model differs from --model; use --rebuild-index")
    dataset_page_ids = {page.page_id for page in bundle.pages}
    if set(index.page_ids) != dataset_page_ids:
        raise ValueError("stored index pages differ from the dataset; use --rebuild-index")

    print(f"Encoding {len(bundle.samples)} queries...")
    query_vectors = encoder.encode_texts(sample.query for sample in bundle.samples)
    results, overall_metrics, split_metrics = _evaluate(bundle.samples, query_vectors, index)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "retrieval_results.json"
    metrics_path = args.output_dir / "metrics.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_payload = {
        "model_name": args.model,
        "page_count": len(bundle.pages),
        "query_count": len(bundle.samples),
        "overall": overall_metrics,
        "by_split": split_metrics,
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"pages: {len(bundle.pages)}")
    print(f"queries: {len(bundle.samples)}")
    print(f"device: {encoder.device}")
    for name, value in overall_metrics.items():
        print(f"{name}: {value:.4f}")
    print(f"index: {_relative(args.index_dir)}")
    print(f"results: {_relative(results_path)}")
    print(f"metrics: {_relative(metrics_path)}")


def _get_or_build_index(
    pages: list,
    encoder: SigLIPEncoder,
    index_dir: Path,
    rebuild: bool,
) -> SigLIPVectorIndex:
    if not rebuild:
        try:
            index = load_siglip_index(index_dir)
            print(f"Loaded {len(index.page_ids)} cached page vectors.")
            return index
        except FileNotFoundError:
            pass
    print(f"Encoding {len(pages)} page images...")
    index = build_siglip_index(pages, encoder, PROJECT_ROOT)
    save_siglip_index(index, index_dir)
    return index


def _evaluate(
    samples: list[DocumentQASample],
    query_vectors: list[list[float]],
    index: SigLIPVectorIndex,
) -> tuple[list[dict[str, object]], dict[str, float], dict[str, dict[str, float]]]:
    results: list[dict[str, object]] = []
    ranks_by_split: dict[str, list[int | None]] = defaultdict(list)
    all_ranks: list[int | None] = []

    for sample, query_vector in zip(samples, query_vectors, strict=True):
        hits = index.search_vector(query_vector, top_k=10)
        evidence = set(sample.evidence_page_ids)
        first_rank = next((hit.rank for hit in hits if hit.page_id in evidence), None)
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
                    {"page_id": hit.page_id, "rank": hit.rank, "score": round(hit.score, 8)}
                    for hit in hits
                ],
            }
        )

    overall = _metrics_from_ranks(all_ranks)
    by_split = {
        split: _metrics_from_ranks(ranks)
        for split, ranks in sorted(ranks_by_split.items())
    }
    return results, overall, by_split


def _metrics_from_ranks(ranks: list[int | None]) -> dict[str, float]:
    if not ranks:
        return {"recall@1": 0.0, "recall@3": 0.0, "recall@10": 0.0, "mrr@10": 0.0}
    count = len(ranks)
    return {
        "recall@1": sum(rank is not None and rank <= 1 for rank in ranks) / count,
        "recall@3": sum(rank is not None and rank <= 3 for rank in ranks) / count,
        "recall@10": sum(rank is not None and rank <= 10 for rank in ranks) / count,
        "mrr@10": sum(1.0 / rank for rank in ranks if rank is not None and rank <= 10) / count,
    }


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
