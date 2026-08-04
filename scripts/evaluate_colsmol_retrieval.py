from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlm_rag.dataset_schema import DocumentQASample, DocumentPage, load_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the lightweight ColSmol late-interaction visual retriever."
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=PROJECT_ROOT / "data" / "real" / "chartqa"
    )
    parser.add_argument(
        "--index-dir", type=Path, default=PROJECT_ROOT / "indexes" / "colsmol_chartqa"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "colsmol_chartqa"
    )
    parser.add_argument("--model", default="vidore/colSmol-500M")
    parser.add_argument("--image-batch-size", type=int, default=2)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from colpali_engine.models import ColIdefics3, ColIdefics3Processor

    if not torch.cuda.is_available():
        raise RuntimeError("this ColSmol baseline expects a CUDA GPU")
    bundle = load_bundle(args.dataset_dir)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    model = ColIdefics3.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="eager",
    ).eval()
    processor = ColIdefics3Processor.from_pretrained(args.model)

    index_path = args.index_dir / "page_embeddings.pt"
    metadata_path = args.index_dir / "index_metadata.json"
    if index_path.exists() and metadata_path.exists() and not args.rebuild_index:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        page_ids = metadata["page_ids"]
        expected_page_ids = [page.page_id for page in bundle.pages]
        if metadata["model_name"] != args.model or page_ids != expected_page_ids:
            raise ValueError("cached ColSmol index differs from current inputs; use --rebuild-index")
        document_embeddings = torch.load(index_path, map_location="cpu", weights_only=True)
        page_encoding_seconds = float(metadata["page_encoding_seconds"])
        print(f"Loaded {len(page_ids)} cached page embeddings.")
    else:
        page_ids, document_embeddings, page_encoding_seconds = _encode_pages(
            bundle.pages,
            model,
            processor,
            Image,
            torch,
            args.image_batch_size,
        )
        args.index_dir.mkdir(parents=True, exist_ok=True)
        torch.save(document_embeddings, index_path)
        metadata = {
            "format_version": "1.0",
            "method": "colsmol_late_interaction",
            "model_name": args.model,
            "page_count": len(page_ids),
            "embedding_dim": int(document_embeddings[0].shape[-1]),
            "page_encoding_seconds": page_encoding_seconds,
            "page_ids": page_ids,
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"Encoding and scoring {len(bundle.samples)} queries...")
    query_started = perf_counter()
    ranked_scores: list[list[float]] = []
    # The cached image vectors use bfloat16 while this model version can emit
    # float32 query vectors. MaxSim requires both operands to share a dtype.
    documents_on_device = [embedding.to(device=device, dtype=torch.float32) for embedding in document_embeddings]
    for start in range(0, len(bundle.samples), args.query_batch_size):
        samples = bundle.samples[start : start + args.query_batch_size]
        inputs = processor.process_queries([sample.query for sample in samples]).to(device)
        with torch.inference_mode():
            query_batch = model(**inputs)
            query_list = [embedding.float() for embedding in torch.unbind(query_batch)]
            scores = processor.score_multi_vector(query_list, documents_on_device)
        ranked_scores.extend(scores.float().cpu().tolist())
    query_seconds = perf_counter() - query_started

    results, overall, by_split = _evaluate(bundle.samples, ranked_scores, page_ids)
    peak_gpu_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "retrieval_results.json"
    metrics_path = args.output_dir / "metrics.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_payload = {
        "method": "colsmol_late_interaction",
        "model_name": args.model,
        "page_count": len(page_ids),
        "query_count": len(bundle.samples),
        "overall": overall,
        "by_split": by_split,
        "runtime": {
            "page_encoding_seconds": page_encoding_seconds,
            "query_encoding_and_scoring_seconds": query_seconds,
            "peak_gpu_mb": peak_gpu_mb,
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison_path = _write_comparison(overall)

    print(f"pages: {len(page_ids)}")
    print(f"queries: {len(bundle.samples)}")
    print(f"device: {device}")
    for name, value in overall.items():
        print(f"{name}: {value:.4f}")
    print(f"page encoding: {page_encoding_seconds:.1f} s")
    print(f"query encoding and scoring: {query_seconds:.1f} s")
    print(f"peak GPU memory: {peak_gpu_mb:.1f} MB")
    print(f"index: {_relative(args.index_dir)}")
    print(f"metrics: {_relative(metrics_path)}")
    print(f"comparison: {_relative(comparison_path)}")


def _encode_pages(
    pages: list[DocumentPage],
    model: object,
    processor: object,
    image_module: object,
    torch_module: object,
    batch_size: int,
) -> tuple[list[str], list[object], float]:
    page_ids: list[str] = []
    embeddings: list[object] = []
    started = perf_counter()
    for start in range(0, len(pages), batch_size):
        page_batch = pages[start : start + batch_size]
        images = []
        try:
            for page in page_batch:
                path = Path(page.image_path)
                path = path if path.is_absolute() else PROJECT_ROOT / path
                with image_module.open(path) as source:
                    images.append(source.convert("RGB"))
            inputs = processor.process_images(images).to(model.device)
            with torch_module.inference_mode():
                batch_embeddings = model(**inputs)
            embeddings.extend(
                embedding.contiguous().cpu() for embedding in torch_module.unbind(batch_embeddings)
            )
            page_ids.extend(page.page_id for page in page_batch)
            print(f"Encoded pages: {len(page_ids)}/{len(pages)}")
        finally:
            for image in images:
                image.close()
    return page_ids, embeddings, perf_counter() - started


def _evaluate(
    samples: list[DocumentQASample],
    score_rows: list[list[float]],
    page_ids: list[str],
) -> tuple[list[dict[str, object]], dict[str, float], dict[str, dict[str, float]]]:
    results: list[dict[str, object]] = []
    all_ranks: list[int | None] = []
    ranks_by_split: dict[str, list[int | None]] = defaultdict(list)
    for sample, scores in zip(samples, score_rows, strict=True):
        order = sorted(range(len(page_ids)), key=lambda index: scores[index], reverse=True)[:10]
        top_10 = [(page_ids[index], scores[index]) for index in order]
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


def _write_comparison(colsmol_metrics: dict[str, float]) -> Path:
    sources = [
        ("siglip_zero_shot", PROJECT_ROOT / "outputs" / "siglip_chartqa" / "metrics.json"),
        ("ppocr_bge", PROJECT_ROOT / "outputs" / "ocr_bge_chartqa" / "metrics.json"),
    ]
    rows: list[dict[str, object]] = []
    for method, path in sources:
        if path.exists():
            rows.append({"method": method, **json.loads(path.read_text(encoding="utf-8"))["overall"]})
    rows.append({"method": "colsmol_late_interaction", **colsmol_metrics})
    path = PROJECT_ROOT / "outputs" / "baseline_retrieval_comparison.csv"
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
