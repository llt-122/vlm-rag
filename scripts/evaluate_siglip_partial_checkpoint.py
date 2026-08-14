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

from vlm_rag.dataset_schema import DocumentPage, DocumentQASample, load_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a partially fine-tuned SigLIP checkpoint.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for tuned SigLIP evaluation")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = str(checkpoint["base_model"])
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    missing, unexpected = model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
    if unexpected:
        raise ValueError(f"unexpected checkpoint keys: {unexpected[:3]}")
    loaded_names = set(checkpoint["trainable_state_dict"])
    if not loaded_names or all(name in missing for name in loaded_names):
        raise ValueError("checkpoint did not match the base SigLIP model")
    device = torch.device("cuda:0")
    model.to(device).eval()
    bundle = load_bundle(args.dataset_dir)

    print(f"Encoding {len(bundle.pages)} pages with the tuned SigLIP model...")
    page_ids, page_vectors = _encode_pages(
        bundle.pages, model, processor, Image, torch, F, device, args.batch_size
    )
    print(f"Encoding {len(bundle.samples)} queries with the tuned SigLIP model...")
    query_vectors = _encode_queries(
        bundle.samples, model, processor, torch, F, device, args.batch_size
    )
    results, overall, by_split = _evaluate(
        bundle.samples, query_vectors, page_ids, page_vectors, torch
    )

    args.index_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"page_ids": page_ids, "vectors": page_vectors}, args.index_dir / "page_embeddings.pt")
    (args.index_dir / "index_metadata.json").write_text(
        json.dumps(
            {
                "format_version": "1.0",
                "method": "siglip_partial_finetune",
                "model_name": model_name,
                "checkpoint": str(args.checkpoint),
                "page_count": len(page_ids),
                "embedding_dim": int(page_vectors.shape[1]),
                "normalized": True,
                "page_ids": page_ids,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "retrieval_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "method": "siglip_partial_finetune",
                "model_name": model_name,
                "checkpoint": str(args.checkpoint),
                "page_count": len(page_ids),
                "query_count": len(bundle.samples),
                "overall": overall,
                "by_split": by_split,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"overall: {_format_metrics(overall)}")
    for split, metrics in by_split.items():
        print(f"{split}: {_format_metrics(metrics)}")
    print(f"index: {args.index_dir}")
    print(f"metrics: {metrics_path}")


def _encode_pages(pages, model, processor, image_module, torch_module, functional, device, batch_size):
    vectors = []
    page_ids = []
    with torch_module.inference_mode():
        for start in range(0, len(pages), batch_size):
            batch = pages[start : start + batch_size]
            images = []
            try:
                for page in batch:
                    with image_module.open(_image_path(page)) as image:
                        images.append(image.convert("RGB"))
                pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
                with torch_module.autocast(device_type="cuda", dtype=torch_module.bfloat16):
                    output = model.get_image_features(pixel_values=pixel_values)
                vectors.append(functional.normalize(_feature_tensor(output).float(), dim=-1).cpu())
                page_ids.extend(page.page_id for page in batch)
            finally:
                for image in images:
                    image.close()
            if len(page_ids) % (batch_size * 20) == 0 or len(page_ids) == len(pages):
                print(f"Encoded pages: {len(page_ids)}/{len(pages)}")
    return page_ids, torch_module.cat(vectors)


def _encode_queries(samples, model, processor, torch_module, functional, device, batch_size):
    vectors = []
    with torch_module.inference_mode():
        for start in range(0, len(samples), batch_size):
            texts = [sample.query for sample in samples[start : start + batch_size]]
            inputs = processor(
                text=texts, padding="max_length", truncation=True, return_tensors="pt"
            )
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            with torch_module.autocast(device_type="cuda", dtype=torch_module.bfloat16):
                output = model.get_text_features(**inputs)
            vectors.append(functional.normalize(_feature_tensor(output).float(), dim=-1).cpu())
    return torch_module.cat(vectors)


def _evaluate(samples, queries, page_ids, pages, torch_module):
    scores = queries @ pages.T
    results = []
    all_ranks = []
    ranks_by_split = defaultdict(list)
    for sample, row in zip(samples, scores, strict=True):
        order = torch_module.argsort(row, descending=True)[:10].tolist()
        evidence = set(sample.evidence_page_ids)
        first_rank = next(
            (rank for rank, index in enumerate(order, start=1) if page_ids[index] in evidence),
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
                    {
                        "page_id": page_ids[index],
                        "rank": rank,
                        "score": round(float(row[index]), 8),
                    }
                    for rank, index in enumerate(order, start=1)
                ],
            }
        )
    return results, _metrics(all_ranks), {
        split: _metrics(ranks) for split, ranks in sorted(ranks_by_split.items())
    }


def _metrics(ranks):
    count = len(ranks)
    return {
        "recall@1": sum(rank is not None and rank <= 1 for rank in ranks) / count,
        "recall@3": sum(rank is not None and rank <= 3 for rank in ranks) / count,
        "recall@10": sum(rank is not None and rank <= 10 for rank in ranks) / count,
        "mrr@10": sum(1.0 / rank for rank in ranks if rank is not None and rank <= 10) / count,
    }


def _image_path(page: DocumentPage) -> Path:
    path = Path(page.image_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _feature_tensor(output):
    pooled = getattr(output, "pooler_output", None)
    return pooled if pooled is not None else output


def _format_metrics(metrics):
    return " ".join(f"{name}={value:.4f}" for name, value in metrics.items())


if __name__ == "__main__":
    main()
