from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlm_rag.dataset_schema import DocumentQASample, load_bundle
from vlm_rag.siglip_encoder import SigLIPConfig, SigLIPEncoder


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    parser = argparse.ArgumentParser(description="Train lightweight SigLIP adapters with InfoNCE.")
    parser.add_argument(
        "--dataset-dir", type=Path, default=PROJECT_ROOT / "data" / "real" / "chartqa"
    )
    parser.add_argument(
        "--base-index-dir", type=Path, default=PROJECT_ROOT / "indexes" / "siglip_chartqa"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "siglip_infonce"
    )
    parser.add_argument(
        "--model-dir", type=Path, default=PROJECT_ROOT / "models" / "siglip_infonce"
    )
    parser.add_argument("--model", default="google/siglip-base-patch16-224")
    parser.add_argument(
        "--adapter-type",
        choices=("diagonal", "dual_low_rank"),
        default="diagonal",
    )
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--patience", type=int, default=40)
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_bundle(args.dataset_dir)
    page_ids, page_vectors = _load_page_vectors(args.base_index_dir, args.model)
    query_vectors = _load_or_encode_queries(bundle.samples, args.base_index_dir, args.model)
    page_tensor = torch.tensor(page_vectors, dtype=torch.float32, device=device)
    query_tensor = torch.tensor(query_vectors, dtype=torch.float32, device=device)

    class ResidualAdapter(nn.Module):
        def __init__(self, dim: int, rank: int) -> None:
            super().__init__()
            self.down = nn.Linear(dim, rank, bias=False)
            self.up = nn.Linear(rank, dim, bias=False)
            nn.init.normal_(self.down.weight, std=0.02)
            nn.init.zeros_(self.up.weight)

        def forward(self, values):
            residual = self.up(F.gelu(self.down(values)))
            return F.normalize(values + residual, dim=-1)

    class DualAdapter(nn.Module):
        def __init__(self, dim: int, rank: int) -> None:
            super().__init__()
            self.query_adapter = ResidualAdapter(dim, rank)
            self.page_adapter = ResidualAdapter(dim, rank)

        def forward(self, queries, pages):
            return self.query_adapter(queries), self.page_adapter(pages)

        def regularization(self):
            return torch.zeros((), device=device)

    class DiagonalAdapter(nn.Module):
        """Learn a shared relevance weight per SigLIP dimension."""

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.log_weights = nn.Parameter(torch.zeros(dim))

        def forward(self, queries, pages):
            weights = self.log_weights.clamp(-2.0, 2.0).exp()
            return F.normalize(queries * weights, dim=-1), F.normalize(pages * weights, dim=-1)

        def regularization(self):
            return self.log_weights.square().mean()

    model = (
        DiagonalAdapter(page_tensor.shape[1])
        if args.adapter_type == "diagonal"
        else DualAdapter(page_tensor.shape[1], args.rank)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    page_index = {page_id: index for index, page_id in enumerate(page_ids)}
    train_indices = [i for i, sample in enumerate(bundle.samples) if sample.split == "train"]
    train_page_ids = sorted({bundle.samples[i].evidence_page_ids[0] for i in train_indices})
    train_page_positions = [page_index[page_id] for page_id in train_page_ids]
    train_target_lookup = {page_id: index for index, page_id in enumerate(train_page_ids)}
    train_targets = torch.tensor(
        [train_target_lookup[bundle.samples[i].evidence_page_ids[0]] for i in train_indices],
        dtype=torch.long,
        device=device,
    )
    train_query_tensor = query_tensor[train_indices]
    train_page_tensor = page_tensor[train_page_positions]

    baseline_ranks = _rank_all(model, query_tensor, page_tensor, bundle.samples, page_ids, torch)
    baseline_metrics = _metrics_by_split(bundle.samples, baseline_ranks)
    best_dev_mrr = baseline_metrics["dev"]["mrr@10"]
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    log_rows: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        projected_queries, projected_pages = model(train_query_tensor, train_page_tensor)
        logits = projected_queries @ projected_pages.T / args.temperature
        loss = F.cross_entropy(logits, train_targets) + 0.01 * model.regularization()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            ranks = _rank_all(model, query_tensor, page_tensor, bundle.samples, page_ids, torch)
        metrics = _metrics_by_split(bundle.samples, ranks)
        dev_mrr = metrics["dev"]["mrr@10"]
        log_rows.append(
            {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "train_mrr@10": metrics["train"]["mrr@10"],
                "dev_mrr@10": dev_mrr,
            }
        )
        if dev_mrr > best_dev_mrr + 1e-12:
            best_dev_mrr = dev_mrr
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 20 == 0:
            print(
                f"epoch={epoch}, loss={float(loss.detach()):.4f}, "
                f"train_mrr@10={metrics['train']['mrr@10']:.4f}, "
                f"dev_mrr@10={dev_mrr:.4f}"
            )
        if stale_epochs >= args.patience:
            print(f"early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        final_ranks, score_rows = _rank_all(
            model, query_tensor, page_tensor, bundle.samples, page_ids, torch, return_scores=True
        )
    final_metrics = _metrics_by_split(bundle.samples, final_ranks)
    overall_metrics = _metrics(list(final_ranks.values()))

    args.model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.model_dir / "adapter.pt")
    (args.model_dir / "config.json").write_text(
        json.dumps(
            {
                "base_model": args.model,
                "adapter_type": args.adapter_type,
                "embedding_dim": int(page_tensor.shape[1]),
                "rank": args.rank,
                "temperature": args.temperature,
                "learning_rate": args.learning_rate,
                "best_epoch": best_epoch,
                "best_dev_mrr@10": best_dev_mrr,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (args.model_dir / "training_log.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "loss", "train_mrr@10", "dev_mrr@10"])
        writer.writeheader()
        writer.writerows(log_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = _result_rows(bundle.samples, final_ranks, score_rows, page_ids)
    (args.output_dir / "retrieval_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics_payload = {
        "method": "siglip_infonce_adapter",
        "base_model": args.model,
        "best_epoch": best_epoch,
        "overall": overall_metrics,
        "by_split": final_metrics,
        "baseline_by_split": baseline_metrics,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"best_epoch: {best_epoch}")
    for split in ("train", "dev", "test"):
        before = baseline_metrics[split]["mrr@10"]
        after = final_metrics[split]["mrr@10"]
        print(f"{split} mrr@10: {before:.4f} -> {after:.4f}")
    for name, value in overall_metrics.items():
        print(f"overall {name}: {value:.4f}")


def _load_page_vectors(index_dir: Path, model_name: str) -> tuple[list[str], list[list[float]]]:
    metadata = json.loads((index_dir / "index_metadata.json").read_text(encoding="utf-8"))
    if metadata["model_name"] != model_name:
        raise ValueError("base index model does not match --model")
    vectors = json.loads((index_dir / "page_vectors.json").read_text(encoding="utf-8"))
    page_ids = metadata["page_ids"]
    return page_ids, [vectors[page_id] for page_id in page_ids]


def _load_or_encode_queries(samples, index_dir: Path, model_name: str) -> list[list[float]]:
    path = index_dir / "query_vectors.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model_name") == model_name and payload.get("query_ids") == [
            sample.query_id for sample in samples
        ]:
            print(f"Loaded {len(samples)} cached query vectors.")
            return payload["vectors"]
    print(f"Encoding {len(samples)} SigLIP query features...")
    encoder = SigLIPEncoder(SigLIPConfig(model_name=model_name, batch_size=16))
    vectors = encoder.encode_texts(sample.query for sample in samples)
    path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "query_ids": [sample.query_id for sample in samples],
                "vectors": vectors,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return vectors


def _rank_all(model, queries, pages, samples, page_ids, torch_module, return_scores=False):
    model.eval()
    projected_queries, projected_pages = model(queries, pages)
    scores = projected_queries @ projected_pages.T
    ranks: dict[str, int | None] = {}
    score_rows: list[list[float]] = scores.detach().cpu().tolist()
    for sample, row in zip(samples, score_rows, strict=True):
        order = sorted(range(len(page_ids)), key=lambda index: row[index], reverse=True)
        evidence = set(sample.evidence_page_ids)
        ranks[sample.query_id] = next(
            (rank for rank, index in enumerate(order[:10], start=1) if page_ids[index] in evidence),
            None,
        )
    return (ranks, score_rows) if return_scores else ranks


def _metrics_by_split(samples, ranks):
    grouped: dict[str, list[int | None]] = defaultdict(list)
    for sample in samples:
        grouped[sample.split].append(ranks[sample.query_id])
    return {split: _metrics(values) for split, values in sorted(grouped.items())}


def _metrics(ranks: list[int | None]) -> dict[str, float]:
    count = len(ranks)
    return {
        "recall@1": sum(rank is not None and rank <= 1 for rank in ranks) / count,
        "recall@3": sum(rank is not None and rank <= 3 for rank in ranks) / count,
        "recall@10": sum(rank is not None and rank <= 10 for rank in ranks) / count,
        "mrr@10": sum(1.0 / rank for rank in ranks if rank is not None and rank <= 10) / count,
    }


def _result_rows(samples, ranks, score_rows, page_ids):
    rows = []
    for sample, scores in zip(samples, score_rows, strict=True):
        order = sorted(range(len(page_ids)), key=lambda index: scores[index], reverse=True)[:10]
        rows.append(
            {
                "query_id": sample.query_id,
                "query": sample.query,
                "split": sample.split,
                "evidence_page_ids": sample.evidence_page_ids,
                "first_evidence_rank": ranks[sample.query_id],
                "top_10": [
                    {"page_id": page_ids[index], "rank": rank, "score": round(scores[index], 8)}
                    for rank, index in enumerate(order, start=1)
                ],
            }
        )
    return rows


if __name__ == "__main__":
    main()
