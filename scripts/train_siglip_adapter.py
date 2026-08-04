from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlm_rag.dataset_schema import DocumentQASample, load_bundle
from vlm_rag.retrieval_adapter import DualTowerAdapter, SharedTowerAdapter
from vlm_rag.siglip_encoder import SigLIPConfig, SigLIPEncoder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train lightweight dual-tower residual heads with InfoNCE."
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=PROJECT_ROOT / "data" / "real" / "chartqa"
    )
    parser.add_argument(
        "--page-index-dir", type=Path, default=PROJECT_ROOT / "indexes" / "siglip_chartqa"
    )
    parser.add_argument(
        "--feature-dir", type=Path, default=PROJECT_ROOT / "features" / "siglip_chartqa"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "siglip_adapter_query"
    )
    parser.add_argument("--model", default="google/siglip-base-patch16-224")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--hard-negatives",
        type=int,
        default=0,
        help="Use this many highest-scoring wrong training pages per query; 0 uses all pages.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-towers", choices=("query", "both", "shared"), default="query")
    args = parser.parse_args()

    import torch
    from torch.nn import functional as F

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = load_bundle(args.dataset_dir)
    page_ids, page_vectors = _load_page_vectors(args.page_index_dir, bundle.pages)
    query_vectors = _load_or_encode_queries(
        bundle.samples, args.feature_dir, args.model
    )
    page_tensor = torch.tensor(page_vectors, dtype=torch.float32, device=device)
    query_tensor = torch.tensor(query_vectors, dtype=torch.float32, device=device)
    embedding_dim = page_tensor.shape[1]
    page_position = {page_id: index for index, page_id in enumerate(page_ids)}

    train_indices = [index for index, sample in enumerate(bundle.samples) if sample.split == "train"]
    train_page_positions = sorted(
        {page_position[page_id] for index in train_indices for page_id in bundle.samples[index].evidence_page_ids}
    )
    train_candidate_lookup = {
        page_position: candidate_index
        for candidate_index, page_position in enumerate(train_page_positions)
    }
    train_targets = torch.tensor(
        [
            train_candidate_lookup[page_position[bundle.samples[index].evidence_page_ids[0]]]
            for index in train_indices
        ],
        dtype=torch.long,
        device=device,
    )
    train_query_tensor = query_tensor[train_indices]
    train_page_tensor = page_tensor[train_page_positions]
    hard_negative_indices = _mine_hard_negatives(
        train_query_tensor,
        train_page_tensor,
        train_targets,
        args.hard_negatives,
    )

    if args.train_towers == "shared":
        adapter = SharedTowerAdapter(embedding_dim=embedding_dim, rank=args.rank).to(device)
    else:
        adapter = DualTowerAdapter(embedding_dim=embedding_dim, rank=args.rank).to(device)
    if args.train_towers == "query":
        for parameter in adapter.page_projection.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in adapter.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    baseline = _evaluate_all(bundle.samples, query_tensor, page_tensor, page_ids, None)
    best_state = copy.deepcopy({name: value.detach().cpu() for name, value in adapter.state_dict().items()})
    best_epoch = 0
    best_dev_mrr = baseline["by_split"]["dev"]["mrr@10"]
    epochs_without_improvement = 0
    log_rows: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        adapted_queries = adapter.encode_queries(train_query_tensor)
        adapted_pages = adapter.encode_pages(train_page_tensor)
        if hard_negative_indices is None:
            logits = adapted_queries @ adapted_pages.T / args.temperature
            loss = F.cross_entropy(logits, train_targets)
        else:
            candidate_indices = torch.cat([train_targets[:, None], hard_negative_indices], dim=1)
            candidate_pages = adapted_pages[candidate_indices]
            logits = torch.einsum("bd,bkd->bk", adapted_queries, candidate_pages)
            logits = logits / args.temperature
            loss = F.cross_entropy(
                logits,
                torch.zeros(len(train_targets), dtype=torch.long, device=device),
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
        optimizer.step()

        adapter.eval()
        evaluation = _evaluate_all(bundle.samples, query_tensor, page_tensor, page_ids, adapter)
        dev_mrr = evaluation["by_split"]["dev"]["mrr@10"]
        log_rows.append(
            {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "train_mrr@10": evaluation["by_split"]["train"]["mrr@10"],
                "dev_mrr@10": dev_mrr,
                "dev_recall@1": evaluation["by_split"]["dev"]["recall@1"],
            }
        )
        if dev_mrr > best_dev_mrr + 1e-9:
            best_dev_mrr = dev_mrr
            best_epoch = epoch
            best_state = copy.deepcopy(
                {name: value.detach().cpu() for name, value in adapter.state_dict().items()}
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"epoch={epoch} loss={float(loss.detach()):.4f} "
                f"train_mrr={evaluation['by_split']['train']['mrr@10']:.4f} "
                f"dev_mrr={dev_mrr:.4f}"
            )
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    adapter.load_state_dict(best_state)
    adapter.to(device).eval()
    tuned = _evaluate_all(bundle.samples, query_tensor, page_tensor, page_ids, adapter)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "adapter.pt"
    torch.save(
        {
            "state_dict": best_state,
            "embedding_dim": embedding_dim,
            "rank": args.rank,
            "model_name": args.model,
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )
    with (args.output_dir / "training_log.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(log_rows[0]))
        writer.writeheader()
        writer.writerows(log_rows)
    metrics = {
        "method": "siglip_frozen_with_dual_residual_adapter",
        "model_name": args.model,
        "train_query_count": len(train_indices),
        "train_page_count": len(train_page_positions),
        "best_epoch": best_epoch,
        "selection_metric": "dev_mrr@10",
        "rank": args.rank,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "seed": args.seed,
        "train_towers": args.train_towers,
        "hard_negatives": args.hard_negatives,
        "baseline": baseline,
        "tuned": tuned,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best_epoch: {best_epoch}")
    for split in ("train", "dev", "test"):
        before = baseline["by_split"][split]
        after = tuned["by_split"][split]
        print(
            f"{split}: MRR@10 {before['mrr@10']:.4f} -> {after['mrr@10']:.4f}, "
            f"Recall@1 {before['recall@1']:.4f} -> {after['recall@1']:.4f}"
        )
    print(f"checkpoint: {_relative(checkpoint_path)}")
    print(f"metrics: {_relative(metrics_path)}")


def _load_page_vectors(index_dir: Path, pages: list[object]) -> tuple[list[str], list[list[float]]]:
    metadata = json.loads((index_dir / "index_metadata.json").read_text(encoding="utf-8"))
    vectors = json.loads((index_dir / "page_vectors.json").read_text(encoding="utf-8"))
    page_ids = [page.page_id for page in pages]
    if metadata["page_ids"] != page_ids or set(vectors) != set(page_ids):
        raise ValueError("SigLIP page index does not match the current dataset")
    return page_ids, [vectors[page_id] for page_id in page_ids]


def _load_or_encode_queries(
    samples: list[DocumentQASample], feature_dir: Path, model_name: str
) -> list[list[float]]:
    vectors_path = feature_dir / "query_vectors.json"
    metadata_path = feature_dir / "query_metadata.json"
    query_ids = [sample.query_id for sample in samples]
    if vectors_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        vectors = json.loads(vectors_path.read_text(encoding="utf-8"))
        if metadata["model_name"] == model_name and metadata["query_ids"] == query_ids:
            print(f"Loaded {len(query_ids)} cached query embeddings.")
            return [vectors[query_id] for query_id in query_ids]

    print(f"Encoding {len(samples)} frozen SigLIP query features...")
    encoder = SigLIPEncoder(SigLIPConfig(model_name=model_name, batch_size=16))
    encoded = encoder.encode_texts(sample.query for sample in samples)
    feature_dir.mkdir(parents=True, exist_ok=True)
    vectors_path.write_text(
        json.dumps(dict(zip(query_ids, encoded, strict=True)), ensure_ascii=False), encoding="utf-8"
    )
    metadata_path.write_text(
        json.dumps({"model_name": model_name, "query_ids": query_ids}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return encoded


def _evaluate_all(
    samples: list[DocumentQASample],
    query_tensor: object,
    page_tensor: object,
    page_ids: list[str],
    adapter: object | None,
) -> dict[str, object]:
    import torch

    with torch.inference_mode():
        if adapter is None:
            queries = query_tensor
            pages = page_tensor
        else:
            queries = adapter.encode_queries(query_tensor)
            pages = adapter.encode_pages(page_tensor)
        scores = queries @ pages.T
    ranks_by_split: dict[str, list[int | None]] = defaultdict(list)
    all_ranks: list[int | None] = []
    for sample, score_row in zip(samples, scores, strict=True):
        order = torch.argsort(score_row, descending=True)[:10].tolist()
        evidence = set(sample.evidence_page_ids)
        rank = next(
            (position for position, index in enumerate(order, start=1) if page_ids[index] in evidence),
            None,
        )
        all_ranks.append(rank)
        ranks_by_split[sample.split].append(rank)
    return {
        "overall": _metrics(all_ranks),
        "by_split": {split: _metrics(ranks) for split, ranks in sorted(ranks_by_split.items())},
    }


def _mine_hard_negatives(
    queries: object,
    pages: object,
    targets: object,
    count: int,
) -> object | None:
    if count <= 0:
        return None
    import torch

    if count >= pages.shape[0]:
        raise ValueError("hard-negative count must be smaller than the train page count")
    with torch.inference_mode():
        scores = queries @ pages.T
        scores[torch.arange(len(targets), device=scores.device), targets] = -torch.inf
        return torch.topk(scores, k=count, dim=1).indices


def _metrics(ranks: list[int | None]) -> dict[str, float]:
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
