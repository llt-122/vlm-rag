from __future__ import annotations

import csv
import json
from pathlib import Path

from .data import Page, Query
from .encoders import EncoderConfig, HashingVLMEncoder, info_nce_loss
from .metrics import mrr_at_k, recall_at_k
from .retriever import DualTowerRetriever


WEIGHT_CANDIDATES: tuple[tuple[float, ...], ...] = (
    (0.2, 0.3, 0.5),
    (0.1, 0.3, 0.6),
    (0.15, 0.35, 0.5),
    (0.25, 0.25, 0.5),
    (0.3, 0.3, 0.4),
)


def train_retriever(
    pages: list[Page],
    queries: list[Query],
    model_dir: Path,
    embedding_dim: int = 384,
    temperature: float = 0.07,
    epochs: int = 5,
) -> dict[str, object]:
    # 真实 VLM 微调会更新模型参数；本项目为了保证无依赖可运行，
    # 用隐藏层池化权重搜索模拟训练过程，并保留 InfoNCE 评估日志。
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_dir / "training_log.csv"
    best_state: dict[str, object] | None = None

    with log_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "weights", "info_nce_loss", "mrr@10", "recall@3"],
        )
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            weights = WEIGHT_CANDIDATES[(epoch - 1) % len(WEIGHT_CANDIDATES)]
            encoder = HashingVLMEncoder(EncoderConfig(dim=embedding_dim, hidden_layer_weights=weights))
            metrics = _evaluate_encoder(encoder, pages, queries, temperature)
            row = {
                "epoch": epoch,
                "weights": ",".join(str(weight) for weight in weights),
                "info_nce_loss": metrics["info_nce_loss"],
                "mrr@10": metrics["mrr@10"],
                "recall@3": metrics["recall@3"],
            }
            writer.writerow(row)
            # 选型时同时看召回效果和对比学习损失，避免只追求单一指标。
            if best_state is None or _score(metrics) > _score(best_state):
                best_state = {
                    "epoch": epoch,
                    "embedding_dim": embedding_dim,
                    "temperature": temperature,
                    "hidden_layer_weights": weights,
                    **metrics,
                }

    if best_state is None:
        raise RuntimeError("training did not produce a best state")

    (model_dir / "retriever_config.json").write_text(
        json.dumps(best_state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return best_state


def _evaluate_encoder(
    encoder: HashingVLMEncoder,
    pages: list[Page],
    queries: list[Query],
    temperature: float,
) -> dict[str, float]:
    retriever = DualTowerRetriever(encoder)
    retriever.index(pages)
    ranked_hits = {query.query_id: retriever.search(query.text, top_k=10) for query in queries}
    # InfoNCE 只需要 Query 向量与正例页面向量；MRR/Recall 用完整检索排序评估。
    query_vectors = [encoder.encode_query(query.text) for query in queries]
    positive_pages = [next(page for page in pages if page.page_id == query.positive_page_ids[0]) for query in queries]
    positive_vectors = [encoder.encode_page(page) for page in positive_pages]
    return {
        "info_nce_loss": round(info_nce_loss(query_vectors, positive_vectors, temperature), 4),
        "mrr@10": round(mrr_at_k(queries, ranked_hits, 10), 4),
        "recall@3": round(recall_at_k(queries, ranked_hits, 3), 4),
    }


def _score(metrics: dict[str, object]) -> float:
    return float(metrics["mrr@10"]) + float(metrics["recall@3"]) - float(metrics["info_nce_loss"]) * 0.01
