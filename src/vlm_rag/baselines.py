from __future__ import annotations

from dataclasses import replace

from .data import Page, Query
from .encoders import EncoderConfig, HashingVLMEncoder
from .generator import WeightedVisualGenerator
from .metrics import accuracy, mrr_at_k, recall_at_k
from .retriever import DualTowerRetriever, SearchHit


class TitleOnlyEncoder(HashingVLMEncoder):
    def encode_page(self, page: Page) -> list[float]:
        # SigLIP 类全局向量基线更偏整体语义，这里弱化页面正文和结构化事实。
        title_only_page = replace(
            page,
            visual_text=page.title,
            layout="global semantic image embedding",
            facts={},
        )
        return super().encode_page(title_only_page)


class LayoutAwareEncoder(HashingVLMEncoder):
    def encode_page(self, page: Page) -> list[float]:
        # ColPali 类方法强调页面 patch/版式交互，这里额外保留 layout token。
        layout_page = replace(
            page,
            visual_text=f"{page.title} {page.visual_text}",
            layout=f"{page.layout} late interaction patch tokens",
        )
        return super().encode_page(layout_page)


def evaluate_method(
    method: str,
    pages: list[Page],
    queries: list[Query],
    top_k: int,
    embedding_dim: int = 384,
) -> dict[str, float | str]:
    # 所有方法共用同一评估协议，只替换页面编码方式或输入质量，
    # 保证对比表里的 MRR/Recall/Accuracy 口径一致。
    if method == "vlm_rag":
        encoder = HashingVLMEncoder(EncoderConfig(dim=embedding_dim, hidden_layer_weights=(0.2, 0.3, 0.5)))
        eval_pages = pages
    elif method == "ocr_rag":
        encoder = HashingVLMEncoder(EncoderConfig(dim=embedding_dim, hidden_layer_weights=(0.3, 0.4, 0.3)))
        eval_pages = [_ocr_corrupt_page(page) for page in pages]
    elif method == "siglip":
        encoder = TitleOnlyEncoder(EncoderConfig(dim=embedding_dim, hidden_layer_weights=(0.2, 0.2, 0.6)))
        eval_pages = pages
    elif method == "colpali":
        encoder = LayoutAwareEncoder(EncoderConfig(dim=embedding_dim, hidden_layer_weights=(0.15, 0.35, 0.5)))
        eval_pages = pages
    else:
        raise ValueError(f"unsupported method: {method}")

    retriever = DualTowerRetriever(encoder)
    retriever.index(eval_pages)
    generator = WeightedVisualGenerator()

    ranked_hits: dict[str, list[SearchHit]] = {}
    predictions: dict[str, str] = {}
    for query in queries:
        hits = retriever.search(query.text, top_k=top_k)
        ranked_hits[query.query_id] = hits
        predictions[query.query_id] = generator.answer(query.text, hits).text

    return {
        "method": method,
        "mrr@10": round(mrr_at_k(queries, ranked_hits, 10), 4),
        f"recall@{top_k}": round(recall_at_k(queries, ranked_hits, top_k), 4),
        "em": round(accuracy(predictions, queries), 4),
        "accuracy": round(accuracy(predictions, queries), 4),
    }


def _ocr_corrupt_page(page: Page) -> Page:
    # OCR baseline 故意引入字符混淆和字段丢失，用来模拟 OCR 误差传导。
    visual_text = (
        page.visual_text.replace("0", "O")
        .replace("1", "I")
        .replace("5", "S")
        .replace("工龄", "工令")
        .replace("金额", "全额")
        .replace("告警", "吉警")
    )
    facts = {
        key: value
        for index, (key, value) in enumerate(page.facts.items())
        if index == 0 or page.doc_type in {"contract", "policy"}
    }
    return replace(page, visual_text=visual_text, layout="ocr text chunks without visual layout", facts=facts)
