from __future__ import annotations

from dataclasses import dataclass

from .data import Page
from .encoders import HashingVLMEncoder, cosine_similarity


@dataclass(frozen=True)
class SearchHit:
    page: Page
    score: float
    rank: int


class DualTowerRetriever:
    def __init__(self, encoder: HashingVLMEncoder) -> None:
        self.encoder = encoder
        self.pages: list[Page] = []
        self.page_vectors: dict[str, list[float]] = {}

    def index(self, pages: list[Page]) -> None:
        self.pages = pages
        # 页面向量在离线阶段预先计算；真实系统中通常会写入 FAISS/Milvus。
        self.page_vectors = {page.page_id: self.encoder.encode_page(page) for page in pages}

    def search(self, query: str, top_k: int = 3) -> list[SearchHit]:
        query_vector = self.encoder.encode_query(query)
        # 双塔检索的线上阶段只需要编码 Query，再与页面向量做相似度排序。
        scored = [
            (page, cosine_similarity(query_vector, self.page_vectors[page.page_id]))
            for page in self.pages
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            SearchHit(page=page, score=score, rank=rank)
            for rank, (page, score) in enumerate(scored[:top_k], start=1)
        ]
