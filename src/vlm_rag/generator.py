from __future__ import annotations

import re
from dataclasses import dataclass

from .retriever import SearchHit


@dataclass(frozen=True)
class Answer:
    text: str
    evidence_page_ids: list[str]
    confidence: float


class WeightedVisualGenerator:
    def answer(self, query: str, hits: list[SearchHit]) -> Answer:
        if not hits:
            return Answer("未检索到相关页面", [], 0.0)

        query_key = _infer_fact_key(query)
        weighted_candidates: dict[str, float] = {}
        evidence: dict[str, list[str]] = {}

        for hit in hits:
            for fact_key, fact_value in hit.page.facts.items():
                # 检索分数表示页面相关性，字段重叠度表示问题与候选事实的匹配度；
                # 两者相加模拟多图问答里的“页面证据加权融合”。
                fact_score = hit.score + _lexical_overlap(query_key, fact_key)
                weighted_candidates[fact_value] = weighted_candidates.get(fact_value, 0.0) + fact_score
                evidence.setdefault(fact_value, []).append(hit.page.page_id)

        best_answer = max(weighted_candidates.items(), key=lambda item: item[1])
        # 将累积分数压到 0-1 区间，作为演示版置信度。
        confidence = 1.0 / (1.0 + pow(2.718281828, -best_answer[1]))
        return Answer(best_answer[0], evidence[best_answer[0]], round(confidence, 4))


def _infer_fact_key(query: str) -> str:
    replacements = {
        "是什么": "",
        "多少": "",
        "多少钱": "",
        "哪个": "",
        "？": "",
        "?": "",
        "的": "",
        "里": "",
        "这次": "",
    }
    result = query
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


def _lexical_overlap(left: str, right: str) -> float:
    left_chars = set(re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", left.lower()))
    right_chars = set(re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", right.lower()))
    if not left_chars or not right_chars:
        return 0.0
    return len(left_chars & right_chars) / len(left_chars | right_chars)
