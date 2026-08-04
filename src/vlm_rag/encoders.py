from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from .data import Page


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")


@dataclass(frozen=True)
class EncoderConfig:
    dim: int = 384
    hidden_layer_weights: tuple[float, ...] = (0.2, 0.3, 0.5)


class HashingVLMEncoder:
    """A local, dependency-free stand-in for a VLM dual-tower encoder.

    It keeps the same interface as a production VLM encoder: text queries and
    page images are projected into one normalized vector space. The demo uses
    deterministic hashing so it can run anywhere without network downloads.
    """

    def __init__(self, config: EncoderConfig | None = None) -> None:
        self.config = config or EncoderConfig()

    def encode_query(self, text: str) -> list[float]:
        tokens = _tokenize(text)
        return self._weighted_pool(tokens)

    def encode_page(self, page: Page) -> list[float]:
        # 生产环境中这里会读取页面图像并送入 VLM；本地演示把页面标题、
        # 版式描述和结构化事实拼成“视觉语义 token”，保持同一调用接口。
        visual_tokens = _tokenize(
            " ".join(
                [
                    page.doc_type,
                    page.title,
                    page.visual_text,
                    page.layout,
                    " ".join(page.facts.keys()),
                    " ".join(page.facts.values()),
                ]
            )
        )
        return self._weighted_pool(visual_tokens)

    def _weighted_pool(self, tokens: list[str]) -> list[float]:
        if not tokens:
            return [0.0 for _ in range(self.config.dim)]

        pooled = [0.0 for _ in range(self.config.dim)]
        for layer_index, layer_weight in enumerate(self.config.hidden_layer_weights):
            # 用不同 layer 前缀模拟 VLM 不同隐藏层。浅层更像视觉/布局特征，
            # 深层更像语义特征，最后按权重融合成一个页面向量。
            layer_vector = [0.0 for _ in range(self.config.dim)]
            for token in tokens:
                index, sign = _hash_token(f"l{layer_index}:{token}", self.config.dim)
                layer_vector[index] += sign
            layer_vector = _l2_normalize(layer_vector)
            for index, value in enumerate(layer_vector):
                pooled[index] += layer_weight * value
        return _l2_normalize(pooled)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def info_nce_loss(
    query_vectors: list[list[float]],
    positive_vectors: list[list[float]],
    temperature: float = 0.07,
) -> float:
    if len(query_vectors) != len(positive_vectors):
        raise ValueError("query_vectors and positive_vectors must have the same length")

    losses: list[float] = []
    for row_index, query_vector in enumerate(query_vectors):
        # 同一个 batch 内，当前位置的页面是正例，其余页面自然充当负例。
        # 这对应双塔检索器常用的 in-batch negatives 训练方式。
        logits = [
            cosine_similarity(query_vector, page_vector) / temperature
            for page_vector in positive_vectors
        ]
        max_logit = max(logits)
        exp_sum = sum(math.exp(logit - max_logit) for logit in logits)
        log_prob = logits[row_index] - max_logit - math.log(exp_sum)
        losses.append(-log_prob)
    return sum(losses) / len(losses)


def _tokenize(text: str) -> list[str]:
    raw_tokens = TOKEN_PATTERN.findall(text.lower())
    tokens: list[str] = []
    for token in raw_tokens:
        tokens.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]+", token) and len(token) > 1:
            # 中文没有天然空格，补充 bigram 可以让“服务期限”和“期限”
            # 这类局部匹配在哈希向量里产生更稳定的相似度。
            tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
    return tokens


def _hash_token(token: str, dim: int) -> tuple[int, float]:
    # signed hashing 能在不引入第三方向量库的前提下，构造稳定的稀疏投影。
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return value % dim, 1.0 if value & 1 else -1.0


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]
