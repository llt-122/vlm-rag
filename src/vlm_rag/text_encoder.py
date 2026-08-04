from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TextEncoderConfig:
    model_name: str = "BAAI/bge-small-en-v1.5"
    device: str = "auto"
    batch_size: int = 32


class SentenceTextEncoder:
    """Dense text encoder for Query-to-OCR-text retrieval."""

    def __init__(self, config: TextEncoderConfig | None = None) -> None:
        self.config = config or TextEncoderConfig()
        if self.config.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Text retrieval requires sentence-transformers in the model environment."
            ) from exc

        if self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device
        self.model = SentenceTransformer(self.config.model_name, device=self.device)

    def encode_queries(self, queries: Iterable[str]) -> list[list[float]]:
        values = _validate_texts(queries)
        if not values:
            return []
        vectors = self.model.encode_query(
            values,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32").tolist()

    def encode_documents(self, documents: Iterable[str]) -> list[list[float]]:
        values = _validate_texts(documents, allow_empty=True)
        if not values:
            return []
        values = [value if value.strip() else "[EMPTY OCR PAGE]" for value in values]
        vectors = self.model.encode_document(
            values,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32").tolist()


def _validate_texts(texts: Iterable[str], allow_empty: bool = False) -> list[str]:
    values = list(texts)
    if not allow_empty and any(not value.strip() for value in values):
        raise ValueError("text inputs must not be empty")
    return values
