from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ColSmolConfig:
    model_name: str = "vidore/colSmol-500M"
    device: str = "cuda:0"
    image_batch_size: int = 1
    query_batch_size: int = 8


class ColSmolEncoder:
    """Small ColPali-family encoder with ColBERT-style multi-vector scoring."""

    def __init__(self, config: ColSmolConfig | None = None) -> None:
        self.config = config or ColSmolConfig()
        try:
            import torch
            from colpali_engine.models import ColIdefics3, ColIdefics3Processor
        except ImportError as exc:
            raise RuntimeError(
                "ColSmol requires the isolated .venv-colpali environment from README.md."
            ) from exc

        self._torch = torch
        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the ColSmol baseline")
        self.model = ColIdefics3.from_pretrained(
            self.config.model_name,
            dtype=torch.bfloat16,
            device_map=self.config.device,
            attn_implementation="eager",
        ).eval()
        self.processor = ColIdefics3Processor.from_pretrained(self.config.model_name)

    @property
    def device(self) -> Any:
        return self.model.device

    def encode_images(self, image_paths: Iterable[Path]) -> list[Any]:
        from PIL import Image

        paths = [Path(path) for path in image_paths]
        embeddings: list[Any] = []
        for batch_paths in _batches(paths, self.config.image_batch_size):
            images = []
            try:
                for path in batch_paths:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                inputs = self.processor.process_images(images).to(self.device)
                with self._torch.inference_mode():
                    batch_embeddings = self.model(**inputs)
                embeddings.extend(vector.detach().cpu().float() for vector in batch_embeddings)
            finally:
                for image in images:
                    image.close()
        return embeddings

    def encode_queries(self, queries: Iterable[str]) -> list[Any]:
        values = list(queries)
        if any(not query.strip() for query in values):
            raise ValueError("queries must not be empty")
        embeddings: list[Any] = []
        for batch in _batches(values, self.config.query_batch_size):
            inputs = self.processor.process_queries(batch).to(self.device)
            with self._torch.inference_mode():
                batch_embeddings = self.model(**inputs)
            embeddings.extend(vector.detach().cpu().float() for vector in batch_embeddings)
        return embeddings

    def score(self, query_embeddings: list[Any], page_embeddings: list[Any]) -> Any:
        query_embeddings = [embedding.float() for embedding in query_embeddings]
        page_embeddings = [embedding.float() for embedding in page_embeddings]
        return self.processor.score_multi_vector(
            query_embeddings,
            page_embeddings,
            batch_size=8,
            device=self.device,
        )


def _batches(values: list[Any], batch_size: int) -> Iterable[list[Any]]:
    if batch_size < 1:
        raise ValueError("batch size must be at least 1")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]
