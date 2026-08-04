from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .dataset_schema import DocumentPage
from .encoders import cosine_similarity
from .siglip_encoder import SigLIPEncoder


@dataclass(frozen=True)
class SigLIPSearchHit:
    page_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class SigLIPVectorIndex:
    """A small exact-search index for normalized SigLIP page vectors."""

    page_ids: list[str]
    vectors: dict[str, list[float]]
    metadata: dict[str, object]

    def search_vector(self, query_vector: list[float], top_k: int = 10) -> list[SigLIPSearchHit]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        embedding_dim = int(self.metadata["embedding_dim"])
        if len(query_vector) != embedding_dim:
            raise ValueError(
                f"query dimension {len(query_vector)} does not match index dimension {embedding_dim}"
            )

        scored = [
            (page_id, cosine_similarity(query_vector, self.vectors[page_id]))
            for page_id in self.page_ids
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            SigLIPSearchHit(page_id=page_id, score=score, rank=rank)
            for rank, (page_id, score) in enumerate(scored[:top_k], start=1)
        ]


def build_siglip_index(
    pages: list[DocumentPage],
    encoder: SigLIPEncoder,
    project_root: Path,
) -> SigLIPVectorIndex:
    if not pages:
        raise ValueError("cannot build an index without pages")

    page_ids = [page.page_id for page in pages]
    if len(page_ids) != len(set(page_ids)):
        raise ValueError("page_id values must be unique")
    image_paths = [_resolve_path(page.image_path, project_root) for page in pages]
    missing_paths = [str(path) for path in image_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"page images are missing: {missing_paths[:3]}")

    encoded = encoder.encode_images(image_paths)
    vectors = dict(zip(page_ids, encoded, strict=True))
    embedding_dim = len(encoded[0])
    metadata: dict[str, object] = {
        "format_version": "1.0",
        "model_name": encoder.config.model_name,
        "page_count": len(page_ids),
        "embedding_dim": embedding_dim,
        "normalized": True,
        "similarity": "cosine",
    }
    return SigLIPVectorIndex(page_ids=page_ids, vectors=vectors, metadata=metadata)


def save_siglip_index(index: SigLIPVectorIndex, index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "page_vectors.json").write_text(
        json.dumps(index.vectors, ensure_ascii=False),
        encoding="utf-8",
    )
    (index_dir / "index_metadata.json").write_text(
        json.dumps({**index.metadata, "page_ids": index.page_ids}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_siglip_index(index_dir: Path) -> SigLIPVectorIndex:
    vectors_path = index_dir / "page_vectors.json"
    metadata_path = index_dir / "index_metadata.json"
    if not vectors_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"SigLIP index is missing at {index_dir}")

    raw_vectors = json.loads(vectors_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    page_ids = metadata.pop("page_ids")
    vectors = {str(page_id): [float(value) for value in vector] for page_id, vector in raw_vectors.items()}
    _validate_loaded_index(page_ids, vectors, metadata)
    return SigLIPVectorIndex(page_ids=page_ids, vectors=vectors, metadata=metadata)


def _validate_loaded_index(
    page_ids: list[str],
    vectors: dict[str, list[float]],
    metadata: dict[str, object],
) -> None:
    if set(page_ids) != set(vectors):
        raise ValueError("index page_ids and vector keys do not match")
    if int(metadata["page_count"]) != len(page_ids):
        raise ValueError("index page_count does not match stored pages")
    embedding_dim = int(metadata["embedding_dim"])
    if any(len(vector) != embedding_dim for vector in vectors.values()):
        raise ValueError("one or more stored vectors have an invalid dimension")


def _resolve_path(path_value: str, project_root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path
