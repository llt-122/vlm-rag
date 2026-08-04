from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Protocol

from .dataset_schema import DocumentPage, load_bundle
from .siglip_encoder import SigLIPConfig, SigLIPEncoder
from .siglip_index import SigLIPVectorIndex, load_siglip_index


class TextEncoder(Protocol):
    def encode_text(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class RetrievalResult:
    page_id: str
    doc_id: str
    page_no: int
    image_path: str
    score: float
    rank: int


class SigLIPRetrievalService:
    """Load a persisted page index and expose query-to-page retrieval.

    The encoder is initialized lazily because loading SigLIP can take several
    seconds and reserve GPU memory. This also lets the API health endpoint start
    before the model is loaded.
    """

    def __init__(
        self,
        project_root: Path,
        dataset_dir: Path,
        index_dir: Path,
        device: str = "auto",
        encoder: TextEncoder | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.dataset_dir = dataset_dir.resolve()
        self.index_dir = index_dir.resolve()
        self.index: SigLIPVectorIndex = load_siglip_index(self.index_dir)
        bundle = load_bundle(self.dataset_dir)
        self.pages: dict[str, DocumentPage] = {page.page_id: page for page in bundle.pages}
        if set(self.index.page_ids) != set(self.pages):
            raise ValueError("stored index pages differ from the configured dataset")
        self.model_name = str(self.index.metadata["model_name"])
        self.device = device
        self._encoder = encoder
        self._encoder_lock = Lock()

    @property
    def model_loaded(self) -> bool:
        return self._encoder is not None

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if top_k < 1 or top_k > min(50, len(self.index.page_ids)):
            raise ValueError(f"top_k must be between 1 and {min(50, len(self.index.page_ids))}")

        vector = self._get_encoder().encode_text(query)
        results: list[RetrievalResult] = []
        for hit in self.index.search_vector(vector, top_k=top_k):
            page = self.pages[hit.page_id]
            image_path = Path(page.image_path)
            resolved = image_path if image_path.is_absolute() else self.project_root / image_path
            results.append(
                RetrievalResult(
                    page_id=page.page_id,
                    doc_id=page.doc_id,
                    page_no=page.page_no,
                    image_path=str(resolved.resolve()),
                    score=hit.score,
                    rank=hit.rank,
                )
            )
        return results

    def _get_encoder(self) -> TextEncoder:
        if self._encoder is None:
            with self._encoder_lock:
                if self._encoder is None:
                    self._encoder = SigLIPEncoder(
                        SigLIPConfig(model_name=self.model_name, device=self.device, batch_size=1)
                    )
        return self._encoder
