from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any


@dataclass(frozen=True)
class OCRPageResult:
    page_id: str
    text: str
    lines: list[str]
    scores: list[float]
    elapsed_ms: float


class PaddleOCRExtractor:
    """Local PP-OCR wrapper used to build the OCR-RAG baseline corpus."""

    def __init__(self, language: str = "en", minimum_score: float = 0.5) -> None:
        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError("minimum_score must be between 0 and 1")
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Use the .venv-ocr environment from README.md."
            ) from exc

        self.minimum_score = minimum_score
        # MKL-DNN is disabled because Paddle 3.3.1's Windows CPU backend cannot
        # execute an attribute used by the PP-OCRv6 detection model.
        self.model = PaddleOCR(
            lang=language,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )

    def extract(self, page_id: str, image_path: Path) -> OCRPageResult:
        started = perf_counter()
        predictions = list(self.model.predict(str(image_path)))
        lines: list[str] = []
        scores: list[float] = []
        for prediction in predictions:
            payload = prediction.json
            result = payload.get("res", payload)
            raw_lines = result.get("rec_texts", [])
            raw_scores = result.get("rec_scores", [])
            for line, score in zip(raw_lines, raw_scores, strict=False):
                clean_line = str(line).strip()
                numeric_score = float(score)
                if clean_line and numeric_score >= self.minimum_score:
                    lines.append(clean_line)
                    scores.append(numeric_score)
        elapsed_ms = (perf_counter() - started) * 1000.0
        return OCRPageResult(
            page_id=page_id,
            text=" ".join(lines),
            lines=lines,
            scores=scores,
            elapsed_ms=elapsed_ms,
        )
