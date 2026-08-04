from __future__ import annotations

import os
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .api_service import SigLIPRetrievalService


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=50)


@lru_cache(maxsize=1)
def get_service() -> SigLIPRetrievalService:
    dataset_dir = Path(
        os.getenv("VLM_RAG_DATASET_DIR", str(PROJECT_ROOT / "data" / "real" / "chartqa"))
    )
    index_dir = Path(
        os.getenv("VLM_RAG_INDEX_DIR", str(PROJECT_ROOT / "indexes" / "siglip_chartqa"))
    )
    return SigLIPRetrievalService(
        project_root=PROJECT_ROOT,
        dataset_dir=dataset_dir,
        index_dir=index_dir,
        device=os.getenv("VLM_RAG_DEVICE", "auto"),
    )


app = FastAPI(
    title="VLM-RAG Page Retrieval API",
    version="0.1.0",
    description="Text Query to document-page image retrieval for Dify or other workflows.",
)


@app.get("/health")
def health() -> dict[str, object]:
    try:
        service = get_service()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ok",
        "model": service.model_name,
        "page_count": len(service.index.page_ids),
        "model_loaded": service.model_loaded,
    }


@app.post("/v1/search")
def search(request: SearchRequest) -> dict[str, object]:
    try:
        service = get_service()
        hits = service.search(request.query, request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "query": request.query,
        "top_k": request.top_k,
        "results": [asdict(hit) for hit in hits],
    }
