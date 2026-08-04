from pathlib import Path

from vlm_rag.api_service import SigLIPRetrievalService


class FakeEncoder:
    def encode_text(self, text: str) -> list[float]:
        assert text == "revenue"
        return [1.0, 0.0]


def test_search_returns_page_metadata(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    index_dir = tmp_path / "index"
    dataset_dir.mkdir()
    index_dir.mkdir()
    (dataset_dir / "pages.jsonl").write_text(
        '{"page_id":"p1","doc_id":"d1","doc_type":"chart","page_no":1,'
        '"image_path":"pages/p1.png","title":"","metadata":{}}\n'
        '{"page_id":"p2","doc_id":"d2","doc_type":"chart","page_no":2,'
        '"image_path":"pages/p2.png","title":"","metadata":{}}\n',
        encoding="utf-8",
    )
    (dataset_dir / "samples.jsonl").write_text("", encoding="utf-8")
    (index_dir / "page_vectors.json").write_text(
        '{"p1":[1.0,0.0],"p2":[0.0,1.0]}', encoding="utf-8"
    )
    (index_dir / "index_metadata.json").write_text(
        '{"format_version":"1.0","model_name":"fake","page_count":2,'
        '"embedding_dim":2,"normalized":true,"similarity":"cosine",'
        '"page_ids":["p1","p2"]}',
        encoding="utf-8",
    )

    service = SigLIPRetrievalService(
        project_root=tmp_path,
        dataset_dir=dataset_dir,
        index_dir=index_dir,
        encoder=FakeEncoder(),
    )
    hits = service.search("revenue", top_k=1)

    assert hits[0].page_id == "p1"
    assert hits[0].doc_id == "d1"
    assert hits[0].rank == 1
    assert hits[0].score == 1.0
