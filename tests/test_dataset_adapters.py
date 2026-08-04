from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.vlm_rag.dataset_adapters import convert_legacy_demo_dataset
from src.vlm_rag.dataset_schema import DatasetBundle, DocumentPage, DocumentQASample, load_bundle, save_bundle


class DatasetSchemaTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        bundle = DatasetBundle(
            pages=[
                DocumentPage(
                    page_id="page_001",
                    doc_id="doc_001",
                    doc_type="contract",
                    page_no=1,
                    image_path="pages/page_001.png",
                )
            ],
            samples=[
                DocumentQASample(
                    query_id="q001",
                    query="付款比例是多少？",
                    answers=["30%"],
                    evidence_page_ids=["page_001"],
                    doc_type="contract",
                    split="train",
                )
            ],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            manifest = save_bundle(bundle, output_dir)
            loaded = load_bundle(output_dir)

        self.assertEqual(manifest["page_count"], 1)
        self.assertEqual(loaded, bundle)

    def test_legacy_conversion(self) -> None:
        pages = [
            {
                "page_id": "page_001",
                "doc_id": "doc_001",
                "doc_type": "contract",
                "page_no": 1,
                "title": "合同",
                "visual_text": "首付款 30%",
                "layout": "table",
                "facts": {"首付款": "30%"},
                "image_path": "page_001.svg",
            }
        ]
        queries = [
            {
                "query_id": "q001",
                "text": "首付款是多少？",
                "answer": "30%",
                "positive_page_ids": ["page_001"],
                "doc_type": "contract",
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pages_path = root / "pages.json"
            queries_path = root / "queries.json"
            pages_path.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
            queries_path.write_text(json.dumps(queries, ensure_ascii=False), encoding="utf-8")
            bundle = convert_legacy_demo_dataset(pages_path, queries_path)

        self.assertEqual(bundle.pages[0].metadata["facts"], {"首付款": "30%"})
        self.assertEqual(bundle.samples[0].answers, ["30%"])
        self.assertEqual(bundle.samples[0].split, "unspecified")


if __name__ == "__main__":
    unittest.main()
