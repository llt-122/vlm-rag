from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlm_rag.siglip_index import SigLIPVectorIndex, load_siglip_index, save_siglip_index


class SigLIPVectorIndexTests(unittest.TestCase):
    def test_search_and_round_trip(self) -> None:
        index = SigLIPVectorIndex(
            page_ids=["p1", "p2"],
            vectors={"p1": [1.0, 0.0], "p2": [0.0, 1.0]},
            metadata={
                "format_version": "1.0",
                "model_name": "test-model",
                "page_count": 2,
                "embedding_dim": 2,
                "normalized": True,
                "similarity": "cosine",
            },
        )
        self.assertEqual([hit.page_id for hit in index.search_vector([0.9, 0.1])], ["p1", "p2"])

        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir)
            save_siglip_index(index, path)
            loaded = load_siglip_index(path)
        self.assertEqual(loaded, index)

    def test_rejects_wrong_query_dimension(self) -> None:
        index = SigLIPVectorIndex(
            page_ids=["p1"],
            vectors={"p1": [1.0, 0.0]},
            metadata={"page_count": 1, "embedding_dim": 2},
        )
        with self.assertRaises(ValueError):
            index.search_vector([1.0])


if __name__ == "__main__":
    unittest.main()
