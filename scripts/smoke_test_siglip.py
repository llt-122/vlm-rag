from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlm_rag.dataset_schema import load_bundle
from vlm_rag.encoders import cosine_similarity
from vlm_rag.siglip_encoder import SigLIPConfig, SigLIPEncoder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode one ChartQA query and two page images with real SigLIP."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "real" / "chartqa",
    )
    parser.add_argument("--model", default="google/siglip-base-patch16-224")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    bundle = load_bundle(args.dataset_dir)
    if len(bundle.pages) < 2 or not bundle.samples:
        raise ValueError("the smoke test needs at least two pages and one QA sample")

    pages_by_id = {page.page_id: page for page in bundle.pages}
    sample = bundle.samples[0]
    evidence_page = pages_by_id[sample.evidence_page_ids[0]]
    unrelated_page = next(page for page in bundle.pages if page.page_id != evidence_page.page_id)
    evidence_path = _resolve_image_path(evidence_page.image_path)
    unrelated_path = _resolve_image_path(unrelated_page.image_path)

    encoder = SigLIPEncoder(
        SigLIPConfig(model_name=args.model, device=args.device, batch_size=2)
    )
    query_vector = encoder.encode_text(sample.query)
    evidence_vector, unrelated_vector = encoder.encode_images([evidence_path, unrelated_path])

    print(f"model: {args.model}")
    print(f"device: {encoder.device}")
    print(f"query: {sample.query}")
    print(f"evidence page: {evidence_page.page_id}")
    print(f"vector dimensions: text={len(query_vector)}, image={len(evidence_vector)}")
    print(f"evidence similarity: {cosine_similarity(query_vector, evidence_vector):.6f}")
    print(f"unrelated similarity: {cosine_similarity(query_vector, unrelated_vector):.6f}")


def _resolve_image_path(image_path: str) -> Path:
    path = Path(image_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
