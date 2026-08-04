from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlm_rag.dataset_schema import load_bundle
from vlm_rag.smolvlm_generator import SmolVLMConfig, SmolVLMGenerator


def main() -> None:
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    parser = argparse.ArgumentParser(description="Evaluate Top-K page collage visual QA.")
    parser.add_argument(
        "--dataset-dir", type=Path, default=PROJECT_ROOT / "data" / "real" / "chartqa"
    )
    parser.add_argument(
        "--retrieval-results",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "colsmol_chartqa" / "retrieval_results.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "top3_collage_qa"
    )
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from PIL import Image, ImageDraw

    bundle = load_bundle(args.dataset_dir)
    samples = bundle.samples[: args.limit or None]
    pages_by_id = {page.page_id: page for page in bundle.pages}
    retrieval_rows = json.loads(args.retrieval_results.read_text(encoding="utf-8"))
    retrieval_by_query = {row["query_id"]: row for row in retrieval_rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    completed = _load_predictions(predictions_path)
    remaining = [sample for sample in samples if sample.query_id not in completed]

    if remaining:
        generator = SmolVLMGenerator(SmolVLMConfig(model_name=args.model))
        with predictions_path.open("a", encoding="utf-8") as output_file:
            for position, sample in enumerate(remaining, start=1):
                top_pages = retrieval_by_query[sample.query_id]["top_10"][: args.top_k]
                page_ids = [hit["page_id"] for hit in top_pages]
                image_paths = [_resolve_image_path(pages_by_id[page_id].image_path) for page_id in page_ids]
                collage = _vertical_collage(image_paths, Image, ImageDraw)
                try:
                    started = perf_counter()
                    prediction = generator.answer_image(collage, sample.query)
                    elapsed_ms = (perf_counter() - started) * 1000.0
                finally:
                    collage.close()
                row = {
                    "query_id": sample.query_id,
                    "query": sample.query,
                    "answers": sample.answers,
                    "evidence_page_ids": sample.evidence_page_ids,
                    "retrieved_page_ids": page_ids,
                    "retrieval_hit": bool(set(page_ids) & set(sample.evidence_page_ids)),
                    "split": sample.split,
                    "prediction": prediction,
                    "elapsed_ms": elapsed_ms,
                }
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_file.flush()
                completed[sample.query_id] = row
                print(
                    f"[{position}/{len(remaining)}] {sample.query_id}: "
                    f"hit={row['retrieval_hit']}, prediction={prediction!r}, elapsed_ms={elapsed_ms:.1f}"
                )

    rows = [completed[sample.query_id] for sample in samples]
    metrics = _evaluate(rows)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "method": f"top{args.top_k}_collage_smolvlm",
                "retriever": "vidore/colSmol-500M",
                "generator": args.model,
                "query_count": len(rows),
                **metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"queries: {len(rows)}")
    print(f"retrieval_recall: {metrics['retrieval_recall']:.4f}")
    print(f"exact_match: {metrics['exact_match']:.4f}")
    print(f"relaxed_accuracy: {metrics['relaxed_accuracy']:.4f}")
    print(f"mean_query_ms: {metrics['mean_query_ms']:.1f}")
    print(f"metrics: {_relative(metrics_path)}")


def _vertical_collage(image_paths, image_module, draw_module):
    images = []
    try:
        for path in image_paths:
            with image_module.open(path) as source:
                images.append(source.convert("RGB"))
        target_width = max(image.width for image in images)
        gap = 28
        resized = []
        for image in images:
            if image.width == target_width:
                resized.append(image.copy())
            else:
                height = max(1, round(image.height * target_width / image.width))
                resized.append(image.resize((target_width, height)))
        canvas = image_module.new(
            "RGB", (target_width, sum(image.height for image in resized) + gap * (len(resized) - 1)), "white"
        )
        draw = draw_module.Draw(canvas)
        y = 0
        for index, image in enumerate(resized):
            canvas.paste(image, (0, y))
            y += image.height
            if index < len(resized) - 1:
                draw.line((0, y + gap // 2, target_width, y + gap // 2), fill="black", width=3)
                y += gap
            image.close()
        return canvas
    finally:
        for image in images:
            image.close()


def _load_predictions(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows.setdefault(row["query_id"], row)
    return rows


def _evaluate(rows: list[dict[str, object]]) -> dict[str, float]:
    exact = relaxed = retrieval_hits = 0
    total_ms = 0.0
    for row in rows:
        prediction = str(row["prediction"])
        answers = [str(answer) for answer in row["answers"]]
        exact += any(_normalize(prediction) == _normalize(answer) for answer in answers)
        relaxed += any(_relaxed_match(prediction, answer) for answer in answers)
        retrieval_hits += bool(row["retrieval_hit"])
        total_ms += float(row["elapsed_ms"])
    count = len(rows)
    return {
        "retrieval_recall": retrieval_hits / count,
        "exact_match": exact / count,
        "relaxed_accuracy": relaxed / count,
        "mean_query_ms": total_ms / count,
    }


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9.+-]", "", value.lower())


def _relaxed_match(prediction: str, answer: str) -> bool:
    if _normalize(prediction) == _normalize(answer):
        return True
    left, right = _number(prediction), _number(answer)
    if left is None or right is None:
        return False
    return math.isclose(left, right, abs_tol=max(abs(right) * 0.05, 1e-9))


def _number(value: str) -> float | None:
    match = re.fullmatch(r"\s*([-+]?\d+(?:\.\d+)?)\s*%?\s*", value.replace(",", ""))
    return float(match.group(1)) if match else None


def _resolve_image_path(image_path: str) -> Path:
    path = Path(image_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
