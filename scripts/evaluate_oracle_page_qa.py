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
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    parser = argparse.ArgumentParser(
        description="Evaluate single-page visual QA using the labeled evidence page."
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=PROJECT_ROOT / "data" / "real" / "chartqa"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "oracle_page_qa"
    )
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all samples")
    args = parser.parse_args()

    bundle = load_bundle(args.dataset_dir)
    samples = bundle.samples[: args.limit or None]
    pages_by_id = {page.page_id: page for page in bundle.pages}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    completed = _load_predictions(predictions_path)
    remaining = [sample for sample in samples if sample.query_id not in completed]

    if remaining:
        generator = SmolVLMGenerator(SmolVLMConfig(model_name=args.model))
        with predictions_path.open("a", encoding="utf-8") as output_file:
            for position, sample in enumerate(remaining, start=1):
                page = pages_by_id[sample.evidence_page_ids[0]]
                image_path = _resolve_image_path(page.image_path)
                started = perf_counter()
                prediction = generator.answer(image_path, sample.query)
                elapsed_ms = (perf_counter() - started) * 1000.0
                row = {
                    "query_id": sample.query_id,
                    "query": sample.query,
                    "answers": sample.answers,
                    "evidence_page_id": page.page_id,
                    "split": sample.split,
                    "prediction": prediction,
                    "elapsed_ms": elapsed_ms,
                }
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_file.flush()
                completed[sample.query_id] = row
                print(
                    f"[{position}/{len(remaining)}] {sample.query_id}: "
                    f"prediction={prediction!r}, elapsed_ms={elapsed_ms:.1f}"
                )

    rows = [completed[sample.query_id] for sample in samples]
    metrics = _evaluate(rows)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "method": "oracle_single_page_smolvlm",
                "model_name": args.model,
                "query_count": len(rows),
                **metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"queries: {len(rows)}")
    print(f"exact_match: {metrics['exact_match']:.4f}")
    print(f"relaxed_accuracy: {metrics['relaxed_accuracy']:.4f}")
    print(f"mean_query_ms: {metrics['mean_query_ms']:.1f}")
    print(f"metrics: {_relative(metrics_path)}")


def _load_predictions(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.setdefault(row["query_id"], row)
    return rows


def _evaluate(rows: list[dict[str, object]]) -> dict[str, float]:
    exact = 0
    relaxed = 0
    elapsed = 0.0
    for row in rows:
        prediction = str(row["prediction"])
        answers = [str(answer) for answer in row["answers"]]
        exact += any(_normalize(prediction) == _normalize(answer) for answer in answers)
        relaxed += any(_relaxed_match(prediction, answer) for answer in answers)
        elapsed += float(row["elapsed_ms"])
    count = len(rows)
    return {
        "exact_match": exact / count,
        "relaxed_accuracy": relaxed / count,
        "mean_query_ms": elapsed / count,
    }


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9.+-]", "", value.lower())


def _relaxed_match(prediction: str, answer: str) -> bool:
    if _normalize(prediction) == _normalize(answer):
        return True
    predicted_number = _number(prediction)
    answer_number = _number(answer)
    if predicted_number is None or answer_number is None:
        return False
    tolerance = max(abs(answer_number) * 0.05, 1e-9)
    return math.isclose(predicted_number, answer_number, abs_tol=tolerance)


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
