from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
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
    parser = argparse.ArgumentParser(
        description="Evaluate Top-K pages one by one, then fuse answers with retrieval scores."
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=PROJECT_ROOT / "data" / "real" / "chartqa"
    )
    parser.add_argument(
        "--retrieval-results",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "colsmol_chartqa" / "retrieval_results.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "top3_sequential_qa"
    )
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    bundle = load_bundle(args.dataset_dir)
    samples = bundle.samples[: args.limit or None]
    pages_by_id = {page.page_id: page for page in bundle.pages}
    retrieval_rows = json.loads(args.retrieval_results.read_text(encoding="utf-8"))
    retrieval_by_query = {row["query_id"]: row for row in retrieval_rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = args.output_dir / "page_candidates.jsonl"
    candidates = _load_candidates(candidates_path)
    required = [
        (sample, hit)
        for sample in samples
        for hit in retrieval_by_query[sample.query_id]["top_10"][: args.top_k]
    ]
    remaining = [pair for pair in required if (pair[0].query_id, pair[1]["page_id"]) not in candidates]

    if remaining:
        generator = SmolVLMGenerator(SmolVLMConfig(model_name=args.model))
        with candidates_path.open("a", encoding="utf-8") as output_file:
            for position, (sample, hit) in enumerate(remaining, start=1):
                page_id = hit["page_id"]
                image_path = _resolve_image_path(pages_by_id[page_id].image_path)
                started = perf_counter()
                answer = generator.answer(image_path, sample.query)
                elapsed_ms = (perf_counter() - started) * 1000.0
                row = {
                    "query_id": sample.query_id,
                    "page_id": page_id,
                    "retrieval_rank": int(hit["rank"]),
                    "retrieval_score": float(hit["score"]),
                    "answer": answer,
                    "elapsed_ms": elapsed_ms,
                }
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_file.flush()
                candidates[(sample.query_id, page_id)] = row
                print(
                    f"[{position}/{len(remaining)}] {sample.query_id}/{page_id}: "
                    f"answer={answer!r}, elapsed_ms={elapsed_ms:.1f}"
                )

    predictions = []
    for sample in samples:
        hits = retrieval_by_query[sample.query_id]["top_10"][: args.top_k]
        rows = [candidates[(sample.query_id, hit["page_id"])] for hit in hits]
        prediction, fusion = _weighted_vote(rows)
        predictions.append(
            {
                "query_id": sample.query_id,
                "query": sample.query,
                "answers": sample.answers,
                "evidence_page_ids": sample.evidence_page_ids,
                "split": sample.split,
                "retrieval_hit": bool(
                    {row["page_id"] for row in rows} & set(sample.evidence_page_ids)
                ),
                "prediction": prediction,
                "fusion": fusion,
                "page_candidates": rows,
            }
        )

    metrics = _evaluate(predictions)
    predictions_path = args.output_dir / "predictions.json"
    metrics_path = args.output_dir / "metrics.json"
    predictions_path.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics_path.write_text(
        json.dumps(
            {
                "method": f"top{args.top_k}_sequential_weighted_vote",
                "retriever": "vidore/colSmol-500M",
                "generator": args.model,
                "query_count": len(predictions),
                **metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"queries: {len(predictions)}")
    print(f"retrieval_recall: {metrics['retrieval_recall']:.4f}")
    print(f"exact_match: {metrics['exact_match']:.4f}")
    print(f"relaxed_accuracy: {metrics['relaxed_accuracy']:.4f}")
    print(f"mean_total_generation_ms: {metrics['mean_total_generation_ms']:.1f}")
    print(f"metrics: {_relative(metrics_path)}")


def _weighted_vote(rows: list[dict[str, object]]) -> tuple[str, dict[str, float]]:
    scores = [float(row["retrieval_score"]) for row in rows]
    maximum = max(scores)
    raw_weights = [math.exp(score - maximum) for score in scores]
    total = sum(raw_weights)
    weights = [weight / total for weight in raw_weights]
    grouped: dict[str, float] = defaultdict(float)
    representatives: dict[str, tuple[float, str]] = {}
    for row, weight in zip(rows, weights, strict=True):
        answer = str(row["answer"])
        key = _normalize(answer)
        grouped[key] += weight
        retrieval_score = float(row["retrieval_score"])
        if key not in representatives or retrieval_score > representatives[key][0]:
            representatives[key] = (retrieval_score, answer)
    winner = max(grouped, key=lambda key: (grouped[key], representatives[key][0]))
    return representatives[winner][1], dict(grouped)


def _load_candidates(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    if not path.exists():
        return {}
    rows: dict[tuple[str, str], dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows.setdefault((row["query_id"], row["page_id"]), row)
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
        total_ms += sum(float(candidate["elapsed_ms"]) for candidate in row["page_candidates"])
    count = len(rows)
    return {
        "retrieval_recall": retrieval_hits / count,
        "exact_match": exact / count,
        "relaxed_accuracy": relaxed / count,
        "mean_total_generation_ms": total_ms / count,
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
