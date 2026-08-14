from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlm_rag.dataset_schema import DocumentQASample, load_bundle
from vlm_rag.smolvlm_generator import SmolVLMConfig, SmolVLMGenerator


@dataclass(frozen=True)
class RetrieverSpec:
    name: str
    path: Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one SmolVLM generator with oracle pages and multiple retrievers. "
            "Page-level answers are cached and reused across Top-1 and Top-K methods."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--retriever",
        action="append",
        required=True,
        metavar="NAME=RESULTS_JSON",
        help="Repeat for each retrieval method.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--retrieval-temperature", type=float, default=1.0)
    parser.add_argument(
        "--collage-retriever",
        action="append",
        default=[],
        help="Retriever name for which a Top-K collage baseline is evaluated.",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.retrieval_temperature <= 0:
        parser.error("--retrieval-temperature must be positive")

    retrievers = [_parse_retriever(value) for value in args.retriever]
    names = [spec.name for spec in retrievers]
    if len(set(names)) != len(names):
        parser.error("retriever names must be unique")
    unknown_collage = sorted(set(args.collage_retriever) - set(names))
    if unknown_collage:
        parser.error(f"unknown collage retrievers: {unknown_collage}")

    bundle = load_bundle(args.dataset_dir)
    samples = [sample for sample in bundle.samples if sample.split == args.split]
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError(f"no samples found for split={args.split!r}")
    pages_by_id = {page.page_id: page for page in bundle.pages}
    retrieval_by_name = {
        spec.name: _load_retrieval(spec.path, samples, args.top_k) for spec in retrievers
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "split": args.split,
        "model": args.model,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "retrieval_temperature": args.retrieval_temperature,
        "retrievers": {spec.name: str(spec.path.resolve()) for spec in retrievers},
        "collage_retrievers": sorted(args.collage_retriever),
    }
    _check_or_write_config(args.output_dir / "run_config.json", config)

    page_cache_path = args.output_dir / "page_answers.jsonl"
    collage_cache_path = args.output_dir / "collage_answers.jsonl"
    page_answers = _load_jsonl_cache(page_cache_path, ("query_id", "page_id"))
    collage_answers = _load_jsonl_cache(
        collage_cache_path, ("query_id", "retriever")
    )

    required_pages: list[tuple[DocumentQASample, str]] = []
    seen_page_keys: set[tuple[str, str]] = set()
    for sample in samples:
        page_ids = list(sample.evidence_page_ids[:1])
        for spec in retrievers:
            hits = retrieval_by_name[spec.name][sample.query_id]["top_10"][: args.top_k]
            page_ids.extend(str(hit["page_id"]) for hit in hits)
        for page_id in page_ids:
            key = (sample.query_id, page_id)
            if key not in seen_page_keys:
                required_pages.append((sample, page_id))
                seen_page_keys.add(key)

    required_collages = [
        (sample, retriever_name)
        for sample in samples
        for retriever_name in args.collage_retriever
    ]
    remaining_pages = [
        pair for pair in required_pages if (pair[0].query_id, pair[1]) not in page_answers
    ]
    remaining_collages = [
        pair
        for pair in required_collages
        if (pair[0].query_id, pair[1]) not in collage_answers
    ]
    print(
        f"split={args.split}; queries={len(samples)}; "
        f"page answers: completed={len(required_pages) - len(remaining_pages)}, "
        f"remaining={len(remaining_pages)}; "
        f"collages: completed={len(required_collages) - len(remaining_collages)}, "
        f"remaining={len(remaining_collages)}"
    )

    generator = None
    started_all = perf_counter()
    if remaining_pages or remaining_collages:
        generator = SmolVLMGenerator(
            SmolVLMConfig(
                model_name=args.model,
                max_new_tokens=args.max_new_tokens,
            )
        )

    if remaining_pages:
        with page_cache_path.open("a", encoding="utf-8") as output_file:
            for position, (sample, page_id) in enumerate(remaining_pages, start=1):
                image_path = _resolve_image_path(pages_by_id[page_id].image_path)
                started = perf_counter()
                answer = generator.answer(image_path, sample.query)
                elapsed_ms = (perf_counter() - started) * 1000.0
                row = {
                    "query_id": sample.query_id,
                    "page_id": page_id,
                    "answer": answer,
                    "elapsed_ms": elapsed_ms,
                }
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_file.flush()
                page_answers[(sample.query_id, page_id)] = row
                print(
                    f"[page {position}/{len(remaining_pages)}] "
                    f"{sample.query_id}/{page_id}: {answer!r}, {elapsed_ms:.1f}ms"
                )

    if remaining_collages:
        from PIL import Image, ImageDraw

        with collage_cache_path.open("a", encoding="utf-8") as output_file:
            for position, (sample, retriever_name) in enumerate(remaining_collages, start=1):
                hits = retrieval_by_name[retriever_name][sample.query_id]["top_10"][
                    : args.top_k
                ]
                page_ids = [str(hit["page_id"]) for hit in hits]
                image_paths = [
                    _resolve_image_path(pages_by_id[page_id].image_path)
                    for page_id in page_ids
                ]
                collage = _vertical_collage(image_paths, Image, ImageDraw)
                try:
                    started = perf_counter()
                    answer = generator.answer_image(collage, sample.query)
                    elapsed_ms = (perf_counter() - started) * 1000.0
                finally:
                    collage.close()
                row = {
                    "query_id": sample.query_id,
                    "retriever": retriever_name,
                    "page_ids": page_ids,
                    "answer": answer,
                    "elapsed_ms": elapsed_ms,
                }
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_file.flush()
                collage_answers[(sample.query_id, retriever_name)] = row
                print(
                    f"[collage {position}/{len(remaining_collages)}] "
                    f"{sample.query_id}/{retriever_name}: {answer!r}, {elapsed_ms:.1f}ms"
                )

    predictions = _build_predictions(
        samples=samples,
        retrievers=retrievers,
        retrieval_by_name=retrieval_by_name,
        page_answers=page_answers,
        collage_answers=collage_answers,
        collage_retrievers=set(args.collage_retriever),
        top_k=args.top_k,
        temperature=args.retrieval_temperature,
    )
    metrics = _aggregate_metrics(predictions)
    runtime_seconds = perf_counter() - started_all
    peak_gpu_mb = 0.0
    if generator is not None:
        torch = generator._torch
        peak_gpu_mb = torch.cuda.max_memory_allocated() / 1024**2

    (args.output_dir / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics_payload = {
        "split": args.split,
        "query_count": len(samples),
        "model": args.model,
        "top_k": args.top_k,
        "runtime_seconds_this_invocation": runtime_seconds,
        "peak_gpu_mb_this_invocation": peak_gpu_mb,
        "methods": metrics,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_comparison(args.output_dir, metrics)

    print("\nStage 4 comparison")
    for row in metrics:
        print(
            f"{row['method']}: retrieval={row['retrieval_recall']:.4f}, "
            f"EM={row['exact_match']:.4f}, "
            f"relaxed={row['relaxed_accuracy']:.4f}, "
            f"mean_ms={row['mean_generation_ms']:.1f}"
        )
    print(f"results: {_relative(args.output_dir)}")


def _parse_retriever(value: str) -> RetrieverSpec:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(
            f"invalid retriever {value!r}; expected NAME=RESULTS_JSON"
        )
    return RetrieverSpec(name=name.strip(), path=Path(raw_path.strip()))


def _load_retrieval(
    path: Path, samples: list[DocumentQASample], top_k: int
) -> dict[str, dict[str, object]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_query = {str(row["query_id"]): row for row in rows}
    missing = [sample.query_id for sample in samples if sample.query_id not in by_query]
    if missing:
        raise ValueError(f"{path}: missing {len(missing)} query IDs, first={missing[:3]}")
    too_short = [
        sample.query_id
        for sample in samples
        if len(by_query[sample.query_id].get("top_10", [])) < top_k
    ]
    if too_short:
        raise ValueError(f"{path}: fewer than Top-{top_k} hits for {too_short[:3]}")
    return by_query


def _check_or_write_config(path: Path, config: dict[str, object]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError(
                f"existing cache configuration differs at {path}; use a new output directory"
            )
        return
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_jsonl_cache(
    path: Path, key_fields: tuple[str, ...]
) -> dict[tuple[str, ...], dict[str, object]]:
    rows: dict[tuple[str, ...], dict[str, object]] = {}
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = tuple(str(row[field]) for field in key_fields)
        if key in rows and rows[key] != row:
            raise ValueError(f"{path}:{line_number}: conflicting cache key {key}")
        rows[key] = row
    return rows


def _build_predictions(
    *,
    samples: list[DocumentQASample],
    retrievers: list[RetrieverSpec],
    retrieval_by_name: dict[str, dict[str, dict[str, object]]],
    page_answers: dict[tuple[str, str], dict[str, object]],
    collage_answers: dict[tuple[str, str], dict[str, object]],
    collage_retrievers: set[str],
    top_k: int,
    temperature: float,
) -> list[dict[str, object]]:
    predictions: list[dict[str, object]] = []
    for sample in samples:
        oracle_page = sample.evidence_page_ids[0]
        oracle = page_answers[(sample.query_id, oracle_page)]
        predictions.append(
            _prediction_row(
                sample=sample,
                method="oracle_page",
                retriever="oracle",
                page_ids=[oracle_page],
                prediction=str(oracle["answer"]),
                retrieval_hit=True,
                generation_ms=float(oracle["elapsed_ms"]),
            )
        )
        for spec in retrievers:
            hits = retrieval_by_name[spec.name][sample.query_id]["top_10"][:top_k]
            candidate_rows = []
            for hit in hits:
                page_id = str(hit["page_id"])
                cached = page_answers[(sample.query_id, page_id)]
                candidate_rows.append(
                    {
                        "page_id": page_id,
                        "rank": int(hit["rank"]),
                        "retrieval_score": float(hit["score"]),
                        "answer": str(cached["answer"]),
                        "elapsed_ms": float(cached["elapsed_ms"]),
                    }
                )
            evidence = set(sample.evidence_page_ids)
            top1 = candidate_rows[0]
            predictions.append(
                _prediction_row(
                    sample=sample,
                    method=f"{spec.name}_top1",
                    retriever=spec.name,
                    page_ids=[str(top1["page_id"])],
                    prediction=str(top1["answer"]),
                    retrieval_hit=str(top1["page_id"]) in evidence,
                    generation_ms=float(top1["elapsed_ms"]),
                    page_candidates=[top1],
                )
            )
            fused_answer, fusion = _weighted_vote(candidate_rows, temperature)
            predictions.append(
                _prediction_row(
                    sample=sample,
                    method=f"{spec.name}_top{top_k}_sequential",
                    retriever=spec.name,
                    page_ids=[str(row["page_id"]) for row in candidate_rows],
                    prediction=fused_answer,
                    retrieval_hit=bool(
                        {str(row["page_id"]) for row in candidate_rows} & evidence
                    ),
                    generation_ms=sum(float(row["elapsed_ms"]) for row in candidate_rows),
                    page_candidates=candidate_rows,
                    fusion=fusion,
                )
            )
            if spec.name in collage_retrievers:
                collage = collage_answers[(sample.query_id, spec.name)]
                page_ids = [str(page_id) for page_id in collage["page_ids"]]
                predictions.append(
                    _prediction_row(
                        sample=sample,
                        method=f"{spec.name}_top{top_k}_collage",
                        retriever=spec.name,
                        page_ids=page_ids,
                        prediction=str(collage["answer"]),
                        retrieval_hit=bool(set(page_ids) & evidence),
                        generation_ms=float(collage["elapsed_ms"]),
                    )
                )
    return predictions


def _prediction_row(
    *,
    sample: DocumentQASample,
    method: str,
    retriever: str,
    page_ids: list[str],
    prediction: str,
    retrieval_hit: bool,
    generation_ms: float,
    page_candidates: list[dict[str, object]] | None = None,
    fusion: dict[str, float] | None = None,
) -> dict[str, object]:
    exact = _exact_match(prediction, sample.answers)
    relaxed = _relaxed_match_any(prediction, sample.answers)
    row: dict[str, object] = {
        "query_id": sample.query_id,
        "query": sample.query,
        "answers": sample.answers,
        "evidence_page_ids": sample.evidence_page_ids,
        "split": sample.split,
        "method": method,
        "retriever": retriever,
        "page_ids": page_ids,
        "retrieval_hit": retrieval_hit,
        "prediction": prediction,
        "exact_match": exact,
        "relaxed_correct": relaxed,
        "generation_ms": generation_ms,
        "error_type": (
            "correct"
            if relaxed
            else "generation_error"
            if retrieval_hit
            else "retrieval_miss"
        ),
    }
    if page_candidates is not None:
        row["page_candidates"] = page_candidates
    if fusion is not None:
        row["fusion"] = fusion
    return row


def _weighted_vote(
    rows: list[dict[str, object]], temperature: float
) -> tuple[str, dict[str, float]]:
    scores = [float(row["retrieval_score"]) / temperature for row in rows]
    maximum = max(scores)
    unnormalized = [math.exp(score - maximum) for score in scores]
    denominator = sum(unnormalized)
    weights = [weight / denominator for weight in unnormalized]
    grouped: dict[str, float] = defaultdict(float)
    representatives: dict[str, tuple[float, str]] = {}
    for row, weight in zip(rows, weights, strict=True):
        answer = str(row["answer"])
        key = _normalize(answer)
        grouped[key] += weight
        score = float(row["retrieval_score"])
        if key not in representatives or score > representatives[key][0]:
            representatives[key] = (score, answer)
    winner = max(grouped, key=lambda key: (grouped[key], representatives[key][0]))
    return representatives[winner][1], dict(grouped)


def _aggregate_metrics(predictions: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in predictions:
        grouped.setdefault(str(row["method"]), []).append(row)
    metrics = []
    for method, rows in grouped.items():
        count = len(rows)
        hits = sum(bool(row["retrieval_hit"]) for row in rows)
        correct_on_hits = sum(
            bool(row["relaxed_correct"]) and bool(row["retrieval_hit"]) for row in rows
        )
        metrics.append(
            {
                "method": method,
                "query_count": count,
                "retrieval_recall": hits / count,
                "exact_match": sum(bool(row["exact_match"]) for row in rows) / count,
                "relaxed_accuracy": sum(bool(row["relaxed_correct"]) for row in rows)
                / count,
                "accuracy_given_retrieval_hit": correct_on_hits / hits if hits else 0.0,
                "mean_generation_ms": sum(float(row["generation_ms"]) for row in rows)
                / count,
                "retrieval_misses": sum(row["error_type"] == "retrieval_miss" for row in rows),
                "generation_errors": sum(
                    row["error_type"] == "generation_error" for row in rows
                ),
            }
        )
    return metrics


def _write_comparison(output_dir: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        "method",
        "query_count",
        "retrieval_recall",
        "exact_match",
        "relaxed_accuracy",
        "accuracy_given_retrieval_hit",
        "mean_generation_ms",
        "retrieval_misses",
        "generation_errors",
    ]
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Stage 4 Generation Comparison",
        "",
        "| Method | Retrieval | EM | Relaxed Acc. | Acc. given hit | Mean ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['retrieval_recall']:.4f} | "
            f"{row['exact_match']:.4f} | {row['relaxed_accuracy']:.4f} | "
            f"{row['accuracy_given_retrieval_hit']:.4f} | "
            f"{row['mean_generation_ms']:.1f} |"
        )
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _vertical_collage(image_paths, image_module, draw_module):
    images = []
    resized = []
    try:
        for path in image_paths:
            with image_module.open(path) as source:
                images.append(source.convert("RGB"))
        target_width = 512
        for image in images:
            target_height = max(1, round(image.height * target_width / image.width))
            resized.append(
                image.resize(
                    (target_width, target_height), image_module.Resampling.LANCZOS
                )
            )
        gap = 16
        canvas = image_module.new(
            "RGB",
            (target_width, sum(image.height for image in resized) + gap * (len(resized) - 1)),
            "white",
        )
        draw = draw_module.Draw(canvas)
        offset = 0
        for index, image in enumerate(resized):
            canvas.paste(image, (0, offset))
            offset += image.height
            if index < len(resized) - 1:
                draw.line((0, offset + gap // 2, target_width, offset + gap // 2), fill="black", width=2)
                offset += gap
        return canvas
    finally:
        for image in images + resized:
            image.close()


def _exact_match(prediction: str, answers: list[str]) -> bool:
    normalized = _normalize(prediction)
    return any(normalized == _normalize(answer) for answer in answers)


def _relaxed_match_any(prediction: str, answers: list[str]) -> bool:
    if _exact_match(prediction, answers):
        return True
    predicted_number = _number(prediction)
    if predicted_number is None:
        return False
    for answer in answers:
        target = _number(answer)
        if target is not None and math.isclose(
            predicted_number, target, rel_tol=0.0, abs_tol=max(abs(target) * 0.05, 1e-9)
        ):
            return True
    return False


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9.+-]", "", value.lower())


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
