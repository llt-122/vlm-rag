from __future__ import annotations

import argparse
import json
import math
import re
import string
import sys
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlm_rag.dataset_schema import load_bundle


PROMPT_TEMPLATE = (
    "Answer the chart question using only this page. "
    "Return only the short answer, with no explanation. Question: {query}"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Top-3 pages separately and fuse page-level VLM answers."
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
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "smolvlm_multipage"
    )
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--retrieval-temperature", type=float, default=1.0)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    bundle = load_bundle(args.dataset_dir)
    pages = {page.page_id: page for page in bundle.pages}
    retrieval = {
        row["query_id"]: row
        for row in json.loads(args.retrieval_results.read_text(encoding="utf-8"))
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = args.output_dir / "page_candidates.jsonl"
    completed = _load_candidates(candidates_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, dtype=dtype, device_map=str(device)
    ).eval()

    total = len(bundle.samples) * 3
    done = sum(
        (sample.query_id, hit["page_id"]) in completed
        for sample in bundle.samples
        for hit in retrieval[sample.query_id]["top_10"][:3]
    )
    print(f"Page candidates: completed={done}, remaining={total - done}")
    with candidates_path.open("a", encoding="utf-8") as output_file:
        for sample in bundle.samples:
            for hit in retrieval[sample.query_id]["top_10"][:3]:
                page_id = hit["page_id"]
                key = (sample.query_id, page_id)
                if key in completed:
                    continue
                path = Path(pages[page_id].image_path)
                path = path if path.is_absolute() else PROJECT_ROOT / path
                with Image.open(path) as source:
                    image = source.convert("RGB")
                try:
                    started = perf_counter()
                    prediction, mean_log_probability = _generate(
                        sample.query,
                        image,
                        model,
                        processor,
                        torch,
                        args.max_new_tokens,
                    )
                    elapsed_ms = (perf_counter() - started) * 1000.0
                finally:
                    image.close()
                row = {
                    "query_id": sample.query_id,
                    "page_id": page_id,
                    "rank": int(hit["rank"]),
                    "retrieval_score": float(hit["score"]),
                    "prediction": prediction,
                    "mean_log_probability": mean_log_probability,
                    "elapsed_ms": elapsed_ms,
                }
                completed[key] = row
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_file.flush()
                done += 1
                print(f"[{done}/{total}] {sample.query_id} page={hit['rank']}: {prediction!r}")

    fused_rows = []
    exact_count = 0
    relaxed_count = 0
    candidate_oracle_count = 0
    evidence_recalled = 0
    for sample in bundle.samples:
        hits = retrieval[sample.query_id]["top_10"][:3]
        candidates = [completed[(sample.query_id, hit["page_id"])] for hit in hits]
        fused = _fuse(candidates, args.retrieval_temperature)
        exact = _exact_match_any(fused["prediction"], sample.answers)
        relaxed = _relaxed_match_any(fused["prediction"], sample.answers)
        candidate_oracle = any(
            _relaxed_match_any(str(candidate["prediction"]), sample.answers)
            for candidate in candidates
        )
        evidence = set(sample.evidence_page_ids)
        recalled = any(candidate["page_id"] in evidence for candidate in candidates)
        exact_count += exact
        relaxed_count += relaxed
        candidate_oracle_count += candidate_oracle
        evidence_recalled += recalled
        fused_rows.append(
            {
                "query_id": sample.query_id,
                "query": sample.query,
                "answers": sample.answers,
                "prediction": fused["prediction"],
                "winning_vote_score": fused["vote_score"],
                "exact_match": exact,
                "relaxed_correct": relaxed,
                "candidate_oracle_correct": candidate_oracle,
                "evidence_recalled_at_3": recalled,
                "candidates": candidates,
            }
        )

    count = len(bundle.samples)
    metrics = {
        "method": "top3_separate_generation_weighted_fusion",
        "model_name": args.model,
        "query_count": count,
        "em": exact_count / count,
        "relaxed_accuracy": relaxed_count / count,
        "candidate_oracle_relaxed_accuracy": candidate_oracle_count / count,
        "evidence_recall@3": evidence_recalled / count,
        "mean_candidate_ms": sum(
            float(row["elapsed_ms"])
            for sample in bundle.samples
            for row in [
                completed[(sample.query_id, hit["page_id"])]
                for hit in retrieval[sample.query_id]["top_10"][:3]
            ]
        )
        / total,
    }
    results_path = args.output_dir / "fused_results.json"
    metrics_path = args.output_dir / "metrics.json"
    results_path.write_text(json.dumps(fused_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"candidates: {_relative(candidates_path)}")
    print(f"results: {_relative(results_path)}")
    print(f"metrics: {_relative(metrics_path)}")


def _generate(
    query: str,
    image: object,
    model: object,
    processor: object,
    torch_module: object,
    max_new_tokens: int,
) -> tuple[str, float]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT_TEMPLATE.format(query=query)},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)
    with torch_module.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
        transition_scores = model.compute_transition_scores(
            generated.sequences,
            generated.scores,
            normalize_logits=True,
        )
    new_tokens = generated.sequences[0, inputs["input_ids"].shape[-1] :]
    prediction = processor.decode(new_tokens, skip_special_tokens=True).strip()
    mean_log_probability = float(transition_scores[0].float().mean().cpu())
    return prediction, mean_log_probability


def _fuse(candidates: list[dict[str, object]], temperature: float) -> dict[str, object]:
    max_score = max(float(candidate["retrieval_score"]) for candidate in candidates)
    votes: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        answer_key = _normalize(str(candidate["prediction"]))
        retrieval_weight = math.exp(
            (float(candidate["retrieval_score"]) - max_score) / temperature
        )
        generation_weight = math.exp(float(candidate["mean_log_probability"]))
        weight = retrieval_weight * generation_weight
        vote = votes.setdefault(
            answer_key,
            {"prediction": candidate["prediction"], "vote_score": 0.0},
        )
        vote["vote_score"] = float(vote["vote_score"]) + weight
    return max(votes.values(), key=lambda item: float(item["vote_score"]))


def _exact_match_any(prediction: str, answers: list[str]) -> bool:
    normalized = _normalize(prediction)
    return any(normalized == _normalize(answer) for answer in answers)


def _relaxed_match_any(prediction: str, answers: list[str]) -> bool:
    if _exact_match_any(prediction, answers):
        return True
    prediction_number = _parse_number(prediction)
    if prediction_number is None:
        return False
    for answer in answers:
        target = _parse_number(answer)
        if target is not None and math.isclose(
            prediction_number, target, rel_tol=0.0, abs_tol=0.05 * abs(target)
        ):
            return True
    return False


def _parse_number(value: str) -> float | None:
    cleaned = value.strip().replace(",", "").replace("%", "")
    match = re.fullmatch(r"[^0-9+\-.]*([+-]?(?:\d+(?:\.\d*)?|\.\d+))[^0-9]*", cleaned)
    return float(match.group(1)) if match else None


def _normalize(value: str) -> str:
    return " ".join(value.lower().translate(str.maketrans("", "", string.punctuation)).split())


def _load_candidates(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    rows: dict[tuple[str, str], dict[str, object]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[(row["query_id"], row["page_id"])] = row
    return rows


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
