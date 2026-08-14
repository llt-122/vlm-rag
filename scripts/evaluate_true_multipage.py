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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate true multi-page evidence coverage and QA.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--retrieval-results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()

    bundle = load_bundle(args.dataset_dir)
    pages = {page.page_id: page for page in bundle.pages}
    retrieval = None
    if args.retrieval_results:
        retrieval = {
            row["query_id"]: row
            for row in json.loads(args.retrieval_results.read_text(encoding="utf-8"))
        }

    retrieval_rows = []
    for sample in bundle.samples:
        retrieved = (
            [hit["page_id"] for hit in retrieval[sample.query_id]["top_10"][: args.top_k]]
            if retrieval
            else list(sample.evidence_page_ids)
        )
        evidence = set(sample.evidence_page_ids)
        selected = set(retrieved)
        retrieval_rows.append(
            {
                "query_id": sample.query_id,
                "query": sample.query,
                "evidence_page_ids": sample.evidence_page_ids,
                "retrieved_page_ids": retrieved,
                "any_evidence_hit": bool(evidence & selected),
                "all_evidence_hit": evidence <= selected,
                "evidence_coverage": len(evidence & selected) / len(evidence),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _retrieval_metrics(retrieval_rows)
    metrics.update({"query_count": len(bundle.samples), "top_k": args.top_k})

    predictions_path = args.output_dir / "predictions.jsonl"
    predictions = _load_completed(predictions_path)
    if not args.skip_generation:
        generator = _MultiImageGenerator(args.model, args.max_new_tokens)
        completed_ids = {row["query_id"] for row in predictions}
        mode = "a" if predictions_path.exists() else "w"
        with predictions_path.open(mode, encoding="utf-8") as output_file:
            for number, (sample, row) in enumerate(zip(bundle.samples, retrieval_rows), start=1):
                if sample.query_id in completed_ids:
                    continue
                image_paths = [
                    _resolve_path(pages[page_id].image_path)
                    for page_id in row["retrieved_page_ids"]
                ]
                started = perf_counter()
                prediction = generator.answer(image_paths, sample.query)
                elapsed_ms = (perf_counter() - started) * 1000
                correct = _relaxed_match_any(prediction, sample.answers)
                result = {
                    **row,
                    "answers": sample.answers,
                    "prediction": prediction,
                    "relaxed_correct": correct,
                    "generation_ms": elapsed_ms,
                }
                predictions.append(result)
                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_file.flush()
                print(f"[{number}/{len(bundle.samples)}] {sample.query_id}: {prediction!r}")
        metrics.update(
            {
                "relaxed_accuracy": sum(row["relaxed_correct"] for row in predictions)
                / len(predictions),
                "accuracy_given_all_evidence": _conditional_accuracy(predictions),
                "mean_generation_ms": sum(row["generation_ms"] for row in predictions)
                / len(predictions),
            }
        )

    (args.output_dir / "retrieval_analysis.json").write_text(
        json.dumps(retrieval_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


class _MultiImageGenerator:
    def __init__(self, model_name: str, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=self.device,
            local_files_only=True,
            attn_implementation="eager",
        ).eval()

    def answer(self, image_paths: list[Path], query: str) -> str:
        from PIL import Image

        images = []
        try:
            for path in image_paths:
                with Image.open(path) as source:
                    images.append(source.convert("RGB"))
            content = [{"type": "image"} for _ in images]
            content.append(
                {
                    "type": "text",
                    "text": (
                        "The images are consecutive pages from one financial report. "
                        "Use all relevant pages, perform the required arithmetic, and return only "
                        f"the final number without units or explanation. Question: {query}"
                    ),
                }
            )
            messages = [{"role": "user", "content": content}]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self.device)
            prompt_length = inputs["input_ids"].shape[-1]
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            return self.processor.decode(
                generated[0, prompt_length:], skip_special_tokens=True
            ).strip()
        finally:
            for image in images:
                image.close()


def _retrieval_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    count = len(rows)
    return {
        "any_evidence_recall": sum(row["any_evidence_hit"] for row in rows) / count,
        "all_evidence_recall": sum(row["all_evidence_hit"] for row in rows) / count,
        "mean_evidence_coverage": sum(float(row["evidence_coverage"]) for row in rows) / count,
    }


def _conditional_accuracy(rows: list[dict[str, object]]) -> float | None:
    eligible = [row for row in rows if row["all_evidence_hit"]]
    if not eligible:
        return None
    return sum(row["relaxed_correct"] for row in eligible) / len(eligible)


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _relaxed_match_any(prediction: str, answers: list[str]) -> bool:
    normalized = _normalize(prediction)
    if any(normalized == _normalize(answer) for answer in answers):
        return True
    predicted = _parse_number(prediction)
    if predicted is None:
        return False
    return any(
        target is not None and math.isclose(predicted, target, rel_tol=0, abs_tol=0.05 * max(abs(target), 1))
        for target in (_parse_number(answer) for answer in answers)
    )


def _parse_number(value: str) -> float | None:
    cleaned = value.strip().replace(",", "").replace("%", "")
    match = re.fullmatch(r"[^0-9+\-.]*([+-]?(?:\d+(?:\.\d*)?|\.\d+))[^0-9]*", cleaned)
    return float(match.group(1)) if match else None


def _normalize(value: str) -> str:
    return " ".join(value.lower().translate(str.maketrans("", "", string.punctuation)).split())


def _load_completed(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
