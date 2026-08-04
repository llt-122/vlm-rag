from __future__ import annotations

import argparse
import json
import math
import re
import string
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlm_rag.dataset_schema import DocumentQASample, load_bundle


MODES = ("oracle", "retrieved_top1", "stitched_top3")
PROMPT_TEMPLATE = (
    "Answer the chart question using only the supplied image. "
    "Return only the short answer, with no explanation. Question: {query}"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate oracle-page, retrieved-page and stitched-page visual QA baselines."
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
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "smolvlm_qa"
    )
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all samples")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    bundle = load_bundle(args.dataset_dir)
    samples = bundle.samples[: args.limit or None]
    pages = {page.page_id: page for page in bundle.pages}
    retrieval = {
        row["query_id"]: row
        for row in json.loads(args.retrieval_results.read_text(encoding="utf-8"))
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    completed = _load_completed(predictions_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, dtype=dtype, device_map=str(device)
    ).eval()

    total = len(samples) * len(MODES)
    done = sum((sample.query_id, mode) in completed for sample in samples for mode in MODES)
    print(f"QA predictions: completed={done}, remaining={total - done}")
    with predictions_path.open("a", encoding="utf-8") as output_file:
        for sample in samples:
            ranked_page_ids = [hit["page_id"] for hit in retrieval[sample.query_id]["top_10"]]
            mode_pages = {
                "oracle": sample.evidence_page_ids[:1],
                "retrieved_top1": ranked_page_ids[:1],
                "stitched_top3": ranked_page_ids[:3],
            }
            for mode in MODES:
                key = (sample.query_id, mode)
                if key in completed:
                    continue
                image = _prepare_image(mode_pages[mode], pages, Image, stitch=mode == "stitched_top3")
                try:
                    started = perf_counter()
                    prediction = _generate(
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
                    "query": sample.query,
                    "mode": mode,
                    "page_ids": mode_pages[mode],
                    "prediction": prediction,
                    "answers": sample.answers,
                    "exact_match": exact_match_any(prediction, sample.answers),
                    "relaxed_correct": relaxed_match_any(prediction, sample.answers),
                    "elapsed_ms": elapsed_ms,
                }
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_file.flush()
                completed[key] = row
                done += 1
                print(f"[{done}/{total}] {sample.query_id} {mode}: {prediction!r}")

    selected_rows = [completed[(sample.query_id, mode)] for sample in samples for mode in MODES]
    metrics = _aggregate(selected_rows)
    metrics_path = args.output_dir / "metrics.json"
    metrics_payload = {
        "model_name": args.model,
        "query_count": len(samples),
        "metrics": metrics,
        "peak_gpu_mb": (
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
        ),
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for mode in MODES:
        values = metrics[mode]
        print(
            f"{mode}: EM={values['em']:.4f}, "
            f"relaxed_accuracy={values['relaxed_accuracy']:.4f}, "
            f"mean_ms={values['mean_ms']:.1f}"
        )
    print(f"predictions: {_relative(predictions_path)}")
    print(f"metrics: {_relative(metrics_path)}")


def _generate(
    query: str,
    image: object,
    model: object,
    processor: object,
    torch_module: object,
    max_new_tokens: int,
) -> str:
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
        )
    return processor.decode(
        generated[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    ).strip()


def _prepare_image(
    page_ids: list[str], pages: dict[str, object], image_module: object, stitch: bool
) -> object:
    images = []
    for page_id in page_ids:
        path = Path(pages[page_id].image_path)
        path = path if path.is_absolute() else PROJECT_ROOT / path
        with image_module.open(path) as source:
            images.append(source.convert("RGB"))
    if not stitch:
        return images[0]
    try:
        target_width = 512
        resized = [
            image.resize(
                (target_width, max(1, round(image.height * target_width / image.width))),
                image_module.Resampling.LANCZOS,
            )
            for image in images
        ]
        margin = 8
        canvas = image_module.new(
            "RGB",
            (target_width, sum(image.height for image in resized) + margin * (len(resized) - 1)),
            "white",
        )
        offset = 0
        for image in resized:
            canvas.paste(image, (0, offset))
            offset += image.height + margin
            image.close()
        return canvas
    finally:
        for image in images:
            image.close()


def exact_match_any(prediction: str, answers: list[str]) -> bool:
    normalized_prediction = _normalize(prediction)
    return any(normalized_prediction == _normalize(answer) for answer in answers)


def relaxed_match_any(prediction: str, answers: list[str]) -> bool:
    if exact_match_any(prediction, answers):
        return True
    predicted_number = _parse_number(prediction)
    if predicted_number is None:
        return False
    for answer in answers:
        target_number = _parse_number(answer)
        if target_number is None:
            continue
        tolerance = 0.05 * abs(target_number)
        if math.isclose(predicted_number, target_number, rel_tol=0.0, abs_tol=tolerance):
            return True
    return False


def _parse_number(value: str) -> float | None:
    cleaned = value.strip().replace(",", "").replace("%", "")
    match = re.fullmatch(r"[^0-9+\-.]*([+-]?(?:\d+(?:\.\d*)?|\.\d+))[^0-9]*", cleaned)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _normalize(value: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return " ".join(value.lower().translate(table).split())


def _load_completed(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    completed: dict[tuple[str, str], dict[str, object]] = {}
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        completed[(row["query_id"], row["mode"])] = row
    return completed


def _aggregate(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    by_mode: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_mode[str(row["mode"])].append(row)
    return {
        mode: {
            "em": sum(bool(row["exact_match"]) for row in mode_rows) / len(mode_rows),
            "relaxed_accuracy": sum(bool(row["relaxed_correct"]) for row in mode_rows)
            / len(mode_rows),
            "mean_ms": sum(float(row["elapsed_ms"]) for row in mode_rows) / len(mode_rows),
        }
        for mode, mode_rows in sorted(by_mode.items())
    }


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
