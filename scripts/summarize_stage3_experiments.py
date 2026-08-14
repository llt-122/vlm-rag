from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = ("recall@1", "recall@3", "recall@10", "mrr@10")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Stage 3 SigLIP retrieval experiments.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--standard", type=Path, required=True)
    parser.add_argument("--hardneg-1", type=Path, required=True)
    parser.add_argument("--hardneg-2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline_payload = _read(args.baseline)
    experiments = {
        "siglip_baseline": baseline_payload["by_split"]["test"],
        "partial_finetune": _read(args.standard)["tuned_test"],
        "partial_finetune_hardneg_1": _read(args.hardneg_1)["tuned_test"],
        "partial_finetune_hardneg_2": _read(args.hardneg_2)["tuned_test"],
    }
    baseline = experiments["siglip_baseline"]
    rows = []
    for method, metrics in experiments.items():
        row: dict[str, str | float] = {"method": method}
        for metric in METRICS:
            value = float(metrics[metric])
            base = float(baseline[metric])
            row[metric] = value
            row[f"{metric}_absolute_gain"] = value - base
            row[f"{metric}_relative_gain_pct"] = 0.0 if base == 0 else 100.0 * (value - base) / base
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "comparison.json"
    csv_path = args.output_dir / "comparison.csv"
    markdown_path = args.output_dir / "comparison.md"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Stage 3 SigLIP Retrieval Comparison",
        "",
        "| Method | Recall@1 | Recall@3 | Recall@10 | MRR@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, metrics in experiments.items():
        lines.append(
            f"| {method} | {metrics['recall@1']:.4f} | {metrics['recall@3']:.4f} | "
            f"{metrics['recall@10']:.4f} | {metrics['mrr@10']:.4f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(markdown_path.read_text(encoding="utf-8"))
    print(f"json: {json_path}")
    print(f"csv: {csv_path}")


def _read(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


if __name__ == "__main__":
    main()
