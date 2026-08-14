from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRIC_NAMES = ("recall@1", "recall@3", "recall@10", "mrr@10")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare available large-scale retrieval baselines.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", action="append", nargs=2, metavar=("NAME", "METRICS_JSON"))
    args = parser.parse_args()
    if not args.method:
        parser.error("at least one --method NAME METRICS_JSON pair is required")

    rows = []
    skipped = []
    for name, raw_path in args.method:
        path = Path(raw_path)
        if not path.exists():
            skipped.append({"method": name, "reason": f"missing {path}"})
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        test_metrics = payload.get("by_split", {}).get("test")
        if not isinstance(test_metrics, dict):
            skipped.append({"method": name, "reason": "missing by_split.test metrics"})
            continue
        rows.append({"method": name, **{metric: float(test_metrics[metric]) for metric in METRIC_NAMES}})
    if not rows:
        raise RuntimeError("no comparable metric files were available")
    rows.sort(key=lambda row: float(row["mrr@10"]), reverse=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps({"ranking": rows, "skipped": skipped}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (args.output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["method", *METRIC_NAMES])
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Large ChartQA Retrieval Baselines (Test Split)",
        "",
        "| Rank | Method | Recall@1 | Recall@3 | Recall@10 | MRR@10 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | {row['method']} | {row['recall@1']:.4f} | "
            f"{row['recall@3']:.4f} | {row['recall@10']:.4f} | {row['mrr@10']:.4f} |"
        )
    if skipped:
        lines.extend(["", "## Skipped", ""])
        lines.extend(f"- {item['method']}: {item['reason']}" for item in skipped)
    report = "\n".join(lines) + "\n"
    (args.output_dir / "comparison.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
