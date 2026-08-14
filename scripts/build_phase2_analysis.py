from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build phase-2 experiment tables and failure analysis.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "phase2_delivery"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stage3 = args.experiment_root / "stage3" / "outputs"
    stage4 = args.experiment_root / "stage4" / "outputs" / "stage4_generation_test"

    retrieval_rows = _retrieval_table(stage3)
    generation_rows = _generation_table(stage4)
    failure_rows, failure_summary = _failure_analysis(stage3)

    _write_csv(args.output_dir / "five_retrieval_baselines.csv", retrieval_rows)
    _write_csv(args.output_dir / "four_smolvlm_qa_schemes.csv", generation_rows)
    _write_csv(args.output_dir / "retrieval_failure_cases.csv", failure_rows)
    _write_json(args.output_dir / "retrieval_failure_cases.json", failure_rows)
    _write_json(args.output_dir / "retrieval_failure_summary.json", failure_summary)
    _write_markdown(
        args.output_dir / "five_retrieval_baselines.md",
        "五种检索方案统一基线表",
        retrieval_rows,
    )
    _write_markdown(
        args.output_dir / "four_smolvlm_qa_schemes.md",
        "四种 SmolVLM 问答方案对比",
        generation_rows,
    )
    _write_failure_markdown(
        args.output_dir / "retrieval_failure_analysis.md", failure_rows, failure_summary
    )
    print(f"retrieval methods: {len(retrieval_rows)}")
    print(f"generation schemes: {len(generation_rows)}")
    print(f"failure cases: {len(failure_rows)}")
    print(f"output: {args.output_dir}")


def _retrieval_table(stage3: Path) -> list[dict[str, object]]:
    baseline = _json(stage3 / "siglip_chartqa_large_baseline" / "metrics.json")["by_split"]["test"]
    partial = _json(stage3 / "siglip_partial_full" / "metrics.json")["tuned_test"]
    hard = _json(stage3 / "siglip_partial_hardneg_1" / "metrics.json")["tuned_test"]
    ocr = _json(stage3 / "ocr_bge_chartqa_large" / "metrics.json")["by_split"]["test"]
    colsmol = _json(stage3 / "colsmol_chartqa_large" / "metrics.json")["by_split"]["test"]
    methods = [
        ("SigLIP zero-shot", "全局图文向量零样本基线", baseline),
        ("SigLIP InfoNCE partial fine-tune", "部分解冻双塔并用 InfoNCE 微调", partial),
        ("SigLIP hard-negative fine-tune", "每个 Query 加入 1 个难负例", hard),
        ("PPOCR + BGE", "传统 OCR 文本链路", ocr),
        ("ColSmol", "多向量页面表示与晚交互", colsmol),
    ]
    return [
        {
            "方案": name,
            "作用": purpose,
            "Recall@1": round(metric["recall@1"], 4),
            "Recall@3": round(metric["recall@3"], 4),
            "Recall@10": round(metric["recall@10"], 4),
            "MRR@10": round(metric["mrr@10"], 4),
        }
        for name, purpose, metric in methods
    ]


def _generation_table(stage4: Path) -> list[dict[str, object]]:
    metrics = _json(stage4 / "metrics.json")["methods"]
    by_name = {row["method"]: row for row in metrics}
    selected = [
        ("Oracle Page", "oracle_page", "直接给标准证据页，测生成上限"),
        ("Top-1", "colsmol_top1", "ColSmol 第一名页面单图问答"),
        ("Top-3 sequential", "colsmol_top3_sequential", "三页分别推理后加权融合"),
        ("Top-3 collage", "colsmol_top3_collage", "三页拼图后一次推理"),
    ]
    return [
        {
            "方案": label,
            "作用": purpose,
            "证据召回率": round(by_name[key]["retrieval_recall"], 4),
            "完全匹配率": round(by_name[key]["exact_match"], 4),
            "宽松准确率": round(by_name[key]["relaxed_accuracy"], 4),
            "命中证据后的准确率": round(by_name[key]["accuracy_given_retrieval_hit"], 4),
            "平均生成耗时(ms)": round(by_name[key]["mean_generation_ms"], 1),
        }
        for label, key, purpose in selected
    ]


def _failure_analysis(stage3: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    paths = {
        "siglip_zero": stage3 / "siglip_chartqa_large_baseline" / "retrieval_results.json",
        "siglip_tuned": stage3 / "siglip_partial_chartqa_large" / "retrieval_results.json",
        "ocr_bge": stage3 / "ocr_bge_chartqa_large" / "retrieval_results.json",
        "colsmol": stage3 / "colsmol_chartqa_large" / "retrieval_results.json",
    }
    maps = {name: {row["query_id"]: row for row in _json(path)} for name, path in paths.items()}
    test_ids = [qid for qid, row in maps["colsmol"].items() if row.get("split") == "test"]
    counts: dict[str, int] = {}
    candidates: list[dict[str, object]] = []
    for query_id in test_ids:
        rows = {name: values[query_id] for name, values in maps.items()}
        ranks = {name: _rank(row) for name, row in rows.items()}
        category, reason = _categorize(ranks)
        counts[category] = counts.get(category, 0) + 1
        candidates.append(
            {
                "query_id": query_id,
                "query": rows["colsmol"]["query"],
                "failure_category": category,
                "reason": reason,
                "siglip_zero_rank": ranks["siglip_zero"],
                "siglip_tuned_rank": ranks["siglip_tuned"],
                "ocr_bge_rank": ranks["ocr_bge"],
                "colsmol_rank": ranks["colsmol"],
                "evidence_page_ids": "|".join(rows["colsmol"]["evidence_page_ids"]),
                "colsmol_top1": rows["colsmol"]["top_10"][0]["page_id"],
            }
        )

    priority = {
        "all_methods_miss": 0,
        "colsmol_recovers": 1,
        "ocr_advantage": 2,
        "finetune_recovers": 3,
        "ranked_below_top3": 4,
        "top3_success": 5,
    }
    selected = []
    category_caps = {key: 6 for key in priority}
    for row in sorted(candidates, key=lambda item: (priority[item["failure_category"]], item["query_id"])):
        category = row["failure_category"]
        if category_caps[category] > 0:
            selected.append(row)
            category_caps[category] -= 1
    summary = {
        "test_query_count": len(test_ids),
        "category_counts": dict(sorted(counts.items())),
        "interpretation": {
            "all_methods_miss": "四条可用检索链路的前10名均无证据页，属于困难样本或查询-页面语义断裂。",
            "colsmol_recovers": "全局向量或 OCR 失败，但晚交互保留局部词元匹配后成功。",
            "ocr_advantage": "OCR 文本直接命中关键字符串，说明纯视觉向量并非所有问题都占优。",
            "finetune_recovers": "InfoNCE 微调改善了正确页排名。",
            "ranked_below_top3": "证据在第4至10名，扩大 Top-K 可提升召回但会增加生成成本。",
            "top3_success": "至少一种方法在前三名命中，不计为主要失败。",
        },
    }
    return selected, summary


def _rank(row: dict[str, object]) -> int:
    value = row.get("first_evidence_rank")
    return int(value) if value is not None else 999


def _categorize(ranks: dict[str, int]) -> tuple[str, str]:
    if all(rank > 10 for rank in ranks.values()):
        return "all_methods_miss", "所有方法前10均未召回证据，需检查视觉相似页面、Query歧义或小文字问题"
    if ranks["colsmol"] <= 3 and all(ranks[name] > 3 for name in ("siglip_zero", "siglip_tuned", "ocr_bge")):
        return "colsmol_recovers", "ColSmol局部晚交互成功，其他全局/OCR表示未进入前三"
    if ranks["ocr_bge"] < min(ranks["siglip_tuned"], ranks["colsmol"]):
        return "ocr_advantage", "问题包含可直接匹配的文本线索，OCR+BGE排名更靠前"
    if ranks["siglip_tuned"] <= 3 < ranks["siglip_zero"]:
        return "finetune_recovers", "InfoNCE微调把证据页提升到前三"
    if min(ranks.values()) <= 10 and min(ranks.values()) > 3:
        return "ranked_below_top3", "证据已进入前10但未进入前三，存在Top-K召回与推理成本权衡"
    return "top3_success", "至少一种方案前三命中，用作相对成功参照"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(path: Path, title: str, rows: list[dict[str, object]]) -> None:
    headers = list(rows[0])
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(row[key]) for key in headers) + " |" for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_failure_markdown(path: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    lines = ["# 检索失败案例原因分析", "", f"测试问题数：{summary['test_query_count']}。", "", "## 原因分布", ""]
    for key, value in summary["category_counts"].items():
        lines.append(f"- {key}: {value} — {summary['interpretation'][key]}")
    lines.extend(["", "## 代表案例", ""])
    for row in rows:
        lines.extend([
            f"### {row['query_id']} · {row['failure_category']}",
            "",
            f"Query：{row['query']}",
            "",
            f"排名：SigLIP零样本 {row['siglip_zero_rank']}，微调SigLIP {row['siglip_tuned_rank']}，OCR+BGE {row['ocr_bge_rank']}，ColSmol {row['colsmol_rank']}。",
            "",
            f"判断：{row['reason']}。",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
