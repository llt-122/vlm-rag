from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlm_rag.dataset_schema import DatasetBundle, DocumentPage, DocumentQASample, save_bundle


COMPANIES = [
    "Aster", "Beacon", "Cedar", "Delta", "Evergreen",
    "Falcon", "Granite", "Harbor", "Indigo", "Juniper",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a controlled true multi-page QA set.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "multipage" / "finance_40",
    )
    args = parser.parse_args()
    pages_dir = args.output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    pages: list[DocumentPage] = []
    samples: list[DocumentQASample] = []
    source_records: list[dict[str, object]] = []

    for index, company in enumerate(COMPANIES, start=1):
        doc_id = f"finance_{index:02d}_{company.lower()}"
        revenue_2025 = 100 + index * 13
        revenue_2026 = revenue_2025 + 8 + index * 2
        cost_2025 = 58 + index * 7
        cost_2026 = cost_2025 + 5 + index
        employees_2025 = 420 + index * 37
        employees_2026 = employees_2025 + 18 + index * 3
        region = ["East", "West", "North", "South", "Central"][index % 5]

        page_specs = [
            (
                1,
                "FY2025 PERFORMANCE",
                [
                    ("Revenue", revenue_2025, "million CNY"),
                    ("Operating cost", cost_2025, "million CNY"),
                    ("Employees", employees_2025, "people"),
                    ("Leading region", region, ""),
                ],
            ),
            (
                2,
                "FY2026 PERFORMANCE",
                [
                    ("Revenue", revenue_2026, "million CNY"),
                    ("Operating cost", cost_2026, "million CNY"),
                    ("Employees", employees_2026, "people"),
                    ("Leading region", region, ""),
                ],
            ),
            (
                3,
                "TWO-YEAR SUMMARY",
                [
                    ("Capital expenditure", 20 + index * 2, "million CNY"),
                    ("R&D expenditure", 9 + index, "million CNY"),
                    ("Customer complaints", 35 - index, "cases"),
                    ("Reporting currency", "CNY", ""),
                ],
            ),
        ]

        page_ids: list[str] = []
        for page_no, title, rows in page_specs:
            page_id = f"{doc_id}_p{page_no}"
            page_ids.append(page_id)
            relative_path = Path("data") / "multipage" / "finance_40" / "pages" / f"{page_id}.png"
            _render_page(PROJECT_ROOT / relative_path, company, page_no, title, rows)
            pages.append(
                DocumentPage(
                    page_id=page_id,
                    doc_id=doc_id,
                    doc_type="financial_report",
                    page_no=page_no,
                    image_path=relative_path.as_posix(),
                    title=f"{company} Holdings - {title}",
                    metadata={"company": company, "synthetic_controlled": True},
                )
            )

        questions = [
            (
                "revenue_growth",
                f"How much did {company} Holdings' revenue increase from 2025 to 2026?",
                str(revenue_2026 - revenue_2025),
                page_ids[:2],
                "subtract page 1 revenue from page 2 revenue",
            ),
            (
                "profit_change",
                f"How much did {company} Holdings' operating profit change from 2025 to 2026?",
                str((revenue_2026 - cost_2026) - (revenue_2025 - cost_2025)),
                page_ids[:2],
                "compute revenue minus cost on pages 1 and 2, then compare",
            ),
            (
                "employee_growth",
                f"How many employees did {company} Holdings add from 2025 to 2026?",
                str(employees_2026 - employees_2025),
                page_ids[:2],
                "subtract page 1 employee count from page 2 employee count",
            ),
            (
                "three_page_total",
                f"What is the sum of {company} Holdings' 2026 operating profit and R&D expenditure?",
                str((revenue_2026 - cost_2026) + (9 + index)),
                [page_ids[1], page_ids[2]],
                "compute page 2 operating profit and add page 3 R&D expenditure",
            ),
        ]
        for q_index, (kind, query, answer, evidence, derivation) in enumerate(questions, start=1):
            query_id = f"{doc_id}_q{q_index}"
            samples.append(
                DocumentQASample(
                    query_id=query_id,
                    query=query,
                    answers=[answer],
                    evidence_page_ids=evidence,
                    doc_type="financial_report",
                    split="test",
                    metadata={
                        "question_type": kind,
                        "requires_all_evidence_pages": True,
                        "derivation": derivation,
                        "synthetic_controlled": True,
                    },
                )
            )
            source_records.append(
                {
                    "query_id": query_id,
                    "document": f"{company} Holdings three-page financial report",
                    "query": query,
                    "answer": answer,
                    "evidence_pages": evidence,
                    "derivation": derivation,
                }
            )

    manifest = save_bundle(DatasetBundle(pages=pages, samples=samples), args.output_dir)
    manifest.update(
        {
            "dataset_name": "Controlled Multi-page Financial QA 40",
            "document_count": len(COMPANIES),
            "pages_per_document": 3,
            "all_questions_require_multiple_pages": True,
            "provenance": "Programmatically generated controlled validation data; not real enterprise data.",
        }
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "annotation_readable.json").write_text(
        json.dumps(source_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _render_page(
    path: Path,
    company: str,
    page_no: int,
    title: str,
    rows: list[tuple[str, object, str]],
) -> None:
    image = Image.new("RGB", (1400, 1900), "white")
    draw = ImageDraw.Draw(image)
    regular = _font(44)
    small = _font(32)
    title_font = _font(68)
    value_font = _font(54)
    navy = "#16324F"
    blue = "#2E6F9E"
    draw.rectangle((0, 0, 1400, 260), fill=navy)
    draw.text((90, 65), f"{company} HOLDINGS", font=title_font, fill="white")
    draw.text((92, 165), "ANNUAL FINANCIAL REPORT", font=small, fill="#D9EAF7")
    draw.text((90, 330), title, font=title_font, fill=navy)
    draw.line((90, 435, 1310, 435), fill=blue, width=6)
    y = 540
    for label, value, unit in rows:
        draw.rounded_rectangle((90, y, 1310, y + 220), radius=18, fill="#F3F6F8", outline="#B8C5CF", width=3)
        draw.text((135, y + 38), label, font=regular, fill="#263746")
        shown = f"{value} {unit}".strip()
        draw.text((135, y + 115), shown, font=value_font, fill=blue)
        y += 270
    draw.text((90, 1770), f"Document page {page_no} of 3", font=small, fill="#586875")
    draw.text((930, 1770), "Internal validation sample", font=small, fill="#586875")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
