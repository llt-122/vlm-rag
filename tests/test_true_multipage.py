from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_true_multipage.py"
SPEC = importlib.util.spec_from_file_location("evaluate_true_multipage", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_retrieval_metrics_distinguish_any_and_all_evidence() -> None:
    rows = [
        {"any_evidence_hit": True, "all_evidence_hit": False, "evidence_coverage": 0.5},
        {"any_evidence_hit": True, "all_evidence_hit": True, "evidence_coverage": 1.0},
    ]
    metrics = MODULE._retrieval_metrics(rows)
    assert metrics["any_evidence_recall"] == 1.0
    assert metrics["all_evidence_recall"] == 0.5
    assert metrics["mean_evidence_coverage"] == 0.75


def test_numeric_relaxed_match() -> None:
    assert MODULE._relaxed_match_any("20", ["20"])
    assert not MODULE._relaxed_match_any("30", ["20"])
