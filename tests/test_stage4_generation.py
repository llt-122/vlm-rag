from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_stage4_generation.py"
SPEC = importlib.util.spec_from_file_location("evaluate_stage4_generation", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_weighted_vote_combines_repeated_answers() -> None:
    rows = [
        {"answer": "10", "retrieval_score": 0.9},
        {"answer": "20", "retrieval_score": 0.8},
        {"answer": "20", "retrieval_score": 0.7},
    ]

    answer, weights = MODULE._weighted_vote(rows, temperature=1.0)

    assert answer == "20"
    assert weights["20"] > weights["10"]


def test_relaxed_numeric_match_uses_five_percent_tolerance() -> None:
    assert MODULE._relaxed_match_any("104", ["100"])
    assert not MODULE._relaxed_match_any("106", ["100"])


def test_aggregate_metrics_separates_retrieval_and_generation_errors() -> None:
    rows = [
        {
            "method": "demo",
            "retrieval_hit": True,
            "exact_match": True,
            "relaxed_correct": True,
            "generation_ms": 10.0,
            "error_type": "correct",
        },
        {
            "method": "demo",
            "retrieval_hit": True,
            "exact_match": False,
            "relaxed_correct": False,
            "generation_ms": 20.0,
            "error_type": "generation_error",
        },
        {
            "method": "demo",
            "retrieval_hit": False,
            "exact_match": False,
            "relaxed_correct": False,
            "generation_ms": 30.0,
            "error_type": "retrieval_miss",
        },
    ]

    metrics = MODULE._aggregate_metrics(rows)[0]

    assert metrics["retrieval_recall"] == 2 / 3
    assert metrics["relaxed_accuracy"] == 1 / 3
    assert metrics["accuracy_given_retrieval_hit"] == 1 / 2
    assert metrics["retrieval_misses"] == 1
    assert metrics["generation_errors"] == 1
