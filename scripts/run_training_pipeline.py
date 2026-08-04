from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible retriever training pipeline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    config_path = _resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = {key: _resolve(Path(value)) for key, value in config["paths"].items()}
    model = config["model"]
    train = config["training"]
    runtime = config.get("runtime", {})
    python = sys.executable

    commands: list[list[str]] = [
        [
            python,
            str(PROJECT_ROOT / "scripts" / "validate_training_dataset.py"),
            "--dataset-dir",
            str(paths["dataset_dir"]),
            "--report-path",
            str(paths["output_dir"] / "dataset_validation.json"),
        ],
        [
            python,
            str(PROJECT_ROOT / "scripts" / "evaluate_siglip_retrieval.py"),
            "--dataset-dir",
            str(paths["dataset_dir"]),
            "--index-dir",
            str(paths["index_dir"]),
            "--output-dir",
            str(paths["baseline_output_dir"]),
            "--model",
            model,
            "--batch-size",
            str(runtime.get("embedding_batch_size", 16)),
            "--device",
            runtime.get("device", "auto"),
        ],
        [
            python,
            str(PROJECT_ROOT / "scripts" / "train_siglip_adapter.py"),
            "--dataset-dir",
            str(paths["dataset_dir"]),
            "--page-index-dir",
            str(paths["index_dir"]),
            "--feature-dir",
            str(paths["feature_dir"]),
            "--output-dir",
            str(paths["output_dir"]),
            "--model",
            model,
            "--rank",
            str(train["rank"]),
            "--epochs",
            str(train["epochs"]),
            "--patience",
            str(train["patience"]),
            "--learning-rate",
            str(train["learning_rate"]),
            "--weight-decay",
            str(train["weight_decay"]),
            "--temperature",
            str(train["temperature"]),
            "--hard-negatives",
            str(train["hard_negatives"]),
            "--seed",
            str(train["seed"]),
            "--train-towers",
            train["train_towers"],
        ],
    ]
    if args.rebuild_index:
        commands[1].append("--rebuild-index")

    print(f"config: {_relative(config_path)}")
    print(f"python: {python}")
    for index, command in enumerate(commands, start=1):
        print(f"step {index}: {subprocess.list2cmdline(command)}")
    if args.dry_run:
        return

    if runtime.get("require_cuda", True):
        _require_cuda()
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    for command in commands:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=_runtime_env())
    completed_at = datetime.now(timezone.utc)
    run_manifest = {
        "config": _relative(config_path),
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "elapsed_seconds": (completed_at - started_at).total_seconds(),
        "commands": commands,
    }
    (paths["output_dir"] / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _require_cuda() -> None:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is not installed in the active environment") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this config, but no CUDA GPU is available")
    print(f"cuda: {torch.version.cuda}; gpu: {torch.cuda.get_device_name(0)}")


def _runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
