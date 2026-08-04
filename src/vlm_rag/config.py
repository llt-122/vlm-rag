from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    data_dir: str = "data"
    output_dir: str = "outputs"
    model_dir: str = "models"
    index_dir: str = "indexes"
    log_dir: str = "logs"
    top_k: int = 3
    embedding_dim: int = 384
    temperature: float = 0.07
    epochs: int = 5
    hidden_layer_weights: tuple[float, ...] = (0.2, 0.3, 0.5)
    train_ratio: float = 0.7
    dev_ratio: float = 0.15


def load_config(path: Path) -> ProjectConfig:
    if not path.exists():
        return ProjectConfig()

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("\"'")

    defaults = ProjectConfig()
    return ProjectConfig(
        data_dir=values.get("data_dir", defaults.data_dir),
        output_dir=values.get("output_dir", defaults.output_dir),
        model_dir=values.get("model_dir", defaults.model_dir),
        index_dir=values.get("index_dir", defaults.index_dir),
        log_dir=values.get("log_dir", defaults.log_dir),
        top_k=int(values.get("top_k", defaults.top_k)),
        embedding_dim=int(values.get("embedding_dim", defaults.embedding_dim)),
        temperature=float(values.get("temperature", defaults.temperature)),
        epochs=int(values.get("epochs", defaults.epochs)),
        hidden_layer_weights=_parse_weights(
            values.get("hidden_layer_weights"),
            defaults.hidden_layer_weights,
        ),
        train_ratio=float(values.get("train_ratio", defaults.train_ratio)),
        dev_ratio=float(values.get("dev_ratio", defaults.dev_ratio)),
    )


def resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def _parse_weights(value: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    weights = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not weights:
        return default
    total = sum(weights)
    return tuple(weight / total for weight in weights)
