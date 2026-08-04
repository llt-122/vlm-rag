from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class SigLIPConfig:
    """Runtime settings for the real image/text dual-tower encoder."""

    model_name: str = "google/siglip-base-patch16-224"
    device: str = "auto"
    batch_size: int = 8


class SigLIPEncoder:
    """Encode text queries and page images into one normalized vector space.

    Heavy ML dependencies are imported lazily so the original lightweight demo
    can still run without PyTorch or Transformers installed.
    """

    def __init__(self, config: SigLIPConfig | None = None) -> None:
        self.config = config or SigLIPConfig()
        if self.config.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "SigLIP requires PyTorch, Transformers and Pillow. "
                "Install the model environment described in README.md."
            ) from exc

        self._torch = torch
        self.device = self._resolve_device(self.config.device)
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(self.config.model_name, dtype=dtype)
        self.model.to(self.device)
        self.model.eval()

    def encode_texts(self, texts: Iterable[str]) -> list[list[float]]:
        values = list(texts)
        if not values:
            return []
        if any(not text.strip() for text in values):
            raise ValueError("text inputs must not be empty")

        vectors: list[list[float]] = []
        for batch in _batches(values, self.config.batch_size):
            inputs = self.processor(
                text=batch,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
            with self._torch.inference_mode():
                output = self.model.get_text_features(**inputs)
                features = _feature_tensor(output)
                features = self._torch.nn.functional.normalize(features.float(), dim=-1)
            vectors.extend(features.cpu().tolist())
        return vectors

    def encode_images(self, image_paths: Iterable[Path]) -> list[list[float]]:
        from PIL import Image

        paths = [Path(path) for path in image_paths]
        if not paths:
            return []

        vectors: list[list[float]] = []
        for batch in _batches(paths, self.config.batch_size):
            images = []
            try:
                for path in batch:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
                with self._torch.inference_mode():
                    output = self.model.get_image_features(**inputs)
                    features = _feature_tensor(output)
                    features = self._torch.nn.functional.normalize(features.float(), dim=-1)
                vectors.extend(features.cpu().tolist())
            finally:
                for image in images:
                    image.close()
        return vectors

    def encode_text(self, text: str) -> list[float]:
        return self.encode_texts([text])[0]

    def encode_image(self, image_path: Path) -> list[float]:
        return self.encode_images([image_path])[0]

    def _resolve_device(self, requested: str) -> Any:
        if requested == "auto":
            requested = "cuda" if self._torch.cuda.is_available() else "cpu"
        device = self._torch.device(requested)
        if device.type == "cuda" and not self._torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU")
        return device


def _feature_tensor(output: Any) -> Any:
    """Support both tensor and ModelOutput return types across Transformers versions."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to read SigLIP features") from exc

    if isinstance(output, torch.Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    if isinstance(pooled, torch.Tensor):
        return pooled
    raise TypeError(f"unsupported SigLIP feature output: {type(output).__name__}")


def _batches(values: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]
