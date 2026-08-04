from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SmolVLMConfig:
    model_name: str = "HuggingFaceTB/SmolVLM-500M-Instruct"
    device: str = "cuda:0"
    max_new_tokens: int = 32


class SmolVLMGenerator:
    """Local lightweight VLM used for the page-image QA baseline."""

    def __init__(self, config: SmolVLMConfig | None = None) -> None:
        self.config = config or SmolVLMConfig()
        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("SmolVLM requires the .venv-colpali model environment") from exc

        self._torch = torch
        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this SmolVLM baseline")
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.config.model_name,
            dtype=torch.bfloat16,
            device_map=self.config.device,
            attn_implementation="eager",
        ).eval()

    @property
    def device(self):
        return self.model.device

    def answer(self, image_path: Path, query: str) -> str:
        from PIL import Image

        with Image.open(image_path) as source:
            image = source.convert("RGB")
        try:
            return self.answer_image(image, query)
        finally:
            image.close()

    def answer_image(self, image, query: str) -> str:
        instruction = (
            "Answer the question using only the chart image. "
            "Return only the short answer without explanation.\n"
            f"Question: {query}"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(self.device)
        prompt_length = inputs["input_ids"].shape[-1]
        with self._torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )
        answer_tokens = generated[0, prompt_length:]
        return self.processor.decode(answer_tokens, skip_special_tokens=True).strip()
