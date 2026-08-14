from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlm_rag.dataset_schema import DocumentPage, DocumentQASample, load_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Partially fine-tune both SigLIP towers with multi-positive InfoNCE."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="google/siglip-base-patch16-224")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--unfreeze-text-layers", type=int, default=2)
    parser.add_argument("--unfreeze-vision-layers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hard-negative-results",
        type=Path,
        help="Baseline retrieval_results.json used to mine static hard-negative pages.",
    )
    parser.add_argument("--hard-negatives-per-query", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-dev-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    args = parser.parse_args()
    _validate_args(args, parser)

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoProcessor, get_linear_schedule_with_warmup

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This partial fine-tuning experiment requires a CUDA GPU")

    bundle = load_bundle(args.dataset_dir)
    pages = {page.page_id: page for page in bundle.pages}
    train_samples = _select_split(bundle.samples, "train", args.max_train_samples)
    dev_samples = _select_split(bundle.samples, "dev", args.max_dev_samples)
    test_samples = _select_split(bundle.samples, "test", args.max_test_samples)
    if not train_samples or not dev_samples or not test_samples:
        raise ValueError("train, dev and test splits must all contain at least one sample")
    hard_negatives = _load_hard_negatives(
        args.hard_negative_results,
        train_samples,
        pages,
        args.hard_negatives_per_query,
    )

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    trainable_names = _configure_trainable_layers(
        model,
        text_layers=args.unfreeze_text_layers,
        vision_layers=args.unfreeze_vision_layers,
    )
    model.to(device)

    class PairDataset(Dataset):
        def __init__(self, samples: list[DocumentQASample]) -> None:
            self.samples = samples

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int) -> tuple[str, str, Path, list[tuple[str, Path]]]:
            sample = self.samples[index]
            page_id = sample.evidence_page_ids[0]
            negative_rows = [
                (negative_id, _resolve_image_path(pages[negative_id]))
                for negative_id in hard_negatives.get(sample.query_id, [])
            ]
            return sample.query, page_id, _resolve_image_path(pages[page_id]), negative_rows

    def collate_pairs(
        rows: list[tuple[str, str, Path, list[tuple[str, Path]]]]
    ) -> dict[str, object]:
        from PIL import Image

        texts = [row[0] for row in rows]
        positive_page_ids = [row[1] for row in rows]
        image_rows = [(row[1], row[2]) for row in rows]
        image_rows.extend(negative for row in rows for negative in row[3])
        image_page_ids = [row[0] for row in image_rows]
        images = []
        try:
            for _, path in image_rows:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(
                text=texts,
                images=images,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        finally:
            for image in images:
                image.close()
        return {
            "inputs": inputs,
            "positive_page_ids": positive_page_ids,
            "image_page_ids": image_page_ids,
        }

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        PairDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_pairs,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_updates = max(1, updates_per_epoch * args.epochs)
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)

    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"device: {device}")
    print(f"train/dev/test: {len(train_samples)}/{len(dev_samples)}/{len(test_samples)}")
    print(
        f"trainable parameters: {trainable_count:,}/{total_count:,} "
        f"({100.0 * trainable_count / total_count:.2f}%)"
    )
    print(f"optimizer updates: {total_updates}; warmup: {warmup_steps}")
    print(f"hard negatives per query: {args.hard_negatives_per_query}")

    baseline_dev = _evaluate(
        model, processor, pages, dev_samples, args.eval_batch_size, device, torch, F
    )
    print(_metric_line("dev baseline", baseline_dev))
    best_dev_mrr = baseline_dev["mrr@10"]
    best_epoch = 0
    best_state = _trainable_state(model)
    stale_epochs = 0
    log_rows: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        optimizer_steps = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            inputs = {
                name: tensor.to(device, non_blocking=True)
                for name, tensor in batch["inputs"].items()
            }
            context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            with context:
                outputs = model(**inputs)
                loss = _multi_positive_infonce(
                    outputs.text_embeds,
                    outputs.image_embeds,
                    batch["positive_page_ids"],
                    batch["image_page_ids"],
                    args.temperature,
                    torch,
                    F,
                )
                scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()
            running_loss += float(loss.detach().cpu())
            should_step = (
                batch_index % args.gradient_accumulation == 0 or batch_index == len(train_loader)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            if batch_index == 1 or batch_index % 50 == 0:
                print(
                    f"epoch={epoch} batch={batch_index}/{len(train_loader)} "
                    f"loss={float(loss.detach()):.4f}"
                )

        dev_metrics = _evaluate(
            model, processor, pages, dev_samples, args.eval_batch_size, device, torch, F
        )
        epoch_loss = running_loss / max(1, len(train_loader))
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "dev_recall@1": dev_metrics["recall@1"],
                "dev_recall@3": dev_metrics["recall@3"],
                "dev_recall@10": dev_metrics["recall@10"],
                "dev_mrr@10": dev_metrics["mrr@10"],
            }
        )
        print(f"epoch={epoch} mean_loss={epoch_loss:.4f}; {_metric_line('dev', dev_metrics)}")
        if dev_metrics["mrr@10"] > best_dev_mrr + 1e-9:
            best_dev_mrr = dev_metrics["mrr@10"]
            best_epoch = epoch
            best_state = _trainable_state(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state, strict=False)
    final_dev = _evaluate(
        model, processor, pages, dev_samples, args.eval_batch_size, device, torch, F
    )
    final_test = _evaluate(
        model, processor, pages, test_samples, args.eval_batch_size, device, torch, F
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "siglip_partial_finetune.pt"
    torch.save(
        {
            "trainable_state_dict": best_state,
            "base_model": args.model,
            "trainable_parameter_names": trainable_names,
            "best_epoch": best_epoch,
            "args": vars(args),
        },
        checkpoint_path,
    )
    if log_rows:
        with (args.output_dir / "training_log.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(log_rows[0]))
            writer.writeheader()
            writer.writerows(log_rows)
    metrics = {
        "method": "siglip_partial_dual_tower_finetune",
        "base_model": args.model,
        "best_epoch": best_epoch,
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "baseline_dev": baseline_dev,
        "tuned_dev": final_dev,
        "tuned_test": final_test,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best_epoch: {best_epoch}")
    print(_metric_line("final dev", final_dev))
    print(_metric_line("final test", final_test))
    print(f"checkpoint: {_relative(checkpoint_path)}")
    print(f"metrics: {_relative(metrics_path)}")


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    positive_names = (
        "epochs",
        "batch_size",
        "eval_batch_size",
        "gradient_accumulation",
        "patience",
    )
    for name in positive_names:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.unfreeze_text_layers < 0 or args.unfreeze_vision_layers < 0:
        parser.error("unfrozen layer counts must be non-negative")
    if args.hard_negatives_per_query < 0:
        parser.error("--hard-negatives-per-query must be non-negative")
    if args.hard_negatives_per_query and args.hard_negative_results is None:
        parser.error("--hard-negative-results is required when hard negatives are enabled")
    if args.learning_rate <= 0 or args.temperature <= 0:
        parser.error("learning rate and temperature must be positive")


def _select_split(
    samples: list[DocumentQASample], split: str, limit: int | None
) -> list[DocumentQASample]:
    selected = [sample for sample in samples if sample.split == split]
    return selected if limit is None else selected[:limit]


def _resolve_image_path(page: DocumentPage) -> Path:
    path = Path(page.image_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_hard_negatives(
    results_path: Path | None,
    train_samples: list[DocumentQASample],
    pages: dict[str, DocumentPage],
    count: int,
) -> dict[str, list[str]]:
    if count == 0:
        return {}
    if results_path is None or not results_path.exists():
        raise FileNotFoundError(f"hard-negative results not found: {results_path}")
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    rows_by_query = {str(row["query_id"]): row for row in payload}
    selected: dict[str, list[str]] = {}
    for sample in train_samples:
        row = rows_by_query.get(sample.query_id)
        if row is None:
            raise ValueError(f"missing baseline retrieval result for {sample.query_id}")
        evidence = set(sample.evidence_page_ids)
        negatives = []
        for hit in row.get("top_10", []):
            page_id = str(hit["page_id"])
            if page_id not in evidence and page_id in pages and page_id not in negatives:
                negatives.append(page_id)
            if len(negatives) == count:
                break
        if len(negatives) < count:
            raise ValueError(
                f"{sample.query_id}: requested {count} hard negatives but found {len(negatives)}"
            )
        selected[sample.query_id] = negatives
    return selected


def _configure_trainable_layers(model: object, text_layers: int, vision_layers: int) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    text_encoder_layers = model.text_model.encoder.layers
    vision_encoder_layers = model.vision_model.encoder.layers
    if text_layers > len(text_encoder_layers) or vision_layers > len(vision_encoder_layers):
        raise ValueError("requested more unfrozen layers than the base SigLIP model contains")
    modules = [model.text_model.final_layer_norm, model.text_model.head]
    modules.extend(text_encoder_layers[-text_layers:] if text_layers else [])
    modules.extend([model.vision_model.post_layernorm, model.vision_model.head])
    modules.extend(vision_encoder_layers[-vision_layers:] if vision_layers else [])
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def _trainable_state(model: object) -> dict[str, object]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.state_dict().items()
        if any(name == trainable or name.startswith(trainable + ".") for trainable in _trainable_roots(model))
    }


def _trainable_roots(model: object) -> set[str]:
    return {
        name.rsplit(".", 1)[0]
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _multi_positive_infonce(
    text_embeddings: object,
    image_embeddings: object,
    positive_page_ids: list[str],
    image_page_ids: list[str],
    temperature: float,
    torch_module: object,
    functional: object,
) -> object:
    queries = functional.normalize(text_embeddings.float(), dim=-1)
    pages = functional.normalize(image_embeddings.float(), dim=-1)
    logits = queries @ pages.T / temperature
    positive_mask = torch_module.tensor(
        [[left == right for right in image_page_ids] for left in positive_page_ids],
        dtype=torch_module.bool,
        device=logits.device,
    )
    if not bool(positive_mask.any(dim=1).all()):
        raise ValueError("every query must have at least one positive image in its batch")
    negative_infinity = torch_module.finfo(logits.dtype).min
    query_loss = -(
        torch_module.logsumexp(logits.masked_fill(~positive_mask, negative_infinity), dim=1)
        - torch_module.logsumexp(logits, dim=1)
    ).mean()
    image_has_positive = positive_mask.any(dim=0)
    image_logits = logits.T[image_has_positive]
    image_mask = positive_mask.T[image_has_positive]
    image_loss = -(
        torch_module.logsumexp(image_logits.masked_fill(~image_mask, negative_infinity), dim=1)
        - torch_module.logsumexp(image_logits, dim=1)
    ).mean()
    return (query_loss + image_loss) / 2


def _evaluate(
    model: object,
    processor: object,
    pages_by_id: dict[str, DocumentPage],
    samples: list[DocumentQASample],
    batch_size: int,
    device: object,
    torch_module: object,
    functional: object,
) -> dict[str, float]:
    from PIL import Image

    model.eval()
    page_ids = list(pages_by_id)
    page_vectors = []
    with torch_module.inference_mode():
        for start in range(0, len(page_ids), batch_size):
            batch_ids = page_ids[start : start + batch_size]
            images = []
            try:
                for page_id in batch_ids:
                    with Image.open(_resolve_image_path(pages_by_id[page_id])) as image:
                        images.append(image.convert("RGB"))
                inputs = processor(images=images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)
                with torch_module.autocast(device_type="cuda", dtype=torch_module.bfloat16):
                    output = model.get_image_features(pixel_values=pixel_values)
                page_vectors.append(functional.normalize(_feature_tensor(output).float(), dim=-1).cpu())
            finally:
                for image in images:
                    image.close()
        query_vectors = []
        texts = [sample.query for sample in samples]
        for start in range(0, len(texts), batch_size):
            inputs = processor(
                text=texts[start : start + batch_size],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            with torch_module.autocast(device_type="cuda", dtype=torch_module.bfloat16):
                output = model.get_text_features(**inputs)
            query_vectors.append(functional.normalize(_feature_tensor(output).float(), dim=-1).cpu())
    pages = torch_module.cat(page_vectors)
    queries = torch_module.cat(query_vectors)
    scores = queries @ pages.T
    page_position = {page_id: index for index, page_id in enumerate(page_ids)}
    ranks: list[int | None] = []
    for sample, row in zip(samples, scores, strict=True):
        evidence_positions = {page_position[page_id] for page_id in sample.evidence_page_ids}
        order = torch_module.argsort(row, descending=True)[:10].tolist()
        ranks.append(
            next(
                (rank for rank, position in enumerate(order, start=1) if position in evidence_positions),
                None,
            )
        )
    return _metrics(ranks)


def _feature_tensor(output: object) -> object:
    pooled = getattr(output, "pooler_output", None)
    return pooled if pooled is not None else output


def _metrics(ranks: list[int | None]) -> dict[str, float]:
    count = len(ranks)
    return {
        "recall@1": sum(rank is not None and rank <= 1 for rank in ranks) / count,
        "recall@3": sum(rank is not None and rank <= 3 for rank in ranks) / count,
        "recall@10": sum(rank is not None and rank <= 10 for rank in ranks) / count,
        "mrr@10": sum(1.0 / rank for rank in ranks if rank is not None and rank <= 10) / count,
    }


def _metric_line(label: str, metrics: dict[str, float]) -> str:
    return (
        f"{label}: R@1={metrics['recall@1']:.4f} R@3={metrics['recall@3']:.4f} "
        f"R@10={metrics['recall@10']:.4f} MRR@10={metrics['mrr@10']:.4f}"
    )


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


if __name__ == "__main__":
    main()
