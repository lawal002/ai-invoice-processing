from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from invoices.services.layoutlmv3_baseline import (  # noqa: E402
    ID_TO_LABEL,
    LABEL_LIST,
    LABEL_TO_ID,
    LayoutLMv3BaselineUnavailable,
    require_layoutlmv3_dependencies,
)


DEFAULT_DATASET_DIR = ROOT / "data" / "annotations" / "layoutlmv3"
DEFAULT_OUTPUT_DIR = ROOT / "models" / "layoutlmv3_baseline_retrained"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class LayoutLMv3TokenDataset:
    def __init__(self, examples: list[dict], processor, *, max_length: int = 512):
        self.examples = examples
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        import torch

        example = self.examples[idx]
        image_path = ROOT / example["image_path"]
        image = Image.open(image_path).convert("RGB")
        word_labels = [LABEL_TO_ID[label] for label in example["labels"]]
        encoding = self.processor(
            image,
            example["words"],
            boxes=example["boxes"],
            word_labels=word_labels,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value.squeeze(0) if isinstance(value, torch.Tensor) else value for key, value in encoding.items()}


def training_arguments(output_dir: Path, args):
    from transformers import TrainingArguments

    common = {
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "report_to": [],
        "remove_unused_columns": False,
        "fp16": args.fp16,
    }
    try:
        return TrainingArguments(evaluation_strategy="epoch", **common)
    except TypeError:
        return TrainingArguments(eval_strategy="epoch", **common)


def label_counts_from_examples(examples: list[dict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for example in examples:
        counts.update(example.get("labels", []))
    return counts


def class_weights_from_examples(examples: list[dict], args):
    import torch

    if args.disable_class_weights:
        return None, {}
    counts = label_counts_from_examples(examples)
    max_count = max(counts.values(), default=1)
    weights = torch.ones(len(LABEL_LIST), dtype=torch.float32)
    weight_report = {}
    for label in LABEL_LIST:
        label_id = LABEL_TO_ID[label]
        count = counts.get(label, 0)
        if count <= 0:
            weight = 1.0
        elif label == "O":
            weight = args.o_label_weight
        else:
            weight = min(args.max_class_weight, max(1.0, (max_count / count) ** 0.5))
        weights[label_id] = float(weight)
        weight_report[label] = {"count": count, "weight": round(float(weight), 4)}
    return weights, weight_report


def weighted_trainer_class():
    import torch
    from transformers import Trainer

    class WeightedTokenClassificationTrainer(Trainer):
        def __init__(self, *args, class_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            if labels is None or self.class_weights is None:
                loss = outputs.loss
            else:
                loss_fct = torch.nn.CrossEntropyLoss(
                    weight=self.class_weights.to(logits.device),
                    ignore_index=-100,
                )
                loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    return WeightedTokenClassificationTrainer


def train(args) -> dict:
    require_layoutlmv3_dependencies()
    from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = ROOT / dataset_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_examples = read_jsonl(dataset_dir / "train.jsonl")
    eval_examples = read_jsonl(dataset_dir / "eval.jsonl")
    if not train_examples:
        raise SystemExit("No train examples found. Run scripts/layoutlmv3_prepare_dataset.py first.")

    class_weights, weight_report = class_weights_from_examples(train_examples, args)
    processor = LayoutLMv3Processor.from_pretrained(args.model_name, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    trainer_cls = weighted_trainer_class()
    trainer = trainer_cls(
        model=model,
        args=training_arguments(output_dir, args),
        train_dataset=LayoutLMv3TokenDataset(train_examples, processor, max_length=args.max_length),
        eval_dataset=LayoutLMv3TokenDataset(eval_examples, processor, max_length=args.max_length) if eval_examples else None,
        tokenizer=processor,
        class_weights=class_weights,
    )
    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    with (output_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump({"label_list": LABEL_LIST, "label_to_id": LABEL_TO_ID, "id_to_label": ID_TO_LABEL}, f, indent=2)

    summary = {
        "model_name": args.model_name,
        "output_dir": str(output_dir.relative_to(ROOT) if output_dir.is_relative_to(ROOT) else output_dir),
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "epochs": args.epochs,
        "max_length": args.max_length,
        "class_weighting": {
            "enabled": class_weights is not None,
            "o_label_weight": args.o_label_weight,
            "max_class_weight": args.max_class_weight,
            "weights": weight_report,
        },
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a small LayoutLMv3 token-classification baseline.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-name", default="microsoft/layoutlmv3-base")
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--disable-class-weights", action="store_true")
    parser.add_argument("--o-label-weight", type=float, default=0.15)
    parser.add_argument("--max-class-weight", type=float, default=8.0)
    args = parser.parse_args()

    try:
        summary = train(args)
    except LayoutLMv3BaselineUnavailable as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
