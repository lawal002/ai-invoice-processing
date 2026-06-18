from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from invoices.services.evaluation import (  # noqa: E402
    FIELDS,
    NORMALIZATION_VERSION,
    EvaluationRecord,
    save_evaluation_outputs,
    values_equal,
)
from invoices.services.layoutlmv3_baseline import (  # noqa: E402
    ID_TO_LABEL,
    LABEL_TO_ID,
    LayoutLMv3BaselineUnavailable,
    decode_token_predictions_with_context,
    require_layoutlmv3_dependencies,
)
from invoices.services.ocr import OCRTokenData  # noqa: E402


DEFAULT_DATASET_DIR = ROOT / "data" / "annotations" / "layoutlmv3"
DEFAULT_MODEL_DIR = ROOT / "models" / "layoutlmv3_baseline_v2"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "layoutlmv3_baseline_v2_decoder"
REQUIRED_FIELDS = ["invoice_number", "date", "vendor_name", "total_amount", "currency"]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_label_map(model_dir: Path) -> tuple[dict[int, str], dict[str, int]]:
    path = model_dir / "label_map.json"
    if not path.exists():
        return ID_TO_LABEL, LABEL_TO_ID
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    id_to_label = {int(key): value for key, value in payload["id_to_label"].items()}
    label_to_id = {key: int(value) for key, value in payload["label_to_id"].items()}
    return id_to_label, label_to_id


def predict_example(example: dict, processor, model, id_to_label: dict[int, str], *, max_length: int) -> tuple[dict, float]:
    import torch

    image = Image.open(ROOT / example["image_path"]).convert("RGB")
    encoding = processor(
        image,
        example["words"],
        boxes=example["boxes"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    word_ids = encoding.word_ids(batch_index=0) if hasattr(encoding, "word_ids") else []
    device = next(model.parameters()).device
    encoding = {key: value.to(device) for key, value in encoding.items()}

    started = time.perf_counter()
    with torch.no_grad():
        outputs = model(**encoding)
        predictions = outputs.logits.argmax(-1).squeeze(0).cpu().tolist()
    elapsed_ms = (time.perf_counter() - started) * 1000

    word_labels: list[str] = ["O"] * len(example["words"])
    seen_word_ids = set()
    for token_idx, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen_word_ids or word_id >= len(word_labels):
            continue
        seen_word_ids.add(word_id)
        word_labels[word_id] = id_to_label.get(int(predictions[token_idx]), "O")

    tokens = [
        OCRTokenData(word, [float(value) for value in box], 1.0)
        for word, box in zip(example["words"], example.get("boxes", []), strict=False)
    ]
    return decode_token_predictions_with_context(tokens, word_labels, category=example.get("category", "")), elapsed_ms


def evaluate(args) -> dict:
    require_layoutlmv3_dependencies()
    import torch
    from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = ROOT / dataset_dir
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_examples = read_jsonl(dataset_dir / "eval.jsonl")
    if not eval_examples:
        raise SystemExit("No eval examples found. Run scripts/layoutlmv3_prepare_dataset.py first.")
    if not model_dir.exists():
        raise SystemExit(f"LayoutLMv3 model folder not found: {model_dir}. Run layoutlmv3_train_baseline.py first.")

    id_to_label, _label_to_id = load_label_map(model_dir)
    processor = LayoutLMv3Processor.from_pretrained(model_dir, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)
    model.eval()

    records: list[EvaluationRecord] = []
    prediction_details = []
    for idx, example in enumerate(eval_examples, start=1):
        prediction, inference_ms = predict_example(
            example,
            processor,
            model,
            id_to_label,
            max_length=args.max_length,
        )
        ground_truth = {field: str(example.get("ground_truth", {}).get(field) or "") for field in FIELDS}
        required_human_correction = any(
            not values_equal(field, ground_truth.get(field, ""), prediction.get(field, ""))
            for field in FIELDS
        )
        records.append(
            EvaluationRecord(
                document_id=example["document_id"],
                method="layoutlmv3",
                ground_truth=ground_truth,
                prediction=prediction,
                inference_ms=inference_ms,
                required_human_correction=required_human_correction,
                category=example.get("category", ""),
            )
        )
        prediction_details.append(
            {
                "document_id": example["document_id"],
                "category": example.get("category", ""),
                "method": "layoutlmv3",
                "prediction": prediction,
                "ground_truth": ground_truth,
                "inference_ms": round(inference_ms, 3),
            }
        )
        print(f"[{idx}/{len(eval_examples)}] {example['document_id']} -> {prediction}")

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ground_truth_file": "data/annotations/final_eval/manual_ground_truth.csv",
        "label_schema_version": "layoutlmv3_bio_token_labels_v1",
        "parser_version": "layoutlmv3_token_classification_baseline_v1",
        "normalization_version": NORMALIZATION_VERSION,
        "ocr_engine": "prepared_jsonl",
        "model_dir": str(model_dir.relative_to(ROOT) if model_dir.is_relative_to(ROOT) else model_dir),
        "methods": ["layoutlmv3"],
        "total_rows": len(eval_examples),
        "labeled_rows": len(eval_examples),
        "skipped_unlabeled_rows": 0,
    }
    paths = save_evaluation_outputs(
        records,
        output_dir,
        metadata=metadata,
        required_document_fields=REQUIRED_FIELDS,
    )
    details_path = output_dir / "prediction_details.json"
    with details_path.open("w", encoding="utf-8") as f:
        json.dump(prediction_details, f, indent=2, ensure_ascii=False)
    return {
        "details": str(paths["details"]),
        "summary": str(paths["summary"]),
        "prediction_details": str(details_path),
        "eval_examples": len(eval_examples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned LayoutLMv3 baseline.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    try:
        summary = evaluate(args)
    except LayoutLMv3BaselineUnavailable as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
