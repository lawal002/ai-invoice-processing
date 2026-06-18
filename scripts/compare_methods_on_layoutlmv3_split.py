from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from invoices.services.anomalies import check_invoice_anomalies  # noqa: E402
from invoices.services.evaluation import (  # noqa: E402
    FIELDS,
    NORMALIZATION_VERSION,
    EvaluationRecord,
    save_evaluation_outputs,
    values_equal,
)
from invoices.services.extraction import extract_layout_aware, extract_with_regex  # noqa: E402
from invoices.services.ocr import OCRDependencyError, run_ocr_with_metadata  # noqa: E402


DEFAULT_DATASET_DIR = ROOT / "data" / "annotations" / "layoutlmv3"
DEFAULT_LAYOUTLMV3_RESULTS = ROOT / "results" / "layoutlmv3_baseline_v2_decoder" / "prediction_details.json"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "final_method_comparison"
REQUIRED_FIELDS = ["invoice_number", "date", "vendor_name", "total_amount", "currency"]


def rel_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_layoutlmv3_predictions(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"LayoutLMv3 prediction details not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return {row["document_id"]: row for row in rows}


def ground_truth_from_example(example: dict) -> dict[str, str]:
    return {field: str(example.get("ground_truth", {}).get(field) or "") for field in FIELDS}


def method_needs_correction(ground_truth: dict, prediction: dict, anomalies: list | None = None) -> bool:
    return bool(anomalies) or any(
        not values_equal(field, ground_truth.get(field, ""), prediction.get(field, ""))
        for field in FIELDS
    )


def evaluate_rule_methods(examples: list[dict], *, ocr_engine: str) -> tuple[list[EvaluationRecord], list[dict]]:
    records: list[EvaluationRecord] = []
    details: list[dict] = []
    seen_invoice_numbers: set[str] = set()

    extractors = [("regex", extract_with_regex), ("layout_aware", extract_layout_aware)]
    for index, example in enumerate(examples, start=1):
        document_id = example["document_id"]
        image_path = ROOT / example["image_path"]
        category = example.get("category", "")
        ground_truth = ground_truth_from_example(example)

        try:
            ocr_started = time.perf_counter()
            tokens, ocr_metadata = run_ocr_with_metadata(image_path, engine=ocr_engine)
            ocr_ms = (time.perf_counter() - ocr_started) * 1000
        except OCRDependencyError as exc:
            details.append(
                {
                    "document_id": document_id,
                    "category": category,
                    "error": str(exc),
                }
            )
            continue

        for method_name, extractor in extractors:
            started = time.perf_counter()
            output = extractor(tokens)
            output.evidence["_ocr_run"] = ocr_metadata
            extraction_ms = (time.perf_counter() - started) * 1000
            anomalies = check_invoice_anomalies(
                output.fields,
                output.confidences,
                seen_invoice_numbers,
                evidence=output.evidence,
            )
            records.append(
                EvaluationRecord(
                    document_id=document_id,
                    method=method_name,
                    ground_truth=ground_truth,
                    prediction=output.fields,
                    inference_ms=ocr_ms + extraction_ms,
                    required_human_correction=method_needs_correction(
                        ground_truth,
                        output.fields,
                        anomalies,
                    ),
                    category=category,
                )
            )
            details.append(
                {
                    "document_id": document_id,
                    "category": category,
                    "method": method_name,
                    "ocr_ms": round(ocr_ms, 3),
                    "extraction_ms": round(extraction_ms, 3),
                    "prediction": output.fields,
                    "confidences": output.confidences,
                    "evidence": output.evidence,
                    "anomalies": [anomaly.to_dict() for anomaly in anomalies],
                }
            )

        if ground_truth.get("invoice_number"):
            seen_invoice_numbers.add(ground_truth["invoice_number"])
        print(f"[{index}/{len(examples)}] evaluated regex/layout-aware for {document_id}")

    return records, details


def layoutlmv3_records(examples: list[dict], prediction_rows: dict[str, dict]) -> tuple[list[EvaluationRecord], list[dict]]:
    records: list[EvaluationRecord] = []
    details: list[dict] = []
    for example in examples:
        document_id = example["document_id"]
        category = example.get("category", "")
        ground_truth = ground_truth_from_example(example)
        row = prediction_rows.get(document_id)
        if row is None:
            details.append(
                {
                    "document_id": document_id,
                    "category": category,
                    "method": "layoutlmv3",
                    "error": "missing_layoutlmv3_prediction",
                }
            )
            continue
        prediction = {field: str(row.get("prediction", {}).get(field) or "") for field in FIELDS}
        records.append(
            EvaluationRecord(
                document_id=document_id,
                method="layoutlmv3",
                ground_truth=ground_truth,
                prediction=prediction,
                inference_ms=float(row.get("inference_ms") or 0.0),
                required_human_correction=method_needs_correction(ground_truth, prediction),
                category=category,
            )
        )
        details.append(
            {
                "document_id": document_id,
                "category": category,
                "method": "layoutlmv3",
                "prediction": prediction,
                "ground_truth": ground_truth,
                "inference_ms": round(float(row.get("inference_ms") or 0.0), 3),
            }
        )
    return records, details


def compare(args) -> dict[str, str]:
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = ROOT / dataset_dir
    layoutlmv3_predictions = Path(args.layoutlmv3_predictions)
    if not layoutlmv3_predictions.is_absolute():
        layoutlmv3_predictions = ROOT / layoutlmv3_predictions
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = read_jsonl(dataset_dir / "eval.jsonl")
    prediction_rows = load_layoutlmv3_predictions(layoutlmv3_predictions)

    rule_records, rule_details = evaluate_rule_methods(examples, ocr_engine=args.ocr_engine)
    lm_records, lm_details = layoutlmv3_records(examples, prediction_rows)
    records = rule_records + lm_records
    details = rule_details + lm_details

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "comparison_name": "same_layoutlmv3_eval_split",
        "dataset_dir": rel_path(dataset_dir),
        "eval_jsonl": rel_path(dataset_dir / "eval.jsonl"),
        "ground_truth_file": "data/annotations/final_eval/manual_ground_truth.csv",
        "layoutlmv3_predictions": rel_path(layoutlmv3_predictions),
        "label_schema_version": "layoutlmv3_eval_split_v1",
        "parser_version": "regex_layout_aware_layoutlmv3_same_split_v1",
        "normalization_version": NORMALIZATION_VERSION,
        "ocr_engine": args.ocr_engine,
        "methods": ["regex", "layout_aware", "layoutlmv3"],
        "total_rows": len(examples),
        "labeled_rows": len(examples),
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
        json.dump(details, f, indent=2, ensure_ascii=False)
    return {
        "details": str(paths["details"]),
        "summary": str(paths["summary"]),
        "prediction_details": str(details_path),
        "eval_examples": str(len(examples)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare regex, layout-aware, and LayoutLMv3 on the same eval split.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--layoutlmv3-predictions", default=str(DEFAULT_LAYOUTLMV3_RESULTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--ocr-engine", default="paddleocr")
    args = parser.parse_args()
    print(json.dumps(compare(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
