from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from invoices.services.anomalies import check_invoice_anomalies
from invoices.services.evaluation import (
    FIELDS,
    NORMALIZATION_VERSION,
    EvaluationRecord,
    save_evaluation_outputs,
    values_equal,
)
from invoices.services.extraction import extract_layout_aware, extract_with_regex
from invoices.services.ocr import OCRDependencyError, run_ocr_with_metadata


DEFAULT_GT = ROOT / "data" / "annotations" / "final_eval" / "manual_ground_truth.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "final_eval_evaluation"
REQUIRED_LABEL_FIELDS = ["invoice_number", "date", "vendor_name", "total_amount", "currency"]


def _is_labeled(row: dict) -> bool:
    return all((row.get(field) or "").strip() for field in REQUIRED_LABEL_FIELDS)


def _ground_truth(row: dict) -> dict:
    return {field: (row.get(field) or "").strip() for field in FIELDS}


def _clean_row_keys(row: dict) -> dict:
    return {(key or "").strip().lstrip("\ufeff"): value for key, value in row.items()}


def _row_file_path(row: dict) -> str:
    return (row.get("file_path") or row.get("local_file") or "").strip()


def _resolve_project_path(path_value: str) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def load_labeled_rows(path: Path, limit: int | None = None) -> tuple[list[dict], list[dict]]:
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [_clean_row_keys(row) for row in csv.DictReader(f)]
    labeled = [row for row in rows if _is_labeled(row)]
    skipped = [row for row in rows if not _is_labeled(row)]
    if limit is not None:
        labeled = labeled[:limit]
    return labeled, skipped


def evaluate_rows(rows: list[dict], methods: set[str], ocr_engine: str | None = None) -> tuple[list[EvaluationRecord], list[dict]]:
    records: list[EvaluationRecord] = []
    prediction_details: list[dict] = []
    seen_invoice_numbers: set[str] = set()

    for row in rows:
        row_path = _row_file_path(row)
        category = (row.get("category") or "").strip()
        local_file = _resolve_project_path(row_path)
        if local_file is None or not local_file.exists():
            prediction_details.append(
                {
                    "document_id": row_path,
                    "category": category,
                    "error": f"file not found: {local_file or '<missing file_path>'}",
                }
            )
            continue

        try:
            ocr_started = time.perf_counter()
            tokens, ocr_metadata = run_ocr_with_metadata(local_file, engine=ocr_engine)
            ocr_ms = (time.perf_counter() - ocr_started) * 1000
        except OCRDependencyError as exc:
            prediction_details.append(
                {
                    "document_id": row_path,
                    "category": category,
                    "error": str(exc),
                }
            )
            continue

        extractors = []
        if "regex" in methods:
            extractors.append(("regex", extract_with_regex))
        if "layout_aware" in methods:
            extractors.append(("layout_aware", extract_layout_aware))

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
            gt = _ground_truth(row)
            required_correction = bool(anomalies) or any(
                not values_equal(field, gt.get(field, ""), output.fields.get(field, ""))
                for field in FIELDS
            )
            records.append(
                EvaluationRecord(
                    document_id=row_path,
                    method=method_name,
                    ground_truth=gt,
                    prediction=output.fields,
                    inference_ms=ocr_ms + extraction_ms,
                    required_human_correction=required_correction,
                    category=category,
                )
            )
            prediction_details.append(
                {
                    "document_id": row_path,
                    "category": category,
                    "method": method_name,
                    "ocr_ms": round(ocr_ms, 3),
                    "extraction_ms": round(extraction_ms, 3),
                    "prediction": output.fields,
                    "confidences": output.confidences,
                    "ocr_metadata": ocr_metadata,
                    "evidence": output.evidence,
                    "anomalies": [anomaly.to_dict() for anomaly in anomalies],
                }
            )

        if row.get("invoice_number"):
            seen_invoice_numbers.add(row["invoice_number"])

    return records, prediction_details


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the final labeled invoice/receipt dataset.")
    parser.add_argument("--ground-truth", default=str(DEFAULT_GT), help="Path to manual_ground_truth.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for evaluation outputs")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum labeled rows to evaluate")
    parser.add_argument("--ocr-engine", default=None, help="Optional OCR engine override: auto, paddleocr, easyocr, tesseract")
    parser.add_argument(
        "--methods",
        default="regex,layout_aware",
        help="Comma-separated rule-based methods: regex,layout_aware.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate labels and file paths without running OCR")
    args = parser.parse_args()

    gt_path = Path(args.ground_truth)
    if not gt_path.is_absolute():
        gt_path = ROOT / gt_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    methods = {part.strip() for part in args.methods.split(",") if part.strip()}

    labeled_rows, skipped_rows = load_labeled_rows(gt_path, args.limit)
    missing_files = []
    for row in labeled_rows:
        row_path = _row_file_path(row)
        resolved_path = _resolve_project_path(row_path)
        if resolved_path is None:
            missing_files.append("<missing file_path>")
        elif not resolved_path.exists():
            missing_files.append(str(resolved_path))
    rows_needing_manual_label = len(skipped_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    readiness = {
        "ground_truth_file": str(gt_path.relative_to(ROOT) if gt_path.is_relative_to(ROOT) else gt_path),
        "total_rows": len(labeled_rows) + len(skipped_rows),
        "labeled_rows": len(labeled_rows),
        "skipped_unlabeled_rows": len(skipped_rows),
        "rows_needing_manual_label": rows_needing_manual_label,
        "required_label_fields": REQUIRED_LABEL_FIELDS,
        "missing_labeled_files": missing_files,
        "methods": sorted(methods),
        "ready_to_run": bool(labeled_rows) and not missing_files,
    }
    readiness_path = output_dir / "evaluation_readiness.json"
    with readiness_path.open("w", encoding="utf-8") as f:
        json.dump(readiness, f, indent=2, ensure_ascii=False)

    if args.dry_run:
        print(json.dumps(readiness, indent=2, ensure_ascii=False))
        print(f"Wrote readiness report to {readiness_path}")
        return

    if missing_files:
        raise FileNotFoundError(f"{len(missing_files)} labeled file(s) are missing. See {readiness_path}.")
    if not labeled_rows:
        raise SystemExit("No complete labeled rows found. Fill data/annotations/final_eval/manual_ground_truth.csv first.")

    records, prediction_details = evaluate_rows(labeled_rows, methods, ocr_engine=args.ocr_engine)
    if not records:
        errors_path = output_dir / "evaluation_errors.json"
        with errors_path.open("w", encoding="utf-8") as f:
            json.dump(prediction_details, f, indent=2, ensure_ascii=False)
        raise SystemExit(
            "Evaluation produced no method records. "
            f"Wrote OCR/error details to {errors_path}; existing metric files were not overwritten."
        )

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ground_truth_file": str(gt_path.relative_to(ROOT) if gt_path.is_relative_to(ROOT) else gt_path),
        "label_schema_version": "final_eval_manual_ground_truth_v1",
        "parser_version": "regex_layout_aware_final_eval_v1",
        "normalization_version": NORMALIZATION_VERSION,
        "ocr_engine": args.ocr_engine or "auto",
        "methods": sorted(methods),
        "total_rows": len(labeled_rows) + len(skipped_rows),
        "labeled_rows": len(labeled_rows),
        "skipped_unlabeled_rows": len(skipped_rows),
    }
    paths = save_evaluation_outputs(
        records,
        output_dir,
        metadata=metadata,
        required_document_fields=REQUIRED_LABEL_FIELDS,
    )

    details_path = output_dir / "prediction_details.json"
    with details_path.open("w", encoding="utf-8") as f:
        json.dump(prediction_details, f, indent=2, ensure_ascii=False)

    print(f"Wrote evaluation details to {paths['details']}")
    print(f"Wrote method summary to {paths['summary']}")
    print(f"Wrote prediction details to {details_path}")


if __name__ == "__main__":
    main()
