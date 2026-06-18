from __future__ import annotations

import csv
import json
import re
import string
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from difflib import SequenceMatcher
from pathlib import Path


FIELDS = ["invoice_number", "date", "vendor_name", "subtotal", "tax_amount", "total_amount", "currency"]
AMOUNT_FIELDS = {"subtotal", "tax_amount", "total_amount"}
DEFAULT_DOCUMENT_REQUIRED_FIELDS = ["invoice_number", "date", "vendor_name", "total_amount", "currency"]
EVALUATION_OUTPUT_VERSION = "phase3_evaluation_v2"
NORMALIZATION_VERSION = "field_normalization_v1"

DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%d-%m-%y",
    "%d/%m/%y",
    "%d.%m.%y",
    "%m/%d/%y",
    "%m-%d-%y",
    "%d-%b-%Y",
    "%d-%B-%Y",
    "%d-%b-%y",
    "%d-%B-%y",
]

SYMBOL_CURRENCIES = {
    "$": "USD",
    "\u00a5": "CNY",
    "\uffe5": "CNY",
    "\u5143": "CNY",
}

CODE_CURRENCIES = {
    "RM": "RM",
    "MYR": "RM",
    "CNY": "CNY",
    "RMB": "CNY",
    "USD": "USD",
    "US$": "USD",
    "BDT": "BDT",
    "EUR": "EUR",
    "NGN": "NGN",
}


@dataclass
class EvaluationRecord:
    document_id: str
    method: str
    ground_truth: dict
    prediction: dict
    inference_ms: float = 0.0
    required_human_correction: bool = False
    category: str = ""


def _text(value) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def _normalize_amount(value) -> str:
    text = _text(value)
    if not text:
        return ""
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not match:
        return text.lower()
    numeric = match.group(0).replace(",", "")
    try:
        amount = Decimal(numeric).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return text.lower()
    return f"{amount:.2f}"


def _normalize_currency(value) -> str:
    text = _text(value).upper()
    if not text:
        return ""
    for symbol, code in SYMBOL_CURRENCIES.items():
        if symbol in text:
            return code
    compact = re.sub(r"[^A-Z$]", "", text)
    return CODE_CURRENCIES.get(compact, compact)


def _normalize_date(value) -> str:
    text = _text(value)
    if not text:
        return ""
    match = re.search(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", text)
    candidate = match.group(0) if match else text
    candidate = candidate.replace(".", "/")
    for fmt in DATE_FORMATS:
        try:
            parsed = datetime.strptime(candidate, fmt)
        except ValueError:
            continue
        return parsed.strftime("%Y-%m-%d")
    return re.sub(r"\s+", "", text).lower()


def _normalize_vendor(value) -> str:
    text = _text(value).casefold()
    text = re.sub(r"[^\w\u3400-\u9fff&]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_invoice_number(value) -> str:
    text = _text(value).upper()
    if not text:
        return ""
    text = re.sub(r"\s+", "", text)
    return text.strip(string.punctuation + "\uff1a\uff0c\u3002\uff1b\u3001")


def normalize(value, field: str | None = None) -> str:
    if field in AMOUNT_FIELDS:
        return _normalize_amount(value)
    if field == "currency":
        return _normalize_currency(value)
    if field == "date":
        return _normalize_date(value)
    if field == "vendor_name":
        return _normalize_vendor(value)
    if field == "invoice_number":
        return _normalize_invoice_number(value)
    return _text(value).casefold().replace(",", "")


def values_equal(field: str, expected, predicted) -> bool:
    expected_norm = normalize(expected, field)
    predicted_norm = normalize(predicted, field)
    if expected_norm == predicted_norm:
        return True
    if field == "vendor_name" and expected_norm and predicted_norm:
        return SequenceMatcher(None, expected_norm, predicted_norm).ratio() >= 0.92
    return False


def _record_category(record: EvaluationRecord) -> str:
    if record.category:
        return record.category
    parts = record.document_id.replace("\\", "/").split("/")
    if "phase3_eval" in parts:
        idx = parts.index("phase3_eval")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "uncategorized"


def _field_has_ground_truth(record: EvaluationRecord, field: str) -> bool:
    return bool(normalize(record.ground_truth.get(field), field))


def _field_is_correct(record: EvaluationRecord, field: str) -> bool:
    gt = normalize(record.ground_truth.get(field), field)
    pred = normalize(record.prediction.get(field), field)
    return bool(gt) and gt == pred


def _field_stats(records: list[EvaluationRecord]) -> tuple[dict, int, int]:
    field_stats = {}
    total_correct = 0
    total_fields = 0

    for field in FIELDS:
        correct = 0
        predicted = 0
        actual = 0
        for record in records:
            gt = normalize(record.ground_truth.get(field), field)
            pred = normalize(record.prediction.get(field), field)
            if gt:
                actual += 1
                total_fields += 1
            if pred:
                predicted += 1
            if gt and pred == gt:
                correct += 1
                total_correct += 1

        precision = correct / predicted if predicted else 0.0
        recall = correct / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        field_stats[field] = {
            "accuracy": correct / actual if actual else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "correct": correct,
            "actual": actual,
            "predicted": predicted,
        }

    return field_stats, total_correct, total_fields


def _category_stats(records: list[EvaluationRecord]) -> dict:
    by_category: dict[str, list[EvaluationRecord]] = {}
    for record in records:
        by_category.setdefault(_record_category(record), []).append(record)

    stats = {}
    for category, category_records in sorted(by_category.items()):
        correct = 0
        actual = 0
        for record in category_records:
            for field in FIELDS:
                if _field_has_ground_truth(record, field):
                    actual += 1
                    if _field_is_correct(record, field):
                        correct += 1
        stats[category] = {
            "documents": len(category_records),
            "correct_fields": correct,
            "labeled_fields": actual,
            "accuracy": correct / actual if actual else 0.0,
        }
    return stats


def _document_pass_rate(records: list[EvaluationRecord], required_fields: list[str]) -> dict:
    passed = 0
    failed_documents = []

    for record in records:
        failed_fields = [
            field
            for field in required_fields
            if not _field_has_ground_truth(record, field) or not _field_is_correct(record, field)
        ]
        if failed_fields:
            failed_documents.append(
                {
                    "document_id": record.document_id,
                    "category": _record_category(record),
                    "failed_fields": failed_fields,
                }
            )
        else:
            passed += 1

    total = len(records)
    return {
        "passed_documents": passed,
        "total_documents": total,
        "pass_rate": passed / total if total else 0.0,
        "failed_documents": failed_documents,
    }


def evaluate_records(
    records: list[EvaluationRecord],
    metadata: dict | None = None,
    required_document_fields: list[str] | None = None,
) -> dict:
    by_method: dict[str, list[EvaluationRecord]] = {}
    for record in records:
        by_method.setdefault(record.method, []).append(record)

    required_fields = required_document_fields or DEFAULT_DOCUMENT_REQUIRED_FIELDS
    summary = {
        "_metadata": {
            "evaluation_output_version": EVALUATION_OUTPUT_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "required_document_fields": required_fields,
            **(metadata or {}),
        }
    }

    for method, method_records in by_method.items():
        field_stats, total_correct, total_fields = _field_stats(method_records)
        summary[method] = {
            "field_stats": field_stats,
            "category_stats": _category_stats(method_records),
            "document_pass_rate": _document_pass_rate(method_records, required_fields),
            "overall_accuracy": total_correct / total_fields if total_fields else 0.0,
            "average_inference_ms": sum(record.inference_ms for record in method_records) / len(method_records),
            "human_correction_rate": sum(1 for record in method_records if record.required_human_correction) / len(method_records),
        }
    return summary


def save_evaluation_outputs(
    records: list[EvaluationRecord],
    output_dir: str | Path,
    metadata: dict | None = None,
    required_document_fields: list[str] | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "evaluation_details.csv"
    summary_path = output_dir / "method_comparison_summary.json"

    with details_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "document_id",
                "category",
                "method",
                "field",
                "ground_truth",
                "prediction",
                "normalized_ground_truth",
                "normalized_prediction",
                "correct",
                "inference_ms",
                "required_human_correction",
            ],
        )
        writer.writeheader()
        for record in records:
            for field in FIELDS:
                gt = record.ground_truth.get(field, "")
                pred = record.prediction.get(field, "")
                writer.writerow(
                    {
                        "document_id": record.document_id,
                        "category": _record_category(record),
                        "method": record.method,
                        "field": field,
                        "ground_truth": gt,
                        "prediction": pred,
                        "normalized_ground_truth": normalize(gt, field),
                        "normalized_prediction": normalize(pred, field),
                        "correct": values_equal(field, gt, pred),
                        "inference_ms": round(record.inference_ms, 3),
                        "required_human_correction": record.required_human_correction,
                    }
                )

    summary = evaluate_records(records, metadata=metadata, required_document_fields=required_document_fields)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return {"details": details_path, "summary": summary_path}
