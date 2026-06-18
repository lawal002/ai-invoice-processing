from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from invoices.services.anomalies import check_invoice_anomalies
from invoices.services.extraction import extract_layout_aware
from invoices.services.ocr import OCRDependencyError, run_ocr_with_metadata


FINAL_EVAL_DIR = ROOT / "data" / "samples" / "final_eval"
ANNOTATION_DIR = ROOT / "data" / "annotations" / "final_eval"
OUTPUT_DIR = ROOT / "results" / "final_eval_labeling"
MANUAL_GT_FILE = ANNOTATION_DIR / "manual_ground_truth.csv"
AUTO_PREDICTIONS_FILE = ANNOTATION_DIR / "auto_predictions.csv"
REVIEW_QUEUE_FILE = ANNOTATION_DIR / "review_queue.csv"
DATASET_SOURCES_FILE = ANNOTATION_DIR / "dataset_sources.csv"
SUMMARY_FILE = OUTPUT_DIR / "labeling_summary.json"

PHASE3_GT_FILE = ROOT / "data" / "annotations" / "phase3" / "manual_ground_truth.csv"
FATURA_ANNOTATION_DIR = (
    ROOT
    / "data"
    / "raw"
    / "fatura"
    / "FATURA"
    / "invoices_dataset_final"
    / "Annotations"
    / "Original_Format"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
GROUND_TRUTH_FIELDS = [
    "file_path",
    "category",
    "invoice_number",
    "date",
    "vendor_name",
    "subtotal",
    "tax_amount",
    "total_amount",
    "currency",
    "vat_rate",
    "amount_in_words",
    "notes",
]
VALUE_FIELDS = [
    "invoice_number",
    "date",
    "vendor_name",
    "subtotal",
    "tax_amount",
    "total_amount",
    "currency",
    "vat_rate",
    "amount_in_words",
]
REQUIRED_FIELDS = ["invoice_number", "date", "vendor_name", "total_amount", "currency"]
AUTO_FIELDNAMES = GROUND_TRUTH_FIELDS + [
    "label_source",
    "review_status",
    "review_reason",
    "confidence_summary",
    "anomaly_codes",
    "ocr_engine",
    "ocr_language",
    "ocr_token_count",
    "ocr_average_confidence",
    "ocr_cache_hit",
    "prelabel_ms",
]
REVIEW_FIELDNAMES = [
    "priority",
    "file_path",
    "category",
    "review_status",
    "review_reason",
    "invoice_number",
    "date",
    "vendor_name",
    "subtotal",
    "tax_amount",
    "total_amount",
    "currency",
    "vat_rate",
    "amount_in_words",
    "label_source",
    "confidence_summary",
    "anomaly_codes",
    "notes",
]
SOURCE_FIELDNAMES = [
    "file_path",
    "category",
    "source_dataset",
    "source_note",
    "label_source",
    "license_or_access",
]


def rel_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def normalize_path(value: str) -> str:
    return value.replace("\\", "/").strip().lower()


def clean_row_keys(row: dict[str, Any]) -> dict[str, str]:
    return {(key or "").strip().lstrip("\ufeff"): (value or "").strip() for key, value in row.items()}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [clean_row_keys(row) for row in csv.DictReader(f)]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def scan_images() -> list[dict[str, str]]:
    if not FINAL_EVAL_DIR.exists():
        return []
    rows: list[dict[str, str]] = []
    for path in sorted(FINAL_EVAL_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem.lower().endswith("_capped"):
            continue
        try:
            category = path.relative_to(FINAL_EVAL_DIR).parts[0]
        except IndexError:
            category = "uncategorized"
        rows.append({"file_path": rel_path(path), "category": category})
    return rows


def rows_by_path(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    output = {}
    for row in rows:
        file_path = (row.get("file_path") or row.get("local_file") or "").strip()
        if file_path:
            output[normalize_path(file_path)] = row
    return output


def phase3_rows_by_name(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    output = {}
    for row in rows:
        file_path = (row.get("file_path") or "").strip()
        if file_path:
            output[Path(file_path).name.lower()] = row
    return output


def empty_gt_row(file_path: str, category: str) -> dict[str, str]:
    row = {field: "" for field in GROUND_TRUTH_FIELDS}
    row["file_path"] = file_path
    row["category"] = category
    return row


def merge_value_fields(base: dict[str, str], values: dict[str, str]) -> dict[str, str]:
    row = dict(base)
    for field in VALUE_FIELDS:
        if values.get(field):
            row[field] = str(values[field]).strip()
    if values.get("notes"):
        row["notes"] = str(values["notes"]).strip()
    return row


def fill_blank_value_fields(base: dict[str, str], values: dict[str, str]) -> dict[str, str]:
    row = dict(base)
    for field in VALUE_FIELDS:
        if not (row.get(field) or "").strip() and values.get(field):
            row[field] = str(values[field]).strip()
    if not (row.get("notes") or "").strip() and values.get("notes"):
        row["notes"] = str(values["notes"]).strip()
    return row


def extract_amount(text: str) -> str:
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text or "")
    if not matches:
        return ""
    value = matches[-1].replace(",", "")
    try:
        return f"{float(value):.2f}"
    except ValueError:
        return value


def extract_currency(text: str) -> str:
    upper = (text or "").upper()
    if "EUR" in upper:
        return "EUR"
    if "USD" in upper:
        return "USD"
    if "RM" in upper or "MYR" in upper:
        return "RM"
    if "CNY" in upper or "RMB" in upper or "￥" in text or "元" in text:
        return "CNY"
    if "NGN" in upper or "₦" in text:
        return "NGN"
    return ""


def extract_date(text: str) -> str:
    value = (text or "").strip()
    value = re.sub(r"^\s*(date|invoice date|issued date)\s*[:：-]?\s*", "", value, flags=re.I)
    return value.strip()


def extract_vat_rate(text: str) -> str:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text or "")
    return f"{match.group(1)}%" if match else ""


def load_fatura_annotation(image_name: str) -> dict[str, str]:
    ann_path = FATURA_ANNOTATION_DIR / f"{Path(image_name).stem}.json"
    if not ann_path.exists():
        return {}
    try:
        with ann_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    seller_name = ""
    for key in ("SELLER_NAME", "SELLER", "TITLE"):
        text = str((payload.get(key) or {}).get("text") or "").strip()
        if text and key != "TITLE":
            seller_name = text.splitlines()[0].strip()
            break

    total_text = str((payload.get("TOTAL") or {}).get("text") or "")
    subtotal_text = str((payload.get("SUB_TOTAL") or {}).get("text") or "")
    tax_text = str((payload.get("TAX") or {}).get("text") or "")
    values = {
        "date": extract_date(str((payload.get("DATE") or {}).get("text") or "")),
        "vendor_name": seller_name,
        "subtotal": extract_amount(subtotal_text),
        "tax_amount": extract_amount(tax_text),
        "total_amount": extract_amount(total_text),
        "currency": extract_currency(" ".join([total_text, subtotal_text, tax_text])),
        "vat_rate": extract_vat_rate(tax_text),
        "amount_in_words": re.sub(
            r"^\s*total in words\s*[:：]?\s*",
            "",
            str((payload.get("TOTAL_WORDS") or {}).get("text") or ""),
            flags=re.I,
        ).replace("\n", " ").strip(),
    }
    return {key: value for key, value in values.items() if value}


def is_complete(row: dict[str, str]) -> bool:
    return all((row.get(field) or "").strip() for field in REQUIRED_FIELDS)


def confidence_summary(confidences: dict[str, float]) -> str:
    parts = []
    for field in REQUIRED_FIELDS:
        if field in confidences:
            parts.append(f"{field}={confidences[field]:.3f}")
    return ";".join(parts)


def review_decision(row: dict[str, str], confidences: dict[str, float], anomaly_codes: list[str], source: str) -> tuple[str, str]:
    missing = [field for field in REQUIRED_FIELDS if not (row.get(field) or "").strip()]
    low_conf = [field for field in REQUIRED_FIELDS if confidences.get(field, 1.0) < 0.55]
    reasons = []
    if missing:
        reasons.append("missing:" + ",".join(missing))
    if low_conf:
        reasons.append("low_confidence:" + ",".join(low_conf))
    if anomaly_codes:
        reasons.append("anomalies:" + ",".join(anomaly_codes))
    if source == "fatura_json" and not row.get("invoice_number"):
        reasons.append("fatura_json_has_no_invoice_number")
    if not reasons:
        return "auto_labeled_review_optional", ""
    return "needs_review", "; ".join(reasons)


def prelabel_with_extractor(file_path: Path, ocr_engine: str | None) -> tuple[dict[str, str], dict[str, Any]]:
    started = time.perf_counter()
    tokens, ocr_metadata = run_ocr_with_metadata(file_path, engine=ocr_engine)
    extracted = extract_layout_aware(tokens)
    anomalies = check_invoice_anomalies(extracted.fields, extracted.confidences, set(), evidence=extracted.evidence)
    elapsed_ms = (time.perf_counter() - started) * 1000
    metadata = {
        "confidences": extracted.confidences,
        "anomaly_codes": [anomaly.code for anomaly in anomalies],
        "ocr_metadata": ocr_metadata,
        "prelabel_ms": elapsed_ms,
    }
    return {field: extracted.fields.get(field, "") for field in VALUE_FIELDS}, metadata


def source_row(file_path: str, category: str, label_source: str) -> dict[str, str]:
    if category == "fatura":
        source_dataset = "FATURA Invoice Dataset"
        license_note = "local downloaded dataset; use according to dataset terms"
    elif category == "vatid":
        source_dataset = "VATID Chinese VAT-style Dataset"
        license_note = "local downloaded dataset; use according to dataset terms"
    elif category == "chinese_invoices":
        source_dataset = "Chinese invoice/taxi samples"
        license_note = "local project sample; document source in report"
    elif category == "clean_invoices":
        source_dataset = "Synthetic starter invoices"
        license_note = "project-generated"
    else:
        source_dataset = "Local project invoice sample"
        license_note = "local project sample; document source in report"
    return {
        "file_path": file_path,
        "category": category,
        "source_dataset": source_dataset,
        "source_note": f"final_eval/{category}",
        "label_source": label_source,
        "license_or_access": license_note,
    }


def build_labels(*, ocr_engine: str | None, no_ocr: bool, limit: int | None) -> dict[str, Any]:
    scanned_rows = scan_images()
    if limit is not None:
        scanned_rows = scanned_rows[:limit]

    existing_manual = rows_by_path(read_csv(MANUAL_GT_FILE))
    phase3_by_name = phase3_rows_by_name(read_csv(PHASE3_GT_FILE))

    manual_rows: list[dict[str, str]] = []
    auto_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, str]] = []
    category_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    errors: list[dict[str, str]] = []

    for index, scanned in enumerate(scanned_rows, start=1):
        file_path = scanned["file_path"]
        category = scanned["category"]
        category_counts[category] += 1
        base = empty_gt_row(file_path, category)
        existing = existing_manual.get(normalize_path(file_path), {})
        label_source = ""
        metadata: dict[str, Any] = {
            "confidences": {},
            "anomaly_codes": [],
            "ocr_metadata": {},
            "prelabel_ms": 0.0,
        }

        if existing:
            row = {field: existing.get(field, base.get(field, "")) for field in GROUND_TRUTH_FIELDS}
            label_source = "manual_existing"
        else:
            phase3 = phase3_by_name.get(Path(file_path).name.lower())
            if phase3 and is_complete(phase3):
                row = merge_value_fields(base, phase3)
                label_source = "phase3_manual_reuse"
            elif category == "fatura":
                fatura_values = load_fatura_annotation(Path(file_path).name)
                if fatura_values:
                    row = merge_value_fields(base, fatura_values)
                    label_source = "fatura_json"
                else:
                    row = base
            else:
                row = base

        if not no_ocr and not is_complete(row):
            try:
                prediction, metadata = prelabel_with_extractor(ROOT / file_path, ocr_engine=ocr_engine)
                row = fill_blank_value_fields(row, prediction)
                label_source = f"{label_source}+layout_aware" if label_source else "layout_aware"
            except OCRDependencyError as exc:
                errors.append({"file_path": file_path, "error": str(exc)})
                label_source = label_source or "unlabeled_ocr_error"
        elif no_ocr and not label_source:
            label_source = "unlabeled_no_ocr"

        status, reason = review_decision(
            row,
            metadata.get("confidences", {}),
            metadata.get("anomaly_codes", []),
            label_source,
        )
        source_counts[label_source] += 1
        row["notes"] = row.get("notes", "")
        manual_rows.append({field: row.get(field, "") for field in GROUND_TRUTH_FIELDS})

        ocr_metadata = metadata.get("ocr_metadata", {}) or {}
        auto_row = dict(row)
        auto_row.update(
            {
                "label_source": label_source,
                "review_status": status,
                "review_reason": reason,
                "confidence_summary": confidence_summary(metadata.get("confidences", {})),
                "anomaly_codes": ";".join(metadata.get("anomaly_codes", [])),
                "ocr_engine": ocr_metadata.get("selected_engine", ""),
                "ocr_language": ",".join(ocr_metadata.get("languages", []) or []),
                "ocr_token_count": ocr_metadata.get("token_count", ""),
                "ocr_average_confidence": ocr_metadata.get("average_confidence", ""),
                "ocr_cache_hit": (ocr_metadata.get("cache") or {}).get("hit", ""),
                "prelabel_ms": f"{metadata.get('prelabel_ms', 0.0):.1f}",
            }
        )
        auto_rows.append(auto_row)
        source_rows.append(source_row(file_path, category, label_source))
        if status == "needs_review":
            review_rows.append({"priority": len(review_rows) + 1, **auto_row})
        print(f"[{index}/{len(scanned_rows)}] {file_path} -> {label_source} ({status})")

    write_csv(MANUAL_GT_FILE, GROUND_TRUTH_FIELDS, manual_rows)
    write_csv(AUTO_PREDICTIONS_FILE, AUTO_FIELDNAMES, auto_rows)
    write_csv(REVIEW_QUEUE_FILE, REVIEW_FIELDNAMES, review_rows)
    write_csv(DATASET_SOURCES_FILE, SOURCE_FIELDNAMES, source_rows)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "final_eval_dir": rel_path(FINAL_EVAL_DIR),
        "manual_ground_truth_file": rel_path(MANUAL_GT_FILE),
        "auto_predictions_file": rel_path(AUTO_PREDICTIONS_FILE),
        "review_queue_file": rel_path(REVIEW_QUEUE_FILE),
        "dataset_sources_file": rel_path(DATASET_SOURCES_FILE),
        "total_images": len(scanned_rows),
        "category_counts": dict(sorted(category_counts.items())),
        "label_source_counts": dict(sorted(source_counts.items())),
        "rows_needing_review": len(review_rows),
        "rows_review_optional": len(scanned_rows) - len(review_rows),
        "ocr_errors": errors,
        "excluded_rule": "Files ending in _capped are ignored to avoid duplicate preprocessed images.",
        "next_action": "Open review_queue.csv and correct only rows marked needs_review. The full draft labels are in manual_ground_truth.csv.",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_FILE.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare final evaluation labels with auto pre-labeling.")
    parser.add_argument("--ocr-engine", default="paddleocr", help="OCR engine for pre-labeling: paddleocr, auto, easyocr, tesseract")
    parser.add_argument("--no-ocr", action="store_true", help="Only reuse existing/FATURA labels; do not run OCR.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for testing the script.")
    args = parser.parse_args()

    summary = build_labels(ocr_engine=args.ocr_engine, no_ocr=args.no_ocr, limit=args.limit)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
