from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from invoices.services.evaluation import FIELDS
from invoices.services.layoutlmv3_baseline import LABEL_LIST, build_layoutlmv3_example
from invoices.services.ocr import OCRDependencyError, run_ocr_with_metadata


DEFAULT_GT = ROOT / "data" / "annotations" / "final_eval" / "manual_ground_truth.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "annotations" / "layoutlmv3"
DEFAULT_COVERAGE_OUTPUT_DIR = ROOT / "results" / "layoutlmv3_label_coverage"
REQUIRED_FIELDS = ["invoice_number", "date", "vendor_name", "total_amount", "currency"]
MIN_LABEL_COVERAGE = {
    "invoice_number": 0.70,
    "date": 0.70,
    "vendor_name": 0.50,
    "subtotal": 0.45,
    "tax_amount": 0.45,
    "total_amount": 0.75,
    "currency": 0.70,
}


def clean_row(row: dict) -> dict[str, str]:
    return {(key or "").strip().lstrip("\ufeff"): (value or "").strip() for key, value in row.items()}


def read_ground_truth(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [clean_row(row) for row in csv.DictReader(f)]


def is_labeled(row: dict[str, str]) -> bool:
    return all(row.get(field, "").strip() for field in REQUIRED_FIELDS)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def rel_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def stratified_split(rows: list[dict[str, str]], train_ratio: float, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_category[row.get("category", "uncategorized")].append(row)

    splits: dict[str, str] = {}
    for category, category_rows in sorted(by_category.items()):
        shuffled = list(category_rows)
        rng.shuffle(shuffled)
        if len(shuffled) <= 1:
            train_count = len(shuffled)
        else:
            train_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * train_ratio)))
        for idx, row in enumerate(shuffled):
            splits[row["file_path"]] = "train" if idx < train_count else "eval"
    return splits


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def label_counts(examples: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for example in examples:
        counts.update(example.get("labels", []))
    return dict(sorted(counts.items()))


def flatten_coverage(example: dict) -> list[dict]:
    rows = []
    coverage = example.get("label_coverage") or {}
    for field in FIELDS:
        field_report = coverage.get(field) or {}
        rows.append(
            {
                "file_path": example.get("document_id", ""),
                "category": example.get("category", ""),
                "split": example.get("split", ""),
                "field": field,
                "ground_truth": field_report.get("ground_truth", example.get("ground_truth", {}).get(field, "")),
                "has_ground_truth": bool(field_report.get("has_ground_truth", False)),
                "matched": bool(field_report.get("matched", False)),
                "reason": field_report.get("reason", ""),
                "score": field_report.get("score", 0.0),
                "start": field_report.get("start", ""),
                "end": field_report.get("end", ""),
                "matched_text": field_report.get("matched_text", ""),
                "token_count": len(example.get("words", [])),
            }
        )
    return rows


def coverage_summary(coverage_rows: list[dict]) -> dict:
    by_field: dict[str, dict[str, int]] = {
        field: {"ground_truth_fields": 0, "matched_fields": 0, "missed_fields": 0}
        for field in FIELDS
    }
    missed_reasons: dict[str, Counter[str]] = {field: Counter() for field in FIELDS}
    for row in coverage_rows:
        field = row["field"]
        if not row["has_ground_truth"]:
            continue
        by_field[field]["ground_truth_fields"] += 1
        if row["matched"]:
            by_field[field]["matched_fields"] += 1
        else:
            by_field[field]["missed_fields"] += 1
            missed_reasons[field][row.get("reason", "") or "unknown"] += 1

    result = {}
    ready_to_train = True
    for field in FIELDS:
        totals = by_field[field]
        actual = totals["ground_truth_fields"]
        matched = totals["matched_fields"]
        coverage = matched / actual if actual else 1.0
        minimum = MIN_LABEL_COVERAGE.get(field, 0.0)
        passes = coverage >= minimum
        if not passes:
            ready_to_train = False
        result[field] = {
            **totals,
            "coverage": coverage,
            "minimum_required": minimum,
            "passes_minimum": passes,
            "missed_reasons": dict(missed_reasons[field]),
        }
    return {"ready_to_train": ready_to_train, "fields": result}


def write_coverage_reports(output_dir: Path, coverage_rows: list[dict], summary: dict) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "label_coverage_report.csv"
    json_path = output_dir / "label_coverage_report.json"
    fieldnames = [
        "file_path",
        "category",
        "split",
        "field",
        "ground_truth",
        "has_ground_truth",
        "matched",
        "reason",
        "score",
        "start",
        "end",
        "matched_text",
        "token_count",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(coverage_rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": coverage_rows}, f, indent=2, ensure_ascii=False)
    return {"coverage_csv": rel_path(csv_path), "coverage_json": rel_path(json_path)}


def prepare_dataset(
    *,
    ground_truth: Path,
    output_dir: Path,
    ocr_engine: str,
    train_ratio: float,
    seed: int,
    limit: int | None,
    coverage_output_dir: Path,
) -> dict:
    rows = [row for row in read_ground_truth(ground_truth) if is_labeled(row)]
    if limit is not None:
        rows = rows[:limit]
    splits = stratified_split(rows, train_ratio=train_ratio, seed=seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    examples_by_split: dict[str, list[dict]] = {"train": [], "eval": []}
    split_rows = []
    coverage_rows = []
    errors = []

    for index, row in enumerate(rows, start=1):
        file_path = row["file_path"]
        image_path = resolve_path(file_path)
        split = splits[file_path]
        category = row.get("category", "")
        if not image_path.exists():
            errors.append({"file_path": file_path, "error": "file_not_found"})
            continue
        try:
            tokens, ocr_metadata = run_ocr_with_metadata(image_path, engine=ocr_engine)
            with Image.open(image_path) as image:
                width, height = image.size
            example = build_layoutlmv3_example(
                document_id=file_path,
                category=category,
                image_path=rel_path(image_path),
                tokens=tokens,
                ground_truth={field: row.get(field, "") for field in FIELDS},
                image_size=(width, height),
            )
        except (OCRDependencyError, OSError) as exc:
            errors.append({"file_path": file_path, "error": str(exc)})
            continue

        example_dict = example.to_dict()
        example_dict["split"] = split
        example_dict["ocr"] = {
            "engine": ocr_metadata.get("selected_engine", ""),
            "languages": ocr_metadata.get("languages", []),
            "token_count": ocr_metadata.get("token_count", 0),
            "average_confidence": ocr_metadata.get("average_confidence", 0.0),
            "cache_hit": (ocr_metadata.get("cache") or {}).get("hit", False),
        }
        examples_by_split[split].append(example_dict)
        split_rows.append({"file_path": file_path, "category": category, "split": split})
        coverage_rows.extend(flatten_coverage(example_dict))
        print(f"[{index}/{len(rows)}] {split:<5} {file_path} labels={Counter(example.labels)}")

    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    write_jsonl(train_path, examples_by_split["train"])
    write_jsonl(eval_path, examples_by_split["eval"])

    splits_path = output_dir / "splits.csv"
    with splits_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "category", "split"])
        writer.writeheader()
        writer.writerows(split_rows)

    label_coverage_summary = coverage_summary(coverage_rows)
    coverage_outputs = write_coverage_reports(coverage_output_dir, coverage_rows, label_coverage_summary)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ground_truth": rel_path(ground_truth),
        "output_dir": rel_path(output_dir),
        "coverage_output_dir": rel_path(coverage_output_dir),
        "ocr_engine": ocr_engine,
        "train_ratio": train_ratio,
        "seed": seed,
        "label_list": LABEL_LIST,
        "total_labeled_rows": len(rows),
        "train_examples": len(examples_by_split["train"]),
        "eval_examples": len(examples_by_split["eval"]),
        "category_split_counts": {
            category: dict(Counter(item["split"] for item in split_rows if item["category"] == category))
            for category in sorted({item["category"] for item in split_rows})
        },
        "train_label_counts": label_counts(examples_by_split["train"]),
        "eval_label_counts": label_counts(examples_by_split["eval"]),
        "label_coverage": label_coverage_summary,
        "errors": errors,
        "outputs": {
            "train_jsonl": rel_path(train_path),
            "eval_jsonl": rel_path(eval_path),
            "splits_csv": rel_path(splits_path),
            "summary": rel_path(output_dir / "dataset_summary.json"),
            **coverage_outputs,
        },
    }
    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LayoutLMv3 token-classification examples from OCR + labels.")
    parser.add_argument("--ground-truth", default=str(DEFAULT_GT), help="Ground-truth CSV")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for JSONL examples")
    parser.add_argument("--ocr-engine", default="paddleocr", help="OCR engine to use: paddleocr, auto, easyocr, tesseract")
    parser.add_argument("--train-ratio", type=float, default=0.75, help="Stratified train split ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--coverage-output-dir", default=str(DEFAULT_COVERAGE_OUTPUT_DIR), help="Where to save label coverage diagnostics")
    args = parser.parse_args()

    gt_path = Path(args.ground_truth)
    if not gt_path.is_absolute():
        gt_path = ROOT / gt_path
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    coverage_out_dir = Path(args.coverage_output_dir)
    if not coverage_out_dir.is_absolute():
        coverage_out_dir = ROOT / coverage_out_dir

    summary = prepare_dataset(
        ground_truth=gt_path,
        output_dir=out_dir,
        ocr_engine=args.ocr_engine,
        train_ratio=args.train_ratio,
        seed=args.seed,
        limit=args.limit,
        coverage_output_dir=coverage_out_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
