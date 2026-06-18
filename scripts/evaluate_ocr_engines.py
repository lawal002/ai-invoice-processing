"""Compare OCR engines on invoice images or stored token fixtures."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import django
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ai_invoice_project.settings")
django.setup()

from invoices.services.extraction import extract_layout_aware
from invoices.services.ocr import OCRTokenData, OCRDependencyError, SUPPORTED_ENGINES


DISPLAY_FIELDS = [
    "invoice_number",
    "date",
    "vendor_name",
    "total_amount",
    "tax_amount",
    "subtotal",
    "currency",
]


def tokens_from_fixture(fixture_path: Path) -> list[OCRTokenData]:
    """Load OCR tokens from a fixture JSON file."""
    with fixture_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    tokens = []
    for t in data.get("tokens", []):
        tokens.append(
            OCRTokenData(
                text=t["text"],
                bbox=t["bbox"],
                confidence=t["confidence"],
                page=t.get("page", 1),
            )
        )
    return tokens


def tokens_from_image(image_path: Path, engine: str) -> tuple[list[OCRTokenData], float]:
    """Run OCR on an image and return (tokens, inference_ms)."""
    from invoices.services.ocr import run_ocr
    t0 = time.perf_counter()
    tokens = run_ocr(image_path, engine=engine)
    ms = (time.perf_counter() - t0) * 1000
    return tokens, ms


def evaluate_document(
    doc_name: str,
    tokens: list[OCRTokenData],
    ocr_ms: float = 0.0,
    ground_truth: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run layout-aware extraction and return a result record."""
    t0 = time.perf_counter()
    result = extract_layout_aware(tokens)
    extraction_ms = (time.perf_counter() - t0) * 1000

    record: dict[str, Any] = {
        "document": doc_name,
        "ocr_ms": round(ocr_ms, 1),
        "extraction_ms": round(extraction_ms, 1),
        "total_ms": round(ocr_ms + extraction_ms, 1),
        "fields": {},
    }
    for field in DISPLAY_FIELDS:
        value = result.fields.get(field, "")
        conf = result.confidences.get(field, 0.0)
        ev = result.evidence.get(field, {})
        entry: dict[str, Any] = {
            "value": value,
            "confidence": round(conf, 3),
            "method": ev.get("method", ""),
        }
        if ground_truth and field in ground_truth:
            gt_val = str(ground_truth[field]).strip()
            entry["ground_truth"] = gt_val
            entry["correct"] = gt_val.lower() == value.lower() if gt_val else None
        record["fields"][field] = entry

    confidences = [
        record["fields"][f]["confidence"]
        for f in DISPLAY_FIELDS
        if record["fields"][f]["value"]
    ]
    record["mean_confidence"] = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
    record["extracted_count"] = sum(1 for f in DISPLAY_FIELDS if record["fields"][f]["value"])
    return record


def _col(s: str, width: int) -> str:
    return str(s)[:width].ljust(width)


def print_comparison_table(
    results_by_engine: dict[str, list[dict]],
    fields_to_show: list[str] | None = None,
) -> None:
    fields_to_show = fields_to_show or DISPLAY_FIELDS
    engines = list(results_by_engine.keys())

    print()
    print("=" * 100)
    print("  OCR ENGINE COMPARISON — layout-aware extraction")
    print("=" * 100)

    all_docs = []
    for records in results_by_engine.values():
        for r in records:
            if r["document"] not in all_docs:
                all_docs.append(r["document"])

    for doc in all_docs:
        print(f"\nDocument: {doc}")
        print("-" * 90)
        header = _col("Engine", 12) + _col("ms", 8) + _col("n/9", 5)
        for f in fields_to_show:
            header += _col(f[:14], 16)
        print(header)
        print("-" * 90)

        for engine in engines:
            engine_results = {r["document"]: r for r in results_by_engine[engine]}
            rec = engine_results.get(doc)
            if rec is None:
                print(_col(engine, 12) + "  — not run")
                continue
            row = _col(engine, 12)
            row += _col(f"{rec['total_ms']:.0f}", 8)
            row += _col(f"{rec['extracted_count']}/{len(DISPLAY_FIELDS)}", 5)
            for f in fields_to_show:
                fdata = rec["fields"].get(f, {})
                val = str(fdata.get("value", ""))[:13]
                correct = fdata.get("correct")
                marker = "" if correct is None else ("✓" if correct else "✗")
                row += _col(f"{val}{marker}", 16)
            print(row)

    print()
    print("=" * 70)
    print("  SUMMARY (across all documents)")
    print("=" * 70)
    print(_col("Engine", 14) + _col("Docs", 6) + _col("Avg ms", 9)
          + _col("Avg conf", 10) + _col("Avg fields", 12))
    print("-" * 70)
    for engine, records in results_by_engine.items():
        n = len(records)
        avg_ms = sum(r["total_ms"] for r in records) / n if n else 0
        avg_conf = sum(r["mean_confidence"] for r in records) / n if n else 0
        avg_fields = sum(r["extracted_count"] for r in records) / n if n else 0
        print(
            _col(engine, 14) + _col(str(n), 6) + _col(f"{avg_ms:.0f}", 9)
            + _col(f"{avg_conf:.3f}", 10) + _col(f"{avg_fields:.1f}/{len(DISPLAY_FIELDS)}", 12)
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, default=None,
                        help="Directory of invoice images to run OCR on")
    parser.add_argument("--engines", nargs="+", default=["easyocr", "paddleocr", "tesseract"],
                        metavar="ENGINE",
                        help="OCR engines to evaluate (default: easyocr paddleocr tesseract)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write results table to this JSON file")
    parser.add_argument("--fixtures", type=Path,
                        default=ROOT / "src" / "invoices" / "fixtures",
                        help="Fixture directory (default: src/invoices/fixtures)")
    args = parser.parse_args()

    for eng in args.engines:
        if eng not in SUPPORTED_ENGINES:
            print(f"ERROR: unknown engine {eng!r}. Supported: {SUPPORTED_ENGINES}")
            sys.exit(1)

    results_by_engine: dict[str, list[dict]] = {}

    if args.images and args.images.is_dir():
        image_files = sorted(
            p for p in args.images.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".pdf"}
        )
        if not image_files:
            print(f"No images found in {args.images}")
            sys.exit(0)

        for engine in args.engines:
            print(f"\nRunning {engine} on {len(image_files)} image(s)...")
            records = []
            for img in image_files:
                print(f"  {img.name} ... ", end="", flush=True)
                try:
                    tokens, ocr_ms = tokens_from_image(img, engine)
                    rec = evaluate_document(img.name, tokens, ocr_ms=ocr_ms)
                    print(f"done ({ocr_ms:.0f} ms OCR, {rec['extraction_ms']:.0f} ms extract)")
                except OCRDependencyError as exc:
                    print(f"SKIPPED — {exc}")
                    break
                records.append(rec)
            if records:
                results_by_engine[engine] = records

    else:
        fixture_files = sorted(args.fixtures.glob("doc*_tokens.json"))
        if not fixture_files:
            print(f"No fixture files found in {args.fixtures}")
            print("Provide --images DIR or add token fixtures.")
            sys.exit(0)

        print(f"\nUsing stored fixture tokens from {args.fixtures}")
        print(f"Found {len(fixture_files)} fixture file(s).\n")

        records: list[dict] = []
        for fpath in fixture_files:
            print(f"  {fpath.name} ... ", end="", flush=True)
            with fpath.open("r", encoding="utf-8") as f:
                fixture_data = json.load(f)
            tokens = tokens_from_fixture(fpath)
            gt = fixture_data.get("current_output", {}).get("fields")
            rec = evaluate_document(
                doc_name=fixture_data.get("original_filename", fpath.stem),
                tokens=tokens,
                ground_truth=gt,
            )
            print(f"done ({rec['extraction_ms']:.0f} ms, {rec['extracted_count']}/{len(DISPLAY_FIELDS)} fields)")
            records.append(rec)

        results_by_engine["fixture_tokens"] = records

        print("\n-- Before/After comparison (fixture baseline vs current extraction) --")
        changed_total = 0
        for fpath in fixture_files:
            with fpath.open("r", encoding="utf-8") as f:
                fixture_data = json.load(f)
            baseline = fixture_data.get("current_output", {}).get("fields", {})
            fname = fixture_data.get("original_filename", fpath.stem)
            current_rec = next(
                (r for r in records if r["document"] == fname), None
            )
            if current_rec is None:
                continue
            changes = []
            for field in DISPLAY_FIELDS:
                old_val = str(baseline.get(field, "")).strip()
                new_val = str(current_rec["fields"].get(field, {}).get("value", "")).strip()
                if old_val != new_val:
                    changes.append(f"  {field}: {old_val!r} → {new_val!r}")
            if changes:
                print(f"\n{fname}:")
                for ch in changes:
                    print(ch)
                changed_total += len(changes)
            else:
                print(f"\n{fname}: no changes (all fields identical to baseline)")

        print(f"\nTotal field changes vs baseline: {changed_total}")

    if not results_by_engine:
        print("\nNo results produced. Check that engines are installed or provide --images.")
        sys.exit(0)

    print_comparison_table(results_by_engine)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(results_by_engine, f, indent=2, ensure_ascii=False)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
