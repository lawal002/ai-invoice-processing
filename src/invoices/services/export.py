from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from django.conf import settings


def export_reviewed_invoice(fields: dict, original_filename: str, fmt: str) -> Path:
    export_dir = Path(settings.AI_INVOICE_EXPORT_DIR)
    export_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(original_filename).stem.replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if fmt == "json":
        path = export_dir / f"{safe_name}_{timestamp}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(fields, f, indent=2, ensure_ascii=False)
        return path
    if fmt == "csv":
        path = export_dir / f"{safe_name}_{timestamp}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(fields.keys()))
            writer.writeheader()
            writer.writerow(fields)
        return path
    raise ValueError("Unsupported export format. Use csv or json.")

