from __future__ import annotations

import re
import string
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .evaluation import FIELDS, normalize
from .ocr import OCRTokenData


class LayoutLMv3BaselineUnavailable(RuntimeError):
    pass


LABEL_FIELDS = [field for field in FIELDS if field != "vat_rate"]
LABEL_LIST = ["O"] + [f"{prefix}-{field.upper()}" for field in LABEL_FIELDS for prefix in ("B", "I")]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

FIELD_PRIORITY = [
    "invoice_number",
    "date",
    "total_amount",
    "tax_amount",
    "subtotal",
    "currency",
    "vendor_name",
]


@dataclass
class LayoutLMv3Example:
    document_id: str
    category: str
    image_path: str
    words: list[str]
    boxes: list[list[int]]
    labels: list[str]
    ground_truth: dict[str, str]
    image_size: tuple[int, int]
    label_coverage: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "category": self.category,
            "image_path": self.image_path,
            "words": self.words,
            "boxes": self.boxes,
            "labels": self.labels,
            "ground_truth": self.ground_truth,
            "image_size": list(self.image_size),
            "label_coverage": self.label_coverage,
        }


def check_layoutlmv3_dependencies() -> dict[str, Any]:
    status: dict[str, Any] = {
        "available": False,
        "torch": "",
        "cuda_available": False,
        "transformers": "",
        "layoutlmv3_import": False,
        "error": "",
        "recommended_fix": "",
    }
    try:
        import torch

        status["torch"] = getattr(torch, "__version__", "")
        status["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        status["error"] = f"torch import failed: {exc}"
        status["recommended_fix"] = (
            "Install PyTorch first, for example: "
            ".\\.venv\\Scripts\\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        )
        return status

    try:
        import transformers

        status["transformers"] = getattr(transformers, "__version__", "")
        from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor  # noqa: F401

        status["layoutlmv3_import"] = True
        status["available"] = True
    except Exception as exc:  # noqa: BLE001
        status["error"] = f"LayoutLMv3 import failed: {exc}"
        status["recommended_fix"] = (
            "Your current transformers build is incompatible with the installed Torch version. "
            "Recommended command: .\\.venv\\Scripts\\python.exe -m pip install --force-reinstall "
            "\"transformers==4.46.3\" \"tokenizers<0.21\""
        )
    return status


def require_layoutlmv3_dependencies() -> None:
    status = check_layoutlmv3_dependencies()
    if not status["available"]:
        raise LayoutLMv3BaselineUnavailable(
            f"{status['error']}\n{status['recommended_fix']}"
        )


def normalize_bbox(bbox: list[float], width: int, height: int) -> list[int]:
    if width <= 0 or height <= 0:
        return [0, 0, 0, 0]
    x1, y1, x2, y2 = [float(value or 0.0) for value in (bbox + [0, 0, 0, 0])[:4]]
    x1, x2 = sorted((max(x1, 0.0), max(x2, 0.0)))
    y1, y2 = sorted((max(y1, 0.0), max(y2, 0.0)))
    x1 = min(x1, width)
    x2 = min(x2, width)
    y1 = min(y1, height)
    y2 = min(y2, height)
    return [
        max(0, min(1000, round(1000 * x1 / width))),
        max(0, min(1000, round(1000 * y1 / height))),
        max(0, min(1000, round(1000 * x2 / width))),
        max(0, min(1000, round(1000 * y2 / height))),
    ]


def _text(value: str) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def _compact(value: str) -> str:
    return re.sub(r"[\s\W_]+", "", _text(value), flags=re.UNICODE).casefold()


def _compact_keep_cjk(value: str) -> str:
    return re.sub(r"[^\w\u3400-\u9fff]+", "", _text(value), flags=re.UNICODE).casefold()


def _ocr_confusion_variants(value: str) -> set[str]:
    value = _text(value).upper()
    compact = re.sub(r"[^A-Z0-9\u3400-\u9fff]+", "", value, flags=re.UNICODE)
    if not compact:
        return set()
    variants = {compact}
    translation_sets = [
        str.maketrans({"O": "0", "Q": "0", "D": "0"}),
        str.maketrans({"I": "1", "L": "1", "|": "1"}),
        str.maketrans({"S": "5"}),
        str.maketrans({"B": "8"}),
        str.maketrans({"Z": "2"}),
    ]
    current = compact
    for table in translation_sets:
        current = current.translate(table)
        variants.add(current)
    all_digit_like = compact.translate(
        str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2"})
    )
    variants.add(all_digit_like)
    return {variant for variant in variants if variant}


def _normalize_amount_candidate(value: str) -> set[str]:
    text = _text(value)
    results: set[str] = set()
    for match in re.finditer(r"[-+]?\d[\d,]*(?:\.\d+)?|[-+]?\d[\d.]*(?:,\d+)", text):
        raw = match.group(0)
        normalized = normalize(raw, "total_amount")
        if not normalized:
            continue
        results.add(normalized)
        results.add(normalized.replace(".", ""))
    normalized = normalize(text, "total_amount")
    if normalized:
        results.add(normalized)
        results.add(normalized.replace(".", ""))
    return {item for item in results if item}


def _normalize_date_candidate(value: str) -> set[str]:
    text = _text(value)
    results: set[str] = set()
    chinese_dates = re.findall(r"\d{2,4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日?", text)
    western_dates = re.findall(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", text)
    candidates = chinese_dates + western_dates + [text]
    for candidate in candidates:
        normalized = normalize(candidate, "date")
        digits = re.sub(r"\D+", "", normalized or candidate)
        if digits:
            results.add(digits)
        if normalized:
            results.add(re.sub(r"\D+", "", normalized))
    return {item for item in results if item}


def _normalize_currency_candidate(value: str) -> set[str]:
    text = _text(value)
    upper = text.upper()
    results: set[str] = set()
    if any(symbol in text for symbol in ("\u00a5", "\uffe5", "\u5143")) or any(word in text for word in ("人民币",)):
        results.add("CNY")
    if any(code in upper for code in ("CNY", "RMB")):
        results.add("CNY")
    if any(code in upper for code in ("RM", "MYR")):
        results.add("RM")
    if "$" in upper or "USD" in upper or "US$" in upper:
        results.add("USD")
    normalized = normalize(text, "currency")
    if normalized:
        results.add(normalized)
    return {item for item in results if item}


def _candidate_norms(value: str, field: str) -> set[str]:
    value = _text(value)
    if not value:
        return set()
    if field in {"subtotal", "tax_amount", "total_amount"}:
        return _normalize_amount_candidate(value)
    if field == "date":
        return _normalize_date_candidate(value)
    if field == "currency":
        return _normalize_currency_candidate(value)
    if field == "invoice_number":
        variants = _ocr_confusion_variants(value)
        normalized = normalize(value, "invoice_number")
        variants.update(_ocr_confusion_variants(normalized))
        variants.add(re.sub(r"[^A-Z0-9\u3400-\u9fff]+", "", normalized.upper(), flags=re.UNICODE))
        return {item for item in variants if item}
    if field == "vendor_name":
        normalized = normalize(value, "vendor_name")
        return {item for item in {_compact_keep_cjk(value), _compact_keep_cjk(normalized), _compact(value)} if item}
    return {_compact_keep_cjk(value)}


def _normalize_for_label(value: str, field: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    if field in {"subtotal", "tax_amount", "total_amount", "currency", "date", "invoice_number", "vendor_name"}:
        value = normalize(value, field)
    if field in {"subtotal", "tax_amount", "total_amount"}:
        return value.replace(".", "")
    if field == "date":
        return re.sub(r"\D+", "", value)
    if field == "currency":
        return value.upper()
    if field == "vendor_name":
        return _compact(value)
    if field == "invoice_number":
        return value.strip(string.punctuation).upper()
    return _compact(value)


def _token_norms(tokens: list[OCRTokenData], field: str) -> list[str]:
    return [
        max(_candidate_norms(token.text, field), key=len, default=_normalize_for_label(token.text, field))
        for token in tokens
    ]


def _window_score(target: str, candidate: str, field: str) -> float:
    if not target or not candidate:
        return 0.0
    if candidate == target:
        return 1.0
    if field != "vendor_name" and (candidate in target or target in candidate):
        if target in candidate and len(target) >= 2:
            return 0.98
        short, long = sorted((len(candidate), len(target)))
        return min(0.96, short / max(long, 1))
    if field == "vendor_name":
        if len(candidate) >= 4 and (candidate in target or target in candidate):
            short, long = sorted((len(candidate), len(target)))
            return max(0.82, short / max(long, 1))
        return SequenceMatcher(None, candidate, target).ratio()
    if field == "invoice_number":
        return SequenceMatcher(None, candidate, target).ratio()
    return SequenceMatcher(None, candidate, target).ratio()


def _score_variants(targets: set[str], candidates: set[str], field: str) -> float:
    best = 0.0
    for target in targets:
        for candidate in candidates:
            best = max(best, _window_score(target, candidate, field))
    return best


def _match_threshold(field: str) -> float:
    if field == "vendor_name":
        return 0.72
    if field == "invoice_number":
        return 0.78
    if field in {"currency", "subtotal", "tax_amount", "total_amount"}:
        return 0.88
    if field == "date":
        return 0.86
    return 0.92


def _best_token_span_candidate(
    tokens: list[OCRTokenData],
    field: str,
    value: str,
    *,
    max_window: int = 14,
) -> tuple[int, int, float] | None:
    targets = _candidate_norms(value, field)
    if not targets:
        return None
    best: tuple[int, int, float] | None = None
    for start in range(len(tokens)):
        window_text_parts: list[str] = []
        for end in range(start, min(len(tokens), start + max_window)):
            token_text = str(tokens[end].text or "").strip()
            if not token_text:
                continue
            window_text_parts.append(token_text)
            window_text = " ".join(window_text_parts)
            candidates = _candidate_norms(window_text, field)
            score = _score_variants(targets, candidates, field)
            if best is None or score > best[2] or (score == best[2] and end - start < best[1] - best[0]):
                best = (start, end + 1, score)
    return best


def find_best_token_span(
    tokens: list[OCRTokenData],
    field: str,
    value: str,
    *,
    max_window: int = 14,
) -> tuple[int, int, float] | None:
    best = _best_token_span_candidate(tokens, field, value, max_window=max_window)
    if best is None:
        return None
    threshold = _match_threshold(field)
    return best if best[2] >= threshold else None


def assign_bio_labels_with_coverage(tokens: list[OCRTokenData], ground_truth: dict[str, str]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    labels = ["O"] * len(tokens)
    coverage: dict[str, dict[str, Any]] = {}
    for field in FIELD_PRIORITY:
        value = str(ground_truth.get(field) or "").strip()
        coverage[field] = {
            "ground_truth": value,
            "has_ground_truth": bool(value),
            "matched": False,
            "reason": "empty_ground_truth" if not value else "no_candidate_above_threshold",
            "score": 0.0,
            "start": None,
            "end": None,
            "matched_text": "",
        }
        if not value:
            continue
        best = _best_token_span_candidate(tokens, field, value)
        if best is None:
            coverage[field]["reason"] = "no_candidate"
            continue
        start, end, score = best
        coverage[field]["score"] = round(score, 4)
        coverage[field]["start"] = start
        coverage[field]["end"] = end
        coverage[field]["matched_text"] = " ".join(str(tokens[idx].text or "").strip() for idx in range(start, end)).strip()
        if score < _match_threshold(field):
            continue
        if any(labels[idx] != "O" for idx in range(start, end)):
            coverage[field]["reason"] = "span_overlaps_existing_label"
            continue
        labels[start] = f"B-{field.upper()}"
        for idx in range(start + 1, end):
            labels[idx] = f"I-{field.upper()}"
        coverage[field]["matched"] = True
        coverage[field]["reason"] = "matched"
    for field in LABEL_FIELDS:
        coverage.setdefault(
            field,
            {
                "ground_truth": str(ground_truth.get(field) or "").strip(),
                "has_ground_truth": bool(str(ground_truth.get(field) or "").strip()),
                "matched": False,
                "reason": "not_attempted",
                "score": 0.0,
                "start": None,
                "end": None,
                "matched_text": "",
            },
        )
    return labels, coverage


def assign_bio_labels(tokens: list[OCRTokenData], ground_truth: dict[str, str]) -> list[str]:
    labels, _coverage = assign_bio_labels_with_coverage(tokens, ground_truth)
    return labels


def _field_from_label(label: str) -> str:
    if "-" not in label:
        return ""
    return label.split("-", 1)[1].lower()


def _label_spans(labels: list[str]) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    current_field = ""
    start = -1
    for idx, label in enumerate(labels):
        if label == "O":
            if current_field:
                spans.append((current_field, start, idx))
            current_field = ""
            start = -1
            continue
        field = _field_from_label(label)
        if not field:
            if current_field:
                spans.append((current_field, start, idx))
            current_field = ""
            start = -1
            continue
        if label.startswith("B-") or field != current_field:
            if current_field:
                spans.append((current_field, start, idx))
            current_field = field
            start = idx
    if current_field:
        spans.append((current_field, start, len(labels)))
    return spans


def decode_token_predictions(tokens: list[OCRTokenData], labels: list[str]) -> dict[str, str]:
    fields: dict[str, list[str]] = {}
    current_field = ""
    current_tokens: list[str] = []

    def flush() -> None:
        nonlocal current_field, current_tokens
        if current_field and current_tokens and current_field not in fields:
            fields[current_field] = list(current_tokens)
        current_field = ""
        current_tokens = []

    for token, label in zip(tokens, labels):
        if label == "O":
            flush()
            continue
        field = _field_from_label(label)
        if not field:
            flush()
            continue
        if label.startswith("B-") or field != current_field:
            flush()
            current_field = field
            current_tokens = [token.text]
        else:
            current_tokens.append(token.text)
    flush()

    decoded = {field: " ".join(values).strip() for field, values in fields.items()}
    for amount_field in ("subtotal", "tax_amount", "total_amount"):
        if amount_field in decoded:
            decoded[amount_field] = normalize(decoded[amount_field], amount_field)
    if "currency" in decoded:
        decoded["currency"] = normalize(decoded["currency"], "currency")
    if "date" in decoded:
        decoded["date"] = normalize(decoded["date"], "date")
    if "invoice_number" in decoded:
        decoded["invoice_number"] = normalize(decoded["invoice_number"], "invoice_number")
    return {field: decoded.get(field, "") for field in FIELDS}


AMOUNT_RE = re.compile(r"[-+]?\s*(?:\d{1,3}(?:[ ,]\d{3})+|\d+)(?:[.,]\d{1,2})?")
DATE_RE = re.compile(
    r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日?"
    r"|\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"
    r"|\d{1,2}[- ](?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[- ]\d{2,4}",
    re.IGNORECASE,
)
MONTH_NAME_RE = re.compile(
    r"(\d{1,2})[- ](jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[- ](\d{2,4})",
    re.IGNORECASE,
)
INVOICE_LABEL_RE = re.compile(
    r"(?:invoice\s*(?:no|number|#)?|receipt\s*(?:no|number|#)?|document\s*(?:no|number|#)?|"
    r"bill\s*(?:no|number|#)?|slip\s*(?:no|number|#)?|trans(?:action)?\s*(?:no|number|#)?|"
    r"ref(?:erence)?\s*(?:no|number|#)?|发票号码|单据号|单号)",
    re.IGNORECASE,
)
DATE_LABEL_RE = re.compile(r"(?:date|invoice\s*date|date\s*of\s*issue|日期|开票日期|收款时间)", re.IGNORECASE)
FIELD_LABELS = {
    "subtotal": re.compile(r"(?:subtotal|sub\s*total|net\s*worth|net\s*amount|amount\s*\(rm\)|金额\s*\(不含税\)|金额)", re.IGNORECASE),
    "tax_amount": re.compile(r"(?:tax|vat|gst|tax\s*amount|税额)", re.IGNORECASE),
    "total_amount": re.compile(r"(?:total\s*amount|amount\s*due|grand\s*total|rounded\s*total|total\s*rm|gross\s*worth|价税合计|合计|金额)", re.IGNORECASE),
}
NON_VALUE_TEXT_RE = re.compile(
    r"^(?:invoice|invoice\s*no|invoice\s*number|commercial\s*invoice|tax\s*invoice|date|seller|client|"
    r"total|total\s*amount|subtotal|tax|vat|gst|currency|description|qty|unit\s*price|line\s*total)[:：\\s]*$",
    re.IGNORECASE,
)


def _normalize_layoutlm_amount(raw: str) -> str:
    text = _text(raw)
    if not text:
        return ""
    text = re.sub(r"[^\d,.\-\s]", "", text).strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    sign = "-" if text.startswith("-") else ""
    text = text.lstrip("+-").strip()
    compact = text.replace(" ", "")
    if not compact or not re.search(r"\d", compact):
        return ""
    if "," in compact and "." in compact:
        decimal_sep = "," if compact.rfind(",") > compact.rfind(".") else "."
        thousands_sep = "." if decimal_sep == "," else ","
        compact = compact.replace(thousands_sep, "").replace(decimal_sep, ".")
    elif "," in compact:
        pieces = compact.split(",")
        compact = "".join(pieces[:-1]) + "." + pieces[-1] if len(pieces[-1]) in {1, 2} else "".join(pieces)
    elif "." in compact:
        pieces = compact.split(".")
        if len(pieces) > 2:
            compact = "".join(pieces[:-1]) + "." + pieces[-1] if len(pieces[-1]) in {1, 2} else "".join(pieces)
    if not re.fullmatch(r"\d+(?:\.\d+)?", compact):
        return ""
    try:
        amount = Decimal(sign + compact).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return ""
    return f"{amount:.2f}"


def _amounts_in_text(text: str) -> list[str]:
    values: list[str] = []
    for match in AMOUNT_RE.finditer(_text(text)):
        if re.match(r"\s*%", text[match.end() :]):
            continue
        normalized = _normalize_layoutlm_amount(match.group(0))
        if normalized and normalized not in values:
            values.append(normalized)
    return values


def _date_in_text(text: str) -> str:
    text = _text(text)
    for match in DATE_RE.finditer(text):
        candidate = match.group(0).strip()
        month_match = MONTH_NAME_RE.fullmatch(candidate)
        if month_match:
            day, month, year = month_match.groups()
            return f"{int(day):02d}-{month[:3].title()}-{year}"
        if "年" in candidate:
            parts = re.findall(r"\d+", candidate)
            if len(parts) >= 3:
                return f"{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"
        normalized = normalize(candidate, "date")
        return normalized or candidate
    return ""


def _currency_in_text(text: str) -> str:
    value = _text(text)
    upper = value.upper()
    if any(symbol in value for symbol in ("\u00a5", "\uffe5", "\u5143")) or "人民币" in value:
        return "CNY"
    for code, normalized in (
        ("MYR", "RM"),
        ("RM", "RM"),
        ("CNY", "CNY"),
        ("RMB", "CNY"),
        ("USD", "USD"),
        ("US$", "USD"),
        ("BDT", "BDT"),
        ("EUR", "EUR"),
        ("NGN", "NGN"),
    ):
        if re.search(rf"(?<![A-Z]){re.escape(code)}(?![A-Z])", upper):
            return normalized
    if "$" in value:
        return "USD"
    return ""


def _strip_label_prefix(text: str, label_re: re.Pattern[str]) -> str:
    value = _text(text)
    value = label_re.sub("", value, count=1)
    value = re.sub(r"^[\s:#：\-]+", "", value)
    return value.strip()


def _clean_invoice_value(text: str) -> str:
    original = _text(text)
    number_match = re.search(r"发票号码\s*[:：]?\s*([A-Z0-9/\-]{4,})", original, re.IGNORECASE)
    if number_match:
        return number_match.group(1).strip(string.punctuation)
    if re.search(r"发票代码|invoice\s*code", original, re.IGNORECASE):
        return ""
    value = _strip_label_prefix(text, INVOICE_LABEL_RE)
    value = re.sub(r"^(?:no|number|#)\s*[:：#-]*\s*", "", value, flags=re.IGNORECASE).strip()
    if not value or NON_VALUE_TEXT_RE.fullmatch(value):
        return ""
    if re.fullmatch(r"(?:commercial|tax|invoice|receipt|bill|document)", value.strip(), flags=re.IGNORECASE):
        return ""
    patterns = [
        r"[A-Z]{1,8}[-/]?\d[A-Z0-9/\-]{2,}",
        r"\d[A-Z0-9/\-]{4,}",
    ]
    for pattern in patterns:
        match = re.search(pattern, value.upper())
        if match:
            candidate = match.group(0).strip(string.punctuation)
            if not re.fullmatch(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}", candidate):
                return candidate
    compact = normalize(value, "invoice_number")
    return compact if len(compact) >= 4 and not NON_VALUE_TEXT_RE.fullmatch(compact) else ""


def _clean_field_value(field: str, text: str) -> str:
    text = _text(text)
    if not text:
        return ""
    if field == "invoice_number":
        return _clean_invoice_value(text)
    if field == "date":
        return _date_in_text(_strip_label_prefix(text, DATE_LABEL_RE)) or _date_in_text(text)
    if field in {"subtotal", "tax_amount", "total_amount"}:
        label_re = FIELD_LABELS.get(field, re.compile("$^"))
        stripped = _strip_label_prefix(text, label_re)
        values = _amounts_in_text(stripped)
        if field in {"subtotal", "total_amount"}:
            values = [value for value in values if value != "0.00"]
        if not values:
            return ""
        lowered = text.lower()
        if field == "total_amount" and re.search(r"\btotal\b", lowered) and not re.search(r"total\s*amount|amount\s*due|grand\s*total|价税合计|金额", lowered):
            return values[-1]
        return values[0]
    if field == "currency":
        return _currency_in_text(text)
    if field == "vendor_name":
        value = re.sub(r"\b(?:seller|client|bill\s*to|ship\s*to|tax\s*invoice|commercial\s*invoice|logo)\b[:：]?", "", text, flags=re.IGNORECASE)
        value = re.split(
            r"\b(?:invoice\s*no|invoice\s*number|business\s*address|date|tax\s*id|currency|vat\s*rate|description)\b[:：]?",
            value,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        value = re.sub(r"^\s*\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\s*", "", value)
        value = re.split(r"\s+\d{3,}[\w\s,.-]*$", value, maxsplit=1)[0]
        value = re.sub(r"\s+", " ", value).strip(" :-")
        return value if value and not NON_VALUE_TEXT_RE.fullmatch(value) else ""
    return text


def _span_text(tokens: list[OCRTokenData], start: int, end: int) -> str:
    return " ".join(str(tokens[idx].text or "").strip() for idx in range(start, min(end, len(tokens)))).strip()


def _nearby_window(tokens: list[OCRTokenData], start: int, end: int, *, before: int = 1, after: int = 5) -> str:
    left = max(0, start - before)
    right = min(len(tokens), end + after)
    return _span_text(tokens, left, right)


def _find_labeled_value(tokens: list[OCRTokenData], field: str) -> str:
    label_re = INVOICE_LABEL_RE if field == "invoice_number" else DATE_LABEL_RE if field == "date" else FIELD_LABELS.get(field)
    if label_re is None:
        return ""
    words = [str(token.text or "").strip() for token in tokens]
    for idx, word in enumerate(words):
        if not label_re.search(word):
            continue
        inline = _clean_field_value(field, word)
        if inline and not label_re.fullmatch(inline):
            return inline
        for end in range(idx + 1, min(len(words), idx + 5)):
            value = _clean_field_value(field, " ".join(words[idx + 1 : end + 1]))
            if value:
                return value
    return ""


def _detect_document_currency(tokens: list[OCRTokenData], category: str = "") -> str:
    joined = " ".join(str(token.text or "") for token in tokens)
    currency = _currency_in_text(joined)
    if currency:
        return currency
    category_lower = (category or "").lower()
    if "chinese" in category_lower or "vatid" in category_lower:
        return "CNY"
    if "malaysian" in category_lower:
        return "RM"
    return ""


def decode_token_predictions_with_context(
    tokens: list[OCRTokenData],
    labels: list[str],
    *,
    category: str = "",
) -> dict[str, str]:
    spans = _label_spans(labels)
    decoded = decode_token_predictions(tokens, labels)
    spans_by_field: dict[str, list[tuple[int, int]]] = {}
    for field, start, end in spans:
        spans_by_field.setdefault(field, []).append((start, end))

    repaired: dict[str, str] = {}
    for field in FIELDS:
        candidates: list[str] = []
        for start, end in spans_by_field.get(field, []):
            raw = _span_text(tokens, start, end)
            candidates.append(_clean_field_value(field, raw))
            before = 1 if field == "vendor_name" else 0
            candidates.append(_clean_field_value(field, _nearby_window(tokens, start, end, before=before)))
        if not candidates:
            candidates.append(_clean_field_value(field, decoded.get(field, "")))
        if field in {"invoice_number", "date", "subtotal", "tax_amount", "total_amount"}:
            candidates.append(_find_labeled_value(tokens, field))
        if field == "currency":
            candidates.append(_detect_document_currency(tokens, category))

        value = ""
        for candidate in candidates:
            if candidate:
                value = candidate
                break
        repaired[field] = value

    if repaired.get("currency"):
        repaired["currency"] = normalize(repaired["currency"], "currency")
    for amount_field in ("subtotal", "tax_amount", "total_amount"):
        if repaired.get(amount_field):
            repaired[amount_field] = _normalize_layoutlm_amount(repaired[amount_field]) or repaired[amount_field]
    if repaired.get("invoice_number"):
        repaired["invoice_number"] = _clean_invoice_value(repaired["invoice_number"]) or repaired["invoice_number"]
    return {field: repaired.get(field, "") for field in FIELDS}


def build_layoutlmv3_example(
    *,
    document_id: str,
    category: str,
    image_path: str,
    tokens: list[OCRTokenData],
    ground_truth: dict[str, str],
    image_size: tuple[int, int],
) -> LayoutLMv3Example:
    width, height = image_size
    filtered = [token for token in tokens if str(token.text or "").strip()]
    words = [str(token.text).strip() for token in filtered]
    boxes = [normalize_bbox(token.bbox, width, height) for token in filtered]
    labels, label_coverage = assign_bio_labels_with_coverage(filtered, ground_truth)
    return LayoutLMv3Example(
        document_id=document_id,
        category=category,
        image_path=image_path,
        words=words,
        boxes=boxes,
        labels=labels,
        ground_truth={field: str(ground_truth.get(field) or "").strip() for field in FIELDS},
        image_size=image_size,
        label_coverage=label_coverage,
    )
