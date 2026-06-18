from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentRoute:
    category: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def evidence(self) -> dict:
        return {
            "category": self.category,
            "confidence": round(self.confidence, 4),
            "scores": {key: round(value, 4) for key, value in sorted(self.scores.items())},
            "reasons": list(self.reasons),
            "method": "lightweight_document_router",
        }


def _text(line) -> str:
    return str(getattr(line, "text", "") or "")


def _confidence(line) -> float:
    try:
        return float(getattr(line, "confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    return re.sub(r"\s+", " ", value).strip()


def _compact(text: str) -> str:
    value = _normalize(text).upper()
    return re.sub(r"[^A-Z0-9\u3400-\u9fff]+", "", value)


def _contains_any(text: str, fragments: tuple[str, ...]) -> bool:
    return any(fragment in text for fragment in fragments)


def _score(value: float) -> float:
    return max(0.0, min(value, 0.99))


def detect_document_category(lines: list) -> DocumentRoute:
    """Classify an OCR document into a lightweight extraction route.

    The categories are intentionally broad. They tune scoring, not parsing, so
    an imperfect route should still leave strong field-level evidence in charge.
    """
    if not lines:
        return DocumentRoute("generic_invoice", 0.0, {"generic_invoice": 0.0}, ["no OCR lines"])

    raw_lines = [_normalize(_text(line)) for line in lines if _text(line).strip()]
    combined = "\n".join(raw_lines)
    compact = _compact(combined)
    upper = combined.upper()
    avg_conf = sum(_confidence(line) for line in lines) / max(len(lines), 1)
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", combined))
    digit_count = len(re.findall(r"\d", combined))
    garbage_count = sum(
        1
        for line in lines
        if _text(line).strip()
        and _confidence(line) < 0.25
        and not re.search(r"[\u3400-\u9fffA-Za-z0-9]", _text(line))
    )
    receipt_cues = sum(
        1
        for fragment in (
            "RECEIPT", "CASH", "CHANGE", "CASHIER", "SLIP", "TRANS", "MEMBER",
            "ITEMCOUNT", "QTY", "PAYMENT", "ROUNDING", "TOTALSALES",
            "单据号", "收款时间", "支付信息", "商品信息", "应收", "实收", "找零",
        )
        if fragment in compact or fragment in combined
    )

    scores = {
        "chinese_invoice": 0.0,
        "malaysian_receipt": 0.0,
        "clean_invoice": 0.0,
        "fatura": 0.0,
        "noisy_receipt": 0.0,
        "generic_invoice": 0.20,
    }
    reasons: dict[str, list[str]] = {key: [] for key in scores}

    chinese_hits = sum(
        1
        for fragment in (
            "发票", "发票号码", "发票代码", "税务局", "价税合计", "税额",
            "纳税人识别号", "税号", "人民币", "金额", "日期", "出租车",
            "单据号", "收款时间", "应收", "实收", "支付宝", "微信", "门店",
        )
        if fragment in combined
    )
    if cjk_count:
        scores["chinese_invoice"] += min(0.36, cjk_count / 120.0)
        reasons["chinese_invoice"].append("CJK text detected")
    if chinese_hits:
        scores["chinese_invoice"] += min(0.50, chinese_hits * 0.08)
        reasons["chinese_invoice"].append(f"{chinese_hits} Chinese invoice/receipt cues")
    if "CNY" in upper or "RMB" in upper or "￥" in combined or "元" in combined:
        scores["chinese_invoice"] += 0.10
        reasons["chinese_invoice"].append("Chinese currency cue")

    malaysian_hits = sum(
        1
        for fragment in (
            "RM", "MYR", "SDNBHD", "S/B", "GST", "SST", "JALAN", "MALAYSIA",
            "KUALALUMPUR", "SELANGOR", "TOTALRM", "ROUNDED", "ROUNDING",
        )
        if fragment in compact or fragment in upper
    )
    if malaysian_hits:
        scores["malaysian_receipt"] += min(0.62, malaysian_hits * 0.08)
        reasons["malaysian_receipt"].append(f"{malaysian_hits} Malaysian/RM receipt cues")
    if receipt_cues and ("RM" in compact or "MYR" in compact):
        scores["malaysian_receipt"] += 0.18
        reasons["malaysian_receipt"].append("receipt cues with RM/MYR")

    fatura_hits = sum(
        1
        for fragment in ("FATURA", "KDV", "TOPLAM", "TUTAR", "VERGI", "FIRMA", "SERI")
        if fragment in compact
    )
    if fatura_hits:
        scores["fatura"] += min(0.75, fatura_hits * 0.12)
        reasons["fatura"].append(f"{fatura_hits} FATURA/KDV cues")

    clean_hits = sum(
        1
        for fragment in (
            "INVOICENO", "INVOICENUMBER", "INVOICEDATE", "DUEDATE",
            "BILLTO", "SHIPTO", "SUBTOTAL", "BALANCEDUE", "AMOUNTDUE",
            "TAXINVOICE", "TOTALAMOUNT",
        )
        if fragment in compact
    )
    if clean_hits:
        scores["clean_invoice"] += min(0.68, clean_hits * 0.09)
        reasons["clean_invoice"].append(f"{clean_hits} clean invoice label cues")
    if clean_hits >= 3 and receipt_cues <= 1 and cjk_count == 0:
        scores["clean_invoice"] += 0.16
        reasons["clean_invoice"].append("explicit invoice labels without receipt noise")

    low_conf_ratio = sum(1 for line in lines if _confidence(line) < 0.45) / max(len(lines), 1)
    weird_line_ratio = sum(
        1
        for line in raw_lines
        if line and len(re.sub(r"[A-Za-z0-9\u3400-\u9fff]", "", line)) / max(len(line), 1) > 0.45
    ) / max(len(raw_lines), 1)
    if avg_conf < 0.62:
        scores["noisy_receipt"] += 0.20
        reasons["noisy_receipt"].append(f"low average OCR confidence {avg_conf:.2f}")
    if low_conf_ratio >= 0.35:
        scores["noisy_receipt"] += 0.18
        reasons["noisy_receipt"].append("many low-confidence OCR lines")
    if weird_line_ratio >= 0.25 or garbage_count:
        scores["noisy_receipt"] += 0.15
        reasons["noisy_receipt"].append("OCR noise/garbage text pattern")
    if receipt_cues:
        scores["noisy_receipt"] += min(0.18, receipt_cues * 0.03)
        reasons["noisy_receipt"].append("receipt-style layout cues")
    if digit_count >= 12 and receipt_cues and scores["clean_invoice"] < 0.40:
        scores["noisy_receipt"] += 0.08
        reasons["noisy_receipt"].append("dense numeric receipt content")

    # Strong language/domain routes should beat noisy when they are clear.
    if scores["chinese_invoice"] >= 0.45:
        scores["noisy_receipt"] *= 0.55
    if scores["malaysian_receipt"] >= 0.45:
        scores["noisy_receipt"] *= 0.70
    if scores["fatura"] >= 0.45:
        scores["clean_invoice"] *= 0.75

    scores = {key: _score(value) for key, value in scores.items()}
    category = max(scores, key=scores.get)
    confidence = scores[category]
    if confidence < 0.28:
        category = "generic_invoice"
        confidence = scores[category]

    return DocumentRoute(
        category=category,
        confidence=confidence,
        scores=scores,
        reasons=reasons.get(category) or ["fallback generic route"],
    )
