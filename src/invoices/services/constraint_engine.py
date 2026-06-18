"""Assign invoice amount roles using financial consistency rules."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from itertools import combinations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Ordered from specific labels to generic ones.
_ROLE_PATTERNS: list[tuple[str, str]] = [
    # Chinese retail and payment labels.
    ("订单优惠", "item"),
    ("消费券抵扣金额", "item"),
    ("消费券", "item"),
    ("优惠", "item"),
    ("折扣", "item"),
    ("件数", "item"),
    ("数量", "item"),
    ("品名", "item"),
    ("找零", "change"),
    ("实收", "cash_paid"),
    ("支付宝", "cash_paid"),
    ("微信", "cash_paid"),
    ("现金", "cash_paid"),
    ("银行卡", "cash_paid"),
    ("应收", "total"),
    ("应付", "total"),
    ("实付", "total"),
    ("收款金额", "total"),
    ("订单原价", "subtotal"),
    ("商品合计", "subtotal"),
    ("原价合计", "subtotal"),
    ("CASH", "cash_paid"),
    ("TENDERED", "cash_paid"),
    ("AMOUNTPAID", "cash_paid"),
    ("PAID", "cash_paid"),
    ("CARD", "cash_paid"),
    ("CREDIT", "cash_paid"),
    ("DEBIT", "cash_paid"),
    ("CHANGE", "change"),
    ("CWAHGE", "change"),
    ("CWANGE", "change"),
    ("CHAHGE", "change"),
    ("CHAMGE", "change"),
    ("GRANDTOTAL", "total"),
    ("TOTALSALES", "total"),
    ("AMOUNTDUE", "total"),
    ("NETTOTAL", "total"),
    ("INVOICETOTAL", "total"),
    ("TOTALINCL", "total"),
    ("INCLGST", "total"),
    ("INCLSST", "total"),
    ("ROUNDEDTOTAL", "rounded_total"),
    ("ROUNDINGTOTAL", "rounded_total"),
    ("ROUNDING", "rounding"),
    # Subtotal labels must precede the generic TOTAL fallback.
    ("SUBTOTAL", "subtotal"),
    ("BEFORETAX", "subtotal"),
    ("TOTALEXCL", "subtotal"),
    ("EXCLGST", "subtotal"),
    ("EXCLSST", "subtotal"),
    ("BEFOREGST", "subtotal"),
    ("BEFORESST", "subtotal"),
    ("TAXAMOUNT", "tax"),
    ("SERVICETAX", "tax"),
    ("SALESTAX", "tax"),
    ("GSTTAX", "tax"),
    # Short labels are intentionally checked last.
    ("TAX", "tax"),
    ("GST", "tax"),
    ("VAT", "tax"),
    # Line-item amounts are excluded from invoice-level assignments.
    ("LINETOTAL", "item"),
    ("LINEITEM", "item"),
    ("ITEMCOUNT", "item"),
    ("DESCRIPTION", "item"),
    ("HRSIQTY", "item"),
    ("DISCOUNT", "item"),
    ("SAVING", "item"),
    ("SAVINGS", "item"),
    ("REBATE", "item"),
    ("RABAT", "item"),
    ("QTY", "item"),
    ("QUANTITY", "item"),
    ("UNITPRICE", "item"),
    ("PRICEUNIT", "item"),
    ("UNITCOST", "item"),
    ("UNIT", "item"),
]

_TOTAL_GENERIC = "TOTAL"


def _classify_role(compact_text: str) -> str:
    """Return the amount role suggested by the compact (upper, stripped) text of a line."""
    for fragment, role in _ROLE_PATTERNS:
        if fragment in compact_text:
            return role
    if _TOTAL_GENERIC in compact_text:
        return "total"
    return "unknown"


@dataclass
class AmountEvidence:
    """A single decimal amount found in the document, with its source context."""
    value: Decimal
    normalized: str
    source_text: str
    bbox: list[float]
    ocr_confidence: float
    label_role: str
    zone_ratio: float
    line: object
    is_repair: bool = False  # True when value was recovered from a damaged OCR token
    repair_edits: int = 0    # Number of confusable-character edits applied


@dataclass
class FinancialAssignment:
    """The best financially-consistent assignment of amounts to invoice roles."""
    subtotal: AmountEvidence | None = None
    tax: AmountEvidence | None = None
    total: AmountEvidence | None = None
    rounded_total: AmountEvidence | None = None
    cash_paid: AmountEvidence | None = None
    change: AmountEvidence | None = None
    rounding: AmountEvidence | None = None
    consistency_score: float = 0.0
    satisfied_laws: list[str] = field(default_factory=list)


def _compact(text: str) -> str:
    """Return upper-case alphanumeric-only form of text."""
    import re
    import unicodedata
    value = unicodedata.normalize("NFKC", str(text or "")).upper()
    return re.sub(r"[^A-Z0-9\u3400-\u9fff]", "", value)


def _to_decimal(value: str) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _role_for_amount(line, amount_cx: float) -> str:
    """Use the nearest label to the left of an amount."""
    # Chinese POS OCR often keeps the label and value in one token.
    for tok in getattr(line, "tokens", []):
        if tok.x1 <= amount_cx <= tok.x2:
            role = _classify_role(_compact(tok.text))
            if role != "unknown":
                return role

    best_role, best_dx = None, None
    for tok in getattr(line, "tokens", []):
        if tok.cx >= amount_cx:
            continue
        role = _classify_role(_compact(tok.text))
        if role == "unknown":
            continue
        dx = amount_cx - tok.cx
        if best_dx is None or dx < best_dx:
            best_dx, best_role = dx, role
    return best_role or _classify_role(_compact(line.text))


def collect_amount_evidence(lines: list) -> list[AmountEvidence]:
    """Collect exact and repaired amount candidates from OCR lines."""
    from .extraction import (
        _extract_amounts,
        _extract_amounts_with_repair,
        _document_bounds,
        _is_adjustment_context,
        _is_line_item_amount_context,
        _y_ratio,
    )
    from .amount_repair import enumerate_repairs

    evidence: list[AmountEvidence] = []
    if not lines:
        return evidence

    _document_bounds(lines)

    existing_normals: set[tuple[str, str]] = set()

    for line in lines:
        zone = float(_y_ratio(line.bbox, lines))
        tokens = getattr(line, "tokens", [])
        forced_role = "item" if (
            _is_line_item_amount_context(getattr(line, "text", ""))
            or _is_adjustment_context(getattr(line, "text", ""))
        ) else ""

        # Token centers preserve column-specific label context.
        token_amounts: list[tuple[str, str, float, float]] = []
        for tok in tokens:
            for norm_str, raw_str in _extract_amounts(tok.text, allow_integer=False):
                token_amounts.append((norm_str, raw_str, tok.cx, float(tok.confidence)))

        # Whole-line parsing handles values split across OCR tokens.
        if not token_amounts:
            for norm_str, raw_str in _extract_amounts_with_repair(line.text, allow_integer=False):
                token_amounts.append((norm_str, raw_str, line.cx, float(line.confidence)))

        for norm_str, _raw_str, amount_cx, tok_conf in token_amounts:
            value = _to_decimal(norm_str)
            if value is None or value < Decimal("0"):
                continue
            if (norm_str, line.text) in existing_normals:
                continue
            role = forced_role or _role_for_amount(line, amount_cx)
            evidence.append(
                AmountEvidence(
                    value=value,
                    normalized=norm_str,
                    source_text=line.text,
                    bbox=list(line.bbox),
                    ocr_confidence=tok_conf,
                    label_role=role,
                    zone_ratio=zone,
                    line=line,
                )
            )
            existing_normals.add((norm_str, line.text))

    # Add plausible repairs that do not duplicate exact parses.
    # A "suspicious" token contains at least one non-digit char that is confusable with
    # a digit (e.g. "J89,80" → 'J' is confusable with '1').
    _SUSPICIOUS_RE = re.compile(r"[A-Za-z|]")  # non-digit, letter-like chars in amounts

    for line in lines:
        zone = float(_y_ratio(line.bbox, lines))

        for token in getattr(line, "tokens", []):
            raw = token.original_text.strip()
            # Only bother if token looks amount-like (has digits and suspicious chars)
            has_digit = any(c.isdigit() for c in raw)
            has_suspicious = bool(_SUSPICIOUS_RE.search(raw))
            has_decimal_sep = "." in raw or "," in raw
            if not (has_digit and has_suspicious and has_decimal_sep):
                continue

            for repaired_str, edit_count in enumerate_repairs(raw, max_edits=2):
                key = (repaired_str, line.text)
                if key in existing_normals:
                    continue
                value = _to_decimal(repaired_str)
                if value is None or value < Decimal("0"):
                    continue
                # Confidence degrades with edit distance; minimum 0.20
                repair_conf = max(0.20, float(token.confidence) - 0.15 * edit_count)
                role = forced_role or _role_for_amount(line, float(token.cx))
                evidence.append(
                    AmountEvidence(
                        value=value,
                        normalized=repaired_str,
                        source_text=f"[repair:{edit_count}e] {line.text}",
                        bbox=list(line.bbox),
                        ocr_confidence=repair_conf,
                        label_role=role,
                        zone_ratio=zone,
                        line=line,
                        is_repair=True,
                        repair_edits=edit_count,
                    )
                )
                existing_normals.add(key)

    return evidence


_LAW_WEIGHTS = {
    "law1_subtotal_tax_total":   3.0,
    "law2_cash_change":          2.5,
    "law3_cash_gte_total":       1.0,
    "law6_cash_below_total":     1.0,
    "law7_tax_lt_subtotal":      0.5,
    "law8_total_in_lower_half":  0.5,
}

_TOLERANCE = Decimal("0.30")
_EXACT_TOLERANCE = Decimal("0.05")


def _score_assignment(a: FinancialAssignment) -> tuple[float, list[str]]:
    """Return (score, list_of_satisfied_law_names)."""
    score = 0.0
    satisfied: list[str] = []

    def _val(ev: AmountEvidence | None) -> Decimal | None:
        return ev.value if ev else None

    subtotal = _val(a.subtotal)
    tax = _val(a.tax)
    total = _val(a.total)
    cash = _val(a.cash_paid)
    change = _val(a.change)

    # Hard violations: immediately disqualify
    if tax is not None and total is not None and tax >= total:
        return -10.0, []
    if cash is not None and total is not None and cash < total:
        return -10.0, []
    if change is not None and change < Decimal("0"):
        return -10.0, []

    # Law 1: subtotal + tax ≈ total
    if subtotal is not None and tax is not None and total is not None:
        diff = abs(subtotal + tax - total)
        if diff <= _EXACT_TOLERANCE:
            score += _LAW_WEIGHTS["law1_subtotal_tax_total"]
            satisfied.append("law1_subtotal_tax_total")
        elif diff <= _TOLERANCE:
            score += _LAW_WEIGHTS["law1_subtotal_tax_total"] * float(1 - diff / _TOLERANCE)
            satisfied.append("law1_subtotal_tax_total_partial")

    # Law 2: cash - total ≈ change
    if cash is not None and total is not None and change is not None:
        diff = abs(cash - total - change)
        if diff <= _EXACT_TOLERANCE:
            score += _LAW_WEIGHTS["law2_cash_change"]
            satisfied.append("law2_cash_change")
        elif diff <= _TOLERANCE:
            score += _LAW_WEIGHTS["law2_cash_change"] * float(1 - diff / _TOLERANCE)
            satisfied.append("law2_cash_change_partial")

    # Law 3: cash ≥ total (soft bonus — hard rejection handled above)
    if cash is not None and total is not None:
        score += _LAW_WEIGHTS["law3_cash_gte_total"]
        satisfied.append("law3_cash_gte_total")

    # Law 6: cash should appear below total in the document
    if a.cash_paid is not None and a.total is not None:
        if a.cash_paid.zone_ratio > a.total.zone_ratio:
            score += _LAW_WEIGHTS["law6_cash_below_total"]
            satisfied.append("law6_cash_below_total")
        else:
            score -= 1.0  # ordering violation

    # Law 7: tax < subtotal (typical for ≤20% tax rates)
    if tax is not None and subtotal is not None and tax < subtotal:
        score += _LAW_WEIGHTS["law7_tax_lt_subtotal"]
        satisfied.append("law7_tax_lt_subtotal")

    # Law 8: total appears in lower half of document
    if a.total is not None and a.total.zone_ratio >= 0.50:
        score += _LAW_WEIGHTS["law8_total_in_lower_half"]
        satisfied.append("law8_total_in_lower_half")

    # Label evidence bonuses
    role_map = {
        "subtotal": a.subtotal,
        "tax": a.tax,
        "total": a.total,
        "cash_paid": a.cash_paid,
        "change": a.change,
    }
    for expected_role, ev in role_map.items():
        if ev is None:
            continue
        if ev.label_role == expected_role:
            score += 1.0 * ev.ocr_confidence
        elif ev.label_role == "unknown":
            score += 0.2 * ev.ocr_confidence
        else:
            # Label says a different role — mild penalty
            score -= 0.5

    return score, satisfied


def find_best_assignment(
    evidence: list[AmountEvidence],
    tolerance: Decimal = _TOLERANCE,
) -> FinancialAssignment | None:
    """Return the highest-scoring financially consistent assignment."""
    if not evidence:
        return None

    def _can_be_total(ev: AmountEvidence) -> bool:
        if ev.label_role in ("cash_paid", "change", "item", "tax", "subtotal", "rounding"):
            return False
        if ev.label_role == "unknown" and ev.zone_ratio < 0.45:
            return False
        return True

    def _can_be_subtotal(ev: AmountEvidence) -> bool:
        return ev.label_role not in ("cash_paid", "change", "item", "rounding", "tax", "total", "rounded_total")

    def _can_be_tax(ev: AmountEvidence) -> bool:
        return ev.label_role not in ("cash_paid", "change", "item", "rounding", "subtotal", "total", "rounded_total")

    def _can_be_cash(ev: AmountEvidence) -> bool:
        return ev.label_role not in ("tax", "subtotal", "change", "item", "rounding")

    def _can_be_change(ev: AmountEvidence) -> bool:
        return ev.label_role not in ("tax", "subtotal", "total", "item", "cash_paid")

    total_candidates = [e for e in evidence if _can_be_total(e)]
    total_candidates.sort(
        key=lambda e: (
            0 if e.label_role == "total" else (1 if e.zone_ratio >= 0.50 else 2),
            -e.ocr_confidence,
        )
    )

    best_assignment: FinancialAssignment | None = None
    best_score: float = -999.0

    for total_ev in total_candidates:
        T = total_ev.value

        # Allow subtotal == total for zero-tax receipts.
        sub_candidates = [e for e in evidence if _can_be_subtotal(e) and e.value <= T and e is not total_ev]
        tax_candidates = [e for e in evidence if _can_be_tax(e) and e.value < T and e is not total_ev]

        for sub_ev in sub_candidates:
            for tax_ev in tax_candidates:
                if sub_ev is tax_ev:
                    continue
                if abs(sub_ev.value + tax_ev.value - T) > tolerance:
                    continue
                if sub_ev.value <= tax_ev.value:
                    sub_ev, tax_ev = tax_ev, sub_ev

                a = FinancialAssignment(
                    subtotal=sub_ev,
                    tax=tax_ev,
                    total=total_ev,
                )

                cash_candidates = [
                    e for e in evidence
                    if _can_be_cash(e) and e.value >= T and e is not total_ev
                ]
                change_candidates = [
                    e for e in evidence
                    if _can_be_change(e) and e.value >= Decimal("0") and e is not total_ev
                ]

                extended = False
                for cash_ev in cash_candidates:
                    C = cash_ev.value
                    for change_ev in change_candidates:
                        if change_ev is cash_ev:
                            continue
                        if abs(C - T - change_ev.value) <= _EXACT_TOLERANCE:
                            a_full = FinancialAssignment(
                                subtotal=sub_ev,
                                tax=tax_ev,
                                total=total_ev,
                                cash_paid=cash_ev,
                                change=change_ev,
                            )
                            s, laws = _score_assignment(a_full)
                            if s > best_score:
                                a_full.consistency_score = s
                                a_full.satisfied_laws = laws
                                best_assignment = a_full
                                best_score = s
                            extended = True

                if not extended:
                    for cash_ev in cash_candidates:
                        a_partial = FinancialAssignment(
                            subtotal=sub_ev,
                            tax=tax_ev,
                            total=total_ev,
                            cash_paid=cash_ev,
                        )
                        s, laws = _score_assignment(a_partial)
                        if s > best_score:
                            a_partial.consistency_score = s
                            a_partial.satisfied_laws = laws
                            best_assignment = a_partial
                            best_score = s

                    s, laws = _score_assignment(a)
                    if s > best_score:
                        a.consistency_score = s
                        a.satisfied_laws = laws
                        best_assignment = a
                        best_score = s

    return best_assignment if (best_assignment and best_assignment.consistency_score > 0.5) else None


def assignment_to_field_candidates(assignment: FinancialAssignment) -> dict:
    """Convert an assignment into extraction candidates."""
    from .extraction import FieldCandidate

    result: dict = {}
    laws_str = "; ".join(assignment.satisfied_laws)

    field_map = [
        ("total_amount", assignment.total,    "total_amount"),
        ("tax_amount",   assignment.tax,      "tax_amount"),
        ("subtotal",     assignment.subtotal, "subtotal"),
    ]

    # Weak evidence should not outscore a clean labeled candidate.
    present = [e for e in (assignment.subtotal, assignment.tax, assignment.total,
                           assignment.cash_paid, assignment.change) if e is not None]
    weakest = min((e.ocr_confidence for e in present), default=0.5)

    for field_name, ev, role in field_map:
        if ev is None:
            continue
        score = 0.90 + ev.ocr_confidence * 0.09
        score = min(score, 0.55 + weakest * 0.40)
        method = "constraint_repaired" if getattr(ev, "is_repair", False) else "financial_constraint_engine"
        repair_note = (
            f"; OCR-repair ({ev.repair_edits} edits applied to damaged token)"
            if getattr(ev, "is_repair", False) else ""
        )
        result[field_name] = FieldCandidate(
            field=field_name,
            value=ev.normalized,
            score=score,
            source_text=ev.source_text,
            source_bbox=ev.bbox,
            method=method,
            explanation=(
                f"Assigned by financial constraint engine — "
                f"financial consistency score {assignment.consistency_score:.2f}; "
                f"laws satisfied: {laws_str}{repair_note}"
            ),
            amount_role=role,
        )

    return result
