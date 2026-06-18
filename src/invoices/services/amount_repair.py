"""Repair OCR-damaged monetary values using common glyph substitutions."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from itertools import combinations
from typing import Iterator

# Characters commonly confused with each digit.
_CONFUSION: dict[str, str] = {
    "0": "0OoDQ",
    "1": "1IlJ|T]",
    "2": "2Zz",
    "3": "38B",
    "4": "4A",
    "5": "5Ss",
    "6": "68Gb",
    "7": "71T?",
    "8": "8B63&",
    "9": "9gq",
}

_GLYPH_TO_DIGITS: dict[str, set[str]] = {}
for _digit, _glyphs in _CONFUSION.items():
    for _g in _glyphs:
        _GLYPH_TO_DIGITS.setdefault(_g, set()).add(_digit)


def _confusable(glyph: str, digit: str) -> bool:
    """True when OCR might produce `glyph` for the true digit `digit`."""
    return glyph == digit or digit in _GLYPH_TO_DIGITS.get(glyph, ())


def token_matches_value(raw_token: str, target: float, max_edits: int = 2) -> tuple[bool, int]:
    """Check whether a damaged token can represent the target value."""
    cand = raw_token.strip().replace(",", ".")
    cand = re.sub(r"[^0-9A-Za-z.|]", "", cand)
    if "." not in cand:
        return (False, 99)
    t_int, t_dec = f"{float(target):.2f}".split(".")
    parts = cand.rsplit(".", 1)
    if len(parts) != 2:
        return (False, 99)
    c_int, c_dec = parts
    if len(c_int) != len(t_int) or len(c_dec) != len(t_dec):
        return (False, 99)
    edits = 0
    for glyph, digit in zip(c_int + c_dec, t_int + t_dec):
        if glyph == digit:
            continue
        if _confusable(glyph, digit):
            edits += 1
        else:
            return (False, 99)
    return (edits <= max_edits, edits)


def enumerate_repairs(amount_str: str, max_edits: int = 2) -> list[tuple[str, int]]:
    """Return valid two-decimal repairs and their edit counts."""
    s = amount_str.strip().replace(",", ".")
    s = re.sub(r"[^0-9A-Za-z.|]", "", s)
    if "." not in s:
        return []
    dot_pos = s.rindex(".")
    int_part = s[:dot_pos]
    dec_part = s[dot_pos + 1:]
    if len(dec_part) != 2 or not int_part:
        return []

    all_chars = list(int_part + dec_part)
    n_int = len(int_part)

    alts: list[tuple[str, list[str]]] = []
    for ch in all_chars:
        ch_upper = ch.upper()
        candidates = sorted(_GLYPH_TO_DIGITS.get(ch_upper, set()) | _GLYPH_TO_DIGITS.get(ch, set()))
        diff = [d for d in candidates if d != ch]
        alts.append((ch, diff))

    results: list[tuple[str, int]] = []

    def _reconstruct(changes: list[tuple[int, str]]) -> str | None:
        chars = all_chars[:]
        for pos, new_ch in changes:
            chars[pos] = new_ch
        if not all(c.isdigit() for c in chars):
            return None
        return "".join(chars[:n_int]) + "." + "".join(chars[n_int:])

    for i, (ch, diff_digits) in enumerate(alts):
        for new_digit in diff_digits:
            repaired = _reconstruct([(i, new_digit)])
            if repaired:
                results.append((repaired, 1))

    if max_edits >= 2:
        suspicious = [i for i, (_, diff) in enumerate(alts) if diff]
        for i, j in combinations(suspicious, 2):
            for d_i in alts[i][1]:
                for d_j in alts[j][1]:
                    repaired = _reconstruct([(i, d_i), (j, d_j)])
                    if repaired:
                        results.append((repaired, 2))

    seen: dict[str, int] = {}
    for val, edits in results:
        if val not in seen or edits < seen[val]:
            seen[val] = edits
    return sorted(seen.items(), key=lambda x: x[1])


def propose_missing_amounts(
    known: dict[str, Decimal | None],
    rate: Decimal | None = None,
) -> list[tuple[str, float, str]]:
    """Derive missing amount fields from known financial relationships."""
    proposals: list[tuple[str, float, str]] = []
    cash = known.get("cash_paid")
    change = known.get("change")
    sub = known.get("subtotal")
    tax = known.get("tax_amount")
    tot = known.get("total_amount")

    if tot is None and cash is not None and change is not None:
        proposals.append(("total_amount", float(cash - change), "law2_cash_change"))
    if tot is None and sub is not None and tax is not None:
        proposals.append(("total_amount", float(sub + tax), "law1_subtotal_tax_total"))
    if sub is None and tot is not None and tax is not None:
        proposals.append(("subtotal", float(tot - tax), "law1_subtotal_tax_total"))
    if tax is None and tot is not None and sub is not None:
        proposals.append(("tax_amount", float(tot - sub), "law1_subtotal_tax_total"))
    if rate is not None and tot is not None and sub is None:
        taxable = round(float(tot) / (1 + float(rate)), 2)
        proposals.append(("subtotal", taxable, "law_gst_inclusive"))
        proposals.append(("tax_amount", round(float(tot) - taxable, 2), "law_gst_inclusive"))

    return proposals
