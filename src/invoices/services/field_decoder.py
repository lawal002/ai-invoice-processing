"""Resolve field candidates with type checks and token-level exclusivity."""

from __future__ import annotations

import re
from typing import Callable


def _token_key(candidate) -> tuple:
    """Return a stable identity for a candidate's source token."""
    bb = getattr(candidate, "source_bbox", None)
    if bb and len(bb) >= 4:
        # Ignore small bounding-box jitter.
        return ("box", round(bb[0] / 5) * 5, round(bb[1] / 5) * 5)
    val = re.sub(r"\s+", "", (candidate.value or "").lower())
    return ("text", val)


def _is_amount(v: str) -> bool:
    return bool(re.fullmatch(r"\d{1,7}([.,]\d{1,2})?", (v or "").strip()))


def _build_type_ok() -> dict[str, Callable[[str], bool]]:
    # Lazy import avoids a circular dependency.
    from .extraction import looks_like_date

    return {
        "invoice_number": lambda v: (
            not looks_like_date(v)
            and bool(re.search(r"[A-Za-z0-9]", v or ""))
            and len((v or "").strip()) >= 3
        ),
        "date": looks_like_date,
        "subtotal": _is_amount,
        "tax_amount": _is_amount,
        "total_amount": _is_amount,
        "vat_rate": lambda v: bool(re.search(r"\d", v or "")),
    }


def decode_fields(
    candidates_by_field: dict[str, list],
    top_k: int = 3,
) -> dict[str, object]:
    """Choose the highest-ranked valid, unclaimed candidate for each field."""
    TYPE_OK = _build_type_ok()

    pool: dict[str, list] = {
        f: [c for c in cs if TYPE_OK.get(f, lambda v: True)(c.value)][:top_k]
        for f, cs in candidates_by_field.items()
    }

    chosen: dict[str, object] = {f: None for f in candidates_by_field}
    claimed: dict[tuple, str] = {}

    field_order = sorted(
        pool,
        key=lambda f: -(pool[f][0].score if pool[f] else 0.0),
    )

    for field_name in field_order:
        for candidate in pool[field_name]:
            key = _token_key(candidate)
            if key in claimed:
                continue
            chosen[field_name] = candidate
            claimed[key] = field_name
            break

    return chosen
