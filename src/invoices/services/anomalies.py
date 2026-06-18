from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation


@dataclass
class Anomaly:
    code: str
    message: str
    severity: str = "warning"

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "severity": self.severity}


def _amount(value) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value).replace(",", "").replace("$", "").replace("¥", "").strip())
    except (InvalidOperation, ValueError):
        return None


def _parse_date(value: str) -> date | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y", "%d/%m/%y", "%m-%d-%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def check_invoice_anomalies(
    fields: dict,
    confidences: dict | None = None,
    previous_invoice_numbers: set[str] | None = None,
    minimum_confidence: float = 0.50,
    evidence: dict | None = None,
) -> list[Anomaly]:
    confidences = confidences or {}
    previous_invoice_numbers = previous_invoice_numbers or set()
    evidence = evidence or {}
    anomalies: list[Anomaly] = []

    required = ["invoice_number", "date", "vendor_name", "total_amount", "currency"]
    for field in required:
        if not fields.get(field):
            anomalies.append(Anomaly("missing_required_field", f"Missing required field: {field}.", "error"))

    invoice_number = fields.get("invoice_number", "")
    if invoice_number and invoice_number in previous_invoice_numbers:
        anomalies.append(Anomaly("duplicate_invoice_number", f"Invoice number {invoice_number} already exists.", "error"))

    invoice_date = _parse_date(fields.get("date", ""))
    if fields.get("date") and invoice_date is None:
        anomalies.append(Anomaly("invalid_date", "Invoice date format is invalid.", "error"))
    elif invoice_date and invoice_date > date.today():
        anomalies.append(Anomaly("future_date", "Invoice date is later than the processing date.", "warning"))

    subtotal = _amount(fields.get("subtotal"))
    tax = _amount(fields.get("tax_amount"))
    total = _amount(fields.get("total_amount"))
    if tax is not None and total is not None and tax >= total:
        anomalies.append(Anomaly("tax_greater_than_total", "Tax amount is greater than or equal to total amount.", "error"))
    if subtotal is not None and tax is not None and total is not None and abs((subtotal + tax) - total) > Decimal("0.02"):
        anomalies.append(Anomaly("subtotal_tax_total_mismatch", "Subtotal plus tax does not match total amount.", "error"))

    # Financial inconsistency: subtotal + tax present but significantly imbalanced with total
    if subtotal is not None and tax is not None and total is not None and abs((subtotal + tax) - total) > Decimal("0.10"):
        anomalies.append(Anomaly("financial_inconsistency",
            "Subtotal, tax, and total do not satisfy standard financial relationships.", "error"))

    # Cash/change consistency: cash - total should equal change (when all available via evidence)
    cash_val = _amount((evidence.get("cash_paid") or {}).get("value", ""))
    change_val = _amount((evidence.get("change") or {}).get("value", ""))
    if cash_val is not None and total is not None and change_val is not None:
        if abs((cash_val - total) - change_val) > Decimal("0.05"):
            anomalies.append(Anomaly("cash_change_mismatch",
                "Cash paid minus total does not equal the change amount — possible misclassification.", "warning"))

    # Warn when extraction evidence shows total or tax came from a payment/cash line
    total_role = evidence.get("total_amount", {}).get("amount_role", "")
    if total_role == "cash_paid":
        anomalies.append(
            Anomaly(
                "total_from_payment_line",
                "Total amount was extracted from a Cash/Payment line — likely incorrect.",
                "error",
            )
        )
    tax_role = evidence.get("tax_amount", {}).get("amount_role", "")
    if tax_role == "cash_paid":
        anomalies.append(
            Anomaly(
                "tax_from_payment_line",
                "Tax amount was extracted from a Cash/Payment line — likely incorrect.",
                "error",
            )
        )
    # Warn when tax equals cash paid (common misclassification signal)
    if tax is not None and total is not None and abs(tax - total) < Decimal("0.01"):
        anomalies.append(
            Anomaly(
                "tax_equals_total",
                "Tax amount equals total amount, which is unusual and may indicate misclassification.",
                "warning",
            )
        )

    vat_rate = fields.get("vat_rate", "")
    if vat_rate:
        try:
            rate = Decimal(str(vat_rate).replace("%", "").strip())
            if rate < 0 or rate > 100:
                anomalies.append(Anomaly("invalid_vat_rate", "VAT/tax rate is outside the valid percentage range.", "error"))
        except InvalidOperation:
            anomalies.append(Anomaly("invalid_vat_rate", "VAT/tax rate format is invalid.", "warning"))

    for index, item in enumerate(fields.get("line_items", []) or [], start=1):
        qty = _amount(item.get("qty"))
        unit_price = _amount(item.get("unit_price"))
        line_total = _amount(item.get("line_total"))
        if qty is not None and unit_price is not None and line_total is not None:
            if abs((qty * unit_price) - line_total) > Decimal("0.02"):
                anomalies.append(
                    Anomaly(
                        "line_item_total_mismatch",
                        f"Line item {index} quantity multiplied by unit price does not match line total.",
                        "error",
                    )
                )

    for field, confidence in confidences.items():
        if field in required and float(confidence or 0.0) < minimum_confidence:
            anomalies.append(Anomaly("low_confidence_key_field", f"Low confidence for key field: {field}.", "warning"))

    if fields.get("currency") and fields.get("currency") not in {"USD", "CNY", "RMB", "EUR", "GBP", "RM", "MYR", "NGN"}:
        anomalies.append(Anomaly("unsupported_currency", f"Currency {fields.get('currency')} is not recognized.", "warning"))

    numeric_amount = _amount(fields.get("total_amount"))
    amount_words = fields.get("amount_in_words", "")
    if numeric_amount is not None and amount_words:
        if "HUNDRED" in amount_words.upper() and numeric_amount < Decimal("100"):
            anomalies.append(Anomaly("amount_words_numeric_mismatch", "Amount in words appears inconsistent with numeric total.", "warning"))

    return anomalies
