"""Regression tests built from real and minimally adapted OCR tokens."""

from __future__ import annotations

import os
import json
import tempfile
from decimal import Decimal
from unittest.mock import patch

from django.test import SimpleTestCase

from .services.amount_repair import enumerate_repairs, token_matches_value
from .services.evaluation import values_equal
from .services.extraction import extract_layout_aware, looks_like_date
from .services.ocr import OCRTokenData


class LooksLikeDateTests(SimpleTestCase):
    """Date detection must reject invoice-like identifiers."""

    def test_dd_slash_mm_slash_yyyy(self):
        self.assertTrue(looks_like_date("22/12/2017"))

    def test_dd_slash_mm_slash_yyyy_with_time(self):
        self.assertTrue(looks_like_date("22/12/2017 14.03"))

    def test_mm_slash_dd_slash_yyyy(self):
        self.assertTrue(looks_like_date("01/31/2025"))

    def test_yyyy_dash_mm_dash_dd(self):
        self.assertTrue(looks_like_date("2025-01-31"))

    def test_dd_dash_mm_dash_yy(self):
        self.assertTrue(looks_like_date("20-11-17"))

    def test_alphanumeric_invoice_number_not_date(self):
        self.assertFalse(looks_like_date("CS67332"))

    def test_long_slash_number_not_date(self):
        self.assertFalse(looks_like_date("18222/102/70341"))

    def test_plain_invoice_number_not_date(self):
        self.assertFalse(looks_like_date("INV-2026-0001"))

    def test_short_standalone_number_not_date(self):
        self.assertFalse(looks_like_date("562936"))


class AmountRepairUnitTests(SimpleTestCase):
    """Tests for common OCR substitutions in monetary values."""

    def test_J89_comma_80_repairs_to_169_dot_80(self):
        repairs = enumerate_repairs("J89,80")
        repair_values = [v for v, _ in repairs]
        self.assertIn(
            "169.80",
            repair_values,
            f"enumerate_repairs('J89,80') must produce '169.80'; got {repair_values}",
        )

    def test_J89_comma_80_repair_edit_count_is_2(self):
        repairs = dict(enumerate_repairs("J89,80"))
        self.assertIn("169.80", repairs)
        self.assertEqual(
            repairs["169.80"],
            2,
            "J89,80 → 169.80 requires exactly 2 confusable-substitution edits",
        )

    def test_token_matches_value_J89_80(self):
        ok, edits = token_matches_value("J89,80", 169.80, max_edits=2)
        self.assertTrue(ok, "token_matches_value must accept 'J89,80' as 169.80")
        self.assertEqual(edits, 2)

    def test_clean_amount_no_1edit_repairs(self):
        repairs = enumerate_repairs("169.80")
        for val, _ in repairs:
            self.assertNotEqual(
                val, "169.80",
                "enumerate_repairs must not include the input value itself",
            )

    def test_common_glyph_confusion_B_to_8(self):
        repairs = dict(enumerate_repairs("1B9.80"))
        self.assertIn("189.80", repairs, "B→8 confusion must produce 189.80")
        self.assertEqual(repairs["189.80"], 1)

    def test_common_glyph_confusion_O_to_0(self):
        repairs = dict(enumerate_repairs("1O9.80"))
        self.assertIn("109.80", repairs)
        self.assertEqual(repairs["109.80"], 1)


class Doc27InvoiceDateRegressionTests(SimpleTestCase):
    """Keep the invoice number, date, and vendor fields separate."""

    # Captured OCR tokens from the original receipt.
    TOKENS = [
        OCRTokenData("HOME MASTeR KARDWARE &",       [143.0, 41.0,  568.0, 90.0],  0.348),
        OCRTokenData("ELECTRICAL",                   [258.0, 80.0,  449.0, 132.0], 0.935),
        OCRTokenData("Na.113G & 115G, JALAN SETIA GEMBILANG",
                                                      [78.0, 113.0, 630.0, 165.0], 0.387),
        OCRTokenData("UtjeG DANDAR SETIA ALAM,",     [165.0, 157.0, 545.0, 200.0], 0.417),
        OCRTokenData("40170 BANDAR SETIA ALAM,",     [173.0, 198.0, 535.0, 239.0], 0.731),
        OCRTokenData("SELANIGOR",                    [270.0, 235.0, 431.0, 274.0], 0.548),
        OCRTokenData("Compzny Reg Nj. ;540371t551-P",[144.0, 275.0, 558.0, 323.0], 0.324),
        OCRTokenData("GST Rey No. \"0016375+1168",   [175.0, 323.0, 525.0, 363.0], 0.461),
        OCRTokenData("Tfk Icice",                    [262.0, 380.0, 436.0, 412.0], 0.194),
        OCRTokenData("Involce Nlo::",                [ 59.0, 425.0, 205.0, 461.0], 0.586),
        OCRTokenData("CS67332",                      [248.0, 430.0, 370.0, 462.0], 0.647),
        OCRTokenData("Date",                         [ 60.0, 468.0, 124.0, 500.0], 0.943),
        OCRTokenData("22/12/2017 14.03",             [244.0, 469.0, 455.0, 508.0], 0.717),
        OCRTokenData("Czahler#;",                    [ 57.0, 508.0, 193.0, 549.0], 0.689),
        OCRTokenData("CaShIER",                      [246.0, 515.0, 368.0, 547.0], 0.229),
        OCRTokenData("RM",                           [463.0, 563.0, 521.0, 599.0], 0.459),
        OCRTokenData("Subtota",                      [182.0, 694.0, 286.0, 726.0], 0.425),
        OCRTokenData("15.90",                        [446.0, 694.0, 518.0, 726.0], 0.928),
        OCRTokenData("Total Excl , 0r GST",          [ 83.0, 739.0, 297.0, 777.0], 0.470),
        OCRTokenData("t5.00",                        [444.0, 746.0, 516.0, 776.0], 0.720),
        OCRTokenData("Totz' Ingi, 0f GST",           [ 91.0, 783.0, 297.0, 821.0], 0.416),
        OCRTokenData("15.90",                        [444.0, 788.0, 516.0, 820.0], 0.925),
        OCRTokenData("Total Amt Rounded",            [ 69.0, 829.0, 297.0, 867.0], 0.882),
        OCRTokenData("15.90",                        [442.0, 834.0, 514.0, 866.0], 0.739),
        OCRTokenData("Payment",                      [170.0, 870.0, 285.0, 910.0], 0.941),
        OCRTokenData("50.00",                        [440.0, 872.0, 514.0, 904.0], 0.593),
        OCRTokenData("Change Duz",                   [127.0, 906.0, 283.0, 951.0], 0.949),
        OCRTokenData("34.10",                        [440.0, 912.0, 512.0, 944.0], 0.972),
        OCRTokenData("6%",                           [188.0, 1058.0, 232.0, 1088.0], 0.549),
        OCRTokenData("15.00",                        [352.0, 1060.0, 424.0, 1090.0], 0.577),
        OCRTokenData("0.90",                         [532.0, 1060.0, 592.0, 1090.0], 0.548),
    ]

    def _run(self):
        return extract_layout_aware(self.TOKENS)

    def test_invoice_number_is_CS67332_not_date(self):
        """Use the value beside the invoice label, not the date below it."""
        result = self._run()
        inv = result.fields.get("invoice_number", "")
        self.assertEqual(
            inv, "CS67332",
            f"invoice_number must be 'CS67332' after date filter; got {inv!r}",
        )

    def test_invoice_number_differs_from_date(self):
        """invoice_number and date must never hold the same value."""
        result = self._run()
        self.assertNotEqual(
            result.fields.get("invoice_number", ""),
            result.fields.get("date", ""),
            "invoice_number and date must not be the same token",
        )

    def test_date_is_22_12_2017(self):
        """The date field must capture '22/12/2017' from the Date label."""
        result = self._run()
        date_val = result.fields.get("date", "")
        self.assertIn(
            "22/12/2017", date_val,
            f"date must contain '22/12/2017'; got {date_val!r}",
        )

    def test_vendor_is_the_business_name_not_address(self):
        """Prefer the receipt header over address lines."""
        result = self._run()
        vendor = result.fields.get("vendor_name", "").upper()
        self.assertIn(
            "KARDWARE", vendor,
            f"vendor must be from the business-name line; got {vendor!r}",
        )

    def test_vendor_is_not_address_line(self):
        """No address component (JALAN, BANDAR, SETIA ALAM) must win vendor."""
        result = self._run()
        vendor = result.fields.get("vendor_name", "").upper()
        for kw in ("JALAN", "BANDAR SETIA", "SELANGOR", "SELANIGOR"):
            self.assertNotIn(
                kw, vendor,
                f"address keyword '{kw}' found in vendor; got {vendor!r}",
            )

    def test_currency_is_RM(self):
        result = self._run()
        self.assertEqual(result.fields.get("currency", ""), "RM")


class VendorAmplersandAssemblyTests(SimpleTestCase):
    """Tests for multi-line business names ending with an ampersand."""

    def test_ampersand_line_joins_continuation(self):
        """Join a business-name continuation on the following line."""
        tokens = [
            OCRTokenData("HOME MASTeR KARDWARE &",  [143, 41,  568,  90],  0.348),
            OCRTokenData("ELECTRICAL",              [258, 95,  449, 140],  0.935),
            OCRTokenData("Na.113G JALAN SETIA GEMBILANG", [78, 165, 630, 210], 0.387),
            OCRTokenData("40170 BANDAR SETIA ALAM",        [173, 215, 535, 255], 0.731),
            OCRTokenData("Involce Nlo::",           [ 59, 420,  205, 460],  0.586),
            OCRTokenData("CS67332",                 [248, 425,  370, 462],  0.647),
            OCRTokenData("Date",                    [ 60, 465,  124, 500],  0.943),
            OCRTokenData("22/12/2017 14.03",        [244, 465,  455, 508],  0.717),
            OCRTokenData("Total Amt Rounded",       [ 69, 829,  297, 867],  0.882),
            OCRTokenData("15.90",                   [442, 834,  514, 866],  0.739),
            OCRTokenData("RM",                      [463, 563,  521, 599],  0.459),
        ]
        result = extract_layout_aware(tokens)
        vendor = result.fields.get("vendor_name", "")

        self.assertIn(
            "ELECTRICAL", vendor.upper(),
            f"vendor assembly must join the ampersand continuation; got {vendor!r}",
        )
        self.assertFalse(
            vendor.rstrip().endswith("&"),
            f"assembled vendor must not end with '&'; got {vendor!r}",
        )
        self.assertIn("KARDWARE", vendor.upper())

    def test_assembled_vendor_excludes_address_after_amp(self):
        """Do not join an address after a trailing ampersand."""
        tokens = [
            OCRTokenData("HOME MASTeR KARDWARE &",        [143, 41,  568,  90], 0.348),
            OCRTokenData("Na.113G JALAN SETIA GEMBILANG", [ 78, 100, 630, 145], 0.387),
            OCRTokenData("Total Amt Rounded",             [ 69, 400, 297, 435], 0.882),
            OCRTokenData("15.90",                         [442, 405, 514, 440], 0.739),
            OCRTokenData("RM",                            [463, 350, 521, 385], 0.459),
        ]
        result = extract_layout_aware(tokens)
        vendor = result.fields.get("vendor_name", "").upper()

        self.assertNotIn(
            "JALAN", vendor,
            f"address line after '&' must NOT be joined into vendor; got {vendor!r}",
        )


class Doc21DamagedTotalRegressionTests(SimpleTestCase):
    """Repair a damaged total when payment values confirm it."""

    TOKENS = [
        # Vendor (top of receipt)
        OCRTokenData("99 SPEED MART S/B (517537-X)",  [ 35, 35,  500,  65], 0.70),
        # Invoice label + number
        OCRTokenData("INVOICE NO",                    [ 35, 155, 180, 182], 0.72),
        OCRTokenData("18222/102/70341",               [250, 155, 500, 182], 0.92),
        # Date
        OCRTokenData("20-11-17",                      [ 35, 195, 155, 222], 0.63),
        # GST Summary — clean values for subtotal / tax
        OCRTokenData("GST Summary",                   [ 35, 460, 210, 490], 0.80),
        OCRTokenData("Amount",                        [235, 500, 325, 528], 0.76),
        OCRTokenData("Tax",                           [420, 500, 470, 528], 0.76),
        OCRTokenData("160.17",                        [235, 538, 330, 567], 0.82),   # subtotal
        OCRTokenData("9.61",                          [420, 538, 488, 567], 0.81),   # tax
        # Total line — J89,80 is the OCR-damaged form of 169.80 (J→1, 8→6, 2 edits)
        OCRTokenData("Total",                         [ 35, 635, 155, 664], 0.85),
        OCRTokenData("RM",                            [330, 635, 370, 664], 0.80),
        OCRTokenData("J89,80",                        [420, 635, 515, 664], 0.24),   # damaged
        # Cash / change — clean values so law2 validates the repair
        OCRTokenData("Cash",                          [ 35, 735, 105, 763], 0.82),
        OCRTokenData("200.00",                        [420, 735, 515, 763], 0.84),
        OCRTokenData("Change",                        [ 35, 780, 125, 808], 0.83),
        OCRTokenData("30.20",                         [420, 780, 500, 808], 0.84),
    ]

    def _run(self):
        return extract_layout_aware(self.TOKENS)

    def test_total_is_not_cash_payment(self):
        """Cash payment 200.00 must never become total_amount."""
        result = self._run()
        self.assertNotEqual(
            result.fields.get("total_amount", ""), "200.00",
            "Cash 200.00 must not be selected as total when a repaired total exists",
        )

    def test_total_repaired_to_169_80(self):
        """Damaged J89,80 must be repaired to 169.80 by the constraint engine."""
        result = self._run()
        self.assertEqual(
            result.fields.get("total_amount", ""), "169.80",
            f"total_amount must be repaired to '169.80'; got "
            f"{result.fields.get('total_amount', '')!r}",
        )

    def test_total_evidence_method_is_constraint_repaired(self):
        """Evidence method for the repaired total must be 'constraint_repaired'."""
        result = self._run()
        method = result.evidence.get("total_amount", {}).get("method", "")
        self.assertEqual(
            method, "constraint_repaired",
            f"evidence method for repaired total must be 'constraint_repaired'; got {method!r}",
        )

    def test_tax_is_9_61_not_cash(self):
        """Tax must come from GST Summary, not from the cash-payment area."""
        result = self._run()
        self.assertNotEqual(result.fields.get("tax_amount", ""), "200.00")
        self.assertEqual(
            result.fields.get("tax_amount", ""), "9.61",
            f"tax_amount must be 9.61; got {result.fields.get('tax_amount', '')!r}",
        )


class AnomalyEvidenceWiringTests(SimpleTestCase):
    """Verify that extraction evidence reaches anomaly checks."""

    TOKENS = [
        OCRTokenData("TEST STORE SDN BHD",   [50, 40, 300, 70],   0.92),
        OCRTokenData("Invoice No",           [50, 120, 200, 148],  0.90),
        OCRTokenData("INV-999",              [210, 120, 340, 148], 0.90),
        OCRTokenData("Date",                 [50, 160, 120, 188],  0.90),
        OCRTokenData("2026-06-01",           [130, 160, 310, 188], 0.90),
        # No explicit total is present.
        OCRTokenData("Cash",                 [50, 700, 130, 730],  0.90),
        OCRTokenData("RM 200.00",            [350, 700, 520, 730], 0.90),
        OCRTokenData("Change",               [50, 745, 145, 775],  0.90),
        OCRTokenData("RM 30.20",             [350, 745, 490, 775], 0.90),
    ]

    def test_evidence_dict_populated_for_amount_fields(self):
        """extract_layout_aware must populate evidence for every extracted amount."""
        result = extract_layout_aware(self.TOKENS)
        for field in ("total_amount", "tax_amount"):
            val = result.fields.get(field, "")
            if val:
                self.assertIn(
                    field, result.evidence,
                    f"result.evidence must contain an entry for {field}",
                )
                self.assertIn(
                    "method", result.evidence[field],
                    f"evidence for {field} must have a 'method' key",
                )

    def test_total_from_payment_anomaly_fires_when_role_is_cash(self):
        """If total_amount evidence has amount_role='cash_paid', the anomaly detector
        must raise 'total_from_payment_line'."""
        from .services.anomalies import check_invoice_anomalies

        evidence = {
            "total_amount": {"amount_role": "cash_paid", "value": "200.00"},
        }
        anomalies = check_invoice_anomalies(
            {
                "invoice_number": "INV-999",
                "date": "2026-06-01",
                "vendor_name": "TEST STORE SDN BHD",
                "total_amount": "200.00",
                "currency": "RM",
            },
            evidence=evidence,
        )
        codes = {a.code for a in anomalies}
        self.assertIn(
            "total_from_payment_line", codes,
            f"anomaly 'total_from_payment_line' must fire; got codes={codes}",
        )


class ContextBoundaryStopTests(SimpleTestCase):
    """Stop invoice-number context at the next labeled field."""

    TOKENS = [
        OCRTokenData("DEMO CORP SDN BHD",  [50, 40,  320, 70],  0.91),
        OCRTokenData("Invoice No",         [50, 140, 200, 168], 0.90),
        OCRTokenData("ABCD-1234",          [210, 140, 370, 168], 0.88),
        OCRTokenData("Date",               [50, 175, 120, 203], 0.92),
        OCRTokenData("15/03/2025",         [130, 175, 320, 203], 0.89),
        OCRTokenData("Total",              [50, 600, 140, 628], 0.93),
        OCRTokenData("RM 500.00",          [350, 600, 510, 628], 0.92),
    ]

    def test_invoice_number_stops_before_date_label(self):
        """Invoice extraction must not cross the Date label and pick up 15/03/2025."""
        result = extract_layout_aware(self.TOKENS)
        inv = result.fields.get("invoice_number", "")
        self.assertFalse(
            looks_like_date(inv),
            f"invoice_number {inv!r} looks like a date — context boundary stop failed",
        )
        self.assertEqual(
            inv, "ABCD-1234",
            f"invoice_number must be 'ABCD-1234'; got {inv!r}",
        )

    def test_date_captured_from_date_label(self):
        result = extract_layout_aware(self.TOKENS)
        self.assertIn("2025", result.fields.get("date", ""))


class HassanbistroVendorRegressionTests(SimpleTestCase):
    """Keep a large vendor heading separate from the address below it."""

    TOKENS = [
        OCRTokenData("RESTORAN HASSANBISTRO",    [35, 20,  400, 55],  0.87),
        OCRTokenData("NO,2-1-1 JALAN SETIA",    [35, 58,  350, 80],  0.72),
        OCRTokenData("PUCHONG, 47100",           [35, 85,  250, 105], 0.80),
        OCRTokenData("SST REG NO:",              [35, 125, 200, 145], 0.76),
        OCRTokenData("B16-2308-32010938",        [210, 125, 430, 145], 0.81),
        OCRTokenData("Invoice No",               [35, 180, 180, 205], 0.88),
        OCRTokenData("RCP-001",                  [195, 180, 320, 205], 0.89),
        OCRTokenData("Date",                     [35, 220, 110, 245], 0.90),
        OCRTokenData("15/03/2023",               [120, 220, 300, 245], 0.91),
        OCRTokenData("Subtotal",                 [35, 580, 160, 605], 0.90),
        OCRTokenData("RM",                       [320, 580, 360, 605], 0.88),
        OCRTokenData("42.00",                    [370, 580, 445, 605], 0.91),
        OCRTokenData("SST 6%",                   [35, 618, 135, 643], 0.89),
        OCRTokenData("RM",                       [320, 618, 360, 643], 0.88),
        OCRTokenData("2.52",                     [370, 618, 445, 643], 0.90),
        OCRTokenData("Total",                    [35, 655, 120, 680], 0.90),
        OCRTokenData("RM",                       [320, 655, 360, 680], 0.91),
        OCRTokenData("44.52",                    [370, 655, 445, 680], 0.93),
        OCRTokenData("Cash",                     [35, 695, 110, 720], 0.88),
        OCRTokenData("RM",                       [320, 695, 360, 720], 0.88),
        OCRTokenData("50.00",                    [370, 695, 445, 720], 0.90),
        OCRTokenData("Change",                   [35, 735, 130, 760], 0.90),
        OCRTokenData("RM",                       [320, 735, 360, 760], 0.88),
        OCRTokenData("5.48",                     [370, 735, 445, 760], 0.91),
    ]

    def _run(self):
        return extract_layout_aware(self.TOKENS)

    def test_vendor_contains_restoran_or_hassan(self):
        """vendor_name must contain the business name, not the address."""
        result = self._run()
        vendor = result.fields.get("vendor_name", "").upper()
        self.assertTrue(
            "RESTORAN" in vendor or "HASSAN" in vendor,
            f"vendor_name must contain RESTORAN or HASSAN; got {vendor!r}",
        )

    def test_vendor_not_address_line(self):
        """The address line must NOT be chosen as vendor."""
        result = self._run()
        vendor = result.fields.get("vendor_name", "").upper()
        self.assertNotIn("JALAN", vendor,
                         f"address keyword JALAN must not appear in vendor; got {vendor!r}")
        self.assertNotIn("PUCHONG", vendor,
                         f"address keyword PUCHONG must not appear in vendor; got {vendor!r}")

    def test_line_groups_keeps_title_separate_from_address(self):
        """Keep the title and address in separate OCR lines."""
        from .services.extraction import _line_groups
        lines = _line_groups(self.TOKENS)
        title_lines = [l for l in lines if "RESTORAN" in l.text.upper() or "HASSAN" in l.text.upper()]
        address_lines = [l for l in lines if "JALAN" in l.text.upper()]
        self.assertTrue(title_lines, "RESTORAN HASSANBISTRO must appear in at least one line")
        self.assertTrue(address_lines, "JALAN address must appear in at least one line")
        for tl in title_lines:
            for al in address_lines:
                self.assertNotEqual(
                    tl.text, al.text,
                    f"Title and address must be separate lines; both in {tl.text!r}",
                )

    def test_total_is_correct(self):
        result = self._run()
        self.assertEqual(result.fields.get("total_amount", ""), "44.52")

    def test_currency_is_RM(self):
        result = self._run()
        self.assertEqual(result.fields.get("currency", ""), "RM")


class SyntheticEURInvoiceRegressionTests(SimpleTestCase):
    """Extract a multi-column EUR invoice with alphabetic month dates."""

    TOKENS = [
        OCRTokenData("ACME CONSULTING GmbH",    [50,  30,  350, 60],  0.92),
        OCRTokenData("Invoice Date:",           [50,  120, 220, 148], 0.90),
        OCRTokenData("18-Mar-2001",             [230, 120, 390, 148], 0.91),
        OCRTokenData("Due Date:",               [50,  158, 190, 183], 0.88),
        OCRTokenData("09-Sep-2010",             [200, 158, 360, 183], 0.89),
        OCRTokenData("BANK DETAILS",            [50,  220, 200, 248], 0.85),
        OCRTokenData("Invoice No:",             [450, 220, 590, 248], 0.88),
        OCRTokenData("PO-2001-042",             [600, 220, 750, 248], 0.90),
        OCRTokenData("Subtotal",               [450, 480, 560, 508], 0.90),
        OCRTokenData("EUR",                    [580, 480, 630, 508], 0.92),
        OCRTokenData("227.73",                 [640, 480, 720, 508], 0.91),
        OCRTokenData("VAT",                    [450, 518, 530, 543], 0.89),
        OCRTokenData("EUR",                    [580, 518, 630, 543], 0.91),
        OCRTokenData("9.89",                   [640, 518, 720, 543], 0.90),
        OCRTokenData("TOTAL",                  [450, 558, 560, 583], 0.93),
        OCRTokenData("EUR",                    [580, 558, 630, 583], 0.94),
        OCRTokenData("234.82",                 [640, 558, 720, 583], 0.93),
    ]

    def _run(self):
        return extract_layout_aware(self.TOKENS)

    def test_invoice_number_not_a_date(self):
        """invoice_number must never be the due-date string '09-Sep-2010'."""
        result = self._run()
        inv = result.fields.get("invoice_number", "")
        self.assertNotEqual(inv, "09-Sep-2010",
                            f"invoice_number must not be the due-date; got {inv!r}")
        self.assertFalse(looks_like_date(inv),
                         f"invoice_number {inv!r} looks like a date")

    def test_date_is_invoice_date(self):
        """date must be 2001-03-18 from the 'Invoice Date: 18-Mar-2001' line."""
        result = self._run()
        date_val = result.fields.get("date", "")
        self.assertEqual(date_val, "2001-03-18",
                         f"date must be '2001-03-18'; got {date_val!r}")

    def test_currency_is_EUR(self):
        result = self._run()
        self.assertEqual(result.fields.get("currency", ""), "EUR")

    def test_total_amount_is_234_82_not_manufactured(self):
        """total_amount must be the labeled 234.82, not a constraint-manufactured value."""
        result = self._run()
        total = result.fields.get("total_amount", "")
        self.assertEqual(total, "234.82",
                         f"total_amount must be '234.82'; got {total!r}")
        self.assertNotEqual(total, "546.13",
                            "Constraint engine must not manufacture wrong total 546.13")

    def test_subtotal_is_not_wrong_manufactured_value(self):
        result = self._run()
        sub = result.fields.get("subtotal", "")
        self.assertNotEqual(sub, "543.01",
                            "subtotal must not be manufactured constraint value 543.01")

    def test_constraint_derived_confidence_capped_by_weakest_input(self):
        """Cap derived confidence by the weakest supporting input."""
        result = self._run()
        for field_name in ("total_amount", "subtotal", "tax_amount"):
            ev = result.evidence.get(field_name, {})
            method = ev.get("method", "")
            if "constraint" in method:
                conf = ev.get("confidence", 1.0)
                self.assertLessEqual(conf, 0.95,
                                     f"{field_name} constraint confidence {conf} not capped")


class ChineseTaxiReceiptRegressionTests(SimpleTestCase):
    """Use format cues when Chinese labels are unreadable."""

    TOKENS = [
        OCRTokenData("VIHIHLAT",          [50,  10,  200, 40],  0.12),
        OCRTokenData("2017-09-27",        [50,  55,  200, 75],  0.99),
        OCRTokenData("AHJFK",            [50,  200, 170, 220], 0.30),
        OCRTokenData("BHJK",             [50,  240, 170, 260], 0.28),
        OCRTokenData("RMB",              [280, 200, 330, 220], 0.85),
        OCRTokenData("10.50",            [335, 200, 395, 220], 0.91),
        OCRTokenData("$0.03",            [50,  350, 120, 370], 0.08),
    ]

    def _run(self):
        return extract_layout_aware(self.TOKENS)

    def test_currency_not_USD(self):
        """$ artifact from 0.08-confidence line must not produce currency=USD."""
        result = self._run()
        self.assertNotEqual(result.fields.get("currency", ""), "USD",
                            "Low-confidence $ must not produce currency=USD")

    def test_vendor_not_seal_garbage(self):
        """VIHIHLAT (0.12 conf) must be rejected by the vendor confidence floor."""
        result = self._run()
        vendor = result.fields.get("vendor_name", "").upper()
        self.assertNotIn("VIHIHLAT", vendor,
                         f"Seal garbage VIHIHLAT must not be vendor; got {vendor!r}")

    def test_format_prior_date(self):
        """format_prior path must extract date=2017-09-27 from the high-conf ISO token."""
        result = self._run()
        self.assertEqual(result.fields.get("date", ""), "2017-09-27",
                         f"date must be '2017-09-27' via format_prior; "
                         f"got {result.fields.get('date', '')!r}")

    def test_format_prior_total_amount(self):
        """format_prior path must extract total_amount=10.50 (RMB adjacent)."""
        result = self._run()
        self.assertEqual(result.fields.get("total_amount", ""), "10.50",
                         f"total_amount must be '10.50' via format_prior; "
                         f"got {result.fields.get('total_amount', '')!r}")

    def test_format_prior_methods_used(self):
        """Evidence methods for date and total must include 'format_prior' or better."""
        result = self._run()
        date_method = result.evidence.get("date", {}).get("method", "")
        total_method = result.evidence.get("total_amount", {}).get("method", "")
        self.assertIn("prior", f"{date_method} {total_method}",
                      f"format_prior must have fired; date method={date_method!r}, "
                      f"total method={total_method!r}")


class ChineseTaxiRealOCRRegressionTests(SimpleTestCase):
    """Regression using real EasyOCR-English tokens from taxi_0207.jpg.

    The Chinese labels are mostly destroyed by English OCR, but the layout still
    exposes a Chinese fapiao pattern: 12-digit invoice code, noisy invoice-number
    row, taxi plate, date, and total fare.
    """

    TOKENS = [
        OCRTokenData("##+3: 137131730001", [188, 797, 1298, 958], 0.238),
        OCRTokenData('"2414411142434', [187, 943, 1271, 1105], 0.366),
        OCRTokenData("8110231", [811, 1746, 1266, 1870], 1.0),
        OCRTokenData("QT-2817", [814, 1864, 1281, 1984], 0.713),
        OCRTokenData("000000", [876, 1981, 1286, 2098], 0.409),
        OCRTokenData("A #:", [215, 2133, 424, 2232], 0.184),
        OCRTokenData("2017-09-27", [628, 2096, 1292, 2219], 0.901),
        OCRTokenData("01:44", [944, 2219, 1291, 2328], 0.998),
        OCRTokenData("01:50", [946, 2338, 1292, 2443], 0.559),
        OCRTokenData("1 .92", [1026, 2456, 1298, 2569], 0.632),
        OCRTokenData("3.8k", [955, 2583, 1305, 2688], 0.978),
        OCRTokenData("00:01.05", [760, 2697, 1310, 2817], 0.995),
        OCRTokenData("43:", [203, 2866, 429, 2975], 0.336),
        OCRTokenData("10.50 #", [842, 2956, 1316, 3078], 0.605),
        OCRTokenData("$IM2$[2017)122+4K657 K1005 ,(44+127)", [168, 3544, 1456, 3625], 0.029),
    ]

    def _run(self):
        return extract_layout_aware(self.TOKENS)

    def test_real_chinese_taxi_invoice_number_from_noisy_fapiao_row(self):
        result = self._run()
        self.assertEqual(result.fields.get("invoice_number", ""), "11142434")

    def test_real_chinese_taxi_currency_is_cny_not_usd(self):
        result = self._run()
        self.assertEqual(result.fields.get("currency", ""), "CNY")

    def test_real_chinese_taxi_total_has_stronger_prior(self):
        result = self._run()
        self.assertEqual(result.fields.get("total_amount", ""), "10.50")
        self.assertGreaterEqual(result.confidences.get("total_amount", 0), 0.50)


class ChineseRetailReceiptRegressionTests(SimpleTestCase):
    """Regression for Chinese POS retail receipts with payment-summary labels."""

    TOKENS = [
        OCRTokenData("吖嘀吖嘀", [99, 24, 297, 92], 0.94),
        OCRTokenData("新一代国民零食", [104, 79, 286, 111], 0.99),
        OCRTokenData(
            "单据号：XLB49239871693749260QHW 收款时间:2026-06-15 18:23:24 南昌新建区庐山中大道店 商品信息 收银员：8342302",
            [31, 147, 357, 245],
            0.99,
        ),
        OCRTokenData("品名 原价/折后 重量500g/数量 总额/金额", [18, 250, 360, 280], 0.98),
        OCRTokenData("元气森林气泡水维C橙味439ml", [18, 288, 290, 314], 0.99),
        OCRTokenData("3.90/3.90 1.000 3.90/3.90", [160, 315, 355, 338], 0.99),
        OCRTokenData("=支付信息=", [114, 480, 260, 500], 0.99),
        OCRTokenData("件数：5.00 订单原价：20.50", [174, 497, 326, 517], 0.99),
        OCRTokenData("应收：20.50 订单优惠：0.00", [14, 512, 315, 535], 0.99),
        OCRTokenData("实收：20.50 找零：0.00", [14, 535, 315, 558], 0.99),
        OCRTokenData(
            "会员手机号：197****9514 本次积分：20 支付宝：20.50 会员积分：593",
            [14, 570, 360, 600],
            0.99,
        ),
        OCRTokenData("门店地址：南昌市南昌经济技术开发区", [11, 615, 360, 642], 0.99),
    ]

    def test_payment_summary_beats_item_count_and_original_price_sum(self):
        result = extract_layout_aware(self.TOKENS)

        self.assertEqual(result.fields.get("invoice_number", ""), "XLB49239871693749260QHW")
        self.assertEqual(result.fields.get("date", ""), "2026-06-15")
        self.assertEqual(result.fields.get("subtotal", ""), "20.50")
        self.assertEqual(result.fields.get("tax_amount", ""), "0.00")
        self.assertEqual(result.fields.get("total_amount", ""), "20.50")
        self.assertEqual(result.fields.get("currency", ""), "CNY")
        self.assertNotEqual(result.fields.get("total_amount", ""), "25.50")
        self.assertIn("应收", result.evidence.get("total_amount", {}).get("source_text", ""))
        self.assertEqual(result.evidence.get("_document_category", {}).get("category"), "chinese_invoice")

    def test_vendor_prefers_top_brand_not_address_or_product_row(self):
        result = extract_layout_aware(self.TOKENS)
        vendor = result.fields.get("vendor_name", "")

        self.assertIn("新一代国民零食", vendor)
        self.assertNotIn("门店地址", vendor)
        self.assertNotIn("元气森林", vendor)


class DocumentCategoryRoutingRegressionTests(SimpleTestCase):
    """Step 5: category-aware routing should be visible in extraction evidence."""

    def _category(self, tokens):
        result = extract_layout_aware(tokens)
        return result.evidence.get("_document_category", {}).get("category"), result

    def test_malaysian_receipt_route_prefers_rm_and_receipt_total(self):
        category, result = self._category([
            OCRTokenData("POPULAR BOOK CO. (M) SDN BHD", [10, 10, 300, 40], 0.90),
            OCRTokenData("Total RM 49.40", [10, 200, 300, 230], 0.90),
            OCRTokenData("Cash 50.00", [10, 240, 300, 270], 0.90),
            OCRTokenData("CHANGE 0.60", [10, 280, 300, 310], 0.90),
        ])

        self.assertEqual(category, "malaysian_receipt")
        self.assertEqual(result.fields.get("currency", ""), "RM")
        self.assertEqual(result.fields.get("total_amount", ""), "49.40")
        self.assertIn("category routing", result.evidence.get("total_amount", {}).get("explanation", ""))

    def test_clean_invoice_route_prefers_explicit_labels(self):
        category, result = self._category([
            OCRTokenData("ACME SERVICES LTD", [10, 10, 250, 40], 0.95),
            OCRTokenData("Invoice No: INV-2026-001", [300, 80, 560, 110], 0.95),
            OCRTokenData("Invoice Date: 2026-06-10", [300, 120, 560, 150], 0.95),
            OCRTokenData("Bill To: Client Company", [10, 160, 260, 190], 0.95),
            OCRTokenData("Subtotal 100.00", [300, 400, 560, 430], 0.95),
            OCRTokenData("Tax Amount 5.00", [300, 440, 560, 470], 0.95),
            OCRTokenData("Amount Due USD 105.00", [300, 480, 620, 510], 0.95),
        ])

        self.assertEqual(category, "clean_invoice")
        self.assertEqual(result.fields.get("invoice_number", ""), "INV-2026-001")
        self.assertEqual(result.fields.get("total_amount", ""), "105.00")
        self.assertIn("explicit labels", result.evidence.get("total_amount", {}).get("explanation", ""))

    def test_fatura_route_detects_fatura_kdv_labels(self):
        category, result = self._category([
            OCRTokenData("FATURA", [10, 10, 180, 50], 0.95),
            OCRTokenData("Fatura No: FTR-2026-01", [300, 80, 560, 110], 0.95),
            OCRTokenData("KDV 18% 18.00", [300, 320, 560, 350], 0.95),
            OCRTokenData("TOPLAM TUTAR 118.00", [300, 380, 620, 410], 0.95),
        ])

        self.assertEqual(category, "fatura")
        self.assertEqual(result.fields.get("invoice_number", ""), "FTR-2026-01")
        self.assertEqual(result.fields.get("total_amount", ""), "118.00")
        self.assertIn("FATURA", result.evidence.get("_document_category", {}).get("reasons", [""])[0])


class ChineseFapiaoLabelRegressionTests(SimpleTestCase):
    """Extract Chinese invoice fields from labels and layout."""

    TOKENS = [
        OCRTokenData("发票代码:137131730001", [188, 797, 1298, 958], 0.994),
        OCRTokenData("发票号码:11142434", [187, 943, 1271, 1105], 0.851),
        OCRTokenData("临沂市出租车", [455, 1122, 1049, 1256], 0.832),
        OCRTokenData("发票专用章", [565, 1232, 945, 1335], 0.983),
        OCRTokenData("日期:", [215, 2133, 424, 2232], 0.633),
        OCRTokenData("2617-53-27", [628, 2096, 1292, 2219], 0.286),
        OCRTokenData("上车:", [209, 2255, 424, 2354], 0.954),
        OCRTokenData("囚1 :44", [944, 2219, 1291, 2328], 0.123),
        OCRTokenData("里程:", [207, 2616, 428, 2720], 0.721),
        OCRTokenData("3.8km", [955, 2583, 1305, 2688], 0.978),
        OCRTokenData("金额:", [206, 2996, 427, 3097], 0.947),
        OCRTokenData("10.50 元", [842, 2956, 1316, 3078], 0.802),
    ]

    def _run(self):
        return extract_layout_aware(self.TOKENS)

    def test_chinese_fapiao_invoice_number_from_label(self):
        result = self._run()
        self.assertEqual(result.fields.get("invoice_number", ""), "11142434")
        self.assertIn("label", result.evidence.get("invoice_number", {}).get("method", ""))

    def test_chinese_fapiao_date_repaired_from_noisy_labeled_date(self):
        result = self._run()
        self.assertEqual(result.fields.get("date", ""), "2017-09-27")

    def test_chinese_fapiao_vendor_currency_and_total(self):
        result = self._run()
        self.assertEqual(result.fields.get("vendor_name", ""), "临沂市出租车")
        self.assertEqual(result.fields.get("currency", ""), "CNY")
        self.assertEqual(result.fields.get("total_amount", ""), "10.50")


class ChineseVATIDNumericRegressionTests(SimpleTestCase):
    """Handle noisy numeric fields in Chinese VAT invoices."""

    def test_noisy_vat_header_recovers_number_date_amounts_and_cny(self):
        tokens = [
            OCRTokenData("1100162130 17}+TC N % + N99750735 ,1001o2130", [80, 60, 760, 95], 0.39),
            OCRTokenData("{reg:61 # # A # 20175F04 H18H 09750785", [80, 115, 790, 150], 0.45),
            OCRTokenData("\u5408\u8ba1 \u91d1\u989d \u7a0e\u989d", [70, 610, 430, 640], 0.52),
            OCRTokenData("32705. 98 5560, 02", [420, 645, 760, 675], 0.48),
            OCRTokenData("\u4ef7\u7a0e\u5408\u8ba1 \u5927\u5199 \u53c1\u4e07\u634c\u4edf\u8d30\u4f70\u9646\u62fe\u9646\u5706\u6574", [70, 720, 820, 750], 0.50),
        ]

        result = extract_layout_aware(tokens)

        self.assertEqual(result.fields.get("invoice_number", ""), "9750785")
        self.assertEqual(result.fields.get("date", ""), "2017-04-18")
        self.assertEqual(result.fields.get("subtotal", ""), "32705.98")
        self.assertEqual(result.fields.get("tax_amount", ""), "5560.02")
        self.assertEqual(result.fields.get("total_amount", ""), "38266.00")
        self.assertEqual(result.fields.get("currency", ""), "CNY")
        self.assertIn("chinese_vat", result.evidence.get("total_amount", {}).get("method", ""))

    def test_noisy_vat_triplet_prefers_consistent_total_over_tax_or_noise(self):
        tokens = [
            OCRTokenData("3100164320 1/}p 452 j% # Ng 34861239 3100164320", [80, 60, 780, 95], 0.55),
            OCRTokenData("2017f058oz8 34861239", [420, 118, 760, 148], 0.32),
            OCRTokenData("\u91d1\u989d \u7a0e\u989d \u4ef7\u7a0e\u5408\u8ba1", [70, 600, 520, 630], 0.50),
            OCRTokenData("377.36 D 22 64 400 . 00", [410, 642, 760, 674], 0.38),
            OCRTokenData("\u5927\u5199 \u8086\u4f70\u5706\u6574", [70, 720, 360, 750], 0.52),
        ]

        result = extract_layout_aware(tokens)

        self.assertEqual(result.fields.get("invoice_number", ""), "34861239")
        self.assertEqual(result.fields.get("date", ""), "2017-05-02")
        self.assertEqual(result.fields.get("subtotal", ""), "377.36")
        self.assertEqual(result.fields.get("tax_amount", ""), "22.64")
        self.assertEqual(result.fields.get("total_amount", ""), "400.00")
        self.assertEqual(result.fields.get("currency", ""), "CNY")

    def test_chinese_taxi_meter_row_beats_header_range_and_foreign_symbol_noise(self):
        tokens = [
            OCRTokenData("X#404: 123001771811", [120, 70, 620, 105], 0.42),
            OCRTokenData("8884.56256956", [120, 125, 560, 160], 0.38),
            OCRTokenData("15.56-16.07 0000017206 65-51286 2017.9.9", [70, 530, 760, 565], 0.47),
            OCRTokenData("#e Ge Jk sh Mt Jjp 1 % 10.00 0.00 3.8 j", [70, 760, 790, 795], 0.49),
            OCRTokenData("GlJG? 75 \u20ac Ai", [70, 900, 760, 940], 0.06),
        ]

        result = extract_layout_aware(tokens)

        self.assertEqual(result.fields.get("invoice_number", ""), "56256956")
        self.assertEqual(result.fields.get("date", ""), "2017-09-09")
        self.assertEqual(result.fields.get("total_amount", ""), "10.00")
        self.assertEqual(result.fields.get("currency", ""), "CNY")
        self.assertEqual(result.fields.get("subtotal", ""), "")
        self.assertEqual(result.fields.get("tax_amount", ""), "")
        self.assertNotEqual(result.fields.get("currency", ""), "EUR")
        self.assertEqual(
            result.evidence.get("total_amount", {}).get("method", ""),
            "chinese_taxi_meter_fare_candidate",
        )


class ChineseOCRRetryDecisionTests(SimpleTestCase):
    """Test when an English OCR pass should retry with Chinese support."""

    CHINESE_TAXI_LIKE_TOKENS = [
        OCRTokenData("##+3: 137131730001", [188, 797, 1298, 958], 0.238),
        OCRTokenData('"2414411142434', [187, 943, 1271, 1105], 0.366),
        OCRTokenData("QT-2817", [814, 1864, 1281, 1984], 0.713),
        OCRTokenData("2017-09-27", [628, 2096, 1292, 2219], 0.901),
        OCRTokenData("10.50 #", [842, 2956, 1316, 3078], 0.605),
    ]

    def test_chinese_taxi_profile_triggers_chinese_retry(self):
        from .services.ocr import _should_retry_chinese_easyocr

        self.assertTrue(_should_retry_chinese_easyocr(self.CHINESE_TAXI_LIKE_TOKENS, ("en",)))

    def test_existing_chinese_language_does_not_retry_again(self):
        from .services.ocr import _should_retry_chinese_easyocr

        self.assertFalse(_should_retry_chinese_easyocr(self.CHINESE_TAXI_LIKE_TOKENS, ("ch_sim", "en")))

    def test_clear_english_invoice_does_not_trigger_chinese_retry(self):
        from .services.ocr import _should_retry_chinese_easyocr

        tokens = [
            OCRTokenData("TAX INVOICE", [50, 30, 300, 70], 0.95),
            OCRTokenData("Invoice No: INV-2026-0001", [600, 150, 920, 180], 0.88),
            OCRTokenData("Date: 2026-05-01", [600, 190, 900, 220], 0.88),
            OCRTokenData("Total Amount: USD 220.00", [620, 865, 940, 895], 0.93),
        ]
        self.assertFalse(_should_retry_chinese_easyocr(tokens, ("en",)))

    def test_chinese_tokens_are_preferred_when_retry_reads_cjk_text(self):
        from .services.ocr import _prefer_chinese_easyocr_tokens

        primary = [
            OCRTokenData("ViHpLat % %", [372, 324, 1036, 513], 0.025),
            OCRTokenData("10.50 #", [842, 2956, 1316, 3078], 0.605),
        ]
        chinese = [
            OCRTokenData("山东省国家税务局", [340, 300, 1050, 520], 0.72),
            OCRTokenData("临沂市出租车", [455, 1122, 1049, 1256], 0.80),
            OCRTokenData("金额 10.50 元", [842, 2956, 1316, 3078], 0.82),
        ]
        self.assertTrue(_prefer_chinese_easyocr_tokens(primary, chinese))


class PaddleOCREngineSelectionTests(SimpleTestCase):
    """The web OCR pipeline should prefer PaddleOCR without requiring tests to run OCR."""

    def test_auto_engine_sequence_prefers_paddleocr(self):
        from .services.ocr import _engine_sequence

        self.assertEqual(_engine_sequence("auto"), ("paddleocr", "easyocr"))

    def test_paddleocr_default_language_is_chinese_plus_english(self):
        from .services.ocr import _default_languages

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_default_languages("paddleocr"), ("ch_sim", "en"))
            self.assertEqual(_default_languages("easyocr"), ("en",))

    def test_paddleocr_optional_preprocessors_are_off_by_default(self):
        from .services.ocr import _paddleocr_v3_kwargs

        with patch.dict(os.environ, {}, clear=True):
            kwargs = _paddleocr_v3_kwargs("ch")

        self.assertEqual(kwargs["lang"], "ch")
        self.assertFalse(kwargs["use_doc_orientation_classify"])
        self.assertFalse(kwargs["use_doc_unwarping"])
        self.assertFalse(kwargs["use_textline_orientation"])

    def test_paddleocr_optional_preprocessors_can_be_enabled_by_env(self):
        from .services.ocr import _paddleocr_v3_kwargs

        with patch.dict(
            os.environ,
            {
                "AI_INVOICE_PADDLE_DOC_ORIENTATION": "true",
                "AI_INVOICE_PADDLE_DOC_UNWARPING": "true",
                "AI_INVOICE_PADDLE_TEXTLINE_ORIENTATION": "true",
            },
            clear=True,
        ):
            kwargs = _paddleocr_v3_kwargs("en")

        self.assertTrue(kwargs["use_doc_orientation_classify"])
        self.assertTrue(kwargs["use_doc_unwarping"])
        self.assertTrue(kwargs["use_textline_orientation"])

    def test_windows_nvidia_dll_bootstrap_is_safe(self):
        from .services.ocr import _add_windows_nvidia_dll_dirs

        _add_windows_nvidia_dll_dirs()

    def test_run_ocr_auto_uses_paddleocr_first(self):
        from .services.ocr import OCRTokenData, run_ocr

        paddle_tokens = [OCRTokenData("发票号码 11142434", [0, 0, 100, 20], 0.90)]
        with patch("invoices.services.ocr._run_paddleocr", return_value=paddle_tokens) as paddle:
            with patch("invoices.services.ocr._run_easyocr") as easy:
                result = run_ocr("dummy.jpg", engine="auto")

        self.assertEqual(result, paddle_tokens)
        paddle.assert_called_once()
        easy.assert_not_called()
        self.assertEqual(paddle.call_args.args[1], ("ch_sim", "en"))

    def test_run_ocr_auto_falls_back_to_easyocr_if_paddle_missing(self):
        from .services.ocr import OCRDependencyError, OCRTokenData, run_ocr

        easy_tokens = [OCRTokenData("Invoice No INV-1", [0, 0, 100, 20], 0.90)]
        with patch("invoices.services.ocr._run_paddleocr", side_effect=OCRDependencyError("missing paddle")):
            with patch("invoices.services.ocr._run_easyocr", return_value=easy_tokens) as easy:
                result = run_ocr("dummy.jpg", engine="auto")

        self.assertEqual(result, easy_tokens)
        easy.assert_called_once()

    def test_auto_retries_easyocr_when_paddle_quality_is_low(self):
        from .services.ocr import OCRTokenData, run_ocr_with_metadata

        paddle_tokens = [
            OCRTokenData("4LJc?,", [0, 0, 80, 20], 0.05),
            OCRTokenData("V6thlm h M /", [0, 30, 140, 50], 0.04),
        ]
        easy_tokens = [
            OCRTokenData("Invoice No INV-100", [0, 0, 180, 24], 0.88),
            OCRTokenData("Total RM 49.40", [0, 40, 160, 64], 0.86),
            OCRTokenData("Date 2026-06-15", [0, 80, 170, 104], 0.86),
        ]

        with patch("invoices.services.ocr._run_paddleocr", return_value=paddle_tokens) as paddle:
            with patch("invoices.services.ocr._run_easyocr", return_value=easy_tokens) as easy:
                result, metadata = run_ocr_with_metadata("dummy.jpg", engine="auto")

        self.assertEqual(result, easy_tokens)
        paddle.assert_called_once()
        easy.assert_called_once()
        self.assertEqual(metadata["selected_engine"], "easyocr")
        self.assertEqual(metadata["fallback_reason"], "paddle_low_quality")
        self.assertEqual(len(metadata["attempts"]), 2)

    def test_auto_keeps_paddle_when_easyocr_is_not_better(self):
        from .services.ocr import OCRTokenData, run_ocr_with_metadata

        paddle_tokens = [
            OCRTokenData("Invoice No INV-100", [0, 0, 180, 24], 0.62),
            OCRTokenData("Total RM 49.40", [0, 40, 160, 64], 0.60),
        ]
        easy_tokens = [OCRTokenData("lNv0", [0, 0, 80, 20], 0.12)]

        with patch.dict(os.environ, {"AI_INVOICE_ALT_ENGINE_RETRY_CONFIDENCE": "0.70"}, clear=False):
            with patch("invoices.services.ocr._run_paddleocr", return_value=paddle_tokens):
                with patch("invoices.services.ocr._run_easyocr", return_value=easy_tokens):
                    result, metadata = run_ocr_with_metadata("dummy.jpg", engine="auto")

        self.assertEqual(result, paddle_tokens)
        self.assertEqual(metadata["selected_engine"], "paddleocr")
        self.assertEqual(len(metadata["attempts"]), 2)

    def test_auto_keeps_paddle_when_easyocr_fallback_errors(self):
        from .services.ocr import OCRDependencyError, OCRTokenData, run_ocr_with_metadata

        paddle_tokens = [OCRTokenData("4LJc?,", [0, 0, 80, 20], 0.05)]

        with patch("invoices.services.ocr._run_paddleocr", return_value=paddle_tokens):
            with patch("invoices.services.ocr._run_easyocr", side_effect=OCRDependencyError("torch dll error")):
                result, metadata = run_ocr_with_metadata("dummy.jpg", engine="auto")

        self.assertEqual(result, paddle_tokens)
        self.assertEqual(metadata["selected_engine"], "paddleocr")
        self.assertIn("fallback_failed", metadata["fallback_reason"])
        self.assertEqual(metadata["attempts"][-1]["error"], "torch dll error")

    def test_ocr_cache_reuses_same_file_and_settings(self):
        from pathlib import Path

        from .services.ocr import OCRTokenData, run_ocr_with_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "invoice.jpg"
            image_path.write_bytes(b"same invoice bytes")
            cache_dir = Path(tmpdir) / "ocr_cache"
            paddle_tokens = [
                OCRTokenData("Invoice No INV-CACHE-1", [0, 0, 180, 24], 0.91),
                OCRTokenData("Total RM 49.40", [0, 40, 160, 64], 0.89),
            ]

            with patch.dict(
                os.environ,
                {
                    "AI_INVOICE_OCR_CACHE": "true",
                    "AI_INVOICE_OCR_CACHE_DIR": str(cache_dir),
                },
                clear=False,
            ):
                with patch("invoices.services.ocr._run_paddleocr", return_value=paddle_tokens) as paddle:
                    first_tokens, first_metadata = run_ocr_with_metadata(image_path, engine="paddleocr")
                    second_tokens, second_metadata = run_ocr_with_metadata(image_path, engine="paddleocr")

        self.assertEqual(paddle.call_count, 1)
        self.assertEqual([token.text for token in first_tokens], [token.text for token in second_tokens])
        self.assertFalse(first_metadata["cache"]["hit"])
        self.assertTrue(second_metadata["cache"]["hit"])
        self.assertTrue(second_metadata["loaded_from_cache"])

    def test_ocr_cache_key_changes_when_ocr_settings_change(self):
        from pathlib import Path

        from .services.ocr import OCRTokenData, run_ocr_with_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "invoice.jpg"
            image_path.write_bytes(b"same invoice bytes")
            cache_dir = Path(tmpdir) / "ocr_cache"
            paddle_tokens = [OCRTokenData("Invoice No INV-CACHE-2", [0, 0, 180, 24], 0.91)]

            with patch("invoices.services.ocr._run_paddleocr", return_value=paddle_tokens) as paddle:
                with patch.dict(
                    os.environ,
                    {
                        "AI_INVOICE_OCR_CACHE": "true",
                        "AI_INVOICE_OCR_CACHE_DIR": str(cache_dir),
                        "AI_INVOICE_OCR_VARIANT_LIMIT": "1",
                    },
                    clear=False,
                ):
                    _, first_metadata = run_ocr_with_metadata(image_path, engine="paddleocr")
                with patch.dict(
                    os.environ,
                    {
                        "AI_INVOICE_OCR_CACHE": "true",
                        "AI_INVOICE_OCR_CACHE_DIR": str(cache_dir),
                        "AI_INVOICE_OCR_VARIANT_LIMIT": "2",
                    },
                    clear=False,
                ):
                    _, second_metadata = run_ocr_with_metadata(image_path, engine="paddleocr")

        self.assertEqual(paddle.call_count, 2)
        self.assertFalse(first_metadata["cache"]["hit"])
        self.assertFalse(second_metadata["cache"]["hit"])
        self.assertNotEqual(first_metadata["cache"]["cache_key"], second_metadata["cache"]["cache_key"])

    def test_extraction_evidence_records_ocr_engine_and_variant(self):
        tokens = [
            OCRTokenData("ACME LTD", [0, 0, 120, 20], 0.90, source_variant="raw", source_engine="paddleocr"),
            OCRTokenData("Total USD 21.00", [0, 50, 180, 74], 0.92, source_variant="contrast", source_engine="paddleocr"),
        ]

        result = extract_layout_aware(tokens)
        ocr_evidence = result.evidence.get("_ocr", {})

        self.assertEqual(ocr_evidence.get("token_count"), 2)
        self.assertEqual(ocr_evidence.get("source_engines"), {"paddleocr": 2})
        self.assertEqual(ocr_evidence.get("source_variants"), {"contrast": 1, "raw": 1})

    def test_paddleocr_tries_preprocessing_variants_for_chinese_invoice(self):
        from pathlib import Path

        from .services.ocr import OCRTokenData, _best_paddle_tokens_for_variants

        raw = [OCRTokenData("发票号码:11142434", [0, 0, 100, 20], 0.80)]
        improved = [
            OCRTokenData("发票号码:11142434", [0, 0, 100, 20], 0.90),
            OCRTokenData("金额:10.50元", [0, 40, 100, 60], 0.92),
            OCRTokenData("日期:2017-09-27", [0, 80, 100, 100], 0.91),
        ]

        with patch("invoices.services.ocr._read_paddle_image", side_effect=[raw, improved]):
            result = _best_paddle_tokens_for_variants(
                object(),
                [("raw", Path("raw.png")), ("sharpened", Path("sharpened.png"))],
                1,
                force_fusion=True,
            )

        texts = {token.text for token in result}
        self.assertIn("发票号码:11142434", texts)
        self.assertIn("金额:10.50元", texts)
        self.assertIn("日期:2017-09-27", texts)

    def test_paddleocr_fusion_preserves_conflicting_total_candidates(self):
        from pathlib import Path

        from .services.ocr import OCRTokenData, _best_paddle_tokens_for_variants

        raw = [
            OCRTokenData("金额:", [0, 40, 40, 60], 0.95),
            OCRTokenData("15.56 元", [90, 40, 170, 60], 0.42),
        ]
        improved = [
            OCRTokenData("金额:", [0, 40, 40, 60], 0.95),
            OCRTokenData("15.50 元", [90, 40, 170, 60], 0.84),
        ]

        with patch("invoices.services.ocr._read_paddle_image", side_effect=[raw, improved]):
            result = _best_paddle_tokens_for_variants(
                object(),
                [("raw", Path("raw.png")), ("contrast", Path("contrast.png"))],
                1,
                force_fusion=True,
            )

        texts = [token.text for token in result]
        self.assertEqual(texts.count("金额:"), 1, "duplicate labels should be merged")
        self.assertIn("15.56 元", texts, "conflicting amount evidence should be preserved")
        self.assertIn("15.50 元", texts, "better amount evidence should be preserved")

        extraction = extract_layout_aware(result)
        self.assertEqual(extraction.fields.get("total_amount", ""), "15.50")

    def test_paddle_variant_quality_prefers_invoice_signals_over_confident_noise(self):
        from .services.ocr import OCRTokenData, _paddle_variant_quality

        confident_noise = [
            OCRTokenData("ALJE WSNTH HNE", [0, 0, 200, 20], 0.96),
            OCRTokenData("MtkE V6thlm", [0, 30, 200, 50], 0.92),
        ]
        invoice_signals = [
            OCRTokenData("\u53d1\u7968\u53f7\u7801:11142434", [0, 0, 200, 20], 0.72),
            OCRTokenData("\u91d1\u989d:10.50\u5143", [0, 30, 200, 50], 0.70),
            OCRTokenData("\u65e5\u671f:2017-09-27", [0, 60, 200, 80], 0.70),
        ]

        self.assertGreater(
            _paddle_variant_quality(invoice_signals),
            _paddle_variant_quality(confident_noise),
        )

    def test_paddleocr_debug_report_records_variants_and_selected_tokens(self):
        from pathlib import Path

        from .services.ocr import OCRTokenData, _best_paddle_tokens_for_variants

        raw = [OCRTokenData("15.56 \u5143", [90, 40, 170, 60], 0.42)]
        improved = [OCRTokenData("15.50 \u5143", [90, 40, 170, 60], 0.84)]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {
                    "AI_INVOICE_OCR_DEBUG": "true",
                    "AI_INVOICE_OCR_DEBUG_DIR": tmpdir,
                },
                clear=False,
            ):
                with patch("invoices.services.ocr._read_paddle_image", side_effect=[raw, improved]):
                    result = _best_paddle_tokens_for_variants(
                        object(),
                        [("raw", Path("raw.png")), ("contrast", Path("contrast.png"))],
                        1,
                        force_fusion=True,
                    )

            debug_files = list(Path(tmpdir).glob("*.json"))
            self.assertEqual(len(debug_files), 1)
            with debug_files[0].open("r", encoding="utf-8") as f:
                payload = json.load(f)

        self.assertEqual(payload["engine"], "paddleocr")
        self.assertEqual(payload["selection_mode"], "fused_variants")
        self.assertEqual(payload["variant_count"], 2)
        self.assertEqual([item["name"] for item in payload["variants"]], ["raw", "contrast"])
        self.assertTrue(any(token.text == "15.50 \u5143" for token in result))
        self.assertTrue(any(token["text"] == "15.50 \u5143" for token in payload["selected_tokens"]))


class AlphaMonthDateRegressionTests(SimpleTestCase):
    """Parse dates that use alphabetic month names."""

    def test_looks_like_date_dd_mon_yyyy(self):
        self.assertTrue(looks_like_date("09-Sep-2010"))

    def test_looks_like_date_dd_mon_yy(self):
        self.assertTrue(looks_like_date("18-Mar-01"))

    def test_looks_like_date_dd_space_mon_yyyy(self):
        self.assertTrue(looks_like_date("18 Mar 2001"))

    def test_looks_like_date_mon_dd_yyyy(self):
        self.assertTrue(looks_like_date("Sep 09 2010"))

    def test_extract_dates_dd_mon_yyyy(self):
        from .services.extraction import _extract_dates
        results = _extract_dates("Invoice Date: 18-Mar-2001")
        self.assertTrue(any(v == "2001-03-18" for v, _ in results),
                        f"_extract_dates must parse '18-Mar-2001'; got {results}")

    def test_extract_dates_dd_mon_yy(self):
        from .services.extraction import _extract_dates
        results = _extract_dates("09-Sep-10")
        self.assertTrue(any(v == "2010-09-09" for v, _ in results),
                        f"_extract_dates must parse '09-Sep-10'; got {results}")

    def test_extract_dates_mon_dd_yyyy(self):
        from .services.extraction import _extract_dates
        results = _extract_dates("Sep 09, 2010")
        self.assertTrue(any(v == "2010-09-09" for v, _ in results),
                        f"_extract_dates must parse 'Sep 09, 2010'; got {results}")


class Phase2HardeningRegressionTests(SimpleTestCase):
    """Additional extraction edge cases."""

    def test_low_confidence_bare_dollar_abstains_currency(self):
        tokens = [
            OCRTokenData("2017-09-27", [50, 50, 180, 75], 0.98),
            OCRTokenData("43 : 10.50 #", [260, 250, 390, 275], 0.90),
            OCRTokenData("$IM2$[2017)122+4K657 K1005", [40, 340, 430, 365], 0.08),
        ]
        result = extract_layout_aware(tokens)
        self.assertEqual(
            result.fields.get("currency", ""),
            "",
            "Low-confidence bare $ noise must abstain instead of producing USD",
        )

    def test_tax_amount_ignores_vat_rate_percent(self):
        tokens = [
            OCRTokenData("ACME CONSULTING GmbH", [50, 30, 350, 60], 0.92),
            OCRTokenData("Invoice No:", [50, 110, 185, 138], 0.90),
            OCRTokenData("INV-EUR-42", [200, 110, 345, 138], 0.91),
            OCRTokenData("date:", [50, 150, 120, 178], 0.95),
            OCRTokenData("09-Sep-2010", [135, 150, 300, 178], 0.96),
            OCRTokenData("subtotal: 227.73 EUR", [430, 470, 730, 500], 0.96),
            OCRTokenData("TAX:VAT (4.34%): 9.89 EUR", [430, 515, 780, 545], 0.90),
            OCRTokenData("total: 234.82 EUR", [430, 560, 710, 590], 0.94),
        ]
        result = extract_layout_aware(tokens)
        self.assertEqual(result.fields.get("tax_amount", ""), "9.89")
        self.assertEqual(result.fields.get("vat_rate", ""), "4.34%")
        self.assertEqual(result.fields.get("total_amount", ""), "234.82")
        self.assertEqual(result.fields.get("currency", ""), "EUR")

    def test_format_prior_ignores_email_local_part(self):
        tokens = [
            OCRTokenData("Email: melvin40@example.net Tel:+(221)488-0938", [50, 40, 560, 68], 0.88),
            OCRTokenData("Alje Wsnt61 hne", [50, 78, 230, 104], 0.18),
            OCRTokenData("Invoice Date:", [50, 120, 210, 148], 0.92),
            OCRTokenData("18-Mar-2001", [225, 120, 390, 148], 0.91),
            OCRTokenData("total: 234.82 EUR", [430, 560, 710, 590], 0.94),
        ]
        result = extract_layout_aware(tokens)
        self.assertNotEqual(
            result.fields.get("invoice_number", ""),
            "melvin40",
            "Email local-part must not be used as invoice_number",
        )
        self.assertNotEqual(
            result.fields.get("invoice_number", ""),
            "Wsnt61",
            "Low-confidence unlabeled OCR noise must not be used as invoice_number",
        )

    def test_vendor_rejects_po_email_and_bank_lines(self):
        tokens = [
            OCRTokenData("PO Number :49", [50, 30, 230, 58], 0.88),
            OCRTokenData("Email: melvin40@example.net", [50, 70, 350, 98], 0.86),
            OCRTokenData("Bank Name Branch Name", [50, 110, 360, 138], 0.85),
            OCRTokenData("Address:16424 Timothy Mission Buyer David Aguirre", [50, 135, 520, 160], 0.84),
            OCRTokenData("Moralesburgh FM 84325 US", [50, 162, 360, 184], 0.89),
            OCRTokenData("ACME CONSULTING GmbH", [50, 185, 350, 213], 0.92),
            OCRTokenData("Invoice No:", [430, 185, 560, 213], 0.90),
            OCRTokenData("INV-42", [575, 185, 660, 213], 0.91),
            OCRTokenData("total: 234.82 EUR", [430, 560, 710, 590], 0.94),
        ]
        result = extract_layout_aware(tokens)
        vendor = result.fields.get("vendor_name", "")
        self.assertIn("ACME", vendor)
        self.assertNotIn("PO NUMBER", vendor.upper())
        self.assertNotIn("EMAIL", vendor.upper())
        self.assertNotIn("BANK", vendor.upper())
        self.assertNotIn("ADDRESS", vendor.upper())


class Step7RegressionCoverageTests(SimpleTestCase):
    """Guards for known evaluation failures."""

    def test_evaluation_normalizes_equivalent_amount_and_date_formats(self):
        self.assertTrue(values_equal("subtotal", "21.0", "21.00"))
        self.assertTrue(values_equal("tax_amount", "34.0", "34.00"))
        self.assertTrue(values_equal("total_amount", "231.0", "231.00"))
        self.assertTrue(values_equal("date", "22/12/2017", "2017-12-22"))
        self.assertTrue(values_equal("date", "2017/09/09", "2017-09-09"))

    def test_vendor_address_line_is_removed_from_candidate(self):
        tokens = [
            OCRTokenData("ACME HARDWARE SDN BHD Business Address: 88 Jalan Tun Razak Kuala Lumpur", [40, 35, 820, 65], 0.91),
            OCRTokenData("Invoice No: INV-ACME-001", [480, 130, 780, 160], 0.93),
            OCRTokenData("Date: 2026-06-01", [480, 175, 740, 205], 0.92),
            OCRTokenData("Total RM 21.00", [520, 790, 760, 820], 0.94),
        ]

        result = extract_layout_aware(tokens)
        vendor = result.fields.get("vendor_name", "")

        self.assertEqual(vendor, "ACME HARDWARE SDN BHD")
        self.assertNotIn("JALAN", vendor.upper())
        self.assertNotIn("ADDRESS", vendor.upper())

    def test_chinese_invoice_number_prefers_fapiao_number_not_tax_id(self):
        tokens = [
            OCRTokenData("\u53d1\u7968\u4ee3\u7801:137131730001", [40, 60, 430, 95], 0.94),
            OCRTokenData("\u53d1\u7968\u53f7\u7801:11142434", [40, 105, 380, 140], 0.93),
            OCRTokenData("\u7eb3\u7a0e\u4eba\u8bc6\u522b\u53f7:91370100MA3ABC1234", [40, 150, 620, 185], 0.90),
            OCRTokenData("\u65e5\u671f:2017-09-27", [40, 205, 340, 238], 0.92),
            OCRTokenData("\u91d1\u989d:10.50\u5143", [500, 820, 760, 855], 0.94),
        ]

        result = extract_layout_aware(tokens)

        self.assertEqual(result.fields.get("invoice_number", ""), "11142434")
        self.assertNotEqual(result.fields.get("invoice_number", ""), "137131730001")
        self.assertNotEqual(result.fields.get("invoice_number", ""), "91370100MA3ABC1234")

    def test_latin_vat_id_is_not_used_as_invoice_number(self):
        tokens = [
            OCRTokenData("VAT ID: MY123456789012", [40, 40, 360, 70], 0.92),
            OCRTokenData("POPULAR TRADING SDN BHD", [40, 90, 380, 120], 0.91),
            OCRTokenData("Invoice No: INV-2026-88", [430, 150, 760, 180], 0.93),
            OCRTokenData("Date: 2026-06-01", [430, 195, 700, 225], 0.93),
            OCRTokenData("Total RM 21.00", [520, 790, 760, 820], 0.94),
        ]

        result = extract_layout_aware(tokens)

        self.assertEqual(result.fields.get("invoice_number", ""), "INV-2026-88")
        self.assertNotEqual(result.fields.get("invoice_number", ""), "MY123456789012")

    def test_malaysian_popular_book_total_not_cash_change_or_savings(self):
        tokens = [
            OCRTokenData("POPULAR BOOK CO. (M) SDN BHD", [40, 35, 470, 70], 0.92),
            OCRTokenData("01/03/18 19:14 Slip No.: 0010104733", [40, 250, 620, 285], 0.88),
            OCRTokenData("Description Amount", [40, 410, 620, 440], 0.90),
            OCRTokenData("TOMBOW C/Tape CX5N 19.78 T", [40, 480, 650, 512], 0.86),
            OCRTokenData("L11 X-VENTURE UNEXPLA[BK] 12.00 Z", [40, 560, 650, 592], 0.86),
            OCRTokenData("Member Discount -1.20", [40, 610, 650, 642], 0.84),
            OCRTokenData("Total RM Incl. of GST 49.39", [40, 725, 650, 758], 0.91),
            OCRTokenData("Rounding Adj 0.01", [40, 768, 650, 800], 0.90),
            OCRTokenData("Total RM 49.40", [40, 810, 650, 842], 0.93),
            OCRTokenData("Cash -50.00", [40, 850, 650, 882], 0.92),
            OCRTokenData("CHANGE 0.60", [40, 890, 650, 922], 0.92),
            OCRTokenData("GST Summary Amount (RM) Tax (RM)", [40, 1000, 650, 1032], 0.90),
            OCRTokenData("T @ 6% 36.41 2.18", [40, 1040, 650, 1072], 0.90),
            OCRTokenData("Z @ 0% 10.80 0.00", [40, 1080, 650, 1112], 0.90),
            OCRTokenData("Total Savings -3.29", [40, 1170, 650, 1202], 0.91),
        ]

        result = extract_layout_aware(tokens)

        self.assertEqual(result.fields.get("vendor_name", ""), "POPULAR BOOK CO. (M) SDN BHD")
        self.assertEqual(result.fields.get("total_amount", ""), "49.40")
        self.assertEqual(result.fields.get("currency", ""), "RM")
        self.assertNotEqual(result.fields.get("total_amount", ""), "50.00")
        self.assertNotEqual(result.fields.get("total_amount", ""), "0.60")
        self.assertNotEqual(result.fields.get("total_amount", ""), "3.29")

    def test_noisy_book_tak_vendor_case_stays_recovered(self):
        tokens = [
            OCRTokenData("BOOKTA_k", [45, 40, 185, 66], 0.88),
            OCRTokenData("(TAMAN DAYA)", [195, 40, 390, 66], 0.87),
            OCRTokenData("SDM", [400, 40, 455, 66], 0.82),
            OCRTokenData("BKD", [465, 40, 520, 66], 0.82),
            OCRTokenData("Document No", [45, 190, 210, 218], 0.91),
            OCRTokenData("TDO1167t04", [250, 190, 405, 218], 0.86),
            OCRTokenData("Date", [45, 235, 110, 263], 0.92),
            OCRTokenData("25/12p018 8 13.39 PM", [250, 235, 560, 263], 0.72),
            OCRTokenData("Rounded Total (RM)", [45, 760, 280, 790], 0.92),
            OCRTokenData("9.00", [460, 760, 535, 790], 0.94),
        ]

        result = extract_layout_aware(tokens)
        vendor = result.fields.get("vendor_name", "")

        self.assertIn(vendor, {"BOOK TA-K (TAMAN DAYA) SDN BHD", "BOOKTA-K (TAMAN DAYA) SDN BHD"})
        self.assertEqual(result.fields.get("invoice_number", ""), "TDO1167t04")
        self.assertEqual(result.fields.get("total_amount", ""), "9.00")


class VendorNamePhase3ImprovementTests(SimpleTestCase):
    """Vendor cleanup regression tests."""

    def test_repeated_chinese_tax_bureau_collapses_to_single_vendor(self):
        repeated = (
            "\u9ed1\u9f99\u6c5f\u7701\u56fd\u5bb6\u7a0e\u52a1\u5c40"
            "\u9ed1\u9f99\u6c5f\u7701\u56fd\u5bb6\u7a0e\u52a1\u5c40"
        )
        tokens = [
            OCRTokenData(repeated, [40, 35, 800, 70], 0.90),
            OCRTokenData("\u53d1\u7968\u53f7\u7801: 56256948", [40, 120, 380, 150], 0.88),
            OCRTokenData("\u91d1\u989d: 34.00 \u5143", [40, 780, 300, 810], 0.88),
        ]

        result = extract_layout_aware(tokens)

        self.assertEqual(result.fields.get("vendor_name", ""), "\u9ed1\u9f99\u6c5f\u7701\u56fd\u5bb6\u7a0e\u52a1\u5c40")

    def test_government_preamble_is_removed_from_english_vendor(self):
        tokens = [
            OCRTokenData(
                "THE GOVERNMENT OF THE PEOPLE'S REPUBLIC OF BANGLADESH MAA ELECTRIC & STATIONERY",
                [40, 80, 1050, 115],
                0.90,
            ),
            OCRTokenData("Invoice No INV-1", [40, 170, 360, 200], 0.91),
            OCRTokenData("Total Amount BDT 10.00", [550, 820, 900, 850], 0.91),
        ]

        result = extract_layout_aware(tokens)

        self.assertEqual(result.fields.get("vendor_name", ""), "MAA ELECTRIC & STATIONERY")
        self.assertNotIn("GOVERNMENT", result.fields.get("vendor_name", ""))

    def test_vendor_address_tail_and_contact_name_do_not_win(self):
        tokens = [
            OCRTokenData("LIGHTROOM GALLERY SDN BHD NO: 28, JALAN ASTANA 1C", [40, 35, 760, 65], 0.88),
            OCRTokenData("Name ESWARAN 012-6369400", [40, 75, 500, 105], 0.95),
            OCRTokenData("BILL NO LCN00212", [40, 130, 320, 160], 0.88),
            OCRTokenData("Total RM 278.80", [500, 800, 760, 830], 0.91),
        ]

        result = extract_layout_aware(tokens)
        vendor = result.fields.get("vendor_name", "")

        self.assertEqual(vendor, "LIGHTROOM GALLERY SDN BHD")
        self.assertNotIn("JALAN", vendor)
        self.assertNotIn("ESWARAN", vendor)

    def test_chinese_store_address_does_not_override_top_brand_line(self):
        tokens = [
            OCRTokenData("\u542f\u5600\u542f\u5600 \u65b0\u4e00\u4ee3\u56fd\u6c11\u96f6\u98df", [40, 35, 600, 70], 0.82),
            OCRTokenData(
                "\u95e8\u5e97\u5730\u5740:\u5357\u660c\u5e02\u5357\u660c\u7ecf\u6d4e\u6280\u672f\u5f00\u53d1\u533a",
                [40, 600, 900, 640],
                0.94,
            ),
            OCRTokenData("\u5e94\u6536: 20.50", [40, 700, 300, 730], 0.90),
            OCRTokenData("\u5b9e\u6536: 25.50", [40, 740, 300, 770], 0.90),
        ]

        result = extract_layout_aware(tokens)
        vendor = result.fields.get("vendor_name", "")

        self.assertEqual(vendor, "\u542f\u5600\u542f\u5600\u65b0\u4e00\u4ee3\u56fd\u6c11\u96f6\u98df")
        self.assertNotIn("\u95e8\u5e97\u5730\u5740", vendor)

    def test_chinese_vat_company_label_extracts_company_not_tax_id(self):
        tokens = [
            OCRTokenData(
                "\u9500\u552e\u65b9\u540d\u79f0:\u4f73\u4f5c\u5929\u6210\uff08\u5317\u4eac\uff09\u79d1\u6280\u6709\u9650\u516c\u53f8 "
                "\u7eb3\u7a0e\u4eba\u8bc6\u522b\u53f7:91110108MA0012345X",
                [40, 100, 900, 130],
                0.90,
            ),
            OCRTokenData("\u53d1\u7968\u53f7\u7801:11142434", [40, 150, 360, 180], 0.90),
            OCRTokenData("\u4ef7\u7a0e\u5408\u8ba1:231.00\u5143", [500, 800, 820, 830], 0.92),
        ]

        result = extract_layout_aware(tokens)
        vendor = result.fields.get("vendor_name", "")

        self.assertEqual(vendor, "\u4f73\u4f5c\u5929\u6210(\u5317\u4eac)\u79d1\u6280\u6709\u9650\u516c\u53f8")
        self.assertNotIn("91110108", vendor)

    def test_chinese_vat_seller_block_wins_over_buyer_table_and_printer_text(self):
        tokens = [
            OCRTokenData("\u5317\u4eac\u589e\u503c\u7a0e\u4e13\u7528\u53d1\u7968", [747, 7, 1172, 47], 0.99),
            OCRTokenData("No 09750785", [1321, 47, 1512, 80], 0.99),
            OCRTokenData("\u5f00\u7968\u65e5\u671f: 2017\u5e7404\u670818\u65e5", [1382, 153, 1780, 190], 0.99),
            OCRTokenData("\u8d2d\u4e70\u65b9", [156, 213, 210, 260], 0.99),
            OCRTokenData("\u540d", [224, 200, 260, 235], 0.99),
            OCRTokenData("\u79f0:", [373, 203, 420, 235], 0.99),
            OCRTokenData("\u4e2d\u56fd\u79d1\u5b66\u9662\u81ea\u52a8\u5316\u7814\u7a76\u6240", [444, 218, 783, 246], 0.99),
            OCRTokenData("\u8d27\u7269\u6216\u5e94\u7a0e\u52b3\u52a1\u3001\u670d\u52a1\u540d\u79f0", [185, 373, 455, 405], 0.99),
            OCRTokenData("[2016]117\u53f7\u5317\u4eac\u5370\u949e\u6709\u9650\u516c\u53f8", [50, 397, 438, 430], 0.99),
            OCRTokenData("\u4f73\u80fd\u76f8\u673a", [154, 420, 278, 452], 0.99),
            OCRTokenData("32705.98", [1373, 452, 1480, 480], 0.99),
            OCRTokenData("17%", [1518, 454, 1570, 482], 0.99),
            OCRTokenData("5560.02", [1717, 457, 1810, 485], 0.99),
            OCRTokenData("\u9500\u552e\u65b9", [98, 826, 170, 860], 0.99),
            OCRTokenData("\u540d", [175, 826, 215, 860], 0.99),
            OCRTokenData("\u79f0:", [335, 832, 390, 865], 0.99),
            OCRTokenData("\u4f73\u4f5c\u5929\u6210\uff08\u5317\u4eac\uff09\u79d1\u6280\u6709\u9650\u516c\u53f8", [406, 837, 820, 882], 0.99),
            OCRTokenData("\u516c\u53f8", [1447, 864, 1510, 900], 0.99),
            OCRTokenData("\u7eb3\u7a0e\u4eba\u8bc6\u522b\u53f7:110116699572706", [172, 864, 700, 900], 0.99),
            OCRTokenData("\uffe532705.98", [1300, 730, 1450, 760], 0.99),
            OCRTokenData("\uffe55560.02", [1656, 739, 1800, 770], 0.99),
            OCRTokenData("\uffe538266.00", [1497, 793, 1660, 825], 0.99),
        ]

        result = extract_layout_aware(tokens)
        vendor = result.fields.get("vendor_name", "")

        self.assertEqual(vendor, "\u4f73\u4f5c\u5929\u6210(\u5317\u4eac)\u79d1\u6280\u6709\u9650\u516c\u53f8")
        self.assertNotIn("\u8d2d\u4e70\u65b9", vendor)
        self.assertNotIn("\u5370\u949e", vendor)
        self.assertNotEqual(vendor, "\u516c\u53f8")


class RegressionGuardX510Tests(SimpleTestCase):
    """Keep the established X510 receipt cases working."""

    def test_x510_still_extracts_correctly(self):
        """X510 clean receipt: GST Summary path, constraint engine, RM currency."""
        tokens = [
            OCRTokenData("99 SPEED MART", [35, 35, 270, 62], 0.83),
            OCRTokenData("S/B", [282, 35, 330, 62], 0.82),
            OCRTokenData("(517537-X)", [342, 35, 500, 62], 0.80),
            OCRTokenData("INWWICE", [35, 155, 145, 182], 0.58),
            OCRTokenData("8Q", [155, 155, 190, 182], 0.55),
            OCRTokenData("18222/102/TD341", [250, 155, 500, 182], 0.76),
            OCRTokenData("z0-14-17", [35, 195, 155, 222], 0.48),
            OCRTokenData("GST Summary", [35, 470, 210, 498], 0.78),
            OCRTokenData("Amount", [235, 510, 325, 536], 0.76),
            OCRTokenData("Tax", [420, 510, 470, 536], 0.76),
            OCRTokenData("160.17", [235, 548, 330, 575], 0.80),
            OCRTokenData("9.61", [420, 548, 488, 575], 0.79),
            OCRTokenData("Tote] Sales", [35, 645, 185, 674], 0.62),
            OCRTokenData("RM", [330, 645, 370, 674], 0.88),
            OCRTokenData("169.80", [420, 645, 515, 674], 0.84),
            OCRTokenData("Rourding", [35, 690, 155, 718], 0.58),
            OCRTokenData("0.02", [420, 690, 485, 718], 0.76),
            OCRTokenData("CaSH", [35, 735, 105, 763], 0.72),
            OCRTokenData("200.00", [420, 735, 515, 763], 0.82),
            OCRTokenData("CWaNGE", [35, 780, 125, 808], 0.68),
            OCRTokenData("30.20", [420, 780, 500, 808], 0.80),
        ]
        result = extract_layout_aware(tokens)
        self.assertEqual(result.fields.get("vendor_name", ""), "99 SPEED MART S/B (517537-X)")
        self.assertEqual(result.fields.get("invoice_number", ""), "18222/102/TD341")
        self.assertEqual(result.fields.get("date", ""), "20-11-17")
        self.assertEqual(result.fields.get("subtotal", ""), "160.17")
        self.assertEqual(result.fields.get("tax_amount", ""), "9.61")
        self.assertEqual(result.fields.get("total_amount", ""), "169.80")
        self.assertEqual(result.fields.get("currency", ""), "RM")
