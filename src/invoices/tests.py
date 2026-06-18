from decimal import Decimal

from django.test import SimpleTestCase

from .services.anomalies import check_invoice_anomalies
from .services.constraint_engine import (
    AmountEvidence,
    collect_amount_evidence,
    find_best_assignment,
)
from .services.evaluation import values_equal
from .services.extraction import _line_groups, extract_layout_aware, extract_with_regex
from .services.layoutlmv3_baseline import (
    assign_bio_labels,
    assign_bio_labels_with_coverage,
    decode_token_predictions,
    decode_token_predictions_with_context,
    normalize_bbox,
)
from .services.ocr import OCRTokenData


class ExtractionServiceTests(SimpleTestCase):
    def test_regex_and_layout_extract_basic_fields(self):
        tokens = [
            OCRTokenData("TAX INVOICE", [50, 30, 300, 70], 0.95),
            OCRTokenData("Blue River Trading Ltd", [50, 150, 330, 180], 0.90),
            OCRTokenData("Invoice No: INV-2026-0001", [600, 150, 920, 180], 0.88),
            OCRTokenData("Date: 2026-05-01", [600, 190, 900, 220], 0.88),
            OCRTokenData("Tax: USD 20.00", [620, 820, 880, 850], 0.92),
            OCRTokenData("Total Amount: USD 220.00", [620, 865, 940, 895], 0.93),
        ]
        regex = extract_with_regex(tokens)
        layout = extract_layout_aware(tokens)
        self.assertEqual(regex.fields["invoice_number"], "INV-2026-0001")
        self.assertEqual(layout.fields["total_amount"], "220.00")
        self.assertEqual(layout.fields["currency"], "USD")

    def test_layout_extracts_noisy_receipt_fields(self):
        tokens = [
            OCRTokenData("BOOKTA_k", [45, 40, 185, 66], 0.88),
            OCRTokenData("(TAMAN DAYA)", [195, 40, 390, 66], 0.87),
            OCRTokenData("SDM", [400, 40, 455, 66], 0.82),
            OCRTokenData("BKD", [465, 40, 520, 66], 0.82),
            OCRTokenData("Document No", [45, 190, 210, 218], 0.91),
            OCRTokenData("TDO1167t04", [250, 190, 405, 218], 0.86),
            OCRTokenData("Date", [45, 235, 110, 263], 0.92),
            OCRTokenData("25/12p018 8 13.39 PM", [250, 235, 560, 263], 0.72),
            OCRTokenData("Qty", [45, 500, 90, 528], 0.90),
            OCRTokenData("Unit Price", [340, 500, 455, 528], 0.90),
            OCRTokenData("Cash", [45, 705, 100, 733], 0.90),
            OCRTokenData("10.00", [460, 705, 535, 733], 0.90),
            OCRTokenData("Rounded Total (RM)", [45, 760, 280, 790], 0.92),
            OCRTokenData("9.00", [460, 760, 535, 790], 0.94),
            OCRTokenData("Change", [45, 810, 125, 838], 0.90),
            OCRTokenData("1.00", [460, 810, 535, 838], 0.90),
        ]

        layout = extract_layout_aware(tokens)

        self.assertIn(layout.fields["vendor_name"], {"BOOK TA-K (TAMAN DAYA) SDN BHD", "BOOKTA-K (TAMAN DAYA) SDN BHD"})
        self.assertEqual(layout.fields["invoice_number"], "TDO1167t04")
        self.assertEqual(layout.fields["date"], "25/12/2018")
        self.assertEqual(layout.fields["total_amount"], "9.00")
        self.assertEqual(layout.fields["currency"], "RM")
        self.assertEqual(layout.fields["tax_amount"], "")
        self.assertIn("source_text", layout.evidence["total_amount"])

    def test_amount_normalization_keeps_decimal_separator(self):
        tokens = [
            OCRTokenData("Rounded Total", [10, 500, 160, 530], 0.95),
            OCRTokenData("9,00", [220, 500, 290, 530], 0.95),
            OCRTokenData("Cash", [10, 550, 80, 580], 0.95),
            OCRTokenData("900", [220, 550, 290, 580], 0.95),
        ]

        layout = extract_layout_aware(tokens)

        self.assertEqual(layout.fields["total_amount"], "9.00")

    def test_layout_extracts_x510_low_quality_receipt(self):
        tokens = [
            OCRTokenData("99 SPEED MART", [35, 35, 270, 62], 0.83),
            OCRTokenData("S/B", [282, 35, 330, 62], 0.82),
            OCRTokenData("(517537-X)", [342, 35, 500, 62], 0.80),
            OCRTokenData("GST ID", [35, 90, 120, 116], 0.74),
            OCRTokenData("001234567890", [160, 90, 335, 116], 0.74),
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

        layout = extract_layout_aware(tokens)

        self.assertEqual(layout.fields["vendor_name"], "99 SPEED MART S/B (517537-X)")
        self.assertEqual(layout.fields["invoice_number"], "18222/102/TD341")
        self.assertEqual(layout.fields["date"], "20-11-17")
        self.assertEqual(layout.fields["subtotal"], "160.17")
        self.assertEqual(layout.fields["tax_amount"], "9.61")
        self.assertEqual(layout.fields["total_amount"], "169.80")
        self.assertEqual(layout.fields["currency"], "RM")
        self.assertIn("source_text", layout.evidence["invoice_number"])
        self.assertIn("financial consistency", layout.evidence["total_amount"]["explanation"].lower())


class LayoutLMv3BaselineUtilityTests(SimpleTestCase):
    def test_normalize_bbox_scales_to_layoutlm_range(self):
        self.assertEqual(normalize_bbox([50, 25, 150, 75], 200, 100), [250, 250, 750, 750])

    def test_assign_bio_labels_matches_ground_truth_spans(self):
        tokens = [
            OCRTokenData("ACME", [0, 0, 20, 10], 0.9),
            OCRTokenData("LTD", [22, 0, 40, 10], 0.9),
            OCRTokenData("Invoice", [0, 20, 40, 30], 0.9),
            OCRTokenData("No", [42, 20, 55, 30], 0.9),
            OCRTokenData("INV-100", [60, 20, 110, 30], 0.9),
            OCRTokenData("Total", [0, 50, 35, 60], 0.9),
            OCRTokenData("USD", [40, 50, 65, 60], 0.9),
            OCRTokenData("21.00", [70, 50, 115, 60], 0.9),
        ]
        labels = assign_bio_labels(
            tokens,
            {
                "vendor_name": "ACME LTD",
                "invoice_number": "INV-100",
                "total_amount": "21.00",
                "currency": "USD",
            },
        )

        self.assertEqual(labels[0:2], ["B-VENDOR_NAME", "I-VENDOR_NAME"])
        self.assertEqual(labels[4], "B-INVOICE_NUMBER")
        self.assertEqual(labels[6], "B-CURRENCY")
        self.assertEqual(labels[7], "B-TOTAL_AMOUNT")

    def test_assign_bio_labels_matches_embedded_label_values(self):
        tokens = [
            OCRTokenData("Invoice No: INV-2026-0001", [0, 0, 180, 20], 0.9),
            OCRTokenData("Date: 2026年06月15日", [0, 25, 180, 45], 0.9),
            OCRTokenData("¥", [0, 50, 20, 70], 0.9),
            OCRTokenData("21", [25, 50, 65, 70], 0.9),
        ]

        labels, coverage = assign_bio_labels_with_coverage(
            tokens,
            {
                "invoice_number": "INV-2026-0001",
                "date": "2026-06-15",
                "total_amount": "21.00",
                "currency": "CNY",
            },
        )

        self.assertEqual(labels[0], "B-INVOICE_NUMBER")
        self.assertEqual(labels[1], "B-DATE")
        self.assertEqual(labels[2], "B-CURRENCY")
        self.assertEqual(labels[3], "B-TOTAL_AMOUNT")
        self.assertTrue(coverage["invoice_number"]["matched"])
        self.assertTrue(coverage["date"]["matched"])
        self.assertTrue(coverage["currency"]["matched"])

    def test_assign_bio_labels_tolerates_invoice_ocr_confusions(self):
        tokens = [
            OCRTokenData("Receipt", [0, 0, 60, 20], 0.9),
            OCRTokenData("N0", [65, 0, 85, 20], 0.75),
            OCRTokenData("TD0I167I04", [90, 0, 190, 20], 0.72),
        ]

        labels = assign_bio_labels(tokens, {"invoice_number": "TDO1167I04"})

        self.assertEqual(labels[2], "B-INVOICE_NUMBER")

    def test_decode_token_predictions_returns_fields(self):
        tokens = [
            OCRTokenData("ACME", [0, 0, 0, 0], 1.0),
            OCRTokenData("LTD", [0, 0, 0, 0], 1.0),
            OCRTokenData("INV-100", [0, 0, 0, 0], 1.0),
            OCRTokenData("USD", [0, 0, 0, 0], 1.0),
            OCRTokenData("21.0", [0, 0, 0, 0], 1.0),
        ]
        fields = decode_token_predictions(
            tokens,
            ["B-VENDOR_NAME", "I-VENDOR_NAME", "B-INVOICE_NUMBER", "B-CURRENCY", "B-TOTAL_AMOUNT"],
        )

        self.assertEqual(fields["vendor_name"], "ACME LTD")
        self.assertEqual(fields["invoice_number"], "INV-100")
        self.assertEqual(fields["currency"], "USD")
        self.assertEqual(fields["total_amount"], "21.00")

    def test_context_decoder_repairs_label_tokens_to_values(self):
        tokens = [
            OCRTokenData("Invoice No:", [0, 0, 20, 10], 1.0),
            OCRTokenData("INV-2026-0001", [25, 0, 80, 10], 1.0),
            OCRTokenData("Date:", [0, 15, 20, 25], 1.0),
            OCRTokenData("2026-05-01", [25, 15, 80, 25], 1.0),
            OCRTokenData("Total Amount:", [0, 30, 35, 40], 1.0),
            OCRTokenData("USD 231.00", [40, 30, 90, 40], 1.0),
        ]

        fields = decode_token_predictions_with_context(
            tokens,
            ["B-INVOICE_NUMBER", "O", "B-DATE", "O", "B-TOTAL_AMOUNT", "O"],
        )

        self.assertEqual(fields["invoice_number"], "INV-2026-0001")
        self.assertEqual(fields["date"], "2026-05-01")
        self.assertEqual(fields["total_amount"], "231.00")
        self.assertEqual(fields["currency"], "USD")

    def test_context_decoder_handles_european_amount_and_month_date(self):
        tokens = [
            OCRTokenData("Invoice no: 72744145", [0, 0, 90, 10], 1.0),
            OCRTokenData("Invoice date: 23-Nov-2019", [0, 15, 110, 25], 1.0),
            OCRTokenData("Total", [0, 30, 20, 40], 1.0),
            OCRTokenData("$ 3 085,50", [25, 30, 80, 40], 1.0),
        ]

        fields = decode_token_predictions_with_context(
            tokens,
            ["B-INVOICE_NUMBER", "B-DATE", "B-TOTAL_AMOUNT", "O"],
        )

        self.assertEqual(fields["invoice_number"], "72744145")
        self.assertEqual(fields["date"], "23-Nov-2019")
        self.assertEqual(fields["total_amount"], "3085.50")
        self.assertEqual(fields["currency"], "USD")

    def test_vendor_trims_embedded_business_address(self):
        tokens = [
            OCRTokenData(
                "Blue River Trading Ltd Business Address: 18 Market Road, City Center Date: 2026-05-01",
                [40, 40, 850, 70],
                0.94,
            ),
            OCRTokenData("Invoice No: INV-1", [500, 120, 760, 150], 0.90),
            OCRTokenData("Total Amount USD 231.00", [500, 820, 820, 850], 0.90),
        ]

        layout = extract_layout_aware(tokens)

        self.assertEqual(layout.fields["vendor_name"], "BLUE RIVER TRADING LTD")
        self.assertNotIn("BUSINESS ADDRESS", layout.fields["vendor_name"])
        self.assertNotIn("DATE", layout.fields["vendor_name"])

    def test_vendor_removes_government_revenue_preamble(self):
        tokens = [
            OCRTokenData("MAA LOGO", [40, 35, 180, 65], 0.95),
            OCRTokenData(
                "THE GOVERNMENT OF THE PEOPLE'S REPUBLIC OF BANGLADESH NATONAL BOARD REVENUE MAA ELECTRIC & STATIONERY",
                [40, 80, 1050, 115],
                0.90,
            ),
            OCRTokenData("Invoice No INV-226483", [40, 170, 360, 200], 0.91),
            OCRTokenData("Total Amount BDT 23250.00", [550, 820, 900, 850], 0.91),
        ]

        layout = extract_layout_aware(tokens)

        self.assertEqual(layout.fields["vendor_name"], "MAA ELECTRIC & STATIONERY")
        self.assertNotIn("GOVERNMENT", layout.fields["vendor_name"])
        self.assertNotIn("LOGO", layout.fields["vendor_name"])

    def test_vendor_rejects_document_title_candidate(self):
        tokens = [
            OCRTokenData("COMMERCIAL INVOICE", [40, 35, 330, 65], 0.97),
            OCRTokenData("Kirk, Murphy and Daniels", [40, 85, 420, 115], 0.91),
            OCRTokenData("Invoice Number 1Y7M4d-846", [500, 130, 850, 160], 0.92),
            OCRTokenData("Total USD 42.00", [560, 840, 780, 870], 0.92),
        ]

        layout = extract_layout_aware(tokens)

        self.assertIn("KIRK", layout.fields["vendor_name"])
        self.assertNotEqual(layout.fields["vendor_name"], "COMMERCIAL INVOICE")

    def test_vendor_preserves_sdn_bhd_when_address_marker_splits_line(self):
        tokens = [
            OCRTokenData("LIGHTROOM GALLERY NO: 28, TALAN ASTANA 1C, SDN BHD", [40, 35, 760, 65], 0.88),
            OCRTokenData("BILL NO LCN00212", [40, 130, 320, 160], 0.88),
            OCRTokenData("Total RM 278.80", [500, 800, 760, 830], 0.91),
        ]

        layout = extract_layout_aware(tokens)

        self.assertEqual(layout.fields["vendor_name"], "LIGHTROOM GALLERY SDN BHD")

    def test_chinese_tax_bureau_can_be_vendor_issuer(self):
        tokens = [
            OCRTokenData("\u9ed1\u9f99\u6c5f\u7701\u56fd\u5bb6\u7a0e\u52a1\u5c40\u901a\u7528\u673a\u6253\u53d1\u7968", [40, 35, 600, 70], 0.90),
            OCRTokenData("\u53d1\u7968\u53f7\u7801: 56256948", [40, 120, 380, 150], 0.88),
            OCRTokenData("\u91d1\u989d: 34.00 \u5143", [40, 780, 300, 810], 0.88),
        ]

        layout = extract_layout_aware(tokens)

        self.assertEqual(layout.fields["vendor_name"], "\u9ed1\u9f99\u6c5f\u7701\u56fd\u5bb6\u7a0e\u52a1\u5c40")

    def test_vendor_evaluation_ignores_punctuation_and_minor_ocr_noise(self):
        self.assertTrue(values_equal("vendor_name", "Barnes, Garcia and Martin", "BARNES GARCIA AND MARTIN"))
        self.assertTrue(values_equal("vendor_name", "PERNIAGAAN ZHENG HUI", "PERNIAGAAN ZHENG KUI"))
        self.assertFalse(values_equal("vendor_name", "MAA Electric & Stationery", "MAA LOGO"))

    def test_line_item_total_does_not_override_invoice_total(self):
        tokens = [
            OCRTokenData("North Star Office Supplies", [45, 45, 380, 75], 0.94),
            OCRTokenData("Invoice No: INV-2026-0002", [600, 120, 930, 150], 0.93),
            OCRTokenData("Date: 2026-05-04", [600, 165, 860, 195], 0.93),
            OCRTokenData("Description Qty Unit Price Line Total", [45, 400, 650, 430], 0.92),
            OCRTokenData("Consulting service 2 127.50 255.00", [45, 445, 650, 475], 0.91),
            OCRTokenData("Training materials 1 144.75 144.75", [45, 490, 650, 520], 0.91),
            OCRTokenData("Subtotal: USD 399.75", [600, 735, 930, 765], 0.96),
            OCRTokenData("Tax: USD 39.98", [600, 780, 900, 810], 0.96),
            OCRTokenData("Total Amount Due", [600, 825, 850, 855], 0.96),
            OCRTokenData("USD 439.", [870, 825, 1020, 855], 0.96),
        ]

        layout = extract_layout_aware(tokens)

        self.assertEqual(layout.fields.get("subtotal", ""), "399.75")
        self.assertEqual(layout.fields.get("tax_amount", ""), "39.98")
        self.assertEqual(layout.fields.get("total_amount", ""), "439.73")
        self.assertNotEqual(layout.fields.get("total_amount", ""), "255.00")
        self.assertNotIn("LINE TOTAL", layout.evidence.get("total_amount", {}).get("source_text", "").upper())

    def test_merged_total_cash_change_line_with_broken_decimal_spacing(self):
        tokens = [
            OCRTokenData("LIGHTROOM GALLERY SDN BHD", [40, 35, 420, 65], 0.90),
            OCRTokenData("Bill No: LCN00211", [40, 130, 320, 160], 0.89),
            OCRTokenData("Date: 20/11/2017", [40, 175, 330, 205], 0.89),
            OCRTokenData("GST Summary Amount (RM) Tax (RM)", [40, 650, 560, 680], 0.84),
            OCRTokenData("SR ZR/OS/EZ 6% 37 . 55 0. 00 2. 25 0. 00", [40, 690, 680, 720], 0.80),
            OCRTokenData("CHANGE : TOTAL CASII RM RM RM 39. 80 39. 80 0_ 00", [40, 740, 720, 785], 0.70),
        ]

        layout = extract_layout_aware(tokens)

        self.assertEqual(layout.fields.get("subtotal", ""), "37.55")
        self.assertEqual(layout.fields.get("tax_amount", ""), "2.25")
        self.assertEqual(layout.fields.get("total_amount", ""), "39.80")
        self.assertEqual(
            layout.evidence.get("total_amount", {}).get("method", ""),
            "merged_payment_total_candidate",
        )

    def test_x51005268408_regression(self):
        """Extract the key fields from a noisy Malaysian receipt."""
        tokens = [
            # Noisy vendor header followed by a clearer address.
            OCRTokenData("99 SPEED MAFT", [35, 35, 250, 62], 0.56),
            OCRTokenData("S/8", [260, 35, 300, 62], 0.52),
            OCRTokenData("{517537-X)", [310, 35, 490, 62], 0.50),
            OCRTokenData("1413-SETIA ALAM 2", [35, 75, 280, 100], 0.72),
            OCRTokenData("INVOICE NI", [35, 145, 180, 170], 0.60),
            OCRTokenData("18222/102/70341", [250, 145, 500, 170], 0.74),
            OCRTokenData("20-14-17", [35, 185, 155, 210], 0.48),
            OCRTokenData("GST Summary", [35, 460, 210, 488], 0.78),
            OCRTokenData("Amount", [235, 500, 325, 526], 0.76),
            OCRTokenData("Tax", [420, 500, 470, 526], 0.76),
            OCRTokenData("160.47", [235, 538, 330, 565], 0.80),
            OCRTokenData("9.61", [420, 538, 488, 565], 0.79),
            OCRTokenData("Tote] Sales", [35, 635, 185, 664], 0.62),
            OCRTokenData("RH", [330, 635, 370, 664], 0.70),
            OCRTokenData("169.80", [420, 635, 515, 664], 0.84),
            OCRTokenData("CaSH", [35, 725, 105, 753], 0.72),
            OCRTokenData("$", [200, 725, 220, 753], 0.45),
            OCRTokenData("200.00", [420, 725, 515, 753], 0.82),
            OCRTokenData("CWaNGE", [35, 770, 125, 798], 0.68),
            OCRTokenData("30.20", [420, 770, 500, 798], 0.80),
        ]

        layout = extract_layout_aware(tokens)

        self.assertNotEqual(layout.fields.get("vendor_name", ""), "1413-SETIA ALAM 2")
        self.assertIn("99 SPEED", layout.fields.get("vendor_name", ""))
        self.assertEqual(layout.fields.get("invoice_number", ""), "18222/102/70341")
        self.assertEqual(layout.fields.get("date", ""), "20-11-17")
        self.assertNotEqual(layout.fields.get("tax_amount", ""), "200.00")
        self.assertEqual(layout.fields.get("tax_amount", ""), "9.61")
        self.assertNotEqual(layout.fields.get("total_amount", ""), "200.00")
        self.assertEqual(layout.fields.get("total_amount", ""), "169.80")
        self.assertNotEqual(layout.fields.get("currency", ""), "USD")
        self.assertEqual(layout.fields.get("currency", ""), "RM")


    def test_x51005268408_spaced_decimal_and_noisy_total(self):
        """Handle noisy GST labels and split decimal amounts."""
        tokens = [
            OCRTokenData("99 SFEOO HART", [35, 35, 250, 62], 0.52),
            OCRTokenData("S/8", [260, 35, 300, 62], 0.50),
            OCRTokenData("{519537-X)", [310, 35, 490, 62], 0.48),
            OCRTokenData("INVOICE NI", [35, 145, 180, 170], 0.60),
            OCRTokenData("18222/102/70341", [250, 145, 500, 170], 0.74),
            OCRTokenData("20-14-17", [35, 185, 155, 210], 0.48),
            OCRTokenData("6S1 Surary", [35, 460, 210, 488], 0.65),
            OCRTokenData("Amount", [235, 500, 325, 526], 0.74),
            OCRTokenData("Tax", [420, 500, 470, 526], 0.74),
            # OCR split 9.61 into two tokens.
            OCRTokenData("160.47", [235, 538, 330, 565], 0.78),
            OCRTokenData("9", [420, 538, 445, 565], 0.72),
            OCRTokenData("61", [448, 538, 488, 565], 0.70),
            OCRTokenData("Tote] Sales", [35, 635, 185, 664], 0.58),
            OCRTokenData("RK", [330, 635, 370, 664], 0.65),
            OCRTokenData("169,80", [420, 635, 515, 664], 0.76),
            OCRTokenData("CaSH", [35, 725, 105, 753], 0.72),
            OCRTokenData("$", [200, 725, 220, 753], 0.45),
            OCRTokenData("200.00", [420, 725, 515, 753], 0.82),
            OCRTokenData("CWaNGE", [35, 770, 125, 798], 0.68),
            OCRTokenData("30.20", [420, 770, 500, 798], 0.80),
        ]

        layout = extract_layout_aware(tokens)

        self.assertNotEqual(
            layout.fields.get("tax_amount", ""), "200.00",
            "Cash payment 200.00 must not be selected as tax_amount",
        )
        self.assertNotEqual(
            layout.fields.get("total_amount", ""), "200.00",
            "Cash payment 200.00 must not be selected as total_amount",
        )

        self.assertEqual(
            layout.fields.get("tax_amount", ""), "9.61",
            "Tax should be 9.61 from GST Summary spaced-decimal repair",
        )
        self.assertEqual(
            layout.fields.get("total_amount", ""), "169.80",
            "Total should be 169.80 from Total Sales line",
        )
        self.assertEqual(layout.fields.get("currency", ""), "RM")
        tax_role = layout.evidence.get("tax_amount", {}).get("amount_role", "")
        total_role = layout.evidence.get("total_amount", {}).get("amount_role", "")
        self.assertNotEqual(tax_role, "cash_paid", "tax_amount evidence must not have amount_role=cash_paid")
        self.assertNotEqual(total_role, "cash_paid", "total_amount evidence must not have amount_role=cash_paid")


class ConstraintEngineTests(SimpleTestCase):
    """Unit tests for the financial constraint engine."""

    def _make_evidence(self, value_str, label_role="unknown", zone_ratio=0.70):
        """Build minimal amount evidence for a test case."""
        return AmountEvidence(
            value=Decimal(value_str),
            normalized=value_str,
            source_text=f"line with {value_str}",
            bbox=[0, 0, 100, 20],
            ocr_confidence=0.80,
            label_role=label_role,
            zone_ratio=zone_ratio,
            line=None,
        )

    def test_label_free_assignment(self):
        """Assign amount roles without label context."""
        evidence = [
            self._make_evidence("160.17", "unknown", 0.60),
            self._make_evidence("9.61",   "unknown", 0.60),
            self._make_evidence("169.80", "unknown", 0.70),
            self._make_evidence("200.00", "unknown", 0.85),
            self._make_evidence("30.20",  "unknown", 0.88),
        ]
        result = find_best_assignment(evidence)
        self.assertIsNotNone(result, "Engine should find a consistent assignment")
        self.assertIsNotNone(result.total)
        self.assertEqual(result.total.normalized, "169.80",
                         "Total should be 169.80 — the only financially consistent total")
        self.assertIsNotNone(result.tax)
        self.assertEqual(result.tax.normalized, "9.61",
                         "Tax should be 9.61 (subtotal+tax≈total)")
        self.assertIsNotNone(result.subtotal)
        self.assertEqual(result.subtotal.normalized, "160.17")

    def test_wrong_label_overridden_by_constraints(self):
        """Financial consistency overrides an incorrect total label."""
        evidence = [
            self._make_evidence("160.17", "unknown", 0.60),
            self._make_evidence("9.61",   "unknown", 0.60),
            self._make_evidence("169.80", "unknown", 0.70),  # unlabeled correct total
            self._make_evidence("200.00", "total",   0.82),  # mislabeled as total
            self._make_evidence("30.20",  "unknown", 0.88),
        ]
        result = find_best_assignment(evidence)
        self.assertIsNotNone(result)
        self.assertEqual(result.total.normalized, "169.80",
                         "Constraint laws must override the mislabeled total=200.00")

    def test_partial_assignment_no_cash(self):
        """Assign subtotal, tax, and total without payment evidence."""
        evidence = [
            self._make_evidence("160.17", "subtotal", 0.60),
            self._make_evidence("9.61",   "tax",      0.60),
            self._make_evidence("169.80", "total",    0.70),
        ]
        result = find_best_assignment(evidence)
        self.assertIsNotNone(result)
        self.assertEqual(result.total.normalized, "169.80")
        self.assertEqual(result.tax.normalized, "9.61")
        self.assertEqual(result.subtotal.normalized, "160.17")
        self.assertTrue(
            result.cash_paid is None or result.change is None,
            "No cash/change evidence should mean those slots are None",
        )
        self.assertIn("law1_subtotal_tax_total", result.satisfied_laws)


class IntegrationConstraintTests(SimpleTestCase):
    """Integration test: constraint engine resolves amounts when GST Summary is absent."""

    def test_no_gst_summary_constraint_engine_recovers_fields(self):
        """Recover amounts when the GST summary label is missing."""
        tokens = [
            OCRTokenData("99 SPEED MAFT", [35, 35, 250, 62], 0.56),
            OCRTokenData("S/8", [260, 35, 300, 62], 0.52),
            OCRTokenData("INVOICE NI", [35, 145, 180, 170], 0.60),
            OCRTokenData("18222/102/70341", [250, 145, 500, 170], 0.74),
            OCRTokenData("20-14-17", [35, 185, 155, 210], 0.48),
            # Deliberately unlabeled subtotal and tax.
            OCRTokenData("160.47", [235, 538, 330, 565], 0.80),   # subtotal (no label)
            OCRTokenData("9.61",   [420, 538, 488, 565], 0.79),   # tax (no label)
            OCRTokenData("Tote] Sales", [35, 635, 185, 664], 0.62),
            OCRTokenData("RH", [330, 635, 370, 664], 0.70),
            OCRTokenData("169.80", [420, 635, 515, 664], 0.84),
            OCRTokenData("CaSH", [35, 725, 105, 753], 0.72),
            OCRTokenData("200.00", [420, 725, 515, 753], 0.82),
            OCRTokenData("CWaNGE", [35, 770, 125, 798], 0.68),
            OCRTokenData("30.20", [420, 770, 500, 798], 0.80),
        ]

        layout = extract_layout_aware(tokens)

        self.assertNotEqual(layout.fields.get("tax_amount", ""), "200.00",
                            "Cash 200.00 must not be selected as tax")
        self.assertNotEqual(layout.fields.get("total_amount", ""), "200.00",
                            "Cash 200.00 must not be selected as total")
        self.assertEqual(layout.fields.get("tax_amount", ""), "9.61",
                         "Constraint engine must recover tax=9.61 from financial laws")
        self.assertEqual(layout.fields.get("total_amount", ""), "169.80",
                         "Total from Total Sales label or constraint engine")
        self.assertEqual(layout.fields.get("currency", ""), "RM")


class AnomalyServiceTests(SimpleTestCase):
    def test_total_from_payment_line_anomaly(self):
        evidence = {
            "total_amount": {"amount_role": "cash_paid", "value": "200.00"},
            "tax_amount": {"amount_role": "tax_amount", "value": "9.61"},
        }
        anomalies = check_invoice_anomalies(
            {
                "invoice_number": "INV-1",
                "date": "2026-01-01",
                "vendor_name": "Test Vendor",
                "total_amount": "200.00",
                "currency": "RM",
            },
            evidence=evidence,
        )
        codes = {a.code for a in anomalies}
        self.assertIn("total_from_payment_line", codes)

    def test_tax_from_payment_line_anomaly(self):
        evidence = {
            "total_amount": {"amount_role": "total_amount", "value": "169.80"},
            "tax_amount": {"amount_role": "cash_paid", "value": "200.00"},
        }
        anomalies = check_invoice_anomalies(
            {
                "invoice_number": "INV-2",
                "date": "2026-01-01",
                "vendor_name": "Test Vendor",
                "total_amount": "169.80",
                "tax_amount": "200.00",
                "currency": "RM",
            },
            evidence=evidence,
        )
        codes = {a.code for a in anomalies}
        self.assertIn("tax_from_payment_line", codes)
        # tax 200 > total 169.80 → also triggers tax_greater_than_total
        self.assertIn("tax_greater_than_total", codes)

    def test_business_rule_anomalies(self):
        anomalies = check_invoice_anomalies(
            {
                "invoice_number": "INV-1",
                "date": "2099-01-01",
                "vendor_name": "Vendor",
                "tax_amount": "120",
                "total_amount": "100",
                "currency": "USD",
            },
            {"invoice_number": 0.9, "date": 0.9, "vendor_name": 0.9, "total_amount": 0.9, "currency": 0.9},
            previous_invoice_numbers={"INV-1"},
        )
        codes = {item.code for item in anomalies}
        self.assertIn("duplicate_invoice_number", codes)
        self.assertIn("future_date", codes)
        self.assertIn("tax_greater_than_total", codes)

    def test_tax_equals_total_anomaly(self):
        """Tax equal to total must trigger both tax_equals_total and tax_greater_than_total."""
        anomalies = check_invoice_anomalies(
            {
                "invoice_number": "INV-10",
                "date": "2026-01-01",
                "vendor_name": "Vendor",
                "total_amount": "169.80",
                "tax_amount": "169.80",
                "currency": "RM",
            }
        )
        codes = {a.code for a in anomalies}
        self.assertIn("tax_equals_total", codes,
                      "Tax equaling total must raise tax_equals_total anomaly")
        self.assertIn("tax_greater_than_total", codes,
                      "tax >= total check (with equality) must also raise tax_greater_than_total")

    def test_tax_zero_no_anomaly(self):
        """Tax of 0.00 is valid — no financial anomaly must be raised."""
        anomalies = check_invoice_anomalies(
            {
                "invoice_number": "INV-11",
                "date": "2026-01-01",
                "vendor_name": "Vendor",
                "subtotal": "100.00",
                "tax_amount": "0.00",
                "total_amount": "100.00",
                "currency": "RM",
            }
        )
        codes = {a.code for a in anomalies}
        self.assertNotIn("tax_greater_than_total", codes,
                         "0.00 tax must not trigger tax_greater_than_total")
        self.assertNotIn("tax_equals_total", codes,
                         "0.00 tax must not trigger tax_equals_total (total is non-zero)")
        self.assertNotIn("subtotal_tax_total_mismatch", codes,
                         "100.00 + 0.00 = 100.00 must not be flagged as mismatch")


class ConstraintEngineExtraTests(SimpleTestCase):
    """Additional rule-coverage tests for the financial constraint engine."""

    def _make_evidence(self, value_str, label_role="unknown", zone_ratio=0.70):
        return AmountEvidence(
            value=Decimal(value_str),
            normalized=value_str,
            source_text=f"line with {value_str}",
            bbox=[0, 0, 100, 20],
            ocr_confidence=0.80,
            label_role=label_role,
            zone_ratio=zone_ratio,
            line=None,
        )

    def test_subtotal_plus_tax_equals_total(self):
        """Law 1: subtotal + tax = total must yield a consistent assignment."""
        evidence = [
            self._make_evidence("80.00", "subtotal", 0.55),
            self._make_evidence("9.00",  "tax",      0.58),
            self._make_evidence("89.00", "total",    0.65),
        ]
        result = find_best_assignment(evidence)
        self.assertIsNotNone(result)
        self.assertEqual(result.total.normalized, "89.00")
        self.assertEqual(result.subtotal.normalized, "80.00")
        self.assertEqual(result.tax.normalized, "9.00")
        self.assertIn("law1_subtotal_tax_total", result.satisfied_laws)

    def test_cash_minus_total_equals_change(self):
        """Law 2: cash_paid - total = change must be satisfied in the best assignment."""
        evidence = [
            self._make_evidence("80.00",  "subtotal",  0.55),
            self._make_evidence("9.00",   "tax",       0.58),
            self._make_evidence("89.00",  "total",     0.65),
            self._make_evidence("100.00", "cash_paid", 0.78),
            self._make_evidence("11.00",  "change",    0.82),
        ]
        result = find_best_assignment(evidence)
        self.assertIsNotNone(result)
        self.assertEqual(result.total.normalized, "89.00")
        self.assertIsNotNone(result.cash_paid)
        self.assertEqual(result.cash_paid.normalized, "100.00")
        self.assertIsNotNone(result.change)
        self.assertEqual(result.change.normalized, "11.00")
        self.assertIn("law2_cash_change", result.satisfied_laws)

    def test_cash_not_selected_as_total_when_change_nonzero(self):
        """Cash paid (100.00) must not be assigned as total when change (11.00) is nonzero."""
        evidence = [
            self._make_evidence("80.00",  "unknown", 0.55),
            self._make_evidence("9.00",   "unknown", 0.58),
            self._make_evidence("89.00",  "unknown", 0.65),
            self._make_evidence("100.00", "unknown", 0.78),
            self._make_evidence("11.00",  "unknown", 0.82),
        ]
        result = find_best_assignment(evidence)
        self.assertIsNotNone(result)
        self.assertEqual(result.total.normalized, "89.00",
                         "89.00 must be selected as total, not cash_paid 100.00")

    def test_tax_zero_evidence_not_in_pool(self):
        """Engine returns None when no valid (subtotal, tax) pair exists for the total."""
        # Single total candidate with no subtotal/tax → cannot form a (S, X) pair.
        evidence = [
            self._make_evidence("100.00", "total", 0.65),
        ]
        result = find_best_assignment(evidence)
        self.assertIsNone(result,
                          "Engine must return None when no valid subtotal+tax pair exists")

    def test_label_noisy_arithmetic_finds_correct_roles(self):
        """Mislabeled amounts: arithmetic must override labels and find the correct roles."""
        # 169.80 is the true total but labeled as "unknown"; 200.00 is mislabeled "total"
        evidence = [
            self._make_evidence("160.17", "unknown", 0.60),
            self._make_evidence("9.61",   "unknown", 0.60),
            self._make_evidence("169.80", "unknown", 0.70),
            self._make_evidence("200.00", "total",   0.82),  # wrong label
            self._make_evidence("30.20",  "unknown", 0.88),
        ]
        result = find_best_assignment(evidence)
        self.assertIsNotNone(result)
        self.assertEqual(result.total.normalized, "169.80",
                         "Financial laws must override the mislabeled total=200.00")

    def test_zero_tax_assignment(self):
        """Zero-tax receipts (GST 6% = 0.00): subtotal + 0.00 = total must be found."""
        # Simulates: Total Excl. GST = 15.00, GST 6% = 0.00, Total Incl. GST = 15.00
        evidence = [
            self._make_evidence("15.00", "subtotal", 0.65),  # Total Excl. GST
            self._make_evidence("0.00",  "tax",      0.65),  # GST 6% RM 0.00
            self._make_evidence("15.00", "total",    0.72),  # Total Incl. GST
        ]
        result = find_best_assignment(evidence)
        self.assertIsNotNone(result, "Engine must find an assignment for zero-tax receipts")
        self.assertEqual(result.total.normalized,   "15.00")
        self.assertEqual(result.subtotal.normalized, "15.00")
        self.assertEqual(result.tax.normalized,      "0.00",
                         "Tax must be 0.00 when GST line clearly shows 0.00")
        self.assertIn("law1_subtotal_tax_total", result.satisfied_laws)


class IntegrationTaxZeroTests(SimpleTestCase):
    """Integration: full extraction pipeline must handle tax=0.00 receipts correctly."""

    def test_tax_zero_extracted_correctly(self):
        """Receipt where tax is explicitly 0.00 must produce tax_amount='0.00' with no error."""
        # "Total Amount" is placed >100px below "Tax RM 0.00" so that _nearby_line()
        # does not return it as the next-line context for the Tax label, which would
        # cause the score for "100.00 next_line_below" to beat "0.00 same_line_right".
        tokens = [
            OCRTokenData("ALPHA STORE",               [50, 40,  300, 70],  0.92),
            OCRTokenData("Invoice No: INV-ZERO-001",  [50, 120, 400, 150], 0.90),
            OCRTokenData("Date: 2026-03-15",          [50, 160, 320, 190], 0.91),
            OCRTokenData("Subtotal",                  [50, 580, 200, 610], 0.93),
            OCRTokenData("RM 100.00",                 [350, 580, 530, 610], 0.93),
            OCRTokenData("Tax",                       [50, 645, 130, 675], 0.92),
            OCRTokenData("RM 0.00",                   [350, 645, 490, 675], 0.91),
            OCRTokenData("Total Amount",              [50, 860, 240, 890], 0.94),
            OCRTokenData("RM 100.00",                 [350, 860, 530, 890], 0.94),
        ]
        layout = extract_layout_aware(tokens)

        self.assertEqual(layout.fields.get("tax_amount", ""), "0.00",
                         "Tax of 0.00 must be extracted, not ignored")
        self.assertEqual(layout.fields.get("total_amount", ""), "100.00")
        self.assertEqual(layout.fields.get("currency", ""), "RM")

        anomalies = check_invoice_anomalies(layout.fields)
        codes = {a.code for a in anomalies}
        self.assertNotIn("tax_greater_than_total", codes)
        self.assertNotIn("tax_equals_total", codes)


class IntegrationGSTReceiptTests(SimpleTestCase):
    """Integration tests for the real GST-zero-tax receipt pattern (CI-0170778 style).

    Covers the bugs found in production:
    - vendor: TERHINAL CI (OCR-noisy TERMINAL CI) selected over real business name
    - date: blank because DATE label shows "28/12/201/" (truncated year OCR artefact)
    - tax_amount: 15.00 instead of 0.00 because "gst" alias matched "Total Incl. GST" line
    """

    def test_gst_incl_excl_receipt(self):
        """Real receipt: Total Excl/Incl GST labels, GST 6% = 0.00, TERHINAL CI rejected."""
        tokens = [
            # Real business name at top
            OCRTokenData("PARKSON CORPORATION SDN BHD",  [50, 35,  450, 65],  0.90),
            # Operational terminal — OCR-noisy "TERMINAL CI" must NOT be selected as vendor
            OCRTokenData("TERHINAL CI",                  [50, 80,  200, 110], 0.86),
            # Date label with truncated/mangled year ("201/" = "2017" after OCR repair)
            OCRTokenData("DATE",                         [50, 140, 120, 170], 0.88),
            OCRTokenData("28/12/201/",                   [140, 140, 300, 170], 0.72),
            # Bill / invoice label
            OCRTokenData("BILL NO:",                     [50, 200, 160, 230], 0.90),
            OCRTokenData("CI-0170778",                   [175, 200, 320, 230], 0.89),
            # Currency present on the excl-GST line
            OCRTokenData("Total Excl. GST",              [50, 700, 270, 730], 0.93),
            OCRTokenData("RM 15.00",                     [350, 700, 470, 730], 0.93),
            # GST line — value is 0.00
            OCRTokenData("GST 6%",                       [50, 745, 165, 775], 0.91),
            OCRTokenData("RM 0.00",                      [350, 745, 470, 775], 0.91),
            # Total including GST
            OCRTokenData("Total Incl. GST",              [50, 790, 270, 820], 0.93),
            OCRTokenData("RM 15.00",                     [350, 790, 470, 820], 0.93),
            # Footer date stamp (fallback for date extraction)
            OCRTokenData("2017-12-28",                   [300, 920, 480, 950], 0.88),
        ]
        layout = extract_layout_aware(tokens)

        # Vendor: OCR-noisy TERMINAL identifier must be rejected; real business name wins
        vendor = layout.fields.get("vendor_name", "")
        self.assertNotIn("TERHINAL", vendor.upper(),
                         "OCR-noisy TERHINAL CI must be rejected as vendor")
        self.assertNotIn("TERMINAL", vendor.upper(),
                         "Operational TERMINAL label must not be selected as vendor")
        self.assertIn("PARKSON", vendor,
                      "Top business-name line must be selected as vendor")

        # Date: recovered from DATE label after "201/" → "2017" repair, or from footer
        date_val = layout.fields.get("date", "")
        self.assertNotEqual(date_val, "",
                            "Date must be extracted from DATE label or footer stamp")
        self.assertIn("2017", date_val,
                      "Extracted date must contain the year 2017")

        # Invoice number
        self.assertEqual(layout.fields.get("invoice_number", ""), "CI-0170778")

        # Currency
        self.assertEqual(layout.fields.get("currency", ""), "RM")

        # GST 6% = 0.00 — must NOT become 15.00
        self.assertEqual(layout.fields.get("tax_amount", ""), "0.00",
                         "GST 6% RM 0.00 must yield tax_amount=0.00, not 15.00")

        # Subtotal from "Total Excl. GST"
        self.assertEqual(layout.fields.get("subtotal", ""), "15.00",
                         "Total Excl. GST RM 15.00 must be the subtotal")

        # Total from "Total Incl. GST"
        self.assertEqual(layout.fields.get("total_amount", ""), "15.00",
                         "Total Incl. GST RM 15.00 must be total_amount")

        # Constraint engine must have injected the amounts (check at least one field)
        tax_method   = layout.evidence.get("tax_amount",   {}).get("method", "")
        total_method = layout.evidence.get("total_amount", {}).get("method", "")
        sub_method   = layout.evidence.get("subtotal",     {}).get("method", "")
        self.assertTrue(
            "financial_constraint_engine" in f"{tax_method} {total_method} {sub_method}",
            "Constraint engine must be active for amount fields on this receipt; "
            f"methods: tax={tax_method!r} total={total_method!r} sub={sub_method!r}",
        )

        # No critical financial anomalies
        anomalies = check_invoice_anomalies(layout.fields)
        codes = {a.code for a in anomalies}
        self.assertNotIn("tax_greater_than_total", codes)
        self.assertNotIn("tax_equals_total", codes)
