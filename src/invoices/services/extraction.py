from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from statistics import mean, median
from typing import Iterable


FIELD_NAMES = [
    "invoice_number",
    "date",
    "vendor_name",
    "tax_amount",
    "total_amount",
    "currency",
    "subtotal",
    "vat_rate",
    "amount_in_words",
]


@dataclass
class ExtractionOutput:
    fields: dict[str, str] = field(default_factory=dict)
    confidences: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, dict] = field(default_factory=dict)


@dataclass
class NormalizedToken:
    original_text: str
    text: str
    search_text: str
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float
    cy: float
    width: float
    height: float
    confidence: float
    page: int = 1

    @property
    def bbox(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class OCRLine:
    page: int
    tokens: list[NormalizedToken]
    text: str
    normalized_text: str
    search_text: str
    confidence: float
    bbox: list[float]

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def height(self) -> float:
        return max(self.y2 - self.y1, 1.0)


@dataclass
class ValueContext:
    text: str
    normalized_text: str
    source_text: str
    bbox: list[float]
    confidence: float
    distance: float
    direction: str
    label: str
    label_line: OCRLine
    value_line: OCRLine


@dataclass
class FieldCandidate:
    field: str
    value: str
    score: float
    source_text: str
    source_bbox: list[float]
    method: str
    explanation: str
    amount_role: str = ""

    def confidence(self) -> float:
        return round(max(0.0, min(self.score, 0.99)), 4)

    def evidence(self) -> dict:
        ev = {
            "value": self.value,
            "confidence": self.confidence(),
            "source_text": self.source_text,
            "source_bbox": [round(float(item), 2) for item in self.source_bbox],
            "method": self.method,
            "explanation": self.explanation,
        }
        if self.amount_role:
            ev["amount_role"] = self.amount_role
        return ev


LABEL_ALIASES: dict[str, list[tuple[str, float]]] = {
    "invoice_number": [
        # Specific receipt anchors — outweigh generic substrings
        ("slip no", 0.50),
        ("trans no", 0.48),
        ("trans", 0.46),
        ("bil no", 0.46),        # Malay: bill number
        ("resit", 0.44),         # Malay: receipt
        # Standard aliases
        ("invoice no", 0.42),
        ("invoice number", 0.42),
        ("document no", 0.46),
        ("doc no", 0.44),
        ("receipt no", 0.44),
        ("receipt number", 0.42),
        ("bill no", 0.42),
        ("cash bill no", 0.44),
        ("transaction no", 0.42),
        ("transaction number", 0.42),
        ("ref no", 0.40),
        ("reference no", 0.40),
        ("发票号码", 0.54),
        ("发票号", 0.50),
        ("票据号码", 0.48),
        ("票号", 0.44),
    ],
    "date": [
        ("invoice date", 0.44),
        ("receipt date", 0.42),
        ("date", 0.40),
        ("bill date", 0.40),
        ("开票日期", 0.48),
        ("日期", 0.44),
    ],
    "total_amount": [
        ("rounded total", 0.50),
        ("rounding total", 0.48),
        ("grand total", 0.48),
        ("amount due", 0.48),
        ("net total", 0.45),
        ("total sales", 0.48),
        ("total amount", 0.44),
        ("total incl", 0.46),   # "Total Incl. GST/SST" — total including tax
        ("incl gst", 0.44),
        ("incl sst", 0.44),
        ("total", 0.35),
        ("价税合计", 0.52),
        ("合计金额", 0.48),
        ("小写金额", 0.48),
        ("应付金额", 0.48),
        ("应收", 0.58),
        ("应付", 0.56),
        ("实付", 0.54),
        ("实收", 0.44),
        ("收款金额", 0.52),
        ("金额", 0.42),
    ],
    "tax_amount": [
        ("tax amount", 0.44),
        ("service tax", 0.42),
        ("sales tax", 0.42),
        ("tax", 0.36),
        ("vat", 0.36),
        ("gst", 0.34),
        ("税额", 0.46),
        ("增值税", 0.42),
    ],
    "subtotal": [
        ("amount", 0.28),
        ("subtotal", 0.44),
        ("sub total", 0.44),
        ("before tax", 0.36),
        ("total excl", 0.46),   # "Total Excl. GST/SST" — subtotal excluding tax
        ("excl gst", 0.44),
        ("excl sst", 0.44),
        ("before gst", 0.44),
        ("before sst", 0.44),
        ("不含税金额", 0.48),
        ("未税金额", 0.44),
        ("订单原价", 0.50),
        ("商品合计", 0.48),
        ("原价合计", 0.46),
    ],
    "vat_rate": [
        ("vat rate", 0.42),
        ("tax rate", 0.42),
        ("gst rate", 0.40),
        ("税率", 0.44),
    ],
    "amount_in_words": [
        ("amount in words", 0.46),
        ("total in words", 0.42),
        ("大写金额", 0.46),
        ("价税合计大写", 0.48),
    ],
}

# Compact labels that disqualify a line for a field.
NEGATIVE_ANCHORS: dict[str, tuple[str, ...]] = {
    "total_amount": ("SAVING", "SAVINGS", "DISCOUNT", "RABAT", "CHANGE",
                     "KASTAM", "VOUCHER", "POINTS"),
    "subtotal":     ("SAVING", "SAVINGS", "DISCOUNT", "CHANGE"),
    "tax_amount":   ("REGNO", "REGISTRATION", "GSTIN", "GSTID", "SSTID",
                     "IDNO", "JALAN", "STREET", "ROAD", "TEL", "PHONE"),
    "date":         ("EXPIRY", "EXP", "EXPIRES", "VALIDTHRU", "VALIDUNTIL",
                     "VALID", "THRU", "CARD"),
    "invoice_number": ("CONO", "COMPANYNO", "GSTNO", "GSTREGNO",
                       "SSTNO", "REGNO", "REGISTRATION", "GSTIN", "TAXNO",
                       "TIN", "ROC"),
}


def _has_negative_anchor(text: str, field: str) -> bool:
    """Match field exclusions without substring false positives."""
    anchors = NEGATIVE_ANCHORS.get(field, ())
    if not anchors:
        return False

    raw_tokens = _searchable(text).split()
    if not raw_tokens:
        return False
    toks = [_compact(w) for w in raw_tokens]

    # N-grams support compact multi-word labels such as GSTREGNO.
    ngrams: set[str] = set()
    n_max = min(4, len(toks))
    for n in range(1, n_max + 1):
        for i in range(len(toks) - n + 1):
            ngrams.add("".join(toks[i : i + n]))

    return any(anchor in ngrams for anchor in anchors)


NEGATIVE_AMOUNT_KEYWORDS = (
    "CASH",
    "CHANGE",
    "GST SUMMARY",
    "TAX",
    "VAT",
    "DISCOUNT",
    "QTY",
    "QUANTITY",
    "UNIT PRICE",
    "PRICE/UNIT",
    "BALANCE",
    "TENDER",
)

# Amounts on payment lines cannot represent tax or invoice total.
PAYMENT_CONTEXT_LABELS = (
    "CASH",
    "CASII",    # OCR noise for CASH
    "CASI",
    "CHANGE",
    "TENDERED",
    "PAID",
    "PAYMENT",
    "AMOUNTPAID",
    "CARD",
    "CREDIT",
    "DEBIT",
    "BALANCE",
    "REFUND",
    "TENDER",
    "CWAHGE",   # common OCR noise for CHANGE
    "CWANGE",
    "CHAHGE",
    "CHAMGE",
    "CWANGE",
)

LINE_ITEM_CONTEXT_LABELS = (
    "DESCRIPTION",
    "ITEM",
    "ITEMCOUNT",
    "QTY",
    "QUANTITY",
    "HRSIQTY",
    "UNIT",
    "UNITPRICE",
    "PRICEUNIT",
    "UNITCOST",
    "LINEITEM",
    "LINETOTAL",
    "PRODUCT",
)

ADJUSTMENT_CONTEXT_LABELS = (
    "ROUNDINGADJ",
    "ROUNDINGADJUSTMENT",
    "ROUNDING",
    "ROURDING",
    "ROUNDIRG",
    "ROUNDIHG",
    "ROUN0ING",
    "SAVING",
    "SAVINGS",
    "DISCOUNT",
    "REBATE",
    "RABAT",
    "VOUCHER",
    "POINTS",
)

FINANCIAL_SUMMARY_CONTEXT_LABELS = (
    "ROUNDEDTOTAL",
    "ROUNDINGTOTAL",
    "GRANDTOTAL",
    "AMOUNTDUE",
    "NETTOTAL",
    "INVOICETOTAL",
    "TOTALAMOUNT",
    "TOTALSALES",
    "TOTALINCL",
    "TOTALEXCL",
    "SUBTOTAL",
    "SUBTOTALAMOUNT",
    "TAXAMOUNT",
    "SERVICETAX",
    "SALESTAX",
    "GSTSUMMARY",
    "SSTSUMMARY",
)
VENDOR_EXCLUDE_KEYWORDS = (
    "DOCUMENT NO",
    "DOC NO",
    "RECEIPT NO",
    "INVOICE NO",
    "INVOICE DATE",
    "BILL NO",
    "PO NUMBER",
    "PO NO",
    "PURCHASE ORDER",
    "DUE DATE",
    "DATE",
    "TIME",
    "CASHIER",
    "TERMINAL",    # hardware terminal / POS counter identifier
    "TABLE",       # restaurant table number
    "TOKEN",       # queue/service token
    "OPERATOR",    # store operator label
    "MEMBER",
    "CUSTOMER",
    "ADDRESS",
    "BUYER",
    "BILL TO",
    "SHIP TO",
    "GST",
    "SST",
    "REGISTRATION",
    "TAX INVOICE",
    "OFFICIAL RECEIPT",
    "TEL",
    "PHONE",
    "FAX",
    "EMAIL",
    "WWW",
    "HTTP",
    "BANK",
    "BRANCH",
    "ACCOUNT",
    "SWIFT",
    "发票代码",
    "发票号码",
    "开票日期",
    "日期",
    "金额",
    "税额",
    "税率",
    "电话",
    "车号",
    "证号",
    "卡号",
    "密码",
    "国家税务局",
    "税务局",
    "机打发票",
    "手写无效",
    "发票联",
    "发票专用章",
    "单据号",
    "收款时间",
    "商品信息",
    "支付信息",
    "门店地址",
    "地址",
    "收银员",
    "会员",
    "服务热线",
)
VENDOR_ADDRESS_KEYWORDS = (
    # Street and road terms.
    "STREET",
    "ROAD",
    "AVENUE",
    "BOULEVARD",
    "EXPRESSWAY",
    "HIGHWAY",
    "FREEWAY",
    "CRESCENT",
    "TERRACE",
    # Building and unit terms.
    "FLOOR",
    "LEVEL",
    "SUITE",
    "APARTMENT",
    "LOT ",
    "NO.",
    # Postal terms.
    "POSTCODE",
    "POSTAL CODE",
    "ZIP CODE",
    "P.O. BOX",
    "PO BOX",
    # Malaysian address terms.
    "JALAN",       # road
    "LORONG",      # lane/alley
    "PERSIARAN",   # boulevard
    "LEBUH",       # street (older Malay)
    "POSKOD",      # postcode
    "TAMAN",       # housing estate
    "SETIA",
    "BANDAR",
    "DAMANSARA",
)
BUSINESS_SUFFIX_RE = re.compile(
    r"\b("
    # Malaysian
    r"SDN\s+BHD|S\.?D\.?N\.?\s+B\.?H\.?D\.?|S\s*/\s*[B8]"
    # UK / Commonwealth
    r"|LTD|LIMITED|PLC|P\.L\.C\."
    r"|PTY\.?\s*LTD"          # Australian
    # USA
    r"|INC\.?|INCORPORATED|CORP\.?|CORPORATION|LLC|L\.L\.C\.|LP|LLP"
    # Europe
    r"|GMBH|G\.M\.B\.H\.|A\.?G\."    # German
    r"|S\.A\.|S\.R\.L\.|S\.L\."       # French/Spanish/Italian
    # Generic entity words
    r"|COMPANY|CO\."
    # Business-type words — protect "Broadway Cafe", "City Lane Clinic", etc.
    r"|STORE|SHOP|ENTERPRISE|TRADING|MARKET|RESTAURANT|SUPERMARKET|MART"
    r"|HARDWARE|ELECTRIC|ELECTRICAL|STATIONERY|BOOK|LOGISTICS"
    r"|CAFE|COFFEE|BAKERY|PHARMACY|CLINIC|HOSPITAL|HOTEL|RESORT"
    r"|SALON|SPA|GYM|GALLERY|STUDIO|CENTRE|CENTER"
    r"|SERVICES|SOLUTIONS|SYSTEMS|TECHNOLOGIES|INDUSTRIES"
    # Malay business-type words (RESTAURANT is above but RESTORAN is missing)
    r"|RESTORAN|KEDAI|SYARIKAT|PERNIAGAAN|WARUNG|GERAI"
    r")\b",
    re.I,
)
CHINESE_BUSINESS_KEYWORDS = (
    "公司",
    "有限",
    "集团",
    "商店",
    "商行",
    "超市",
    "市场",
    "餐饮",
    "酒店",
    "医院",
    "学校",
    "出租车",
    "运输",
    "服务",
    "科技",
    "中心",
    "门店",
    "分店",
)
CHINESE_VENDOR_LABELS = (
    "销售方名称",
    "销方名称",
    "购买方名称",
    "开票单位",
    "收款单位",
    "单位名称",
    "公司名称",
    "纳税人名称",
)
CHINESE_VENDOR_EXCLUDE_FRAGMENTS = (
    "发票代码",
    "发票号码",
    "国家税务局",
    "税务局",
    "通用机打发票",
    "机打发票",
    "手写无效",
    "发票联",
    "发票专用章",
    "单位代码",
    "电话",
    "车号",
    "证号",
    "日期",
    "上车",
    "下车",
    "单价",
    "里程",
    "等候",
    "状态",
    "金额",
    "卡号",
    "密码",
    "单据号",
    "收款时间",
    "商品信息",
    "支付信息",
    "门店地址",
    "地址",
    "收银员",
    "会员",
    "服务热线",
)

_INLINE_CJK_STOP_LABELS: tuple[str, ...] = (
    "订单优惠",
    "实收",
    "找零",
    "消费券抵扣金额",
    "消费券",
    "支付宝",
    "微信",
    "现金",
    "会员手机号",
    "会员积分",
    "本次积分",
    "门店地址",
    "商品信息",
    "支付信息",
    "收款时间",
    "订单原价",
    "件数",
)

# Reject obvious seal/logo noise while retaining faded vendor headers.
_VENDOR_MIN_CONFIDENCE = 0.30
CURRENCY_PATTERNS = [
    (re.compile(r"\bRM\b", re.I), "RM"),
    (re.compile(r"\bMYR\b", re.I), "MYR"),
    # Common OCR variants of RM.
    (re.compile(r"\b(?:RH|RK|RNG|RHT)\b"), "RM"),
    # Match explicit codes before symbols.
    (re.compile(r"\bEUR\b", re.I), "EUR"),
    (re.compile(r"€"), "EUR"),
    (re.compile(r"\bGBP\b", re.I), "GBP"),
    (re.compile(r"£"), "GBP"),
    (re.compile(r"\bRMB\b", re.I), "RMB"),
    (re.compile(r"\bCNY\b", re.I), "CNY"),
    (re.compile(r"¥"), "CNY"),
    (re.compile(r"元"), "CNY"),
    (re.compile(r"\bUSD\b", re.I), "USD"),
    (re.compile(r"\$"), "USD"),
    (re.compile(r"\bNGN\b", re.I), "NGN"),
    (re.compile(r"₦"), "NGN"),
]

_CURRENCY_CONFIDENCE_FLOOR = 0.45
_RM_EVIDENCE_RE = re.compile(r"\b(?:RM|MYR|RH|RK|RNG|RHT)\b", re.I)
_INVOICE_FORMAT_PRIOR_EXCLUDE_FRAGMENTS = frozenset({
    "EMAIL",
    "TEL",
    "TELEPHONE",
    "PHONE",
    "MOBILE",
    "WWW",
    "HTTP",
    "BANK",
    "BRANCH",
    "ACCOUNT",
    "SWIFT",
    "PONUMBER",
    "PONO",
    "PURCHASEORDER",
})
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.I)
DECIMAL_AMOUNT_RE = re.compile(
    r"(?<![A-Z0-9])(?:RM|MYR|RMB|CNY|USD|NGN)?\s*[$¥₦]?\s*([0-9]{1,3}(?:[,.][0-9]{3})+[,.][0-9]{2}|[0-9]+[,.][0-9]{2})(?![A-Z0-9])",
    re.I,
)
# Used only in financial contexts to repair values such as "9 61".
_SPACED_DECIMAL_AMOUNT_RE = re.compile(
    r"(?<![0-9,.])([0-9]{1,4})\s{1,3}([0-9]{2})(?![0-9,.])",
)
_BROKEN_DECIMAL_AMOUNT_RE = re.compile(
    r"(?<![0-9,.])([0-9]{1,6})\s*[,.]\s+([0-9]{2})(?![0-9,.])",
)
INTEGER_AMOUNT_RE = re.compile(
    r"(?<![A-Z0-9])(?:RM|MYR|RMB|CNY|USD|NGN)?\s*[$¥₦]?\s*([0-9]{1,7})(?![A-Z0-9%])",
    re.I,
)
VAT_RATE_RE = re.compile(r"(?<![A-Z0-9])([0-9]{1,2}(?:[,.][0-9]+)?\s*%)(?![A-Z0-9])", re.I)
INVOICE_ID_RE = re.compile(r"\b(?=[A-Z0-9][A-Z0-9\-/]{4,}\b)(?=[A-Z0-9\-/]*[A-Z])(?=[A-Z0-9\-/]*[0-9])[A-Z0-9][A-Z0-9\-/]{3,}[A-Z0-9]\b", re.I)
PURE_DIGIT_ID_NEAR_LABEL_RE = re.compile(r"(?<!\d)(\d{6,12})(?!\d)")


def _token_text(token) -> str:
    if hasattr(token, "text"):
        return str(token.text or "")
    return str(token.get("text", "") or "")


def _token_bbox(token) -> list[float]:
    if hasattr(token, "bbox"):
        bbox = token.bbox
    elif isinstance(token, dict) and "bbox" in token:
        bbox = token.get("bbox")
    elif all(hasattr(token, attr) for attr in ("x1", "y1", "x2", "y2")):
        bbox = [token.x1, token.y1, token.x2, token.y2]
    else:
        bbox = [0, 0, 0, 0]
    if len(bbox) < 4:
        bbox = [0, 0, 0, 0]
    return [float(value or 0.0) for value in bbox[:4]]


def _token_confidence(token) -> float:
    if hasattr(token, "confidence"):
        return float(token.confidence or 0.0)
    return float(token.get("confidence", 0.0) or 0.0)


def _token_page(token) -> int:
    if hasattr(token, "page"):
        return int(token.page or 1)
    return int(token.get("page", 1) or 1)


def _normalize_ocr_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    replacements = {
        "：": ":",
        "（": "(",
        "）": ")",
        "–": "-",
        "—": "-",
        "¬": "-",
        "Â¥": "¥",
        "â‚¬": "€",
        "Â£": "£",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _searchable(text: str) -> str:
    value = _normalize_ocr_text(text).upper()
    value = value.replace("_", " ")
    value = re.sub(r"[\s:;,#|]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text or ""))


def _compact(text: str) -> str:
    return re.sub(r"[^A-Z0-9%\u3400-\u9fff]+", "", _searchable(text))


def _canonical_compact(text: str) -> str:
    value = _compact(text)
    replacements = [
        ("INWWICE", "INVOICE"),
        ("INV0ICE", "INVOICE"),
        ("INVO1CE", "INVOICE"),
        ("INV01CE", "INVOICE"),
        ("INWICE", "INVOICE"),
        ("8Q", "NO"),
        ("BQ", "NO"),
        ("NQ", "NO"),
        ("N0", "NO"),
        # "INVOICE NI" is a common OCR misread of "INVOICE NO"
        ("INVOICENI", "INVOICENO"),
        ("INVOICEN0", "INVOICENO"),
        ("TOTAI", "TOTAL"),
        ("T0TAL", "TOTAL"),
        ("TOTE", "TOTAL"),
        ("T0TE", "TOTAL"),
        ("ROURDING", "ROUNDING"),
        ("ROUNDIRG", "ROUNDING"),
        ("ROUNDIHG", "ROUNDING"),
        ("ROUN0ING", "ROUNDING"),
        ("CASH", "CASH"),
        ("CASII", "CASH"),
        ("CASI", "CASH"),
        ("CWAHGE", "CHANGE"),
        ("CWANGE", "CHANGE"),
        ("CHAHGE", "CHANGE"),
        ("CHAMGE", "CHANGE"),
        # OCR noise variants of operational labels
        ("TERHINAL", "TERMINAL"),
        ("TERMIHAL", "TERMINAL"),
        ("TERMIMAL", "TERMINAL"),
    ]
    for old, new in replacements:
        value = value.replace(old, new)
    return value


def _fuzzy_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _valid_amount_value(value: str, field: str) -> bool:  # noqa: ARG001 (field reserved for future per-field rules)
    """Return True only when `value` is a properly-formatted 2-decimal money amount.

    Rejects bare integers (e.g. '8') and negative amounts, which are never
    valid for tax_amount, subtotal, or total_amount.
    """
    v = value.strip()
    if v.startswith("-"):
        return False
    return bool(re.fullmatch(
        r"\d{1,3}(?:[,.]\d{3})*[.,]\d{2}|\d+[.,]\d{2}",
        v,
    ))


def _valid_invoice_value(value: str) -> bool:
    """Return True when `value` is plausibly an invoice/receipt number.

    Rejects dates, emails/URLs, and company-registration-number shapes
    like '(13825-W)'.
    """
    v = value.strip()
    if looks_like_date(v):
        return False
    if re.search(r"@|HTTP|WWW|GSTIN", v, re.I):
        return False
    # Company-reg shapes: (13825-W), 13825-W, (1234-A)
    if re.fullmatch(r"\(?\s*\d{3,6}\s*-\s*[A-Z]\s*\)?", v, re.I):
        return False
    return bool(re.search(r"[A-Za-z0-9]", v))


def _compact_contains_fuzzy(text: str, target: str, threshold: float = 0.78) -> bool:
    haystack = _canonical_compact(text)
    needle = _canonical_compact(target)
    if not needle:
        return False
    if needle in haystack:
        return True
    if len(haystack) < max(3, len(needle) - 2):
        return _fuzzy_ratio(haystack, needle) >= threshold
    window_min = max(3, len(needle) - 2)
    window_max = min(len(haystack), len(needle) + 3)
    for size in range(window_min, window_max + 1):
        for start in range(0, len(haystack) - size + 1):
            if _fuzzy_ratio(haystack[start : start + size], needle) >= threshold:
                return True
    return False


def _normalize_tokens(tokens: Iterable) -> list[NormalizedToken]:
    normalized_tokens: list[NormalizedToken] = []
    for token in tokens:
        original = _token_text(token).strip()
        if not original:
            continue
        x1, y1, x2, y2 = _token_bbox(token)
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        text = _normalize_ocr_text(original)
        normalized_tokens.append(
            NormalizedToken(
                original_text=original,
                text=text,
                search_text=_searchable(text),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                cx=(x1 + x2) / 2,
                cy=(y1 + y2) / 2,
                width=max(x2 - x1, 1.0),
                height=max(y2 - y1, 1.0),
                confidence=_token_confidence(token),
                page=_token_page(token),
            )
        )
    return normalized_tokens


def _make_line(tokens: list[NormalizedToken]) -> OCRLine:
    sorted_tokens = sorted(tokens, key=lambda token: token.x1)
    x1 = min(token.x1 for token in sorted_tokens)
    y1 = min(token.y1 for token in sorted_tokens)
    x2 = max(token.x2 for token in sorted_tokens)
    y2 = max(token.y2 for token in sorted_tokens)
    text = " ".join(token.original_text for token in sorted_tokens).strip()
    normalized_text = " ".join(token.text for token in sorted_tokens).strip()
    confidences = [token.confidence for token in sorted_tokens if token.text]
    return OCRLine(
        page=sorted_tokens[0].page,
        tokens=sorted_tokens,
        text=text,
        normalized_text=normalized_text,
        search_text=_searchable(normalized_text),
        confidence=round(mean(confidences), 4) if confidences else 0.0,
        bbox=[x1, y1, x2, y2],
    )


def _vertical_overlap_ratio(token: NormalizedToken, line: OCRLine) -> float:
    overlap = min(token.y2, line.y2) - max(token.y1, line.y1)
    if overlap <= 0:
        return 0.0
    return overlap / max(min(token.height, line.height), 1.0)


def _line_groups(tokens: Iterable) -> list[OCRLine]:
    normalized_tokens = sorted(_normalize_tokens(tokens), key=lambda token: (token.page, token.cy, token.x1))
    if not normalized_tokens:
        return []

    typical_height = median(token.height for token in normalized_tokens)
    y_threshold = max(typical_height * 0.65, 12.0)
    grouped: list[list[NormalizedToken]] = []
    line_objects: list[OCRLine] = []

    for token in normalized_tokens:
        placed = False
        for idx in range(len(grouped) - 1, -1, -1):
            line = line_objects[idx]
            if line.page != token.page:
                continue
            close_centers = abs(token.cy - line.cy) <= y_threshold
            overlapping = _vertical_overlap_ratio(token, line) >= 0.35
            if close_centers or overlapping:
                grouped[idx].append(token)
                line_objects[idx] = _make_line(grouped[idx])
                placed = True
                break
            if token.cy - line.cy > y_threshold * 2.5:
                break
        if not placed:
            grouped.append([token])
            line_objects.append(_make_line([token]))

    return sorted((_make_line(line) for line in grouped), key=lambda line: (line.page, line.cy, line.x1))


def _bbox_union(items: list[NormalizedToken] | list[OCRLine]) -> list[float]:
    return [
        min(item.x1 for item in items),
        min(item.y1 for item in items),
        max(item.x2 for item in items),
        max(item.y2 for item in items),
    ]


def _document_bounds(lines: list[OCRLine]) -> tuple[float, float, float]:
    if not lines:
        return 0.0, 1.0, 1.0
    min_y = min(line.y1 for line in lines)
    max_y = max(line.y2 for line in lines)
    height = max(max_y - min_y, 1.0)
    return min_y, max_y, height


def _y_ratio(bbox: list[float], lines: list[OCRLine]) -> float:
    min_y, _, height = _document_bounds(lines)
    return ((bbox[1] + bbox[3]) / 2 - min_y) / height


def _line_has_alias(line: OCRLine, alias: str) -> bool:
    return _compact_contains_fuzzy(line.search_text, alias)


def _find_label_span(line: OCRLine, alias: str) -> tuple[int, int] | None:
    alias_compact = _canonical_compact(alias)
    for start in range(len(line.tokens)):
        merged = ""
        for end in range(start, min(len(line.tokens), start + 7)):
            merged += _canonical_compact(line.tokens[end].search_text)
            if alias_compact in merged or _fuzzy_ratio(merged, alias_compact) >= 0.78:
                return start, end
    return None


def _text_after_alias(text: str, alias: str) -> str:
    normalized = _normalize_ocr_text(text)
    if _contains_cjk(alias):
        index = normalized.find(alias)
        if index >= 0:
            after = normalized[index + len(alias):].strip(" :：#-")
            stop_positions = [
                pos
                for stop_label in _INLINE_CJK_STOP_LABELS
                if stop_label != alias and (pos := after.find(stop_label)) > 0
            ]
            if stop_positions:
                after = after[: min(stop_positions)]
            return after.strip(" :：#-")
    words = [re.escape(word) for word in re.split(r"\s+", alias.strip()) if word]
    if not words:
        return ""
    pattern = r"\b" + r"\s*\.?\s*".join(words) + r"\b\s*[:#\-]?\s*(.*)$"
    match = re.search(pattern, normalized, re.I)
    return match.group(1).strip() if match else ""


def _nearby_line(lines: list[OCRLine], index: int) -> OCRLine | None:
    if index + 1 >= len(lines):
        return None
    current = lines[index]
    candidate = lines[index + 1]
    if candidate.page != current.page:
        return None
    vertical_gap = max(candidate.y1 - current.y2, 0.0)
    allowed_gap = max(current.height, candidate.height) * 2.5 + 18.0
    if vertical_gap <= allowed_gap:
        return candidate
    return None


def _token_is_label(tok: NormalizedToken) -> bool:
    """Return True when a token's text matches any known field label alias."""
    comp = _canonical_compact(tok.search_text)
    if not comp:
        return False
    for aliases in LABEL_ALIASES.values():
        for alias_text, _ in aliases:
            alias_comp = _canonical_compact(alias_text)
            if alias_comp and (alias_comp in comp or _fuzzy_ratio(comp, alias_comp) >= 0.82):
                return True
    return False


def _value_tokens_after_label(tokens_after_label: list[NormalizedToken]) -> list[NormalizedToken]:
    """Return value tokens up to the next field label."""
    out: list[NormalizedToken] = []
    for tok in tokens_after_label:
        if out and _token_is_label(tok):
            break
        out.append(tok)
    return out


def _contexts_for_label(lines: list[OCRLine], index: int, alias: str) -> list[ValueContext]:
    line = lines[index]
    span = _find_label_span(line, alias)
    contexts: list[ValueContext] = []
    label_bbox = line.bbox

    if span:
        label_tokens = line.tokens[span[0] : span[1] + 1]
        label_bbox = _bbox_union(label_tokens)
        right_tokens = _value_tokens_after_label(line.tokens[span[1] + 1 :])

        # Stop before a distant value in a multi-column line.
        if right_tokens and len(line.tokens) > 1:
            widths = [t.width for t in line.tokens if t.width > 0]
            med_width = median(widths) if widths else 1.0
            gap_threshold = med_width * 3.0
            trimmed: list[NormalizedToken] = []
            prev_x2 = label_tokens[-1].x2
            for tok in right_tokens:
                if trimmed and tok.x1 - prev_x2 > gap_threshold:
                    break
                trimmed.append(tok)
                prev_x2 = tok.x2
            right_tokens = trimmed

        if right_tokens:
            right_text = " ".join(token.original_text for token in right_tokens).strip()
            right_bbox = _bbox_union(right_tokens)
            contexts.append(
                ValueContext(
                    text=right_text,
                    normalized_text=_normalize_ocr_text(right_text),
                    source_text=f"{alias}: {right_text}",
                    bbox=right_bbox,
                    confidence=round(mean(token.confidence for token in right_tokens), 4),
                    distance=max(right_bbox[0] - label_bbox[2], 0.0),
                    direction="same_line_right",
                    label=alias,
                    label_line=line,
                    value_line=line,
                )
            )

    after_text = _text_after_alias(line.normalized_text, alias)
    if after_text and not any(context.text == after_text for context in contexts):
        contexts.append(
            ValueContext(
                text=after_text,
                normalized_text=_normalize_ocr_text(after_text),
                source_text=line.text,
                bbox=line.bbox,
                confidence=line.confidence,
                distance=0.0,
                direction="same_line_after_label",
                label=alias,
                label_line=line,
                value_line=line,
            )
        )

    if not contexts:
        contexts.append(
            ValueContext(
                text=line.text,
                normalized_text=line.normalized_text,
                source_text=line.text,
                bbox=line.bbox,
                confidence=line.confidence,
                distance=0.0,
                direction="same_line_full",
                label=alias,
                label_line=line,
                value_line=line,
            )
        )

    below = _nearby_line(lines, index)
    if below and not _below_starts_with_stop_label(below.text):
        distance = max(below.y1 - line.y2, 0.0)
        contexts.append(
            ValueContext(
                text=below.text,
                normalized_text=below.normalized_text,
                source_text=f"{line.text} -> {below.text}",
                bbox=below.bbox,
                confidence=below.confidence,
                distance=distance,
                direction="next_line_below",
                label=alias,
                label_line=line,
                value_line=below,
            )
        )
    return contexts


def _normalize_amount(value: str) -> str:
    raw = re.sub(r"[^0-9,.\-]", "", str(value or "")).strip()
    raw = raw.strip("-")
    if not raw:
        return ""

    if "," in raw and "." in raw:
        decimal_sep = "," if raw.rfind(",") > raw.rfind(".") else "."
        thousands_sep = "." if decimal_sep == "," else ","
        normalized = raw.replace(thousands_sep, "").replace(decimal_sep, ".")
    elif "," in raw:
        pieces = raw.split(",")
        if len(pieces[-1]) in {1, 2}:
            normalized = "".join(pieces[:-1]) + "." + pieces[-1]
        else:
            normalized = "".join(pieces)
    elif "." in raw:
        pieces = raw.split(".")
        if len(pieces) > 2:
            normalized = "".join(pieces[:-1]) + "." + pieces[-1] if len(pieces[-1]) in {1, 2} else "".join(pieces)
        elif len(pieces[-1]) == 3 and len(pieces[0]) <= 3:
            normalized = "".join(pieces)
        else:
            normalized = raw
    else:
        normalized = raw

    if not re.fullmatch(r"\d+(?:\.\d+)?", normalized):
        return ""
    if "." in normalized:
        whole, decimal = normalized.split(".", 1)
        decimal = (decimal + "00")[:2]
        return f"{int(whole)}.{decimal}"
    return str(int(normalized))


def _clean_amount(value: str) -> str:
    amounts = _extract_amounts(value, allow_integer=True)
    return amounts[-1][0] if amounts else ""


def _number_is_percent(text: str, end_index: int) -> bool:
    return bool(re.match(r"\s*%", str(text or "")[end_index:]))


def _extract_amounts(text: str, allow_integer: bool = False) -> list[tuple[str, str]]:
    amounts: list[tuple[str, str]] = []
    for match in DECIMAL_AMOUNT_RE.finditer(text):
        if _number_is_percent(text, match.end(1)):
            continue
        normalized = _normalize_amount(match.group(1))
        if normalized:
            amounts.append((normalized, match.group(0).strip()))

    if allow_integer:
        decimal_spans = [match.span() for match in DECIMAL_AMOUNT_RE.finditer(text)]
        for match in INTEGER_AMOUNT_RE.finditer(text):
            if any(start <= match.start() < end for start, end in decimal_spans):
                continue
            if _number_is_percent(text, match.end(1)):
                continue
            if re.search(r"[/.-]\s*$", text[: match.start()]) or re.search(r"^\s*[/.-]", text[match.end() :]):
                continue
            normalized = _normalize_amount(match.group(1))
            if normalized:
                amounts.append((normalized, match.group(0).strip()))
    return amounts


def _extract_amounts_with_repair(text: str, allow_integer: bool = False) -> list[tuple[str, str]]:
    """Like _extract_amounts but also handles OCR-spaced decimals such as '9 61' → '9.61'."""
    amounts = _extract_amounts(text, allow_integer=allow_integer)
    existing_normalized = {a[0] for a in amounts}
    decimal_spans = [m.span() for m in DECIMAL_AMOUNT_RE.finditer(text)]
    for m in _SPACED_DECIMAL_AMOUNT_RE.finditer(text):
        # Skip positions already covered by a standard decimal match
        if any(s <= m.start() < e or s < m.end() <= e for s, e in decimal_spans):
            continue
        normalized = _normalize_amount(f"{m.group(1)}.{m.group(2)}")
        if normalized and normalized not in existing_normalized:
            amounts.append((normalized, m.group(0)))
            existing_normalized.add(normalized)
    for m in _BROKEN_DECIMAL_AMOUNT_RE.finditer(text):
        if any(s <= m.start() < e or s < m.end() <= e for s, e in decimal_spans):
            continue
        normalized = _normalize_amount(f"{m.group(1)}.{m.group(2)}")
        if normalized and normalized not in existing_normalized:
            amounts.append((normalized, m.group(0)))
            existing_normalized.add(normalized)
    return amounts


def _detect_currency(text: str) -> str:
    for pattern, value in CURRENCY_PATTERNS:
        if pattern.search(text or ""):
            return value
    return ""


def _detect_all_currencies(text: str) -> list[str]:
    values = []
    for pattern, value in CURRENCY_PATTERNS:
        if pattern.search(text or "") and value not in values:
            values.append(value)
    return values


_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1
)}


def _month_from_name(s: str) -> int | None:
    return _MONTHS.get((s or "").strip().lower()[:3])


def _correct_date_ocr(text: str) -> str:
    value = _normalize_ocr_text(text)
    value = re.sub(r"(?<![A-Za-z])[Zz](?=\d[-/])", "2", value)
    value = re.sub(r"(?<![A-Za-z])[Zz](?=\d)", "2", value)
    value = re.sub(r"(?<=\d)[Pp](?=\d{3}\b)", "/2", value)
    value = re.sub(r"(?<=\d)[Oo](?=\d)", "0", value)
    value = re.sub(r"(?<=\d)[Il|](?=\d)", "1", value)
    # Truncated year: "28/12/201/" — OCR confuses trailing digit (often "7") with "/".
    # Replace the trailing "/" when it follows exactly 3 year digits in a date-like fragment.
    value = re.sub(r"(\d{1,2}[-/]\d{1,2}[-/]\d{3})/(?!\d)", r"\g<1>7", value)
    return value


def _four_digit_year(year: str) -> int:
    value = int(year)
    if len(year) == 2:
        return 2000 + value if value < 50 else 1900 + value
    return value


def _valid_date_parts(year: int, month: int, day: int) -> bool:
    return 1900 <= year <= 2099 and 1 <= month <= 12 and 1 <= day <= 31


def _repair_header_date_parts(day: int, month: int, year: int) -> tuple[int, int, int] | None:
    if _valid_date_parts(year, month, day):
        return day, month, year
    if month == 14 and _valid_date_parts(year, 11, day):
        return day, 11, year
    return None


_DATE_DIGIT_REPAIRS = {
    "6": ("6", "0"),
    "5": ("5", "0", "9"),
    "3": ("3", "8", "9"),
    "8": ("8", "0", "6"),
    "9": ("9", "0"),
}


def _date_digit_variants(value: str, max_variants: int = 96) -> list[str]:
    variants = [""]
    for char in value:
        choices = _DATE_DIGIT_REPAIRS.get(char, (char,))
        variants = [prefix + choice for prefix in variants for choice in choices]
        if len(variants) > max_variants:
            variants = variants[:max_variants]
    out: list[str] = []
    for item in variants:
        if item not in out:
            out.append(item)
    return out


def _repair_ymd_date_parts(year_text: str, month_text: str, day_text: str) -> tuple[int, int, int] | None:
    """Repair invalid YYYY-MM-DD OCR digits using a small confusion map.

    This is intentionally only called when the caller has already decided that
    the text is in a date context. It fixes common dot-matrix/thermal OCR
    confusions without applying those guesses to random unlabeled numbers.
    """
    month_variants = _date_digit_variants(month_text)
    # Dot-matrix taxi receipts often lose the upper/right strokes of "09",
    # which can become "53". Prefer that repair before looser valid months.
    if month_text == "53" and "09" in month_variants:
        month_variants = ["09"] + [item for item in month_variants if item != "09"]

    for y_text in _date_digit_variants(year_text):
        year = int(y_text)
        if not 1900 <= year <= 2099:
            continue
        for m_text in month_variants:
            month = int(m_text)
            if not 1 <= month <= 12:
                continue
            for d_text in _date_digit_variants(day_text):
                day = int(d_text)
                if _valid_date_parts(year, month, day):
                    return year, month, day
    return None


def _extract_dates(text: str, repair_invalid: bool = False) -> list[tuple[str, str]]:
    corrected = _correct_date_ocr(text)
    results: list[tuple[str, str]] = []

    for match in re.finditer(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)", corrected):
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if _valid_date_parts(year, month, day):
            results.append((f"{year:04d}-{month:02d}-{day:02d}", match.group(0)))
        elif repair_invalid:
            repaired = _repair_ymd_date_parts(match.group(1), match.group(2), match.group(3))
            if repaired:
                year, month, day = repaired
                results.append((f"{year:04d}-{month:02d}-{day:02d}", match.group(0)))

    for match in re.finditer(r"(?<!\d)(\d{1,2})([-/.])(\d{1,2})[-/.](\d{2,4})(?!\d)", corrected):
        day = int(match.group(1))
        separator = match.group(2)
        month = int(match.group(3))
        raw_year = match.group(4)
        year = _four_digit_year(raw_year)
        repaired = _repair_header_date_parts(day, month, year) if repair_invalid else None
        if repaired:
            day, month, year = repaired
        if _valid_date_parts(year, month, day):
            if len(raw_year) == 2:
                results.append((f"{day:02d}{separator}{month:02d}{separator}{raw_year}", match.group(0)))
            else:
                results.append((f"{day:02d}{separator}{month:02d}{separator}{year:04d}", match.group(0)))

    # DD-Mon-YYYY  (09-Sep-2010, 18 Mar 2001, 18/Mar/01)
    for m in re.finditer(r"(?<![A-Za-z0-9])(\d{1,2})[-/ ]([A-Za-z]{3,9})[-/ ](\d{2,4})(?![A-Za-z0-9])", corrected):
        day = int(m.group(1))
        mon = _month_from_name(m.group(2))
        year = _four_digit_year(m.group(3))
        if mon and _valid_date_parts(year, mon, day):
            results.append((f"{year:04d}-{mon:02d}-{day:02d}", m.group(0)))

    # Mon-DD-YYYY  (Sep 09, 2010)
    for m in re.finditer(r"(?<![A-Za-z0-9])([A-Za-z]{3,9})[-/ ](\d{1,2}),?[-/ ](\d{2,4})(?![A-Za-z0-9])", corrected):
        mon = _month_from_name(m.group(1))
        day = int(m.group(2))
        year = _four_digit_year(m.group(3))
        if mon and _valid_date_parts(year, mon, day):
            results.append((f"{year:04d}-{mon:02d}-{day:02d}", m.group(0)))

    return results


def looks_like_date(value: str) -> bool:
    """Return True when a string is almost certainly a calendar date, not an invoice ID.

    Handles DD/MM/YYYY, YYYY-MM-DD, DD-MM-YY, and strips trailing time parts
    so "22/12/2017 14.03" is caught even with a time component.
    Also handles alpha-month formats like "18 Mar 2001" and "Sep 09 2010".
    """
    v_full = (value or "").strip()
    v = v_full.split()[0]   # drop trailing time "22/12/2017 14.03"
    # DD/MM/YYYY  DD-MM-YYYY  DD/MM/YY  DD-MM-YY  (also MM/DD/YYYY variants)
    m = re.match(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})$", v)
    if m:
        a, b, _ = (int(x) for x in m.groups())
        if (1 <= a <= 31 and 1 <= b <= 12) or (1 <= b <= 31 and 1 <= a <= 12):
            return True
    # YYYY/MM/DD  YYYY-MM-DD
    m = re.match(r"^(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})$", v)
    if m:
        _, mo, d = (int(x) for x in m.groups())
        if 1 <= d <= 31 and 1 <= mo <= 12:
            return True
    # Alpha-month formats — must check the full string because space is the separator
    # (split()[0] would reduce "18 Mar 2001" to "18", breaking the match).
    # Also check split "v" to handle "18-Mar-2001 14:30" after stripping trailing time.
    for s in (v_full, v):
        m = re.match(r"^(\d{1,2})[-/ ]([A-Za-z]{3,9})[-/ ](\d{2,4})$", s)
        if m and _month_from_name(m.group(2)):
            return True
        m = re.match(r"^([A-Za-z]{3,9})[-/ ](\d{1,2}),?[-/ ](\d{2,4})$", s)
        if m and _month_from_name(m.group(1)):
            return True
    return False


def _extract_invoice_ids(text: str) -> list[str]:
    values = []
    for match in INVOICE_ID_RE.finditer(_normalize_ocr_text(text)):
        value = match.group(0).strip(":-# ")
        compact_value = _compact(value)
        if compact_value in {"DOCUMENT", "RECEIPT", "INVOICE", "BILL", "DATE", "TOTAL", "CASH", "CHANGE"}:
            continue
        if looks_like_date(value):
            continue
        if not re.search(r"[A-Z]", value, re.I) and _normalize_amount(value):
            continue
        if value not in values:
            values.append(value)
    return values


# Looser pattern for label-confirmed contexts: accepts digit+separator invoice IDs
# like "18222/102/70341" which have no letters but are clearly invoice numbers
_INVOICE_ID_NEAR_LABEL_RE = re.compile(
    r"\b(?=[0-9A-Z\-/]{5,})(?=[0-9A-Z\-/]*[0-9])(?=[0-9A-Z\-/]*[\-/])[0-9A-Z][0-9A-Z\-/]{3,}[0-9A-Z]\b",
    re.I,
)


def _extract_invoice_ids_near_label(text: str) -> list[str]:
    """Like _extract_invoice_ids but also accepts digit/slash/dash patterns when a label is confirmed."""
    strict = _extract_invoice_ids(text)
    if strict:
        return strict
    values = []
    normalized_text = _normalize_ocr_text(text)
    for match in _INVOICE_ID_NEAR_LABEL_RE.finditer(normalized_text):
        value = match.group(0).strip(":-# ")
        compact_value = _compact(value)
        if compact_value in {"DOCUMENT", "RECEIPT", "INVOICE", "BILL", "DATE", "TOTAL", "CASH", "CHANGE"}:
            continue
        if looks_like_date(value):
            continue
        if value not in values:
            values.append(value)
    if values:
        return values
    for match in PURE_DIGIT_ID_NEAR_LABEL_RE.finditer(normalized_text):
        value = match.group(1)
        if value == "0" * len(value):
            continue
        if looks_like_date(value):
            continue
        if value not in values:
            values.append(value)
    return values


def _exclude_invoice_format_prior_line(text: str) -> bool:
    canonical = _canonical_compact(text)
    return bool(_EMAIL_RE.search(text or "")) or any(
        fragment in canonical for fragment in _INVOICE_FORMAT_PRIOR_EXCLUDE_FRAGMENTS
    )


def _extract_vat_rates(text: str) -> list[str]:
    return [match.group(1).replace(" ", "").replace(",", ".") for match in VAT_RATE_RE.finditer(text)]


def _to_decimal(value: str) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _is_payment_context(text: str) -> bool:
    """Return True if text contains a label that marks it as a payment/cash line."""
    canonical = _canonical_compact(text)
    return any(label in canonical for label in PAYMENT_CONTEXT_LABELS)


def _has_context_label(text: str, labels: tuple[str, ...]) -> bool:
    canonical = _canonical_compact(text)
    return any(label in canonical for label in labels)


def _is_financial_summary_context(text: str) -> bool:
    """Return True for subtotal/tax/final-total summary rows, not item tables."""
    canonical = _canonical_compact(text)
    if any(label in canonical for label in FINANCIAL_SUMMARY_CONTEXT_LABELS):
        return True
    # Allow a standalone TOTAL row, but not table headers such as LINE TOTAL.
    return "TOTAL" in canonical and "LINETOTAL" not in canonical


def _is_line_item_amount_context(text: str) -> bool:
    """Return True for item-table headers/rows that should not become totals."""
    canonical = _canonical_compact(text)
    if not any(label in canonical for label in LINE_ITEM_CONTEXT_LABELS):
        return False
    # Table headers often include a "Subtotal" or "Line Total" column.  Do not
    # let those generic words turn the next line item into an invoice summary.
    strong_summary = (
        "ROUNDEDTOTAL",
        "ROUNDINGTOTAL",
        "GRANDTOTAL",
        "AMOUNTDUE",
        "NETTOTAL",
        "INVOICETOTAL",
        "TOTALAMOUNT",
        "TOTALSALES",
        "TOTALINCL",
        "TOTALEXCL",
        "GSTSUMMARY",
        "SSTSUMMARY",
        "TAXAMOUNT",
        "SERVICETAX",
        "SALESTAX",
    )
    return not any(label in canonical for label in strong_summary)


def _is_adjustment_context(text: str) -> bool:
    canonical = _canonical_compact(text)
    if any(label in canonical for label in ("ROUNDEDTOTAL", "ROUNDINGTOTAL")):
        return False
    return any(label in canonical for label in ADJUSTMENT_CONTEXT_LABELS)


# Labels that stop a neighboring-field search.
_CONTEXT_STOP_PREFIXES: frozenset[str] = frozenset({
    "DATE", "TIME", "CASHIER", "TERMINAL", "TABLE", "TOKEN",
    "TOTAL", "TAX", "GST", "SST", "VAT", "SUBTOTAL",
    "CASH", "CHANGE", "PAYMENT", "INVOICENO", "INVOICENUM",
    "RECEIPTNO", "DOCUMENTNO", "BILLNO",
    "DUE", "DUEDATE",
    "发票代码", "发票号码", "开票日期", "日期", "时间",
    "金额", "税额", "税率", "价税合计", "电话", "车号",
    "证号", "上车", "下车", "单价", "里程", "等候",
    "状态", "卡号", "密码",
})

_DATE_LABEL_FRAGMENTS: frozenset[str] = frozenset({
    "INVOICEDATE", "BILLDATE", "RECEIPTDATE", "INVOICED",
})


def _below_starts_with_stop_label(text: str) -> bool:
    """Return True when the first ~20 chars of the compact form start with a stop label."""
    compact = _canonical_compact(text)[:25]
    return any(compact.startswith(prefix) for prefix in _CONTEXT_STOP_PREFIXES)


# Known OCR variants of GST/SST Summary.
_GST_SUMMARY_CANONICAL_VARIANTS = frozenset({
    "GSTSUMMARY",
    "SSTSUMMARY",
    "6STSUMMARY",
    "GSTSUMARY",
    "GSTSUMXARY",
    "6S1SUMMARY",
    "6S1SURARY",
    "GSISUMMARY",
    "GSTSTMMARY",
    "GSTSURARY",
})

_GST_SUMMARY_STOP_LABELS = (
    "total",
    "grand total",
    "total sales",
    "amount due",
    "rounded total",
    "net total",
    "rounding",
    "total amount",
)


def _is_gst_summary_line(line: OCRLine) -> bool:
    """Detect GST/SST Summary header lines including noisy OCR variants."""
    canonical = _canonical_compact(line.text)
    if any(v in canonical for v in _GST_SUMMARY_CANONICAL_VARIANTS):
        return True
    for needle in ("GST Summary", "SST Summary", "GST Sumary", "GSI Summary", "6ST Summary"):
        if _compact_contains_fuzzy(line.text, needle, threshold=0.70):
            return True
    return False


def _candidate_score(
    base: float,
    alias_weight: float,
    context: ValueContext,
    lines: list[OCRLine],
    field: str,
) -> tuple[float, str]:
    score = base + alias_weight + context.confidence * 0.18
    reasons = [
        f"label '{context.label}'",
        context.direction.replace("_", " "),
        f"OCR confidence {context.confidence:.2f}",
    ]

    # Keep excluded contexts below viable candidates.
    if (_has_negative_anchor(context.label_line.search_text, field) or
            _has_negative_anchor(context.value_line.search_text, field)):
        score -= 1.0
        reasons.append(f"NEGATIVE anchor for {field} — disqualified")

    # Specific labels are more reliable than short generic labels.
    if len(_canonical_compact(context.label)) >= 6:
        score += 0.08
        reasons.append("specific anchor")

    if context.direction == "same_line_right":
        score += 0.12
        reasons.append("value is to the right of the label")
    elif context.direction == "same_line_after_label":
        score += 0.10
        reasons.append("value appears after the label")
    elif context.direction == "next_line_below":
        score += 0.06
        reasons.append("value is on the nearby line below")

    if context.distance:
        distance_penalty = min(context.distance / 700.0, 0.12)
        score -= distance_penalty
        reasons.append(f"label-value distance penalty {distance_penalty:.2f}")

    if field in {"total_amount", "tax_amount"}:
        label_text = _canonical_compact(context.label_line.search_text)
        value_text = _canonical_compact(context.value_line.search_text)
        combined = f"{label_text} {value_text}"

        if _is_payment_context(combined):
            score -= 0.70
            reasons.append("STRONG payment-context exclusion: cash/change/tendered line")
        elif any(_canonical_compact(kw) in combined for kw in ("DISCOUNT", "QTY", "QUANTITY", "UNITPRICE", "PRICEUNIT")):
            score -= 0.30
            reasons.append("near discount/quantity keyword")

    if field in {"tax_amount", "subtotal"}:
        # The receipt footer is usually the payment area.
        y = _y_ratio(context.bbox, lines)
        if y > 0.88:
            score -= 0.20
            reasons.append(f"zone-mismatch penalty: {field} candidate in deep payment zone ({y:.2f})")

    if field == "total_amount":
        label_text = _canonical_compact(context.label_line.search_text)
        value_text = _canonical_compact(context.value_line.search_text)
        y = _y_ratio(context.bbox, lines)
        if y >= 0.70:
            score += 0.16
            reasons.append("value is in the lower document region")
        elif y >= 0.50:
            score += 0.10
            reasons.append("value is in the lower half")
        if "SUBTOTAL" in label_text or "SUBTOTAL" in value_text:
            score -= 0.22
            reasons.append("line looks like subtotal, not final total")
        if context.label == "rounding" and "TOTAL" not in label_text:
            score -= 0.28
            reasons.append("rounding adjustment alone is not payable total")
        if "ROUNDEDTOTAL" in label_text or "GRANDTOTAL" in label_text or "AMOUNTDUE" in label_text or "TOTALSALES" in label_text:
            score += 0.12
            reasons.append("strong final-total keyword")

    if field in {"total_amount", "tax_amount", "subtotal"} and context.confidence < 0.45:
        cap = 0.32 + context.confidence * 0.50
        if score > cap:
            score = cap
            reasons.append(f"low OCR confidence cap for numeric value ({context.confidence:.2f})")

    return score, "; ".join(reasons)


def _value_candidates_from_label(lines: list[OCRLine]) -> dict[str, list[FieldCandidate]]:
    candidates: dict[str, list[FieldCandidate]] = {field: [] for field in FIELD_NAMES}
    for index, line in enumerate(lines):
        for field, aliases in LABEL_ALIASES.items():
            for alias, alias_weight in aliases:
                if not _line_has_alias(line, alias):
                    continue
                if field in {"tax_amount", "subtotal", "total_amount"}:
                    if _is_line_item_amount_context(line.text) or _is_adjustment_context(line.text):
                        continue
                if field == "total_amount" and alias == "total" and ("SUBTOTAL" in line.search_text or "SUB TOTAL" in line.search_text):
                    continue
                # "Total Incl/Excl GST" describes total or subtotal, not tax.
                if field == "tax_amount" and alias in ("gst", "vat", "tax"):
                    if "TOTAL" in _canonical_compact(line.text):
                        continue
                contexts = _contexts_for_label(lines, index, alias)
                for context in contexts:
                    if field in {"tax_amount", "subtotal", "total_amount"}:
                        context_text = f"{context.label_line.text} {context.value_line.text}"
                        if _is_line_item_amount_context(context_text) or _is_adjustment_context(context_text):
                            continue
                    if field == "invoice_number":
                        if any(term in _canonical_compact(context.source_text) for term in ("GST", "SST", "REGISTRATION")):
                            continue
                        # Invoice Date and Bill Date are date fields.
                        line_compact = _canonical_compact(line.search_text)
                        if any(frag in line_compact for frag in _DATE_LABEL_FRAGMENTS):
                            continue
                        values = [(value, value) for value in _extract_invoice_ids_near_label(context.text)]
                        base = 0.25
                    elif field == "date":
                        values = _extract_dates(context.text, repair_invalid=True)
                        base = 0.28
                    elif field == "tax_amount":
                        for rate in _extract_vat_rates(context.text):
                            candidates["vat_rate"].append(
                                FieldCandidate(
                                    field="vat_rate",
                                    value=rate,
                                    score=0.34 + alias_weight + context.confidence * 0.18,
                                    source_text=context.source_text,
                                    source_bbox=context.bbox,
                                    method="tax_label_rate_candidate",
                                    explanation=(
                                        f"VAT/tax rate '{rate}' found in tax context; "
                                        "kept separate from tax_amount"
                                    ),
                                )
                            )
                        values = _extract_amounts_with_repair(context.text, allow_integer=True)
                        base = 0.22
                    elif field == "subtotal":
                        values = _extract_amounts_with_repair(context.text, allow_integer=True)
                        base = 0.22
                    elif field == "total_amount":
                        values = _extract_amounts_with_repair(context.text, allow_integer=False)
                        base = 0.22
                    elif field == "vat_rate":
                        values = [(value, value) for value in _extract_vat_rates(context.text)]
                        base = 0.22
                    elif field == "amount_in_words":
                        value = context.text.strip(" :-")
                        values = [(value, value)] if value and re.search(r"[A-Z]{3,}", value, re.I) else []
                        base = 0.20
                    else:
                        values = []
                        base = 0.0

                    for value, raw_value in values:
                        if (_has_negative_anchor(context.label_line.search_text, field) or
                                _has_negative_anchor(context.value_line.search_text, field)):
                            continue
                        if field in {"tax_amount", "subtotal", "total_amount"}:
                            if not _valid_amount_value(value, field):
                                continue
                        if field == "invoice_number" and not _valid_invoice_value(value):
                            continue
                        score, explanation = _candidate_score(base, alias_weight, context, lines, field)
                        source_bbox = context.bbox
                        role = ""
                        if field in {"tax_amount", "subtotal", "total_amount"}:
                            ctx_text = _canonical_compact(context.source_text)
                            if _is_payment_context(ctx_text):
                                role = "cash_paid"
                            elif field == "tax_amount":
                                role = "tax_amount"
                            elif field == "subtotal":
                                role = "subtotal"
                            elif field == "total_amount":
                                role = "total_amount"
                            amount_confidence = _amount_confidence_in_line(context.value_line, raw_value)
                            source_bbox = _amount_bbox_in_line(context.value_line, raw_value)
                            score += (amount_confidence - context.confidence) * 0.28
                            explanation += f"; numeric-token confidence {amount_confidence:.2f}"
                            if amount_confidence < 0.45:
                                amount_cap = 0.32 + amount_confidence * 0.50
                                if score > amount_cap:
                                    score = amount_cap
                                    explanation += f"; low numeric-token confidence cap ({amount_confidence:.2f})"
                        candidates[field].append(
                            FieldCandidate(
                                field=field,
                                value=value,
                                score=score,
                                source_text=context.source_text,
                                source_bbox=source_bbox,
                                method="label_neighbor_candidate",
                                explanation=explanation + f"; extracted value '{raw_value}'",
                                amount_role=role,
                            )
                        )

                if field == "currency":
                    continue
                for currency in _detect_all_currencies(line.text):
                    if field == "total_amount":
                        candidates["currency"].append(
                            FieldCandidate(
                                field="currency",
                                value=currency,
                                score=0.84 + alias_weight * 0.15,
                                source_text=line.text,
                                source_bbox=line.bbox,
                                method="currency_in_total_label",
                                explanation=f"currency found in total label '{alias}'",
                            )
                        )
    return candidates


def _currency_candidates(lines: list[OCRLine]) -> list[FieldCandidate]:
    candidates: list[FieldCandidate] = []
    # Strong RM context suppresses isolated dollar-sign artifacts.
    rm_evidence = sum(1 for line in lines if _RM_EVIDENCE_RE.search(line.text))
    chinese_invoice_profile = _looks_like_chinese_fapiao_profile(lines)
    for line in lines:
        for currency in _detect_all_currencies(line.text):
            score = 0.58 + line.confidence * 0.18
            explanation = "currency symbol/code detected in OCR line"
            if chinese_invoice_profile and currency in {"USD", "EUR", "GBP"}:
                explicit_code = re.search(rf"\b{currency}\b", line.text, re.I)
                if not explicit_code:
                    score -= 0.55
                    explanation = "bare foreign currency symbol suppressed in Chinese invoice profile"
            if currency == "USD" and not re.search(r"\bUSD\b", line.text, re.I):
                if rm_evidence >= 1 or line.confidence < 0.35:
                    score -= 0.45
                    explanation = "bare $ from low-confidence / RM-context line — likely OCR artifact"
            if any(_line_has_alias(line, alias) for alias, _ in LABEL_ALIASES["total_amount"]):
                score += 0.20
                explanation = "currency detected in total-related line"
            candidates.append(
                FieldCandidate(
                    field="currency",
                    value=currency,
                    score=score,
                    source_text=line.text,
                    source_bbox=line.bbox,
                    method="currency_detection",
                    explanation=explanation,
                )
            )
    return candidates


def _gst_summary_candidates(lines: list[OCRLine]) -> dict[str, list[FieldCandidate]]:
    candidates: dict[str, list[FieldCandidate]] = {"subtotal": [], "tax_amount": []}
    for index, line in enumerate(lines):
        if not _is_gst_summary_line(line):
            continue

        # Keep the GST table window above totals and payment rows.
        section: list[OCRLine] = [line]
        for i in range(index + 1, min(len(lines), index + 8)):
            candidate = lines[i]
            sc = _canonical_compact(candidate.text)
            if _is_payment_context(sc):
                break
            if any(_line_has_alias(candidate, lbl) for lbl in _GST_SUMMARY_STOP_LABELS):
                break
            section.append(candidate)

        # Column centers preserve roles on merged OCR lines.
        amount_col_cx: float | None = None
        tax_col_cx: float | None = None
        for section_line in section:
            for token in section_line.tokens:
                tc = _canonical_compact(token.text)
                if tc == "AMOUNT" and amount_col_cx is None:
                    amount_col_cx = token.cx
                elif tc == "TAX" and tax_col_cx is None:
                    tax_col_cx = token.cx

        # Keep each amount's x-position for column matching.
        amount_items: list[tuple[str, str, float, float, OCRLine]] = []

        for section_line in section:
            if _is_gst_summary_line(section_line):
                continue
            tokens = section_line.tokens
            used: set[int] = set()
            for i, token in enumerate(tokens):
                if i in used:
                    continue
                if i + 1 < len(tokens) and (i + 1) not in used:
                    pair_text = f"{token.text} {tokens[i + 1].text}"
                    matched = False
                    for m in _SPACED_DECIMAL_AMOUNT_RE.finditer(pair_text):
                        normalized = _normalize_amount(f"{m.group(1)}.{m.group(2)}")
                        if normalized:
                            pair_cx = (token.cx + tokens[i + 1].cx) / 2
                            pair_conf = (token.confidence + tokens[i + 1].confidence) / 2
                            amount_items.append((normalized, pair_text, pair_cx, pair_conf, section_line))
                            used.add(i)
                            used.add(i + 1)
                            matched = True
                            break
                    if matched:
                        continue
                for value, raw in _extract_amounts(token.text, allow_integer=False):
                    amount_items.append((value, raw, token.cx, token.confidence, section_line))
                    used.add(i)
                    break

            # Recover decimals split around punctuation.
            existing_values = {(value, src_line.text) for value, _raw, _cx, _conf, src_line in amount_items}
            repaired_line_amounts = _extract_amounts_with_repair(section_line.text, allow_integer=False)
            if repaired_line_amounts:
                line_width = max(section_line.x2 - section_line.x1, 1.0)
                step = line_width / max(len(repaired_line_amounts) + 1, 2)
                for idx_amount, (value, raw) in enumerate(repaired_line_amounts, start=1):
                    if (value, section_line.text) in existing_values:
                        continue
                    amount_items.append(
                        (
                            value,
                            raw,
                            section_line.x1 + step * idx_amount,
                            section_line.confidence,
                            section_line,
                        )
                    )
                    existing_values.add((value, section_line.text))

        if not amount_items:
            continue

        amount_items.sort(key=lambda item: (item[4].cy, item[2]))

        if amount_col_cx is not None and tax_col_cx is not None:
            for value, raw, cx, conf, src_line in amount_items:
                dist_amount = abs(cx - amount_col_cx)
                dist_tax = abs(cx - tax_col_cx)
                if dist_amount <= dist_tax:
                    field = "subtotal"
                    role = "subtotal"
                    expl = f"GST Summary Amount column (cx {cx:.0f}≈{amount_col_cx:.0f}): '{raw}'"
                else:
                    field = "tax_amount"
                    role = "tax_amount"
                    expl = f"GST Summary Tax column (cx {cx:.0f}≈{tax_col_cx:.0f}): '{raw}'"
                candidates[field].append(
                    FieldCandidate(
                        field=field,
                        value=value,
                        score=0.88 + conf * 0.10,
                        source_text=src_line.text,
                        source_bbox=src_line.bbox,
                        method="gst_summary_column_candidate",
                        explanation=expl,
                        amount_role=role,
                    )
                )
        else:
            # GST rows usually list subtotal before tax.
            positive_items = [
                item for item in amount_items
                if (_to_decimal(item[0]) or Decimal("0")) > Decimal("0")
            ]
            chosen_items = positive_items[:2] if len(positive_items) >= 2 else amount_items[:2]
            for idx2, (value, raw, cx, conf, src_line) in enumerate(chosen_items):
                field = "subtotal" if idx2 == 0 else "tax_amount"
                candidates[field].append(
                    FieldCandidate(
                        field=field,
                        value=value,
                        score=0.82 + conf * 0.10,
                        source_text=src_line.text,
                        source_bbox=src_line.bbox,
                        method="gst_summary_table_candidate",
                        explanation=f"GST Summary positional {field} from '{raw}'",
                        amount_role=field,
                    )
                )
    return candidates


def _amount_line_candidates(lines: list[OCRLine]) -> list[tuple[str, str, OCRLine]]:
    values: list[tuple[str, str, OCRLine]] = []
    for line in lines:
        canonical = _canonical_compact(line.text)
        if _is_payment_context(canonical):
            continue
        if _is_line_item_amount_context(line.text) or _is_adjustment_context(line.text):
            continue
        for value, raw in _extract_amounts_with_repair(line.text, allow_integer=False):
            values.append((value, raw, line))
    return values


def _amount_bbox_in_line(line: OCRLine, raw: str) -> list[float]:
    raw_norm = _normalize_amount(raw)
    for token in line.tokens:
        for value, _ in _extract_amounts_with_repair(token.text, allow_integer=False):
            if value == raw_norm:
                return list(token.bbox)
    return list(line.bbox)


def _amount_confidence_in_line(line: OCRLine, raw: str) -> float:
    raw_norm = _normalize_amount(raw)
    for token in line.tokens:
        for value, _ in _extract_amounts_with_repair(token.text, allow_integer=False):
            if value == raw_norm:
                return token.confidence
    return line.confidence


def _add_financial_consistency_candidates(lines: list[OCRLine], candidates: dict[str, list[FieldCandidate]]) -> None:
    subtotal_candidates = candidates.get("subtotal", [])
    tax_candidates = candidates.get("tax_amount", [])
    total_candidates = candidates.get("total_amount", [])

    # Complete a known subtotal and tax pair.
    if subtotal_candidates and tax_candidates:
        best_subtotal = max(subtotal_candidates, key=lambda item: item.score)
        best_tax = max(tax_candidates, key=lambda item: item.score)
        subtotal = _to_decimal(best_subtotal.value)
        tax = _to_decimal(best_tax.value)
        if subtotal is not None and tax is not None:
            expected = subtotal + tax
            if expected > Decimal("0"):
                matched_existing_total = False
                for candidate in total_candidates:
                    amount = _to_decimal(candidate.value)
                    if amount is not None and abs(amount - expected) <= Decimal("0.05"):
                        matched_existing_total = True
                        candidate.score += 0.24
                        candidate.explanation += f"; supported by financial consistency: subtotal + tax ({subtotal} + {tax} ~= {candidate.value})"
                for value, raw, line in _amount_line_candidates(lines):
                    amount = _to_decimal(value)
                    if amount is None or abs(amount - expected) > Decimal("0.05"):
                        continue
                    matched_existing_total = True
                    y = _y_ratio(line.bbox, lines)
                    score = 0.76 + line.confidence * 0.12
                    if y > 0.50:
                        score += 0.08
                    if _line_has_alias(line, "total sales") or _line_has_alias(line, "rounded total") or _line_has_alias(line, "amount due"):
                        score += 0.12
                        source = "total label and financial consistency"
                    else:
                        source = "financial consistency"
                    candidates["total_amount"].append(
                        FieldCandidate(
                            field="total_amount",
                            value=value,
                            score=score,
                            source_text=line.text,
                            source_bbox=line.bbox,
                            method="financial_consistency_candidate",
                            explanation=f"{source}: value '{raw}' is within 0.05 of subtotal + tax ({subtotal} + {tax})",
                        )
                    )
                if not matched_existing_total:
                    final_total_aliases = (
                        "total amount",
                        "amount due",
                        "grand total",
                        "net total",
                        "rounded total",
                        "total sales",
                        "total incl",
                    )
                    for line in lines:
                        if _is_line_item_amount_context(line.text) or _is_adjustment_context(line.text):
                            continue
                        if not any(_line_has_alias(line, alias) for alias in final_total_aliases):
                            continue
                        has_complete_amount = bool(_extract_amounts_with_repair(line.text, allow_integer=False))
                        has_incomplete_decimal = bool(re.search(r"\d+[,.]\s*(?:$|[^\d])", line.text))
                        if has_complete_amount and not has_incomplete_decimal:
                            continue
                        expected_value = format(expected.quantize(Decimal("0.01")), "f")
                        score = min(
                            0.93,
                            0.78 + min(best_subtotal.confidence(), best_tax.confidence()) * 0.14,
                        )
                        candidates["total_amount"].append(
                            FieldCandidate(
                                field="total_amount",
                                value=expected_value,
                                score=score,
                                source_text=line.text,
                                source_bbox=line.bbox,
                                method="computed_total_from_subtotal_tax",
                                explanation=(
                                    f"explicit total label has incomplete/missing amount; "
                                    f"computed subtotal + tax ({subtotal} + {tax} = {expected_value})"
                                ),
                                amount_role="total_amount",
                            )
                        )
                        break

    # Infer subtotal and tax from a known total.
    if _looks_like_chinese_taxi_profile(lines):
        return
    if total_candidates and (not subtotal_candidates or not tax_candidates):
        best_total = max(total_candidates, key=lambda item: item.score)
        total_val = _to_decimal(best_total.value)
        if total_val is not None and total_val > Decimal("0"):
            all_amounts = _amount_line_candidates(lines)
            for (val_a, raw_a, line_a), (val_b, raw_b, line_b) in (
                (a, b) for i, a in enumerate(all_amounts) for b in all_amounts[i + 1:]
            ):
                dec_a, dec_b = _to_decimal(val_a), _to_decimal(val_b)
                if dec_a is None or dec_b is None:
                    continue
                if dec_a < dec_b:
                    dec_a, dec_b = dec_b, dec_a
                    val_a, raw_a, line_a, val_b, raw_b, line_b = val_b, raw_b, line_b, val_a, raw_a, line_a
                if abs(dec_a + dec_b - total_val) > Decimal("0.05"):
                    continue
                if dec_b >= total_val * Decimal("0.25"):
                    continue
                base_score = 0.72 + (line_a.confidence + line_b.confidence) / 2 * 0.10
                if not subtotal_candidates:
                    candidates["subtotal"].append(
                        FieldCandidate(
                            field="subtotal",
                            value=val_a,
                            score=base_score,
                            source_text=line_a.text,
                            source_bbox=line_a.bbox,
                            method="reverse_consistency_candidate",
                            explanation=f"reverse financial consistency: '{raw_a}' + '{raw_b}' ≈ total {best_total.value}",
                            amount_role="subtotal",
                        )
                    )
                if not tax_candidates:
                    candidates["tax_amount"].append(
                        FieldCandidate(
                            field="tax_amount",
                            value=val_b,
                            score=base_score,
                            source_text=line_b.text,
                            source_bbox=line_b.bbox,
                            method="reverse_consistency_candidate",
                            explanation=f"reverse financial consistency: '{raw_a}' + '{raw_b}' ≈ total {best_total.value}",
                            amount_role="tax_amount",
                        )
                    )


def _merged_payment_total_candidates(lines: list[OCRLine]) -> list[FieldCandidate]:
    """Recover totals from OCR lines that merge TOTAL/CASH/CHANGE labels.

    Some receipt OCR engines collapse adjacent summary rows into one line, for
    example "CHANGE : TOTAL CASH RM RM RM 39. 80 39. 80 0.00".  The normal
    payment-context guard correctly blocks cash/change lines, but a line that
    explicitly contains TOTAL can still provide useful payable-total evidence.
    Prefer a repeated non-zero amount because duplicated values often represent
    "Total" and "Cash" columns with the same payable value.
    """
    candidates: list[FieldCandidate] = []
    for line in lines:
        canonical = _canonical_compact(line.text)
        if "TOTAL" not in canonical or not _is_payment_context(canonical):
            continue
        amounts = _extract_amounts_with_repair(line.text, allow_integer=False)
        nonzero = [(value, raw) for value, raw in amounts if _to_decimal(value) not in {None, Decimal("0")}]
        if not nonzero:
            continue

        counts: dict[str, int] = {}
        first_raw: dict[str, str] = {}
        for value, raw in nonzero:
            counts[value] = counts.get(value, 0) + 1
            first_raw.setdefault(value, raw)

        repeated = [value for value, count in counts.items() if count > 1]
        if repeated:
            value = max(repeated, key=lambda item: (_to_decimal(item) or Decimal("0")))
            raw = first_raw[value]
            reason = "repeated non-zero amount in merged total/cash/change line"
        else:
            value, raw = nonzero[0]
            reason = "first non-zero amount after merged total/payment labels"

        score = 0.76 + line.confidence * 0.10
        if _y_ratio(line.bbox, lines) >= 0.50:
            score += 0.06
        candidates.append(
            FieldCandidate(
                field="total_amount",
                value=value,
                score=score,
                source_text=line.text,
                source_bbox=_amount_bbox_in_line(line, raw),
                method="merged_payment_total_candidate",
                explanation=f"{reason}; extracted '{raw}' from OCR-merged summary/payment line",
                amount_role="total_amount",
            )
        )
    return candidates


def _clean_vendor_text(text: str) -> str:
    value = _normalize_ocr_text(text).upper()
    value = value.replace("_", "-")
    value = _strip_vendor_prefix_noise(value)
    value = re.sub(r"\bSDM\s+BKD\b", "SDN BHD", value)
    value = re.sub(r"\bSDM\s+BHD\b", "SDN BHD", value)
    value = re.sub(r"\bSDN\s+BKD\b", "SDN BHD", value)
    value = re.sub(r"\bSOM\s+BHD\b", "SDN BHD", value)
    value = re.sub(r"\bBHD\b", "BHD", value)
    value = re.sub(r"\bBOOKTA([-\s])", r"BOOK TA\1", value)
    value = _strip_vendor_trailing_noise(value)
    value = re.sub(r"\s+", " ", value).strip(" -,:")
    return _finalize_latin_vendor_text(value)


def _vendor_line_is_valid(text: str) -> bool:
    search = _searchable(text)
    if len(search) < 4:
        return False
    if _vendor_has_hard_exclude(text):
        return False
    if any(keyword in search for keyword in VENDOR_EXCLUDE_KEYWORDS):
        return False
    if re.match(r"^(?:NAME|NAMA)\b", search) and re.search(r"\d{3,}", search):
        return False
    # Also check canonical-compact form so OCR-noisy operational labels (e.g.
    # "TERHINAL" → "TERMINAL") are caught even when they don't match literally.
    canonical = _canonical_compact(text)
    if any(_compact(kw) in canonical for kw in VENDOR_EXCLUDE_KEYWORDS):
        return False
    if _has_vendor_address_keyword(search) and not _has_legal_business_suffix(search):
        return False
    if re.search(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\s+(?:US|USA)\b", search) and not BUSINESS_SUFFIX_RE.search(search):
        return False
    if re.search(r"\b\d{5}(?:-\d{4})?\s+(?:US|USA)\b", search) and not BUSINESS_SUFFIX_RE.search(search):
        return False
    # Pipe characters in raw text signal OCR-noisy address fragments (e.g. "SHAT | AM" → "SHAH ALAM")
    # or table separators — neither is a valid business name.
    if "|" in text and not BUSINESS_SUFFIX_RE.search(text.upper()):
        return False
    # "City, State" / "City, Country" comma pattern (works for any country without hardcoding city names):
    # reject when every comma-separated part is ≤ 2 words and ≤ 15 chars — this matches geographic
    # fragments ("SHAH ALAM, SELANGOR", "LONDON, UK", "PARIS, FRANCE") but not business names like
    # "ALICE & BOB'S DINER, GRILL & BAR" (too many words per part).
    if "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if (len(parts) >= 2
                and all(len(p.split()) <= 2 and len(p) <= 15 for p in parts)
                and not BUSINESS_SUFFIX_RE.search(text.upper())):
            return False
    # Lines like "1413-SETIA ALAM 2" (3+ leading digits then dash/space) are lot/address numbers
    if re.match(r"^\d{3,}[-\s]", search) and not BUSINESS_SUFFIX_RE.search(search):
        return False
    if not re.search(r"[A-Z]", search):
        return False
    if re.fullmatch(r"[0-9\s:/.,#-]+", search):
        return False
    if len(re.sub(r"[^A-Z0-9]", "", search)) >= 14 and " " not in search and not BUSINESS_SUFFIX_RE.search(search):
        return False
    digit_ratio = len(re.findall(r"\d", search)) / max(len(search), 1)
    return digit_ratio <= 0.45


# Only parenthesised registration numbers (e.g. "(517537-X)") trigger a multi-line join.
# Bare 6+-digit numbers are GST IDs, phone numbers, etc. — too ambiguous to join on.
_REG_NUM_RE = re.compile(r"\(\s*\d{4,}[-\s]*[A-Za-z]?\s*\)")

VENDOR_HARD_EXCLUDE_PHRASES = (
    "LOGO",
    "COMMERCIAL INVOICE",
    "CREDIT NOTE",
    "CLIENT DETAILS",
    "CLIENT'S DETAILS",
    "CUSTOMER DETAILS",
    "CUSTOMER INFORMATION",
    "BILL TO",
    "SHIP TO",
    "SOLD TO",
    "PURCHASE ORDER",
)
VENDOR_PREFIX_DROP_PATTERNS = (
    re.compile(r"^\s*FROM\s+", re.I),
    re.compile(r"^\s*VENDOR\s*[:#-]?\s+", re.I),
    re.compile(r"^\s*SUPPLIER\s*[:#-]?\s+", re.I),
    re.compile(r"^.*\bGOVERNMENT\s+OF\s+THE\s+PEOPLE'?S\s+REPUBLIC\s+OF\s+BANGLADESH\s+", re.I),
    re.compile(r"^.*\b(?:NATIONAL|NATONAL)\s+BOARD\s+REVENUE\s+", re.I),
    re.compile(r"^.*\bBOARD\s+REVENUE\s+", re.I),
    re.compile(r"^.*\bGOVERNMENT\s+OF\b.*?\bREVENUE\s+", re.I),
)
VENDOR_TRAILING_STOP_PATTERNS = (
    re.compile(r"\bBUSINESS\s+ADDRESS\b.*$", re.I),
    re.compile(r"\bCLIENT'?S?\s+DETAILS\b.*$", re.I),
    re.compile(r"\bCUSTOMER\s+(?:DETAILS|INFORMATION)\b.*$", re.I),
    re.compile(r"\b(?:BILL|SHIP|SOLD)\s+TO\b.*$", re.I),
    re.compile(r"\b(?:TAX\s+)?INVOICE\s*(?:NO|NUMBER|DATE|#)\b.*$", re.I),
    re.compile(r"\b(?:DOCUMENT|RECEIPT|BILL)\s*(?:NO|NUMBER|DATE|#)\b.*$", re.I),
    re.compile(r"\bBUSINESS\s+REG(?:ISTRATION)?\b.*$", re.I),
    re.compile(r"\b(?:GST|SST|VAT)\s*(?:REG|ID|NO|NUMBER)\b.*$", re.I),
    re.compile(r"\bCO\.?\s*NO\b.*$", re.I),
    re.compile(r"\bREG\s*NO\b.*$", re.I),
    re.compile(r"\bDATE\s*[:#-].*$", re.I),
    re.compile(r"\b(?:TEL|TELEPHONE|PHONE|FAX)\b.*$", re.I),
    re.compile(r"\bNO\s*[:：.]?\s*\d.*$", re.I),
    re.compile(r"\b(?:JALAN|ROAD|STREET)\b.*$", re.I),
)
VENDOR_ENTITY_TERMINAL_RE = re.compile(
    r"\b("
    r"SDN\s+BHD|S\s*/\s*[B8]|LTD|LIMITED|GMBH|INC\.?|CORP\.?|LLC"
    r"|STATIONERY|LOGISTICS|HARDWARE|ELECTRICAL|RESTAURANT|RESTORAN"
    r"|GALLERY|BISTRO|HOTEL|TECHNOLOGIES|TECHNOLOGY|SERVICES"
    r")\b",
    re.I,
)
VENDOR_LEGAL_SUFFIX_RE = re.compile(
    r"\b("
    r"SDN\s+BHD|S\.?D\.?N\.?\s+B\.?H\.?D\.?|S\s*/\s*[B8]"
    r"|LTD|LIMITED|PLC|PTY\.?\s*LTD"
    r"|INC\.?|INCORPORATED|CORP\.?|CORPORATION|LLC|L\.L\.C\.|LP|LLP"
    r"|GMBH|A\.?G\.|S\.A\.|S\.R\.L\.|S\.L\."
    r"|COMPANY|CO\.|ENTERPRISE|TRADING|PERNIAGAAN|SYARIKAT|RESTORAN"
    r")\b",
    re.I,
)
CHINESE_VENDOR_POSITIVE_KEYWORDS = (
    "\u7a0e\u52a1\u5c40",
    "\u51fa\u79df\u8f66",
    "\u6709\u9650\u516c\u53f8",
    "\u516c\u53f8",
    "\u9152\u5e97",
    "\u79d1\u6280",
)
CHINESE_VENDOR_REMOVE_SUFFIXES = (
    "\u901a\u7528\u673a\u6253\u53d1\u7968",
    "\u673a\u6253\u53d1\u7968",
    "\u624b\u5199\u65e0\u6548",
    "\u53d1\u7968\u8054",
    "\u53d1\u7968\u4e13\u7528\u7ae0",
)
CHINESE_VENDOR_NEGATIVE_FRAGMENTS = (
    "\u53d1\u7968\u4ee3\u7801",
    "\u53d1\u7968\u53f7\u7801",
    "\u65e5\u671f",
    "\u8d2d\u4e70\u65b9",
    "\u8d2d\u65b9",
    "\u4e70\u65b9",
    "\u8d27\u7269\u6216\u5e94\u7a0e\u52b3\u52a1",
    "\u670d\u52a1\u540d\u79f0",
    "\u89c4\u683c\u578b\u53f7",
    "\u5355\u4f4d",
    "\u6570\u91cf",
    "\u5355\u4ef7",
    "\u7535\u8bdd",
    "\u8f66\u53f7",
    "\u8bc1\u53f7",
    "\u91d1\u989d",
    "\u7a0e\u7387",
    "\u7a0e\u989d",
    "\u5361\u53f7",
    "\u5bc6\u7801",
)
CHINESE_COMPANY_RE = re.compile(
    r"[\u3400-\u9fffA-Za-z0-9（）()·\-]{2,60}?(?:\u6709\u9650\u516c\u53f8|\u516c\u53f8|\u9152\u5e97)"
)
CHINESE_TAX_BUREAU_RE = re.compile(r"[\u3400-\u9fff]{2,18}?\u7a0e\u52a1\u5c40")
CHINESE_VENDOR_LABEL_ALIASES = CHINESE_VENDOR_LABELS + (
    "\u9500\u552e\u65b9\u540d\u79f0",
    "\u9500\u552e\u65b9",
    "\u9500\u65b9\u540d\u79f0",
    "\u9500\u65b9",
    "\u8d2d\u4e70\u65b9\u540d\u79f0",
    "\u5f00\u7968\u5355\u4f4d",
    "\u6536\u6b3e\u5355\u4f4d",
    "\u5355\u4f4d\u540d\u79f0",
    "\u516c\u53f8\u540d\u79f0",
    "\u7eb3\u7a0e\u4eba\u540d\u79f0",
)
CHINESE_VENDOR_SELLER_LABELS = (
    "\u9500\u552e\u65b9\u540d\u79f0",
    "\u9500\u65b9\u540d\u79f0",
    "\u9500\u552e\u65b9",
    "\u9500\u65b9",
)
CHINESE_VENDOR_BUYER_LABELS = (
    "\u8d2d\u4e70\u65b9\u540d\u79f0",
    "\u8d2d\u4e70\u65b9",
    "\u8d2d\u65b9",
    "\u4e70\u65b9",
)
CHINESE_VENDOR_TABLE_LABELS = (
    "\u8d27\u7269\u6216\u5e94\u7a0e\u52b3\u52a1",
    "\u670d\u52a1\u540d\u79f0",
    "\u89c4\u683c\u578b\u53f7",
    "\u5355\u4f4d",
    "\u6570\u91cf",
    "\u5355\u4ef7",
    "\u91d1\u989d",
    "\u7a0e\u7387",
    "\u7a0e\u989d",
)
CHINESE_VENDOR_STOP_MARKERS = (
    "\u7eb3\u7a0e\u4eba\u8bc6\u522b\u53f7",
    "\u7eb3\u7a0e\u4eba\u8bc6\u522b\u7801",
    "\u7a0e\u53f7",
    "\u5730\u5740",
    "\u7535\u8bdd",
    "\u5f00\u6237\u884c",
    "\u94f6\u884c",
    "\u8d26\u53f7",
    "\u53d1\u7968\u4ee3\u7801",
    "\u53d1\u7968\u53f7\u7801",
)
CHINESE_VENDOR_BAD_PREFIXES = (
    "\u95e8\u5e97\u5730\u5740",
    "\u5730\u5740",
    "\u7535\u8bdd",
    "\u8f66\u53f7",
    "\u8bc1\u53f7",
    "\u65e5\u671f",
    "\u91d1\u989d",
    "\u5361\u53f7",
    "\u5bc6\u7801",
    "\u4f1a\u5458",
    "\u6536\u94f6\u5458",
)
CHINESE_VENDOR_ALLOWED_ISSUER_FRAGMENTS = (
    "\u7a0e\u52a1\u5c40",
    "\u56fd\u5bb6\u7a0e\u52a1\u5c40",
)


def _strip_vendor_prefix_noise(value: str) -> str:
    for pattern in VENDOR_PREFIX_DROP_PATTERNS:
        value = pattern.sub("", value).strip()
    return value


def _strip_vendor_trailing_noise(value: str) -> str:
    original = value
    for pattern in VENDOR_TRAILING_STOP_PATTERNS:
        next_value = pattern.sub("", value).strip(" -,:")
        if next_value != value:
            if "SDN BHD" in original and "SDN BHD" not in next_value and next_value:
                next_value = f"{next_value} SDN BHD"
            value = next_value
            break
    return value


def _collapse_repeated_cjk_text(value: str) -> str:
    compact = re.sub(r"\s+", "", value or "")
    if len(re.findall(r"[\u3400-\u9fff]", compact)) < 4:
        return value.strip()
    for size in range(3, (len(compact) // 2) + 1):
        if len(compact) % size:
            continue
        chunk = compact[:size]
        if chunk and chunk * (len(compact) // size) == compact:
            return chunk
    return value.strip()


def _strip_chinese_vendor_label_prefix(value: str) -> str:
    for label in CHINESE_VENDOR_LABEL_ALIASES:
        if label and label in value:
            value = value.split(label, 1)[1]
            break
    return value.strip(" :\uff1a,\uff0c;\uff1b#-")


def _cut_chinese_vendor_after_stop(value: str) -> str:
    cut_at: int | None = None
    for marker in CHINESE_VENDOR_STOP_MARKERS:
        idx = value.find(marker)
        if idx > 0 and (cut_at is None or idx < cut_at):
            cut_at = idx
    if cut_at is not None:
        value = value[:cut_at]
    return value.strip(" :\uff1a,\uff0c;\uff1b#-")


def _extract_chinese_vendor_core(text: str) -> str:
    value = _clean_chinese_vendor_text(text)
    if not value:
        return ""
    tax_bureau = CHINESE_TAX_BUREAU_RE.search(value)
    if tax_bureau:
        return _collapse_repeated_cjk_text(tax_bureau.group(0))
    company_matches = [
        match.group(0).strip(" :\uff1a,\uff0c;\uff1b#-")
        for match in CHINESE_COMPANY_RE.finditer(value)
    ]
    company_matches = [match for match in company_matches if len(match) >= 3]
    if company_matches:
        return _collapse_repeated_cjk_text(max(company_matches, key=len))
    return _collapse_repeated_cjk_text(value)


def _truncate_after_vendor_terminal(value: str) -> str:
    matches = list(VENDOR_ENTITY_TERMINAL_RE.finditer(value))
    if not matches:
        return value
    # Prefer legal suffixes such as SDN BHD; otherwise use the first business
    # terminal when the following text looks like duplicated OCR spillover.
    legal_match = next(
        (
            match for match in matches
            if re.search(r"SDN\s+BHD|S\s*/\s*[B8]|LTD|LIMITED|GMBH|INC|CORP|LLC", match.group(0), re.I)
        ),
        None,
    )
    match = legal_match or matches[0]
    tail = value[match.end():].strip()
    if not tail:
        return value
    if legal_match and _REG_NUM_RE.fullmatch(tail):
        return value
    if legal_match or len(tail.split()) >= 2:
        return value[:match.end()].strip(" -,:")
    return value


def _vendor_segment_score(value: str) -> float:
    search = _searchable(value)
    score = 0.0
    if VENDOR_LEGAL_SUFFIX_RE.search(search):
        score += 2.0
    if VENDOR_ENTITY_TERMINAL_RE.search(search):
        score += 1.2
    if re.search(r"\b(?:FROM|GOVERNMENT|ADDRESS|JALAN|PHONE|TEL|EMAIL|NAME)\b", search):
        score -= 1.4
    digit_ratio = len(re.findall(r"\d", search)) / max(len(search), 1)
    score -= digit_ratio * 3.0
    words = [word for word in re.findall(r"[A-Z&]+", search) if len(word) > 1]
    score += min(len(words), 5) * 0.12
    score -= max(len(words) - 5, 0) * 0.08
    return score


def _split_repeated_vendor_segments(value: str) -> list[str]:
    words = value.split()
    if len(words) < 4:
        return [value]
    anchor = next((word for word in words if re.search(r"[A-Z]", word) and len(re.sub(r"[^A-Z0-9]", "", word)) >= 3), "")
    if not anchor:
        return [value]
    anchor_key = re.sub(r"[^A-Z0-9]", "", anchor.upper())
    positions = [
        idx for idx, word in enumerate(words)
        if re.sub(r"[^A-Z0-9]", "", word.upper()) == anchor_key
    ]
    if len(positions) < 2:
        return [value]
    segments = [value]
    for idx, start in enumerate(positions):
        end = positions[idx + 1] if idx + 1 < len(positions) else len(words)
        segment = " ".join(words[start:end]).strip()
        if segment:
            segments.append(segment)
    return segments


def _finalize_latin_vendor_text(value: str) -> str:
    value = _strip_vendor_prefix_noise(value)
    value = _strip_vendor_trailing_noise(value)
    value = re.sub(r"\s+", " ", value).strip(" -,:")
    candidates = []
    for segment in _split_repeated_vendor_segments(value):
        segment = _strip_vendor_trailing_noise(_truncate_after_vendor_terminal(segment))
        segment = re.sub(r"\s+", " ", segment).strip(" -,:")
        if segment and _vendor_line_is_valid(segment):
            candidates.append(segment)
    if candidates:
        value = max(candidates, key=_vendor_segment_score)
    value = _truncate_after_vendor_terminal(value)
    value = re.sub(r"\bBOOK\s+TA\s*\.\s*K\b", "BOOK TA-K", value, flags=re.I)
    value = re.sub(r"\bBOOKTA\s*[-.]?\s*K\b", "BOOK TA-K", value, flags=re.I)
    value = re.sub(r"\bLOG(?:L|I)ATICS\b", "LOGISTICS", value, flags=re.I)
    value = re.sub(r"\bSTATLONERY\b", "STATIONERY", value, flags=re.I)
    return re.sub(r"\s+", " ", value).strip(" -,:")


def _vendor_has_hard_exclude(value: str) -> bool:
    search = _searchable(value)
    return any(phrase in search for phrase in VENDOR_HARD_EXCLUDE_PHRASES)


def _has_vendor_address_keyword(value: str) -> bool:
    search = _searchable(value)
    return any(keyword in search for keyword in VENDOR_ADDRESS_KEYWORDS)


def _has_legal_business_suffix(value: str) -> bool:
    return bool(VENDOR_LEGAL_SUFFIX_RE.search(_searchable(value)))


def _vendor_text_before_embedded_label(line: OCRLine) -> str:
    prefix_tokens: list[NormalizedToken] = []
    for token in line.tokens:
        if _token_is_label(token):
            break
        prefix_tokens.append(token)
    if prefix_tokens and len(prefix_tokens) < len(line.tokens):
        return " ".join(token.original_text for token in prefix_tokens).strip()
    return line.text


def _clean_chinese_vendor_text(text: str) -> str:
    value = _normalize_ocr_text(text)
    value = re.sub(r"\s+(?=[\u3400-\u9fff])", "", value)
    value = re.sub(r"(?<=[\u3400-\u9fff])\s+", "", value)
    value = _strip_chinese_vendor_label_prefix(value)
    if any(value.startswith(prefix) for prefix in CHINESE_VENDOR_BAD_PREFIXES):
        return ""
    for suffix in CHINESE_VENDOR_REMOVE_SUFFIXES:
        value = value.replace(suffix, "")
    value = re.sub(r"\s+(?=[\u3400-\u9fff])", "", value)
    value = re.sub(r"(?<=[\u3400-\u9fff])\s+", "", value)
    value = _cut_chinese_vendor_after_stop(value)
    return _collapse_repeated_cjk_text(value.strip(" :：,，;；#-"))


def _chinese_vendor_text_after_label(text: str) -> str:
    normalized = _normalize_ocr_text(text)
    for label in CHINESE_VENDOR_LABEL_ALIASES:
        if label in normalized:
            return _extract_chinese_vendor_core(normalized.split(label, 1)[1])
    return ""


def _chinese_vendor_line_is_valid(text: str) -> bool:
    value = _extract_chinese_vendor_core(text)
    if len(value) < 3 or not _contains_cjk(value):
        return False
    if any(fragment in value for fragment in CHINESE_VENDOR_NEGATIVE_FRAGMENTS):
        return False
    is_allowed_issuer = any(fragment in value for fragment in CHINESE_VENDOR_ALLOWED_ISSUER_FRAGMENTS)
    if not is_allowed_issuer and any(fragment in value for fragment in CHINESE_VENDOR_EXCLUDE_FRAGMENTS):
        return False
    if any(keyword in value for keyword in CHINESE_VENDOR_POSITIVE_KEYWORDS):
        digit_ratio = len(re.findall(r"\d", value)) / max(len(value), 1)
        return digit_ratio <= 0.35
    if re.fullmatch(r"[\d\s:：,，;；#.\-A-Za-z]+", value):
        return False
    digit_ratio = len(re.findall(r"\d", value)) / max(len(value), 1)
    return digit_ratio <= 0.35


def _has_substantive_chinese_company_name(value: str) -> bool:
    if not value:
        return False
    if CHINESE_TAX_BUREAU_RE.search(value) or "\u51fa\u79df\u8f66" in value or "\u9152\u5e97" in value:
        return True
    for suffix in ("\u6709\u9650\u516c\u53f8", "\u516c\u53f8"):
        idx = value.find(suffix)
        if idx <= 0:
            continue
        prefix = re.sub(r"[^\u3400-\u9fff]", "", value[:idx])
        if len(prefix) >= 2:
            return True
    return False


def _chinese_vendor_has_company_shape(text: str) -> bool:
    value = _extract_chinese_vendor_core(text)
    return bool(value and (_has_substantive_chinese_company_name(value) or any(
        keyword in value for keyword in ("\u7a0e\u52a1\u5c40", "\u51fa\u79df\u8f66", "\u9152\u5e97")
    )))


def _chinese_vendor_text_is_blocked(text: str) -> bool:
    value = _normalize_ocr_text(text)
    return any(fragment in value for fragment in (CHINESE_VENDOR_BUYER_LABELS + CHINESE_VENDOR_TABLE_LABELS))


def _line_tokens_sorted(lines: list[OCRLine]) -> list[NormalizedToken]:
    tokens: list[NormalizedToken] = []
    for line in lines:
        tokens.extend(line.tokens)
    return sorted(tokens, key=lambda token: (token.page, token.cy, token.x1))


def _token_text_for_vendor(token: NormalizedToken) -> str:
    return _normalize_ocr_text(token.original_text or token.text)


def _candidate_from_chinese_seller_span(
    tokens: list[NormalizedToken],
    label_token: NormalizedToken,
    reason: str,
) -> FieldCandidate | None:
    if not tokens:
        return None
    source_text = "".join(_token_text_for_vendor(token) for token in tokens)
    if _chinese_vendor_text_is_blocked(source_text):
        return None
    value = _extract_chinese_vendor_core(source_text)
    if not value or not _chinese_vendor_line_is_valid(value):
        return None
    if not _chinese_vendor_has_company_shape(value):
        return None
    if len(re.findall(r"[\u3400-\u9fff]", value)) < 4:
        return None
    bbox = [
        min(token.x1 for token in tokens),
        min(token.y1 for token in tokens),
        max(token.x2 for token in tokens),
        max(token.y2 for token in tokens),
    ]
    distance = abs(mean(token.cy for token in tokens) - label_token.cy) + max(0.0, min(token.x1 for token in tokens) - label_token.x2) / 20.0
    confidence = mean(token.confidence for token in tokens)
    score = 0.86 + min(confidence, 0.99) * 0.11
    if "\u6709\u9650\u516c\u53f8" in value or "\u516c\u53f8" in value:
        score += 0.04
    score -= min(distance / 400.0, 0.08)
    return FieldCandidate(
        field="vendor_name",
        value=value,
        score=score,
        source_text=source_text,
        source_bbox=bbox,
        method="chinese_vat_seller_block_candidate",
        explanation=f"Chinese VAT seller block near {reason}; company-like name selected from nearby OCR tokens",
    )


def _chinese_vat_seller_candidates(lines: list[OCRLine]) -> list[FieldCandidate]:
    candidates: list[FieldCandidate] = []
    if not _looks_like_chinese_fapiao_profile(lines) or _looks_like_chinese_retail_receipt_profile(lines):
        return candidates
    tokens = _line_tokens_sorted(lines)
    if not tokens:
        return candidates

    for label_token in tokens:
        label_text = _token_text_for_vendor(label_token)
        if not any(label in label_text for label in CHINESE_VENDOR_SELLER_LABELS):
            continue

        nearby = [
            token
            for token in tokens
            if token.page == label_token.page
            and token.confidence >= 0.20
            and _contains_cjk(_token_text_for_vendor(token))
            and (label_token.cy - 85) <= token.cy <= (label_token.cy + 90)
            and token.x1 >= (label_token.x1 - 12)
        ]
        nearby = [
            token for token in nearby
            if not any(fragment in _token_text_for_vendor(token) for fragment in (
                CHINESE_VENDOR_SELLER_LABELS
                + CHINESE_VENDOR_BUYER_LABELS
                + CHINESE_VENDOR_TABLE_LABELS
                + CHINESE_VENDOR_STOP_MARKERS
            ))
        ]

        row_groups: list[list[NormalizedToken]] = []
        for token in sorted(nearby, key=lambda item: (item.cy, item.x1)):
            if not row_groups or abs(mean(item.cy for item in row_groups[-1]) - token.cy) > 28:
                row_groups.append([token])
            else:
                row_groups[-1].append(token)

        for group in row_groups:
            group = sorted(group, key=lambda item: item.x1)
            for index in range(len(group)):
                span: list[NormalizedToken] = []
                previous_x2: float | None = None
                for token in group[index:index + 5]:
                    if previous_x2 is not None and token.x1 - previous_x2 > 140:
                        break
                    span.append(token)
                    previous_x2 = token.x2
                    candidate = _candidate_from_chinese_seller_span(span, label_token, "seller label")
                    if candidate:
                        candidates.append(candidate)

    # Fallback for OCR engines that merge the whole seller area into a single line:
    # split after the seller label, then select a company-like core from the suffix.
    for line in lines:
        normalized = _normalize_ocr_text(line.text)
        if not any(label in normalized for label in CHINESE_VENDOR_SELLER_LABELS):
            continue
        seller_part = normalized
        for label in CHINESE_VENDOR_SELLER_LABELS:
            if label in seller_part:
                seller_part = seller_part.split(label, 1)[1]
                break
        value = _extract_chinese_vendor_core(seller_part)
        if value and _chinese_vendor_line_is_valid(value) and _chinese_vendor_has_company_shape(value):
            candidates.append(
                FieldCandidate(
                    field="vendor_name",
                    value=value,
                    score=0.80 + line.confidence * 0.12,
                    source_text=line.text,
                    source_bbox=line.bbox,
                    method="chinese_vat_seller_line_candidate",
                    explanation="Chinese VAT seller label found in OCR line; company-like suffix selected",
                )
            )
    return candidates


def _chinese_vendor_candidates(lines: list[OCRLine]) -> list[FieldCandidate]:
    candidates: list[FieldCandidate] = []
    if not lines:
        return candidates
    min_y, _, doc_height = _document_bounds(lines)

    for line in sorted(lines, key=lambda item: item.cy):
        if line.confidence < 0.20 or not _contains_cjk(line.text):
            continue
        y = (line.cy - min_y) / doc_height

        explicit_value = _chinese_vendor_text_after_label(line.text)
        if explicit_value and _chinese_vendor_line_is_valid(explicit_value):
            candidates.append(
                FieldCandidate(
                    field="vendor_name",
                    value=explicit_value,
                    score=0.70 + line.confidence * 0.16,
                    source_text=line.text,
                    source_bbox=line.bbox,
                    method="chinese_vendor_label_candidate",
                    explanation="Chinese vendor/issuer label followed by a business name",
                )
            )

        value = _extract_chinese_vendor_core(line.text)
        if not _chinese_vendor_line_is_valid(value):
            continue
        has_business_keyword = (
            any(keyword in value for keyword in CHINESE_BUSINESS_KEYWORDS)
            or any(keyword in value for keyword in CHINESE_VENDOR_POSITIVE_KEYWORDS)
        )
        if not has_business_keyword and y > 0.32:
            continue
        if not has_business_keyword and y > 0.20:
            continue

        score = 0.34 + line.confidence * 0.20
        reasons = [f"Chinese top/layout line at {y:.2f}", f"OCR confidence {line.confidence:.2f}"]
        if y <= 0.32:
            score += 0.12
            reasons.append("inside upper document region")
        if has_business_keyword:
            score += 0.26
            reasons.append("Chinese business/issuer keyword detected")

        candidates.append(
            FieldCandidate(
                field="vendor_name",
                value=value,
                score=score,
                source_text=line.text,
                source_bbox=line.bbox,
                method="chinese_vendor_layout_candidate",
                explanation="; ".join(reasons),
            )
        )
    return candidates


def _vendor_candidates(lines: list[OCRLine]) -> list[FieldCandidate]:
    candidates: list[FieldCandidate] = []
    if not lines:
        return candidates
    min_y, _, doc_height = _document_bounds(lines)
    # Sort top-to-bottom so "first valid" is reliably the geometrically topmost.
    sorted_lines = sorted(lines, key=lambda l: l.cy)
    top_lines = [
        ln for ln in sorted_lines
        if (ln.cy - min_y) / doc_height <= 0.32
    ]
    chinese_retail_table_y = None
    if _looks_like_chinese_retail_receipt_profile(lines):
        for ln in sorted_lines:
            if any(fragment in _normalize_ocr_text(ln.text) for fragment in ("商品信息", "品名")):
                chinese_retail_table_y = ln.cy
                break
    first_valid_seen = False

    for idx, line in enumerate(top_lines):
        if chinese_retail_table_y is not None and line.cy >= chinese_retail_table_y:
            continue
        if line.confidence < _VENDOR_MIN_CONFIDENCE:
            continue
        top_ratio = (line.cy - min_y) / doc_height
        raw_vendor_text = _vendor_text_before_embedded_label(line)
        value = _clean_vendor_text(raw_vendor_text)
        if not _vendor_line_is_valid(value):
            continue

        assembled = value
        joined_lines = [line]

        for j in range(idx + 1, min(idx + 3, len(top_lines))):
            nxt = top_lines[j]
            nxt_value = _clean_vendor_text(nxt.text)
            if not nxt_value:
                continue
            # Continue only when the next line still looks like a business name.
            ends_amp = assembled.rstrip().endswith("&")
            nxt_has_suffix = bool(BUSINESS_SUFFIX_RE.search(nxt_value))
            nxt_has_reg = bool(_REG_NUM_RE.search(nxt_value))
            if not (ends_amp or nxt_has_suffix or nxt_has_reg):
                break
            nxt_search = _searchable(nxt_value)
            if _has_vendor_address_keyword(nxt_search) and not _has_legal_business_suffix(nxt_search):
                break
            assembled = (assembled.rstrip(" &") + " " + nxt_value.lstrip()).strip()
            joined_lines.append(nxt)
        score = 0.30 + line.confidence * 0.25
        reasons = [f"top-region line at {top_ratio:.2f}", f"OCR confidence {line.confidence:.2f}"]
        if top_ratio <= 0.25:
            score += 0.12
            reasons.append("inside top 25%")
        if not first_valid_seen:
            score += 0.15
            reasons.append("topmost valid vendor line")
            first_valid_seen = True
        if BUSINESS_SUFFIX_RE.search(assembled):
            score += 0.28
            reasons.append("business suffix detected")
        if len(joined_lines) > 1:
            score += 0.12 * (len(joined_lines) - 1)
            reasons.append(f"multi-line assembly ({len(joined_lines)} lines)")
        if len(re.findall(r"[A-Z]", assembled)) >= 6:
            score += 0.06
            reasons.append("business-name-like letter content")

        if len(joined_lines) > 1:
            all_bboxes = [ln.bbox for ln in joined_lines]
            source_bbox: list[float] = [
                min(b[0] for b in all_bboxes), min(b[1] for b in all_bboxes),
                max(b[2] for b in all_bboxes), max(b[3] for b in all_bboxes),
            ]
            source_text = " | ".join(ln.text for ln in joined_lines)
        else:
            source_bbox = list(line.bbox)
            source_text = raw_vendor_text

        candidates.append(
            FieldCandidate(
                field="vendor_name",
                value=assembled,
                score=score,
                source_text=source_text,
                source_bbox=source_bbox,
                method="top_region_vendor_candidate",
                explanation="; ".join(reasons),
            )
        )
    return candidates


def _global_date_candidates(lines: list[OCRLine]) -> list[FieldCandidate]:
    candidates: list[FieldCandidate] = []
    for line in lines:
        if _has_negative_anchor(line.search_text, "date"):
            continue
        y = _y_ratio(line.bbox, lines)
        header_region = y <= 0.35
        footer_region = y >= 0.80  # receipts sometimes print a date stamp at the very bottom
        has_date_label = any(_line_has_alias(line, alias) for alias, _ in LABEL_ALIASES["date"])
        if not has_date_label and not header_region and not footer_region:
            continue
        for value, raw in _extract_dates(line.text, repair_invalid=has_date_label or header_region):
            score = 0.48 + line.confidence * 0.18
            explanation = f"date-like value '{raw}' found with OCR correction"
            if has_date_label:
                score += 0.14
                explanation += " on a Date line"
            elif header_region:
                score += 0.08
                explanation += " in receipt header area"
            else:
                score += 0.04
                explanation += " in receipt footer area (fallback)"
            candidates.append(
                FieldCandidate(
                    field="date",
                    value=value,
                    score=score,
                    source_text=line.text,
                    source_bbox=line.bbox,
                    method="date_line_candidate",
                    explanation=explanation,
                )
            )
    return candidates


def _global_invoice_id_candidates(lines: list[OCRLine]) -> list[FieldCandidate]:
    candidates: list[FieldCandidate] = []
    for line in lines:
        canonical = _canonical_compact(line.text)
        if "GST" in canonical or "SST" in canonical or "REGISTRATION" in canonical:
            continue
        if _has_negative_anchor(line.search_text, "invoice_number"):
            continue
        if not any(_line_has_alias(line, alias) for alias, _ in LABEL_ALIASES["invoice_number"]):
            continue
        for value in _extract_invoice_ids_near_label(line.text):
            if not _valid_invoice_value(value):
                continue
            candidates.append(
                FieldCandidate(
                    field="invoice_number",
                    value=value,
                    score=0.58 + line.confidence * 0.18,
                    source_text=line.text,
                    source_bbox=line.bbox,
                    method="invoice_id_line_candidate",
                    explanation="alphanumeric identifier found on document/receipt/invoice number line",
                )
            )
    return candidates


def _digit_runs(text: str) -> list[str]:
    return re.findall(r"\d{6,}", str(text or ""))


def _has_any_fragment(text: str, fragments: tuple[str, ...]) -> bool:
    return any(fragment in (text or "") for fragment in fragments)


def _looks_like_chinese_fapiao_profile(lines: list[OCRLine]) -> bool:
    if not lines:
        return False
    has_header_code = False
    has_number_label = False
    has_fapiao_label = False
    has_valid_date = False
    has_amount = False
    has_issuer_marker = False
    has_top_invoice_number = False
    clear_latin_labels = 0
    for line in lines:
        y = _y_ratio(line.bbox, lines)
        text = _normalize_ocr_text(line.text)
        digit_runs = _digit_runs(line.text)
        if y <= 0.45 and any(len(run) >= 10 for run in digit_runs):
            has_header_code = True
        if y <= 0.45 and any(7 <= len(run) <= 8 for run in digit_runs):
            has_top_invoice_number = True
        if "发票代码" in text:
            has_header_code = True
            has_fapiao_label = True
        if "发票号码" in text or "发票号" in text:
            has_number_label = True
            has_fapiao_label = True
        if _has_any_fragment(text, ("发票", "税务局", "发票联", "机打发票")):
            has_fapiao_label = True
        if _has_any_fragment(text, ("出租车", "公司", "有限", "运输", "服务")):
            has_issuer_marker = True
        if re.search(r"\b[A-Z]{1,3}-\d{3,5}\b", line.text, re.I):
            has_issuer_marker = True
        if _extract_dates(line.text):
            has_valid_date = True
        if re.search(r"\b(invoice|receipt|total|subtotal|tax|vendor|date)\b", line.text, re.I):
            clear_latin_labels += 1
        if _detect_currency(line.text) == "CNY":
            has_amount = True
        if y >= 0.35 and any(_to_decimal(value) is not None for value, _ in _extract_amounts_with_repair(line.text)):
            has_amount = True
    labeled_profile = has_fapiao_label and (has_header_code or has_number_label) and (has_amount or has_issuer_marker)
    noisy_profile = has_header_code and has_valid_date and has_amount and has_issuer_marker and clear_latin_labels <= 2
    vatid_numeric_profile = has_header_code and has_top_invoice_number and has_amount and clear_latin_labels <= 2
    return labeled_profile or noisy_profile or vatid_numeric_profile


_TAXI_ONE_DECIMAL_DISTANCE_RE = re.compile(r"(?<!\d)(?:[1-9]\d?|0?[1-9])\.\d\s*(?:K|KM)?(?!\d)", re.I)


def _looks_like_chinese_taxi_profile(lines: list[OCRLine]) -> bool:
    if not _looks_like_chinese_fapiao_profile(lines):
        return False
    combined = " ".join(line.text for line in lines)
    if any(fragment in combined for fragment in ("出租车", "车号", "上车", "下车", "里程", "等候")):
        return True
    if any("KM" in line.text.upper() for line in lines):
        return True
    for line in lines:
        values = [
            amount
            for value, _ in _extract_amounts_with_repair(line.text, allow_integer=False)
            if (amount := _to_decimal(value)) is not None
        ]
        positive_values = [amount for amount in values if amount > Decimal("0")]
        has_zero = any(amount == Decimal("0") for amount in values)
        has_distance_like = (
            any(Decimal("1.0") <= amount <= Decimal("99.9") for amount in positive_values[1:])
            or bool(_TAXI_ONE_DECIMAL_DISTANCE_RE.search(line.text))
        )
        if positive_values and has_zero and has_distance_like:
            return True
    return False


def _chinese_vat_amount_items(lines: list[OCRLine]) -> list[tuple[Decimal, str, str, OCRLine]]:
    items: list[tuple[Decimal, str, str, OCRLine]] = []
    seen: set[tuple[str, str]] = set()
    for line in lines:
        if _y_ratio(line.bbox, lines) < 0.12:
            continue
        if re.search(r"\b\d{1,2}:\d{2}\b", line.text):
            continue
        for value, raw in _extract_amounts_with_repair(line.text, allow_integer=False):
            amount = _to_decimal(value)
            if amount is None or amount <= Decimal("0"):
                continue
            if amount < Decimal("1.00"):
                continue
            key = (value, line.text)
            if key in seen:
                continue
            seen.add(key)
            items.append((amount, value, raw, line))
    return items


def _tax_ratio_ok(subtotal: Decimal, tax: Decimal) -> bool:
    if subtotal <= Decimal("0") or tax < Decimal("0"):
        return False
    ratio = tax / subtotal
    return Decimal("0.01") <= ratio <= Decimal("0.25")


def _find_chinese_vat_assignment(
    lines: list[OCRLine],
) -> tuple[tuple[Decimal, str, str, OCRLine], tuple[Decimal, str, str, OCRLine], tuple[Decimal, str, str, OCRLine] | None] | None:
    items = _chinese_vat_amount_items(lines)
    if len(items) < 2:
        return None

    best_triplet = None
    best_triplet_score = Decimal("-1")
    for total_item in items:
        total = total_item[0]
        for sub_source in items:
            if sub_source is total_item:
                continue
            for tax_source in items:
                if tax_source is total_item or tax_source is sub_source:
                    continue
                sub_item, tax_item = sub_source, tax_source
                subtotal, tax = sub_item[0], tax_item[0]
                if subtotal < tax:
                    subtotal, tax = tax, subtotal
                    sub_item, tax_item = tax_item, sub_item
                if not _tax_ratio_ok(subtotal, tax):
                    continue
                if abs(subtotal + tax - total) > Decimal("0.05"):
                    continue
                score = total + Decimal(str(total_item[3].confidence)) + Decimal("1000")
                if score > best_triplet_score:
                    best_triplet_score = score
                    best_triplet = (sub_item, tax_item, total_item)
    if best_triplet:
        return best_triplet

    best_pair = None
    best_pair_score = Decimal("-1")
    for left in items:
        for right in items:
            if left is right:
                continue
            subtotal_item, tax_item = (left, right) if left[0] >= right[0] else (right, left)
            subtotal, tax = subtotal_item[0], tax_item[0]
            if not _tax_ratio_ok(subtotal, tax):
                continue
            score = subtotal + tax
            # Prefer pairs close to each other in the VAT amount/tax table.
            if abs(subtotal_item[3].cy - tax_item[3].cy) < max(subtotal_item[3].height, tax_item[3].height) * 4:
                score += Decimal("500")
            if score > best_pair_score:
                best_pair_score = score
                best_pair = (subtotal_item, tax_item, None)
    return best_pair


def _noisy_chinese_vat_dates(text: str) -> list[str]:
    """Recover dates from VAT OCR fragments like '20175F04 H18H' or '2017f058oz8'."""
    raw = _normalize_ocr_text(text)
    out: list[str] = []
    confusables = str.maketrans({
        "O": "0", "o": "0",
        "Z": "2", "z": "2",
        "I": "1", "l": "1",
        "S": "5", "s": "5",
    })
    for match in re.finditer(r"20\d{2}", raw):
        year = int(match.group(0))
        tail = raw[match.end(): match.end() + 18].translate(confusables)
        digits = "".join(ch for ch in tail if ch.isdigit())
        for month_start in range(0, min(5, max(len(digits) - 3, 0))):
            month_text = digits[month_start: month_start + 2]
            if len(month_text) < 2:
                continue
            month = int(month_text)
            if not (1 <= month <= 12):
                continue
            for day_start in range(month_start + 2, min(month_start + 5, len(digits) - 1)):
                day_text = digits[day_start: day_start + 2]
                day = int(day_text)
                if 1 <= day <= 31:
                    try:
                        date(year, month, day)
                    except ValueError:
                        continue
                    out.append(f"{year:04d}-{month:02d}-{day:02d}")
                    return out
    return out


def _chinese_taxi_meter_amount_candidates(lines: list[OCRLine]) -> list[FieldCandidate]:
    candidates: list[FieldCandidate] = []
    if not _looks_like_chinese_taxi_profile(lines):
        return candidates
    for line in lines:
        if "-" in line.text:
            continue
        values = [
            (amount, value, raw)
            for value, raw in _extract_amounts_with_repair(line.text, allow_integer=False)
            if (amount := _to_decimal(value)) is not None
        ]
        positive_values = [(amount, value, raw) for amount, value, raw in values if amount > Decimal("0")]
        has_zero = any(amount == Decimal("0") for amount, _, _ in values)
        has_distance_like = (
            any(Decimal("1.0") <= amount <= Decimal("99.9") for amount, _, _ in positive_values[1:])
            or bool(_TAXI_ONE_DECIMAL_DISTANCE_RE.search(line.text))
        )
        if not positive_values or not has_zero or not has_distance_like:
            continue
        amount, value, raw = positive_values[0]
        if not (Decimal("3.00") <= amount <= Decimal("500.00")):
            continue
        candidates.append(
            FieldCandidate(
                field="total_amount",
                value=value,
                score=0.80 + line.confidence * 0.10,
                source_text=line.text,
                source_bbox=_amount_bbox_in_line(line, raw),
                method="chinese_taxi_meter_fare_candidate",
                explanation="taxi meter row: first positive fare amount before zero/distance-like values",
                amount_role="total_amount",
            )
        )
    return candidates


def _looks_like_chinese_retail_receipt_profile(lines: list[OCRLine]) -> bool:
    if not lines:
        return False
    combined = " ".join(_normalize_ocr_text(line.text) for line in lines)
    if any(fragment in combined for fragment in ("发票代码", "发票号码", "税务局", "发票联")):
        return False
    has_receipt_id = any(fragment in combined for fragment in ("单据号", "小票", "流水号", "收款时间"))
    has_payment = any(fragment in combined for fragment in ("应收", "应付", "实收", "实付", "找零", "支付宝", "微信", "现金"))
    has_retail_body = any(fragment in combined for fragment in ("商品信息", "品名", "订单原价", "件数", "数量", "会员", "门店"))
    return has_payment and (has_receipt_id or has_retail_body)


def _amounts_after_chinese_label(line: OCRLine, label: str) -> list[tuple[str, str]]:
    text = _normalize_ocr_text(line.text)
    index = text.find(label)
    if index < 0:
        return []
    after = text[index + len(label):].strip(" :：#-")
    stop_positions = [
        pos
        for stop_label in _INLINE_CJK_STOP_LABELS
        if stop_label != label and (pos := after.find(stop_label)) > 0
    ]
    if stop_positions:
        after = after[: min(stop_positions)]
    return _extract_amounts_with_repair(after, allow_integer=False)


def _chinese_retail_receipt_candidates(lines: list[OCRLine]) -> dict[str, list[FieldCandidate]]:
    candidates: dict[str, list[FieldCandidate]] = {
        "total_amount": [],
        "subtotal": [],
        "currency": [],
    }
    if not _looks_like_chinese_retail_receipt_profile(lines):
        return candidates

    total_labels = (
        ("应收", 0.91, "Chinese retail payable/amount-due label"),
        ("应付", 0.90, "Chinese retail payable/amount-due label"),
        ("实付", 0.82, "Chinese retail paid amount label"),
        ("收款金额", 0.84, "Chinese retail collected amount label"),
        ("实收", 0.76, "Chinese retail received-payment fallback"),
        ("支付宝", 0.68, "Chinese retail payment-method fallback"),
        ("微信", 0.68, "Chinese retail payment-method fallback"),
        ("现金", 0.66, "Chinese retail payment-method fallback"),
    )
    subtotal_labels = (
        ("订单原价", 0.72, "Chinese retail original order price"),
        ("商品合计", 0.72, "Chinese retail item subtotal"),
        ("原价合计", 0.70, "Chinese retail original-price subtotal"),
    )

    for line in lines:
        for label, base_score, reason in total_labels:
            for value, raw in _amounts_after_chinese_label(line, label):
                if not _valid_amount_value(value, "total_amount"):
                    continue
                amount_confidence = _amount_confidence_in_line(line, raw)
                candidates["total_amount"].append(
                    FieldCandidate(
                        field="total_amount",
                        value=value,
                        score=base_score + amount_confidence * 0.08,
                        source_text=line.text,
                        source_bbox=_amount_bbox_in_line(line, raw),
                        method="chinese_retail_payment_total",
                        explanation=f"{reason}: label '{label}' produced payable value '{raw}'",
                        amount_role="total_amount",
                    )
                )
        for label, base_score, reason in subtotal_labels:
            for value, raw in _amounts_after_chinese_label(line, label):
                if not _valid_amount_value(value, "subtotal"):
                    continue
                amount_confidence = _amount_confidence_in_line(line, raw)
                candidates["subtotal"].append(
                    FieldCandidate(
                        field="subtotal",
                        value=value,
                        score=base_score + amount_confidence * 0.08,
                        source_text=line.text,
                        source_bbox=_amount_bbox_in_line(line, raw),
                        method="chinese_retail_subtotal",
                        explanation=f"{reason}: label '{label}' produced subtotal value '{raw}'",
                        amount_role="subtotal",
                    )
                )

    if not any(candidate.value == "CNY" for candidate in candidates["currency"]):
        source = next((line for line in lines if _contains_cjk(line.text)), lines[0])
        candidates["currency"].append(
            FieldCandidate(
                field="currency",
                value="CNY",
                score=0.84,
                source_text=source.text,
                source_bbox=source.bbox,
                method="chinese_retail_currency_prior",
                explanation="Chinese retail receipt profile implies CNY/RMB currency",
            )
        )
    return candidates


def _chinese_fapiao_candidates(lines: list[OCRLine]) -> dict[str, list[FieldCandidate]]:
    candidates: dict[str, list[FieldCandidate]] = {
        "invoice_number": [],
        "date": [],
        "total_amount": [],
        "currency": [],
        "subtotal": [],
        "tax_amount": [],
        "amount_in_words": [],
    }
    if not _looks_like_chinese_fapiao_profile(lines):
        return candidates
    if _looks_like_chinese_retail_receipt_profile(lines):
        return candidates

    for line in lines:
        text = _normalize_ocr_text(line.text)
        if "发票代码" in text and "发票号码" not in text:
            continue
        if "发票号码" in text or "发票号" in text:
            after = _text_after_alias(text, "发票号码") or _text_after_alias(text, "发票号")
            for value in _extract_invoice_ids_near_label(after or text):
                if value == "0" * len(value) or looks_like_date(value):
                    continue
                candidates["invoice_number"].append(
                    FieldCandidate(
                        field="invoice_number",
                        value=value,
                        score=0.78 + line.confidence * 0.14,
                        source_text=line.text,
                        source_bbox=line.bbox,
                        method="chinese_fapiao_number_label",
                        explanation="Chinese fapiao invoice-number label followed by a numeric identifier",
                    )
                )
                break

    code_seen = False
    for line in sorted(lines, key=lambda item: item.cy):
        if _y_ratio(line.bbox, lines) > 0.45:
            continue
        if "发票代码" in line.text and "发票号码" not in line.text:
            code_seen = True
            continue
        code_seen_before_line = code_seen
        long_code_on_line = any(len(run) >= 10 for run in _digit_runs(line.text))
        date_like_on_line = bool(_extract_dates(line.text, repair_invalid=True) or _noisy_chinese_vat_dates(line.text))
        for run in _digit_runs(line.text):
            if not code_seen and len(run) >= 10:
                code_seen = True
                continue
            if len(run) == 8:
                value = run
            elif code_seen and 9 <= len(run) <= 14:
                value = run[-8:]
            else:
                continue
            if value == "00000000" or looks_like_date(value):
                continue
            values = [value]
            if value.startswith("0") and len(value) == 8 and value[1] != "0":
                values.insert(0, value[1:])
            for candidate_value in values:
                score = 0.64 + line.confidence * 0.06
                if code_seen_before_line:
                    score += 0.10
                if date_like_on_line:
                    score += 0.06
                if long_code_on_line and not code_seen_before_line:
                    score -= 0.08
                if candidate_value != value:
                    score += 0.04
                candidates["invoice_number"].append(
                    FieldCandidate(
                        field="invoice_number",
                        value=candidate_value,
                        score=score,
                        source_text=line.text,
                        source_bbox=line.bbox,
                        method="chinese_fapiao_prior",
                        explanation=(
                            "Chinese fapiao/VAT layout: invoice number recovered "
                            "from top numeric invoice-number region"
                        ),
                    )
                )
            if len(run) >= 10:
                code_seen = True

    for line in lines:
        if not any(label in line.text for label in ("日期", "开票日期")):
            continue
        for value, raw in _extract_dates(line.text, repair_invalid=True):
            candidates["date"].append(
                FieldCandidate(
                    field="date",
                    value=value,
                    score=0.62 + line.confidence * 0.16,
                    source_text=line.text,
                    source_bbox=line.bbox,
                    method="chinese_fapiao_date_label",
                    explanation=f"Chinese date label with OCR repair produced '{value}' from '{raw}'",
                )
            )
            break

    if not candidates["date"]:
        for line in lines:
            for value in _noisy_chinese_vat_dates(line.text):
                candidates["date"].append(
                    FieldCandidate(
                        field="date",
                        value=value,
                        score=0.58 + line.confidence * 0.10,
                        source_text=line.text,
                        source_bbox=line.bbox,
                        method="chinese_vat_noisy_date",
                        explanation="Chinese VAT date recovered from noisy year/month/day OCR fragment",
                    )
                )
                break
            if candidates["date"]:
                break

    taxi_profile = _looks_like_chinese_taxi_profile(lines)
    vat_assignment = None if taxi_profile else _find_chinese_vat_assignment(lines)
    if vat_assignment:
        subtotal_item, tax_item, total_item = vat_assignment
        subtotal, tax = subtotal_item[0], tax_item[0]
        total = total_item[0] if total_item else subtotal + tax
        amount_fields = [
            ("subtotal", subtotal_item, subtotal, "Chinese VAT subtotal/amount column"),
            ("tax_amount", tax_item, tax, "Chinese VAT tax amount column"),
        ]
        if total_item:
            amount_fields.append(("total_amount", total_item, total, "Chinese VAT total observed as subtotal + tax"))
        else:
            amount_fields.append(("total_amount", subtotal_item, total, "Chinese VAT total computed from subtotal + tax"))
        for field_name, item, amount, reason in amount_fields:
            value = format(amount.quantize(Decimal("0.01")), "f")
            candidates[field_name].append(
                FieldCandidate(
                    field=field_name,
                    value=value,
                    score=0.86 + min(item[3].confidence, 0.80) * 0.10,
                    source_text=item[3].text,
                    source_bbox=_amount_bbox_in_line(item[3], item[2]),
                    method="chinese_vat_amount_structure",
                    explanation=reason,
                    amount_role=field_name,
                )
            )

    labeled_amounts: list[tuple[Decimal, str, str, OCRLine]] = []
    for line in lines:
        text = _normalize_ocr_text(line.text)
        if not any(label in text for label in ("价税合计", "小写金额", "应付金额", "合计金额", "金额")):
            continue
        if any(label in text for label in ("单价", "里程", "等候")):
            continue
        for value, raw in _extract_amounts(line.text):
            amount = _to_decimal(value)
            if amount is not None and amount > Decimal("0"):
                labeled_amounts.append((amount, value, raw, line))

    for _, value, raw, line in labeled_amounts:
        amount_confidence = _amount_confidence_in_line(line, raw)
        candidates["total_amount"].append(
            FieldCandidate(
                field="total_amount",
                value=value,
                score=0.38 + amount_confidence * 0.32,
                source_text=line.text,
                source_bbox=_amount_bbox_in_line(line, raw),
                method="chinese_fapiao_amount_label",
                explanation=(
                    f"Chinese amount/payable label produced amount '{raw}'; "
                    f"value OCR confidence {amount_confidence:.2f}"
                ),
                amount_role="total_amount",
            )
        )

    chinese_money_words = re.compile(r"[零壹贰貳叁參肆伍陆陸柒捌玖拾佰仟万萬亿億元圆圓角分整]+")
    for line in lines:
        if not _contains_cjk(line.text):
            continue
        if "大写" not in line.text and "圆" not in line.text and "元" not in line.text:
            continue
        match = chinese_money_words.search(line.text)
        if match and len(match.group(0)) >= 3:
            candidates["amount_in_words"].append(
                FieldCandidate(
                    field="amount_in_words",
                    value=match.group(0),
                    score=0.70 + line.confidence * 0.12,
                    source_text=line.text,
                    source_bbox=line.bbox,
                    method="chinese_amount_in_words",
                    explanation="Chinese uppercase amount-in-words detected",
                )
            )

    candidates["total_amount"].extend(_chinese_taxi_meter_amount_candidates(lines))

    best_amount: tuple[Decimal, str, str, OCRLine] | None = None
    for line in lines:
        text_upper = line.text.upper()
        if re.search(r"\b\d{1,2}:\d{2}\b", line.text) or "K" in text_upper:
            continue
        if any(label in line.text for label in ("单价", "里程", "等候")):
            continue
        for value, raw in _extract_amounts(line.text):
            amount = _to_decimal(value)
            if amount is None or amount <= Decimal("0"):
                continue
            if best_amount is None or amount > best_amount[0]:
                best_amount = (amount, value, raw, line)

    if best_amount is not None and not candidates["total_amount"]:
        _, value, raw, line = best_amount
        candidates["total_amount"].append(
            FieldCandidate(
                field="total_amount",
                value=value,
                score=0.62 + line.confidence * 0.06,
                source_text=line.text,
                source_bbox=_amount_bbox_in_line(line, raw),
                method="chinese_fapiao_prior",
                explanation=(
                    f"Chinese fapiao/taxi layout: selected largest fare-like "
                    f"amount '{raw}' after excluding time and distance rows"
                ),
                amount_role="total_amount",
            )
        )

    currency_source = None
    for line in lines:
        if _detect_currency(line.text) == "CNY":
            currency_source = line
            break
    if currency_source is None and candidates["total_amount"]:
        currency_source = lines[0] if lines else None
    if currency_source is not None and not candidates["currency"]:
        candidates["currency"].append(
            FieldCandidate(
                field="currency",
                value="CNY",
                score=0.82 + (currency_source.confidence * 0.08 if _detect_currency(currency_source.text) == "CNY" else 0.0),
                source_text=currency_source.text,
                source_bbox=[],
                method="chinese_fapiao_currency_prior",
                explanation="Chinese fapiao/taxi receipt profile implies CNY currency",
            )
        )

    if vat_assignment and not any(candidate.value == "CNY" for candidate in candidates["currency"]):
        source_line = vat_assignment[0][3]
        candidates["currency"].append(
            FieldCandidate(
                field="currency",
                value="CNY",
                score=0.84,
                source_text=source_line.text,
                source_bbox=[],
                method="chinese_vat_currency_prior",
                explanation="Chinese VAT numeric invoice profile implies CNY/RMB currency",
            )
        )

    return candidates


def _format_prior_candidates(lines: list[OCRLine]) -> dict[str, list[FieldCandidate]]:
    """Scan ALL lines for value-shaped tokens without needing a label match.

    Emits low-priority candidates (method="format_prior") so that labeled
    extraction still wins when available, but non-Latin / label-destroyed
    documents still get useful values from high-confidence formatted tokens.

    Fields covered:
      date         — ISO / DD-Mon-YYYY / DD/MM/YYYY patterns anywhere
      total_amount — decimal adjacent to a currency marker, or the largest
                     standalone decimal not in a payment/time/km/% context
      invoice_number — ID-shaped token in the header region with no readable label
    """
    candidates: dict[str, list[FieldCandidate]] = {
        "date": [], "total_amount": [], "invoice_number": [],
    }
    if not lines:
        return candidates

    _time_re = re.compile(r"\b\d{1,2}:\d{2}\b")
    _km_re = re.compile(r"\d+\.\d+\s*km", re.I)
    _pct_re = re.compile(r"\d+\s*%")

    min_y, max_y, _ = _document_bounds(lines)
    header_cy_limit = min_y + (max_y - min_y) * 0.35

    for line in lines:
        y = _y_ratio(line.bbox, lines)
        header_region = y <= 0.35
        footer_region = y >= 0.80
        for value, raw in _extract_dates(line.text):
            score = 0.45 + line.confidence * 0.10
            if header_region:
                score += 0.05
            elif footer_region:
                score += 0.02
            candidates["date"].append(
                FieldCandidate(
                    field="date",
                    value=value,
                    score=score,
                    source_text=line.text,
                    source_bbox=line.bbox,
                    method="format_prior",
                    explanation=f"format-prior date '{raw}' (no label required)",
                )
            )

    # Prefer currency-adjacent totals, then the largest eligible amount.
    best_decimal: tuple[str, str, OCRLine, list[float]] | None = None
    best_decimal_val: Decimal | None = None
    currency_adjacent: list[tuple[str, str, OCRLine, list[float]]] = []

    for line in lines:
        if line.confidence < 0.20:
            continue
        canonical = _canonical_compact(line.text)
        if _is_payment_context(canonical):
            continue
        if _is_line_item_amount_context(line.text) or _is_adjustment_context(line.text):
            continue
        if _time_re.search(line.text) or _km_re.search(line.text) or _pct_re.search(line.text):
            continue
        has_total_label = any(_line_has_alias(line, alias) for alias, _ in LABEL_ALIASES["total_amount"])
        has_non_total_amount_label = any(
            _line_has_alias(line, alias)
            for field_name in ("subtotal", "tax_amount", "vat_rate")
            for alias, _ in LABEL_ALIASES[field_name]
        )
        if has_non_total_amount_label and not has_total_label:
            continue
        has_currency = bool(_detect_currency(line.text))
        for token in line.tokens:
            for value, raw in _extract_amounts_with_repair(token.text, allow_integer=False):
                dec = _to_decimal(value)
                if dec is None or dec <= Decimal("0"):
                    continue
                tok_bbox = list(token.bbox)
                if has_currency:
                    currency_adjacent.append((value, raw, line, tok_bbox))
                if best_decimal_val is None or dec > best_decimal_val:
                    best_decimal = (value, raw, line, tok_bbox)
                    best_decimal_val = dec

    chosen_totals: list[tuple[str, str, OCRLine, list[float], float]] = []
    for value, raw, line, tok_bbox in currency_adjacent:
        score = 0.40 + line.confidence * 0.08
        chosen_totals.append((value, raw, line, tok_bbox, score))

    if not chosen_totals and best_decimal is not None:
        value, raw, line, tok_bbox = best_decimal
        score = 0.38 + line.confidence * 0.06
        chosen_totals.append((value, raw, line, tok_bbox, score))

    for value, raw, line, tok_bbox, score in chosen_totals:
        candidates["total_amount"].append(
            FieldCandidate(
                field="total_amount",
                value=value,
                score=score,
                source_text=line.text,
                source_bbox=tok_bbox,
                method="format_prior",
                explanation=f"format-prior total_amount '{raw}' (no label required)",
                amount_role="total_amount",
            )
        )

    # Label-free invoice IDs are limited to the header.
    for line in lines:
        if line.cy > header_cy_limit:
            continue
        if line.confidence < 0.45:
            continue
        if _exclude_invoice_format_prior_line(line.text):
            continue
        if any(_line_has_alias(line, alias) for aliases in LABEL_ALIASES.values() for alias, _ in aliases):
            continue
        for value in _extract_invoice_ids(line.text):
            if looks_like_date(value):
                continue
            if not _valid_invoice_value(value):
                continue
            score = 0.40 + line.confidence * 0.08
            candidates["invoice_number"].append(
                FieldCandidate(
                    field="invoice_number",
                    value=value,
                    score=score,
                    source_text=line.text,
                    source_bbox=line.bbox,
                    method="format_prior",
                    explanation=f"format-prior invoice_number '{value}' in header (no label)",
                )
            )

    return candidates


def _generate_candidates(lines: list[OCRLine]) -> dict[str, list[FieldCandidate]]:
    candidates = _value_candidates_from_label(lines)
    gst_candidates = _gst_summary_candidates(lines)
    candidates["subtotal"].extend(gst_candidates["subtotal"])
    candidates["tax_amount"].extend(gst_candidates["tax_amount"])
    candidates["currency"].extend(_currency_candidates(lines))
    candidates["vendor_name"].extend(_vendor_candidates(lines))
    candidates["vendor_name"].extend(_chinese_vendor_candidates(lines))
    candidates["vendor_name"].extend(_chinese_vat_seller_candidates(lines))
    candidates["date"].extend(_global_date_candidates(lines))
    candidates["invoice_number"].extend(_global_invoice_id_candidates(lines))
    chinese_retail = _chinese_retail_receipt_candidates(lines)
    for field_name, fc_list in chinese_retail.items():
        candidates[field_name].extend(fc_list)
    chinese_fapiao = _chinese_fapiao_candidates(lines)
    for field_name, fc_list in chinese_fapiao.items():
        candidates[field_name].extend(fc_list)
    candidates["total_amount"].extend(_merged_payment_total_candidates(lines))
    _add_financial_consistency_candidates(lines, candidates)
    # Low-score fallback for documents without readable labels.
    format_prior = _format_prior_candidates(lines)
    for field_name, fc_list in format_prior.items():
        candidates[field_name].extend(fc_list)
    return candidates


def _candidate_y_ratio(candidate: FieldCandidate, lines: list[OCRLine]) -> float:
    if candidate.source_bbox and len(candidate.source_bbox) >= 4:
        return _y_ratio(candidate.source_bbox, lines)
    return 0.50


def _route_note(candidate: FieldCandidate, delta: float, note: str) -> None:
    candidate.score += delta
    sign = "+" if delta >= 0 else ""
    candidate.explanation += f"; category routing {sign}{delta:.2f}: {note}"


def _apply_category_routing(candidates: dict[str, list[FieldCandidate]], route, lines: list[OCRLine]) -> None:
    category = getattr(route, "category", "generic_invoice")
    if not lines:
        return

    for field_name, field_candidates in candidates.items():
        for candidate in field_candidates:
            source_text = str(candidate.source_text or "")
            source_compact = _canonical_compact(source_text)
            method = str(candidate.method or "")
            y = _candidate_y_ratio(candidate, lines)

            if category == "chinese_invoice":
                if field_name == "currency":
                    if candidate.value == "CNY":
                        _route_note(candidate, 0.14, "Chinese document prefers CNY/RMB")
                    elif candidate.value in {"USD", "EUR", "GBP", "RM", "MYR"}:
                        _route_note(candidate, -0.12, "non-CNY currency is less likely in Chinese route")
                elif method.startswith("chinese_") or _contains_cjk(source_text):
                    _route_note(candidate, 0.06, "Chinese route prefers Chinese-label evidence")
                if field_name in {"total_amount", "subtotal", "tax_amount"} and method.startswith("chinese_"):
                    _route_note(candidate, 0.05, "Chinese amount rule matched route")

            elif category == "malaysian_receipt":
                if field_name == "currency":
                    if candidate.value in {"RM", "MYR"}:
                        _route_note(candidate, 0.16, "Malaysian route prefers RM/MYR")
                    elif candidate.value in {"CNY", "USD", "EUR", "GBP"}:
                        _route_note(candidate, -0.12, "foreign currency is weaker in Malaysian route")
                if field_name == "total_amount":
                    if y >= 0.48:
                        _route_note(candidate, 0.08, "receipt totals usually appear in lower half")
                    incl_or_excl = any(fragment in source_compact for fragment in ("TOTALINCL", "TOTALRMINCL", "INCLGST", "INCLSST", "TOTALEXCL", "EXCLGST", "EXCLSST"))
                    if incl_or_excl:
                        _route_note(candidate, -0.12, "incl/excl total is weaker than final payable total")
                    final_total_label = (
                        any(fragment in source_compact for fragment in ("ROUNDEDTOTAL", "ROUNDINGTOTAL", "TOTALSALES", "AMOUNTDUE"))
                        or ("TOTALRM" in source_compact and not incl_or_excl)
                    )
                    if final_total_label:
                        _route_note(candidate, 0.10, "Malaysian receipt final-total label")
                    if method in {"financial_constraint_engine", "merged_payment_total_candidate"}:
                        _route_note(candidate, 0.08, "financial/payment evidence fits receipt route")
                if field_name == "vendor_name" and re.search(r"\b(?:SDN\s+BHD|S\s*/\s*B|BHD)\b", source_text, re.I):
                    _route_note(candidate, 0.08, "Malaysian business suffix")

            elif category == "clean_invoice":
                explicit_method = method in {
                    "label_neighbor_candidate",
                    "invoice_id_line_candidate",
                    "tax_label_rate_candidate",
                    "currency_in_total_label",
                    "financial_constraint_engine",
                }
                if field_name in {"invoice_number", "date", "subtotal", "tax_amount", "total_amount", "vat_rate"}:
                    if explicit_method:
                        _route_note(candidate, 0.08, "clean invoice route prefers explicit labels")
                    elif method == "format_prior":
                        _route_note(candidate, -0.08, "clean invoice route deprioritizes label-free guesses")
                if field_name == "total_amount" and any(fragment in source_compact for fragment in ("BALANCEDUE", "AMOUNTDUE", "INVOICETOTAL", "TOTALAMOUNT")):
                    _route_note(candidate, 0.10, "clean invoice final amount label")

            elif category == "fatura":
                if any(fragment in source_compact for fragment in ("FATURA", "KDV", "TOPLAM", "TUTAR", "VERGI")):
                    _route_note(candidate, 0.10, "FATURA/KDV label fits route")
                if method == "format_prior" and field_name in {"invoice_number", "date", "total_amount"}:
                    _route_note(candidate, -0.05, "FATURA route prefers labeled fields")

            elif category == "noisy_receipt":
                if field_name == "total_amount":
                    if y >= 0.50:
                        _route_note(candidate, 0.08, "noisy receipt route prefers lower-half totals")
                    if method in {"financial_constraint_engine", "merged_payment_total_candidate"}:
                        _route_note(candidate, 0.10, "financial/payment consistency fits noisy receipt")
                    if method == "format_prior":
                        _route_note(candidate, -0.07, "noisy receipt route deprioritizes bare amount guesses")
                if field_name == "vendor_name" and y <= 0.25:
                    _route_note(candidate, 0.05, "vendor usually appears near receipt top")


def _attach_document_route(output: ExtractionOutput, route) -> ExtractionOutput:
    if hasattr(route, "evidence"):
        output.evidence["_document_category"] = route.evidence()
    return output


def _count_strings(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _ocr_metadata_evidence(tokens: list) -> dict:
    confidence_values = []
    engine_values: list[str] = []
    variant_values: list[str] = []
    for token in tokens:
        try:
            confidence_values.append(max(min(float(getattr(token, "confidence", 0.0)), 1.0), 0.0))
        except (TypeError, ValueError):
            pass
        engine = (getattr(token, "source_engine", "") or "").strip()
        if engine:
            engine_values.append(engine)
        for part in (getattr(token, "source_variant", "") or "").split("+"):
            part = part.strip()
            if part:
                variant_values.append(part)
    average_confidence = round(mean(confidence_values), 6) if confidence_values else 0.0
    return {
        "token_count": len(tokens),
        "average_confidence": average_confidence,
        "source_engines": _count_strings(engine_values),
        "source_variants": _count_strings(variant_values),
        "preprocessing_variants": sorted(set(variant_values)),
        "method": "ocr_metadata",
        "explanation": "OCR engine and preprocessing evidence from source tokens",
    }


def _attach_ocr_metadata(output: ExtractionOutput, tokens: list) -> ExtractionOutput:
    output.evidence["_ocr"] = _ocr_metadata_evidence(tokens)
    return output


def _choose_best(candidates: dict[str, list[FieldCandidate]]) -> ExtractionOutput:
    output = ExtractionOutput()
    for field_name in FIELD_NAMES:
        field_candidates = [candidate for candidate in candidates.get(field_name, []) if candidate.value]
        if not field_candidates:
            output.fields[field_name] = ""
            output.confidences[field_name] = 0.0
            continue
        best = max(field_candidates, key=lambda candidate: candidate.score)
        # Abstain on currency when the best candidate is below the confidence floor —
        # better to emit an empty value than a hallucinated currency code.
        if field_name == "currency" and best.score < _CURRENCY_CONFIDENCE_FLOOR:
            output.fields[field_name] = ""
            output.confidences[field_name] = 0.0
            continue
        output.fields[field_name] = best.value
        output.confidences[field_name] = best.confidence()
        output.evidence[field_name] = best.evidence()
    return output


def _ensure_defaults(output: ExtractionOutput) -> ExtractionOutput:
    for field_name in FIELD_NAMES:
        output.fields.setdefault(field_name, "")
        output.confidences.setdefault(field_name, 0.0)
    return output


def _line_text(line: OCRLine | list) -> str:
    if isinstance(line, OCRLine):
        return line.text
    return " ".join(_token_text(token) for token in line).strip()


def _line_confidence(line: OCRLine | list) -> float:
    if isinstance(line, OCRLine):
        return line.confidence
    values = [_token_confidence(token) for token in line if _token_text(token).strip()]
    return round(mean(values), 4) if values else 0.0


def extract_with_regex(tokens: Iterable) -> ExtractionOutput:
    token_list = list(tokens)
    lines = _line_groups(token_list)
    from .document_router import detect_document_category
    route = detect_document_category(lines)
    full_text = "\n".join(line.text for line in lines)
    output = ExtractionOutput()

    regex_specs = {
        "invoice_number": re.compile(
            r"(?:invoice\s*(?:no|number|#)|document\s*no|doc\s*no|receipt\s*(?:no|number|#)|bill\s*(?:no|number)|cash\s*bill\s*no|transaction\s*(?:no|number)|ref\s*no)\s*[:#-]?\s*([A-Z0-9][A-Z0-9\-\/]{3,})",
            re.I,
        ),
        "date": re.compile(r"(?:date|invoice\s*date|receipt\s*date)\s*[:#-]?\s*([0-9OIlpP/.\- ]{6,24})", re.I),
        "tax_amount": re.compile(r"(?:tax|vat|gst)\s*[:#-]?\s*((?:RM|MYR|USD|CNY|RMB|NGN)?\s*[$¥₦]?\s*[0-9][0-9,.]*)", re.I),
        "subtotal": re.compile(r"(?:subtotal|sub\s*total)\s*[:#-]?\s*((?:RM|MYR|USD|CNY|RMB|NGN)?\s*[$¥₦]?\s*[0-9][0-9,.]*)", re.I),
        "total_amount": re.compile(
            r"(?:rounded\s*total|grand\s*total|amount\s*due|net\s*total|total\s*(?:amount)?)\s*(?:\([A-Z]{2,3}\))?\s*[:#-]?\s*((?:RM|MYR|USD|CNY|RMB|NGN)?\s*[$¥₦]?\s*[0-9][0-9,.]*)",
            re.I,
        ),
        "vat_rate": re.compile(r"(?:vat\s*rate|tax\s*rate|gst\s*rate)\s*[:#-]?\s*([0-9]{1,2}(?:[,.][0-9]+)?\s*%)", re.I),
        "amount_in_words": re.compile(r"(?:amount\s*in\s*words|total\s*in\s*words)\s*[:#-]?\s*([A-Z \-]+)", re.I),
    }

    for field_name, pattern in regex_specs.items():
        match = pattern.search(full_text)
        if not match:
            continue
        raw = match.group(1).strip()
        value = raw
        if field_name in {"tax_amount", "total_amount", "subtotal"}:
            value = _clean_amount(raw)
        elif field_name == "date":
            dates = _extract_dates(raw)
            value = dates[0][0] if dates else ""
        if not value:
            continue
        output.fields[field_name] = value
        output.confidences[field_name] = 0.62
        output.evidence[field_name] = {
            "value": value,
            "confidence": 0.62,
            "source_text": match.group(0),
            "source_bbox": [],
            "method": "regex_baseline",
            "explanation": "value matched by full-text regular expression",
        }

    currency = _detect_currency(full_text)
    if currency:
        output.fields["currency"] = currency
        output.confidences["currency"] = 0.62
        output.evidence["currency"] = {
            "value": currency,
            "confidence": 0.62,
            "source_text": currency,
            "source_bbox": [],
            "method": "regex_currency",
            "explanation": "currency code/symbol matched in OCR text",
        }

    vendor_candidates = _vendor_candidates(lines)
    if vendor_candidates:
        best_vendor = max(vendor_candidates, key=lambda candidate: candidate.score)
        output.fields["vendor_name"] = best_vendor.value
        output.confidences["vendor_name"] = max(0.55, min(best_vendor.confidence() - 0.12, 0.80))
        output.evidence["vendor_name"] = best_vendor.evidence()
        output.evidence["vendor_name"]["method"] = "regex_top_region_vendor"

    return _attach_ocr_metadata(_attach_document_route(_ensure_defaults(output), route), token_list)


def extract_layout_aware(tokens: Iterable) -> ExtractionOutput:
    token_list = list(tokens)
    lines = _line_groups(token_list)
    from .document_router import detect_document_category
    route = detect_document_category(lines)
    candidates = _generate_candidates(lines)

    # Add amount candidates supported by financial consistency.
    from .constraint_engine import collect_amount_evidence, find_best_assignment, assignment_to_field_candidates
    evidence_pool = collect_amount_evidence(lines)
    assignment = find_best_assignment(evidence_pool)
    if assignment is not None and assignment.consistency_score > 0.5:
        for field_name, fc in assignment_to_field_candidates(assignment).items():
            candidates[field_name].append(fc)

    _apply_category_routing(candidates, route, lines)

    for field_name in candidates:
        candidates[field_name].sort(key=lambda c: c.score, reverse=True)

    from .field_decoder import decode_fields
    decoded = decode_fields(candidates, top_k=3)

    output = ExtractionOutput()
    for field_name in FIELD_NAMES:
        best = decoded.get(field_name)
        if best is None:
            field_candidates = [c for c in candidates.get(field_name, []) if c.value]
            best = field_candidates[0] if field_candidates else None
        if best is None or not best.value:
            output.fields[field_name] = ""
            output.confidences[field_name] = 0.0
        elif field_name == "currency" and best.score < _CURRENCY_CONFIDENCE_FLOOR:
            output.fields[field_name] = ""
            output.confidences[field_name] = 0.0
        else:
            output.fields[field_name] = best.value
            output.confidences[field_name] = best.confidence()
            output.evidence[field_name] = best.evidence()

    return _attach_ocr_metadata(_attach_document_route(_ensure_defaults(output), route), token_list)
