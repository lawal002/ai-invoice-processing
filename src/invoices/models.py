from django.db import models


class InvoiceDocument(models.Model):
    class Status(models.TextChoices):
        UPLOADED = "uploaded", "Uploaded"
        PROCESSED = "processed", "Processed"
        REVIEWED = "reviewed", "Reviewed"
        ERROR = "error", "Error"

    file = models.FileField(upload_to="documents/%Y/%m/%d/")
    original_filename = models.CharField(max_length=255)
    status = models.CharField(max_length=32, choices=Status.choices, default=Status.UPLOADED)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.original_filename} ({self.status})"


class OCRToken(models.Model):
    document = models.ForeignKey(InvoiceDocument, on_delete=models.CASCADE, related_name="ocr_tokens")
    page = models.PositiveIntegerField(default=1)
    text = models.TextField()
    x1 = models.FloatField(default=0)
    y1 = models.FloatField(default=0)
    x2 = models.FloatField(default=0)
    y2 = models.FloatField(default=0)
    confidence = models.FloatField(default=0)

    class Meta:
        ordering = ["page", "y1", "x1"]

    def to_service_dict(self):
        return {
            "page": self.page,
            "text": self.text,
            "bbox": [self.x1, self.y1, self.x2, self.y2],
            "confidence": self.confidence,
        }


class ExtractionResult(models.Model):
    class Method(models.TextChoices):
        REGEX = "regex", "OCR + Regex"
        LAYOUT_AWARE = "layout_aware", "Lightweight Layout-Aware"
        LAYOUTLMV3 = "layoutlmv3", "LayoutLMv3 Baseline"

    document = models.ForeignKey(InvoiceDocument, on_delete=models.CASCADE, related_name="extractions")
    method = models.CharField(max_length=32, choices=Method.choices)
    fields = models.JSONField(default=dict)
    confidences = models.JSONField(default=dict)
    evidence = models.JSONField(default=dict)
    anomalies = models.JSONField(default=list)
    inference_ms = models.FloatField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]


class ReviewedInvoice(models.Model):
    document = models.OneToOneField(InvoiceDocument, on_delete=models.CASCADE, related_name="review")
    final_fields = models.JSONField(default=dict)
    approved = models.BooleanField(default=False)
    reviewer_notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

