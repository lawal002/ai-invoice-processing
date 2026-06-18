from django.contrib import admin

from .models import ExtractionResult, InvoiceDocument, OCRToken, ReviewedInvoice


@admin.register(InvoiceDocument)
class InvoiceDocumentAdmin(admin.ModelAdmin):
    list_display = ("original_filename", "status", "created_at", "updated_at")
    search_fields = ("original_filename",)
    list_filter = ("status",)


@admin.register(OCRToken)
class OCRTokenAdmin(admin.ModelAdmin):
    list_display = ("document", "page", "text", "confidence")
    search_fields = ("text",)


@admin.register(ExtractionResult)
class ExtractionResultAdmin(admin.ModelAdmin):
    list_display = ("document", "method", "inference_ms", "created_at")
    list_filter = ("method",)


@admin.register(ReviewedInvoice)
class ReviewedInvoiceAdmin(admin.ModelAdmin):
    list_display = ("document", "approved", "updated_at")

