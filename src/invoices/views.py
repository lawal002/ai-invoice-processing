from __future__ import annotations

import time

from django.contrib import messages
from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404, redirect, render

from .forms import InvoiceUploadForm, ReviewForm
from .models import ExtractionResult, InvoiceDocument, OCRToken, ReviewedInvoice
from .services.anomalies import check_invoice_anomalies
from .services.export import export_reviewed_invoice
from .services.extraction import extract_layout_aware, extract_with_regex
from .services.ocr import OCRDependencyError, run_ocr_with_metadata


def dashboard(request):
    documents = InvoiceDocument.objects.order_by("-created_at")[:50]
    return render(request, "invoices/dashboard.html", {"documents": documents})


def upload_document(request):
    if request.method == "POST":
        form = InvoiceUploadForm(request.POST, request.FILES)
        if form.is_valid():
            document = form.save(commit=False)
            document.original_filename = request.FILES["file"].name
            document.save()
            messages.success(request, "Document uploaded.")
            return redirect("invoices:detail", pk=document.pk)
    else:
        form = InvoiceUploadForm()
    return render(request, "invoices/upload.html", {"form": form})


def document_detail(request, pk):
    document = get_object_or_404(InvoiceDocument, pk=pk)
    regex_result = document.extractions.filter(method=ExtractionResult.Method.REGEX).first()
    layout_result = document.extractions.filter(method=ExtractionResult.Method.LAYOUT_AWARE).first()
    return render(
        request,
        "invoices/detail.html",
        {
            "document": document,
            "regex_result": regex_result,
            "layout_result": layout_result,
            "review": getattr(document, "review", None),
            "ocr_tokens": document.ocr_tokens.all()[:80],
        },
    )


def process_document(request, pk):
    document = get_object_or_404(InvoiceDocument, pk=pk)
    OCRToken.objects.filter(document=document).delete()
    document.extractions.all().delete()
    previous_numbers = set(
        ReviewedInvoice.objects.exclude(document=document)
        .values_list("final_fields__invoice_number", flat=True)
    )
    try:
        tokens, ocr_metadata = run_ocr_with_metadata(document.file.path)
    except OCRDependencyError as exc:
        document.status = InvoiceDocument.Status.ERROR
        document.error_message = str(exc)
        document.save(update_fields=["status", "error_message", "updated_at"])
        messages.error(request, str(exc))
        return redirect("invoices:detail", pk=document.pk)

    OCRToken.objects.bulk_create(
        [
            OCRToken(
                document=document,
                page=token.page,
                text=token.text,
                x1=token.bbox[0],
                y1=token.bbox[1],
                x2=token.bbox[2],
                y2=token.bbox[3],
                confidence=token.confidence,
            )
            for token in tokens
        ]
    )

    for method, extractor in [
        (ExtractionResult.Method.REGEX, extract_with_regex),
        (ExtractionResult.Method.LAYOUT_AWARE, extract_layout_aware),
    ]:
        started = time.perf_counter()
        result = extractor(tokens)
        result.evidence["_ocr_run"] = ocr_metadata
        anomalies = check_invoice_anomalies(result.fields, result.confidences, previous_numbers, evidence=result.evidence)
        ExtractionResult.objects.create(
            document=document,
            method=method,
            fields=result.fields,
            confidences=result.confidences,
            evidence=result.evidence,
            anomalies=[anomaly.to_dict() for anomaly in anomalies],
            inference_ms=(time.perf_counter() - started) * 1000,
        )

    document.status = InvoiceDocument.Status.PROCESSED
    document.error_message = ""
    document.save(update_fields=["status", "error_message", "updated_at"])
    messages.success(request, "Document processed with OCR + Regex and layout-aware methods.")
    return redirect("invoices:detail", pk=document.pk)


def review_document(request, pk):
    document = get_object_or_404(InvoiceDocument, pk=pk)
    extraction = document.extractions.filter(method=ExtractionResult.Method.LAYOUT_AWARE).first()
    if extraction is None:
        messages.error(request, "Process the document before review.")
        return redirect("invoices:detail", pk=document.pk)

    review = getattr(document, "review", None)
    initial = extraction.fields if review is None else review.final_fields
    if review is not None:
        initial = {**initial, "reviewer_notes": review.reviewer_notes}

    if request.method == "POST":
        form = ReviewForm(request.POST)
        if form.is_valid():
            fields = {key: form.cleaned_data.get(key, "") for key in ["invoice_number", "date", "vendor_name", "tax_amount", "total_amount", "currency"]}
            review, _ = ReviewedInvoice.objects.update_or_create(
                document=document,
                defaults={
                    "final_fields": fields,
                    "approved": True,
                    "reviewer_notes": form.cleaned_data.get("reviewer_notes", ""),
                },
            )
            document.status = InvoiceDocument.Status.REVIEWED
            document.save(update_fields=["status", "updated_at"])
            messages.success(request, "Reviewed values saved.")
            return redirect("invoices:detail", pk=document.pk)
    else:
        form = ReviewForm(initial=initial)

    return render(request, "invoices/review.html", {"document": document, "form": form, "extraction": extraction})


def export_document(request, pk, fmt):
    document = get_object_or_404(InvoiceDocument, pk=pk)
    review = getattr(document, "review", None)
    if review is None:
        messages.error(request, "Approve the document before exporting.")
        return redirect("invoices:detail", pk=document.pk)
    try:
        path = export_reviewed_invoice(review.final_fields, document.original_filename, fmt)
    except ValueError as exc:
        raise Http404(str(exc))
    return FileResponse(open(path, "rb"), as_attachment=True, filename=path.name)
