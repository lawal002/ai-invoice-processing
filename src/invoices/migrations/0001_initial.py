from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="InvoiceDocument",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("file", models.FileField(upload_to="documents/%Y/%m/%d/")),
                ("original_filename", models.CharField(max_length=255)),
                ("status", models.CharField(choices=[("uploaded", "Uploaded"), ("processed", "Processed"), ("reviewed", "Reviewed"), ("error", "Error")], default="uploaded", max_length=32)),
                ("error_message", models.TextField(blank=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.CreateModel(
            name="ExtractionResult",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("method", models.CharField(choices=[("regex", "OCR + Regex"), ("layout_aware", "Lightweight Layout-Aware"), ("layoutlmv3", "LayoutLMv3 Baseline")], max_length=32)),
                ("fields", models.JSONField(default=dict)),
                ("confidences", models.JSONField(default=dict)),
                ("evidence", models.JSONField(default=dict)),
                ("anomalies", models.JSONField(default=list)),
                ("inference_ms", models.FloatField(default=0)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("document", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="extractions", to="invoices.invoicedocument")),
            ],
            options={"ordering": ["-created_at"]},
        ),
        migrations.CreateModel(
            name="OCRToken",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("page", models.PositiveIntegerField(default=1)),
                ("text", models.TextField()),
                ("x1", models.FloatField(default=0)),
                ("y1", models.FloatField(default=0)),
                ("x2", models.FloatField(default=0)),
                ("y2", models.FloatField(default=0)),
                ("confidence", models.FloatField(default=0)),
                ("document", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="ocr_tokens", to="invoices.invoicedocument")),
            ],
            options={"ordering": ["page", "y1", "x1"]},
        ),
        migrations.CreateModel(
            name="ReviewedInvoice",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("final_fields", models.JSONField(default=dict)),
                ("approved", models.BooleanField(default=False)),
                ("reviewer_notes", models.TextField(blank=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("document", models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name="review", to="invoices.invoicedocument")),
            ],
        ),
    ]

