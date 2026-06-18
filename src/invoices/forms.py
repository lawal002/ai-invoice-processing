from django import forms

from .models import InvoiceDocument


class InvoiceUploadForm(forms.ModelForm):
    class Meta:
        model = InvoiceDocument
        fields = ["file"]

    def clean_file(self):
        uploaded = self.cleaned_data["file"]
        allowed = {".pdf", ".jpg", ".jpeg", ".png"}
        suffix = "." + uploaded.name.rsplit(".", 1)[-1].lower() if "." in uploaded.name else ""
        if suffix not in allowed:
            raise forms.ValidationError("Only PDF, JPG, JPEG, and PNG files are supported.")
        return uploaded


class ReviewForm(forms.Form):
    invoice_number = forms.CharField(required=False)
    date = forms.CharField(required=False)
    vendor_name = forms.CharField(required=False)
    tax_amount = forms.CharField(required=False)
    total_amount = forms.CharField(required=False)
    currency = forms.CharField(required=False)
    reviewer_notes = forms.CharField(required=False, widget=forms.Textarea(attrs={"rows": 3}))

