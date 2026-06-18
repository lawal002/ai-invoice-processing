# AI-Powered Invoice and Receipt Processing

A Django-based intelligent document processing system for extracting, verifying,
reviewing, storing, and exporting structured information from invoices and
receipts.

The completed project compares three approaches on the same labeled evaluation
split:

1. OCR + Regex baseline
2. Fine-tuned LayoutLMv3 baseline
3. Proposed lightweight layout-aware method

## Final Results

The fair comparison uses the same 20-document evaluation split for all methods.

| Method | Overall accuracy | Document pass rate |
|---|---:|---:|
| OCR + Regex | 33.33% | 0.00% |
| Fine-tuned LayoutLMv3 | 58.91% | 5.00% |
| Proposed layout-aware method | 75.97% | 35.00% |

The proposed method achieved the highest overall accuracy. Full field-level and
category-level results are in `results/final_method_comparison/`.

## Main Features

-  JPG, JPEG, and PNG upload
- PaddleOCR with EasyOCR fallback and optional Tesseract support
- Image preprocessing, OCR retry, multi-variant fusion, and OCR caching
- OCR + Regex baseline extraction
- Layout-aware candidate generation and confidence scoring
- Document-category routing for mixed invoice and receipt formats
- Invoice-specific financial constraint and anomaly checks
- Chinese VAT, taxi-invoice, retail-receipt, Malaysian, FATURA, and English
  invoice handling
- Human review and correction workflow
- SQLite record storage
- CSV and JSON export
- Fine-tuned LayoutLMv3 token-classification baseline
- Reproducible evaluation scripts and report-ready result tables

## Architecture

```text
Uploaded document
       |
       v
Preprocessing and OCR
(PaddleOCR / EasyOCR / Tesseract)
       |
       v
Normalized OCR tokens
(text, confidence, page, bounding box)
       |
       +--------------------+
       |                    |
       v                    v
OCR + Regex          Layout-aware extraction
baseline             + category routing
                     + financial constraints
                     + confidence/evidence
       |                    |
       +----------+---------+
                  |
                  v
Invoice anomaly verification
                  |
                  v
Human review and correction
                  |
                  v
SQLite storage and CSV/JSON export
```

LayoutLMv3 is trained and evaluated separately as the deep-learning comparison
baseline using the prepared OCR tokens, bounding boxes, images, and BIO labels.

## Final Dataset

The final dataset contains 83 field-level labeled documents across seven
categories:

| Category | Documents |
|---|---:|
| Chinese invoices | 13 |
| Clean invoices | 10 |
| English invoices | 15 |
| FATURA invoices | 18 |
| Malaysian invoices and receipts | 8 |
| Noisy invoices and receipts | 7 |
| VATID Chinese VAT-style invoices | 12 |

The stratified LayoutLMv3 split contains 63 training documents and 20 evaluation
documents, with no overlap.

Label provenance is recorded rather than hidden: 63 rows reused existing manual
labels, 18 low-confidence or anomalous pre-labeled rows were placed in the
review queue and corrected, and 2 layout-aware pre-labels were retained as
review-optional rows. This should be reported as a limitation when discussing
ground-truth quality.

Key files:

- `data/samples/final_eval/`: final document images
- `data/annotations/final_eval/manual_ground_truth.csv`: field-level ground truth
- `data/annotations/final_eval/README.md`: label creation and review provenance
- `data/annotations/final_eval/dataset_sources.csv`: source and label provenance
- `data/annotations/layoutlmv3/train.jsonl`: prepared training split
- `data/annotations/layoutlmv3/eval.jsonl`: shared final evaluation split

## Project Structure

```text
data/
  annotations/final_eval/       Final labels and provenance
  annotations/layoutlmv3/       Prepared LayoutLMv3 train/eval data
  processed/ocr_cache/          Generated OCR cache
  samples/final_eval/           Final 83-document dataset
docs/
  FINAL_PROJECT_STATUS.md       Verified final project status
  final_method_comparison_report.md
  history/                      Earlier investigation and phase records
models/
  layoutlmv3_baseline_v2/       Final fine-tuned LayoutLMv3 model
results/
  final_eval_evaluation/        Full regex/layout-aware evaluation
  layoutlmv3_baseline_v2_decoder/
  final_method_comparison/      Final same-split three-method comparison
scripts/                        Dataset, model, and evaluation utilities
src/                            Django application and processing services
```

## Environment

The final tested environment is:

- Windows
- Python 3.12
- NVIDIA GPU
- PyTorch 2.5.1 with CUDA 12.1
- PaddlePaddle GPU 3.3.1 with CUDA 12.6 packages

Create the environment from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first OCR run may download PaddleOCR or EasyOCR model files. PDF conversion
requires Poppler on the system `PATH`. Tesseract is an optional fallback and
requires the native Tesseract executable.

On Windows, CUDA PyTorch and CUDA PaddlePaddle can conflict if both frameworks
are imported manually in the wrong order. The project OCR loader isolates the
PaddleOCR import path and was verified on `gpu:0`; use the application services
instead of importing both GPU frameworks in an ad hoc script.

## Run the Web Application

From the project root:

```powershell
.\.venv\Scripts\python.exe src\manage.py migrate
.\.venv\Scripts\python.exe src\manage.py check
.\.venv\Scripts\python.exe src\manage.py runserver
```

Open `http://127.0.0.1:8000/`.

The workflow is:

1. Upload an invoice or receipt.
2. Process OCR and extraction.
3. Compare regex and layout-aware results.
4. Review and correct the proposed extraction.
5. Export the approved record as CSV or JSON.

## Reproduce the Experiments

Evaluate regex and layout-aware extraction on the full labeled dataset:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_labeled_dataset.py
```

Evaluate the saved LayoutLMv3 model:

```powershell
.\.venv\Scripts\python.exe scripts\layoutlmv3_evaluate_baseline.py
```

Regenerate the fair same-split comparison:

```powershell
.\.venv\Scripts\python.exe scripts\compare_methods_on_layoutlmv3_split.py
```

Dataset preparation and model retraining are available through:

```powershell
.\.venv\Scripts\python.exe scripts\layoutlmv3_prepare_dataset.py
.\.venv\Scripts\python.exe scripts\layoutlmv3_train_baseline.py
```

Retraining writes to `models/layoutlmv3_baseline_retrained/` by default so the
preserved final model is not overwritten.

## Verification

Run the complete verification suite:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe src\manage.py makemigrations --check --dry-run
.\.venv\Scripts\python.exe src\manage.py check
.\.venv\Scripts\python.exe src\manage.py test invoices
```

The final verified state contains 144 passing tests and no Django system or
migration issues.

## Known Limitations

- Vendor-name extraction remains the weakest field because invoices may contain
  several organizations, logos, addresses, buyer/seller tables, or noisy OCR.
- The final evaluation split is small, so the experimental conclusions apply to
  this course-project dataset rather than every invoice domain.
- The LayoutLMv3 currency BIO-label coverage was below the preparation
  diagnostic threshold because currency symbols were frequently merged with
  amount tokens. Training proceeded with weighted loss and contextual decoding;
  final currency accuracy was 95%.
- The recorded method timings are not a strictly controlled speed benchmark:
  rule-based timings include OCR, while saved LayoutLMv3 timings cover model
  prediction. Accuracy is the primary fair comparison.

