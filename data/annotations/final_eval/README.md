# Final Evaluation Labels

This folder contains the 83-row field-level label set used by the final project.

## Files

- `manual_ground_truth.csv`: final field values used for training and evaluation
- `auto_predictions.csv`: pre-label source, review status, confidence, and OCR metadata
- `review_queue.csv` / `review_queue.xlsx`: rows selected for manual correction
- `dataset_sources.csv`: document source and label provenance

## Label Provenance

The final labels were not all typed from blank rows:

- 63 rows reused existing manual labels.
- 18 rows were flagged as `needs_review` and corrected through the review queue.
- 2 layout-aware pre-labels were retained as review-optional rows.

The two retained pre-labels are a small potential source of evaluation bias and
must be disclosed in the final report. An ideal future dataset would be
independently annotated from each source image by at least one reviewer.
