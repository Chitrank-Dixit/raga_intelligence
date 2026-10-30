# Evaluation Report: BASELINE (Synthetic Dataset)
**Dataset**: Synthetic

## Overall Summary
- **Macro-F1 Score**: `0.8667`
- **Weighted-F1 Score**: `0.8667`
- **Accuracy (Supplementary)**: `0.9000`
- **Expected Calibration Error (ECE)**: `0.6505`

## Per-Raga Detailed Metrics
| Raga | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Bhairav** | 1.0000 | 1.0000 | 1.0000 | 1 |
| **Bhairavi** | 1.0000 | 1.0000 | 1.0000 | 1 |
| **Yaman** | 1.0000 | 1.0000 | 1.0000 | 1 |
| **Bhoopali** | 0.0000 | 0.0000 | 0.0000 | 1 |
| **Deshkar** | 0.5000 | 1.0000 | 0.6667 | 1 |
| **Bageshree** | 1.0000 | 1.0000 | 1.0000 | 1 |
| **Malkauns** | 1.0000 | 1.0000 | 1.0000 | 1 |
| **Todi** | 1.0000 | 1.0000 | 1.0000 | 1 |
| **Darbari Kanada** | 1.0000 | 1.0000 | 1.0000 | 1 |
| **Kafi** | 1.0000 | 1.0000 | 1.0000 | 1 |

## Confusion Matrix
Rows represent Ground Truth, Columns represent Model Predictions.

| True \ Pred | Bhaira | Bhaira | Yaman | Bhoopa | Deshka | Bagesh | Malkau | Todi | Darbar | Kafi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bhairav** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Bhairavi** | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Yaman** | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Bhoopali** | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| **Deshkar** | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| **Bageshree** | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| **Malkauns** | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| **Todi** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| **Darbari Kanada** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| **Kafi** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |

## Experimental Observations & Limitations
Trained with baseline architecture. Checkpoint persisted to `artifacts\checkpoints\baseline_svm.joblib`.