# Evaluation Report: BASELINE (Hindustani Dataset)
**Dataset**: Hindustani

## Overall Summary
- **Macro-F1 Score**: `0.6067`
- **Weighted-F1 Score**: `0.6067`
- **Accuracy (Supplementary)**: `0.6000`
- **Expected Calibration Error (ECE)**: `0.2003`

## Per-Raga Detailed Metrics
| Raga | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Bhairav** | 0.3333 | 0.3333 | 0.3333 | 3 |
| **Bhairavi** | 1.0000 | 0.6667 | 0.8000 | 3 |
| **Yaman** | 1.0000 | 0.3333 | 0.5000 | 3 |
| **Bhoopali** | 0.5000 | 0.3333 | 0.4000 | 3 |
| **Deshkar** | 0.5000 | 1.0000 | 0.6667 | 3 |
| **Bageshree** | 1.0000 | 0.3333 | 0.5000 | 3 |
| **Malkauns** | 1.0000 | 1.0000 | 1.0000 | 3 |
| **Todi** | 0.2857 | 0.6667 | 0.4000 | 3 |
| **Darbari Kanada** | 0.6667 | 0.6667 | 0.6667 | 3 |
| **Kafi** | 1.0000 | 0.6667 | 0.8000 | 3 |

## Confusion Matrix
Rows represent Ground Truth, Columns represent Model Predictions.

| True \ Pred | Bhaira | Bhaira | Yaman | Bhoopa | Deshka | Bagesh | Malkau | Todi | Darbar | Kafi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bhairav** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| **Bhairavi** | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| **Yaman** | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| **Bhoopali** | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| **Deshkar** | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
| **Bageshree** | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 |
| **Malkauns** | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| **Todi** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| **Darbari Kanada** | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 0 |
| **Kafi** | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 2 |

## Experimental Observations & Limitations
Trained with baseline architecture. Checkpoint persisted to `artifacts\checkpoints\baseline_svm.joblib`.