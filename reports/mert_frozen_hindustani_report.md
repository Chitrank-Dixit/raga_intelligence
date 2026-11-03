# Evaluation Report: MERT_FROZEN (Hindustani Dataset)
**Dataset**: Hindustani

## Overall Summary
- **Macro-F1 Score**: `0.3440`
- **Weighted-F1 Score**: `0.3440`
- **Accuracy (Supplementary)**: `0.4000`
- **Expected Calibration Error (ECE)**: `0.1794`

## Per-Raga Detailed Metrics
| Raga | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Bhairav** | 1.0000 | 1.0000 | 1.0000 | 3 |
| **Bhairavi** | 0.0000 | 0.0000 | 0.0000 | 3 |
| **Yaman** | 0.4000 | 0.6667 | 0.5000 | 3 |
| **Bhoopali** | 0.2500 | 0.3333 | 0.2857 | 3 |
| **Deshkar** | 0.0000 | 0.0000 | 0.0000 | 3 |
| **Bageshree** | 0.5000 | 0.6667 | 0.5714 | 3 |
| **Malkauns** | 0.6000 | 1.0000 | 0.7500 | 3 |
| **Todi** | 0.3333 | 0.3333 | 0.3333 | 3 |
| **Darbari Kanada** | 0.0000 | 0.0000 | 0.0000 | 3 |
| **Kafi** | 0.0000 | 0.0000 | 0.0000 | 3 |

## Confusion Matrix
Rows represent Ground Truth, Columns represent Model Predictions.

| True \ Pred | Bhaira | Bhaira | Yaman | Bhoopa | Deshka | Bagesh | Malkau | Todi | Darbar | Kafi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bhairav** | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Bhairavi** | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 2 |
| **Yaman** | 0 | 0 | 2 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| **Bhoopali** | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| **Deshkar** | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| **Bageshree** | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 1 |
| **Malkauns** | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| **Todi** | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 |
| **Darbari Kanada** | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| **Kafi** | 0 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 0 |

## Experimental Observations & Limitations
Trained with mert_frozen architecture. Checkpoint persisted to `artifacts\checkpoints\mert_frozen_hindustani_model.pt`.