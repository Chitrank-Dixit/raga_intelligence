# Evaluation Report: HYBRID (Hindustani Dataset)
**Dataset**: Hindustani

## Overall Summary
- **Macro-F1 Score**: `0.7400`
- **Weighted-F1 Score**: `0.7400`
- **Accuracy (Supplementary)**: `0.7667`
- **Expected Calibration Error (ECE)**: `0.1808`

## Per-Raga Detailed Metrics
| Raga | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Bhairav** | 0.7500 | 1.0000 | 0.8571 | 3 |
| **Bhairavi** | 0.7500 | 1.0000 | 0.8571 | 3 |
| **Yaman** | 1.0000 | 1.0000 | 1.0000 | 3 |
| **Bhoopali** | 0.5000 | 0.3333 | 0.4000 | 3 |
| **Deshkar** | 0.7500 | 1.0000 | 0.8571 | 3 |
| **Bageshree** | 1.0000 | 0.3333 | 0.5000 | 3 |
| **Malkauns** | 1.0000 | 1.0000 | 1.0000 | 3 |
| **Todi** | 0.7500 | 1.0000 | 0.8571 | 3 |
| **Darbari Kanada** | 1.0000 | 0.3333 | 0.5000 | 3 |
| **Kafi** | 0.5000 | 0.6667 | 0.5714 | 3 |

## Confusion Matrix
Rows represent Ground Truth, Columns represent Model Predictions.

| True \ Pred | Bhaira | Bhaira | Yaman | Bhoopa | Deshka | Bagesh | Malkau | Todi | Darbar | Kafi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bhairav** | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Bhairavi** | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Yaman** | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Bhoopali** | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| **Deshkar** | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
| **Bageshree** | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 |
| **Malkauns** | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| **Todi** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 |
| **Darbari Kanada** | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| **Kafi** | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 2 |

## Experimental Observations & Limitations
Trained with hybrid architecture. Checkpoint persisted to `artifacts\checkpoints\hybrid_hindustani_model.pt`.