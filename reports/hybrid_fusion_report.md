# Evaluation Report: HYBRID (Synthetic Dataset)
**Dataset**: Synthetic

## Overall Summary
- **Macro-F1 Score**: `0.1000`
- **Weighted-F1 Score**: `0.1000`
- **Accuracy (Supplementary)**: `0.1000`
- **Expected Calibration Error (ECE)**: `0.0396`

## Per-Raga Detailed Metrics
| Raga | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Bhairav** | 0.0000 | 0.0000 | 0.0000 | 1 |
| **Bhairavi** | 0.0000 | 0.0000 | 0.0000 | 1 |
| **Yaman** | 0.0000 | 0.0000 | 0.0000 | 1 |
| **Bhoopali** | 0.0000 | 0.0000 | 0.0000 | 1 |
| **Deshkar** | 0.0000 | 0.0000 | 0.0000 | 1 |
| **Bageshree** | 0.0000 | 0.0000 | 0.0000 | 1 |
| **Malkauns** | 1.0000 | 1.0000 | 1.0000 | 1 |
| **Todi** | 0.0000 | 0.0000 | 0.0000 | 1 |
| **Darbari Kanada** | 0.0000 | 0.0000 | 0.0000 | 1 |
| **Kafi** | 0.0000 | 0.0000 | 0.0000 | 1 |

## Confusion Matrix
Rows represent Ground Truth, Columns represent Model Predictions.

| True \ Pred | Bhaira | Bhaira | Yaman | Bhoopa | Deshka | Bagesh | Malkau | Todi | Darbar | Kafi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bhairav** | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Bhairavi** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| **Yaman** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Bhoopali** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| **Deshkar** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| **Bageshree** | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| **Malkauns** | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| **Todi** | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Darbari Kanada** | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Kafi** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |

## Experimental Observations & Limitations
Trained with hybrid architecture. Checkpoint persisted to `artifacts\checkpoints\hybrid_raga_model.pt`.