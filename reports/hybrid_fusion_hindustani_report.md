# Evaluation Report: HYBRID (Hindustani Dataset)
**Dataset**: Hindustani

## Overall Summary
- **Macro-F1 Score**: `0.0917`
- **Weighted-F1 Score**: `0.0917`
- **Accuracy (Supplementary)**: `0.1333`
- **Expected Calibration Error (ECE)**: `0.4443`

## Per-Raga Detailed Metrics
| Raga | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Bhairav** | 0.0000 | 0.0000 | 0.0000 | 3 |
| **Bhairavi** | 0.0000 | 0.0000 | 0.0000 | 3 |
| **Yaman** | 0.0000 | 0.0000 | 0.0000 | 3 |
| **Bhoopali** | 0.0000 | 0.0000 | 0.0000 | 3 |
| **Deshkar** | 0.2222 | 0.6667 | 0.3333 | 3 |
| **Bageshree** | 0.0000 | 0.0000 | 0.0000 | 3 |
| **Malkauns** | 0.2000 | 0.3333 | 0.2500 | 3 |
| **Todi** | 0.0000 | 0.0000 | 0.0000 | 3 |
| **Darbari Kanada** | 0.0000 | 0.0000 | 0.0000 | 3 |
| **Kafi** | 0.3333 | 0.3333 | 0.3333 | 3 |

## Confusion Matrix
Rows represent Ground Truth, Columns represent Model Predictions.

| True \ Pred | Bhaira | Bhaira | Yaman | Bhoopa | Deshka | Bagesh | Malkau | Todi | Darbar | Kafi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bhairav** | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | 0 |
| **Bhairavi** | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| **Yaman** | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 1 |
| **Bhoopali** | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 |
| **Deshkar** | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| **Bageshree** | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 1 |
| **Malkauns** | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 1 | 0 |
| **Todi** | 0 | 0 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 0 |
| **Darbari Kanada** | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 0 |
| **Kafi** | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 1 |

## Experimental Observations & Limitations
Trained with hybrid architecture. Checkpoint persisted to `artifacts\checkpoints\hybrid_hindustani_model.pt`.