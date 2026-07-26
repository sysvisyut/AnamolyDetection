# Model Evaluation Report
**Run ID:** `testrun123`
**Generated At:** 2026-07-26T16:01:35.513356Z

## Overall Metrics
- **Macro-F1:** 0.1053
- **AUPRC (Anomaly vs Normal):** 0.1000
- **AUROC (One-vs-Rest Macro):** 0.0000
- **Precision @ 1% (Alert Budget):** 0.0000
- **FPR @ TPR=0.9:** 1.0000

## Per-Class Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| normal | 0.9000 | 1.0000 | 0.9474 |
| brute_force | 0.0000 | 0.0000 | 0.0000 |
| impossible_travel | 0.0000 | 0.0000 | 0.0000 |
| credential_stuffing | 0.0000 | 0.0000 | 0.0000 |
| lateral_movement | 0.0000 | 0.0000 | 0.0000 |
| device_spoofing | 0.0000 | 0.0000 | 0.0000 |
| low_and_slow | 0.0000 | 0.0000 | 0.0000 |
| insider_drift | 0.0000 | 0.0000 | 0.0000 |
| unclassified | 0.0000 | 0.0000 | 0.0000 |

## True Positive Risk Tier Distribution
Shows how correctly identified anomalies were triaged by the fusion logic.
- **Critical:** 0
- **High:** 0
- **Medium:** 0
- **Low:** 0