"""
Evaluation report generator (M15).
"""

from typing import Dict, Any
import os
from datetime import datetime

def generate_report(metrics: Dict[str, Any], output_dir: str, run_id: str) -> str:
    """
    Produces a human-readable evaluation report artifact.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"evaluation_report_{run_id}.md")
    
    lines = [
        f"# Model Evaluation Report",
        f"**Run ID:** `{run_id}`",
        f"**Generated At:** {datetime.utcnow().isoformat()}Z",
        "",
        "## Overall Metrics",
        f"- **Macro-F1:** {metrics.get('macro_f1', 0.0):.4f}",
        f"- **AUPRC (Anomaly vs Normal):** {metrics.get('auprc', 0.0):.4f}",
        f"- **AUROC (One-vs-Rest Macro):** {metrics.get('auroc_ovr', 0.0):.4f}",
        f"- **Precision @ 1% (Alert Budget):** {metrics.get('precision_at_1_percent', 0.0):.4f}",
        f"- **FPR @ TPR=0.9:** {metrics.get('fpr_at_tpr_90', 0.0):.4f}",
        "",
        "## Per-Class Performance",
        "| Class | Precision | Recall | F1-Score |",
        "|-------|-----------|--------|----------|"
    ]
    
    per_class = metrics.get('per_class', {})
    for cls, scores in per_class.items():
        lines.append(
            f"| {cls} | {scores.get('precision', 0.0):.4f} | {scores.get('recall', 0.0):.4f} | {scores.get('f1', 0.0):.4f} |"
        )
        
    lines.append("")
    lines.append("## True Positive Risk Tier Distribution")
    lines.append("Shows how correctly identified anomalies were triaged by the fusion logic.")
    tp_dist = metrics.get('tp_risk_tier_distribution', {})
    lines.append(f"- **Critical:** {tp_dist.get('critical', 0)}")
    lines.append(f"- **High:** {tp_dist.get('high', 0)}")
    lines.append(f"- **Medium:** {tp_dist.get('medium', 0)}")
    lines.append(f"- **Low:** {tp_dist.get('low', 0)}")
    
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
        
    return report_path
