"""
Automated error analysis, directional confusion asymmetry, audio quality correlation,
and musicological hypothesis attribution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Established musicological hypotheses for frequently confused Hindustani raga pairs
MUSICOLOGICAL_HYPOTHESES: Dict[Tuple[str, str], str] = {
    ("Bhoopali", "Deshkar"): (
        "Both Bhoopali and Deshkar are Audav (pentatonic) ragas belonging to Bilawal thhat "
        "sharing the identical swara scale: Sa, Re, Ga, Pa, Dha (omitting Ma and Ni). They are "
        "distinguished primarily by Vadi/Samvadi hierarchy (Gandhar in Bhoopali vs Dhaivat in Deshkar), "
        "register emphasis (Bhoopali favors Mandra and Madhya saptak with characteristic glide G \\ R S, "
        "whereas Deshkar emphasizes Taar saptak and prayogas like D-P, G-P-D-S'), and sangatis. "
        "Without sequential melodic phrase contours (chalan) or octave register weighting, static pitch "
        "histograms cannot reliably separate them."
    ),
    ("Deshkar", "Bhoopali"): (
        "Identical pentatonic scale set (S R G P D). Deshkar recordings evaluated in isolation without "
        "octave-separated pitch tracking or temporal phrase modeling map to the same pitch-class profile "
        "as Bhoopali, causing static classifiers to mispredict Bhoopali."
    ),
    ("Bhairav", "Todi"): (
        "Both ragas feature prominent komal Re and komal Dha with slow, wide oscillations (andolan). "
        "Todi uses komal Ga and teevra Ma, while Bhairav uses shuddha Ga and shuddha Ma. Neural encoders "
        "or acoustic features can confuse them due to shared microtonal pitch inflections, vocal production "
        "styles, and tanpura drone textures."
    ),
    ("Todi", "Bhairav"): (
        "Oscillation on komal Re and komal Dha in Todi can obscure the semitone boundaries of komal Ga "
        "and teevra Ma if the performance has low vocal energy or heavy tanpura resonance, leading to Bhairav confusion."
    ),
    ("Bageshree", "Kafi"): (
        "Both belong to Kafi thhat (komal Ga, komal Ni). Bageshree is Audav-Sampurna (omitting Pa in arohana, "
        "highlighting Ma and Ni), while Kafi uses all seven notes with strong Pa. A short recording snippet "
        "focusing on Gandhar-Madhyam or omitting Pa can cause Kafi to be misclassified as Bageshree."
    ),
    ("Kafi", "Bageshree"): (
        "When Kafi phrases emphasize Madhyam or omit Pancham in specific avarohana passages, its pitch distribution "
        "closely resembles Bageshree."
    ),
    ("Bhairavi", "Malkauns"): (
        "Malkauns is an Audav raga from Bhairavi thhat (S g m d n), omitting Re and Pa. When a Bhairavi "
        "performance lingers in the upper tetrachord (uttaranga) and transiently under-emphasizes Re and Pa "
        "in a 10-30 second analysis window, pitch histograms closely mirror Malkauns."
    ),
    ("Malkauns", "Bhairavi"): (
        "If background drone or vocal artifacts introduce residual energy around Re or Pa, Malkauns may be "
        "erroneously assigned to the parent Bhairavi scale."
    ),
    ("Yaman", "Bhoopali"): (
        "Bhoopali (S R G P D) is an Audav subset of the Kalyan scale (Yaman); all 5 notes of Bhoopali belong to "
        "Yaman. Yaman adds teevra Ma (M') and shuddha Ni (N). In fast taans or sthayi passages where teevra Ma is "
        "skipped or transient, the local pitch distribution overlaps heavily with Bhoopali."
    ),
    ("Bhoopali", "Yaman"): (
        "Teevra Ma leakage or ambient overtones in Bhoopali recordings can trigger Yaman predictions due to the "
        "underlying Kalyan thhat parent-child relationship."
    ),
    ("Darbari Kanada", "Kafi"): (
        "Both ragas utilize komal Ga and komal Ni. Darbari Kanada (Asavari thhat) is characterized by heavy, "
        "slow andolans on komal Ga and komal Dha with descending glides. If temporal dynamics and oscillations "
        "are averaged out in static feature vectors, pitch-class distributions can lead to mutual confusion."
    ),
    ("Bhairav", "Bhairavi"): (
        "Both share komal Re and komal Dha. Bhairavi adds komal Ga and komal Ni. If melodic phrases in Bhairavi "
        "emphasize the tonic and dominant with komal Re and komal Dha, pitch distributions partially overlap."
    ),
}


@dataclass
class ConfusionPairDetail:
    """Detailed record for a specific confused pair of ragas."""
    true_raga: str
    pred_raga: str
    count: int
    error_percentage: float
    musical_hypothesis: str


@dataclass
class AsymmetryDetail:
    """Directional confusion asymmetry between two ragas."""
    pair: Tuple[str, str]
    count_a_to_b: int
    count_b_to_a: int
    net_asymmetry: int
    dominant_direction: str


@dataclass
class ErrorAnalysisReport:
    """Consolidated error analysis report."""
    total_samples: int
    total_errors: int
    error_rate: float
    top_confusion_pairs: List[ConfusionPairDetail]
    directional_asymmetries: List[AsymmetryDetail]
    poor_recall_classes: List[Dict[str, Any]]
    poor_precision_classes: List[Dict[str, Any]]
    quality_correlation: Optional[Dict[str, Any]] = None
    tonic_correlation: Optional[Dict[str, Any]] = None


def get_musical_hypothesis(raga_a: str, raga_b: str) -> str:
    """Retrieves or synthesizes a musicological explanation hypothesis for a confusion pair."""
    if (raga_a, raga_b) in MUSICOLOGICAL_HYPOTHESES:
        return MUSICOLOGICAL_HYPOTHESES[(raga_a, raga_b)]
    if (raga_b, raga_a) in MUSICOLOGICAL_HYPOTHESES:
        return MUSICOLOGICAL_HYPOTHESES[(raga_b, raga_a)]
    return (
        f"Acoustic or modal overlap between '{raga_a}' and '{raga_b}'. Possible shared swara subsets, "
        f"tetrachord structural affinity, or ambiguous melodic phrases in short segments lacking "
        f"defining prayogas or vadi-samvadi emphasis."
    )


def analyze_classification_errors(
    y_true: Sequence[str] | np.ndarray,
    y_pred: Sequence[str] | np.ndarray,
    class_names: List[str],
    y_probs: Optional[np.ndarray] = None,
    voiced_ratios: Optional[Sequence[float] | np.ndarray] = None,
    spectral_flatness: Optional[Sequence[float] | np.ndarray] = None,
    tonic_errors_cents: Optional[Sequence[float] | np.ndarray] = None,
    recall_threshold: float = 0.50,
    precision_threshold: float = 0.50,
) -> ErrorAnalysisReport:
    """
    Performs comprehensive diagnostic error analysis on model predictions.
    
    Identifies:
    1. Most frequently confused raga pairs with musical hypotheses.
    2. Directional confusion asymmetry.
    3. Ragas with low recall and low precision.
    4. Error correlations with acoustic quality (voiced ratio, spectral flatness).
    5. Error correlations with tonic estimation inaccuracies (> 50 cents).
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    n_samples = len(y_true_arr)

    errors_mask = (y_true_arr != y_pred_arr)
    total_errors = int(np.sum(errors_mask))
    error_rate = float(total_errors / n_samples) if n_samples > 0 else 0.0

    # 1. Confusion pairs
    pair_counts: Dict[Tuple[str, str], int] = {}
    for t, p in zip(y_true_arr[errors_mask], y_pred_arr[errors_mask]):
        pair_counts[(t, p)] = pair_counts.get((t, p), 0) + 1

    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
    top_confusion_pairs: List[ConfusionPairDetail] = []
    for (t, p), cnt in sorted_pairs:
        pct = (cnt / total_errors * 100.0) if total_errors > 0 else 0.0
        hypothesis = get_musical_hypothesis(t, p)
        top_confusion_pairs.append(
            ConfusionPairDetail(
                true_raga=t,
                pred_raga=p,
                count=cnt,
                error_percentage=round(pct, 2),
                musical_hypothesis=hypothesis,
            )
        )

    # 2. Directional confusion asymmetry
    evaluated_pairs = set()
    directional_asymmetries: List[AsymmetryDetail] = []
    for (t, p), cnt_ab in sorted_pairs:
        canonical_key = tuple(sorted([t, p]))
        if canonical_key in evaluated_pairs:
            continue
        evaluated_pairs.add(canonical_key)

        raga_a, raga_b = canonical_key
        cnt_a_to_b = pair_counts.get((raga_a, raga_b), 0)
        cnt_b_to_a = pair_counts.get((raga_b, raga_a), 0)
        net = cnt_a_to_b - cnt_b_to_a

        if cnt_a_to_b > cnt_b_to_a:
            dom = f"{raga_a} -> {raga_b} ({cnt_a_to_b} vs {cnt_b_to_a})"
        elif cnt_b_to_a > cnt_a_to_b:
            dom = f"{raga_b} -> {raga_a} ({cnt_b_to_a} vs {cnt_a_to_b})"
        else:
            dom = f"Symmetric ({cnt_a_to_b} vs {cnt_b_to_a})"

        directional_asymmetries.append(
            AsymmetryDetail(
                pair=(raga_a, raga_b),
                count_a_to_b=cnt_a_to_b,
                count_b_to_a=cnt_b_to_a,
                net_asymmetry=abs(net),
                dominant_direction=dom,
            )
        )

    directional_asymmetries.sort(key=lambda x: x.net_asymmetry, reverse=True)

    # 3. Class-level recall and precision screening
    poor_recall_classes: List[Dict[str, Any]] = []
    poor_precision_classes: List[Dict[str, Any]] = []

    for raga in class_names:
        tp = np.sum((y_true_arr == raga) & (y_pred_arr == raga))
        fn = np.sum((y_true_arr == raga) & (y_pred_arr != raga))
        fp = np.sum((y_true_arr != raga) & (y_pred_arr == raga))

        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

        if recall < recall_threshold:
            poor_recall_classes.append({
                "raga": raga,
                "recall": round(recall, 4),
                "support": int(tp + fn),
                "missed_as": [p for t, p in zip(y_true_arr, y_pred_arr) if t == raga and p != raga],
            })

        if precision < precision_threshold:
            poor_precision_classes.append({
                "raga": raga,
                "precision": round(precision, 4),
                "false_positives_count": int(fp),
                "intruding_true_ragas": [t for t, p in zip(y_true_arr, y_pred_arr) if p == raga and t != raga],
            })

    poor_recall_classes.sort(key=lambda x: x["recall"])
    poor_precision_classes.sort(key=lambda x: x["precision"])

    # 4. Audio quality correlation (voiced ratio, spectral flatness)
    quality_correlation: Optional[Dict[str, Any]] = None
    if voiced_ratios is not None:
        vr = np.asarray(voiced_ratios)
        if len(vr) == n_samples and n_samples > 0:
            median_vr = float(np.median(vr))
            low_vr_mask = vr < median_vr
            high_vr_mask = ~low_vr_mask

            err_low_vr = float(np.mean(errors_mask[low_vr_mask])) if np.sum(low_vr_mask) > 0 else 0.0
            err_high_vr = float(np.mean(errors_mask[high_vr_mask])) if np.sum(high_vr_mask) > 0 else 0.0

            quality_correlation = {
                "median_voiced_ratio": round(median_vr, 4),
                "error_rate_low_voiced_ratio": round(err_low_vr, 4),
                "error_rate_high_voiced_ratio": round(err_high_vr, 4),
                "low_voiced_error_inflation": round(err_low_vr - err_high_vr, 4),
            }

            if spectral_flatness is not None:
                sf = np.asarray(spectral_flatness)
                if len(sf) == n_samples:
                    median_sf = float(np.median(sf))
                    high_sf_mask = sf > median_sf
                    err_high_sf = float(np.mean(errors_mask[high_sf_mask])) if np.sum(high_sf_mask) > 0 else 0.0
                    quality_correlation["median_spectral_flatness"] = round(median_sf, 4)
                    quality_correlation["error_rate_high_spectral_flatness"] = round(err_high_sf, 4)

    # 5. Tonic estimation correlation (> 50 cents deviation)
    tonic_correlation: Optional[Dict[str, Any]] = None
    if tonic_errors_cents is not None:
        te = np.asarray(tonic_errors_cents)
        if len(te) == n_samples and n_samples > 0:
            large_tonic_err_mask = np.abs(te) > 50.0  # more than a quarter semitone error
            normal_tonic_mask = ~large_tonic_err_mask

            err_large_tonic = float(np.mean(errors_mask[large_tonic_err_mask])) if np.sum(large_tonic_err_mask) > 0 else 0.0
            err_normal_tonic = float(np.mean(errors_mask[normal_tonic_mask])) if np.sum(normal_tonic_mask) > 0 else 0.0

            tonic_correlation = {
                "large_tonic_error_count": int(np.sum(large_tonic_err_mask)),
                "error_rate_with_large_tonic_deviation": round(err_large_tonic, 4),
                "error_rate_with_accurate_tonic": round(err_normal_tonic, 4),
                "tonic_error_relative_risk": round(err_large_tonic / max(err_normal_tonic, 1e-4), 2),
            }

    return ErrorAnalysisReport(
        total_samples=n_samples,
        total_errors=total_errors,
        error_rate=round(error_rate, 4),
        top_confusion_pairs=top_confusion_pairs,
        directional_asymmetries=directional_asymmetries,
        poor_recall_classes=poor_recall_classes,
        poor_precision_classes=poor_precision_classes,
        quality_correlation=quality_correlation,
        tonic_correlation=tonic_correlation,
    )


def format_error_analysis_markdown(report: ErrorAnalysisReport) -> str:
    """Formats an ErrorAnalysisReport into a clear, evidence-driven Markdown document."""
    lines = [
        "## Automated Error Analysis & Musicological Diagnostics",
        f"- **Evaluated Samples**: `{report.total_samples}`",
        f"- **Total Misclassifications**: `{report.total_errors}`",
        f"- **Overall Error Rate**: `{report.error_rate:.4f}` ({report.error_rate * 100:.1f}%)",
        "",
        "### Top Frequently Confused Raga Pairs",
    ]

    if not report.top_confusion_pairs:
        lines.append("No misclassifications observed.")
    else:
        lines.extend([
            "| True Raga | Predicted As | Count | % of Errors | Musical Explanation (Hypothesis) |",
            "| :--- | :--- | :---: | :---: | :--- |",
        ])
        for p in report.top_confusion_pairs[:8]:
            lines.append(
                f"| **{p.true_raga}** | **{p.pred_raga}** | {p.count} | {p.error_percentage:.1f}% | "
                f"*Hypothesis: {p.musical_hypothesis}* |"
            )

    lines.extend([
        "",
        "### Directional Confusion Asymmetry",
        "| Raga Pair | Direction A -> B | Direction B -> A | Net Bias | Dominant Direction |",
        "| :--- | :---: | :---: | :---: | :--- |",
    ])
    for a in report.directional_asymmetries[:6]:
        lines.append(
            f"| **{a.pair[0]} & {a.pair[1]}** | {a.count_a_to_b} | {a.count_b_to_a} | "
            f"{a.net_asymmetry} | {a.dominant_direction} |"
        )

    if report.poor_recall_classes:
        lines.extend([
            "",
            "### Classes with Poor Recall (< 0.50)",
            "| Raga | Recall | Support | Frequent Misclassifications |",
            "| :--- | :---: | :---: | :--- |",
        ])
        for pr in report.poor_recall_classes:
            misses = ", ".join([f"`{m}`" for m in pr['missed_as']]) or "None"
            lines.append(f"| **{pr['raga']}** | `{pr['recall']:.4f}` | {pr['support']} | {misses} |")

    if report.poor_precision_classes:
        lines.extend([
            "",
            "### Classes with Poor Precision (< 0.50)",
            "| Raga | Precision | False Positives | Intruding True Classes |",
            "| :--- | :---: | :---: | :--- |",
        ])
        for pp in report.poor_precision_classes:
            intruders = ", ".join([f"`{c}`" for c in pp['intruding_true_ragas']]) or "None"
            lines.append(f"| **{pp['raga']}** | `{pp['precision']:.4f}` | {pp['false_positives_count']} | {intruders} |")

    if report.quality_correlation:
        lines.extend([
            "",
            "### Acoustic Quality vs. Error Correlation",
            f"- **Error Rate in Low-Voiced Segments (< median {report.quality_correlation.get('median_voiced_ratio', 0):.2f})**: "
            f"`{report.quality_correlation.get('error_rate_low_voiced_ratio', 0):.4f}`",
            f"- **Error Rate in High-Voiced Segments**: "
            f"`{report.quality_correlation.get('error_rate_high_voiced_ratio', 0):.4f}`",
            f"- **Low-Voiced Error Inflation**: "
            f"`{report.quality_correlation.get('low_voiced_error_inflation', 0):+.4f}`",
        ])

    if report.tonic_correlation:
        lines.extend([
            "",
            "### Tonic Accuracy vs. Error Correlation",
            f"- **Clips with Large Tonic Inaccuracy (> 50 cents)**: `{report.tonic_correlation.get('large_tonic_error_count', 0)}`",
            f"- **Error Rate under Large Tonic Inaccuracy**: `{report.tonic_correlation.get('error_rate_with_large_tonic_deviation', 0):.4f}`",
            f"- **Error Rate under Accurate Tonic**: `{report.tonic_correlation.get('error_rate_with_accurate_tonic', 0):.4f}`",
            f"- **Relative Error Risk**: `{report.tonic_correlation.get('tonic_error_relative_risk', 0):.2f}x`",
        ])

    return "\n".join(lines)

