"""
CLI evaluation script to analyze an individual audio recording or song file.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from raga_intelligence.inference.pipeline import InferencePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze an audio recording and output ranked raga similarity matches."
    )
    parser.add_argument("audio_file", type=str, help="Path to input audio recording (WAV, MP3, FLAC)")
    parser.add_argument("--tonic", type=float, default=None, help="Known tonic frequency (Sa) in Hz")
    parser.add_argument("--top-k", type=int, default=5, help="Number of candidate ragas to display")
    parser.add_argument("--model-path", type=str, default="./artifacts/checkpoints/baseline_svm.joblib", help="Optional classifier checkpoint")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        logger.error(f"Audio file '{audio_path}' does not exist.")
        return

    pipe = InferencePipeline(
        baseline_classifier_path=args.model_path if Path(args.model_path).exists() else None
    )
    result = pipe.analyze_audio(audio_path, ground_truth_tonic_hz=args.tonic, top_k=args.top_k)

    if args.json:
        data = {
            "file": str(audio_path),
            "duration_sec": result.duration_sec,
            "estimated_tonic_hz": result.estimated_tonic_hz,
            "voiced_ratio": result.voiced_ratio,
            "detected_dominant_swaras": result.detected_dominant_swaras,
            "abstention_triggered": result.abstention_triggered,
            "abstention_reason": result.abstention_reason,
            "strict_classification_winner": result.strict_classification_winner,
            "matches": [
                {
                    "raga": m.raga_name,
                    "score": m.similarity_score,
                    "confidence": m.confidence_level,
                    "thhat": m.thhat,
                    "matching_swaras": m.matching_swaras,
                    "foreign_swaras": m.foreign_swaras,
                    "peak_windows": m.evidence_segments,
                }
                for m in result.ranked_matches
            ],
            "caveat": result.caveat_statement,
        }
        print(json.dumps(data, indent=2))
        return

    print("\n" + "=" * 65)
    print(" INDIAN RAGA INTELLIGENCE: MELODIC ANALYSIS REPORT")
    print("=" * 65)
    print(f"Source Recording    : {audio_path.name}")
    print(f"Duration            : {result.duration_sec:.2f}s")
    print(f"Estimated Tonic (Sa): {result.estimated_tonic_hz:.2f} Hz")
    print(f"Voiced Ratio        : {result.voiced_ratio:.1%}")
    print(f"Dominant Swaras     : {', '.join(result.detected_dominant_swaras) if result.detected_dominant_swaras else 'None'}")
    if result.strict_classification_winner:
        print(f"Classifier Winner   : {result.strict_classification_winner} (Confidence: {result.strict_classification_confidence:.2%})")

    if result.abstention_triggered:
        print("\n[!] CALIBRATED ABSTENTION TRIGGERED:")
        print(f"    Reason: {result.abstention_reason}")
    else:
        print("\nRanked Melodic Similarity Matches:")
        print(f"{'Raga':<18} {'Score':<8} {'Confidence':<15} {'Thhat':<12} {'Aligned Swaras'}")
        print("-" * 65)
        for m in result.ranked_matches:
            swaras_str = " ".join(m.matching_swaras)
            print(f"{m.raga_name:<18} {m.similarity_score:<8.3f} {m.confidence_level:<15} {m.thhat:<12} {swaras_str}")
            if m.foreign_swaras:
                print(f"  -> Discrepancy (Omitted Notes in Raga): {', '.join(m.foreign_swaras)}")
            if m.evidence_segments:
                print(f"  -> Supporting Windows: {', '.join(m.evidence_segments)}")

    print("\n" + "-" * 65)
    print(result.caveat_statement)
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

