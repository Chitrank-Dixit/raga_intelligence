"""
Indian classical swara nomenclature and interval mappings.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# 12 Swaras in Hindustani Music relative to Shadja (Sa)
SWARA_NAMES: List[str] = [
    "S",   # Shadja (Tonic)
    "r",   # Komal Rishabh (Minor 2nd)
    "R",   # Shuddha Rishabh (Major 2nd)
    "g",   # Komal Gandhar (Minor 3rd)
    "G",   # Shuddha Gandhar (Major 3rd)
    "M",   # Shuddha Madhyam (Perfect 4th)
    "M'",  # Teevra Madhyam (Augmented 4th / Tritone)
    "P",   # Pancham (Perfect 5th)
    "d",   # Komal Dhaivat (Minor 6th)
    "D",   # Shuddha Dhaivat (Major 6th)
    "n",   # Komal Nishad (Minor 7th)
    "N",   # Shuddha Nishad (Major 7th)
]

SWARA_SEMITONES: Dict[str, int] = {name: i for i, name in enumerate(SWARA_NAMES)}
SEMITONE_TO_SWARA: Dict[int, str] = {i: name for i, name in enumerate(SWARA_NAMES)}


def cents_to_swara(cents: float) -> Tuple[str, float]:
    """
    Maps an interval in cents (relative to tonic Sa, modulo 1200) to the nearest
    swara and the microtonal deviation in cents.
    """
    norm_cents = cents % 1200.0
    semitone_float = norm_cents / 100.0
    nearest_semitone = int(round(semitone_float)) % 12
    deviation = (semitone_float - round(semitone_float)) * 100.0
    return SEMITONE_TO_SWARA[nearest_semitone], round(deviation, 1)

