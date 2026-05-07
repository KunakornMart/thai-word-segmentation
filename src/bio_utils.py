"""
Utility functions for Thai word segmentation BIO decoding.
"""

from __future__ import annotations

import unicodedata
from typing import Iterable, List

import numpy as np


LABEL_B = "B_WORD"
LABEL_I = "I_WORD"
LABEL_E = "E_WORD"
VALID_LABELS = {LABEL_B, LABEL_I, LABEL_E}


def char_type(ch: str) -> int:
    """
    Return a simple character type ID.

    0 = padding
    1 = Thai base character
    2 = Thai combining mark
    3 = digit
    4 = Latin
    5 = punctuation / symbol
    6 = other
    """
    cp = ord(ch)
    cat = unicodedata.category(ch)

    if 0x0E00 <= cp <= 0x0E7F:
        if cat.startswith("M"):
            return 2
        return 1

    if ch.isdigit():
        return 3

    if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
        return 4

    if cat.startswith("P") or cat.startswith("S"):
        return 5

    return 6


def boundaries_to_labels(boundaries: np.ndarray, single_char_label: str = LABEL_B) -> List[str]:
    """
    Convert binary end-of-word boundaries into B/I/E labels.

    `boundaries[i] = 1` means character i ends a word.
    """
    if single_char_label not in VALID_LABELS:
        raise ValueError(f"Invalid single_char_label: {single_char_label}")

    n = len(boundaries)
    if n == 0:
        return []

    b = boundaries.astype(np.int64).copy()
    b[-1] = 1

    labels: list[str] = []
    start = 0

    for end in np.where(b == 1)[0]:
        length = end - start + 1

        if length <= 0:
            continue

        if length == 1:
            labels.append(single_char_label)
        elif length == 2:
            labels.extend([LABEL_B, LABEL_E])
        else:
            labels.append(LABEL_B)
            labels.extend([LABEL_I] * (length - 2))
            labels.append(LABEL_E)

        start = end + 1

    if len(labels) != n:
        raise ValueError(f"Decoded label length mismatch: expected {n}, got {len(labels)}")

    return labels


def validate_labels(labels: Iterable[str]) -> bool:
    """Return True if all labels are valid."""
    return set(labels).issubset(VALID_LABELS)
