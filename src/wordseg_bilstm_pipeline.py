"""
Cleaned pipeline skeleton for Thai Word Segmentation Challenge.

The full experimental workflow is available in the notebook.
This script summarizes the main production-like steps for portfolio readability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from bio_utils import boundaries_to_labels, char_type


@dataclass
class SegmentSample:
    text: str
    end_labels: np.ndarray


def word_to_end_labels(word: str) -> list[int]:
    """Return end-of-word boundary labels for a word."""
    labels = [0] * len(word)
    labels[-1] = 1
    return labels


def split_test_into_nonspace_segments(text: str) -> list[tuple[str, list[int]]]:
    """
    Split test text into non-space segments.

    The sample submission skips spaces, so prediction IDs are mapped only to non-space characters.
    """
    segments: list[tuple[str, list[int]]] = []
    chars: list[str] = []
    ids: list[int] = []

    for idx, ch in enumerate(text, start=1):
        if ch == " ":
            if chars:
                segments.append(("".join(chars), ids.copy()))
                chars = []
                ids = []
        else:
            chars.append(ch)
            ids.append(idx)

    if chars:
        segments.append(("".join(chars), ids.copy()))

    return segments


def build_vocab_from_texts(texts: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Build a simple character vocabulary."""
    charset = sorted(set("".join(texts)))
    stoi = {"<PAD>": 0, "<UNK>": 1}
    for ch in charset:
        stoi[ch] = len(stoi)
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def make_submission(
    sample_submission_path: str | Path,
    segments: list[tuple[str, list[int]]],
    boundaries_list: list[np.ndarray],
    output_path: str | Path,
    single_char_label: str = "B_WORD",
) -> None:
    """
    Convert predicted boundaries into the official Id,Predicted submission format.
    """
    sample = pd.read_csv(sample_submission_path)

    id_to_label: dict[int, str] = {}

    for (segment_text, segment_ids), boundaries in zip(segments, boundaries_list):
        labels = boundaries_to_labels(boundaries, single_char_label=single_char_label)

        if len(labels) != len(segment_text) or len(labels) != len(segment_ids):
            raise ValueError("Label length does not match segment length.")

        for idx, label in zip(segment_ids, labels):
            id_to_label[idx] = label

    sub = sample.copy()
    sub["Predicted"] = sub["Id"].map(id_to_label)

    if sub["Predicted"].isna().any():
        raise ValueError("Some IDs were not assigned predictions.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)


if __name__ == "__main__":
    print("This is a cleaned pipeline skeleton. Run the notebook for the full training and inference workflow.")
