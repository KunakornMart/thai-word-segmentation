"""
Validate submission format for Super AI Engineer S6 Word Segmentation Challenge.

Expected CSV columns:
Id,Predicted
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


VALID_LABELS = {"B_WORD", "I_WORD", "E_WORD"}
REQUIRED_COLUMNS = ["Id", "Predicted"]


def validate_submission(path: str | Path, sample_path: str | Path | None = None) -> None:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Submission file not found: {path}")

    sub = pd.read_csv(path)

    if list(sub.columns) != REQUIRED_COLUMNS:
        raise ValueError(f"Invalid columns. Expected {REQUIRED_COLUMNS}, got {list(sub.columns)}")

    if sub["Id"].duplicated().any():
        duplicated = sub.loc[sub["Id"].duplicated(), "Id"].tolist()
        raise ValueError(f"Duplicate IDs found: {duplicated[:20]}")

    if sub["Predicted"].isna().any():
        raise ValueError("Missing labels found in Predicted column.")

    invalid_labels = sorted(set(sub["Predicted"].astype(str)) - VALID_LABELS)
    if invalid_labels:
        raise ValueError(f"Invalid labels found: {invalid_labels}")

    if sample_path is not None:
        sample_path = Path(sample_path)
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample submission file not found: {sample_path}")

        sample = pd.read_csv(sample_path)

        if len(sub) != len(sample):
            raise ValueError(f"Expected {len(sample)} rows, got {len(sub)} rows")

        if not sub["Id"].equals(sample["Id"]):
            raise ValueError("Submission IDs do not match sample submission order.")

    print("Submission format is valid.")
    print(f"Rows: {len(sub):,}")
    print("Columns: Id,Predicted")
    print("Labels: B_WORD, I_WORD, E_WORD")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", help="Path to submission CSV")
    parser.add_argument("--sample", default=None, help="Optional path to ws_sample_submission.csv")
    args = parser.parse_args()

    validate_submission(args.submission, args.sample)


if __name__ == "__main__":
    main()
