"""Split utilities — works against the public-bus-mil manifest 'fold' column.

The public-bus-mil benchmark writes a 1..5 fold column to its manifest.csv
(see https://github.com/jbaggett/public-bus-mil). This module just gives
you a convenient way to ask "which bag_ids are in the train / val / test
partition for fold N?" without re-deriving anything from scratch.

Default convention (matches public-bus-mil):
  - test  = fold N (you specify which)
  - val   = one fold drawn from the remaining 4 (default: the next fold)
  - train = the remaining 3 folds
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd


def fold_split(
    manifest_csv: str | Path,
    test_fold: int = 5,
    val_fold: int | None = None,
) -> Tuple[List[str], List[str], List[str]]:
    """Return (train_ids, val_ids, test_ids) for a single fold split.

    If ``val_fold`` is None, use the fold immediately preceding the test
    fold (wrapping around: test_fold=1 → val_fold=5).
    """
    df = pd.read_csv(manifest_csv)
    if "fold" not in df.columns:
        raise ValueError(
            f"manifest at {manifest_csv} has no 'fold' column — rebuild "
            "your public-bus-mil cohort with v0.2.0+ (which writes folds)."
        )
    if test_fold not in (1, 2, 3, 4, 5):
        raise ValueError(f"test_fold must be 1..5, got {test_fold}")
    if val_fold is None:
        val_fold = test_fold - 1 if test_fold > 1 else 5
    if val_fold == test_fold:
        raise ValueError("val_fold must differ from test_fold")

    test_ids = df.loc[df["fold"] == test_fold, "bag_id"].astype(str).tolist()
    val_ids = df.loc[df["fold"] == val_fold, "bag_id"].astype(str).tolist()
    train_ids = df.loc[~df["fold"].isin([test_fold, val_fold]), "bag_id"].astype(str).tolist()
    return train_ids, val_ids, test_ids


def all_folds(manifest_csv: str | Path) -> List[int]:
    """Return sorted list of fold values present in the manifest."""
    df = pd.read_csv(manifest_csv)
    if "fold" not in df.columns:
        return []
    return sorted(df["fold"].dropna().astype(int).unique().tolist())
