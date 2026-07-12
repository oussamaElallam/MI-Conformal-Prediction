"""Leakage-safe group-aware data splitting helpers.

The paper uses record-level labels but must keep every subject/group in exactly one
partition.  These helpers prefer stratified group k-fold candidates and fall back
to repeated group shuffle candidates when an exact fold construction is not
possible.  The selected candidate minimizes both size and class-ratio deviation.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


def make_group_id(
    frame: pd.DataFrame,
    preferred_columns: Sequence[str] = ("patient_id", "subject_id", "record_id", "ecg_id"),
    output_column: str = "group_id",
) -> pd.DataFrame:
    """Return a copy with a non-missing string group identifier.

    The first available preferred column is used. Missing values are filled with
    a record-level fallback so they cannot silently merge unrelated examples.
    """
    result = frame.copy()
    source = next((column for column in preferred_columns if column in result.columns), None)
    if source is None:
        result[output_column] = [f"row-{i}" for i in range(len(result))]
        result[f"{output_column}_source"] = "row_index"
        return result

    values = result[source].astype("string")
    fallback_column = next(
        (column for column in ("record_id", "ecg_id", "filename_lr") if column in result.columns),
        None,
    )
    if fallback_column is not None:
        fallback = result[fallback_column].astype("string").map(lambda value: f"record:{value}")
    else:
        fallback = pd.Series([f"row-{i}" for i in range(len(result))], index=result.index, dtype="string")

    missing = values.isna() | values.str.strip().isin(["", "nan", "None", "<NA>"])
    result[output_column] = values.mask(missing, fallback).astype(str)
    result[f"{output_column}_source"] = np.where(missing, "record_fallback", source)
    return result


def _candidate_score(
    frame: pd.DataFrame,
    test_indices: np.ndarray,
    label_column: str,
    requested_fraction: float,
) -> float:
    overall_ratio = frame[label_column].mean()
    test = frame.iloc[test_indices]
    size_error = abs(len(test) / len(frame) - requested_fraction)
    class_error = abs(test[label_column].mean() - overall_ratio)
    return float(size_error + class_error)


def _valid_binary_candidate(
    frame: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    label_column: str,
    group_column: str,
) -> bool:
    train = frame.iloc[train_indices]
    test = frame.iloc[test_indices]
    if train[label_column].nunique() < 2 or test[label_column].nunique() < 2:
        return False
    return set(train[group_column]).isdisjoint(set(test[group_column]))


def stratified_group_holdout(
    frame: pd.DataFrame,
    *,
    group_column: str,
    label_column: str,
    test_size: float,
    random_state: int,
    max_shuffle_candidates: int = 128,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into group-disjoint, approximately stratified partitions."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must lie strictly between zero and one")
    if group_column not in frame or label_column not in frame:
        raise KeyError(f"Missing required columns: {group_column}, {label_column}")
    if frame[label_column].nunique() != 2:
        raise ValueError("Binary stratification requires exactly two labels")
    if frame[group_column].nunique() < 4:
        raise ValueError("At least four distinct groups are required")

    frame = frame.reset_index(drop=True).copy()
    y = frame[label_column].to_numpy()
    groups = frame[group_column].astype(str).to_numpy()
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []

    requested_splits = max(2, int(round(1.0 / test_size)))
    requested_splits = min(requested_splits, int(frame[group_column].nunique()))
    try:
        splitter = StratifiedGroupKFold(
            n_splits=requested_splits,
            shuffle=True,
            random_state=random_state,
        )
        for train_indices, test_indices in splitter.split(frame, y, groups):
            if _valid_binary_candidate(frame, train_indices, test_indices, label_column, group_column):
                score = _candidate_score(frame, test_indices, label_column, test_size)
                candidates.append((score, train_indices, test_indices))
    except ValueError:
        pass

    shuffle = GroupShuffleSplit(
        n_splits=max_shuffle_candidates,
        test_size=test_size,
        random_state=random_state,
    )
    for train_indices, test_indices in shuffle.split(frame, y, groups):
        if _valid_binary_candidate(frame, train_indices, test_indices, label_column, group_column):
            score = _candidate_score(frame, test_indices, label_column, test_size)
            candidates.append((score, train_indices, test_indices))

    if not candidates:
        raise RuntimeError(
            "Could not construct a binary, group-disjoint split. "
            "Inspect group counts and class coverage."
        )

    _, train_indices, test_indices = min(candidates, key=lambda item: item[0])
    train = frame.iloc[train_indices].copy().reset_index(drop=True)
    test = frame.iloc[test_indices].copy().reset_index(drop=True)
    assert_group_disjoint(("train", train), ("holdout", test), group_column=group_column)
    return train, test


def assert_group_disjoint(
    *named_frames: tuple[str, pd.DataFrame],
    group_column: str = "group_id",
) -> None:
    """Raise when any pair of named partitions shares a group."""
    for (name_a, frame_a), (name_b, frame_b) in combinations(named_frames, 2):
        overlap = set(frame_a[group_column].astype(str)) & set(frame_b[group_column].astype(str))
        if overlap:
            examples = sorted(overlap)[:5]
            raise AssertionError(
                f"Group leakage between {name_a} and {name_b}: "
                f"{len(overlap)} overlapping groups, examples={examples}"
            )


def split_summary(
    named_frames: Iterable[tuple[str, pd.DataFrame]],
    *,
    group_column: str = "group_id",
    label_column: str = "label",
) -> pd.DataFrame:
    rows = []
    for name, frame in named_frames:
        rows.append(
            {
                "split": name,
                "records": int(len(frame)),
                "groups": int(frame[group_column].nunique()),
                "normal": int((frame[label_column] == 0).sum()),
                "mi": int((frame[label_column] == 1).sum()),
                "mi_fraction": float(frame[label_column].mean()),
            }
        )
    return pd.DataFrame(rows)
