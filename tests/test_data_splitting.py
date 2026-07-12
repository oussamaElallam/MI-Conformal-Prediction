import numpy as np
import pandas as pd

from experiments.data_splitting import (
    assert_group_disjoint,
    make_group_id,
    stratified_group_holdout,
)


def synthetic_frame(n_groups=80, records_per_group=2):
    groups = np.repeat(np.arange(n_groups), records_per_group)
    labels = np.repeat(np.arange(n_groups) % 2, records_per_group)
    return pd.DataFrame({"patient_id": groups, "ecg_id": np.arange(len(groups)), "label": labels})


def test_group_holdout_is_disjoint_and_binary():
    frame = make_group_id(synthetic_frame())
    train, test = stratified_group_holdout(
        frame,
        group_column="group_id",
        label_column="label",
        test_size=0.2,
        random_state=42,
    )
    assert_group_disjoint(("train", train), ("test", test), group_column="group_id")
    assert train.label.nunique() == 2
    assert test.label.nunique() == 2
    assert abs(len(test) / len(frame) - 0.2) < 0.08


def test_four_way_sequential_group_split_has_no_overlap():
    frame = make_group_id(synthetic_frame(n_groups=120))
    development, test = stratified_group_holdout(
        frame, group_column="group_id", label_column="label", test_size=0.1, random_state=1
    )
    pool, validation = stratified_group_holdout(
        development,
        group_column="group_id",
        label_column="label",
        test_size=1 / 9,
        random_state=2,
    )
    train, calibration = stratified_group_holdout(
        pool,
        group_column="group_id",
        label_column="label",
        test_size=0.2,
        random_state=3,
    )
    assert_group_disjoint(
        ("train", train),
        ("calibration", calibration),
        ("validation", validation),
        ("test", test),
        group_column="group_id",
    )


def test_missing_patient_id_uses_record_fallback():
    frame = pd.DataFrame(
        {"patient_id": [1, np.nan], "ecg_id": [10, 11], "label": [0, 1]}
    )
    result = make_group_id(frame)
    assert result.loc[0, "group_id"] == "1.0"
    assert result.loc[1, "group_id"] == "record:11"
    assert result.loc[1, "group_id_source"] == "record_fallback"
